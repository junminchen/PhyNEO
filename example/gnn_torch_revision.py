import pickle
import re
import sys
from collections import OrderedDict
from functools import partial

import torch
import torch.nn.functional as F
import numpy as np
from graph_torch_revision import MAX_VALENCE, TopGraph, from_pdb
np.set_printoptions(threshold=np.inf)
torch.set_printoptions(profile="full")


r'''
def prm_transform_f2i(params, n_layers):
    p = {}
    for k in params:
        if isinstance(params[k], np.ndarray):
            p[k] = torch.tensor(params[k])
        elif isinstance(params[k], torch.Tensor):
            p[k] = params[k]
        else:
            p[k] = params[k]
    for i_nn in [0, 1]:
        nn_name = 'fc%d' % i_nn
        p['%s.weight' % nn_name] = []
        p['%s.bias' % nn_name] = []
        for i_layer in range(n_layers[i_nn]):
            k_w = '%s.%d.weight' % (nn_name, i_layer)
            k_b = '%s.%d.bias' % (nn_name, i_layer)
            p['%s.weight' % nn_name].append(p.pop(k_w, None))
            p['%s.bias' % nn_name].append(p.pop(k_b, None))
    return p
'''


def prm_transform_f2i(params, n_layers):
    p = {}
    for k in params:
        if isinstance(params[k], np.ndarray):
            p[k] = torch.tensor(params[k], dtype=torch.float32)
        elif isinstance(params[k], torch.Tensor):
            # p[k] = params[k]
            p[k] = torch.tensor(params[k], dtype=torch.float32)
        else:
            p[k] = params[k]
    return p

def prm_transform_i2f(params, n_layers):
    p = {}
    p['w'] = params['w']
    p['fc_final.weight'] = params['fc_final.weight']
    p['fc_final.bias'] = params['fc_final.bias']
    for i_nn in range(2):
        nn_name = 'fc%d' % i_nn
        for i_layer in range(n_layers[i_nn]):
            p[nn_name + '.%d.weight' %
                   i_layer] = params[nn_name + '.weight'][i_layer]
            p[nn_name +
                   '.%d.bias' % i_layer] = params[nn_name +
                                                  '.bias'][i_layer]
    return p


class MolGNNForce(torch.nn.Module):

    def __init__(self, G, n_layers=(3, 2), sizes=[(40, 20, 20), (20, 10)], 
                  nn=1, sigma=162.13039087945623, 
                 mu=117.41975505778706, seed=12345, device='cpu'):
        super(MolGNNForce, self).__init__()
        
        
        self.device = torch.device(device)
        self.nn = nn
        self.G = G
        # self.device = G.device
        self._prepare_graph()

        torch.manual_seed(seed)

        self.n_layers = n_layers
        self.sizes = sizes
        self.sigma = sigma
        self.mu = mu

        self._w = torch.nn.Parameter(torch.rand(1, device=self.device))

        dim_in = self.G.n_features

        self.fc0_layers = torch.nn.ModuleList()
        self.fc1_layers = torch.nn.ModuleList()
        
        for i_layer in range(n_layers[0]):
            dim_out = sizes[0][i_layer]
            layer = torch.nn.Linear(dim_in, dim_out)
            torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='tanh')
            torch.nn.init.zeros_(layer.bias)
            self.fc0_layers.append(layer)
            dim_in = dim_out
        

        for i_layer in range(n_layers[1]):
            dim_out = sizes[1][i_layer]
            layer = torch.nn.Linear(dim_in, dim_out)
            torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='tanh')
            torch.nn.init.zeros_(layer.bias)
            self.fc1_layers.append(layer)
            dim_in = dim_out
        
        self.fc_final = torch.nn.Linear(dim_in, 1)
        torch.nn.init.kaiming_uniform_(self.fc_final.weight)
        torch.nn.init.uniform_(self.fc_final.bias)

        self.to(self.device)

    def _prepare_graph(self):
        # if not isinstance(self.G, TopGraph):
        #     raise TypeError('MolGNNForce requires a TopGraph instance as input graph.')

        # target_device = self.device
        # self.G.device = target_device


        import time
        start = time.time()
        self.G.get_all_subgraphs(self.nn, typify=True)
        print(f"self.G.get_all_subgraphs: {time.time()-start:.2f}")

        if hasattr(self.G, '_process_unique_subgraphs'):
            start = time.time()
            self.G._process_unique_subgraphs(id_chiral=True)
            print(f"self.G._process_unique_subgraphs: {time.time()-start:.2f}")

        # if getattr(self.G, 'positions', None) is not None and hasattr(self.G, 'set_positions'):
        #     start = time.time()
        #     self.G.set_positions(self.G.positions, update_subgraph=True)
        #     print(f"self.G.set_positions: {time.time()-start:.2f}")
        
        start = time.time()
        self.G.prepare_subgraph_feature_calc()
        print(f"self.G.prepare_subgraph_feature_calc: {time.time()-start:.2f}")

        attr_dict = vars(self.G)
        for attr, value in attr_dict.items():#['bonds', 'positions', 'box', 'box_inv', 'angles']:
            # value = getattr(self.G, attr, None)
            if value is None:
                print(attr, 'is none')
            if isinstance(value, torch.Tensor):
                print(attr)
                setattr(self.G, attr, value.to(self.device))
            if attr == 'feature_indices':
                for sub_attr, sub_value in self.G.feature_indices.items():
                    if isinstance(sub_value, torch.Tensor):
                        self.G.feature_indices[sub_attr] = sub_value.to(self.device)
                        print(sub_attr)
        print(self.G.bonds.device)

    def _ensure_tensor(self, value, device, dtype):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            needs_move = value.device != device or value.dtype != dtype
            return value.to(device=device, dtype=dtype) if needs_move else value
        return torch.tensor(value, device=device, dtype=dtype)

    def fc0(self, f_in):
        f = f_in
        for layer in self.fc0_layers:
            f = torch.tanh(layer(f))
        return f

    def fc1(self, f_in):
        f = f_in
        for layer in self.fc1_layers:
            f = torch.tanh(layer(f))
        return f

    def message_pass(self, f_in, nb_connect, w, nn):
        is_batched = f_in.dim() == 4
        if nn == 0:
            return f_in[:, :, 0, :] if is_batched else f_in[:, 0, :]
        elif nn == 1:
            if nb_connect is None:
                raise ValueError('Neighbor connectivity is required when nn == 1.')

            w = w.to(dtype=f_in.dtype, device=f_in.device)
            nb_connect = nb_connect.to(dtype=f_in.dtype, device=f_in.device)

            eps = torch.tensor(1e-5, dtype=f_in.dtype, device=f_in.device)

            if is_batched:
                nb_connect0 = nb_connect.unsqueeze(0)[:, :, :MAX_VALENCE - 1]
                nb_connect1 = nb_connect.unsqueeze(0)[:, :, MAX_VALENCE - 1:2 * (MAX_VALENCE - 1)]
                nb0 = torch.sum(nb_connect0, dim=2, keepdim=True)
                nb1 = torch.sum(nb_connect1, dim=2, keepdim=True)

                f_center = f_in[:, :, 0, :]
                f_nb0 = f_in[:, :, 1:MAX_VALENCE, :]
                f_nb1 = f_in[:, :, MAX_VALENCE:2 * MAX_VALENCE - 1, :]

                weighted_nb0 = torch.sum(nb_connect0.unsqueeze(-1) * f_nb0, dim=2)
                weighted_nb1 = torch.sum(nb_connect1.unsqueeze(-1) * f_nb1, dim=2)

                nb0_safe = torch.where(nb0 < eps, eps, nb0)
                nb1_safe = torch.where(nb1 < eps, eps, nb1)

                heaviside_nb0 = (nb0 > 0).to(f_in.dtype)
                heaviside_nb1 = (nb1 > 0).to(f_in.dtype)

                f = f_center * (1 - heaviside_nb0 * w - heaviside_nb1 * w) + \
                    w * weighted_nb0 / nb0_safe + \
                    w * weighted_nb1 / nb1_safe
            else:
                nb_connect0 = nb_connect[:, :MAX_VALENCE - 1]
                nb_connect1 = nb_connect[:, MAX_VALENCE - 1:2 * (MAX_VALENCE - 1)]
                nb0 = torch.sum(nb_connect0, dim=1, keepdim=True)
                nb1 = torch.sum(nb_connect1, dim=1, keepdim=True)

                f_center = f_in[:, 0, :]
                f_nb0 = f_in[:, 1:MAX_VALENCE, :]
                f_nb1 = f_in[:, MAX_VALENCE:2 * MAX_VALENCE - 1, :]

                weighted_nb0 = torch.sum(nb_connect0.unsqueeze(-1) * f_nb0, dim=1)
                weighted_nb1 = torch.sum(nb_connect1.unsqueeze(-1) * f_nb1, dim=1)

                nb0_safe = torch.where(nb0 < eps, eps, nb0)
                nb1_safe = torch.where(nb1 < eps, eps, nb1)

                heaviside_nb0 = (nb0 > 0).to(f_in.dtype)
                heaviside_nb1 = (nb1 > 0).to(f_in.dtype)

                f = f_center * (1 - heaviside_nb0 * w - heaviside_nb1 * w) + \
                    w * weighted_nb0 / nb0_safe + \
                    w * weighted_nb1 / nb1_safe

            return f

    def forward(self, positions, box=None):
        # positions = positions*10
        dtype = self.w.dtype

        if isinstance(positions, np.ndarray):
            positions_tensor = torch.tensor(positions, device=self.device, dtype=dtype)
        else:
            positions_tensor = self._ensure_tensor(positions, self.device, dtype)

        is_batched = positions_tensor.dim() == 3

        if box is not None:
            if isinstance(box, np.ndarray):
                box_tensor = torch.tensor(box, device=self.device, dtype=dtype)
            else:
                box_tensor = self._ensure_tensor(box, self.device, dtype)
        else:
            box_tensor = None

        features = self.G.calc_subgraph_features(positions_tensor, box_tensor).to(self.device)
        # print(f"features of self.G.calc_subgraph_features(positions, box): {features.detach().cpu().numpy()}")
        
        features = self.fc0(features)

        nb_connect = getattr(self.G, 'nb_connect', None)
        features = self.message_pass(features, nb_connect, self.w, self.G.nn)
        features = self.fc1(features)

        energies = self.fc_final(features).squeeze(-1)

        weights = self.G.weights.to(device=self.device, dtype=dtype)

        if is_batched:
            total_energy = torch.sum(weights.unsqueeze(0) * energies, dim=1) * self.sigma + self.mu
        else:
            total_energy = torch.sum(weights * energies) * self.sigma + self.mu

        return total_energy

    def batch_forward(self, positions_batch, box_batch):
        return self.forward(positions_batch, box_batch)

    def get_energy(self, positions, box, params=None):
        return self.forward(positions, box)

    @property
    def params(self):
        return self.parameters_dict()

    @property
    def w(self):
        return getattr(self, '_w', torch.nn.Parameter(torch.rand(1, device=self.device)))
    
    @w.setter
    def w(self, value):
        if isinstance(value, torch.Tensor):
            self._w = torch.nn.Parameter(value.to(device=self.device))
        else:
            self._w = torch.nn.Parameter(torch.tensor(value, device=self.device))

    def load_params(self, ifn):
        with open(ifn, 'rb') as ifile:
            params = pickle.load(ifile)
        
        for k in params.keys():
            if isinstance(params[k], np.ndarray):
                params[k] = torch.tensor(params[k], device=self.device)
            elif isinstance(params[k], torch.Tensor):
                params[k] = params[k].to(device=self.device)
        
        params_internal = prm_transform_f2i(params, self.n_layers)
        
        for k in params_internal.keys():
            if isinstance(params_internal[k], torch.Tensor):
                params_internal[k] = params_internal[k].to(device=self.device)
            elif isinstance(params_internal[k], list):
                for i, item in enumerate(params_internal[k]):
                    if isinstance(item, torch.Tensor):
                        params_internal[k][i] = item.to(device=self.device)
        
        with torch.no_grad():
            self.w = params_internal['w'].to(device=self.device)
            
            for i, layer in enumerate(self.fc0_layers):
                if i < len(params_internal['fc0.weight']):
                    saved_weight = params_internal['fc0.weight'][i]
                    saved_bias = params_internal['fc0.bias'][i]
                    if saved_weight.shape == layer.weight.shape:
                        layer.weight.copy_(saved_weight)
                    if saved_bias.shape == layer.bias.shape:
                        layer.bias.copy_(saved_bias)
            
            for i, layer in enumerate(self.fc1_layers):
                if i < len(params_internal['fc1.weight']):
                    saved_weight = params_internal['fc1.weight'][i]
                    saved_bias = params_internal['fc1.bias'][i]
                    if saved_weight.shape == layer.weight.shape:
                        layer.weight.copy_(saved_weight)
                    if saved_bias.shape == layer.bias.shape:
                        layer.bias.copy_(saved_bias)
            
            saved_final_weight = params_internal['fc_final.weight'].to(device=self.device)
            saved_final_bias = params_internal['fc_final.bias'].to(device=self.device)
            
            if saved_final_weight.shape == self.fc_final.weight.shape:
                self.fc_final.weight.copy_(saved_final_weight)
            if saved_final_bias.shape == self.fc_final.bias.shape:
                self.fc_final.bias.copy_(saved_final_bias)
        
        return

    def save_params(self, ofn):
        """ Save the network parameters to a pickle file

        Parameters
        ----------
        ofn: string
            the output file name

        """
        params_internal = OrderedDict()
        
        params_internal['w'] = self.w.detach().cpu()
        
        params_internal['fc0.weight'] = []
        params_internal['fc0.bias'] = []
        for layer in self.fc0_layers:
            params_internal['fc0.weight'].append(layer.weight.detach().cpu().T)
            params_internal['fc0.bias'].append(layer.bias.detach().cpu())
        
        params_internal['fc1.weight'] = []
        params_internal['fc1.bias'] = []
        for layer in self.fc1_layers:
            params_internal['fc1.weight'].append(layer.weight.detach().cpu().T)
            params_internal['fc1.bias'].append(layer.bias.detach().cpu())
        
        params_internal['fc_final.weight'] = self.fc_final.weight.detach().cpu()
        params_internal['fc_final.bias'] = self.fc_final.bias.detach().cpu()
        
        params = prm_transform_i2f(params_internal, self.n_layers)
        

        for k in params:
            if isinstance(params[k], torch.Tensor):
                params[k] = params[k].numpy()
        
        with open(ofn, 'wb') as ofile:
            pickle.dump(params, ofile)
        return

    def parameters_dict(self):
        params = OrderedDict()
        params['w'] = self.w
        for i, layer in enumerate(self.fc0_layers):
            params[f'fc0.{i}.weight'] = layer.weight.T
            params[f'fc0.{i}.bias'] = layer.bias
        for i, layer in enumerate(self.fc1_layers):
            params[f'fc1.{i}.weight'] = layer.weight.T
            params[f'fc1.{i}.bias'] = layer.bias
        params['fc_final.weight'] = self.fc_final.weight
        params['fc_final.bias'] = self.fc_final.bias
        return params

    def set_parameters_dict(self, params):
        with torch.no_grad():
            if 'w' in params:
                self.w = params['w']
            
            for i, layer in enumerate(self.fc0_layers):
                weight_key = f'fc0.{i}.weight'
                bias_key = f'fc0.{i}.bias'
                if weight_key in params:
                    layer.weight.copy_(params[weight_key].T)
                if bias_key in params:
                    layer.bias.copy_(params[bias_key])
            
            for i, layer in enumerate(self.fc1_layers):
                weight_key = f'fc1.{i}.weight'
                bias_key = f'fc1.{i}.bias'
                if weight_key in params:
                    layer.weight.copy_(params[weight_key].T)
                if bias_key in params:
                    layer.bias.copy_(params[bias_key])
            
            if 'fc_final.weight' in params:
                self.fc_final.weight.copy_(params['fc_final.weight'])
            if 'fc_final.bias' in params:
                self.fc_final.bias.copy_(params['fc_final.bias'])

    def train(self, mode: bool = True):
        super().train(mode)
        return self

    def eval_mode(self):
        self.eval()
        return self

    def to_device(self, device):
        self.device = torch.device(device)
        self.to(self.device)
        # self._prepare_graph()
        return self

    def compute_gradients(self, positions, box):
        dtype = self.w.dtype
        positions_tensor = self._ensure_tensor(positions, self.device, dtype)
        if not positions_tensor.requires_grad:
            positions_tensor = positions_tensor.clone().detach().requires_grad_(True)
        energy = self.forward(positions_tensor, box)

        is_batched = positions_tensor.dim() == 3
        if is_batched:
            # Sum energies for batched gradient calculation
            total_energy = torch.sum(energy)
            gradients = torch.autograd.grad(total_energy, positions_tensor, create_graph=True)[0]
        else:
            gradients = torch.autograd.grad(energy, positions_tensor, create_graph=True)[0]
            
        return energy, -gradients  # Forces are negative gradients

    def energy_and_forces(self, positions, box):
        return self.compute_gradients(positions, box)

