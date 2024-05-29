#!/usr/bin/env python
import sys
import jax
import jax.numpy as jnp
from jax import vmap, jit, value_and_grad
import numpy as np
from dmff.utils import jit_condition, regularize_pairs, pair_buffer_scales
from dmff.admp.pairwise import distribute_scalar, distribute_v3
from dmff.admp.spatial import pbc_shift
from functools import partial
import jax.nn.initializers
import pickle
# from jax.config import config
# config.update("jax_debug_nans", True)

# Make printing parameters a little more readable
def parameter_shapes(params):
    return jax.tree_util.tree_map(lambda p: p.shape, params)

@jit_condition(static_argnums=())
@partial(vmap, in_axes=(0, 0, None, None, None, None, None), out_axes=0)
def get_gto(i_atom, r, pairs, rc, rs, inta, species):
    gto_i = jnp.exp(-inta[species[pairs[i_atom][1]]] * (r - rs[species[pairs[i_atom][1]]])**2)
    gto_j = jnp.exp(-inta[species[pairs[i_atom][0]]] * (r - rs[species[pairs[i_atom][0]]])**2)
    return gto_i, gto_j

@jit_condition(static_argnums=())
@partial(vmap, in_axes=(0, None), out_axes=0)
def cutoff_cosine(distances, cutoff):
    return jnp.square(0.5 * jnp.cos(distances * (jnp.pi / cutoff)) + 0.5)

@jit_condition(static_argnums=())
@partial(vmap, in_axes=(0, 0, None), out_axes=0)
def distribute_pair_cij(i_elem, j_elem, cij):
    return cij[j_elem]
    
@jit_condition(static_argnums=())
@partial(vmap, in_axes=(0, None, None, None), out_axes=(0)) 
def reduce_atoms(i_atom, wfs, indices, buffer_scales):
    mask = (indices == i_atom)
    res = jnp.einsum('ijk,i,i', wfs, mask, buffer_scales)
    return res

def layer_norm(x, weight, bias, axis=-1, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    std = jnp.sqrt(var + eps)
    y = (x - mean) / std * weight + bias
    return y

# calculate neural network energy through features
# Linear, LayerNorm, Relu_Like, Linear, LayerNorm, Relu_Like, Linear
@jit_condition(static_argnums=())
@partial(vmap, in_axes=(0, 0, None), out_axes=(0))
def get_atomic_energies(features, elem_index, params):
    # 0 Linear
    features1 = features.dot(params['w'][0][elem_index]) + params['b'][0][elem_index]
    # 1 LayerNorm
    features2 = layer_norm(features1, params['w'][1][elem_index], params['b'][1][elem_index])
    # 2 Relu_Like
    features3 = params['w'][2][elem_index] * jax.nn.silu(features2 * params['b'][2][elem_index])
    # 3 Linear 
    features4 = features3.dot(params['w'][3][elem_index]) + params['b'][3][elem_index]
    # 4 LayerNorm
    features5 = layer_norm(features4, params['w'][4][elem_index], params['b'][4][elem_index])
    # 5 Relu_Like 
    features6 = params['w'][5][elem_index] * jax.nn.silu(features5 * params['b'][5][elem_index])
    # 6 Linear 
    features7 = features6.dot(params['w'][6][elem_index]) + params['b'][6][elem_index]
    return features7


class EANNForce:

    def __init__(self, n_elem, elem_indices, n_gto, rc, nipsin=2, beta=0.2, sizes=(64, 64), seed=12345):
        """ Constructor

        Parameters
        ----------
        n_elem: int
            Number of elements in the model.
        elem_indices: array of ints
            Element type of each atom in the system.
        n_gto: int
            Number of GTOs used in EANN.
        rc: float
            Cutoff distances, used to determine initial rs and inta.
        nipsin: int, optional
            Largest L in angular channel. Default 2
        beta: float, optional
            beta used to determine initial \Delta rs. Default 0.2
        sizes: tupple, ints, optional
            Number of hidden neurons in the model, the length is number of layers.
            Default (20, 20)
        seed: int, optional
            Seed for random number generator, default 12345

        Examples
        ----------

        """
        self.n_elem = n_elem
        self.n_gto = n_gto
        self.rc = rc
        self.beta = beta
        self.sizes = sizes
        self.n_layers = len(sizes)
        self.nipsin = nipsin
        self.elem_indices = elem_indices
        self.n_atoms = len(elem_indices)

        # n_elements * n_features
        self.n_features = (nipsin+1) * n_gto
        cij = jnp.ones((n_elem, n_gto)) * 0.0
        rs, inta = self.get_init_rs(n_gto, beta, rc)
        initpot = jnp.ones(1) * 0.0

        # initialize NN params
        key = jax.random.PRNGKey(seed)
        initializer = jax.nn.initializers.he_uniform()
        weights = []
        bias = []

        dim_in = self.n_features
        W = []
        B = []
        # Linear, LayerNorm, Relu_Like, Linear, LayerNorm, Relu_Like, Linear
        for i_layer in range(self.n_layers):
            dim_out = sizes[i_layer]
            key, subkey = jax.random.split(key)
            W.append(initializer(subkey, (n_elem, dim_in, dim_out)))
            B.append(jnp.zeros((n_elem, dim_out)))
            # LayerNorm
            W.append(initializer(subkey, (n_elem, dim_out)))
            B.append(jnp.zeros((n_elem, dim_out)))
            # Relu_like 
            W.append(initializer(subkey, (n_elem, 1, dim_out)))
            B.append(jnp.zeros((n_elem, 1, dim_out)))            
            dim_in = dim_out
        key, subkey = jax.random.split(key)
        W.append(initializer(subkey, (n_elem, dim_in)))
        key, subkey = jax.random.split(key)
        B.append(jax.random.uniform(subkey, shape=(n_elem,)))

        # prepare input parameters
        # weights: weights[i_layer][n_elem, dim_in, dim_out]
        # bias: bias[i_layer][n_elem, dim_out]
        self.params = {
                'w': W,
                'b': B,
                'c': cij,
                'rs': rs,
                'inta': inta,
                'initpot': initpot
                }
        # prepare angular channels
        npara = [1]
        for i in range(1,self.nipsin+1):
            npara.append(3**i)
        self.index_para = jnp.concatenate([jnp.ones((npara[i],), dtype=jnp.int32) * i for i in range(len(npara))])

        # generate get_energy
        self.get_energy = self.generate_get_energy()

        return

    def get_init_rs(self, n_gto, beta, rc):
        """
        Generate initial values for rs and inta (exponents)

        Parameters
        ----------
        n_gto: int
            number of radial GTOs used in EANN
        beta: float
            beta used to determine initial \Delta rs. Default 0.2
        rc: float
            cutoff distance

        Returns
        ----------
        rs: 
            (3, n_gto): list of rs (for different radial channels)
        inta:
            (3, n_gto): list of inta
        """
        drs = rc / (n_gto - 1 + 0.3333333333)
        a = beta / drs / drs
        # rs = jnp.arange(0, rc, drs)
        # inta = jnp.ones(n_gto) * a
        rs=jnp.stack([jnp.arange(0, rc, drs) for itype in range(self.n_elem)],axis=0)
        inta=jnp.stack([jnp.ones(n_gto) * a for itype in range(self.n_elem)],axis=0)
        return rs, inta

    def get_features(self, radial, dr, pairs, buffer_scales, orb_coeff):
        """ Get atomic features from pairwise gto arrays
        
        Parameters
        ----------
        gtos(radial): array, (2, n_pairs, nipsin+1, n_gtos)
            pairwise gto values, that is, 
            cij * exp(-inta * (r-rs)**2) * 0.25*(cos(r/rc*pi) + 1)**2
        dr: array
            dr_vec for each pair, pbc shifted
        pairs: int array
            Indices of interacting pairs
        buffer_scales: float (0 or 1)
            neighbor list buffer masks

        Returns
        ----------
        features: (n_atom, n_features) array
            Atomic features

        Examples
        ----------
        """

        dist_vec = jnp.concatenate((dr,-dr),axis=0)
        dr_norm = jnp.linalg.norm(dist_vec, axis=1)
        f_cut = cutoff_cosine(dr_norm, self.rc)
        neigh_list = jnp.concatenate((pairs,pairs[:,[1,0]]),axis=0)
        buffer_scales_ = jnp.concatenate((buffer_scales,buffer_scales),axis=0)
        totneighbour = len(neigh_list)
        prefacs = f_cut.reshape(1, -1)
        angular = prefacs
        for ipsin in range(1,self.nipsin+1):
            prefacs = jnp.einsum("ji,ki->jki", prefacs, dist_vec.T).reshape(-1, totneighbour)
            angular = jnp.vstack((angular, prefacs))
        orbital = jnp.einsum("ji,ik->ijk", angular, radial)
        expandpara = orb_coeff[neigh_list[:,1],:] 
        worbital = jnp.einsum("ijk,ik,i->ijk", orbital, expandpara, buffer_scales_) 
        sum_worbital = jnp.zeros((self.n_atoms, orbital.shape[1], self.rs.shape[1]), dtype=orbital.dtype) 
        sum_worbital = sum_worbital.at[neigh_list[:,0], :, :].add(worbital)
        features = jnp.zeros((self.n_atoms, self.nipsin+1, self.rs.shape[1]), dtype=orbital.dtype) 
        features = features.at[:,self.index_para,:].add(jnp.square(sum_worbital)) 
        features = features.reshape(self.n_atoms,-1)
        return features


    def generate_get_energy(self):

        @jit_condition(static_argnums=())
        def get_energy(positions, box, pairs, params):
            """ Get energy
            This function returns the EANN energy.

            Parameters
            ----------
            positions: (n_atom, 3) array
                The positions of all atoms, in cartesian
            box: (3, 3) array
                The box array, arranged in rows
            pairs: jax_md nbl index
                The neighbor list, in jax_md.partition.OrderedSparse format
            params: dict
                The parameter dictionary, including the following keys:
                c: ${c_{ij}} of all exponent prefactors, (n_elem, n_elem)
                rs: distance shifts of all radial gaussian functions, (n_gto,)
                inta: the exponents, (n_gto,)
                w: weights of NN, list of (n_elem, dim_in, dime_out) array, with a length of n_layer
                b: bias of NN, list of (n_elem, dim_out) array, with a length of n_layer
            
            Returns:
            ----------
            energy: float or double
                EANN energy

            Examples:
            ----------
            """
            pairs = pairs[:,:2]
            pairs = regularize_pairs(pairs)
            buffer_scales = pair_buffer_scales(pairs)

            # get distances
            box_inv = jnp.linalg.inv(box)
            ri = distribute_v3(positions, pairs[:, 0])
            rj = distribute_v3(positions, pairs[:, 1])
            dr = rj - ri
            dr = pbc_shift(dr, box, box_inv)

            dr_norm = jnp.linalg.norm(dr, axis=1)

            buffer_scales2 = jnp.piecewise(buffer_scales, (dr_norm <= 4, dr_norm > 4),
                            (lambda x: jnp.array(1), lambda x: jnp.array(0)))
            buffer_scales = buffer_scales2 * buffer_scales

            self.rs = params['rs']
            self.inta = params['inta']

            radial_i, radial_j = get_gto(jnp.arange(len(dr_norm)), dr_norm, pairs, self.rc, self.rs, self.inta, self.elem_indices)
            radial = jnp.concatenate((radial_i,radial_j), axis=0)
            orb_coeff = params['c'][self.elem_indices,:] # (48,16)

            features = self.get_features(radial, dr, pairs, buffer_scales, orb_coeff)
            atomic_energies = get_atomic_energies(features, self.elem_indices, params)
            return jnp.sum(atomic_energies + params['initpot'][0])

        return get_energy


def validation():
    # O H H
    from jax_md import partition, space
    rc = 4.0
    n_gto = 16
    n_elem = 3
    nipsin = 2
    n_layers = 2

    import MDAnalysis as mda
    atomtype = ['H','C','O']
    u = mda.Universe('ff_files/peg4oh2_dimer.pdb')
    elements = u.atoms.elements
    n_atoms = len(u.atoms)
    species = []
    for i in range(n_atoms):
        species.append(atomtype.index(elements[i]))
    elem_indices = jnp.array(species)
    box = jnp.array(u.trajectory[0].triclinic_dimensions)
    pos = jnp.array(u.atoms.positions)
    
    displacement_fn, shift_fn = space.periodic_general(box, fractional_coordinates=False)
    neighborlist_fn = partition.neighbor_list(displacement_fn, box, rc, 0, format=partition.OrderedSparse)
    nblist = neighborlist_fn.allocate(pos)
    pairs = nblist.idx.T
    pairs = jnp.hstack((pairs,jnp.expand_dims(pairs[:,1],1)))
    eann_force = EANNForce(n_elem, elem_indices, n_gto, rc)
    params = eann_force.params
    E = eann_force.get_energy(pos, box, pairs, params)
    print(E)


    # # transfer params to jax params
    # import torch
    # from collections import OrderedDict
    # state_dict = torch.load("ff_files/EANN.pth",map_location='cpu')
    # new_state_dict = OrderedDict()
    # for k, v in state_dict['eannparam'].items():
    #     if k[0:7]=="module.":
    #         name = k[7:] # remove `module.`
    #         new_state_dict[name] = v
    #     else:
    #         name = k
    #         new_state_dict[name] = v
    # param = new_state_dict
    # with open('ff_files/params_eann_test.pickle', 'wb') as f:
    #     pickle.dump(param, f, protocol=pickle.HIGHEST_PROTOCOL)

    params = {}
    with open('ff_files/params_eann_test.pickle', 'rb') as f:
        param = pickle.load(f)
    parameter_shapes(param)
    for key in param:
        param[key] = jnp.array(param[key])
    # build params_init
    params['c'] = param['density.params']
    params['rs'] = param['density.rs']
    params['inta'] = param['density.inta']
    params['initpot'] = param['nnmod.initpot']
    params['w'] = []
    params['b'] = []
    params['w'].append(jnp.stack((param['nnmod.elemental_nets.H.0.weight'].T,param['nnmod.elemental_nets.C.0.weight'].T,param['nnmod.elemental_nets.O.0.weight'].T)))
    params['w'].append(jnp.stack((param['nnmod.elemental_nets.H.1.weight'],param['nnmod.elemental_nets.C.1.weight'],param['nnmod.elemental_nets.O.1.weight'])))
    params['w'].append(jnp.stack((param['nnmod.elemental_nets.H.2.alpha'],param['nnmod.elemental_nets.C.2.alpha'],param['nnmod.elemental_nets.O.2.alpha'])))
    params['w'].append(jnp.stack((param['nnmod.elemental_nets.H.3.weight'].T,param['nnmod.elemental_nets.C.3.weight'].T,param['nnmod.elemental_nets.O.3.weight'].T)))
    params['w'].append(jnp.stack((param['nnmod.elemental_nets.H.4.weight'],param['nnmod.elemental_nets.C.4.weight'],param['nnmod.elemental_nets.O.4.weight'])))
    params['w'].append(jnp.stack((param['nnmod.elemental_nets.H.5.alpha'],param['nnmod.elemental_nets.C.5.alpha'],param['nnmod.elemental_nets.O.5.alpha'])))
    params['w'].append(jnp.stack((param['nnmod.elemental_nets.H.6.weight'].T,param['nnmod.elemental_nets.C.6.weight'].T,param['nnmod.elemental_nets.O.6.weight'].T)))

    params['b'].append(jnp.stack((param['nnmod.elemental_nets.H.0.bias'],param['nnmod.elemental_nets.C.0.bias'],param['nnmod.elemental_nets.O.0.bias'])))
    params['b'].append(jnp.stack((param['nnmod.elemental_nets.H.1.bias'],param['nnmod.elemental_nets.C.1.bias'],param['nnmod.elemental_nets.O.1.bias'])))
    params['b'].append(jnp.stack((param['nnmod.elemental_nets.H.2.beta'],param['nnmod.elemental_nets.C.2.beta'],param['nnmod.elemental_nets.O.2.beta'])))
    params['b'].append(jnp.stack((param['nnmod.elemental_nets.H.3.bias'],param['nnmod.elemental_nets.C.3.bias'],param['nnmod.elemental_nets.O.3.bias'])))
    params['b'].append(jnp.stack((param['nnmod.elemental_nets.H.4.bias'],param['nnmod.elemental_nets.C.4.bias'],param['nnmod.elemental_nets.O.4.bias'])))
    params['b'].append(jnp.stack((param['nnmod.elemental_nets.H.5.beta'],param['nnmod.elemental_nets.C.5.beta'],param['nnmod.elemental_nets.O.5.beta'])))
    params['b'].append(jnp.stack((param['nnmod.elemental_nets.H.6.bias'],param['nnmod.elemental_nets.C.6.bias'],param['nnmod.elemental_nets.O.6.bias'])))
    parameter_shapes(params)
    get_energy = eann_force.get_energy
    E = get_energy(pos, box, pairs, params)
    print(E)

    get_energy = jit(value_and_grad(eann_force.get_energy, argnums=(0,1)))
    E = get_energy(pos, box, pairs, params)
    print(E)

    with open('ff_files/params_eann.pickle', 'wb') as f:
        pickle.dump(params, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    validation()
