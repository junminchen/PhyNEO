#!/usr/bin/env python
import sys
import jax
import jax.numpy as jnp
from jax import grad, value_and_grad, jit
import numpy as np
import dmff
from dmff.utils import jit_condition
from dmff.sgnn.gnn import MolGNNForce
from dmff.sgnn.graph import TopGraph, from_pdb
import optax
import pickle
# use pytorch data loader
from torch.utils.data import DataLoader

class MolDataSet():

    def __init__(self, pdb, pickle_fn):
        self.file = pickle_fn
        with open(pickle_fn, 'rb') as f:
            self.data = pickle.load(f)
        self.n_data = len(self.data['positions'])
        self.pickle = pickle_fn
        self.pdb = pdb
        return

    def __getitem__(self, i):
        return [self.data['positions'][i], self.data['energies'][i]]

    def __len__(self):
        return self.n_data


if __name__ == "__main__":

    set_file = 'dataset_test_pe16.pickle'
    set_wo_nb_file = 'dataset_test_pe16_remove_nb.pickle'
    # training and testing data
    dataset = MolDataSet('pe16.pdb', set_wo_nb_file)
    test_loader = DataLoader(dataset, batch_size=1000, shuffle=False)
    box = jnp.eye(3) * 50

    # Graph and model
    G = from_pdb('pe16.pdb')
    model = MolGNNForce(G, nn=1)
    model.batch_forward = jax.vmap(model.forward, in_axes=(0, None, None), out_axes=(0))
    model.load_params('params_sgnn.pickle')

    # evaluate test
    ene_preds = []
    for pos, e in test_loader:
        pos = jnp.array(pos.numpy())
        ene_pred = model.batch_forward(pos, box, model.params)
        ene_preds.append(ene_pred)

    with open(set_file, 'rb') as f:
        data = pickle.load(f)
    with open(set_wo_nb_file, 'rb') as f:
        data_wo_nb = pickle.load(f)

    ene_nb = data['energies'] - data_wo_nb['energies']
    ene_pred = ene_nb + np.array(ene_preds).reshape(-1)
    ene_ref = data['energies']
    ene_pred -= np.average(ene_pred)
    ene_ref -= np.average(ene_ref)
    n_data = len(ene_ref)
    rmsd = np.sqrt(np.average((ene_pred - ene_ref)**2))
    print('#', rmsd)
    # print test data
    with open('test_data_pe16.xvg', 'w') as f:
        print('#', rmsd, file=f)
        for i in range(n_data):
            e1 = ene_pred[i]
            e0 = ene_ref[i]
            print(e0, e0, e1, file=f)
