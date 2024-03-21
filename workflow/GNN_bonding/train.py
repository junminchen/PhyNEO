#!/usr/bin/env python
import sys
import jax
import time
import jax.numpy as jnp
from jax import grad, value_and_grad, jit
import numpy as np
import dmff
from dmff.utils import jit_condition
# from gnn import MolGNNForce
# from graph import TopGraph, from_pdb

from dmff.sgnn.gnn import MolGNNForce
from dmff.sgnn.graph import TopGraph, from_pdb

import optax
import pickle
# use pytorch data loader
from torch.utils.data import DataLoader

class MolDataSet():

    def __init__(self, pdb, data):
        self.data = data
        self.n_data = len(self.data['positions'])
        self.pdb = pdb
        return

    def __getitem__(self, i):
        return [self.data['positions'][i], self.data['energies'][i]]

    def __len__(self):
        return self.n_data


if __name__ == "__main__":
    restart = None
    with open('dataset_train_remove_nb.pickle', 'rb') as ifile:
        data_train = pickle.load(ifile)
    with open('dataset_test_remove_nb.pickle', 'rb') as ifile:
        data_test = pickle.load(ifile)    

    pdb = 'pe8.pdb'
    # training and testing data
    dataset = MolDataSet(pdb, data_train)
    train_loader = DataLoader(dataset, shuffle=True, batch_size=64)
    dataset_test = MolDataSet(pdb, data_test)
    test_loader = DataLoader(dataset_test, batch_size=500)
    box = jnp.eye(3) * 50

    # Graph and model
    G = from_pdb(pdb)
    model = MolGNNForce(G, nn=1)
    model.batch_forward = jax.vmap(model.forward, in_axes=(0, None, None), out_axes=(0))
    if restart is not None:
        model.load_params(restart)

    # optmizer
    lr = 0.001
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(model.params)

    # mean square loss function
    def MSELoss(params, positions, box, ene_ref):
        ene = model.batch_forward(positions, box, params)
        err = ene - ene_ref
        # we do not care about constant shifts
        err -= jnp.average(err)
        return jnp.average(err**2)
    MSELoss = jit(MSELoss)

    # train
    best_loss = jnp.array(1e30)
    n_epochs = 3000
    fout=open('nn.err','w')
    fout.write("Sub-Graph Nerual Network Package used for intramolecular energy\n")
    fout.write(time.strftime("%Y-%m-%d-%H_%M_%S \n", time.localtime()))
    fout.flush()
    for i_epoch in range(n_epochs):
        # train an epoch
        lossprop = 0
        for ibatch, (pos, e) in enumerate(train_loader):
            pos = jnp.array(pos.numpy())
            ene_ref = jnp.array(e.numpy())
            loss, gradients = value_and_grad(MSELoss, argnums=(0))(model.params, pos, box, ene_ref)
            lossprop += loss
            updates, opt_state = optimizer.update(gradients, opt_state)
            model.params = optax.apply_updates(model.params, updates)
        lossprop = jnp.sqrt(lossprop)
        print(lossprop)        
        
        if lossprop < best_loss:
            # save model after each epoch
            model.save_params('params_sgnn.pickle') 
            best_loss = lossprop      

            # evaluate test
            ene_refs = []
            ene_preds = []
            for pos, e in test_loader:
                ene_ref = jnp.array(e.numpy())
                pos = jnp.array(pos.numpy())
                ene_pred = model.batch_forward(pos, box, model.params)
                ene_preds.append(ene_pred)
                ene_refs.append(ene_ref)
            ene_ref = jnp.concatenate(ene_refs)
            ene_ref = ene_ref - jnp.average(ene_ref)
            ene_pred = jnp.concatenate(ene_preds)
            ene_pred = ene_pred - jnp.average(ene_pred)
            err = ene_pred - ene_ref
            test_loss = jnp.sqrt(jnp.average(err**2))

            fout.write("{:5} {:4} {:15} {:5e}  {} ".format("Epoch=",i_epoch,"learning rate",lr,"train error:"))
            fout.write('{:10.5f} '.format(lossprop))
            fout.write('{} '.format("test error:"))
            fout.write('{:10.5f} \n'.format(test_loss))
            fout.flush()
            # print test data
            with open('test_data.xvg', 'w') as f:
                print('# RMSE = %10.5f'%test_loss, file=f)
                for e1, e2 in zip(ene_pred, ene_ref):
                    print(e2, e2, e1, file=f)
            
    fout.write(time.strftime("%Y-%m-%d-%H_%M_%S \n", time.localtime()))
    fout.write("terminated normal\n")
    fout.close()
