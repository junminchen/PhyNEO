#!/usr/bin/env python3
import os
import sys
import numpy as np
import pickle
import re

import jax
import jax.numpy as jnp
from jax import jit, vmap, value_and_grad
import openmm
from openmm import *
from openmm.app import *
from openmm.unit import *
import dmff
from dmff.api import Hamiltonian
from dmff.common import nblist
from dmff.utils import jit_condition

data_file = sys.argv[1]
pdbfile = sys.argv[2]
# load data.pickle
with open(data_file, 'rb') as ifile:
    data = pickle.load(ifile)

mol = PDBFile(pdbfile)
pos = jnp.array(mol.positions._value) 
box = jnp.array(mol.topology.getPeriodicBoxVectors()._value)
L = box[0][0]

rc = 2.4
H = Hamiltonian('output.xml')
pots = H.createPotential(mol.topology, nonbondedCutoff=rc*nanometer, nonbondedMethod=PME, ethresh=1e-4, step_pol=5)
efunc_nb = pots.getPotentialFunc()
params_nb = H.getParameters()

# neighbor list
nbl = nblist.NeighborListFreud(box, rc, pots.meta['cov_map'])
nbl.allocate(pos, box)
pairs = nbl.pairs

@jit_condition(static_argnums=())
def cal_dmff(pos):
    E_nb = efunc_nb(pos, box, pairs, params_nb)
    return E_nb

E_tot = cal_dmff(pos)
print(E_tot)
data['tot_full'] = data['energies'].copy()
data['ff'] = data['energies'].copy()

for i in range(len(data['energies'])):
    pos = data['positions'][i]
    E_nb = cal_dmff(pos*0.1)
    data['ff'][i] = E_nb
    data['energies'][i] -= E_nb
    print(E_nb)

# write it down.
filename = re.sub('.pickle','',data_file)
with open(filename+'_remove_nb.pickle', 'wb') as ofile:
    pickle.dump(data, ofile)
