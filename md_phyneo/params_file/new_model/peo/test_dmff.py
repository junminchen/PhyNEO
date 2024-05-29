#!/usr/bin/env python3
import os
import sys
import driver
import numpy as np
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
from graph import TopGraph, from_pdb
from gnn import MolGNNForce
from eann import EANNForce
import pickle
from jax.config import config


config.update("jax_enable_x64", True)


pdb, ff_xml, psr1, psr2, psr3 = 'init.pdb', 'output.xml', \
                                '_params.pickle', \
                                '_params_sgnn.pickle', \
                                '_params_eann.pickle'

# set up force calculators
mol = PDBFile(pdb)
pos = jnp.array(mol.positions._value) 
box = jnp.array(mol.topology.getPeriodicBoxVectors()._value) 
atomtype = ['H', 'C', 'O']
n_elem = len(atomtype)
species = []
# Loop over all atoms in the topology
for atom in mol.topology.atoms():
    # Get the element of the atom
    element = atom.element.symbol
    mass = atom.element.mass
    species.append(atomtype.index(atom.element.symbol))
elem_indices = jnp.array(species)
L = box[0][0]

H = Hamiltonian(ff_xml)
rc = 0.6
pots = H.createPotential(mol.topology, nonbondedCutoff=rc*nanometer, nonbondedMethod=PME, ethresh=1e-4, step_pol=5)
params = H.getParameters()

# neighbor list
nbl = nblist.NeighborListFreud(box, rc, pots.meta['cov_map'])
nbl.capacity_multiplier = 1000000
nbl.allocate(pos, box)
pairs = nbl.pairs

# load parameters
with open(psr1, 'rb') as ifile:
    param = pickle.load(ifile)
with open(psr3, 'rb') as ifile:
    params_eann = pickle.load(ifile)

# set up eann calculators
pot_eann = EANNForce(n_elem, elem_indices, n_gto=16, rc=4)

# set up gnn calculators
G = from_pdb(pdb)
model = MolGNNForce(G, nn=1)
model.load_params(psr2)

def dmff_calculator(pos, L, pairs):
    box = jnp.array([[L,0,0],[0,L,0],[0,0,L]])          
    E_ml = pot_eann.get_energy(pos*10, box*10, pairs, params_eann)
    E_gnn = model.forward(pos*10, box*10, model.params)
    E_nb = pots.getPotentialFunc()(pos, box, pairs, params)        

    return E_nb, E_ml, E_gnn

print(dmff_calculator(pos, L, pairs))

# # set up various force calculators
# calc_dmff = jit(value_and_grad(dmff_calculator,argnums=(0,1)))

# # compile tot_force function
# energy, (grad, virial) = calc_dmff(pos, L, pairs)