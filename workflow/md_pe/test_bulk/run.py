#!/usr/bin/env python
import sys
import jax
import jax.numpy as jnp
from openmm import *
from openmm.app import *
import openmm.app as app
import openmm.unit as unit
from dmff.api import Hamiltonian
from dmff.common import nblist
from jax import value_and_grad
import pickle

if __name__ == '__main__':
    
    H = Hamiltonian('pe.xml')
    pdb = app.PDBFile("bulk.pdb")
    rc = 0.6
    # generator stores all force field parameters
    pots = H.createPotential(pdb.topology, nonbondedMethod=PME, nonbondedCutoff=rc*unit.nanometer, ethresh=5e-4, step_pol=5)

    # construct inputs
    positions = jnp.array(pdb.positions._value)
    a, b, c = pdb.topology.getPeriodicBoxVectors()
    box = jnp.array([a._value, b._value, c._value])
    # neighbor list
    nbl = nblist.NeighborList(box, rc, pots.meta['cov_map']) 
    nbl.allocate(positions)
    paramset = H.getParameters()
    efunc = jax.jit(value_and_grad(pots.getPotentialFunc(),argnums=(0, 1)))
    print(efunc(positions, box, nbl.pairs, paramset))


