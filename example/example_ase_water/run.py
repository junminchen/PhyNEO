#!/usr/bin/env python
import sys
import jax
import jax.numpy as jnp
import openmm.app as app
import openmm.unit as unit
from dmff.api import Hamiltonian
from dmff.common import nblist
from jax import value_and_grad
import pickle
from jax import config
config.update("jax_enable_x64", True)
if __name__ == '__main__':
   
    rc = 0.6
    H = Hamiltonian('ff.xml')
    app.Topology.loadBondDefinitions("residues.xml")
    pdb = app.PDBFile("init.pdb")
    # generator stores all force field parameters
    pots = H.createPotential(pdb.topology, ethresh=1e-4, step_pol=5)

    # construct inputs
    positions = jnp.array(pdb.positions._value)
    a, b, c = pdb.topology.getPeriodicBoxVectors()
    box = jnp.array([a._value, b._value, c._value])
    # neighbor list
    nbl = nblist.NeighborList(box, rc, pots.meta['cov_map']) 
    nbl.allocate(positions)

  
    paramset = H.getParameters()
    # params = paramset.parameters

    efunc = jax.jit(pots.getPotentialFunc())
    print(efunc(positions, box, nbl.pairs, paramset))

