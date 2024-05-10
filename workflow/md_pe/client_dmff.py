#!/usr/bin/env python3
import os
import sys
import driver
import numpy as np
import openmm
from openmm import *
from openmm.app import *
from openmm.unit import *
import pickle

from dmff.api import Hamiltonian
from dmff.common import nblist
from jax import jit, value_and_grad
import jax.numpy as jnp

class DMFFDriver(driver.BaseDriver):

    def __init__(self, addr, port, pdb, ff_xml, psr, socktype):
        #addr = addr + '_%s'%os.environ['SLURM_JOB_ID']
        # set up the interface with ipi
        driver.BaseDriver.__init__(self, port, addr, socktype)

        pdb, ff_xml = 'bulk.pdb', 'pe.xml'

        mol = PDBFile(pdb) 
        self.topology = mol.topology
        pos = jnp.array(mol.positions._value) 
        box = jnp.array(mol.topology.getPeriodicBoxVectors()._value)
        L = box[0][0]
        
        rc = 0.6
        H = Hamiltonian(ff_xml)
        pots = H.createPotential(mol.topology, nonbondedCutoff=rc*nanometer, nonbondedMethod=PME, ethresh=1e-4, step_pol=5)
        efunc_nb = pots.getPotentialFunc()
        self.params = H.getParameters()
        
        # neighbor list
        self.nbl = nblist.NeighborList(box, rc, pots.meta['cov_map'])
        self.nbl.capacity_multiplier = 5000000
        self.nbl.allocate(pos, box)
        pairs = self.nbl.pairs
        pots.getPotentialFunc()

        def dmff_calculator(pos, L, pairs):
            box = jnp.array([[L,0,0],[0,L,0],[0,0,L]])  
            e = pots.getPotentialFunc()(pos, box, pairs, self.params)        
            return e

        self.calc_dmff = jit(value_and_grad(dmff_calculator,argnums=(0, 1)))

        # compile tot_force function
        energy, (grad, virial) = self.calc_dmff(pos, L, pairs)
        print(energy)
        return

    def grad(self, crd, cell): # receive SI input, return SI values
        pos = np.array(crd*1e9) # convert to nanometer
        box = np.array(cell*1e9) # convert to nanometer
        L = box[0][0]

        # nb list
        self.nbl.update(pos, box)
        pairs = self.nbl.pairs

        energy, (grad, virial) = self.calc_dmff(pos, L, pairs)
        virial = np.diag((-grad * pos).sum(axis=0) - virial*L/3).ravel()

        energy = np.array((energy*kilojoule_per_mole/AVOGADRO_CONSTANT_NA).value_in_unit(joule))
        grad = np.array((grad*kilojoule_per_mole/nanometer/AVOGADRO_CONSTANT_NA).value_in_unit(joule/meter))
        virial = np.array((virial*kilojoule_per_mole/AVOGADRO_CONSTANT_NA).value_in_unit(joule))
        return energy, grad, virial


if __name__ == '__main__':
    # the forces are composed by three parts: 
    # the long range part computed using openmm, parameters in xml
    # the short range part writen by hand, parameters in psr
    fn_pdb = sys.argv[1] # pdb file used to define openmm topology, this one should contain all virtual sites
    ff_xml = sys.argv[2] # xml file that defines the force field
    fn_psr = sys.argv[3] # sgnn parameter file

    addr = sys.argv[4]
    port = int(sys.argv[5])
    socktype = sys.argv[6]

    driver_dmff = DMFFDriver(addr, port, fn_pdb, ff_xml, fn_psr, socktype)
    while True:
        driver_dmff.parse()
