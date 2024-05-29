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

class DMFFDriver(driver.BaseDriver):

    def __init__(self, addr, port, pdb, ff_xml, psr, socktype):
        #addr = addr + '_%s'%os.environ['SLURM_JOB_ID']
        # set up the interface with ipi
        driver.BaseDriver.__init__(self, port, addr, socktype)


        pdb, ff_xml, psr1, psr2, psr3 = 'init.pdb', 'peo.xml', \
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
        self.nbl = nblist.NeighborListFreud(box, rc, pots.meta['cov_map'])
        self.nbl.capacity_multiplier = 1000000
        self.nbl.allocate(pos, box)
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
