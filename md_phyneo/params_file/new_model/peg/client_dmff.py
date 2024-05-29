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

def params_align(params, params0):
    # setting up params for all calculators
    params_ex = {}
    params_sr_es = {}
    params_sr_pol = {}
    params_sr_disp = {}
    params_dhf = {}
    params_dmp_es = {}
    params_dmp_disp = {}
    for k in ['B', 'mScales']:
        params_ex[k] = params[k]
        params_sr_es[k] = params[k]
        params_sr_pol[k] = params[k]
        params_sr_disp[k] = params[k]
        params_dhf[k] = params[k]
        params_dmp_es[k] = params[k]
        params_dmp_disp[k] = params[k]
    params_ex['A'] = params['A_ex']
    params_sr_es['A'] = params['A_es']
    params_sr_pol['A'] = params['A_pol']
    params_sr_disp['A'] = params['A_disp']
    params_dhf['A'] = params['A_dhf']
    # damping parameters
    params_dmp_es['Q'] = params['Q']
    params_dmp_disp['C6'] = params['C6']
    params_dmp_disp['C8'] = params['C8']
    params_dmp_disp['C10'] = params['C10']
    # long range parameters
    params_espol = {}
    for k in ['mScales', 'pScales', 'dScales', 'Q_local', 'pol', 'tholes']:
        params_espol[k] = params[k]
    params_disp = {}
    for k in ['B', 'C6', 'C8', 'C10', 'mScales']:
        params_disp[k] = params[k]

    paramtree = {}
    paramtree['ADMPPmeForce'] = params0['ADMPPmeForce'] # read from the XML file force field
    paramtree['ADMPDispPmeForce'] = params_disp
    paramtree['SlaterExForce'] = params_ex
    paramtree['SlaterSrEsForce'] = params_sr_es
    paramtree['SlaterSrPolForce'] = params_sr_pol
    paramtree['SlaterSrDispForce'] = params_sr_disp
    paramtree['SlaterDhfForce'] = params_dhf
    paramtree['QqTtDampingForce'] = params_dmp_es
    paramtree['SlaterDampingForce'] = params_dmp_disp
    # paramtree['NonbondedForce'] = params0['NonbondedForce']
    return paramtree

class DMFFDriver(driver.BaseDriver):

    def __init__(self, addr, port, pdb, ff_xml, psr1, psr2, psr3, socktype):
        addr = addr + '_%s'%os.environ['SLURM_JOB_ID']
        
        # set up the interface with ipi
        driver.BaseDriver.__init__(self, port, addr, socktype)

        # set up force calculators
        mol = PDBFile(pdb)
        pos = jnp.array(mol.positions._value) * 10
        box = jnp.array(mol.topology.getPeriodicBoxVectors()._value) * 10
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
        rc = 8
        pots = H.createPotential(mol.topology, nonbondedCutoff=rc*angstrom, nonbondedMethod=PME, ethresh=1e-4, step_pol=5)

        # neighbor list
        self.nbl = nblist.NeighborListFreud(box, rc, H.getGenerators()[0].covalent_map)
        self.nbl.allocate(pos, box)
        pairs = self.nbl.pairs

        # load parameters
        with open(psr1, 'rb') as ifile:
            param = pickle.load(ifile)
        with open(psr3, 'rb') as ifile:
            params_eann = pickle.load(ifile)
        params0 = H.getParameters()
        paramtree = params_align(param, params0)

        pot_pme = pots.dmff_potentials['ADMPPmeForce']
        pot_disp = pots.dmff_potentials['ADMPDispPmeForce']
        pot_ex = pots.dmff_potentials['SlaterExForce']
        pot_sr_es = pots.dmff_potentials['SlaterSrEsForce']
        pot_sr_pol = pots.dmff_potentials['SlaterSrPolForce']
        pot_sr_disp = pots.dmff_potentials['SlaterSrDispForce']
        pot_dhf = pots.dmff_potentials['SlaterDhfForce']
        pot_dmp_es = pots.dmff_potentials['QqTtDampingForce']
        pot_dmp_disp = pots.dmff_potentials['SlaterDampingForce']
    
        # set up eann calculators
        pot_eann = EANNForce(n_elem, elem_indices, n_gto=16, rc=6)

        # set up gnn calculators
        G = from_pdb(pdb)
        model = MolGNNForce(G, nn=1)
        model.load_params(psr2)
        
        def dmff_calculator(pos, L, pairs):
            box = jnp.array([[L,0,0],[0,L,0],[0,0,L]])          
            E_es = pot_pme(pos, box, pairs, paramtree)
            E_ex = pot_ex(pos, box, pairs, paramtree)
            E_dmp_es = pot_dmp_es(pos, box, pairs, paramtree) 
            E_sr_es = pot_sr_es(pos, box, pairs, paramtree) 
            E_sr_pol = pot_sr_pol(pos, box, pairs, paramtree) 
            E_disp = pot_disp(pos, box, pairs, paramtree)
            E_dmp_disp = pot_dmp_disp(pos, box, pairs, paramtree) 
            E_sr_disp = pot_sr_disp(pos, box, pairs, paramtree) 
            E_dhf = pot_dhf(pos, box, pairs, paramtree)

        
            E_ml = pot_eann.get_energy(pos, box, pairs, params_eann)
            E_gnn = model.forward(pos, box, model.params)
            E_nb = E_es + E_ex + E_dmp_es + E_sr_es + E_sr_pol + E_disp + E_dmp_disp + E_sr_disp + E_dhf

            return E_nb + E_gnn + E_ml

        # set up various force calculators
        self.calc_dmff = jit(value_and_grad(dmff_calculator,argnums=(0,1)))

        # compile tot_force function
        energy, (grad, virial) = self.calc_dmff(pos, L, pairs)

        return

    def grad(self, crd, cell): # receive SI input, return SI values
        pos = jnp.array(crd*1e10) # convert to angstrom
        box = jnp.array(cell*1e10)      # convert to angstrom
        L = box[0][0]

        # nb list
        self.nbl.update(pos, box)
        pairs = self.nbl.pairs

        energy, (grad, virial) = self.calc_dmff(pos, L, pairs)
        virial = np.diag((-grad * pos).sum(axis=0) - virial*L/3).ravel()

        energy = np.array((energy*kilojoule_per_mole/AVOGADRO_CONSTANT_NA).value_in_unit(joule))
        grad = np.array((grad*kilojoule_per_mole/angstrom/AVOGADRO_CONSTANT_NA).value_in_unit(joule/meter))
        virial = np.array((virial*kilojoule_per_mole/AVOGADRO_CONSTANT_NA).value_in_unit(joule))
        return energy, grad, virial


if __name__ == '__main__':
    # the forces are composed by three parts: 
    # the long range part computed using openmm, parameters in xml
    # the short range part writen by hand, parameters in psr
    fn_pdb = sys.argv[1] # pdb file used to define openmm topology, this one should contain all virtual sites
    ff_xml = sys.argv[2] # xml file that defines the force field
    fn_psr1 = sys.argv[3] # short range parameter file
    fn_psr2 = sys.argv[4] # sgnn parameter file
    fn_psr3 = sys.argv[5] # sgnn parameter file

    addr = sys.argv[6]
    port = int(sys.argv[7])
    socktype = sys.argv[8]

    driver_dmff = DMFFDriver(addr, port, fn_pdb, ff_xml, fn_psr1, fn_psr2, fn_psr3, socktype)
    while True:
        driver_dmff.parse()
