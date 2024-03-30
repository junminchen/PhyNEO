#!/usr/bin/env python
import numpy as np

import jax
import jax.numpy as jnp
from jax import value_and_grad, vmap, jit

from openmm.app import PDBFile
from openmm.unit import angstrom
from openmm.app import CutoffPeriodic
from functools import partial
import pickle

from dmff.api import Hamiltonian
from dmff.utils import jit_condition
from dmff.common import nblist

import time 
import optax
import sys 

def padding(i):
    s = '%d'%i
    while len(s) < 3:
        s = '0' + s
    return s

def params_convert(params):
    params_ex = {}
    params_sr_es = {}
    params_sr_pol = {}
    params_sr_disp = {}
    params_dhf = {}
    params_dmp_es = {}  # electrostatic damping
    params_dmp_disp = {} # dispersion damping
    for k in ['B']:
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
    p = {}
    p['SlaterExForce'] = params_ex
    p['SlaterSrEsForce'] = params_sr_es
    p['SlaterSrPolForce'] = params_sr_pol
    p['SlaterSrDispForce'] = params_sr_disp
    p['SlaterDhfForce'] = params_dhf
    p['QqTtDampingForce'] = params_dmp_es
    p['SlaterDampingForce'] = params_dmp_disp
    return p

class BasePairs:
    def __init__(self, ff, pdb, pdb_A, pdb_B):
        pdb = PDBFile(pdb)
        pdb_A = PDBFile(pdb_A)
        pdb_B = PDBFile(pdb_B)
        self.H = Hamiltonian(ff)
        self.H_A = Hamiltonian(ff)
        self.H_B = Hamiltonian(ff)
        self.pots = self.H.createPotential(pdb.topology, nonbondedCutoff=25*angstrom, nonbondedMethod=CutoffPeriodic, ethresh=1e-4, step_pol=5)
        self.generators = self.H.getGenerators()        
        self.pots_A = self.H_A.createPotential(pdb_A.topology, nonbondedCutoff=25*angstrom, nonbondedMethod=CutoffPeriodic, ethresh=1e-4, step_pol=5)
        self.generators_A = self.H_A.getGenerators()        
        self.pots_B = self.H_B.createPotential(pdb_B.topology, nonbondedCutoff=25*angstrom, nonbondedMethod=CutoffPeriodic, ethresh=1e-4, step_pol=5)
        self.generators_B = self.H_B.getGenerators()

        self.pos = jnp.array(pdb.positions._value) * 10
        self.pos_A = jnp.array(pdb_A.positions._value) * 10
        self.pos_B = jnp.array(pdb_B.positions._value) * 10

        self.box = jnp.array(pdb.topology.getPeriodicBoxVectors()._value) * 10
        self.rc = 24
        self.nblist = nblist.NeighborList(self.box, self.rc, self.pots.meta['cov_map'])
        self.nblist_A = nblist.NeighborList(self.box, self.rc, self.pots_A.meta['cov_map'])
        self.nblist_B = nblist.NeighborList(self.box, self.rc, self.pots_B.meta['cov_map'])
        self.nblist.allocate(self.pos)
        self.nblist_A.allocate(self.pos_A)
        self.nblist_B.allocate(self.pos_B)
        self.pairs = self.nblist.pairs
        self.pairs_A = self.nblist_A.pairs
        self.pairs_B = self.nblist_B.pairs
        self.pairs_AB = self.pairs[self.pairs[:, 0] < self.pairs[:, 1]]
        self.pairs_A = self.pairs_A[self.pairs_A[:, 0] < self.pairs_A[:, 1]]
        self.pairs_B = self.pairs_B[self.pairs_B[:, 0] < self.pairs_B[:, 1]]

        self.potentials_names = ['es', 'disp']
        self.potentials_mapping = {
            'es': 'ADMPPmeForce',
            'disp': 'ADMPDispPmeForce',
        }
        
        for potentials_name in self.potentials_names:
            print(potentials_name)
            setattr(self, f'pots_{potentials_name}', self.pots.dmff_potentials[self.potentials_mapping[potentials_name]])
            setattr(self, f'pots_{potentials_name}_A', self.pots_A.dmff_potentials[self.potentials_mapping[potentials_name]])
            setattr(self, f'pots_{potentials_name}_B', self.pots_B.dmff_potentials[self.potentials_mapping[potentials_name]])

    def cal_E(self, params, pos_A, pos_B):
        # get position array
        pos_A *= 0.1
        pos_B *= 0.1
        pos_AB = jnp.concatenate([pos_A, pos_B], axis=0)
        box = self.box
        #####################
        # electrostatic + pol
        #####################
        E_espol_A = self.pots_es_A(pos_A, box, self.pairs_A, params)
        E_espol_B = self.pots_es_B(pos_B, box, self.pairs_B, params)
        E_espol = self.pots_es(pos_AB, box, self.pairs_AB, params) \
                    - E_espol_A \
                    - E_espol_B

        ###################################
        # use induced dipole of monomers to compute electrostatic interaction
        ###################################
        pme_generator_AB = self.generators[0]
        pme_generator_A = self.generators_A[0]
        pme_generator_B = self.generators_B[0]
        U_ind_AB = jnp.vstack((pme_generator_A.pme_force.U_ind, pme_generator_B.pme_force.U_ind))        
        params_pme = params['ADMPPmeForce']
        map_atypes = self.pots.meta['ADMPPmeForce_map_atomtype']
        map_poltypes = self.pots.meta['ADMPPmeForce_map_poltype']
        Q_local = params_pme['Q_local'][map_atypes]
        pol = params_pme['pol'][map_poltypes]
        tholes = params_pme['thole'][map_poltypes]
        pme_force = pme_generator_AB.pme_force
        E_nonpol_AB = pme_force.energy_fn(pos_AB*10, box*10, self.pairs_AB, Q_local, U_ind_AB, pol, tholes, \
                    pme_generator_AB.mScales, pme_generator_AB.pScales, pme_generator_AB.dScales)
        E_es = E_nonpol_AB - E_espol_A - E_espol_B

        ###################################
        # polarization (induction) energy
        ###################################
        E_pol = E_espol - E_es

        #############
        # dispersion
        #############
        E_disp = self.pots_disp(pos_AB, box, self.pairs_AB, params) \
                - self.pots_disp_A(pos_A, box, self.pairs_A, params) \
                - self.pots_disp_B(pos_B, box, self.pairs_B, params)

        return E_es, E_pol, E_disp

if __name__ == '__main__':
    params = Hamiltonian('dmff_forcefield.xml').getParameters()

    data = {}

    with open('data.pickle', 'rb') as ifile:
        data['Pairs_pe_dimer'] = pickle.load(ifile)

    pair_classes = [
        ("_pe_dimer", "pe6_dimer.pdb", "pe6.pdb", "pe6.pdb"),
        # Add other definition
    ]

    class_instances = {}
    # Loop to create subclasses and add them to the global namespace
    for pair_class_name, dimer_file, monomer_A_file, monomer_B_file in pair_classes:
        class_definition = f"""
class Pairs{pair_class_name}(BasePairs):
    def __init__(self):
        super().__init__('dmff_forcefield.xml', 'dimer/{dimer_file}', 'monomer/{monomer_A_file}', 'monomer/{monomer_B_file}')
    """
        exec(class_definition)

        # Instantiate the class and add it to the dictionary
        class_instances[f'Pairs{pair_class_name}'] = globals()[f'Pairs{pair_class_name}']()

    cal_energy = {}
    for class_name, class_instance in class_instances.items():
        cal_energy[class_name] = jit(vmap(class_instance.cal_E, in_axes=(None, 0, 0), out_axes=(0, 0, 0)))
    
    batch = padding(0)
    for key in cal_energy:
        print(cal_energy[key](params, data[key][batch]['posA'], data[key][batch]['posB']))

    from tqdm import tqdm

    # Initialize the scan_res_lr dictionary
    data_lr = {}

    # Loop through keys in the data dictionary
    for key in tqdm(data.keys()):
        
        # Initialize the scan_res_lr dictionary for the current key
        data_lr[key] = {}
        
        for sid in data[key].keys():
            scan_res = data[key][sid]
            scan_res['tot_full'] = scan_res['tot'].copy()
            npts = len(scan_res['tot'])
            
            # Initialize arrays in scan_res_lr for the current key and sid
            data_lr[key][sid] = {
                'es': np.zeros(npts),
                'pol': np.zeros(npts),
                'disp': np.zeros(npts),
                'tot': np.zeros(npts)
            }
            
            # Calculate energy values
            E_es, E_pol, E_disp = cal_energy[key](params, scan_res['posA'], scan_res['posB'])
            
            # Loop through and process each energy component
            for component in ['es', 'pol', 'disp', 'tot']:
                # Remove long range
                scan_res[component] -= E_es if component == 'es' else 0
                scan_res[component] -= E_pol if component == 'pol' else 0
                scan_res[component] -= E_disp if component == 'disp' else 0
                scan_res[component] -= (E_es + E_pol + E_disp) if component == 'tot' else 0
                
                # Save long range
                data_lr[key][sid][component] = E_es if component == 'es' else \
                                                    E_pol if component == 'pol' else \
                                                    E_disp if component == 'disp' else \
                                                    (E_es + E_pol + E_disp)
    with open('data_sr.pickle', 'wb') as ofile:
        pickle.dump(data, ofile)

    with open('data_lr.pickle', 'wb') as ofile:
        pickle.dump(data_lr, ofile)