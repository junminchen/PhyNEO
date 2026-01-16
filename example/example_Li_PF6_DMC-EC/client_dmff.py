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
from jax import jit, value_and_grad, vmap
import jax.numpy as jnp

from dmff.sgnn.gnn import MolGNNForce
from dmff.sgnn.graph import TopGraph, from_pdb
from eapnn import *

class DMFFDriver(driver.BaseDriver):

    def __init__(self, addr, port, socktype):
        #addr = addr + '_%s'%os.environ['SLURM_JOB_ID']
        # set up the interface with ipi
        driver.BaseDriver.__init__(self, port, addr, socktype)

        pdb, ff_xml, psr, psr_, psr1 = 'init.pdb', 'phyneo_ecl.xml', 'params_sgnn.pickle', 'params_sgnn_ABn.pickle', 'params_ml.pickle'
        residue_names = ['PF6', 'DFP', 'BF4']

        mol = PDBFile(pdb) 
        pos = jnp.array(mol.positions._value) 
        box = jnp.array(mol.topology.getPeriodicBoxVectors()._value)
        L = box[0][0]
        
        rc = 0.6
        H = Hamiltonian(ff_xml)
        pots = H.createPotential(mol.topology, nonbondedCutoff=rc*nanometer, nonbondedMethod=PME, ethresh=1e-4, step_pol=10)
        efunc_nb = pots.getPotentialFunc()
        params_nb = H.getParameters()

        # neighbor list
        self.nbl = nblist.NeighborListFreud(box, rc, pots.meta['cov_map'])
        # self.nbl.capacity_multiplier = 500000 # avoid pairs leaking
        self.nbl.allocate(pos, box)
        self.pairs = self.nbl.pairs


        # define atomic symbols and corresponding indexes
        atom_elements = []
        for atom in mol.topology.atoms():
            atom_elements.append(atom.element.atomic_number)
        z_atomnum = jnp.array(atom_elements)

        zindex = [1, 3, 5, 6, 7, 8, 9, 11, 15, 16]
        n_atype = len(zindex)
        z_atomnum_list = [float(num) for num in np.array(z_atomnum)]
        zindex_dict = {float(num): i for i, num in enumerate(zindex)}
        self.atype_indices = jnp.array([zindex_dict.get(num, -1) for num in z_atomnum_list])

        mol_ID = []
        for atom in mol.topology.atoms():
            mol_ID.append(atom.residue.index)
        mol_ID = jnp.array(mol_ID)

        topo_nblist, topo_mask = get_topology_neighbors(pdb, connectivity=4, max_neighbors=20, max_n_atoms=None)

        n_atoms = len(pos)
        atomic_nums = jnp.array([atom.element.atomic_number for atom in mol.topology.atoms()], dtype=int)
        # 标记Li(3)和Na(11)原子
        target_mask = (atomic_nums == 3) | (atomic_nums == 11)
        target_indices = jnp.where(target_mask)[0]
        self.max_pairs = len(target_indices)*100

        self.valid_pairs, self.valid_mask = filter_and_pad_pairs(self.pairs, self.atype_indices, max_pairs=self.max_pairs)


        model_nb = EAPNNForce(
            n_atoms=n_atoms, 
            n_atype=n_atype, 
            rc=6.0,  
            acsf_nmu=20,
            apsf_nmu=20,
            acsf_eta=100,
            apsf_eta=50
        )

        key = jax.random.PRNGKey(0)
        model_nb.init(key, pos*10, box*10, self.valid_pairs, self.valid_mask, topo_nblist, topo_mask, mol_ID, self.atype_indices)

        with open(psr1, 'rb') as ifile:
            params = pickle.load(ifile)	

        # 检查Topology中是否存在特定的残基名
        def has_residue(topology, residue_name):
            for residue in topology.residues():
                if residue.name in residue_names:
                    print(f"Topology contains residue named {residue.name}.")
                    return True
            return False

        # 判断Topology是否有该残基名
        if has_residue(mol.topology, residue_names):
            residues = []
            for atom in mol.topology.atoms():
                residues.append(atom.residue.name)            
            enumerated_strings = list(enumerate(residues))
            # 找到特定元素的所有索引
            target_indices = [index for index, string in enumerated_strings if string in residue_names]
            residue_name = residues[target_indices[0]]

            # set up gnn calculators
            G = from_pdb('init_remaining.pdb')
            model = MolGNNForce(G, nn=1)
            with open(psr, 'rb') as ifile:
                params_bond = pickle.load(ifile)

            # set up gnn calculators
            G_ = from_pdb('init_extracted.pdb')
            G_pf6 = from_pdb(f'pdb_bank/{residue_name}.pdb')
            model_ = MolGNNForce(G_pf6, nn=0, max_valence=6)
            with open(psr_, 'rb') as ifile:
                params_bond_ = pickle.load(ifile)
            model_.batch_forward = vmap(model_.forward, in_axes=(0, None, None), out_axes=(0))

            def dmff_calculator(pos, L, pairs, valid_pairs, valid_mask, atype_indices):
                box = jnp.array([[L,0,0],[0,L,0],[0,0,L]])          
                E_nb = efunc_nb(pos, box, pairs, params_nb)
                pos_ABn = pos[target_indices[0]:target_indices[-1]+1]
                pos_else = jnp.concatenate((pos[:target_indices[0]],pos[target_indices[-1]+1:]), axis=0)
                pos_ABn = pos_ABn.reshape((int(G_.positions.shape[0]/G_pf6.positions.shape[0]), 
                                                        G_pf6.positions.shape[0], 
                                                        G_pf6.positions.shape[1]))
                E_bond_ = jnp.sum(model_.batch_forward(pos_ABn*10, box*10, params_bond_))
                E_bond = model.forward(pos_else*10, box*10, params_bond)
                # E_nb_ml = model_nb.apply(params, pos*10, box*10, valid_pairs, valid_mask, topo_nblist, topo_mask, mol_ID, atype_indices)
                return E_nb+E_bond+E_bond_#+E_nb_ml

        else:
            print(f"Topology does not contain ABn residue.")
            G = from_pdb('init.pdb')
            model = MolGNNForce(G, nn=1)
            with open(psr, 'rb') as ifile:
                params_bond = pickle.load(ifile)
                
            def dmff_calculator(pos, L, pairs, valid_pairs, valid_mask, atype_indices):
                box = jnp.array([[L,0,0],[0,L,0],[0,0,L]])          
                E_nb = efunc_nb(pos, box, pairs, params_nb)
                E_bond = model.forward(pos*10, box*10, params_bond)
                #E_nb_ml = model_nb.apply(params, pos*10, box*10, valid_pairs, valid_mask, topo_nblist, topo_mask, mol_ID, atype_indices)
                return E_nb+E_bond#+E_nb_ml

        self.calc_dmff = jit(value_and_grad(dmff_calculator,argnums=(0, 1)))

        # compile tot_force function
        energy, (grad, virial) = self.calc_dmff(pos, L, self.pairs, self.valid_pairs, self.valid_mask, self.atype_indices)
        print(energy, grad, virial)
        return

    def grad(self, crd, cell): # receive SI input, return SI values
        pos = np.array(crd*1e9) # convert to nanometer
        box = np.array(cell*1e9) # convert to nanometer
        L = box[0][0]

        # nb list
        self.nbl.update(pos, box)
        pairs = self.nbl.pairs
        valid_pairs, valid_mask = filter_and_pad_pairs(pairs, self.atype_indices, max_pairs=self.max_pairs)

        energy, (grad, virial) = self.calc_dmff(pos, L, pairs, valid_pairs, valid_mask, self.atype_indices)
        virial = np.diag((-grad * pos).sum(axis=0) - virial*L/3).ravel()

        energy = np.array((energy*kilojoule_per_mole/AVOGADRO_CONSTANT_NA).value_in_unit(joule))
        grad = np.array((grad*kilojoule_per_mole/nanometer/AVOGADRO_CONSTANT_NA).value_in_unit(joule/meter))
        virial = np.array((virial*kilojoule_per_mole/AVOGADRO_CONSTANT_NA).value_in_unit(joule))
        return energy, grad, virial


if __name__ == '__main__':
    # the forces are composed by three parts: 
    # the long range part computed using openmm, parameters in xml
    # the short range part writen by hand, parameters in psr
    addr = sys.argv[1]
    port = int(sys.argv[2])
    socktype = sys.argv[3]

    driver_dmff = DMFFDriver(addr, port, socktype)
    while True:
        driver_dmff.parse()
