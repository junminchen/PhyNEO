
import time 
import optax
import sys 
import os 
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
import jax.numpy as jnp
from openmm.app import PDBFile, CutoffPeriodic, PME
from openmm.unit import angstrom
from dmff.api import Hamiltonian
from dmff.common import nblist

class DMFFEnergyCalculator:
    def __init__(self, ff_file, pdb_file):
        self.ff = ff_file
        self.pdb = PDBFile(pdb_file)
        self.positions = jnp.array(self.pdb.positions._value)
        self.box = jnp.eye(3) * 6.0
        self.rc = 2.5

        # Build Hamiltonian and Potential
        self.H = Hamiltonian(self.ff)
        self.potentials_obj = self.H.createPotential(
            self.pdb.topology,
            nonbondedCutoff=25 * angstrom,
            nonbondedMethod=CutoffPeriodic,
            ethresh=1e-4,
            step_pol=20
        )
        self.params = self.H.getParameters()

        # Neighbor list
        self.nblist = nblist.NeighborList(
            self.box,
            self.rc,
            self.potentials_obj.meta['cov_map']
        )
        self.nblist.allocate(self.positions)
        self.pairs = self.nblist.pairs
        self.pairs = self.pairs[self.pairs[:, 0] < self.pairs[:, 1]]

        # Define potential keys and mapping
        self.potentials_keys = [
            'espol', 'disp', 'ex', 'sr_es', 'sr_pol', 'sr_disp', 'dhf', 'dmp_es', 'dmp_disp'
        ]
        self.potentials_mapping = {
            'espol': 'ADMPPmeForce',
            'disp': 'ADMPDispPmeForce',
            'ex': 'SlaterExForce',
            'sr_es': 'SlaterSrEsForce',
            'sr_pol': 'SlaterSrPolForce',
            'sr_disp': 'SlaterSrDispForce',
            'dhf': 'SlaterDhfForce',
            'dmp_es': 'QqTtDampingForce',
            'dmp_disp': 'SlaterDampingForce'
        }

    def compute_components(self):
        energy_dict = {}
        for key in self.potentials_keys:
            force_name = self.potentials_mapping[key]
            potential_func = self.potentials_obj.getPotentialFunc(force_name)
            energy = potential_func(self.positions, self.box, self.pairs, self.params)
            energy_dict[key] = energy
        return energy_dict

    def compute_total(self):
        etotal = self.potentials_obj.getPotentialFunc()
        return etotal(self.positions, self.box, self.pairs, self.params)

if __name__ == '__main__':

    # 实例化并计算
    # calc = DMFFEnergyCalculator('peo.xml', 'peo3.pdb')
    calc = DMFFEnergyCalculator('EC.xml', 'EC.pdb')

    # calc = DMFFEnergyCalculator('peo.xml', 'init.pdb')

    # 获取各项分能量
    energy_components = calc.compute_components()
    for k, v in energy_components.items():
        print(f"{k}: {v}")

    # 获取总能量（包含所有项）
    total_energy = calc.compute_total()
    print(f"Dispersion+Damping Energy: {total_energy - energy_components['espol']}")

    print(f"Total Energy: {total_energy}")
