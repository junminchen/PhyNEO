#!/usr/bin/env python3
import os
import sys
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

# from dmff.sgnn.gnn import MolGNNForce
from gnn import MolGNNForce
# from dmff.sgnn.graph import TopGraph, from_pdb
from graph import TopGraph, from_pdb

from ase import Atoms
from ase.io import read, Trajectory, write
from ase.calculators.calculator import Calculator, all_changes
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.stress import full_3x3_to_voigt_6_stress

from ase import units
from ase.md.npt import NPT
from ase.md.nptberendsen import NPTBerendsen
from ase.md.langevin import Langevin
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md import MDLogger
from ase.io.trajectory import Trajectory

from jax import config
config.update("jax_debug_nans", True)

import jax
from jax.lib import xla_bridge 
print(jax.devices()[0]) 
print(xla_bridge.get_backend().platform)

import time

# 自定义 ASE Calculator 类
class DMFFCalculator(Calculator):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.implemented_properties = ['energy', 'forces', 'stress']
        self.results = {}

        pdb, ff_xml, psr, psr_ = 'init.pdb', 'phyneo_ecl.xml', 'params_sgnn.pickle', 'params_sgnn_ABn.pickle'

        mol = PDBFile(pdb) 
        self.topology = mol.topology
        pos = jnp.array(mol.positions._value) 
        box = jnp.array(mol.topology.getPeriodicBoxVectors()._value)
        L = box[0][0]
        
        rc = 0.6
        H = Hamiltonian(ff_xml)
        pots = H.createPotential(mol.topology, nonbondedCutoff=rc*nanometer, nonbondedMethod=PME, ethresh=1e-4, step_pol=5)
        efunc_nb = pots.getPotentialFunc()
        params_nb = H.getParameters()
        
        # neighbor list
        self.nbl = nblist.NeighborListFreud(box, rc, pots.meta['cov_map'])
        self.nbl.allocate(pos, box)
        pairs = self.nbl.pairs
        
        # set up gnn calculators
        G = from_pdb('init.pdb')
        model = MolGNNForce(G, nn=1)
        with open(psr, 'rb') as ifile:
            params_bond = pickle.load(ifile)

        def dmff_calculator(pos, L, pairs):
            box = jnp.array([[L,0,0],[0,L,0],[0,0,L]])          
            E_nb = efunc_nb(pos, box, pairs, params_nb)
            E_bond = model.forward(pos*10, box*10, params_bond)
            return E_nb+E_bond

        self.calc_dmff = jit(value_and_grad(dmff_calculator,argnums=(0, 1)))

        # compile tot_force function
        energy, (grad, virial) = self.calc_dmff(pos, L, pairs)
        print(energy, grad, virial)
        return

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        positions = jnp.array(atoms.get_positions()) / 10.0 
        L = atoms.get_cell()[0, 0] / 10.0 
        box = atoms.get_cell() / 10.0
        print(atoms.get_cell())

        box = jnp.array([[L, 0, 0], [0, L, 0], [0, 0, L]])
        self.nbl.update(positions, box)
        pairs = self.nbl.pairs

        energy, (grad, virial) = self.calc_dmff(positions, L, pairs)
        # 3x3 virial tensor
        virial = np.diag((-grad * positions).sum(axis=0) - virial*L/3) #.ravel()
        volume = np.linalg.det(box)
        
        self.results['energy'] = energy.item() * 0.010364  # kj/mol to eV
        self.results['forces'] = np.array(-grad) * 0.010364 / 10.0
        stress = -full_3x3_to_voigt_6_stress(virial/volume) * 0.010364 / 1000 # eV/Å³
        self.results['stress'] = stress

        return self.results
        
def print_energy(a, ofile='energy_output.txt'):
    """Function to print the potential, kinetic, total energy and density to a file."""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    cell_volume = a.get_volume()  # Å³
    total_mass = sum(a.get_masses())  # 原子质量单位
    density = total_mass * 1.66053907e-24 / (cell_volume * 1e-24)  # g/cm³
    
    energy_str = 'Epot = %.4feV  Ekin = %.4feV (T=%3.0fK)  Etot = %.4feV  Density = %.4f g/cm³\n' % \
                 (epot, ekin, ekin / (1.5 * units.kB), epot + ekin, density)
    
    with open(ofile, 'a') as f:
        f.write(energy_str)
    
    print(energy_str, end='')


if __name__ == '__main__':
    pdb = 'init.pdb'
    L = 25.689
    atoms = read(pdb)
    atoms.set_cell([L, L, L])  
    atoms.set_pbc([True, True, True])  

    # atoms.set_calculator(DMFFCalculator())
    time1=time.time()
    atoms.calc = DMFFCalculator()

    print(atoms.get_potential_energy())
    print(atoms.get_forces())
    print(atoms.get_stress())

    time2=time.time()
    print('time',time2-time1)

    temperature = 298.15
    pressure = 1.01325 * units.bar
    # Set initial velocities using Maxwell-Boltzmann distribution
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)

    timestep = 1.0 * units.fs  # 时间步长为 1 飞秒
    friction = 0.02  # 摩擦系数
    # dyn = Langevin(atoms, timestep, temperature_K=temperature, friction=friction)
    # dyn = NoseHooverChainNVT(atoms, timestep, temperature_K=temperature, tdamp=100*timestep)

    # Define the NPT ensemble
    NPTdamping_timescale = 1000 * units.fs  # Time constant for NPT dynamics
    NVTdamping_timescale = 1000 * units.fs  # Time constant for NVT dynamics (NPT includes both)
    # dyn = NPT(atoms, timestep=1.0 * units.fs, temperature_K=temperature,
    #         ttime=100 * units.fs, #NVTdamping_timescale, 
    #         pfactor=100000 * units.fs, #0.1*NPTdamping_timescale**2 #None,
    #         externalstress=6.2415e-6) #0.0) #1bar to eV/A3; 6.3242e-6 1atom to eV/A3 
    # dyn = NPTBerendsen(atoms, timestep=1 * units.fs, temperature_K=temperature,
    #                 taut=NVTdamping_timescale, pressure_au=0.0,
    #                 taup=NPTdamping_timescale, compressibility_au=1.)

    dyn = NPTBerendsen(atoms, timestep=timestep, temperature_K=temperature,
                    taut=100 * units.fs, pressure_au=pressure,
                    taup=1000 * units.fs, compressibility_au=4.57e-5 / units.bar)

    dyn.attach(MDLogger(dyn, atoms, 'npt.log', header=True, stress=True,
            peratom=False, mode="w"), interval=100)    

    def write_frame():
        dyn.atoms.write('npt.xyz', append=True)
    dyn.attach(write_frame, interval=500)

    traj = Trajectory('npt.traj', 'w', atoms)
    dyn.attach(traj.write, interval=500)
    n_steps = 500
    for step in range(n_steps):
        print_energy(atoms)  
        dyn.run(100)
    traj.close()

    # dyn = NPTBerendsen(atoms, timestep=1 * units.fs, temperature_K=temperature,
    #                   taut=NVTdamping_timescale, pressure_au=1.0 * units.bar,
    #                   taup=NPTdamping_timescale, compressibility_au=1.)




