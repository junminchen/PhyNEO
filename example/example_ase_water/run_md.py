#!/usr/bin/env python
#from dmff.mdtools.asetools import *
from asetools import *
from ase import units
from ase.md.npt import NPT
from ase.md.nptberendsen import NPTBerendsen
from ase.md.langevin import Langevin
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md import MDLogger
from ase.io import read, write
from ase.optimize import BFGS
from ase.io.trajectory import Trajectory
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from ase.constraints import FixBondLengths
import jax.numpy as jnp

from jax import config
config.update("jax_debug_nans", True)

import jax
from jax.lib import xla_bridge 
print(jax.devices()[0]) 
print(xla_bridge.get_backend().platform)

import time
import argparse


def print_energy(a, step, time_ps, ofile='energy_output.txt'):
    """Print system energy, temperature, density and time information"""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    etot = epot + ekin
    temp = ekin / (1.5 * units.kB)
    
    # Calculate density
    cell_volume = a.get_volume()  # Å³
    total_mass = sum(a.get_masses())  # atomic mass units
    density = total_mass * 1.66053907e-24 / (cell_volume * 1e-24)  # g/cm³
    
    energy_str = (f'Step: {step:6d} | Time: {time_ps:.4f} ps | Epot = {epot:.4f} eV | '
                  f'Ekin = {ekin:.4f} eV | T = {temp:.0f} K | Etot = {etot:.4f} eV | '
                  f'Density = {density:.4f} g/cm³\n')
    
    with open(ofile, 'a') as f:
        f.write(energy_str)
    
    print(energy_str, end='')
    
    # Check energy conservation (for NVE simulation)
    return etot

def run_optimization(atoms, args):
    """Run structure optimization"""
    
    # Set up trajectory file
    traj = Trajectory(args.output_prefix + ".traj", 'w', atoms)
    
    # Function to write XYZ file
    def write_xyz():
        write(args.output_prefix + ".xyz", atoms, append=True)
    
    # Perform optimization
    print("Starting structure optimization...")
    optimizer = BFGS(atoms, trajectory=traj)  # 使用轨迹文件
    optimizer.attach(write_xyz, interval=1)  # 每步写入XYZ文件
    optimizer.run(fmax=0.02, steps=1000)  # 设置更严格的力收敛标准和最大步数

    print("Optimization completed!")
    traj.close()  # 关闭轨迹文件


def run_nve_simulation(atoms, args):
    """Run NVE molecular dynamics simulation"""
    
    # Get initial energy, forces and stress
    print("Calculating initial energy and forces...")
    pot_energy = atoms.get_potential_energy()
    print(f"Initial potential energy: {pot_energy} eV")
    
    # Set initial velocity distribution
    print(f"Setting initial velocities (temperature: {args.temperature} K)...")
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature)
    
    # Set up NVE integrator
    timestep = args.timestep * units.fs
    dyn = VelocityVerlet(atoms, timestep=timestep)
    
    # Set up logging
    log_file = args.output_prefix + ".log"
    dyn.attach(MDLogger(dyn, atoms, log_file, header=True, stress=True,
                      peratom=False, mode="w"), interval=args.log_interval)
    
    # Set up trajectory saving
    def write_frame():
        dyn.atoms.write(args.output_prefix + ".xyz", append=True)
    dyn.attach(write_frame, interval=args.traj_interval)
    
    # Set up trajectory file
    traj = Trajectory(args.output_prefix + ".traj", 'w', atoms)
    dyn.attach(traj.write, interval=args.traj_interval)
    
    # Initialize energy output file with time column
    with open(args.output_prefix + "_energy.txt", 'w') as f:
        f.write("# Step | Time(ps) | Epot(eV) | Ekin(eV) | Temp(K) | Etot(eV) | Density(g/cm³)\n")
    
    # Run simulation
    print(f"\nStarting NVE simulation, total steps: {args.n_steps}, timestep: {args.timestep} fs")
    print("-" * 80)
    
    # Record initial total energy (time=0)
    initial_time_ps = 0.0
    initial_etot = print_energy(atoms, 0, initial_time_ps, args.output_prefix + "_energy.txt")
    
    total_start_time = time.time()
    for step in range(1, args.n_steps + 1):
        dyn.run(args.steps_per_print)
        
        # Calculate current step and time
        current_step = step * args.steps_per_print
        current_time_ps = current_step * args.timestep / 1000.0  # fs to ps
        
        # Print energy with time
        current_etot = print_energy(atoms, current_step, current_time_ps, 
                                  args.output_prefix + "_energy.txt")
        
        # Check energy drift
        energy_drift = abs((current_etot - initial_etot) / initial_etot)
        if energy_drift > 0.01:  # 1% energy drift
            print(f"WARNING: Large energy drift ({energy_drift*100:.2f}%), check simulation stability")
    
    traj.close()
    
    # Print performance statistics
    total_time = time.time() - total_start_time
    steps_per_second = args.n_steps * args.steps_per_print / total_time
    print("-" * 80)
    print(f"Simulation completed!")
    print(f"Total simulation steps: {args.n_steps * args.steps_per_print}")
    print(f"Total simulation time: {args.n_steps * args.steps_per_print * args.timestep / 1000:.2f} ps")
    print(f"Computation time: {total_time:.2f} seconds")
    print(f"Performance: {steps_per_second:.2f} steps/second")



def run_nvt_simulation(atoms, args):
    """Run NVT molecular dynamics simulation using Nose-Hoover Chain"""
    
    # Set initial velocity distribution
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature)
    
    # Set up NVT integrator using Nose-Hoover Chain
    timestep = args.timestep * units.fs
    tdamp = 100 * units.fs  # Temperature damping time constant
    dyn = NoseHooverChainNVT(atoms, timestep, temperature_K=args.temperature, tdamp=100*timestep)
    
    # Set up trajectory saving
    def write_frame():
        dyn.atoms.write(args.output_prefix + "_nvt.xyz", append=True)
    dyn.attach(write_frame, interval=args.traj_interval)
    
    # Set up trajectory file
    traj = Trajectory(args.output_prefix + "_nvt.traj", 'w', atoms)
    dyn.attach(traj.write, interval=args.traj_interval)
    
    # Initialize energy output file with time column
    with open(args.output_prefix + "_nvt_energy.txt", 'w') as f:
        f.write("# Step | Time(ps) | Epot(eV) | Ekin(eV) | Temp(K) | Etot(eV) | Density(g/cm³)\n")
    
    # Run simulation
    print(f"\nStarting NVT simulation using Nose-Hoover Chain, total steps: {args.n_steps}, timestep: {args.timestep} fs")
    print("-" * 80)
    
    # Record initial total energy (time=0)
    initial_time_ps = 0.0
    initial_etot = print_energy(atoms, 0, initial_time_ps, args.output_prefix + "_nvt_energy.txt")
    
    total_start_time = time.time()
    for step in range(1, args.n_steps + 1):
        dyn.run(args.steps_per_print)
        
        # Calculate current step and time
        current_step = step * args.steps_per_print
        current_time_ps = current_step * args.timestep / 1000.0  # fs to ps
        
        # Print energy with time
        current_etot = print_energy(atoms, current_step, current_time_ps, 
                                  args.output_prefix + "_nvt_energy.txt")
        
        # Check energy drift
        energy_drift = abs((current_etot - initial_etot) / initial_etot)
        if energy_drift > 0.01:  # 1% energy drift
            print(f"WARNING: Large energy drift ({energy_drift*100:.2f}%), check simulation stability")
    
    traj.close()
    
    # Print performance statistics
    total_time = time.time() - total_start_time
    steps_per_second = args.n_steps * args.steps_per_print / total_time
    print("-" * 80)
    print(f"Simulation completed!")
    print(f"Total simulation steps: {args.n_steps * args.steps_per_print}")
    print(f"Total simulation time: {args.n_steps * args.steps_per_print * args.timestep / 1000:.2f} ps")
    print(f"Computation time: {total_time:.2f} seconds")
    print(f"Performance: {steps_per_second:.2f} steps/second")

  
def run_npt_simulation(atoms, args):
    # print("Step 1: Minimizing energy...")
    # opt = BFGS(atoms)
    # opt.run(fmax=0.05) 

    # # 2. 现在加约束（只加两条 O-H 键）
    # print("Step 2: Applying constraints...")
    # bonds = []
    # for i in range(len(atoms) // 3):
    #     bonds.append([3 * i, 3 * i + 1])
    #     bonds.append([3 * i, 3 * i + 2])
    # atoms.set_constraint(FixBondLengths(bonds))

    # # 3. 再次进行微调优化（确保约束后的结构也是受力平衡的）
    # opt.run(fmax=0.05)
    # atoms.set_constraint(FixBondLengths(bonds))


    # Set initial velocity distribution
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature)
    
    # Set up NPT integrator using Berendsen thermostat and barostat
    timestep = args.timestep * units.fs
    pressure = 1.0 * units.bar  # Target pressure
    dyn = NPTBerendsen(atoms, timestep=timestep, temperature_K=args.temperature,
                   taut=100 * units.fs, pressure_au=pressure,
                   taup=2000 * units.fs, compressibility_au=4.57e-5 / units.bar)
    
    # Set up trajectory saving
    def write_frame():
        dyn.atoms.write(args.output_prefix + "_npt.xyz", append=True)
    dyn.attach(write_frame, interval=args.traj_interval)
    
    # Set up trajectory file
    traj = Trajectory(args.output_prefix + "_npt.traj", 'w', atoms)
    dyn.attach(traj.write, interval=args.traj_interval)
    
    # Initialize energy output file with time column
    with open(args.output_prefix + "_npt_energy.txt", 'w') as f:
        f.write("# Step | Time(ps) | Epot(eV) | Ekin(eV) | Temp(K) | Etot(eV) | Density(g/cm³)\n")
    
    # Run simulation
    print(f"\nStarting NPT simulation using Berendsen thermostat and barostat, total steps: {args.n_steps}, timestep: {args.timestep} fs")
    print("-" * 80)
    
    # Record initial total energy (time=0)
    initial_time_ps = 0.0
    initial_etot = print_energy(atoms, 0, initial_time_ps, args.output_prefix + "_npt_energy.txt")
    
    total_start_time = time.time()
    for step in range(1, args.n_steps + 1):
        dyn.run(args.steps_per_print)
        
        # Calculate current step and time
        current_step = step * args.steps_per_print
        current_time_ps = current_step * args.timestep / 1000.0  # fs to ps
        
        # Print energy with time
        current_etot = print_energy(atoms, current_step, current_time_ps, 
                                  args.output_prefix + "_npt_energy.txt")
        
        # # Check energy drift
        # energy_drift = abs((current_etot - initial_etot) / initial_etot)
        # if energy_drift > 0.01:  # 1% energy drift
        #     print(f"WARNING: Large energy drift ({energy_drift*100:.2f}%), check simulation stability")
    
    traj.close()
    
    # Print performance statistics
    total_time = time.time() - total_start_time
    steps_per_second = args.n_steps * args.steps_per_print / total_time
    print("-" * 80)
    print(f"Simulation completed!")
    print(f"Total simulation steps: {args.n_steps * args.steps_per_print}")
    print(f"Total simulation time: {args.n_steps * args.steps_per_print * args.timestep / 1000:.2f} ps")
    print(f"Computation time: {total_time:.2f} seconds")
    print(f"Performance: {steps_per_second:.2f} steps/second")

if __name__ == '__main__':
    # Command line argument parsing
    parser = argparse.ArgumentParser(description='Molecular Dynamics Simulation with Time Tracking')
    parser.add_argument('--pdb', type=str, default='init.pdb', help='Initial PDB file')
    parser.add_argument('--ff_xml', type=str, default='ff.xml', help='Force field XML file')
    parser.add_argument('--cutoff', type=float, default=0.8, help='Cutoff radius (nm)')
    parser.add_argument('--temperature', type=float, default=298.15, help='Initial temperature (K)')
    parser.add_argument('--timestep', type=float, default=1.0, help='Time step (fs)')
    parser.add_argument('--n_steps', type=int, default=5000, help='Number of simulation steps')
    parser.add_argument('--steps_per_print', type=int, default=100, help='Steps between each print')
    parser.add_argument('--log_interval', type=int, default=100, help='Log recording interval')
    parser.add_argument('--traj_interval', type=int, default=500, help='Trajectory saving interval')
    parser.add_argument('--output_prefix', type=str, default='test', help='Output file prefix')
    
    args = parser.parse_args()
    
    # Run simulation
    print("=" * 80)
    print(f"Starting NPT Molecular Dynamics Simulation")
    print(f"Initial structure: {args.pdb}")
    print(f"Force field file: {args.ff_xml}")
    print("=" * 80)
   
    # Load initial structure, sanity check
    atoms = read(args.pdb)
    cell = jnp.array(atoms.get_cell())
    if jnp.sum(cell*cell) == 0.0:
        sys.exit('Error reading box dimension from pdb')
    if len(atoms.get_positions()) == 0:
        sys.exit('Error reading atom positions')

    # Set up calculator
    kwargs = {}
    # rc should be in cutoff, and is used for two purposes:
    # 1. feed in createPotential as nonbondedCutoff
    # 2. build initial pair list for the pilot run of potential energy
    # atoms.calc = DMFFCalculator(pdb=args.pdb, ff_xml=args.ff_xml, rc=args.cutoff, **kwargs)
    atoms.calc = DMFFCalculator(pdb=args.pdb, ff_xml=args.ff_xml, rc=args.cutoff)
    # run_nve_simulation(atoms, args)
    # run_optimization(atoms, args)
    # run_nvt_simulation(atoms, args)
    run_npt_simulation(atoms, args)
