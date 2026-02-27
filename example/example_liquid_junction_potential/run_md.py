import os
import sys
from openmm.app import *
from openmm import *
from openmm.unit import *

# ==========================================================
# 参数设置
# ==========================================================
pdb_file = 'sandwich_box.pdb'
ff_files = ['opls_salt.xml', 'opls_solvent.xml']
temp = 298.15 * kelvin
pressure = 1.0 * bar

# 模拟时长 (生产级别)
steps_nvt_eq = 100000  # 100 ps
steps_npt_eq = 200000  # 200 ps
steps_production = 1000000 # 1000 ps = 1 ns

report_steps = 5000

# ==========================================================
# 1. 能量最小化 (EM) - 无约束
# ==========================================================
print("Loading PDB and ForceField...")
pdb = PDBFile(pdb_file)
forcefield = ForceField(*ff_files)
# 初始盒子设为足够大
pdb.topology.setUnitCellDimensions(Vec3(5.0, 5.0, 32.0)*nanometers)

print("Step 1: Energy Minimization (No Constraints)...")
system_em = forcefield.createSystem(pdb.topology, nonbondedMethod=PME, 
                                    nonbondedCutoff=1.0*nanometers, constraints=None)
integrator_em = LangevinMiddleIntegrator(temp, 1/picosecond, 0.5*femtoseconds)
sim_em = Simulation(pdb.topology, system_em, integrator_em)
sim_em.context.setPositions(pdb.positions)
sim_em.minimizeEnergy()

state = sim_em.context.getState(getEnergy=True, getPositions=True)
print(f"Potential Energy after EM: {state.getPotentialEnergy()}")
positions = state.getPositions()

# ==========================================================
# 2. NVT 平衡 - 1.0 fs
# ==========================================================
print("\nStep 2: NVT Equilibration (1.0 fs)...")
system_nvt = forcefield.createSystem(pdb.topology, nonbondedMethod=PME, 
                                     nonbondedCutoff=1.0*nanometers, constraints=HBonds)
integrator_nvt = LangevinMiddleIntegrator(temp, 1/picosecond, 1.0*femtoseconds)
sim_nvt = Simulation(pdb.topology, system_nvt, integrator_nvt)
sim_nvt.context.setPositions(positions)
sim_nvt.context.setVelocitiesToTemperature(temp)

sim_nvt.reporters.append(StateDataReporter(sys.stdout, report_steps, step=True,
    potentialEnergy=True, temperature=True, progress=True, speed=True, totalSteps=steps_nvt_eq))
sim_nvt.step(steps_nvt_eq)

state = sim_nvt.context.getState(getPositions=True, getVelocities=True)
positions = state.getPositions()
velocities = state.getVelocities()

# ==========================================================
# 3. NPT 平衡 - 1.0 fs
# ==========================================================
print("\nStep 3: NPT Equilibration (1.0 fs)...")
system_npt = forcefield.createSystem(pdb.topology, nonbondedMethod=PME, 
                                     nonbondedCutoff=1.0*nanometers, constraints=HBonds)
system_npt.addForce(MonteCarloBarostat(pressure, temp))
integrator_npt = LangevinMiddleIntegrator(temp, 1/picosecond, 1.0*femtoseconds)
sim_npt = Simulation(pdb.topology, system_npt, integrator_npt)
sim_npt.context.setPositions(positions)
sim_npt.context.setVelocities(velocities)

sim_npt.reporters.append(StateDataReporter(sys.stdout, report_steps, step=True,
    potentialEnergy=True, temperature=True, density=True, progress=True, speed=True, totalSteps=steps_npt_eq))
sim_npt.step(steps_npt_eq)

# 重要：在 NPT 结束时获取位置、速度以及最重要的【盒子大小】
state = sim_npt.context.getState(getPositions=True, getVelocities=True, getEnergy=True)
positions = state.getPositions()
velocities = state.getVelocities()
box_vectors = state.getPeriodicBoxVectors()
print(f"Final NPT Box Vectors: {box_vectors}")

# ==========================================================
# 4. NVT 生产 - 1.0 fs
# ==========================================================
print("\nStep 4: NVT Production (1.0 fs)...")
system_prod = forcefield.createSystem(pdb.topology, nonbondedMethod=PME, 
                                      nonbondedCutoff=1.0*nanometers, constraints=HBonds)
integrator_prod = LangevinMiddleIntegrator(temp, 1/picosecond, 1.0*femtoseconds)
sim_prod = Simulation(pdb.topology, system_prod, integrator_prod)

# 必须先设置盒子向量，再设置位置
sim_prod.context.setPeriodicBoxVectors(*box_vectors)
sim_prod.context.setPositions(positions)
sim_prod.context.setVelocities(velocities)

sim_prod.reporters.append(DCDReporter('production.dcd', report_steps))
sim_prod.reporters.append(StateDataReporter(sys.stdout, report_steps, step=True,
    potentialEnergy=True, temperature=True, progress=True, speed=True, totalSteps=steps_production))

sim_prod.step(steps_production)

print("\nSimulation Finished!")
