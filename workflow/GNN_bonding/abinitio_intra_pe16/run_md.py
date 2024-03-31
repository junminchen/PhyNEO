#!/usr/bin/env python
import sys
from sys import stdout
import numpy as np
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *

# from http://zarbi.chem.yale.edu/ligpargen/openMM_tutorial.html
def OPLS_LJ(system):
    forces = {system.getForce(index).__class__.__name__: system.getForce(
        index) for index in range(system.getNumForces())}
    nonbonded_force = forces['NonbondedForce']
    lorentz = CustomNonbondedForce(
        '4*epsilon*((sigma/r)^12-(sigma/r)^6); sigma=sqrt(sigma1*sigma2); epsilon=sqrt(epsilon1*epsilon2)')
    # lorentz.setNonbondedMethod(nonbonded_force.getNonbondedMethod())
    lorentz.setNonbondedMethod(NonbondedForce.CutoffPeriodic)
    lorentz.addPerParticleParameter('sigma')
    lorentz.addPerParticleParameter('epsilon')
    lorentz.setCutoffDistance(nonbonded_force.getCutoffDistance())
    system.addForce(lorentz)
    LJset = {}
    for index in range(nonbonded_force.getNumParticles()):
        charge, sigma, epsilon = nonbonded_force.getParticleParameters(index)
        LJset[index] = (sigma, epsilon)
        lorentz.addParticle([sigma, epsilon])
        nonbonded_force.setParticleParameters(
            index, charge, sigma, epsilon * 0)
    for i in range(nonbonded_force.getNumExceptions()):
        (p1, p2, q, sig, eps) = nonbonded_force.getExceptionParameters(i)
        # ALL THE 12,13 and 14 interactions are EXCLUDED FROM CUSTOM NONBONDED
        # FORCE
        lorentz.addExclusion(p1, p2)
        if eps._value != 0.0:
            #print p1,p2,sig,eps
            sig14 = sqrt(LJset[p1][0] * LJset[p2][0])
            eps14 = sqrt(LJset[p1][1] * LJset[p2][1])
            nonbonded_force.setExceptionParameters(i, p1, p2, q, sig14, eps)
    return system

# create system
mol = PDBFile('pe16.pdb')
forcefield = ForceField('pe.xml')
system = forcefield.createSystem(mol.topology, nonbondedMethod=PME, nonbondedCutoff=1.0*nanometer, constraints=None, rigidWater=True, removeCMMotion=True)
system = OPLS_LJ(system)

# create simulation
integrator = LangevinIntegrator(300*kelvin, 1.0/picoseconds, 1.0*femtosecond)
platform = Platform.getPlatformByName('CUDA')
if platform.getName() == 'CUDA':
    properties = {'CudaPrecision': 'mixed'}
else:
    properties = {}
simulation = Simulation(mol.topology, system, integrator, platform, properties)
simulation.context.setPositions(mol.positions)

# outputs
simulation.reporters.append(DCDReporter('traj.dcd', 5000))
simulation.reporters.append(StateDataReporter(stdout, 5000, step=True, potentialEnergy=True, temperature=True))

# simulate
simulation.step(50000000)

positions = simulation.context.getState(getPositions=True).getPositions()
with open('output.pdb', 'w') as ifile:
    PDBFile.writeFile(simulation.topology, positions, ifile)
