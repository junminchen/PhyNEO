#!/usr/bin/env python
from openmm.app import Modeller
import openmm as mm 
import openmm.app as app
import openmm.unit as unit 
import numpy as np
import sys
from dmff import Hamiltonian
from dmff.common import nblist
from jax import jit
import jax.numpy as jnp

def forcegroupify(system):
    forcegroups = {}
    for i in range(system.getNumForces()):
        force = system.getForce(i)
        force.setForceGroup(i)
        forcegroups[force] = i
    return forcegroups

def getEnergyDecomposition(context, forcegroups):
    energies = {}
    for f, i in forcegroups.items():
        energies[f] = context.getState(getEnergy=True, groups=2**i).getPotentialEnergy()
    return energies

if __name__ == "__main__":

    print("MM Reference Energy:")
    # app.Topology.loadBondDefinitions("lig-top.xml")
    pdb = app.PDBFile("dimer_bank/dimer_003_EC_EC.pdb")
    # pdb = app.PDBFile("pdb_bank/EC.pdb")

    # ff = app.ForceField("xml/opls_solvent.xml")
    ff = app.ForceField("ff_openmm_EC.xml")
    system = ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, constraints=None, removeCMMotion=False, defaultTholeWidth=5.0)
    
    # mpid_force.set14ScaleFactor(0.5) 

    # customNonbondedForce = [f for f in [system.getForce(i) for i in range(system.getNumForces())] if type(f) == mm.CustomNonbondedForce][0]
    # customNonbondedForce.setNonbondedMethod(min(nbondedForce.getNonbondedMethod(),NonbondedForce.CutoffPeriodic))
    # customNonbondedForce.setUseLongRangeCorrection(False)
    # customNonbondedForce.setUseLongRangeCorrection(True)

    for force in system.getForces():
        if isinstance(force, mm.CustomNonbondedForce):
            # 通过检查参数名称或能量公式来区分
            # 比如 Dispersion 力包含 "C6" 参数
            is_dispersion = True
            for i in range(force.getNumPerParticleParameters()):
                if force.getPerParticleParameterName(i) == "Aexch":
                    is_dispersion = False
                    break
            
            if is_dispersion:
                print("Found Dispersion Force: Enabling Long Range Correction")
                force.setUseLongRangeCorrection(True)
            else:
                print("Found Repulsion/Elec Force: Disabling Long Range Correction")
                force.setUseLongRangeCorrection(False)


    # print("Dih info:")
    # for force in system.getForces():
    #     if isinstance(force, mm.PeriodicTorsionForce):
    #         print("No. of dihs:", force.getNumTorsions())

    forcegroups = forcegroupify(system)
    integrator = mm.VerletIntegrator(0.1)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("Reference"))
    context.setPositions(pdb.positions)
    state = context.getState(getEnergy=True)
    energy = state.getPotentialEnergy()
    energies = getEnergyDecomposition(context, forcegroups)
    print(energy)
    for key in energies.keys():
        print(key.getName(), energies[key])
