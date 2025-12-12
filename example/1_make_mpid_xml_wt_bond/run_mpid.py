#!/usr/bin/env python
import sys
from openmm import *
from openmm.app import *
from openmm.unit import *
# import mpidplugin


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

if __name__ == '__main__':
    pdb_file = 'peo3.pdb'
    pdb_file = 'EC.pdb'
    # pdb_file = sys.argv[1]
    ff_file = sys.argv[1]

    print("MM Reference Energy:")

    ff = ForceField(ff_file)
    # ff = ForceField('../EC_Li_mpid.xml')
    # ff = ForceField('phyneo_ecl_z_b.xml')

    pdb = PDBFile(pdb_file)
    rc = 8
    # system = ff.createSystem(pdb.topology, nonbondedCutoff=rc*angstrom, nonbondedMethod=PME, defaultTholeWidth=8)
    system = ff.createSystem(pdb.topology, nonbondedCutoff=rc*angstrom, nonbondedMethod=NoCutoff)

    # system = ff.createSystem(pdb.topology, nonbondedCutoff=rc*angstrom, nonbondedMethod=PME, constraints=HBonds)
    # system = ff.createSystem(pdb.topology, nonbondedCutoff=rc*unit.angstrom, nonbondedMethod=app.CutoffPeriodic)
    # system = ff.createSystem(pdb.topology, nonbondedMethod=NoCutoff, constraints=None, removeCMMotion=False)

    # try:
    #     myplatform = Platform.getPlatformByName('CUDA')
    #     # Figure out which GPU to run on, i.e. did the user tell us?
    #     deviceid = argv[1] if len(argv) > 1 else '0'
    #     myproperties = {'DeviceIndex': deviceid, 'Precision': 'mixed'}
    #     myproperties = {'DeviceIndex': deviceid, 'Precision': 'double'}
    # except:
    #     print("CUDA NOT FOUND!!!!!!!!!!")
    #     myplatform = None
    #     deviceid = "N/A"

    # if myplatform:
    #     simulation = Simulation(modeller.topology, system, integrator, myplatform, myproperties)
    # else:
    #     simulation = Simulation(modeller.topology, system, integrator)

    # context = simulation.context
    # if pdb.topology.getPeriodicBoxVectors():
    #     context.setPeriodicBoxVectors(*pdb.topology.getPeriodicBoxVectors())

    # print("Running on ", context.getPlatform().getName(), " Device ID:", deviceid)

    forcegroups = forcegroupify(system)
    integrator = VerletIntegrator(0.1)
    # platform = Platform.getPlatformByName('CUDA')
    platform = Platform.getPlatformByName('Reference')

    context = Context(system, integrator, platform)
    
    context.setPositions(pdb.positions)
    state = context.getState(getEnergy=True,getForces=True)
    energy = state.getPotentialEnergy()
    forces = state.getForces()

    # # Set distance cutoffs, constraints, and other force-specific options for each
    # # force we might encounter
    # forces = {system.getForce(index).__class__.__name__: system.getForce(
    #     index) for index in range(system.getNumForces())}
    # # nonbonded_force = forces['NonbondedForce'] 
    # # two_body_cutoff = nonbonded_force.getCutoffDistance()
    # two_body_cutoff = 0.8*nanometers
    # for force in system.getForces():
    #     if isinstance(force, CustomHbondForce):
    #         force.setNonbondedMethod(CustomHbondForce.CutoffPeriodic)
    #         force.setCutoffDistance(two_body_cutoff)
    #     elif isinstance(force, CustomNonbondedForce):
    #         force.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
    #         force.setCutoffDistance(two_body_cutoff)
    #         force.setUseLongRangeCorrection(True)
    #     elif isinstance(force, AmoebaMultipoleForce):
    #         force.setNonbondedMethod(AmoebaMultipoleForce.PME)
    #     elif isinstance(force, NonbondedForce):
    #         force.setNonbondedMethod(NonbondedForce.LJPME)
    #         force.setCutoffDistance(two_body_cutoff)
    #     elif isinstance(force, CustomManyParticleForce):
    #         force.setNonbondedMethod(CustomManyParticleForce.CutoffPeriodic)
    #         force.setCutoffDistance(three_body_cutoff)
    #     else:
    #         pass

    # print(force)
    energies = getEnergyDecomposition(context, forcegroups)
    print('TotalEnergy', energy)
    for key in energies.keys():
        print(key.getName(), energies[key])
        
    # import numpy as np
    # from run_openmm import *    
    # calc = nb_ff_calculator(pdb_file, 'forcefield_amoeba.xml', 'params_sr.dat')
    # pdb = app.PDBFile(pdb_file)
    # pos = np.array(pdb.positions._value) * 10
    # E_nb, E_nb_sr, E_nb_lr = calc.get_energy(pos)
    # # print("E_nb:",E_nb) 
    # # print("E_nb_lr:",E_nb_lr) 
    # print("E_nb_sr:",E_nb_sr) 
