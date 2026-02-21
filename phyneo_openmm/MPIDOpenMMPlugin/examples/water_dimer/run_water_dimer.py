from openmm.app import *
from openmm import *
from openmm.unit import *
import openmm.app.element as elem
from sys import stdout, argv
import mpidplugin
import numpy as np

rvals = np.arange(1.0, 6.0, 0.1)
rvals = np.arange(6.0, 1.0, -0.1)

pdb = PDBFile('waterdimer.pdb')
pdb = PDBFile('waterdimer_aligned.pdb')

forcefield = ForceField('charmm_polar_2019.xml')
modeller = Modeller(pdb.topology, pdb.positions)
modeller.addExtraParticles(forcefield)
system = forcefield.createSystem(modeller.topology, nonbondedMethod=LJPME, nonbondedCutoff=8*angstrom, constraints=HBonds)
integrator = DrudeSCFIntegrator(1e-10*femtoseconds)
integrator.setMinimizationErrorTolerance(1e-12)

try:
    myplatform = Platform.getPlatformByName('CUDA')
    # Figure out which GPU to run on, i.e. did the user tell us?
    deviceid = argv[1] if len(argv) > 1 else '0'
    myproperties = {'DeviceIndex': deviceid, 'Precision': 'mixed'}
    myproperties = {'DeviceIndex': deviceid, 'Precision': 'double'}
except:
    print("CUDA NOT FOUND!!!!!!!!!!")
    myplatform = None
    deviceid = "N/A"

if myplatform:
    simulation = Simulation(modeller.topology, system, integrator, myplatform, myproperties)
else:
    simulation = Simulation(modeller.topology, system, integrator)

context = simulation.context
if pdb.topology.getPeriodicBoxVectors():
    context.setPeriodicBoxVectors(*pdb.topology.getPeriodicBoxVectors())

print("Running on ", context.getPlatform().getName(), " Device ID:", deviceid)

context.setPositions(modeller.positions)
drude = []
for r in rvals:
    new_coords = []
    for n,c in enumerate(modeller.positions):
        c = c.value_in_unit(angstrom)
        if n > 4:
            c = Vec3(c[0], c[1], c[2]+r)
        new_coords.append(c)
    new_coords = new_coords*angstrom
    context.setPositions(new_coords)
    integrator.step(1)
    state = context.getState(getEnergy=True, getPositions=True)
    drude.append(state.getPotentialEnergy().value_in_unit(kilocalories_per_mole))
#    
#MPID
#    
forcefield = ForceField('mpidwater.xml')
system = forcefield.createSystem(pdb.topology, nonbondedMethod=LJPME, nonbondedCutoff=8*angstrom, constraints=HBonds, defaultTholeWidth=8)
integrator = VerletIntegrator(1e-10*femtoseconds)
if myplatform:
    simulation = Simulation(pdb.topology, system, integrator, myplatform, myproperties)
else:
    simulation = Simulation(pdb.topology, system, integrator)
context = simulation.context
context.setPositions(pdb.positions)
mpid = []
for r in rvals:
    new_coords = []
    for n,c in enumerate(pdb.positions):
        c = c.value_in_unit(angstrom)
        if n > 2:
            c = Vec3(c[0], c[1], c[2]+r)
        new_coords.append(c)
    new_coords = new_coords*angstrom
    context.setPositions(new_coords)
    integrator.step(1)
    state = context.getState(getEnergy=True, getPositions=True)
    mpid.append(state.getPotentialEnergy().value_in_unit(kilocalories_per_mole))

for n,r in enumerate(rvals):
    print(f'{r:.2f}, {drude[n]:8.3f}, {mpid[n]:8.3f}')
