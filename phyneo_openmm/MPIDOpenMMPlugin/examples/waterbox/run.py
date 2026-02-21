from openmm.app import *
from openmm import *
from openmm.unit import *
import openmm.app.element as elem
import openmm.app.forcefield as forcefield
from sys import stdout, argv
import mpidplugin
import numpy as np

pdb = PDBFile('waterbox_31ang.pdb')

#forcefield = ForceField('../parameters/tip3p.xml')
#forcefield = ForceField('../parameters/swm4.xml')
forcefield = ForceField('../parameters/swm6.xml')
system = forcefield.createSystem(pdb.topology, nonbondedMethod=LJPME, nonbondedCutoff=8*angstrom, constraints=HBonds,
                                 defaultTholeWidth=8)
integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 2*femtoseconds)
system.addForce(MonteCarloBarostat(1*atmosphere, 300*kelvin, 25))

# Make sure all the forces we expect are present
for force in range(system.getNumForces()):
    print(system.getForce(force))

try:
    myplatform = Platform.getPlatformByName('CUDA')
    # Figure out which GPU to run on, i.e. did the user tell us?
    deviceid = argv[1] if len(argv) > 1 else '0'
    myproperties = {'DeviceIndex': deviceid, 'Precision': 'mixed'}
except:
    print("CUDA NOT FOUND!!!!!!!!!!")
    myplatform = None
    deviceid = "N/A"

if myplatform:
    simulation = Simulation(pdb.topology, system, integrator, myplatform, myproperties)
else:
    simulation = Simulation(pdb.topology, system, integrator)

context = simulation.context
if pdb.topology.getPeriodicBoxVectors():
    context.setPeriodicBoxVectors(*pdb.topology.getPeriodicBoxVectors())

if os.path.isfile('restart.xml'):
    simulation.loadState('restart.xml')

print("Running on ", context.getPlatform().getName(), " Device ID:", deviceid)

# Initialize
context.setPositions(pdb.positions)

nsteps = 50000

# Dump trajectory info every 10ps
#simulation.reporters.append(DCDReporter('output.dcd', 5000))
# Dump simulation info every 1ps
simulation.reporters.append(StateDataReporter(stdout, 500, progress=True, totalSteps=nsteps,
                            step=True, potentialEnergy=True, totalEnergy=True, temperature=True, density=True, speed=True))
simulation.step(nsteps)

simulation.saveState('restart.xml')
