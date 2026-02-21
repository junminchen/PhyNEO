from openmm.app import *
from openmm import *
from openmm.unit import *
import openmm.app.element as elem
from sys import stdout, argv
import mpidplugin
import numpy as np
from mdtraj.reporters import NetCDFReporter

pdb = PDBFile('solvated_ethane_from_openmm_setup.pdb')
forcefield = ForceField('ethane_water.xml')
system = forcefield.createSystem(pdb.topology, nonbondedMethod=PME, polarization='direct',
                                 nonbondedCutoff=8*angstrom, constraints=HBonds, defaultTholeWidth=8)
temperature = 298*kelvin
system.addForce(MonteCarloBarostat(1*atmosphere, temperature, 25))
integrator = LangevinIntegrator(temperature, 1/picosecond, 2*femtoseconds)

try:
    myplatform = Platform.getPlatformByName('CUDA')
    # Figure out which GPU to run on, i.e. did the user tell us?
    deviceid = argv[1] if len(argv) > 1 else '0'
    #myproperties = {'DeviceIndex': deviceid, 'Precision': 'double'}
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

print("Running on ", context.getPlatform().getName(), " Device ID:", deviceid)

# Initialize
context.setPositions(pdb.positions)
#simulation.loadState('equilibrated.xml')

# Get a list of the forces.  We use a dictionary here because there is only one force present
# that identifies itself as just a "Force".  If multiple plugins are used, this needs to change.
forces = {system.getForce(i).__class__.__name__: system.getForce(i) for i in range(system.getNumForces())}
if mpidplugin.MPIDForce.isinstance(forces['Force']):
    mpidforce = mpidplugin.MPIDForce.cast(forces['Force'])
else:
    raise "Unknown custom force detected!"

print(f'initial energy = ',context.getState(getEnergy=True).getPotentialEnergy())

# Specify lambda windows for alchemical changes
lambdas = np.arange(0.0, 1.05, 0.1)
selection = range(8)  # We know the first 8 atoms are ethane

# Turn on the electrostatic terms in increments, after storing the initial values
# Order of parameters is
# charge, dipole, qpole, opole, axis, atomZ, atomX, atomY, thole, alphas
elec_params = [ mpidforce.getMultipoleParameters(i) for i in selection ]

for lam in lambdas:
    for i in selection:
        params = elec_params[i].copy()
        # charge
        params[0] *= lam
        # dipoles
        params[1] = [d * lam for d in params[1]]
        # quadrupoles
        params[2] = [q * lam for q in params[2]]
        # octopoles
        params[3] = [o * lam for o in params[3]]
        # polarizabilities
        params[9] = [a * lam for a in params[9]]
        mpidforce.setMultipoleParameters(i, *params)
    mpidforce.updateParametersInContext(context)
    print(f'elec lambda = {lam:.2f}, energy = ',context.getState(getEnergy=True).getPotentialEnergy())

# Turn on the VDW terms in increments, after storing the initial values
# Order of paramters is
# charge, sigma, epsilon
vdw_params = [ forces['NonbondedForce'].getParticleParameters(i) for i in selection ]

for lam in lambdas:
    for i in selection:
        params = [lam*p for p in vdw_params[i]]
        forces['NonbondedForce'].setParticleParameters(i, *params)
    forces['NonbondedForce'].updateParametersInContext(context)
    print(f'vdw lambda = {lam:.2f}, energy = ',context.getState(getEnergy=True).getPotentialEnergy())

# Make sure the final energy is the same as the original (unperturbed) value, with roundoff error
print(f'final energy = ',context.getState(getEnergy=True).getPotentialEnergy())
