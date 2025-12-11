#!/usr/bin/env python

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

    # print("MM Reference Energy:")
    # app.Topology.loadBondDefinitions("lig-top.xml")

    pdb = app.PDBFile("dimer_bank/dimer_003_EC_EC.pdb")
    # pdb = app.PDBFile("pdb_bank/EC.pdb")

    # ff = app.ForceField("xml/opls_solvent.xml")
    # ff = app.ForceField("output.customnb.xml")
    # system = ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, constraints=None, removeCMMotion=False)
    
    # print("Dih info:")
    # for force in system.getForces():
    #     if isinstance(force, mm.PeriodicTorsionForce):
    #         print("No. of dihs:", force.getNumTorsions())

    # forcegroups = forcegroupify(system)
    # integrator = mm.VerletIntegrator(0.1)
    # context = mm.Context(system, integrator, mm.Platform.getPlatformByName("Reference"))
    # context.setPositions(pdb.positions)
    # state = context.getState(getEnergy=True)
    # energy = state.getPotentialEnergy()
    # energies = getEnergyDecomposition(context, forcegroups)
    # print(energy)
    # for key in energies.keys():
    #     print(key.getName(), energies[key])

    # print()
    print("Jax Energy:")
    
    
    # h = Hamiltonian('example_Li_PF6_DMC-EC/phyneo_ecl.xml')
    h = Hamiltonian('ff_dmff_EC.xml')
    pot = h.createPotential(pdb.topology, nonbondedMethod=app.NoCutoff)
    params = h.getParameters()

    positions = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    positions = jnp.array(positions)
    box = np.array([
        [10.0,  0.0,  0.0],
        [ 0.0, 10.0,  0.0],
        [ 0.0,  0.0, 10.0]
    ])
    
    # neighbor list
    rc = 1.5
    nbl = nblist.NeighborList(box, rc, pot.meta['cov_map'])
    nbl.allocate(positions)
    pairs = nbl.pairs
    potentials_names = ['es', 'disp', 'ex', 'sr_es', 'sr_pol', 'sr_disp', 'dhf', 'dmp_es', 'dmp_disp']
    potentials_mapping = {
        'es': 'ADMPPmeForce',
        'disp': 'ADMPDispPmeForce',
        'ex': 'SlaterExForce',
        'sr_es': 'SlaterSrEsForce',
        'sr_pol': 'SlaterSrPolForce',
        'sr_disp': 'SlaterSrDispForce',
        'dhf': 'SlaterDhfForce',
        'dmp_es': 'QqTtDampingForce',
        'dmp_disp': 'SlaterDampingForce',
    }
    
    for potentials_name in potentials_names:
        setattr(pot, f'pots_{potentials_name}', pot.dmff_potentials[potentials_mapping[potentials_name]])
        print(potentials_name, getattr(pot, f'pots_{potentials_name}')(positions, box, pairs, params))


    print("Total:", pot.getPotentialFunc()(positions, box, pairs, params))
    # print("Electrostatic:", getattr(pot, 'pots_es')(positions, box, pairs, params))
    # print("Total - Electrostatic:", pot.getPotentialFunc()(positions, box, pairs, params) - getattr(pot, 'pots_es')(positions, box, pairs, params))