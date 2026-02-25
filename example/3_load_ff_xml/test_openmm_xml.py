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

try:
    import mpidplugin  # noqa: F401
except Exception:
    pass

# DMFF-aligned scaling profiles (12,13,14,15,16).
M_SCALES = [0.0, 0.0, 0.0, 0.0, 0.0]
P_SCALES = [0.0, 0.0, 0.0, 0.0, 0.0]
D_SCALES = [1.0, 1.0, 1.0, 1.0, 1.0]

def _get_mpid_force(system):
    for i in range(system.getNumForces()):
        f = system.getForce(i)
        if "mpidplugin" in globals() and mpidplugin.MPIDForce.isinstance(f):
            return mpidplugin.MPIDForce.cast(f)
    return None


def _bonded_shells(topology, max_depth=5):
    atoms = list(topology.atoms())
    n = len(atoms)
    adj = [set() for _ in range(n)]
    for bond in topology.bonds():
        i = bond[0].index
        j = bond[1].index
        adj[i].add(j)
        adj[j].add(i)

    shells_all = []
    for i in range(n):
        visited = {i}
        frontier = {i}
        shells = {d: set() for d in range(1, max_depth + 1)}
        for d in range(1, max_depth + 1):
            nxt = set()
            for u in frontier:
                nxt |= (adj[u] - visited)
            shells[d] = nxt
            visited |= nxt
            frontier = nxt
        shells_all.append(shells)
    return shells_all


def apply_dmff_like_intra_exclusions(system, topology, m_scales, p_scales, d_scales):
    """
    Align MPID intramolecular scaling with DMFF by in-script m/p/d scale lists.
    """
    mpid_force = _get_mpid_force(system)
    if mpid_force is None:
        return False

    shells_all = _bonded_shells(topology, max_depth=5)
    residue_atoms = [[atom.index for atom in residue.atoms()] for residue in topology.residues()]
    atom_to_residue = {a: r for r, atoms in enumerate(residue_atoms) for a in atoms}
    covalent15 = getattr(mpid_force, "Covalent15", 3)

    for atom_index in range(mpid_force.getNumMultipoles()):
        shells = shells_all[atom_index]
        residue_index = atom_to_residue[atom_index]
        full_intra = tuple(sorted([a for a in residue_atoms[residue_index] if a != atom_index]))

        if float(m_scales[4]) == 0.0:
            c12, c13, c14, c15 = full_intra, tuple(), tuple(), tuple()
        else:
            c12 = tuple(sorted(shells[1])) if float(m_scales[0]) == 0.0 else tuple()
            c13 = tuple(sorted(shells[2])) if float(m_scales[1]) == 0.0 else tuple()
            c14 = tuple(sorted(shells[3])) if float(m_scales[2]) == 0.0 else tuple()
            c15 = tuple(sorted(shells[4])) if float(m_scales[3]) == 0.0 else tuple()
        mpid_force.setCovalentMap(atom_index, mpid_force.Covalent12, c12)
        mpid_force.setCovalentMap(atom_index, mpid_force.Covalent13, c13)
        mpid_force.setCovalentMap(atom_index, mpid_force.Covalent14, c14)
        mpid_force.setCovalentMap(atom_index, covalent15, c15)

        p11 = (atom_index,)
        if float(p_scales[4]) == 0.0:
            p12, p13, p14 = full_intra, tuple(), tuple()
        else:
            p12 = tuple(sorted(shells[1])) if float(p_scales[0]) == 0.0 else tuple()
            p13 = tuple(sorted(shells[2])) if float(p_scales[1]) == 0.0 else tuple()
            p14 = tuple(sorted(shells[3])) if float(p_scales[2]) == 0.0 else tuple()
        mpid_force.setCovalentMap(atom_index, mpid_force.PolarizationCovalent11, p11)
        mpid_force.setCovalentMap(atom_index, mpid_force.PolarizationCovalent12, p12)
        mpid_force.setCovalentMap(atom_index, mpid_force.PolarizationCovalent13, p13)
        mpid_force.setCovalentMap(atom_index, mpid_force.PolarizationCovalent14, p14)

    if float(m_scales[4]) == 0.0:
        print("Info: mScale16=0 detected; applied full same-residue exclusion fallback for mScale maps.")
    if float(p_scales[4]) == 0.0:
        print("Info: pScale16=0 detected; applied full same-residue exclusion fallback for pScale maps.")
    elif float(p_scales[3]) == 0.0:
        print("Warning: pScale15=0 requested, but MPIDForce exposes polarization maps only up to 1-4.")
    if any(float(x) != 1.0 for x in d_scales):
        print("Warning: dScales are recorded but not directly configurable through MPIDForce covalent maps.")
    return True

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
    system = ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, constraints=None, removeCMMotion=False)#, defaultTholeWidth=5.0)

    patched = apply_dmff_like_intra_exclusions(system, pdb.topology, M_SCALES, P_SCALES, D_SCALES)
    if patched:
        print(f"Applied MPID intramolecular exclusions with m={M_SCALES}, p={P_SCALES}, d={D_SCALES}.")
    
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
