#!/usr/bin/env python3
import argparse

from openmm import Context, Platform, VerletIntegrator
from openmm.app import ForceField, NoCutoff, PDBFile, PME
from openmm.unit import angstrom


def forcegroupify(system):
    forcegroups = {}
    for i in range(system.getNumForces()):
        force = system.getForce(i)
        force.setForceGroup(i)
        forcegroups[force] = i
    return forcegroups


def get_energy_decomposition(context, forcegroups):
    energies = {}
    for force, idx in forcegroups.items():
        state = context.getState(getEnergy=True, groups=2**idx)
        energies[force.getName()] = state.getPotentialEnergy()
    return energies


def main():
    parser = argparse.ArgumentParser(description="Compute OpenMM energy decomposition")
    parser.add_argument("--xml", required=True, help="OpenMM forcefield XML")
    parser.add_argument("--pdb", default="EC.pdb", help="PDB file")
    parser.add_argument("--method", choices=["nocutoff", "pme"], default="nocutoff")
    parser.add_argument("--cutoff-angstrom", type=float, default=8.0)
    parser.add_argument("--platform", default="Reference", help="OpenMM platform name")
    parser.add_argument(
        "--no-mpid-plugin",
        action="store_true",
        help="Disable loading mpidplugin before parsing XML",
    )
    args = parser.parse_args()

    if not args.no_mpid_plugin:
        try:
            import mpidplugin  # noqa: F401
            print("mpidplugin: loaded")
        except Exception as e:
            print(f"mpidplugin: not loaded ({e})")

    ff = ForceField(args.xml)
    pdb = PDBFile(args.pdb)

    if args.method == "pme":
        system = ff.createSystem(
            pdb.topology,
            nonbondedCutoff=args.cutoff_angstrom * angstrom,
            nonbondedMethod=PME,
        )
    else:
        system = ff.createSystem(
            pdb.topology,
            nonbondedCutoff=args.cutoff_angstrom * angstrom,
            nonbondedMethod=NoCutoff,
        )

    forcegroups = forcegroupify(system)
    integrator = VerletIntegrator(0.1)
    platform = Platform.getPlatformByName(args.platform)
    context = Context(system, integrator, platform)
    context.setPositions(pdb.positions)

    total_energy = context.getState(getEnergy=True).getPotentialEnergy()
    print(f"OpenMM Total Energy: {total_energy}")

    print("OpenMM Energy Components:")
    for name, e in get_energy_decomposition(context, forcegroups).items():
        print(f"{name}: {e}")


if __name__ == "__main__":
    main()
