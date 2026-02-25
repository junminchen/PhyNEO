#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import openmm as mm
import openmm.app as app
import openmm.unit as unit

PDB_BANK = {
    "Li": "../pdb_bank/Li.pdb",
    "PF6": "../pdb_bank/PF6.pdb",
    "EC": "../pdb_bank/EC.pdb",
    "DMC": "../pdb_bank/DMC.pdb",
}


def main() -> None:
    here = Path(__file__).resolve().parent
    cfg = json.loads((here / "config.json").read_text())

    # build a tiny test system: 1 molecule per component
    first = app.PDBFile(str((here / PDB_BANK["Li"]).resolve()))
    modeller = app.Modeller(first.topology, first.positions)
    shift = mm.Vec3(0.6, 0.6, 0.6) * unit.nanometer

    for key in ("PF6", "EC", "DMC"):
        pdb = app.PDBFile(str((here / PDB_BANK[key]).resolve()))
        pos = [p + shift for p in pdb.positions]
        modeller.add(pdb.topology, pos)
        shift += mm.Vec3(0.6, 0.0, 0.0) * unit.nanometer

    ff_files = [str((here / p).resolve()) for p in cfg["forcefield_xml"]]
    ff = app.ForceField(*ff_files)
    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=cfg["openmm"]["nonbonded_cutoff_nm"] * unit.nanometer,
        constraints=app.HBonds,
    )

    print("ForceField mapping check: PASS")
    print(f"Num particles: {system.getNumParticles()}")
    print(f"Num forces: {system.getNumForces()}")
    for i in range(system.getNumForces()):
        print(f"  {i}: {system.getForce(i).__class__.__name__}")


if __name__ == "__main__":
    main()
