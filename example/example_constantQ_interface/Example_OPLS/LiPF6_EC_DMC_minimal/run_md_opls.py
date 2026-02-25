#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import openmm as mm
import openmm.app as app
import openmm.unit as unit


def pick_platform(requested: str | None = None) -> mm.Platform:
    if requested:
        return mm.Platform.getPlatformByName(requested)
    for name in ("CUDA", "CPU", "Reference"):
        try:
            return mm.Platform.getPlatformByName(name)
        except Exception:
            continue
    raise RuntimeError("No usable OpenMM platform found")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OPLS LiPF6-EC-DMC MD")
    parser.add_argument("--equil-steps", type=int, default=None)
    parser.add_argument("--prod-steps", type=int, default=None)
    parser.add_argument("--report-interval", type=int, default=None)
    parser.add_argument("--platform", choices=["CPU", "Reference", "OpenCL", "CUDA"], default=None)
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    cfg = json.loads((here / "config.json").read_text())
    equil_steps = int(cfg["md"]["equil_steps"] if args.equil_steps is None else args.equil_steps)
    prod_steps = int(cfg["md"]["prod_steps"] if args.prod_steps is None else args.prod_steps)
    report_interval = int(cfg["md"]["report_interval"] if args.report_interval is None else args.report_interval)
    pdb_path = here / "start.pdb"
    if not pdb_path.exists():
        raise FileNotFoundError("start.pdb not found. Build with packmol first.")

    pdb = app.PDBFile(str(pdb_path))
    if pdb.topology.getPeriodicBoxVectors() is None:
        bx, by, bz = cfg["box_angstrom"]
        a = mm.Vec3(float(bx), 0.0, 0.0) * unit.angstrom
        b = mm.Vec3(0.0, float(by), 0.0) * unit.angstrom
        c = mm.Vec3(0.0, 0.0, float(bz)) * unit.angstrom
        pdb.topology.setPeriodicBoxVectors((a, b, c))
        print(f"Applied periodic box from config.json: {bx} x {by} x {bz} A")
    ff = app.ForceField(*[str((here / p).resolve()) for p in cfg["forcefield_xml"]])

    method = app.PME if cfg["openmm"]["nonbonded_method"].upper() == "PME" else app.NoCutoff
    system = ff.createSystem(
        pdb.topology,
        nonbondedMethod=method,
        nonbondedCutoff=cfg["openmm"]["nonbonded_cutoff_nm"] * unit.nanometer,
        constraints=app.HBonds if cfg["openmm"]["constraints"] == "HBonds" else None,
        rigidWater=bool(cfg["openmm"].get("rigid_water", False)),
        ewaldErrorTolerance=float(cfg["openmm"].get("ewald_error_tolerance", 5e-4)),
    )

    integrator = mm.LangevinMiddleIntegrator(
        cfg["md"]["temperature_k"] * unit.kelvin,
        cfg["md"]["friction_ps"] / unit.picosecond,
        cfg["md"]["timestep_fs"] * unit.femtosecond,
    )
    barostat = mm.MonteCarloBarostat(
        cfg["md"]["pressure_bar"] * unit.bar,
        cfg["md"]["temperature_k"] * unit.kelvin,
        25,
    )
    system.addForce(barostat)

    platform = pick_platform(args.platform)
    print(f"Using OpenMM platform: {platform.getName()}")
    sim = app.Simulation(pdb.topology, system, integrator, platform)
    sim.context.setPositions(pdb.positions)

    sim.reporters.append(app.StateDataReporter(
        str(here / "npt.log"),
        report_interval,
        step=True,
        potentialEnergy=True,
        kineticEnergy=True,
        totalEnergy=True,
        temperature=True,
        density=True,
        volume=True,
        speed=True,
    ))
    sim.reporters.append(app.DCDReporter(str(here / "traj.dcd"), report_interval))

    print("Minimizing...")
    sim.minimizeEnergy(maxIterations=1000)
    print("Equilibrating...")
    sim.step(equil_steps)
    print("Production...")
    sim.step(prod_steps)

    state = sim.context.getState(getPositions=True, getEnergy=True)
    try:
        with open(here / "final.pdb", "w") as f:
            app.PDBFile.writeFile(sim.topology, state.getPositions(), f)
    except TypeError:
        # Some generated topologies carry malformed box metadata for PDB CRYST1 writing.
        sim.topology.setPeriodicBoxVectors(None)
        with open(here / "final.pdb", "w") as f:
            app.PDBFile.writeFile(sim.topology, state.getPositions(), f)
    print("Done")


if __name__ == "__main__":
    main()
