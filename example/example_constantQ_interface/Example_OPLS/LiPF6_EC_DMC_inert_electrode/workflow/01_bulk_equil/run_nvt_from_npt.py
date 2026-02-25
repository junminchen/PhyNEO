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
    p = argparse.ArgumentParser(description="Run NVT continuation from bulk NPT final.pdb")
    p.add_argument("--bulk-dir", default="../LiPF6_EC_DMC_minimal")
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--report-interval", type=int, default=500)
    p.add_argument("--platform", choices=["CPU", "Reference", "OpenCL", "CUDA"], default=None)
    args = p.parse_args()

    here = Path(__file__).resolve().parents[2]
    bulk = (here / args.bulk_dir).resolve()
    cfg = json.loads((bulk / "config.json").read_text())

    pdb_path = bulk / "final.pdb"
    if not pdb_path.exists():
        raise FileNotFoundError(f"{pdb_path} not found. Run bulk NPT first.")

    pdb = app.PDBFile(str(pdb_path))
    if pdb.topology.getPeriodicBoxVectors() is None:
        bx, by, bz = cfg["box_angstrom"]
        a = mm.Vec3(float(bx), 0.0, 0.0) * unit.angstrom
        b = mm.Vec3(0.0, float(by), 0.0) * unit.angstrom
        c = mm.Vec3(0.0, 0.0, float(bz)) * unit.angstrom
        pdb.topology.setPeriodicBoxVectors((a, b, c))

    ff = app.ForceField(*[str((bulk / p).resolve()) for p in cfg["forcefield_xml"]])
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
    platform = pick_platform(args.platform)
    print(f"Using OpenMM platform: {platform.getName()}")
    sim = app.Simulation(pdb.topology, system, integrator, platform)
    sim.context.setPositions(pdb.positions)

    sim.reporters.append(app.StateDataReporter(
        str(bulk / "nvt.log"),
        args.report_interval,
        step=True,
        potentialEnergy=True,
        kineticEnergy=True,
        totalEnergy=True,
        temperature=True,
        density=True,
        volume=True,
        speed=True,
    ))
    sim.reporters.append(app.DCDReporter(str(bulk / "traj_nvt.dcd"), args.report_interval))

    print("Running bulk NVT continuation...")
    sim.step(int(args.steps))

    state = sim.context.getState(getPositions=True)
    try:
        with open(bulk / "final_nvt.pdb", "w") as f:
            app.PDBFile.writeFile(sim.topology, state.getPositions(), f)
    except TypeError:
        sim.topology.setPeriodicBoxVectors(None)
        with open(bulk / "final_nvt.pdb", "w") as f:
            app.PDBFile.writeFile(sim.topology, state.getPositions(), f)

    print(f"Wrote {bulk / 'nvt.log'}")
    print(f"Wrote {bulk / 'traj_nvt.dcd'}")
    print(f"Wrote {bulk / 'final_nvt.pdb'}")


if __name__ == "__main__":
    main()
