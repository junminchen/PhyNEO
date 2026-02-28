#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import openmm as mm
import openmm.app as app
import openmm.unit as unit


def get_nonbonded(system: mm.System) -> mm.NonbondedForce:
    for i in range(system.getNumForces()):
        f = system.getForce(i)
        if isinstance(f, mm.NonbondedForce):
            return f
    raise RuntimeError("NonbondedForce not found")


def collect_electrode_atoms(topology: app.Topology, cathode_chain_idx: int = 0, anode_chain_idx: int = 1):
    cath = []
    ano = []
    for ch in topology.chains():
        if ch.index == cathode_chain_idx:
            cath.extend([a.index for a in ch.atoms() if (a.element is not None and a.element.symbol != "H")])
        if ch.index == anode_chain_idx:
            ano.extend([a.index for a in ch.atoms() if (a.element is not None and a.element.symbol != "H")])
    if not cath or not ano:
        raise RuntimeError("Could not find electrode atoms on chains 0/1")
    return cath, ano


def pick_platform(requested: str | None = None) -> mm.Platform:
    if requested:
        return mm.Platform.getPlatformByName(requested)
    for name in ("CUDA", "CPU", "Reference"):
        try:
            return mm.Platform.getPlatformByName(name)
        except Exception:
            continue
    raise RuntimeError("No usable OpenMM platform found")


def build_residue_templates(topology: app.Topology):
    residue_templates = {}
    known = {"CAT", "ANO", "LiA", "PF6", "ECA", "DMC"}
    for res in topology.residues():
        if res.name in known:
            residue_templates[res] = res.name
    return residue_templates


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-layer inert graphene electrode with fixed-charge MD")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--pdb", default="start_fixedcharge_graphene.pdb")
    parser.add_argument("--equil-steps", type=int, default=None)
    parser.add_argument("--prod-steps", type=int, default=None)
    parser.add_argument("--report-interval", type=int, default=None)
    parser.add_argument("--platform", choices=["CPU", "Reference", "OpenCL", "CUDA"], default=None)
    parser.add_argument("--fixed-charge-per-atom-e", type=float, default=None)
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    cfg = json.loads((here / args.config).read_text())
    md = cfg["md"]
    omm = cfg["openmm"]
    ele = cfg["electrode"]

    equil_steps = int(md["equil_steps"] if args.equil_steps is None else args.equil_steps)
    prod_steps = int(md["prod_steps"] if args.prod_steps is None else args.prod_steps)
    report_interval = int(md["report_interval"] if args.report_interval is None else args.report_interval)
    qfix = float(ele["fixed_charge_per_atom_e"] if args.fixed_charge_per_atom_e is None else args.fixed_charge_per_atom_e)

    platform = pick_platform(args.platform if args.platform else omm.get("platform", None))
    print(f"Using OpenMM platform: {platform.getName()}")

    pdb_path = (here / args.pdb).resolve()
    if not pdb_path.exists():
        raise FileNotFoundError(f"{pdb_path.name} not found. Run assemble_singlelayer_graphene_system.py first.")

    pdb = app.PDBFile(str(pdb_path))
    if pdb.topology.getPeriodicBoxVectors() is None:
        cell = cfg["cell"]
        a = mm.Vec3(float(cell["box_x_angstrom"]), 0.0, 0.0) * unit.angstrom
        b = mm.Vec3(0.0, float(cell["box_y_angstrom"]), 0.0) * unit.angstrom
        c = mm.Vec3(0.0, 0.0, float(cell["box_z_angstrom"])) * unit.angstrom
        pdb.topology.setPeriodicBoxVectors((a, b, c))

    ff = app.ForceField(*[str((here / p).resolve()) for p in cfg["forcefield_xml"]])
    residue_templates = build_residue_templates(pdb.topology)

    method = app.PME if str(omm["nonbonded_method"]).upper() == "PME" else app.NoCutoff
    system = ff.createSystem(
        pdb.topology,
        nonbondedMethod=method,
        nonbondedCutoff=float(omm["nonbonded_cutoff_nm"]) * unit.nanometer,
        constraints=app.HBonds if str(omm["constraints"]) == "HBonds" else None,
        rigidWater=bool(omm.get("rigid_water", False)),
        ewaldErrorTolerance=float(omm.get("ewald_error_tolerance", 5e-4)),
        removeCMMotion=bool(omm.get("remove_cm_motion", True)),
        residueTemplates=residue_templates,
    )

    nb = get_nonbonded(system)
    cath_atoms, ano_atoms = collect_electrode_atoms(pdb.topology)

    if bool(ele.get("fix_electrode_positions", True)):
        for idx in cath_atoms + ano_atoms:
            system.setParticleMass(int(idx), 0.0 * unit.dalton)

    # Set fixed charges on electrodes directly in NonbondedForce.
    for idx in cath_atoms:
        q, sig, eps = nb.getParticleParameters(int(idx))
        nb.setParticleParameters(int(idx), qfix * unit.elementary_charge, sig, eps)
    for idx in ano_atoms:
        q, sig, eps = nb.getParticleParameters(int(idx))
        nb.setParticleParameters(int(idx), -qfix * unit.elementary_charge, sig, eps)

    # Update exception charge products for pairs involving electrode atoms.
    cath_set = set(cath_atoms)
    ano_set = set(ano_atoms)
    for j in range(nb.getNumExceptions()):
        p1, p2, qprod, sig, eps = nb.getExceptionParameters(j)
        p1 = int(p1)
        p2 = int(p2)
        if p1 in cath_set:
            q1 = qfix
        elif p1 in ano_set:
            q1 = -qfix
        else:
            q1 = nb.getParticleParameters(p1)[0].value_in_unit(unit.elementary_charge)

        if p2 in cath_set:
            q2 = qfix
        elif p2 in ano_set:
            q2 = -qfix
        else:
            q2 = nb.getParticleParameters(p2)[0].value_in_unit(unit.elementary_charge)

        scale = 1.0
        if abs(sig.value_in_unit(unit.nanometer)) > 0 or abs(eps.value_in_unit(unit.kilojoule_per_mole)) > 0:
            # Keep original 1-4 electrostatic scaling if it exists.
            qprod_old = qprod.value_in_unit(unit.elementary_charge**2)
            q1_old = nb.getParticleParameters(p1)[0].value_in_unit(unit.elementary_charge)
            q2_old = nb.getParticleParameters(p2)[0].value_in_unit(unit.elementary_charge)
            denom = q1_old * q2_old
            if abs(denom) > 1e-12:
                scale = qprod_old / denom
        qprod_new = scale * q1 * q2
        nb.setExceptionParameters(j, p1, p2, qprod_new * unit.elementary_charge**2, sig, eps)

    if bool(md.get("use_barostat", False)):
        system.addForce(mm.MonteCarloBarostat(md["pressure_bar"] * unit.bar, md["temperature_k"] * unit.kelvin, 25))

    integrator = mm.LangevinMiddleIntegrator(
        md["temperature_k"] * unit.kelvin,
        md["friction_ps"] / unit.picosecond,
        md["timestep_fs"] * unit.femtosecond,
    )

    sim = app.Simulation(pdb.topology, system, integrator, platform)
    sim.context.setPositions(pdb.positions)

    sim.reporters.append(app.StateDataReporter(str(here / "nvt_fixedcharge.log"), report_interval,
                                              step=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True,
                                              temperature=True, density=True, volume=True, speed=True))
    sim.reporters.append(app.DCDReporter(str(here / "traj_fixedcharge.dcd"), report_interval))

    charge_log = open(here / "electrode_fixed_charges.log", "w")
    charge_log.write("# step q_per_atom_e Q_cathode(e) Q_anode(e)\n")

    def log_charges():
        q_c = qfix * len(cath_atoms)
        q_a = -qfix * len(ano_atoms)
        charge_log.write(f"{sim.currentStep} {qfix:.8f} {q_c:.8f} {q_a:.8f}\n")
        charge_log.flush()

    print(f"Fixed electrode charge per atom: cathode +{qfix:.6f} e, anode -{qfix:.6f} e")
    print("Minimizing...")
    sim.minimizeEnergy(maxIterations=1000)
    log_charges()

    print("Equilibrating...")
    nblock_e = max(1, equil_steps // report_interval)
    rem_e = equil_steps - nblock_e * report_interval
    for _ in range(nblock_e):
        sim.step(report_interval)
        log_charges()
    if rem_e > 0:
        sim.step(rem_e)
        log_charges()

    print("Production...")
    nblock_p = max(1, prod_steps // report_interval)
    rem_p = prod_steps - nblock_p * report_interval
    for _ in range(nblock_p):
        sim.step(report_interval)
        log_charges()
    if rem_p > 0:
        sim.step(rem_p)
        log_charges()

    state = sim.context.getState(getPositions=True)
    with open(here / "final_fixedcharge.pdb", "w") as f:
        try:
            app.PDBFile.writeFile(sim.topology, state.getPositions(), f)
        except TypeError:
            sim.topology.setPeriodicBoxVectors(None)
            app.PDBFile.writeFile(sim.topology, state.getPositions(), f)

    charge_log.close()
    print("Done")


if __name__ == "__main__":
    main()
