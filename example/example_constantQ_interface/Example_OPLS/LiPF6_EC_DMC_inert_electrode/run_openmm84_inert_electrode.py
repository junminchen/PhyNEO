#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import openmm as mm
import openmm.app as app
import openmm.unit as unit

KJMOL_PER_E_PER_VOLT = 96.48533212331002


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


def main() -> None:
    parser = argparse.ArgumentParser(description="OPLS LiPF6-EC-DMC with inert electrodes under native OpenMM84 CPF")
    parser.add_argument("--equil-steps", type=int, default=None)
    parser.add_argument("--prod-steps", type=int, default=None)
    parser.add_argument("--report-interval", type=int, default=None)
    parser.add_argument("--platform", choices=["CPU", "Reference", "OpenCL", "CUDA"], default=None)
    parser.add_argument("--pdb", default=None, help="Input PDB. Default: start_with_electrodes_mc.pdb if present, else start_with_electrodes.pdb")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    cfg = json.loads((here / "config.json").read_text())
    md = cfg["md"]
    omm = cfg["openmm"]
    ele = cfg["electrode"]

    equil_steps = int(md["equil_steps"] if args.equil_steps is None else args.equil_steps)
    prod_steps = int(md["prod_steps"] if args.prod_steps is None else args.prod_steps)
    report_interval = int(md["report_interval"] if args.report_interval is None else args.report_interval)
    platform = pick_platform(args.platform)
    platform_name = platform.getName()
    print(f"Using OpenMM platform: {platform_name}")

    if args.pdb is not None:
        pdb_path = here / args.pdb
    else:
        mc_pdb = here / str(cfg.get("mc_gap", {}).get("output_pdb", "start_with_electrodes_mc.pdb"))
        pdb_path = mc_pdb if mc_pdb.exists() else (here / "start_with_electrodes.pdb")
    if not pdb_path.exists():
        raise FileNotFoundError(f"{pdb_path.name} not found. Run assemble_inert_electrode_system.py (and optional mc_gap_equilibrate.py) first.")

    pdb = app.PDBFile(str(pdb_path))
    if pdb.topology.getPeriodicBoxVectors() is None:
        cell = cfg["cell"]
        a = mm.Vec3(float(cell["box_x_angstrom"]), 0.0, 0.0) * unit.angstrom
        b = mm.Vec3(0.0, float(cell["box_y_angstrom"]), 0.0) * unit.angstrom
        c = mm.Vec3(0.0, 0.0, float(cell["box_z_angstrom"])) * unit.angstrom
        pdb.topology.setPeriodicBoxVectors((a, b, c))
        print(f'Applied periodic box from config: {cell["box_x_angstrom"]} x {cell["box_y_angstrom"]} x {cell["box_z_angstrom"]} A')
    ff = app.ForceField(*[str((here / p).resolve()) for p in cfg["forcefield_xml"]])

    method = app.PME if str(omm["nonbonded_method"]).upper() == "PME" else app.NoCutoff
    system = ff.createSystem(
        pdb.topology,
        nonbondedMethod=method,
        nonbondedCutoff=float(omm["nonbonded_cutoff_nm"]) * unit.nanometer,
        constraints=app.HBonds if str(omm["constraints"]) == "HBonds" else None,
        rigidWater=bool(omm.get("rigid_water", False)),
        ewaldErrorTolerance=float(omm.get("ewald_error_tolerance", 5e-4)),
        removeCMMotion=False,
    )

    nb = get_nonbonded(system)
    cath_atoms, ano_atoms = collect_electrode_atoms(pdb.topology)

    cpf = mm.ConstantPotentialForce()
    cpf.setCutoffDistance(float(omm["nonbonded_cutoff_nm"]))
    cpf.setConstantPotentialMethod(mm.ConstantPotentialForce.CG)
    cpf.setCGErrorTolerance(float(ele["cg_error_tol"]))
    cpf.setUseChargeConstraint(bool(ele.get("use_charge_constraint", False)))
    if bool(ele.get("use_charge_constraint", False)):
        cpf.setChargeConstraintTarget(float(ele.get("charge_constraint_target_e", 0.0)))

    for i in range(system.getNumParticles()):
        q, sig, eps = nb.getParticleParameters(i)
        q_e = q.value_in_unit(unit.elementary_charge)
        cpf.addParticle(float(q_e))
        nb.setParticleParameters(i, 0.0 * unit.elementary_charge, sig, eps)

    for j in range(nb.getNumExceptions()):
        p1, p2, qprod, sig, eps = nb.getExceptionParameters(j)
        cpf.addException(int(p1), int(p2), float(qprod.value_in_unit(unit.elementary_charge**2)))
        nb.setExceptionParameters(j, p1, p2, 0.0 * unit.elementary_charge**2, sig, eps)

    v = float(ele["voltage_v"])
    cath_pot = v * KJMOL_PER_E_PER_VOLT
    ano_pot = -v * KJMOL_PER_E_PER_VOLT
    cpf.addElectrode(set(cath_atoms), cath_pot, float(ele["gaussian_width_nm"]), float(ele["thomas_fermi_scale_invnm"]))
    cpf.addElectrode(set(ano_atoms), ano_pot, float(ele["gaussian_width_nm"]), float(ele["thomas_fermi_scale_invnm"]))
    system.addForce(cpf)

    barostat = mm.MonteCarloBarostat(md["pressure_bar"] * unit.bar, md["temperature_k"] * unit.kelvin, 25)
    system.addForce(barostat)

    for i in range(system.getNumForces()):
        system.getForce(i).setForceGroup(i)

    integrator = mm.LangevinMiddleIntegrator(
        md["temperature_k"] * unit.kelvin,
        md["friction_ps"] / unit.picosecond,
        md["timestep_fs"] * unit.femtosecond,
    )

    sim = app.Simulation(pdb.topology, system, integrator, platform)
    sim.context.setPositions(pdb.positions)

    sim.reporters.append(app.StateDataReporter(str(here / "npt_inert.log"), report_interval,
                                              step=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True,
                                              temperature=True, density=True, volume=True, speed=True))
    sim.reporters.append(app.DCDReporter(str(here / "traj_inert.dcd"), report_interval))

    charge_log = open(here / "electrode_charges.log", "w")
    charge_log.write("# step Q_cathode(e) Q_anode(e) Q_total(e)\n")

    def log_charges():
        charges = list(cpf.getCharges(sim.context))
        q_c = sum(float(charges[i].value_in_unit(unit.elementary_charge)) for i in cath_atoms)
        q_a = sum(float(charges[i].value_in_unit(unit.elementary_charge)) for i in ano_atoms)
        charge_log.write(f"{sim.currentStep} {q_c:.10f} {q_a:.10f} {(q_c+q_a):.10f}\n")
        charge_log.flush()

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
    with open(here / "final_inert.pdb", "w") as f:
        try:
            app.PDBFile.writeFile(sim.topology, state.getPositions(), f)
        except TypeError:
            sim.topology.setPeriodicBoxVectors(None)
            app.PDBFile.writeFile(sim.topology, state.getPositions(), f)

    charge_log.close()
    print("Done")


if __name__ == "__main__":
    main()
