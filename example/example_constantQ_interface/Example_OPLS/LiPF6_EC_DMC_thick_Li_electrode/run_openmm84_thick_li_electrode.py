#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import openmm as mm
import openmm.app as app
import openmm.unit as unit

KJMOL_PER_E_PER_VOLT = 96.48533212331002
AMU_TO_G = 1.66053906660e-24
NM3_TO_ML = 1.0e-21


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


def collect_electrolyte_atoms(topology: app.Topology):
    idx = []
    for res in topology.residues():
        if res.name in {"CAT", "ANO"}:
            continue
        idx.extend([a.index for a in res.atoms()])
    return idx


def electrolyte_mass_g(topology: app.Topology, atom_indices):
    mass_amu = 0.0
    atoms = list(topology.atoms())
    for i in atom_indices:
        e = atoms[i].element
        if e is not None:
            mass_amu += e.mass.value_in_unit(unit.dalton)
    return mass_amu * AMU_TO_G


def slab_density_g_ml(positions_ang, topology: app.Topology, cath_atoms, ano_atoms, electrolyte_indices):
    box = topology.getPeriodicBoxVectors()
    if box is None:
        return float("nan"), float("nan")
    ax_nm = box[0][0].value_in_unit(unit.nanometer)
    by_nm = box[1][1].value_in_unit(unit.nanometer)
    area_nm2 = float(ax_nm * by_nm)
    zc = max(float(positions_ang[i][2]) for i in cath_atoms)
    za = min(float(positions_ang[i][2]) for i in ano_atoms)
    gap_ang = za - zc
    if gap_ang <= 0:
        return float("nan"), gap_ang
    mass_g = electrolyte_mass_g(topology, electrolyte_indices)
    vol_ml = area_nm2 * (gap_ang * 0.1) * NM3_TO_ML
    return mass_g / vol_ml, gap_ang


def parse_tail_density_from_bulk_log(path: Path):
    if not path.exists():
        return None
    vals = []
    with path.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        cols = [h.replace("#", "").replace('"', "").strip() for h in header]
        if "Density (g/mL)" not in cols:
            return None
        i = cols.index("Density (g/mL)")
        for row in reader:
            if not row:
                continue
            try:
                vals.append(float(row[i]))
            except (ValueError, IndexError):
                continue
    if not vals:
        return None
    n = max(5, len(vals) // 3)
    return sum(vals[-n:]) / n


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
    # Disambiguate single-Li residue matching (CAT/ANO vs LiA).
    residue_templates = {}
    known = {"CAT", "ANO", "LiA", "PF6", "ECA", "DMC"}
    for res in topology.residues():
        if res.name in known:
            residue_templates[res] = res.name
    return residue_templates


def main() -> None:
    parser = argparse.ArgumentParser(description="OPLS LiPF6-EC-DMC with thick Li electrodes under native OpenMM84 CPF")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--equil-steps", type=int, default=None)
    parser.add_argument("--prod-steps", type=int, default=None)
    parser.add_argument("--report-interval", type=int, default=None)
    parser.add_argument("--voltage-v", type=float, default=None)
    parser.add_argument("--platform", choices=["CPU", "Reference", "OpenCL", "CUDA"], default=None)
    parser.add_argument("--pdb", default=None, help="Input PDB. Default: start_with_electrodes_mc.pdb if present, else start_with_electrodes.pdb")
    parser.add_argument("--state-log", default="npt_thick_li.log")
    parser.add_argument("--traj", default="traj_thick_li.dcd")
    parser.add_argument("--charge-log", default="electrode_charges.log")
    parser.add_argument("--final-pdb", default="final_thick_li.pdb")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    cfg = json.loads((here / args.config).read_text())
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
        raise FileNotFoundError(f"{pdb_path.name} not found. Run assemble_thick_li_electrode_system.py (and optional mc_gap_equilibrate.py) first.")

    pdb = app.PDBFile(str(pdb_path))
    if pdb.topology.getPeriodicBoxVectors() is None:
        cell = cfg["cell"]
        a = mm.Vec3(float(cell["box_x_angstrom"]), 0.0, 0.0) * unit.angstrom
        b = mm.Vec3(0.0, float(cell["box_y_angstrom"]), 0.0) * unit.angstrom
        c = mm.Vec3(0.0, 0.0, float(cell["box_z_angstrom"])) * unit.angstrom
        pdb.topology.setPeriodicBoxVectors((a, b, c))
        print(f'Applied periodic box from config: {cell["box_x_angstrom"]} x {cell["box_y_angstrom"]} x {cell["box_z_angstrom"]} A')
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
        removeCMMotion=False,
        residueTemplates=residue_templates,
    )

    nb = get_nonbonded(system)
    cath_atoms, ano_atoms = collect_electrode_atoms(pdb.topology)
    electrolyte_indices = collect_electrolyte_atoms(pdb.topology)
    for idx in cath_atoms + ano_atoms:
        system.setParticleMass(int(idx), 0.0 * unit.dalton)

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

    v = float(ele["voltage_v"] if args.voltage_v is None else args.voltage_v)
    cath_pot = v * KJMOL_PER_E_PER_VOLT
    ano_pot = -v * KJMOL_PER_E_PER_VOLT
    cpf.addElectrode(set(cath_atoms), cath_pot, float(ele["gaussian_width_nm"]), float(ele["thomas_fermi_scale_invnm"]))
    cpf.addElectrode(set(ano_atoms), ano_pot, float(ele["gaussian_width_nm"]), float(ele["thomas_fermi_scale_invnm"]))
    system.addForce(cpf)

    if bool(md.get("use_barostat", True)):
        barostat = mm.MonteCarloBarostat(md["pressure_bar"] * unit.bar, md["temperature_k"] * unit.kelvin, 25)
        system.addForce(barostat)
        ensemble_label = "NPT"
    else:
        ensemble_label = "NVT"

    for i in range(system.getNumForces()):
        system.getForce(i).setForceGroup(i)

    integrator = mm.LangevinMiddleIntegrator(
        md["temperature_k"] * unit.kelvin,
        md["friction_ps"] / unit.picosecond,
        md["timestep_fs"] * unit.femtosecond,
    )

    sim = app.Simulation(pdb.topology, system, integrator, platform)
    sim.context.setPositions(pdb.positions)
    init_pos_ang = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    rho0, gap0 = slab_density_g_ml(init_pos_ang, pdb.topology, cath_atoms, ano_atoms, electrolyte_indices)
    bulk_ref = parse_tail_density_from_bulk_log((here / "../LiPF6_EC_DMC_minimal/npt.log").resolve())
    if bulk_ref is not None:
        print(f"Bulk reference density (tail mean): {bulk_ref:.4f} g/mL")
    if gap0 == gap0:
        print(f"Initial slab gap: {gap0:.3f} A")
    if rho0 == rho0:
        print(f"Initial slab density: {rho0:.4f} g/mL")

    state_log_path = (here / args.state_log).resolve()
    traj_path = (here / args.traj).resolve()
    charge_log_path = (here / args.charge_log).resolve()
    final_pdb_path = (here / args.final_pdb).resolve()
    state_log_path.parent.mkdir(parents=True, exist_ok=True)
    traj_path.parent.mkdir(parents=True, exist_ok=True)
    charge_log_path.parent.mkdir(parents=True, exist_ok=True)
    final_pdb_path.parent.mkdir(parents=True, exist_ok=True)

    sim.reporters.append(app.StateDataReporter(str(state_log_path), report_interval,
                                              step=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True,
                                              temperature=True, density=True, volume=True, speed=True))
    sim.reporters.append(app.DCDReporter(str(traj_path), report_interval))

    charge_log = open(charge_log_path, "w")
    charge_log.write("# step Q_cathode(e) Q_anode(e) Q_total(e)\n")

    def log_charges():
        charges = list(cpf.getCharges(sim.context))
        q_c = sum(float(charges[i].value_in_unit(unit.elementary_charge)) for i in cath_atoms)
        q_a = sum(float(charges[i].value_in_unit(unit.elementary_charge)) for i in ano_atoms)
        charge_log.write(f"{sim.currentStep} {q_c:.10f} {q_a:.10f} {(q_c+q_a):.10f}\n")
        charge_log.flush()

    print(f"Running ensemble: {ensemble_label}")
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
    final_pos_ang = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    rho1, gap1 = slab_density_g_ml(final_pos_ang, pdb.topology, cath_atoms, ano_atoms, electrolyte_indices)
    with open(final_pdb_path, "w") as f:
        try:
            app.PDBFile.writeFile(sim.topology, state.getPositions(), f)
        except TypeError:
            sim.topology.setPeriodicBoxVectors(None)
            app.PDBFile.writeFile(sim.topology, state.getPositions(), f)

    charge_log.close()
    print(f"Voltage used: {v:.3f} V")
    print(f"Wrote state log: {state_log_path}")
    print(f"Wrote traj: {traj_path}")
    print(f"Wrote charge log: {charge_log_path}")
    print(f"Wrote final pdb: {final_pdb_path}")
    if gap1 == gap1:
        print(f"Final slab gap: {gap1:.3f} A")
    if rho1 == rho1:
        print(f"Final slab density: {rho1:.4f} g/mL")
        if bulk_ref is not None:
            print(f"Final vs bulk density delta: {abs(rho1 - bulk_ref):.4f} g/mL")
    print("Done")


if __name__ == "__main__":
    main()
