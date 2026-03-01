#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import openmm as mm
import openmm.app as app
import openmm.unit as unit

AMU_TO_G = 1.66053906660e-24
NM3_TO_ML = 1.0e-21
KJMOL_PER_E_PER_VOLT = 96.48533212331002


@dataclass
class Stage1Summary:
    target_density_g_ml: float
    picked_density_g_ml: float
    picked_step: int
    density_abs_error_g_ml: float
    fixed_lz_angstrom: float
    n_samples: int
    lz_samples: list[float]
    density_samples: list[float]


@dataclass
class ResidueGroup:
    name: str
    atom_indices: list[int]
    atom_masses_amu: list[float]


def pick_platform(requested: str | None = None) -> mm.Platform:
    if requested:
        return mm.Platform.getPlatformByName(requested)
    for name in ("CUDA", "CPU", "Reference"):
        try:
            return mm.Platform.getPlatformByName(name)
        except Exception:
            continue
    raise RuntimeError("No usable OpenMM platform found")


def parse_bulk_density(log_path: Path) -> float:
    dens = []
    with log_path.open() as f:
        r = csv.reader(f)
        h = [x.replace("#", "").replace('"', "").strip() for x in next(r)]
        idx = {k: i for i, k in enumerate(h)}
        i_den = idx["Density (g/mL)"]
        for row in r:
            if not row:
                continue
            try:
                dens.append(float(row[i_den]))
            except (ValueError, IndexError):
                continue
    if not dens:
        raise RuntimeError(f"No density rows found in {log_path}")
    tail_n = max(5, len(dens) // 3)
    return sum(dens[-tail_n:]) / tail_n


def build_residue_templates(topology: app.Topology) -> dict[app.Topology.Residue, str]:
    templates = {}
    known = {"CAT", "ANO", "LiA", "PF6", "ECA", "DMC"}
    for res in topology.residues():
        if res.name in known:
            templates[res] = res.name
    return templates


def get_nonbonded(system: mm.System) -> mm.NonbondedForce:
    for i in range(system.getNumForces()):
        force = system.getForce(i)
        if isinstance(force, mm.NonbondedForce):
            return force
    raise RuntimeError("NonbondedForce not found")


def collect_electrode_atoms(topology: app.Topology) -> tuple[list[int], list[int]]:
    left = []
    right = []
    for res in topology.residues():
        rn = res.name.strip()
        if rn == "CAT":
            left.extend([a.index for a in res.atoms() if a.element is not None and a.element.symbol != "H"])
        elif rn == "ANO":
            right.extend([a.index for a in res.atoms() if a.element is not None and a.element.symbol != "H"])
    if not left or not right:
        raise RuntimeError("Failed to identify electrode atoms from CAT/ANO residues")
    return sorted(set(left)), sorted(set(right))


def electrolyte_mass_amu(topology: app.Topology) -> float:
    mass = 0.0
    for res in topology.residues():
        if res.name.strip() in {"CAT", "ANO"}:
            continue
        for atom in res.atoms():
            if atom.element is not None:
                mass += atom.element.mass.value_in_unit(unit.dalton)
    return mass


def slab_density_from_state(
    state: mm.State,
    left_electrode_atoms: list[int],
    right_electrode_atoms: list[int],
    electrolyte_mass_g: float,
) -> float:
    pos = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    box = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.angstrom)
    lx_nm = float(box[0][0]) * 0.1
    ly_nm = float(box[1][1]) * 0.1
    z_left_surface = max(float(pos[i][2]) for i in left_electrode_atoms)
    z_right_surface = min(float(pos[i][2]) for i in right_electrode_atoms)
    gap_angstrom = z_right_surface - z_left_surface
    if gap_angstrom <= 0:
        raise RuntimeError(f"Invalid gap along z: {gap_angstrom:.3f} A")
    gap_nm = gap_angstrom * 0.1
    vol_ml = lx_nm * ly_nm * gap_nm * NM3_TO_ML
    return electrolyte_mass_g / vol_ml


def set_periodic_box_with_lz(context: mm.Context, lz_angstrom: float) -> None:
    state = context.getState()
    box = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.angstrom)
    ax = float(box[0][0])
    by = float(box[1][1])
    a = mm.Vec3(ax, 0.0, 0.0) * unit.angstrom
    b = mm.Vec3(0.0, by, 0.0) * unit.angstrom
    c = mm.Vec3(0.0, 0.0, float(lz_angstrom)) * unit.angstrom
    context.setPeriodicBoxVectors(a, b, c)


def build_electrolyte_residue_groups(topology: app.Topology) -> list[ResidueGroup]:
    groups: list[ResidueGroup] = []
    for res in topology.residues():
        rn = res.name.strip()
        if rn in {"CAT", "ANO"}:
            continue
        atom_indices: list[int] = []
        atom_masses_amu: list[float] = []
        for atom in res.atoms():
            atom_indices.append(int(atom.index))
            if atom.element is None:
                atom_masses_amu.append(0.0)
            else:
                atom_masses_amu.append(float(atom.element.mass.value_in_unit(unit.dalton)))
        groups.append(ResidueGroup(name=rn, atom_indices=atom_indices, atom_masses_amu=atom_masses_amu))
    return groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Three-stage LiPF6/EC/DMC interfacial simulation: "
            "(1) Neutral z-only NPT, (2) CP activation NVT, (3) Production NVT"
        )
    )
    parser.add_argument("--config", default="config_three_stage.json")
    parser.add_argument("--platform", choices=["CPU", "Reference", "OpenCL", "CUDA"], default=None)
    parser.add_argument("--neutral-relax-steps", type=int, default=None)
    parser.add_argument("--cp-activation-steps", type=int, default=None)
    parser.add_argument("--production-steps", type=int, default=None)
    parser.add_argument("--report-interval", type=int, default=None)
    parser.add_argument("--bulk-log", default=None, help="Bulk NPT log used for target density matching")
    parser.add_argument("--density-only", action="store_true", help="Run only Stage 1 density matching and stop before CP")
    parser.add_argument("--delta-phi-v", type=float, default=None, help="DeltaPhi = Phi_R - Phi_L (volts)")
    parser.add_argument("--phi-center-v", type=float, default=None, help="Average potential center (volts)")
    parser.add_argument("--z-density-bins", type=int, default=180, help="Number of z bins for number-density profile")
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args()


def make_base_system(pdb: app.PDBFile, cfg: dict, freeze_electrodes: bool = True) -> tuple[mm.System, list[int], list[int]]:
    here = Path(__file__).resolve().parent
    ff = app.ForceField(*[str((here / f).resolve()) for f in cfg["forcefield_xml"]])
    omm = cfg["openmm"]
    method = app.PME if str(omm["nonbonded_method"]).upper() == "PME" else app.NoCutoff
    templates = build_residue_templates(pdb.topology)
    system = ff.createSystem(
        pdb.topology,
        nonbondedMethod=method,
        nonbondedCutoff=float(omm["nonbonded_cutoff_nm"]) * unit.nanometer,
        constraints=app.HBonds if str(omm["constraints"]) == "HBonds" else None,
        rigidWater=bool(omm.get("rigid_water", False)),
        ewaldErrorTolerance=float(omm.get("ewald_error_tolerance", 5e-4)),
        removeCMMotion=bool(omm.get("remove_cm_motion", True)),
        residueTemplates=templates,
    )
    left_atoms, right_atoms = collect_electrode_atoms(pdb.topology)
    if freeze_electrodes:
        for atom_index in left_atoms + right_atoms:
            system.setParticleMass(int(atom_index), 0.0 * unit.dalton)
    return system, left_atoms, right_atoms


def run_stage1_neutral_relaxation(
    pdb: app.PDBFile,
    cfg: dict,
    platform: mm.Platform,
    neutral_steps: int,
    report_interval: int,
    target_density_g_ml: float,
    out_dir: Path,
) -> tuple[app.PDBFile, Stage1Summary]:
    md = cfg["md"]
    system, left_atoms, right_atoms = make_base_system(pdb, cfg)

    pressure = float(md["pressure_bar"])
    temperature = float(md["temperature_k"])
    barostat_interval = int(md.get("barostat_interval", 25))
    # Density-matching stage: keep slab gap stable (Z fixed) and relax in-plane area.
    barostat = mm.MonteCarloMembraneBarostat(
        pressure * unit.bar,
        0.0 * unit.bar * unit.nanometer,
        temperature * unit.kelvin,
        mm.MonteCarloMembraneBarostat.XYIsotropic,
        mm.MonteCarloMembraneBarostat.ZFixed,
        barostat_interval,
    )
    system.addForce(barostat)

    integrator = mm.LangevinMiddleIntegrator(
        temperature * unit.kelvin,
        float(md["friction_ps"]) / unit.picosecond,
        float(md["timestep_fs"]) * unit.femtosecond,
    )
    sim = app.Simulation(pdb.topology, system, integrator, platform)
    sim.context.setPositions(pdb.positions)

    stage1_dir = out_dir / "stage1_neutral_relaxation"
    stage1_dir.mkdir(parents=True, exist_ok=True)
    log_path = stage1_dir / "neutral_relax.log"
    traj_path = stage1_dir / "neutral_relax.dcd"
    sim.reporters.append(
        app.StateDataReporter(
            str(log_path),
            report_interval,
            step=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True,
            temperature=True,
            density=True,
            volume=True,
            speed=True,
        )
    )
    sim.reporters.append(app.DCDReporter(str(traj_path), report_interval))

    mass_g = electrolyte_mass_amu(pdb.topology) * AMU_TO_G
    lz_samples: list[float] = []
    density_samples: list[float] = []

    print("[Stage 1] Neutral Relaxation: CP off, density matching against bulk target")
    print(f"  Target slab density: {target_density_g_ml:.6f} g/mL")
    sim.minimizeEnergy(maxIterations=1000)

    n_blocks = max(1, neutral_steps // report_interval)
    remainder = neutral_steps - n_blocks * report_interval
    best_err = float("inf")
    best_step = -1
    best_pos = None
    best_box = None
    best_rho = None

    def sample_lz_and_density() -> None:
        nonlocal best_err, best_step, best_pos, best_box, best_rho
        st = sim.context.getState(getPositions=True, enforcePeriodicBox=True)
        box = st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.angstrom)
        lz = float(box[2][2])
        rho = slab_density_from_state(st, left_atoms, right_atoms, mass_g)
        lz_samples.append(lz)
        density_samples.append(rho)
        err = abs(rho - target_density_g_ml)
        if err < best_err:
            best_err = err
            best_step = int(sim.currentStep)
            best_pos = st.getPositions(asNumpy=True)
            best_box = st.getPeriodicBoxVectors()
            best_rho = rho

    for _ in range(n_blocks):
        sim.step(report_interval)
        sample_lz_and_density()
    if remainder > 0:
        sim.step(remainder)
        sample_lz_and_density()

    if best_pos is None or best_box is None or best_rho is None:
        raise RuntimeError("Stage 1 did not capture a valid best frame")
    lz_from_best = float(best_box[2][2].value_in_unit(unit.angstrom))
    picked_pdb_path = stage1_dir / "stage1_fixed_lz_start.pdb"
    with picked_pdb_path.open("w") as fh:
        pdb.topology.setPeriodicBoxVectors(best_box)
        app.PDBFile.writeFile(pdb.topology, best_pos, fh)

    summary = Stage1Summary(
        target_density_g_ml=target_density_g_ml,
        picked_density_g_ml=best_rho,
        picked_step=best_step,
        density_abs_error_g_ml=best_err,
        fixed_lz_angstrom=lz_from_best,
        n_samples=len(lz_samples),
        lz_samples=lz_samples,
        density_samples=density_samples,
    )
    summary_path = stage1_dir / "stage1_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "target_density_g_ml": summary.target_density_g_ml,
                "picked_density_g_ml": summary.picked_density_g_ml,
                "picked_step": summary.picked_step,
                "density_abs_error_g_ml": summary.density_abs_error_g_ml,
                "fixed_lz_angstrom": summary.fixed_lz_angstrom,
                "n_samples": summary.n_samples,
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n"
    )

    return app.PDBFile(str(picked_pdb_path)), summary


def add_constant_potential_force(
    system: mm.System,
    left_atoms: list[int],
    right_atoms: list[int],
    cfg: dict,
    delta_phi_v: float,
    phi_center_v: float,
) -> mm.ConstantPotentialForce:
    ele = cfg["electrode"]
    omm = cfg["openmm"]

    nb = get_nonbonded(system)
    cpf = mm.ConstantPotentialForce()
    cpf.setCutoffDistance(float(omm["nonbonded_cutoff_nm"]))
    cpf.setConstantPotentialMethod(mm.ConstantPotentialForce.CG)
    cpf.setCGErrorTolerance(float(ele["cg_error_tol"]))
    cpf.setUseChargeConstraint(bool(ele.get("use_charge_constraint", False)))
    if bool(ele.get("use_charge_constraint", False)):
        cpf.setChargeConstraintTarget(float(ele.get("charge_constraint_target_e", 0.0)))

    for idx in range(system.getNumParticles()):
        q, sigma, epsilon = nb.getParticleParameters(idx)
        cpf.addParticle(float(q.value_in_unit(unit.elementary_charge)))
        nb.setParticleParameters(idx, 0.0 * unit.elementary_charge, sigma, epsilon)

    for j in range(nb.getNumExceptions()):
        p1, p2, qprod, sigma, epsilon = nb.getExceptionParameters(j)
        cpf.addException(int(p1), int(p2), float(qprod.value_in_unit(unit.elementary_charge**2)))
        nb.setExceptionParameters(j, p1, p2, 0.0 * unit.elementary_charge**2, sigma, epsilon)

    phi_left_v = phi_center_v - 0.5 * delta_phi_v
    phi_right_v = phi_center_v + 0.5 * delta_phi_v
    cpf.addElectrode(
        set(left_atoms),
        float(phi_left_v) * KJMOL_PER_E_PER_VOLT,
        float(ele["gaussian_width_nm"]),
        float(ele["thomas_fermi_scale_invnm"]),
    )
    cpf.addElectrode(
        set(right_atoms),
        float(phi_right_v) * KJMOL_PER_E_PER_VOLT,
        float(ele["gaussian_width_nm"]),
        float(ele["thomas_fermi_scale_invnm"]),
    )
    system.addForce(cpf)
    return cpf


def run_stage2_stage3_cp(
    pdb: app.PDBFile,
    cfg: dict,
    platform: mm.Platform,
    avg_lz_angstrom: float,
    activation_steps: int,
    production_steps: int,
    report_interval: int,
    delta_phi_v: float,
    phi_center_v: float,
    z_density_bins: int,
    out_dir: Path,
) -> None:
    md = cfg["md"]
    system, left_atoms, right_atoms = make_base_system(pdb, cfg)
    cpf = add_constant_potential_force(system, left_atoms, right_atoms, cfg, delta_phi_v, phi_center_v)

    integrator = mm.LangevinMiddleIntegrator(
        float(md["temperature_k"]) * unit.kelvin,
        float(md["friction_ps"]) / unit.picosecond,
        float(md["timestep_fs"]) * unit.femtosecond,
    )
    sim = app.Simulation(pdb.topology, system, integrator, platform)
    sim.context.setPositions(pdb.positions)
    set_periodic_box_with_lz(sim.context, avg_lz_angstrom)

    stage2_dir = out_dir / "stage2_cp_activation"
    stage3_dir = out_dir / "stage3_production"
    stage2_dir.mkdir(parents=True, exist_ok=True)
    stage3_dir.mkdir(parents=True, exist_ok=True)

    act_log = stage2_dir / "cp_activation.log"
    act_traj = stage2_dir / "cp_activation.dcd"
    prod_log = stage3_dir / "production.log"
    prod_traj = stage3_dir / "production.dcd"
    total_charge_log = out_dir / "electrode_total_charge_timeseries.dat"
    atom_charge_log = out_dir / "electrode_atom_charges.csv"
    z_density_profile = stage3_dir / "z_number_density_profile.csv"

    left_set = set(left_atoms)
    right_set = set(right_atoms)
    residue_groups = build_electrolyte_residue_groups(pdb.topology)
    species = sorted(set(r.name for r in residue_groups))
    accum_counts = {sp: [0.0] * z_density_bins for sp in species}
    n_density_frames = 0

    def accumulate_z_number_density_frame() -> None:
        nonlocal n_density_frames
        st = sim.context.getState(getPositions=True, enforcePeriodicBox=True)
        pos = st.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        box = st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.angstrom)
        lz = float(box[2][2])
        if lz <= 0.0:
            raise RuntimeError(f"Invalid lz: {lz}")
        for group in residue_groups:
            mass_sum = sum(group.atom_masses_amu)
            if mass_sum > 0.0:
                z = 0.0
                for idx, m in zip(group.atom_indices, group.atom_masses_amu):
                    z += float(pos[idx][2]) * m
                z /= mass_sum
            else:
                z = sum(float(pos[idx][2]) for idx in group.atom_indices) / max(1, len(group.atom_indices))
            z_wrapped = z % lz
            bin_id = int(z_wrapped / lz * z_density_bins)
            if bin_id >= z_density_bins:
                bin_id = z_density_bins - 1
            accum_counts[group.name][bin_id] += 1.0
        n_density_frames += 1

    with total_charge_log.open("w") as f_total, atom_charge_log.open("w", newline="") as f_atom:
        f_total.write("# step Q_left_e Q_right_e Q_total_e\n")
        atom_writer = csv.writer(f_atom)
        atom_writer.writerow(["step", "atom_index", "electrode", "charge_e"])

        def log_charges() -> None:
            charge_vec = list(cpf.getCharges(sim.context))
            q_left = 0.0
            q_right = 0.0
            step = sim.currentStep
            for atom_index, q in enumerate(charge_vec):
                q_e = float(q.value_in_unit(unit.elementary_charge))
                if atom_index in left_set:
                    q_left += q_e
                    atom_writer.writerow([step, atom_index, "L", f"{q_e:.10f}"])
                elif atom_index in right_set:
                    q_right += q_e
                    atom_writer.writerow([step, atom_index, "R", f"{q_e:.10f}"])
            f_total.write(f"{step} {q_left:.10f} {q_right:.10f} {(q_left + q_right):.10f}\n")
            f_total.flush()
            f_atom.flush()

        print("[Stage 2] CP-Activation: set DeltaPhi and run NVT")
        print(f"  DeltaPhi = {delta_phi_v:.4f} V, Phi_center = {phi_center_v:.4f} V")
        sim.reporters.append(
            app.StateDataReporter(
                str(act_log),
                report_interval,
                step=True,
                potentialEnergy=True,
                kineticEnergy=True,
                totalEnergy=True,
                temperature=True,
                density=True,
                volume=True,
                speed=True,
            )
        )
        sim.reporters.append(app.DCDReporter(str(act_traj), report_interval))

        n_blocks = max(1, activation_steps // report_interval)
        rem = activation_steps - n_blocks * report_interval
        for _ in range(n_blocks):
            sim.step(report_interval)
            log_charges()
        if rem > 0:
            sim.step(rem)
            log_charges()

        sim.reporters.clear()

        print("[Stage 3] Production: keep NVT and collect CPF charges")
        sim.reporters.append(
            app.StateDataReporter(
                str(prod_log),
                report_interval,
                step=True,
                potentialEnergy=True,
                kineticEnergy=True,
                totalEnergy=True,
                temperature=True,
                density=True,
                volume=True,
                speed=True,
            )
        )
        sim.reporters.append(app.DCDReporter(str(prod_traj), report_interval))

        n_blocks = max(1, production_steps // report_interval)
        rem = production_steps - n_blocks * report_interval
        for _ in range(n_blocks):
            sim.step(report_interval)
            log_charges()
            accumulate_z_number_density_frame()
        if rem > 0:
            sim.step(rem)
            log_charges()
            accumulate_z_number_density_frame()

    if n_density_frames > 0:
        st = sim.context.getState(getPositions=True, enforcePeriodicBox=True)
        box = st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.angstrom)
        lx_nm = float(box[0][0]) * 0.1
        ly_nm = float(box[1][1]) * 0.1
        lz_nm = float(box[2][2]) * 0.1
        area_nm2 = lx_nm * ly_nm
        dz_nm = lz_nm / float(z_density_bins)
        total_counts = [0.0] * z_density_bins
        for sp in species:
            for i in range(z_density_bins):
                total_counts[i] += accum_counts[sp][i]

        with z_density_profile.open("w", newline="") as fz:
            writer = csv.writer(fz)
            writer.writerow(["z_center_angstrom", "number_density_total_nm^-3"] + [f"number_density_{sp}_nm^-3" for sp in species])
            for i in range(z_density_bins):
                z_center_a = (i + 0.5) * (lz_nm * 10.0) / float(z_density_bins)
                total_rho = total_counts[i] / (n_density_frames * area_nm2 * dz_nm)
                row = [f"{z_center_a:.6f}", f"{total_rho:.10f}"]
                for sp in species:
                    rho_sp = accum_counts[sp][i] / (n_density_frames * area_nm2 * dz_nm)
                    row.append(f"{rho_sp:.10f}")
                writer.writerow(row)

    final_pdb = stage3_dir / "production_final.pdb"
    state = sim.context.getState(getPositions=True)
    with final_pdb.open("w") as fh:
        app.PDBFile.writeFile(sim.topology, state.getPositions(), fh)

    summary_path = out_dir / "cp_setup_summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                f"delta_phi_v: {delta_phi_v:.6f}",
                f"phi_center_v: {phi_center_v:.6f}",
                f"phi_left_v: {phi_center_v - 0.5 * delta_phi_v:.6f}",
                f"phi_right_v: {phi_center_v + 0.5 * delta_phi_v:.6f}",
                f"fixed_lz_angstrom: {avg_lz_angstrom:.6f}",
                f"activation_steps: {activation_steps}",
                f"production_steps: {production_steps}",
                f"z_density_bins: {z_density_bins}",
                f"z_number_density_profile: {z_density_profile}",
            ]
        )
        + "\n"
    )


def main() -> None:
    args = parse_args()
    here = Path(__file__).resolve().parent
    cfg = json.loads((here / args.config).read_text())

    md = cfg["md"]
    ele = cfg["electrode"]
    bulk_cfg = cfg.get("bulk_reference", {})
    neutral_steps = int(md["neutral_relax_steps"] if args.neutral_relax_steps is None else args.neutral_relax_steps)
    activation_steps = int(md["cp_activation_steps"] if args.cp_activation_steps is None else args.cp_activation_steps)
    production_steps = int(md["production_steps"] if args.production_steps is None else args.production_steps)
    report_interval = int(md["report_interval"] if args.report_interval is None else args.report_interval)
    delta_phi_v = float(ele["delta_phi_v"] if args.delta_phi_v is None else args.delta_phi_v)
    phi_center_v = float(ele.get("phi_center_v", 0.0) if args.phi_center_v is None else args.phi_center_v)
    z_density_bins = int(args.z_density_bins)
    bulk_log_rel = bulk_cfg.get("npt_log", "../../LiPF6_EC_DMC_minimal/npt.log")
    bulk_log = (here / (bulk_log_rel if args.bulk_log is None else args.bulk_log)).resolve()
    target_density_g_ml = parse_bulk_density(bulk_log)

    pdb_path = (here / cfg["input"]["pdb"]).resolve()
    if not pdb_path.exists():
        raise FileNotFoundError(f"Input PDB not found: {pdb_path}")
    pdb = app.PDBFile(str(pdb_path))
    if pdb.topology.getPeriodicBoxVectors() is None:
        cell = cfg.get("cell", {})
        if not cell:
            raise RuntimeError("Input PDB has no periodic box and config.cell is missing")
        a = mm.Vec3(float(cell["box_x_angstrom"]), 0.0, 0.0) * unit.angstrom
        b = mm.Vec3(0.0, float(cell["box_y_angstrom"]), 0.0) * unit.angstrom
        c = mm.Vec3(0.0, 0.0, float(cell["box_z_angstrom"])) * unit.angstrom
        pdb.topology.setPeriodicBoxVectors((a, b, c))

    platform = pick_platform(args.platform)
    print(f"Using OpenMM platform: {platform.getName()}")

    out_dir = (here / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    stage1_pdb, stage1_summary = run_stage1_neutral_relaxation(
        pdb=pdb,
        cfg=cfg,
        platform=platform,
        neutral_steps=neutral_steps,
        report_interval=report_interval,
        target_density_g_ml=target_density_g_ml,
        out_dir=out_dir,
    )

    print(
        "[Stage 1 Summary] "
        f"picked step = {stage1_summary.picked_step}, "
        f"target rho = {stage1_summary.target_density_g_ml:.6f} g/mL, "
        f"picked rho = {stage1_summary.picked_density_g_ml:.6f} g/mL, "
        f"|drho| = {stage1_summary.density_abs_error_g_ml:.6f} g/mL, "
        f"fixed Lz = {stage1_summary.fixed_lz_angstrom:.4f} A"
    )

    if args.density_only:
        print("Density-only mode enabled. Stop after Stage 1.")
        print(f"Outputs directory: {out_dir}")
        return

    run_stage2_stage3_cp(
        pdb=stage1_pdb,
        cfg=cfg,
        platform=platform,
        avg_lz_angstrom=stage1_summary.fixed_lz_angstrom,
        activation_steps=activation_steps,
        production_steps=production_steps,
        report_interval=report_interval,
        delta_phi_v=delta_phi_v,
        phi_center_v=phi_center_v,
        z_density_bins=z_density_bins,
        out_dir=out_dir,
    )

    print("Three-stage workflow finished.")
    print(f"Outputs directory: {out_dir}")


if __name__ == "__main__":
    main()
