#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

import openmm as mm
import openmm.app as app
import openmm.unit as unit

KJMOL_PER_E_PER_VOLT = 96.48533212331002
KB_KJ_MOL_K = 0.00831446261815324
AMU_TO_G = 1.66053906660e-24
NM3_TO_ML = 1.0e-21


def get_nonbonded(system: mm.System) -> mm.NonbondedForce:
    for i in range(system.getNumForces()):
        f = system.getForce(i)
        if isinstance(f, mm.NonbondedForce):
            return f
    raise RuntimeError("NonbondedForce not found")


def collect_atoms(topology: app.Topology):
    cath, ano, ele = [], [], []
    for ch in topology.chains():
        if ch.index == 0:
            cath.extend([a.index for a in ch.atoms() if (a.element is not None and a.element.symbol != "H")])
        elif ch.index == 1:
            ano.extend([a.index for a in ch.atoms() if (a.element is not None and a.element.symbol != "H")])
        else:
            ele.extend([a.index for a in ch.atoms()])
    if not cath or not ano:
        raise RuntimeError("Electrode atoms not found on chains 0/1")
    return cath, ano, ele


def surface_z_ang(pos_ang, idxs, which: str):
    zvals = [pos_ang[i][2] for i in idxs]
    if which == "cathode":
        return max(zvals)
    if which == "anode":
        return min(zvals)
    raise ValueError(f"Unknown electrode kind: {which}")


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


def make_cpf(system: mm.System, nb: mm.NonbondedForce, cath_atoms, ano_atoms, cfg):
    ele = cfg["electrode"]
    omm = cfg["openmm"]

    cpf = mm.ConstantPotentialForce()
    cpf.setCutoffDistance(float(omm["nonbonded_cutoff_nm"]))
    cpf.setConstantPotentialMethod(mm.ConstantPotentialForce.CG)
    cpf.setCGErrorTolerance(float(ele["cg_error_tol"]))
    cpf.setUseChargeConstraint(bool(ele.get("use_charge_constraint", False)))
    if bool(ele.get("use_charge_constraint", False)):
        cpf.setChargeConstraintTarget(float(ele.get("charge_constraint_target_e", 0.0)))

    for i in range(system.getNumParticles()):
        q, sig, eps = nb.getParticleParameters(i)
        cpf.addParticle(float(q.value_in_unit(unit.elementary_charge)))
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
    return cpf


def parse_bulk_density_tail(log_path: Path) -> float:
    densities = []
    with log_path.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        cols = [h.replace("#", "").replace('"', "").strip() for h in header]
        try:
            i_den = cols.index("Density (g/mL)")
        except ValueError as exc:
            raise RuntimeError(f"Density column not found in {log_path}") from exc
        for row in reader:
            if not row:
                continue
            try:
                densities.append(float(row[i_den]))
            except (ValueError, IndexError):
                continue
    if not densities:
        raise RuntimeError(f"No density rows found in {log_path}")
    n_tail = max(5, len(densities) // 3)
    return sum(densities[-n_tail:]) / n_tail


def compute_electrolyte_mass_g(topology: app.Topology, electrolyte_indices) -> float:
    atoms = list(topology.atoms())
    mass_amu = 0.0
    for i in electrolyte_indices:
        elem = atoms[i].element
        if elem is not None:
            mass_amu += elem.mass.value_in_unit(unit.dalton)
    return mass_amu * AMU_TO_G


def density_from_gap_angstrom(mass_g: float, area_nm2: float, gap_angstrom: float) -> float:
    gap_nm = gap_angstrom * 0.1
    vol_ml = area_nm2 * gap_nm * NM3_TO_ML
    return mass_g / vol_ml


def gap_from_density_angstrom(mass_g: float, area_nm2: float, density_g_ml: float) -> float:
    gap_nm = mass_g / (density_g_ml * area_nm2 * NM3_TO_ML)
    return gap_nm / 0.1


def main() -> None:
    here = Path(__file__).resolve().parent
    cfg = json.loads((here / "config.json").read_text())
    mc = cfg.get("mc_gap", {})
    if not bool(mc.get("enabled", True)):
        print("mc_gap.enabled is false, skip")
        return

    pdb_in = here / "start_with_electrodes.pdb"
    if not pdb_in.exists():
        raise FileNotFoundError("start_with_electrodes.pdb not found. Run assemble_thick_li_electrode_system.py first.")

    seed = int(mc.get("seed", 20260223))
    rng = random.Random(seed)

    temperature = float(mc.get("temperature_k", cfg["md"]["temperature_k"]))
    pressure_bar = float(mc.get("pressure_bar", cfg["md"]["pressure_bar"]))
    n_trials = int(mc.get("n_trials", 200))
    max_shift_ang = float(mc.get("max_shift_angstrom", 0.5))
    min_gap_ang = float(mc.get("min_gap_angstrom", 40.0))
    max_gap_ang = float(mc.get("max_gap_angstrom", 70.0))
    report_interval = int(mc.get("report_interval", 20))
    target_density = mc.get("target_density_g_ml")
    use_bulk_density_target = bool(mc.get("use_bulk_density_target", True))
    bulk_density_log = (here / str(mc.get("bulk_density_log", "../LiPF6_EC_DMC_minimal/npt.log"))).resolve()
    auto_expand_gap_bounds = bool(mc.get("auto_expand_gap_bounds", True))
    target_gap_buffer = float(mc.get("target_gap_buffer_angstrom", 2.0))
    target_gap_bias_k = float(mc.get("target_gap_bias_k_kj_mol_per_A2", 0.0))
    target_gap_pull = float(mc.get("target_gap_pull", 0.0))
    initialize_to_target_gap = bool(mc.get("initialize_to_target_gap", True))
    final_project_to_target_gap = bool(mc.get("final_project_to_target_gap", True))
    enforce_target_density = bool(mc.get("enforce_target_density", True))
    target_density_tol = float(mc.get("target_density_tolerance_g_ml", 0.05))

    beta = 1.0 / (KB_KJ_MOL_K * temperature)
    p_kj_mol_nm3 = pressure_bar * 0.0602214076

    pdb = app.PDBFile(str(pdb_in))
    if pdb.topology.getPeriodicBoxVectors() is None:
        cell = cfg["cell"]
        a = mm.Vec3(float(cell["box_x_angstrom"]), 0.0, 0.0) * unit.angstrom
        b = mm.Vec3(0.0, float(cell["box_y_angstrom"]), 0.0) * unit.angstrom
        c = mm.Vec3(0.0, 0.0, float(cell["box_z_angstrom"])) * unit.angstrom
        pdb.topology.setPeriodicBoxVectors((a, b, c))

    cath_atoms, ano_atoms, ele_atoms = collect_atoms(pdb.topology)

    ff = app.ForceField(*[str((here / p).resolve()) for p in cfg["forcefield_xml"]])
    residue_templates = build_residue_templates(pdb.topology)
    omm = cfg["openmm"]
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
    make_cpf(system, nb, cath_atoms, ano_atoms, cfg)

    integ = mm.VerletIntegrator(1.0 * unit.femtosecond)
    platform = pick_platform(None)
    print(f"Using OpenMM platform: {platform.getName()}")
    sim = app.Simulation(pdb.topology, system, integ, platform)

    pos = pdb.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    sim.context.setPositions(pos * unit.angstrom)
    e_old = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    box = pdb.topology.getPeriodicBoxVectors()
    ax = box[0][0].value_in_unit(unit.nanometer)
    by = box[1][1].value_in_unit(unit.nanometer)
    area_nm2 = float(ax * by)
    mass_g = compute_electrolyte_mass_g(pdb.topology, ele_atoms)

    if target_density is None and use_bulk_density_target and bulk_density_log.exists():
        target_density = parse_bulk_density_tail(bulk_density_log)
    elif target_density is None and use_bulk_density_target:
        print(f"Bulk density log not found: {bulk_density_log}; proceed without target-density bias.")

    target_gap_ang = None
    if target_density is not None:
        target_density = float(target_density)
        target_gap_ang = gap_from_density_angstrom(mass_g, area_nm2, target_density)
        print(f"Target slab density: {target_density:.4f} g/mL")
        print(f"Target gap from composition+area: {target_gap_ang:.3f} A")
        if auto_expand_gap_bounds:
            min_gap_ang = min(min_gap_ang, target_gap_ang - target_gap_buffer)
            max_gap_ang = max(max_gap_ang, target_gap_ang + target_gap_buffer)
        if target_gap_bias_k <= 0.0:
            target_gap_bias_k = 2.0
        if target_gap_pull <= 0.0:
            target_gap_pull = 0.25
        print(f"MC gap bounds used: [{min_gap_ang:.3f}, {max_gap_ang:.3f}] A")
        print(f"Target-gap steering: pull={target_gap_pull:.3f}, bias_k={target_gap_bias_k:.3f} kJ/mol/A^2")

    zc_old = surface_z_ang(pos, cath_atoms, "cathode")
    za_old = surface_z_ang(pos, ano_atoms, "anode")
    gap_old_ang = za_old - zc_old

    if target_gap_ang is not None and initialize_to_target_gap and abs(gap_old_ang - target_gap_ang) > 1e-6:
        scale = target_gap_ang / gap_old_ang
        dz = target_gap_ang - gap_old_ang
        init_pos = pos.copy()
        init_pos[ano_atoms, 2] += dz
        z_ele = init_pos[ele_atoms, 2]
        init_pos[ele_atoms, 2] = zc_old + (z_ele - zc_old) * scale
        pos = init_pos
        gap_old_ang = target_gap_ang
        za_old = za_old + dz
        sim.context.setPositions(pos * unit.angstrom)
        e_old = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        print(f"Initialized slab gap directly to target: {target_gap_ang:.3f} A")

    accepted = 0
    out_log = here / str(mc.get("output_log", "mc_gap.log"))
    with out_log.open("w") as flog:
        if target_gap_ang is not None:
            flog.write(f"# target_density_g_ml {target_density:.8f}\n")
            flog.write(f"# target_gap_angstrom {target_gap_ang:.8f}\n")
            flog.write(f"# gap_bounds_angstrom {min_gap_ang:.8f} {max_gap_ang:.8f}\n")
            flog.write(f"# initialize_to_target_gap {1 if initialize_to_target_gap else 0}\n")
        flog.write("# trial accept dz_A gap_A E_kJmol density_g_ml\n")

        for it in range(1, n_trials + 1):
            if target_gap_ang is not None and target_gap_pull > 0.0:
                drift = target_gap_pull * (target_gap_ang - gap_old_ang)
                drift = max(-max_shift_ang, min(max_shift_ang, drift))
                dz = drift + rng.uniform(-max_shift_ang, max_shift_ang)
                dz = max(-max_shift_ang, min(max_shift_ang, dz))
            else:
                dz = rng.uniform(-max_shift_ang, max_shift_ang)
            gap_new_ang = gap_old_ang + dz
            if gap_new_ang < min_gap_ang or gap_new_ang > max_gap_ang:
                if it % report_interval == 0:
                    dens = density_from_gap_angstrom(mass_g, area_nm2, gap_old_ang)
                    flog.write(f"{it} 0 {dz:.6f} {gap_old_ang:.6f} {e_old:.6f} {dens:.6f}\n")
                continue

            scale = gap_new_ang / gap_old_ang
            trial = pos.copy()

            # move anode atoms rigidly
            trial[ano_atoms, 2] += dz
            # affine scale electrolyte between cathode plane and anode plane
            z_ele = trial[ele_atoms, 2]
            z_ele_new = zc_old + (z_ele - zc_old) * scale
            trial[ele_atoms, 2] = z_ele_new

            sim.context.setPositions(trial * unit.angstrom)
            e_new = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

            v_old = area_nm2 * (gap_old_ang * 0.1)
            v_new = area_nm2 * (gap_new_ang * 0.1)
            dH = (e_new - e_old) + p_kj_mol_nm3 * (v_new - v_old)
            if target_gap_ang is not None and target_gap_bias_k > 0.0:
                dH += 0.5 * target_gap_bias_k * (
                    (gap_new_ang - target_gap_ang) ** 2 - (gap_old_ang - target_gap_ang) ** 2
                )

            accept = False
            if dH <= 0.0:
                accept = True
            else:
                if rng.random() < math.exp(-beta * dH):
                    accept = True

            if accept:
                pos = trial
                e_old = e_new
                gap_old_ang = gap_new_ang
                za_old = za_old + dz
                accepted += 1
            else:
                sim.context.setPositions(pos * unit.angstrom)

            if it % report_interval == 0:
                dens = density_from_gap_angstrom(mass_g, area_nm2, gap_old_ang)
                flog.write(f"{it} {1 if accept else 0} {dz:.6f} {gap_old_ang:.6f} {e_old:.6f} {dens:.6f}\n")

    dens_final = density_from_gap_angstrom(mass_g, area_nm2, gap_old_ang)
    print(f"MC done: accepted {accepted}/{n_trials} ({accepted/max(1,n_trials):.3f})")
    print(f"Final gap: {gap_old_ang:.3f} A")
    print(f"Final slab density (electrolyte region): {dens_final:.4f} g/mL")
    if target_density is not None:
        diff = abs(dens_final - target_density)
        print(f"Target-density |final-target|: {diff:.4f} g/mL")
        if final_project_to_target_gap and diff > target_density_tol and target_gap_ang is not None:
            scale = target_gap_ang / gap_old_ang
            dz = target_gap_ang - gap_old_ang
            proj_pos = pos.copy()
            proj_pos[ano_atoms, 2] += dz
            z_ele = proj_pos[ele_atoms, 2]
            proj_pos[ele_atoms, 2] = zc_old + (z_ele - zc_old) * scale
            pos = proj_pos
            gap_old_ang = target_gap_ang
            dens_final = density_from_gap_angstrom(mass_g, area_nm2, gap_old_ang)
            diff = abs(dens_final - target_density)
            print(f"Applied final projection to target gap: {target_gap_ang:.3f} A")
            print(f"Projected slab density: {dens_final:.4f} g/mL (|delta|={diff:.4f})")
        if enforce_target_density and diff > target_density_tol:
            raise RuntimeError(
                f"Final slab density {dens_final:.4f} g/mL deviates from target {target_density:.4f} "
                f"by {diff:.4f} > tolerance {target_density_tol:.4f}. "
                "Increase n_trials/max_shift, widen gap bounds, or adjust composition."
            )
    out_pdb = here / str(mc.get("output_pdb", "start_with_electrodes_mc.pdb"))
    with out_pdb.open("w") as f:
        # avoid CRYST1 formatting issues
        pdb.topology.setPeriodicBoxVectors(None)
        app.PDBFile.writeFile(pdb.topology, pos * unit.angstrom, f)
    print(f"Wrote {out_pdb}")
    print(f"Wrote {out_log}")


if __name__ == "__main__":
    main()
