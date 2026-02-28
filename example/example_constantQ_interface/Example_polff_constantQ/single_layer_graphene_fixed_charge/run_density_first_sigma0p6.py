#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import sys
from pathlib import Path

import MDAnalysis as mda
import openmm.app as app
import openmm.unit as unit
from openmm import Vec3

NA = 6.02214076e23


def read_density_tail(log_path: Path, tail_fraction: float) -> tuple[float, float, int]:
    rows = []
    with log_path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            if not r:
                continue
            d = float(r['Density (g/mL)'])
            rows.append(d)
    if not rows:
        raise RuntimeError(f"No density rows in {log_path}")
    n_tail = max(10, int(len(rows) * tail_fraction))
    tail = rows[-n_tail:]
    avg = sum(tail) / len(tail)
    var = sum((x - avg) ** 2 for x in tail) / len(tail)
    return avg, math.sqrt(var), len(tail)


def make_single_layer_graphene_electrodes(
    box_x: float, box_y: float, z_cathode: float, z_anode: float, spacing: float, margin: float
):
    nx = int((box_x - 2.0 * margin) / spacing) + 1
    ny = int((box_y - 2.0 * margin) / spacing) + 1
    top = app.Topology()
    chain_c = top.addChain("A")
    chain_a = top.addChain("B")
    pos = []
    x0 = 0.5 * (box_x - (nx - 1) * spacing)
    y0 = 0.5 * (box_y - (ny - 1) * spacing)
    for ix in range(nx):
        for iy in range(ny):
            x = x0 + ix * spacing
            y = y0 + iy * spacing
            rc = top.addResidue("CAT", chain_c)
            top.addAtom("CG", app.element.carbon, rc)
            pos.append(Vec3(x, y, z_cathode))
    for ix in range(nx):
        for iy in range(ny):
            x = x0 + ix * spacing
            y = y0 + iy * spacing
            ra = top.addResidue("ANO", chain_a)
            top.addAtom("CG", app.element.carbon, ra)
            pos.append(Vec3(x, y, z_anode))
    return top, unit.Quantity(pos, unit.angstrom), nx, ny


def get_residue_masses(u: mda.Universe) -> dict[str, float]:
    mass = {}
    for res in u.residues:
        if res.resname not in mass:
            mass[res.resname] = float(res.atoms.masses.sum())
    return mass


def compute_density_g_ml(counts: dict[str, int], masses_amu: dict[str, float], vol_a3: float) -> float:
    mw_total = sum(masses_amu[k] * v for k, v in counts.items())
    mass_g = mw_total / NA
    vol_cm3 = vol_a3 * 1e-24
    return mass_g / vol_cm3


def main() -> None:
    p = argparse.ArgumentParser(description="Bulk-density-first workflow for sigma=0.6 e/nm^2 fixed-charge run")
    p.add_argument("--bulk-log", default="../../Example_OPLS/LiPF6_EC_DMC_minimal/npt.log")
    p.add_argument("--source-electrolyte", default="../../Example_OPLS/LiPF6_EC_DMC_inert_electrode/electrolyte_start.pdb")
    p.add_argument("--sigma-enm2", type=float, default=0.6)
    p.add_argument("--tail-fraction", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=20260228)
    p.add_argument("--run-dir", default="runs/sigma0p6_densityfirst")
    p.add_argument("--equil-steps", type=int, default=100000)
    p.add_argument("--prod-steps", type=int, default=300000)
    p.add_argument("--report-interval", type=int, default=1000)
    p.add_argument("--platform", choices=["CPU", "Reference", "OpenCL", "CUDA"], default="CUDA")
    args = p.parse_args()

    here = Path(__file__).resolve().parent
    run_dir = (here / args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg_base = json.loads((here / "config.json").read_text())
    cell = cfg_base["cell"]
    ele = cfg_base["electrode"]

    box_x = float(cell["box_x_angstrom"])
    box_y = float(cell["box_y_angstrom"])
    z_cath = float(cell["z_cathode_angstrom"])
    z_anode = float(cell["z_anode_angstrom"])
    z_center = 0.5 * (z_cath + z_anode)
    wall_buffer = 2.0
    max_thickness = (z_anode - z_cath) - 2.0 * wall_buffer
    if max_thickness <= 1.0:
        raise RuntimeError("Electrode gap too small for liquid slab")

    bulk_log = (here / args.bulk_log).resolve()
    src_pdb = (here / args.source_electrolyte).resolve()
    rho_bulk, rho_std, n_tail = read_density_tail(bulk_log, args.tail_fraction)

    u = mda.Universe(str(src_pdb))
    rng = random.Random(args.seed)

    counts_src: dict[str, int] = {}
    for res in u.residues:
        counts_src[res.resname] = counts_src.get(res.resname, 0) + 1
    masses = get_residue_masses(u)
    mw_total_src = sum(masses[k] * v for k, v in counts_src.items())
    vol_needed_a3 = (mw_total_src / NA) / rho_bulk / 1e-24
    thickness_needed = vol_needed_a3 / (box_x * box_y)

    if thickness_needed <= max_thickness:
        z_liq_min = z_center - 0.5 * thickness_needed
        z_liq_max = z_center + 0.5 * thickness_needed
        density_mode = "keep_all_molecules_adjust_thickness"
    else:
        z_liq_min = z_cath + wall_buffer
        z_liq_max = z_anode - wall_buffer
        density_mode = "downsample_to_fit_gap"

    vol_liq_a3 = box_x * box_y * (z_liq_max - z_liq_min)
    rho_src = compute_density_g_ml(counts_src, masses, vol_liq_a3)
    scale = rho_bulk / rho_src

    target_counts = {k: int(round(v * min(1.0, scale))) for k, v in counts_src.items()}
    if "LiA" in target_counts and "PF6" in target_counts:
        n_salt = min(target_counts["LiA"], target_counts["PF6"])
        target_counts["LiA"] = n_salt
        target_counts["PF6"] = n_salt

    by_resname: dict[str, list] = {}
    for res in u.residues:
        by_resname.setdefault(res.resname, []).append(res)

    selected = []
    for rn, residues in by_resname.items():
        n_pick = min(len(residues), max(0, target_counts.get(rn, 0)))
        selected.extend(rng.sample(residues, n_pick))
    if not selected:
        raise RuntimeError("No residues selected for electrolyte slab")

    merged = mda.Merge(*[r.atoms for r in selected])
    coords = merged.atoms.positions.copy()
    coords[:, 0] = coords[:, 0] % box_x
    coords[:, 1] = coords[:, 1] % box_y
    zmin = float(coords[:, 2].min())
    zmax = float(coords[:, 2].max())
    if zmax - zmin < 1e-6:
        raise RuntimeError("Source electrolyte has degenerate z extent")
    coords[:, 2] = z_liq_min + (coords[:, 2] - zmin) * ((z_liq_max - z_liq_min) / (zmax - zmin))
    merged.atoms.positions = coords
    merged.dimensions = [box_x, box_y, float(cell["box_z_angstrom"]), 90.0, 90.0, 90.0]

    electrolyte_out = run_dir / "electrolyte_density_matched.pdb"
    merged.atoms.write(str(electrolyte_out))

    top_e, pos_e, nx, ny = make_single_layer_graphene_electrodes(
        float(cell["box_x_angstrom"]),
        float(cell["box_y_angstrom"]),
        float(cell["z_cathode_angstrom"]),
        float(cell["z_anode_angstrom"]),
        float(ele["spacing_angstrom"]),
        float(ele["edge_margin_angstrom"]),
    )
    modeller = app.Modeller(top_e, pos_e)
    pdb_liq = app.PDBFile(str(electrolyte_out))
    modeller.add(pdb_liq.topology, pdb_liq.positions)
    modeller.topology.setPeriodicBoxVectors(None)

    start_pdb = run_dir / "start_fixedcharge_graphene.pdb"
    with start_pdb.open("w") as f:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, f)

    area_nm2 = float(cell["box_x_angstrom"]) * float(cell["box_y_angstrom"]) / 100.0
    natom_sheet = nx * ny
    q_per_atom = args.sigma_enm2 * area_nm2 / natom_sheet

    cfg_run = dict(cfg_base)
    cfg_run["input"] = {"electrolyte_pdb": str(electrolyte_out)}
    cfg_run["electrode"] = dict(cfg_base["electrode"])
    cfg_run["electrode"]["fixed_charge_per_atom_e"] = q_per_atom
    cfg_run["forcefield_xml"] = [
        str((here / "../../Example_OPLS/opls_salt.xml").resolve()),
        str((here / "../../Example_OPLS/opls_solvent.xml").resolve()),
        str((here / "electrode_residues.xml").resolve()),
        str((here / "electrode_ff.xml").resolve()),
    ]
    cfg_run["md"] = dict(cfg_base["md"])
    cfg_run["md"]["equil_steps"] = int(args.equil_steps)
    cfg_run["md"]["prod_steps"] = int(args.prod_steps)
    cfg_run["md"]["report_interval"] = int(args.report_interval)

    cfg_path = run_dir / "config_densityfirst_sigma0p6.json"
    cfg_path.write_text(json.dumps(cfg_run, indent=2))

    report = run_dir / "density_workflow_report.txt"
    report.write_text(
        "\n".join(
            [
                f"bulk_log: {bulk_log}",
                f"source_electrolyte: {src_pdb}",
                f"bulk_density_avg_g_ml: {rho_bulk:.6f}",
                f"bulk_density_std_g_ml: {rho_std:.6f}",
                f"bulk_tail_points: {n_tail}",
                f"density_mode: {density_mode}",
                f"slab_liquid_z_min_A: {z_liq_min:.6f}",
                f"slab_liquid_z_max_A: {z_liq_max:.6f}",
                f"slab_liquid_thickness_A: {(z_liq_max-z_liq_min):.6f}",
                f"slab_liquid_volume_A3: {vol_liq_a3:.3f}",
                f"source_slab_density_g_ml: {rho_src:.6f}",
                f"density_scale_factor: {scale:.6f}",
                f"selected_counts: {json.dumps(target_counts, sort_keys=True)}",
                f"electrode_grid_nx_ny: {nx} {ny}",
                f"sheet_area_nm2: {area_nm2:.6f}",
                f"target_sigma_e_per_nm2: {args.sigma_enm2:.6f}",
                f"q_per_atom_e: {q_per_atom:.8f}",
                f"run_dir: {run_dir}",
            ]
        )
        + "\n"
    )

    cmd = [
        sys.executable,
        str((here / "run_fixedcharge_singlelayer_graphene.py").resolve()),
        "--config",
        str(cfg_path),
        "--pdb",
        str(start_pdb),
        "--equil-steps",
        str(args.equil_steps),
        "--prod-steps",
        str(args.prod_steps),
        "--report-interval",
        str(args.report_interval),
        "--platform",
        args.platform,
        "--fixed-charge-per-atom-e",
        f"{q_per_atom:.8f}",
        "--outdir",
        str(run_dir),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(here))

    traj = run_dir / "traj_fixedcharge.dcd"
    if traj.exists():
        u2 = mda.Universe(str(start_pdb), str(traj))
        u2.trajectory[-1]
        u2.atoms.write(str(run_dir / "traj_lastframe_sigma0p6.pdb"))

    print(f"Done. Outputs in: {run_dir}")


if __name__ == "__main__":
    main()
