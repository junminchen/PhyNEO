#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import MDAnalysis as mda
import numpy as np


def export_one(run_dir: Path, n_bins: int, stride: int) -> tuple[Path, Path]:
    top = run_dir / "stage3_charged_vacuum_nvt_prod_final.pdb"
    traj = run_dir / "stage3_charged_vacuum_nvt_prod_traj.dcd"
    if not top.exists() or not traj.exists():
        raise FileNotFoundError(f"Missing stage3 files in {run_dir}")

    u = mda.Universe(str(top), str(traj))
    ele_idx = []
    for ch in u.atoms.residues.resindices:  # placeholder to keep linter quiet for mda typed gaps
        _ = ch
        break

    # Electrode chains are first two chains (0/1) in this workflow.
    # Keep electrolyte atoms only for number-density profile.
    electrolyte_sel = u.select_atoms("not (segid A or segid B)")
    if len(electrolyte_sel) == 0:
        electrolyte_sel = u.atoms[0:0]
        ele0 = list(u.segments)
        if len(ele0) >= 2:
            ele_atoms = np.concatenate([ele0[0].atoms.indices, ele0[1].atoms.indices])
            mask = np.ones(u.atoms.n_atoms, dtype=bool)
            mask[ele_atoms] = False
            electrolyte_sel = u.atoms[np.where(mask)[0]]

    # Last-frame PDB from trajectory.
    u.trajectory[-1]
    last_pdb = run_dir / "stage3_lastframe_from_traj.pdb"
    u.atoms.write(str(last_pdb))

    # z-number-density (electrolyte atoms): #/A^3 and #/nm^3
    u.trajectory[0]
    lz0 = float(u.trajectory.ts.dimensions[2])
    z_edges = np.linspace(0.0, lz0, n_bins + 1)
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    dz = lz0 / n_bins

    dens_acc = np.zeros(n_bins, dtype=np.float64)
    n_frames = 0
    for ts in u.trajectory[::max(1, stride)]:
        lx, ly, lz, _, _, _ = ts.dimensions
        area_a2 = float(lx * ly)
        bin_vol_a3 = area_a2 * dz
        z = (electrolyte_sel.positions[:, 2] % lz) * (lz0 / float(lz))
        h, _ = np.histogram(z, bins=z_edges)
        dens_acc += h / bin_vol_a3
        n_frames += 1
    if n_frames == 0:
        raise RuntimeError(f"No frames sampled in {traj}")

    dens_a3 = dens_acc / n_frames
    dens_nm3 = dens_a3 * 1000.0

    z_csv = run_dir / "z_number_density.csv"
    with z_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["z_angstrom", "number_density_per_A3", "number_density_per_nm3"])
        for i in range(n_bins):
            w.writerow([f"{z_centers[i]:.6f}", f"{dens_a3[i]:.10e}", f"{dens_nm3[i]:.10e}"])

    return last_pdb, z_csv


def main() -> None:
    p = argparse.ArgumentParser(description="Export Stage3 last-frame PDB and z-number-density for sigma series.")
    p.add_argument("--run-root", default="runs/stage3_from_neutral_sigma_series_v2")
    p.add_argument("--n-bins", type=int, default=240)
    p.add_argument("--stride", type=int, default=20)
    args = p.parse_args()

    root = Path(args.run_root).resolve()
    sigma_dirs = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("sigma")])
    if not sigma_dirs:
        raise RuntimeError(f"No sigma directories found under {root}")

    manifest = root / "uploaded_manifest.csv"
    with manifest.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_dir", "lastframe_pdb", "z_number_density_csv"])
        for d in sigma_dirs:
            pdb_out, z_out = export_one(d, n_bins=int(args.n_bins), stride=int(args.stride))
            w.writerow([str(d), str(pdb_out), str(z_out)])
            print(f"[done] {d.name}")
    print(f"Wrote {manifest}")


if __name__ == "__main__":
    main()
