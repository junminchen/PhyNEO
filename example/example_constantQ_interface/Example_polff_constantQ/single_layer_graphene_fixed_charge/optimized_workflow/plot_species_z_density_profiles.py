#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import MDAnalysis as mda
import matplotlib.pyplot as plt
import numpy as np


def parse_sigma_from_dirname(name: str) -> float:
    # e.g. sigma0p600_q0p0375 -> 0.600
    key = name.split("_")[0].replace("sigma", "").replace("p", ".")
    try:
        return float(key)
    except Exception:
        return float("nan")


def species_density_from_traj(run_dir: Path, n_bins: int, stride: int, species: list[str]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    top = run_dir / "stage3_charged_vacuum_nvt_prod_final.pdb"
    traj = run_dir / "stage3_charged_vacuum_nvt_prod_traj.dcd"
    u = mda.Universe(str(top), str(traj))

    u.trajectory[0]
    lz0 = float(u.trajectory.ts.dimensions[2])
    z_edges = np.linspace(0.0, lz0, n_bins + 1)
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    dz = lz0 / n_bins

    dens = {sp: np.zeros(n_bins, dtype=np.float64) for sp in species}
    n_frames = 0
    for ts in u.trajectory[::max(1, stride)]:
        lx, ly, lz, _, _, _ = ts.dimensions
        area_a2 = float(lx * ly)
        bin_vol_a3 = area_a2 * dz
        scale = lz0 / float(lz)
        for sp in species:
            g = u.select_atoms(f"resname {sp}")
            if len(g.residues) == 0:
                continue
            com = g.center_of_geometry(compound="residues")
            z = (com[:, 2] % lz) * scale
            h, _ = np.histogram(z, bins=z_edges)
            dens[sp] += h / bin_vol_a3 * 1000.0  # #/nm^3
        n_frames += 1

    if n_frames == 0:
        raise RuntimeError(f"No frames sampled in {traj}")
    for sp in species:
        dens[sp] /= n_frames
    return z_centers, dens


def main() -> None:
    p = argparse.ArgumentParser(description="Plot species z-number-density profiles for sigma-series Stage3 runs.")
    p.add_argument("--run-root", default="runs/stage3_from_neutral_sigma_series_v2")
    p.add_argument("--species", default="LiA,PF6,ECA,DMC")
    p.add_argument("--n-bins", type=int, default=240)
    p.add_argument("--stride", type=int, default=20)
    args = p.parse_args()

    root = Path(args.run_root).resolve()
    run_dirs = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("sigma")], key=lambda x: parse_sigma_from_dirname(x.name))
    species = [x.strip() for x in args.species.split(",") if x.strip()]
    if not run_dirs:
        raise RuntimeError(f"No sigma directories found in {root}")

    summary_rows = []
    # per-run plots and csv
    for d in run_dirs:
        sigma = parse_sigma_from_dirname(d.name)
        z, dens = species_density_from_traj(d, n_bins=int(args.n_bins), stride=int(args.stride), species=species)

        csv_path = d / "z_density_profile_by_species.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["z_angstrom"] + [f"{sp}_number_density_per_nm3" for sp in species])
            for i in range(len(z)):
                w.writerow([f"{z[i]:.6f}"] + [f"{dens[sp][i]:.10e}" for sp in species])

        fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
        for sp in species:
            ax.plot(z, dens[sp], label=sp, lw=1.8)
        ax.set_xlabel("z (Angstrom)")
        ax.set_ylabel("Number density (#/nm^3)")
        ax.set_title(f"Species z-density profile (sigma={sigma:.3f} e/nm^2)")
        ax.legend(frameon=False, ncol=2)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        png_path = d / "z_density_profile_by_species.png"
        fig.savefig(png_path)
        plt.close(fig)

        summary_rows.append((d.name, sigma, str(csv_path), str(png_path)))
        print(f"[done] {d.name}")

    # cross-sigma comparison per species
    z_ref = None
    all_data: dict[str, list[tuple[float, np.ndarray]]] = {sp: [] for sp in species}
    for d in run_dirs:
        sigma = parse_sigma_from_dirname(d.name)
        z, dens = species_density_from_traj(d, n_bins=int(args.n_bins), stride=int(args.stride), species=species)
        if z_ref is None:
            z_ref = z
        for sp in species:
            all_data[sp].append((sigma, dens[sp]))

    for sp in species:
        fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
        for sigma, y in sorted(all_data[sp], key=lambda t: t[0]):
            ax.plot(z_ref, y, label=f"sigma={sigma:.3f}", lw=1.8)
        ax.set_xlabel("z (Angstrom)")
        ax.set_ylabel("Number density (#/nm^3)")
        ax.set_title(f"{sp} z-density profile across sigma")
        ax.legend(frameon=False, ncol=2)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(root / f"z_density_compare_{sp}.png")
        plt.close(fig)

    manifest = root / "z_density_profile_manifest.csv"
    with manifest.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_dir", "sigma_e_per_nm2", "csv", "png"])
        w.writerows(summary_rows)
    print(f"Wrote {manifest}")


if __name__ == "__main__":
    main()
