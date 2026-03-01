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


def species_density_from_traj(run_dir: Path, n_bins: int, stride: int, species: list[str]) -> tuple[np.ndarray, dict[str, np.ndarray], float]:
    top = run_dir / "stage3_charged_vacuum_nvt_prod_final.pdb"
    traj = run_dir / "stage3_charged_vacuum_nvt_prod_traj.dcd"
    u = mda.Universe(str(top), str(traj))

    if len(u.segments) < 2:
        raise RuntimeError(f"{run_dir}: expected at least 2 segments/chains for electrodes")
    cath_atoms = u.segments[0].atoms
    ano_atoms = u.segments[1].atoms
    ele_idx = np.concatenate([cath_atoms.indices, ano_atoms.indices])
    mask = np.ones(u.atoms.n_atoms, dtype=bool)
    mask[ele_idx] = False
    elec_atoms = u.atoms[np.where(mask)[0]]

    # Use first sampled frame gap as reference x-axis length.
    u.trajectory[0]
    lz_first = float(u.trajectory.ts.dimensions[2])
    z_all_first = u.atoms.positions[:, 2] % lz_first
    zc_first = float(np.mean(z_all_first[cath_atoms.indices]))
    za_first = float(np.mean(z_all_first[ano_atoms.indices]))
    gap_ref = za_first - zc_first
    if gap_ref <= 0.0:
        gap_ref += lz_first
    z_edges = np.linspace(0.0, gap_ref, n_bins + 1)
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    ds = gap_ref / n_bins

    dens = {sp: np.zeros(n_bins, dtype=np.float64) for sp in species}
    n_frames = 0
    for ts in u.trajectory[::max(1, stride)]:
        lx, ly, lz, _, _, _ = ts.dimensions
        area_a2 = float(lx * ly)
        z_all = u.atoms.positions[:, 2] % lz
        zc = float(np.mean(z_all[cath_atoms.indices]))
        za = float(np.mean(z_all[ano_atoms.indices]))
        gap = za - zc
        if gap <= 0.0:
            gap += lz
        bin_vol_a3 = area_a2 * ds

        # Project any z to distance from cathode along cathode->anode direction.
        def project(z_vals: np.ndarray) -> np.ndarray:
            d = z_vals - zc
            d[d < 0.0] += lz
            d = d * (gap_ref / gap)
            return d[(d >= 0.0) & (d <= gap_ref)]

        for sp in species:
            if sp == "CAT":
                z = project(z_all[cath_atoms.indices])
                h, _ = np.histogram(z, bins=z_edges)
                dens[sp] += h / bin_vol_a3 * 1000.0
            elif sp == "ANO":
                z = project(z_all[ano_atoms.indices])
                h, _ = np.histogram(z, bins=z_edges)
                dens[sp] += h / bin_vol_a3 * 1000.0
            else:
                g = elec_atoms.select_atoms(f"resname {sp}")
                if len(g.residues) == 0:
                    continue
                com = g.center_of_geometry(compound="residues")
                z = project(com[:, 2] % lz)
                h, _ = np.histogram(z, bins=z_edges)
                dens[sp] += h / bin_vol_a3 * 1000.0
        n_frames += 1

    if n_frames == 0:
        raise RuntimeError(f"No frames sampled in {traj}")
    for sp in species:
        dens[sp] /= n_frames
    return z_centers, dens, gap_ref


def main() -> None:
    p = argparse.ArgumentParser(description="Plot species z-number-density profiles for sigma-series Stage3 runs.")
    p.add_argument("--run-root", default="runs/stage3_from_neutral_sigma_series_v2")
    p.add_argument("--species", default="LiA,PF6,ECA,DMC,CAT,ANO")
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
        z, dens, gap_ref = species_density_from_traj(d, n_bins=int(args.n_bins), stride=int(args.stride), species=species)

        csv_path = d / "z_density_profile_by_species.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["z_angstrom"] + [f"{sp}_number_density_per_nm3" for sp in species])
            for i in range(len(z)):
                w.writerow([f"{z[i]:.6f}"] + [f"{dens[sp][i]:.10e}" for sp in species])

        fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
        for sp in species:
            ax.plot(z, dens[sp], label=sp, lw=1.8)
        ax.set_xlabel("z from cathode to anode (Angstrom)")
        ax.set_ylabel("Number density (#/nm^3)")
        ax.set_title(f"Species z-density profile in electrolyte gap (sigma={sigma:.3f} e/nm^2)")
        ax.set_xlim(0.0, gap_ref)
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
        z, dens, gap_ref = species_density_from_traj(d, n_bins=int(args.n_bins), stride=int(args.stride), species=species)
        if z_ref is None:
            z_ref = z
        for sp in species:
            all_data[sp].append((sigma, dens[sp]))

    for sp in species:
        fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
        for sigma, y in sorted(all_data[sp], key=lambda t: t[0]):
            ax.plot(z_ref, y, label=f"sigma={sigma:.3f}", lw=1.8)
        ax.set_xlabel("z from cathode to anode (Angstrom)")
        ax.set_ylabel("Number density (#/nm^3)")
        ax.set_title(f"{sp} z-density profile across sigma")
        ax.set_xlim(float(z_ref[0]), float(z_ref[-1]))
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
