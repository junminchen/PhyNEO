#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import struct
from pathlib import Path

import numpy as np
import openmm.app as app


RESIDUE_GROUPS = {
    "Li": {"LiA"},
    "PF6": {"PF6"},
    "EC": {"ECA"},
    "DMC": {"DMC"},
}

RESIDUE_NET_CHARGE_E = {
    "LiA": +1.0,
    "PF6": -1.0,
    "ECA": 0.0,
    "DMC": 0.0,
}


def _read_record(f):
    head = f.read(4)
    if len(head) == 0:
        return None
    if len(head) < 4:
        raise EOFError("Truncated DCD record head")
    n = struct.unpack("<i", head)[0]
    payload = f.read(n)
    if len(payload) != n:
        raise EOFError("Truncated DCD record payload")
    tail = f.read(4)
    if len(tail) < 4:
        raise EOFError("Truncated DCD record tail")
    n2 = struct.unpack("<i", tail)[0]
    if n2 != n:
        raise ValueError("DCD record length mismatch")
    return payload


def dcd_frame_iterator(dcd_path: Path):
    with dcd_path.open("rb") as f:
        # header
        rec = _read_record(f)
        if rec is None:
            return
        if len(rec) < 4 or rec[:4] != b"CORD":
            raise ValueError("Unsupported DCD header (expected CORD)")

        # title block
        rec = _read_record(f)
        if rec is None:
            raise ValueError("Missing DCD title block")

        # natoms block
        rec = _read_record(f)
        if rec is None or len(rec) != 4:
            raise ValueError("Missing DCD natoms block")
        natom = struct.unpack("<i", rec)[0]

        while True:
            rec = _read_record(f)
            if rec is None:
                break

            # optional unit cell block (48 bytes)
            if len(rec) == 48:
                rec = _read_record(f)
                if rec is None:
                    break

            # x, y, z coordinate blocks as float32 arrays
            if len(rec) != 4 * natom:
                raise ValueError("Unexpected DCD x block length")
            x = np.frombuffer(rec, dtype=np.float32, count=natom)

            recy = _read_record(f)
            recz = _read_record(f)
            if recy is None or recz is None:
                raise EOFError("Truncated DCD coordinate blocks")
            if len(recy) != 4 * natom or len(recz) != 4 * natom:
                raise ValueError("Unexpected DCD y/z block length")
            y = np.frombuffer(recy, dtype=np.float32, count=natom)
            z = np.frombuffer(recz, dtype=np.float32, count=natom)

            yield x, y, z


def get_box_from_config(cfg_path: Path):
    cfg = json.loads(cfg_path.read_text())
    cell = cfg["cell"]
    return float(cell["box_x_angstrom"]), float(cell["box_y_angstrom"]), float(cell["box_z_angstrom"])


def histogram_one(zcom_A: float, z_edges_A: np.ndarray) -> np.ndarray:
    h, _ = np.histogram([zcom_A], bins=z_edges_A)
    return h


def integrate_window(z_centers_A: np.ndarray, rho: np.ndarray, z_lo_A: float, z_hi_A: float, dz_nm: float) -> float:
    mask = (z_centers_A >= z_lo_A) & (z_centers_A < z_hi_A)
    if not np.any(mask):
        return 0.0
    # rho unit: 1/nm^3 ; integral over z gives 1/nm^2
    return float(np.sum(rho[mask]) * dz_nm)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze interfacial z-distribution for Li/PF6/EC/DMC (no external traj deps).")
    parser.add_argument("--top", default="start_with_electrodes_mc.pdb", help="Topology PDB")
    parser.add_argument("--traj", default="traj_thick_li.dcd", help="Trajectory DCD")
    parser.add_argument("--config", default="config.json", help="Config for box size")
    parser.add_argument("--nbins", type=int, default=120)
    parser.add_argument("--zmin", type=float, default=None, help="A")
    parser.add_argument("--zmax", type=float, default=None, help="A")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--nblocks", type=int, default=5, help="Block count for uncertainty estimate")
    parser.add_argument("--electrode_margin", type=float, default=1.0, help="A; trim around electrode planes")
    parser.add_argument("--interface_width", type=float, default=5.0, help="A; adsorption integration width near each electrode")
    parser.add_argument("--between-electrodes-only", action="store_true", help="Restrict profile to (z_cath+margin, z_anode-margin)")
    parser.add_argument("--csv", default="results/interface_distribution.csv")
    parser.add_argument("--png", default="results/interface_distribution.png")
    parser.add_argument("--summary", default="results/interface_distribution_summary.txt")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    top_path = (here / args.top).resolve()
    traj_path = (here / args.traj).resolve()
    cfg_path = (here / args.config).resolve()

    pdb = app.PDBFile(str(top_path))
    top = pdb.topology
    atoms = list(top.atoms())
    natom = len(atoms)

    masses = np.array([
        (a.element.mass._value if a.element is not None else 0.0) for a in atoms
    ], dtype=float)

    group_residue_atom_indices: dict[str, list[list[int]]] = {k: [] for k in RESIDUE_GROUPS}
    charge_residue_atom_indices: list[tuple[list[int], float]] = []
    cath_atoms, ano_atoms = [], []

    for ch in top.chains():
        if ch.index == 0:
            cath_atoms.extend([a.index for a in ch.atoms()])
        elif ch.index == 1:
            ano_atoms.extend([a.index for a in ch.atoms()])

    for res in top.residues():
        rname = res.name.strip()
        atom_idx = [a.index for a in res.atoms()]
        for g, rset in RESIDUE_GROUPS.items():
            if rname in rset:
                group_residue_atom_indices[g].append(atom_idx)
        q = RESIDUE_NET_CHARGE_E.get(rname, 0.0)
        if abs(q) > 0.0:
            charge_residue_atom_indices.append((atom_idx, q))

    box_x_A, box_y_A, box_z_A = get_box_from_config(cfg_path)
    area_nm2 = (box_x_A * 0.1) * (box_y_A * 0.1)

    zmin_A = 0.0 if args.zmin is None else float(args.zmin)
    zmax_A = box_z_A if args.zmax is None else float(args.zmax)
    z_edges_A = np.linspace(zmin_A, zmax_A, args.nbins + 1)
    dz_nm = (z_edges_A[1] - z_edges_A[0]) * 0.1
    z_centers_A = 0.5 * (z_edges_A[:-1] + z_edges_A[1:])

    frame_hist_counts = {g: [] for g in RESIDUE_GROUPS}
    frame_hist_charge = []

    z_cath_A = np.nan
    z_ano_A = np.nan

    nframe = 0
    used = 0
    for iframe, (_, _, z) in enumerate(dcd_frame_iterator(traj_path), start=1):
        if args.stride > 1 and ((iframe - 1) % args.stride != 0):
            continue
        if z.shape[0] != natom:
            raise ValueError(f"DCD natom mismatch: frame {iframe} has {z.shape[0]}, topology has {natom}")

        # For multilayer electrodes, use physical surface planes (not slab mean z).
        if not np.isfinite(z_cath_A) and cath_atoms:
            z_cath_A = float(np.max(z[cath_atoms]))
        if not np.isfinite(z_ano_A) and ano_atoms:
            z_ano_A = float(np.min(z[ano_atoms]))

        frame_counts = {g: np.zeros(args.nbins, dtype=float) for g in RESIDUE_GROUPS}
        frame_charge = np.zeros(args.nbins, dtype=float)

        for g, res_list in group_residue_atom_indices.items():
            for atom_idx in res_list:
                w = masses[atom_idx]
                zcom = float(np.dot(z[atom_idx], w) / np.sum(w))
                frame_counts[g] += histogram_one(zcom, z_edges_A)

        for atom_idx, q in charge_residue_atom_indices:
            w = masses[atom_idx]
            zcom = float(np.dot(z[atom_idx], w) / np.sum(w))
            frame_charge += q * histogram_one(zcom, z_edges_A)

        for g in RESIDUE_GROUPS:
            frame_hist_counts[g].append(frame_counts[g])
        frame_hist_charge.append(frame_charge)

        used += 1
        nframe = iframe

    if used == 0:
        raise RuntimeError("No frames analyzed")

    frames_per_block = max(1, used // max(1, args.nblocks))
    eff_blocks = max(1, used // frames_per_block)

    def to_density(arr: np.ndarray, nfrm: int) -> np.ndarray:
        return arr / (max(1, nfrm) * area_nm2 * dz_nm)

    all_counts = {g: np.stack(frame_hist_counts[g], axis=0) for g in RESIDUE_GROUPS}
    all_charge = np.stack(frame_hist_charge, axis=0)

    rho = {g: to_density(np.sum(all_counts[g], axis=0), used) for g in RESIDUE_GROUPS}
    rho_q = to_density(np.sum(all_charge, axis=0), used)

    rho_std = {g: np.zeros(args.nbins, dtype=float) for g in RESIDUE_GROUPS}
    rho_q_std = np.zeros(args.nbins, dtype=float)
    if eff_blocks >= 2:
        block_rho = {g: [] for g in RESIDUE_GROUPS}
        block_rho_q = []
        for b in range(eff_blocks):
            lo = b * frames_per_block
            hi = min((b + 1) * frames_per_block, used)
            if hi <= lo:
                continue
            nfrm_b = hi - lo
            for g in RESIDUE_GROUPS:
                block_rho[g].append(to_density(np.sum(all_counts[g][lo:hi], axis=0), nfrm_b))
            block_rho_q.append(to_density(np.sum(all_charge[lo:hi], axis=0), nfrm_b))
        if len(block_rho_q) >= 2:
            for g in RESIDUE_GROUPS:
                rho_std[g] = np.std(np.stack(block_rho[g], axis=0), axis=0, ddof=1)
            rho_q_std = np.std(np.stack(block_rho_q, axis=0), axis=0, ddof=1)

    if args.between_electrodes_only and np.isfinite(z_cath_A) and np.isfinite(z_ano_A):
        zmin_eff_A = z_cath_A + args.electrode_margin
        zmax_eff_A = z_ano_A - args.electrode_margin
    else:
        zmin_eff_A = zmin_A
        zmax_eff_A = zmax_A

    csv_path = (here / args.csv).resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "z_angstrom",
            "rho_Li_1_per_nm3",
            "rho_PF6_1_per_nm3",
            "rho_EC_1_per_nm3",
            "rho_DMC_1_per_nm3",
            "rho_Li_std_1_per_nm3",
            "rho_PF6_std_1_per_nm3",
            "rho_EC_std_1_per_nm3",
            "rho_DMC_std_1_per_nm3",
            "rho_charge_e_per_nm3",
            "rho_charge_std_e_per_nm3",
            "z_cathode_angstrom",
            "z_anode_angstrom",
        ])
        for i in range(args.nbins):
            zc = float(z_centers_A[i])
            if (zc < zmin_eff_A) or (zc >= zmax_eff_A):
                continue
            w.writerow([
                zc,
                float(rho["Li"][i]),
                float(rho["PF6"][i]),
                float(rho["EC"][i]),
                float(rho["DMC"][i]),
                float(rho_std["Li"][i]),
                float(rho_std["PF6"][i]),
                float(rho_std["EC"][i]),
                float(rho_std["DMC"][i]),
                float(rho_q[i]),
                float(rho_q_std[i]),
                float(z_cath_A),
                float(z_ano_A),
            ])

    import matplotlib.pyplot as plt

    png_path = (here / args.png).resolve()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.5, 5.4), dpi=160)
    mask_plot = (z_centers_A >= zmin_eff_A) & (z_centers_A < zmax_eff_A)
    z_plot = z_centers_A[mask_plot]
    li = rho["Li"][mask_plot]
    pf6 = rho["PF6"][mask_plot]
    ec = rho["EC"][mask_plot]
    dmc = rho["DMC"][mask_plot]
    li_std = rho_std["Li"][mask_plot]
    pf6_std = rho_std["PF6"][mask_plot]
    ec_std = rho_std["EC"][mask_plot]
    dmc_std = rho_std["DMC"][mask_plot]

    plt.plot(z_plot, li, label="Li", lw=1.8)
    plt.plot(z_plot, pf6, label="PF6", lw=1.8)
    plt.plot(z_plot, ec, label="EC", lw=1.6)
    plt.plot(z_plot, dmc, label="DMC", lw=1.6)
    if eff_blocks >= 2:
        plt.fill_between(z_plot, li - li_std, li + li_std, alpha=0.15)
        plt.fill_between(z_plot, pf6 - pf6_std, pf6 + pf6_std, alpha=0.15)
        plt.fill_between(z_plot, ec - ec_std, ec + ec_std, alpha=0.12)
        plt.fill_between(z_plot, dmc - dmc_std, dmc + dmc_std, alpha=0.12)
    if np.isfinite(z_cath_A):
        plt.axvline(z_cath_A, color="k", ls="--", lw=1.0, alpha=0.8, label="Cathode")
    if np.isfinite(z_ano_A):
        plt.axvline(z_ano_A, color="k", ls=":", lw=1.0, alpha=0.8, label="Anode")
    plt.xlabel("z (A)")
    plt.ylabel("Number Density (1/nm^3)")
    plt.title("Interfacial Distribution: LiPF6-EC-DMC")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()

    qpng_path = png_path.with_name(png_path.stem + "_charge.png")
    plt.figure(figsize=(7.5, 4.6), dpi=160)
    q_plot = rho_q[mask_plot]
    q_std = rho_q_std[mask_plot]
    plt.plot(z_plot, q_plot, label="Charge density", lw=1.8)
    if eff_blocks >= 2:
        plt.fill_between(z_plot, q_plot - q_std, q_plot + q_std, alpha=0.2)
    if np.isfinite(z_cath_A):
        plt.axvline(z_cath_A, color="k", ls="--", lw=1.0, alpha=0.8)
    if np.isfinite(z_ano_A):
        plt.axvline(z_ano_A, color="k", ls=":", lw=1.0, alpha=0.8)
    plt.xlabel("z (A)")
    plt.ylabel("Charge Density (e/nm^3)")
    plt.title("Interfacial Charge Density")
    plt.tight_layout()
    plt.savefig(qpng_path)
    plt.close()

    summary_path = (here / args.summary).resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as f:
        f.write("Interfacial distribution summary\n")
        f.write(f"frames_used: {used}\n")
        f.write(f"blocks_used: {eff_blocks}\n")
        f.write(f"z_cathode_A: {z_cath_A:.6f}\n")
        f.write(f"z_anode_A: {z_ano_A:.6f}\n")
        f.write(f"analysis_zmin_A: {zmin_eff_A:.6f}\n")
        f.write(f"analysis_zmax_A: {zmax_eff_A:.6f}\n")
        if np.isfinite(z_cath_A) and np.isfinite(z_ano_A):
            w = float(args.interface_width)
            cath_lo = z_cath_A + args.electrode_margin
            cath_hi = cath_lo + w
            ano_hi = z_ano_A - args.electrode_margin
            ano_lo = ano_hi - w
            f.write(f"interface_width_A: {w:.6f}\n")
            f.write(f"cathode_window_A: [{cath_lo:.6f}, {cath_hi:.6f})\n")
            f.write(f"anode_window_A: [{ano_lo:.6f}, {ano_hi:.6f})\n")
            for g in RESIDUE_GROUPS:
                cath_ads = integrate_window(z_centers_A, rho[g], cath_lo, cath_hi, dz_nm)
                ano_ads = integrate_window(z_centers_A, rho[g], ano_lo, ano_hi, dz_nm)
                f.write(f"adsorption_{g}_cathode_1_per_nm2: {cath_ads:.8f}\n")
                f.write(f"adsorption_{g}_anode_1_per_nm2: {ano_ads:.8f}\n")
            cath_q = integrate_window(z_centers_A, rho_q, cath_lo, cath_hi, dz_nm)
            ano_q = integrate_window(z_centers_A, rho_q, ano_lo, ano_hi, dz_nm)
            f.write(f"charge_adsorption_cathode_e_per_nm2: {cath_q:.8f}\n")
            f.write(f"charge_adsorption_anode_e_per_nm2: {ano_q:.8f}\n")

    print(f"Frames read: {nframe}, frames used: {used}")
    print(f"Cathode z: {z_cath_A:.3f} A, Anode z: {z_ano_A:.3f} A")
    print(f"Analysis range: [{zmin_eff_A:.3f}, {zmax_eff_A:.3f}) A")
    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")
    print(f"Wrote {qpng_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
