#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import struct
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openmm.app as app


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
        rec = _read_record(f)
        if rec is None:
            return
        if len(rec) < 4 or rec[:4] != b"CORD":
            raise ValueError("Unsupported DCD header")
        _ = _read_record(f)  # title
        natom_rec = _read_record(f)
        if natom_rec is None or len(natom_rec) != 4:
            raise ValueError("Missing natom record in DCD")
        natom = struct.unpack("<i", natom_rec)[0]
        while True:
            rec = _read_record(f)
            if rec is None:
                break
            if len(rec) == 48:
                rec = _read_record(f)
                if rec is None:
                    break
            if len(rec) != 4 * natom:
                raise ValueError("Unexpected x record length")
            x = np.frombuffer(rec, dtype=np.float32, count=natom)
            yrec = _read_record(f)
            zrec = _read_record(f)
            if yrec is None or zrec is None:
                raise EOFError("Truncated DCD coordinate records")
            y = np.frombuffer(yrec, dtype=np.float32, count=natom)
            z = np.frombuffer(zrec, dtype=np.float32, count=natom)
            yield x, y, z


def load_box(cfg_path: Path):
    cfg = json.loads(cfg_path.read_text())
    cell = cfg["cell"]
    return float(cell["box_x_angstrom"]), float(cell["box_y_angstrom"]), float(cell["box_z_angstrom"])


def main():
    parser = argparse.ArgumentParser(description="Visualize last-frame interfacial ionic charge distribution.")
    parser.add_argument("--top", default="start_with_electrodes_mc.pdb")
    parser.add_argument("--traj", default="traj_inert.dcd")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--interface-width", type=float, default=5.0, help="A")
    parser.add_argument("--electrode-margin", type=float, default=1.0, help="A")
    parser.add_argument("--z-bins", type=int, default=120)
    parser.add_argument("--xy-bins", type=int, default=40)
    parser.add_argument("--png", default="results/last_frame_interface_charge.png")
    parser.add_argument("--csv", default="results/last_frame_interface_charge_profile.csv")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    top_path = (here / args.top).resolve()
    traj_path = (here / args.traj).resolve()
    cfg_path = (here / args.config).resolve()
    out_png = (here / args.png).resolve()
    out_csv = (here / args.csv).resolve()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    box_x_A, box_y_A, box_z_A = load_box(cfg_path)
    box_x_nm = box_x_A * 0.1
    box_y_nm = box_y_A * 0.1

    pdb = app.PDBFile(str(top_path))
    top = pdb.topology
    atoms = list(top.atoms())
    natom = len(atoms)
    masses = np.array([(a.element.mass._value if a.element is not None else 0.0) for a in atoms], dtype=float)

    cath_atoms = []
    anode_atoms = []
    residue_items = []
    for ch in top.chains():
        if ch.index == 0:
            cath_atoms.extend([a.index for a in ch.atoms()])
        elif ch.index == 1:
            anode_atoms.extend([a.index for a in ch.atoms()])
    for res in top.residues():
        q = RESIDUE_NET_CHARGE_E.get(res.name.strip(), 0.0)
        if abs(q) < 1e-12:
            continue
        idx = [a.index for a in res.atoms()]
        residue_items.append((idx, q, res.name.strip()))

    last_frame = None
    nframe = 0
    for fr in dcd_frame_iterator(traj_path):
        last_frame = fr
        nframe += 1
    if last_frame is None:
        raise RuntimeError("No frame found in trajectory")
    x, y, z = last_frame
    if x.shape[0] != natom:
        raise ValueError("Topology/trajectory atom count mismatch")

    z_cath_A = float(np.mean(z[cath_atoms])) if cath_atoms else np.nan
    z_anode_A = float(np.mean(z[anode_atoms])) if anode_atoms else np.nan

    z_edges_A = np.linspace(0.0, box_z_A, args.z_bins + 1)
    z_centers_A = 0.5 * (z_edges_A[:-1] + z_edges_A[1:])
    dz_nm = (z_edges_A[1] - z_edges_A[0]) * 0.1
    area_nm2 = box_x_nm * box_y_nm
    rho_z = np.zeros(args.z_bins, dtype=float)

    cath_lo = z_cath_A + args.electrode_margin
    cath_hi = cath_lo + args.interface_width
    anode_hi = z_anode_A - args.electrode_margin
    anode_lo = anode_hi - args.interface_width

    cath_xy = np.zeros((args.xy_bins, args.xy_bins), dtype=float)
    anode_xy = np.zeros((args.xy_bins, args.xy_bins), dtype=float)

    for atom_idx, q, _ in residue_items:
        w = masses[atom_idx]
        wsum = float(np.sum(w))
        if wsum <= 0.0:
            continue
        xcom = float(np.dot(x[atom_idx], w) / wsum) % box_x_A
        ycom = float(np.dot(y[atom_idx], w) / wsum) % box_y_A
        zcom = float(np.dot(z[atom_idx], w) / wsum)

        hz, _ = np.histogram([zcom], bins=z_edges_A)
        rho_z += q * hz

        ix = int(np.floor((xcom / box_x_A) * args.xy_bins))
        iy = int(np.floor((ycom / box_y_A) * args.xy_bins))
        if ix >= args.xy_bins:
            ix = args.xy_bins - 1
        if iy >= args.xy_bins:
            iy = args.xy_bins - 1

        if cath_lo <= zcom < cath_hi:
            cath_xy[iy, ix] += q
        if anode_lo <= zcom < anode_hi:
            anode_xy[iy, ix] += q

    rho_z /= (area_nm2 * dz_nm)  # e / nm^3

    dx_nm = box_x_nm / args.xy_bins
    dy_nm = box_y_nm / args.xy_bins
    bin_area_nm2 = dx_nm * dy_nm
    cath_sigma = cath_xy / bin_area_nm2  # e / nm^2
    anode_sigma = anode_xy / bin_area_nm2

    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["z_angstrom", "rho_charge_e_per_nm3", "z_cathode_angstrom", "z_anode_angstrom"])
        for i in range(args.z_bins):
            w.writerow([float(z_centers_A[i]), float(rho_z[i]), z_cath_A, z_anode_A])

    vlim = max(np.max(np.abs(cath_sigma)), np.max(np.abs(anode_sigma)), 1e-12)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), dpi=180)
    ax0, ax1, ax2 = axes

    ax0.plot(z_centers_A, rho_z, lw=1.8)
    if np.isfinite(z_cath_A):
        ax0.axvline(z_cath_A, color="k", ls="--", lw=1.0, alpha=0.8)
    if np.isfinite(z_anode_A):
        ax0.axvline(z_anode_A, color="k", ls=":", lw=1.0, alpha=0.8)
    ax0.set_xlabel("z (A)")
    ax0.set_ylabel("rho_q (e/nm^3)")
    ax0.set_title("Last-frame charge profile")

    im1 = ax1.imshow(
        cath_sigma,
        origin="lower",
        cmap="coolwarm",
        vmin=-vlim,
        vmax=vlim,
        extent=[0.0, box_x_A, 0.0, box_y_A],
        aspect="equal",
    )
    ax1.set_title(f"Cathode-side map ({cath_lo:.1f}-{cath_hi:.1f} A)")
    ax1.set_xlabel("x (A)")
    ax1.set_ylabel("y (A)")

    im2 = ax2.imshow(
        anode_sigma,
        origin="lower",
        cmap="coolwarm",
        vmin=-vlim,
        vmax=vlim,
        extent=[0.0, box_x_A, 0.0, box_y_A],
        aspect="equal",
    )
    ax2.set_title(f"Anode-side map ({anode_lo:.1f}-{anode_hi:.1f} A)")
    ax2.set_xlabel("x (A)")
    ax2.set_ylabel("y (A)")

    cbar = fig.colorbar(im2, ax=[ax1, ax2], 
                        fraction=0.046, 
                        pad=0.04,      
                        shrink=0.8,   
                        aspect=25)     
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

    print(f"Frames read: {nframe} (used last frame)")
    print(f"Cathode z: {z_cath_A:.3f} A, Anode z: {z_anode_A:.3f} A")
    print(f"Wrote {out_png}")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
