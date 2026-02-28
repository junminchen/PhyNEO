#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import struct
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
from matplotlib.gridspec import GridSpec

KJMOL_PER_E_PER_VOLT = 96.48533212331002


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


def dcd_last_frame(dcd_path: Path):
    last = None
    nframe = 0
    with dcd_path.open("rb") as f:
        rec = _read_record(f)
        if rec is None or len(rec) < 4 or rec[:4] != b"CORD":
            raise ValueError("Unsupported DCD")
        _ = _read_record(f)  # title
        natom_rec = _read_record(f)
        if natom_rec is None or len(natom_rec) != 4:
            raise ValueError("Missing natom record")
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
            x = np.frombuffer(rec, dtype=np.float32, count=natom).copy()
            y = np.frombuffer(_read_record(f), dtype=np.float32, count=natom).copy()
            z = np.frombuffer(_read_record(f), dtype=np.float32, count=natom).copy()
            last = (x, y, z)
            nframe += 1
    if last is None:
        raise RuntimeError("No frame in DCD")
    return nframe, last


def get_nonbonded(system: mm.System) -> mm.NonbondedForce:
    for i in range(system.getNumForces()):
        f = system.getForce(i)
        if isinstance(f, mm.NonbondedForce):
            return f
    raise RuntimeError("NonbondedForce not found")


def collect_electrode_atoms(topology: app.Topology):
    cath = []
    ano = []
    for ch in topology.chains():
        if ch.index == 0:
            cath.extend([a.index for a in ch.atoms() if (a.element is not None and a.element.symbol != "H")])
        elif ch.index == 1:
            ano.extend([a.index for a in ch.atoms() if (a.element is not None and a.element.symbol != "H")])
    if not cath or not ano:
        raise RuntimeError("Electrode atoms not found on chains 0/1")
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


def build_residue_templates(topology: app.Topology):
    residue_templates = {}
    known = {"CAT", "ANO", "LiA", "PF6", "ECA", "DMC"}
    for res in topology.residues():
        if res.name in known:
            residue_templates[res] = res.name
    return residue_templates


def main():
    parser = argparse.ArgumentParser(description="Visualize last-frame cathode/anode atomic charge maps from ConstantPotentialForce.")
    parser.add_argument("--top", default="start_with_electrodes_mc.pdb")
    parser.add_argument("--traj", default="traj_thick_li.dcd")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--voltage-v", type=float, default=None, help="Override voltage in config")
    parser.add_argument("--platform", choices=["CPU", "Reference", "OpenCL", "CUDA"], default=None)
    parser.add_argument("--xy-bins", type=int, default=40)
    parser.add_argument("--vlim", type=float, default=None, help="Fixed color scale limit in e/nm^2 (symmetric +/-vlim)")
    parser.add_argument("--png", default="results/last_frame_electrode_charge_maps.png")
    parser.add_argument("--csv", default="results/last_frame_electrode_atom_charges.csv")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    cfg = json.loads((here / args.config).read_text())
    omm = cfg["openmm"]
    ele = cfg["electrode"]
    cell = cfg["cell"]

    top_path = (here / args.top).resolve()
    traj_path = (here / args.traj).resolve()
    out_png = (here / args.png).resolve()
    out_csv = (here / args.csv).resolve()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    pdb = app.PDBFile(str(top_path))
    if pdb.topology.getPeriodicBoxVectors() is None:
        a = mm.Vec3(float(cell["box_x_angstrom"]), 0.0, 0.0) * unit.angstrom
        b = mm.Vec3(0.0, float(cell["box_y_angstrom"]), 0.0) * unit.angstrom
        c = mm.Vec3(0.0, 0.0, float(cell["box_z_angstrom"])) * unit.angstrom
        pdb.topology.setPeriodicBoxVectors((a, b, c))

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

    v = float(ele["voltage_v"] if args.voltage_v is None else args.voltage_v)
    cpf.addElectrode(set(cath_atoms), v * KJMOL_PER_E_PER_VOLT, float(ele["gaussian_width_nm"]), float(ele["thomas_fermi_scale_invnm"]))
    cpf.addElectrode(set(ano_atoms), -v * KJMOL_PER_E_PER_VOLT, float(ele["gaussian_width_nm"]), float(ele["thomas_fermi_scale_invnm"]))
    system.addForce(cpf)

    integrator = mm.VerletIntegrator(1.0 * unit.femtosecond)
    platform = pick_platform(args.platform)
    print(f"Using OpenMM platform: {platform.getName()}")
    sim = app.Simulation(pdb.topology, system, integrator, platform)

    nframe, (x, y, z) = dcd_last_frame(traj_path)
    pos = [mm.Vec3(float(xi), float(yi), float(zi)) for xi, yi, zi in zip(x, y, z)]
    sim.context.setPositions(pos * unit.angstrom)

    charges = list(cpf.getCharges(sim.context))
    q_arr = np.array([float(q.value_in_unit(unit.elementary_charge)) for q in charges], dtype=float)

    box_x_A = float(cell["box_x_angstrom"])
    box_y_A = float(cell["box_y_angstrom"])
    box_x_nm = box_x_A * 0.1
    box_y_nm = box_y_A * 0.1
    dx_nm = box_x_nm / args.xy_bins
    dy_nm = box_y_nm / args.xy_bins
    area_bin_nm2 = dx_nm * dy_nm

    cath_map = np.zeros((args.xy_bins, args.xy_bins), dtype=float)
    ano_map = np.zeros((args.xy_bins, args.xy_bins), dtype=float)

    for idx in cath_atoms:
        xx = float(x[idx]) % box_x_A
        yy = float(y[idx]) % box_y_A
        ix = min(args.xy_bins - 1, int((xx / box_x_A) * args.xy_bins))
        iy = min(args.xy_bins - 1, int((yy / box_y_A) * args.xy_bins))
        cath_map[iy, ix] += q_arr[idx]
    for idx in ano_atoms:
        xx = float(x[idx]) % box_x_A
        yy = float(y[idx]) % box_y_A
        ix = min(args.xy_bins - 1, int((xx / box_x_A) * args.xy_bins))
        iy = min(args.xy_bins - 1, int((yy / box_y_A) * args.xy_bins))
        ano_map[iy, ix] += q_arr[idx]

    cath_sigma = cath_map / area_bin_nm2  # e/nm^2
    ano_sigma = ano_map / area_bin_nm2

    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["atom_index", "electrode", "x_angstrom", "y_angstrom", "z_angstrom", "q_e"])
        for idx in cath_atoms:
            w.writerow([idx, "cathode", float(x[idx]), float(y[idx]), float(z[idx]), float(q_arr[idx])])
        for idx in ano_atoms:
            w.writerow([idx, "anode", float(x[idx]), float(y[idx]), float(z[idx]), float(q_arr[idx])])

    cath_auto_vlim = max(np.max(np.abs(cath_sigma)), 1e-12)
    ano_auto_vlim = max(np.max(np.abs(ano_sigma)), 1e-12)
    if args.vlim is not None:
        cath_vlim = float(args.vlim)
        ano_vlim = float(args.vlim)
    else:
        cath_vlim = cath_auto_vlim
        ano_vlim = ano_auto_vlim

    fig = plt.figure(figsize=(12.2, 4.8), dpi=180, constrained_layout=True)
    gs = GridSpec(nrows=1, ncols=4, figure=fig, width_ratios=[1.0, 0.045, 1.0, 0.045])
    ax1 = fig.add_subplot(gs[0, 0])
    cax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    cax2 = fig.add_subplot(gs[0, 3])

    im1 = ax1.imshow(
        cath_sigma,
        origin="lower",
        extent=[0.0, box_x_A, 0.0, box_y_A],
        cmap="coolwarm",
        vmin=-cath_vlim,
        vmax=cath_vlim,
        aspect="equal",
    )
    ax1.set_title("Cathode atomic charge map")
    ax1.set_xlabel("x (A)")
    ax1.set_ylabel("y (A)")
    im2 = ax2.imshow(
        ano_sigma,
        origin="lower",
        extent=[0.0, box_x_A, 0.0, box_y_A],
        cmap="coolwarm",
        vmin=-ano_vlim,
        vmax=ano_vlim,
        aspect="equal",
    )
    ax2.set_title("Anode atomic charge map")
    ax2.set_xlabel("x (A)")
    ax2.set_ylabel("y (A)")
    cbar1 = fig.colorbar(im1, cax=cax1)
    cbar1.set_label("cathode sigma_q (e/nm^2)")
    cbar2 = fig.colorbar(im2, cax=cax2)
    cbar2.set_label("anode sigma_q (e/nm^2)")
    fig.suptitle(
        "Last-frame electrode charges "
        f"(frame count={nframe}, cathode vlim={cath_vlim:.4f}, anode vlim={ano_vlim:.4f} e/nm^2)"
    )
    fig.savefig(out_png)
    plt.close(fig)

    print(f"Frames read: {nframe} (used last frame)")
    print(f"Cathode total charge: {np.sum(q_arr[cath_atoms]):.8f} e")
    print(f"Anode total charge: {np.sum(q_arr[ano_atoms]):.8f} e")
    print(f"Wrote {out_png}")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
