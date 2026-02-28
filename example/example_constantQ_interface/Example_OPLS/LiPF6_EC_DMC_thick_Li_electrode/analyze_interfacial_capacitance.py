#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

E_CHARGE_C = 1.602176634e-19
ANGSTROM2_TO_M2 = 1.0e-20
F_PER_M2_TO_UF_PER_CM2 = 100.0


def parse_charge_log(path: Path):
    steps = []
    q_cath = []
    q_anode = []
    q_total = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if (not s) or s.startswith("#"):
            continue
        toks = s.split()
        if len(toks) < 4:
            continue
        steps.append(int(float(toks[0])))
        q_cath.append(float(toks[1]))
        q_anode.append(float(toks[2]))
        q_total.append(float(toks[3]))
    if not steps:
        raise RuntimeError(f"No data rows in {path}")
    return np.asarray(steps), np.asarray(q_cath), np.asarray(q_anode), np.asarray(q_total)


def block_stats(values: np.ndarray, nblocks: int):
    n = values.shape[0]
    if n < 2:
        return float(values.mean()), np.nan, 1
    bsize = max(1, n // max(1, nblocks))
    nblk = max(1, n // bsize)
    if nblk < 2:
        return float(values.mean()), np.nan, nblk
    means = []
    for b in range(nblk):
        lo = b * bsize
        hi = min((b + 1) * bsize, n)
        if hi <= lo:
            continue
        means.append(float(np.mean(values[lo:hi])))
    if len(means) < 2:
        return float(values.mean()), np.nan, len(means)
    arr = np.asarray(means)
    return float(arr.mean()), float(arr.std(ddof=1)), len(means)


def main():
    parser = argparse.ArgumentParser(description="Compute interfacial capacitance from constant-potential electrode charge log.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--charge-log", default="electrode_charges.log")
    parser.add_argument("--voltage-v", type=float, default=None, help="Override voltage in config")
    parser.add_argument("--skip", type=int, default=0, help="Number of initial rows to skip as equilibration")
    parser.add_argument("--nblocks", type=int, default=5)
    parser.add_argument("--summary", default="results/interfacial_capacitance_summary.txt")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    cfg = json.loads((here / args.config).read_text())
    steps, q_cath, q_anode, q_total = parse_charge_log((here / args.charge_log).resolve())

    if args.skip > 0:
        if args.skip >= len(steps):
            raise RuntimeError(f"--skip {args.skip} >= samples {len(steps)}")
        steps = steps[args.skip:]
        q_cath = q_cath[args.skip:]
        q_anode = q_anode[args.skip:]
        q_total = q_total[args.skip:]

    area_A2 = float(cfg["cell"]["box_x_angstrom"]) * float(cfg["cell"]["box_y_angstrom"])
    area_m2 = area_A2 * ANGSTROM2_TO_M2
    voltage_V = float(cfg["electrode"]["voltage_v"] if args.voltage_v is None else args.voltage_v)
    if voltage_V <= 0.0:
        raise RuntimeError("electrode.voltage_v must be > 0 for capacitance")

    # One-electrode interfacial capacitance from |Q_electrode| / (A * DeltaV)
    c_cath = np.abs(q_cath) * E_CHARGE_C / (area_m2 * voltage_V)
    c_anode = np.abs(q_anode) * E_CHARGE_C / (area_m2 * voltage_V)
    c_avg = 0.5 * (c_cath + c_anode)

    mean_cath, std_cath, nblk_c = block_stats(c_cath, args.nblocks)
    mean_ano, std_ano, nblk_a = block_stats(c_anode, args.nblocks)
    mean_avg, std_avg, nblk_m = block_stats(c_avg, args.nblocks)

    out = (here / args.summary).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write("Interfacial capacitance summary\n")
        f.write("definition: C_int = |Q_electrode|/(A*DeltaV)\n")
        f.write(f"samples_used: {len(steps)}\n")
        f.write(f"step_min: {int(steps.min())}\n")
        f.write(f"step_max: {int(steps.max())}\n")
        f.write(f"area_A2: {area_A2:.6f}\n")
        f.write(f"voltage_V: {voltage_V:.6f}\n")
        f.write(f"Q_cath_mean_e: {np.mean(q_cath):.10f}\n")
        f.write(f"Q_anode_mean_e: {np.mean(q_anode):.10f}\n")
        f.write(f"Q_total_mean_e: {np.mean(q_total):.10f}\n")
        f.write(f"C_cathode_F_per_m2: {mean_cath:.10f}\n")
        f.write(f"C_anode_F_per_m2: {mean_ano:.10f}\n")
        f.write(f"C_interfacial_avg_F_per_m2: {mean_avg:.10f}\n")
        f.write(f"C_cathode_uF_per_cm2: {mean_cath * F_PER_M2_TO_UF_PER_CM2:.6f}\n")
        f.write(f"C_anode_uF_per_cm2: {mean_ano * F_PER_M2_TO_UF_PER_CM2:.6f}\n")
        f.write(f"C_interfacial_avg_uF_per_cm2: {mean_avg * F_PER_M2_TO_UF_PER_CM2:.6f}\n")
        if np.isfinite(std_avg):
            f.write(f"C_interfacial_avg_block_std_uF_per_cm2: {std_avg * F_PER_M2_TO_UF_PER_CM2:.6f}\n")
            f.write(f"blocks_used: {nblk_m}\n")
        else:
            f.write("C_interfacial_avg_block_std_uF_per_cm2: nan\n")
            f.write(f"blocks_used: {min(nblk_c, nblk_a, nblk_m)}\n")
        f.write("note: Differential capacitance requires a voltage sweep (multiple DeltaV points).\n")

    print(f"Samples used: {len(steps)}")
    print(f"Voltage: {voltage_V:.3f} V, area: {area_A2:.2f} A^2")
    print(f"C_interfacial_avg = {mean_avg:.6f} F/m^2 = {mean_avg * F_PER_M2_TO_UF_PER_CM2:.3f} uF/cm^2")
    if np.isfinite(std_avg):
        print(f"Block std = {std_avg * F_PER_M2_TO_UF_PER_CM2:.3f} uF/cm^2")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
