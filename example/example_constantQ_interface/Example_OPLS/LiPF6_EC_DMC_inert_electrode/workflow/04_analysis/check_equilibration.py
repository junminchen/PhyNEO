#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CheckResult:
    name: str
    status: str
    value: str
    criterion: str


def parse_state_density(log_path: Path) -> tuple[np.ndarray, np.ndarray]:
    steps, dens = [], []
    with log_path.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        hmap = {h.replace("#", "").replace('"', "").strip(): i for i, h in enumerate(header)}
        step_i = hmap.get("Step")
        dens_i = hmap.get("Density (g/mL)")
        if step_i is None or dens_i is None:
            raise RuntimeError(f"Missing Step/Density in {log_path}")
        for row in reader:
            if not row:
                continue
            try:
                steps.append(float(row[step_i]))
                dens.append(float(row[dens_i]))
            except (ValueError, IndexError):
                continue
    if not steps:
        raise RuntimeError(f"No data rows in {log_path}")
    return np.asarray(steps), np.asarray(dens)


def tail_stats(x: np.ndarray, y: np.ndarray, tail_fraction: float = 1/3) -> tuple[float, float, float]:
    n = len(y)
    n_tail = max(5, int(n * tail_fraction))
    xt = x[-n_tail:]
    yt = y[-n_tail:]
    mean = float(np.mean(yt))
    std = float(np.std(yt, ddof=1)) if n_tail > 1 else 0.0
    cv_pct = 100.0 * std / max(abs(mean), 1e-12)
    p = np.polyfit(xt, yt, deg=1)
    drift = (p[0] * (xt[-1] - xt[0]))
    drift_pct = 100.0 * float(drift) / max(abs(mean), 1e-12)
    return mean, cv_pct, drift_pct


def parse_charge_log(path: Path):
    step, qc, qa, qt = [], [], [], []
    for line in path.read_text().splitlines():
        s = line.strip()
        if (not s) or s.startswith("#"):
            continue
        t = s.split()
        if len(t) < 4:
            continue
        step.append(float(t[0]))
        qc.append(float(t[1]))
        qa.append(float(t[2]))
        qt.append(float(t[3]))
    if not step:
        raise RuntimeError(f"No rows in {path}")
    return np.asarray(step), np.asarray(qc), np.asarray(qa), np.asarray(qt)


def parse_distribution(path: Path):
    z, rho_tot, rho_q = [], [], []
    with path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                zz = float(r["z_angstrom"])
                rt = float(r["rho_Li_1_per_nm3"]) + float(r["rho_PF6_1_per_nm3"]) + float(r["rho_EC_1_per_nm3"]) + float(r["rho_DMC_1_per_nm3"])
                rq = float(r["rho_charge_e_per_nm3"])
            except (ValueError, KeyError):
                continue
            z.append(zz)
            rho_tot.append(rt)
            rho_q.append(rq)
    if not z:
        raise RuntimeError(f"No rows in {path}")
    return np.asarray(z), np.asarray(rho_tot), np.asarray(rho_q)


def pass_warn(ok: bool) -> str:
    return "PASS" if ok else "WARN"


def main() -> None:
    p = argparse.ArgumentParser(description="Equilibration quality checks for bulk->interface->CPM workflow")
    p.add_argument("--bulk-log", default="../LiPF6_EC_DMC_minimal/npt.log")
    p.add_argument("--interface-log", default="npt_inert.log")
    p.add_argument("--charge-log", default="electrode_charges.log")
    p.add_argument("--distribution-csv", default="results/interface_distribution.csv")
    p.add_argument("--out", default="results/check_equilibration_report.txt")
    args = p.parse_args()

    here = Path(__file__).resolve().parents[2]
    bulk_log = (here / args.bulk_log).resolve()
    int_log = (here / args.interface_log).resolve()
    charge_log = (here / args.charge_log).resolve()
    dist_csv = (here / args.distribution_csv).resolve()
    out = (here / args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    checks: list[CheckResult] = []

    xb, db = parse_state_density(bulk_log)
    bulk_mean, bulk_cv, bulk_drift = tail_stats(xb, db)
    ok_bulk = (bulk_cv <= 3.0) and (abs(bulk_drift) <= 3.0)
    checks.append(CheckResult(
        "Bulk density plateau",
        pass_warn(ok_bulk),
        f"mean={bulk_mean:.4f} g/mL, cv={bulk_cv:.2f}%, drift={bulk_drift:.2f}%",
        "cv<=3% and |drift|<=3%",
    ))

    xi, di = parse_state_density(int_log)
    int_mean, int_cv, int_drift = tail_stats(xi, di)
    rel_diff = 100.0 * abs(int_mean - bulk_mean) / max(abs(bulk_mean), 1e-12)
    ok_int = (int_cv <= 5.0) and (abs(int_drift) <= 5.0) and (rel_diff <= 5.0)
    checks.append(CheckResult(
        "Interface slab density vs bulk",
        pass_warn(ok_int),
        f"interface={int_mean:.4f} g/mL, bulk={bulk_mean:.4f} g/mL, diff={rel_diff:.2f}%",
        "interface cv<=5%, |drift|<=5%, bulk diff<=5%",
    ))

    z, rho_tot, rho_q = parse_distribution(dist_csv)
    zmin, zmax = float(np.min(z)), float(np.max(z))
    lo = zmin + 0.3 * (zmax - zmin)
    hi = zmin + 0.7 * (zmax - zmin)
    mask = (z >= lo) & (z <= hi)
    mid_tot = rho_tot[mask]
    mid_q = rho_q[mask]
    mid_cv = 100.0 * float(np.std(mid_tot, ddof=1)) / max(float(np.mean(mid_tot)), 1e-12)
    mid_q_abs = float(np.mean(np.abs(mid_q)))
    ok_mid = (mid_cv <= 10.0) and (mid_q_abs <= 0.10)
    checks.append(CheckResult(
        "Mid-slab profile quality",
        pass_warn(ok_mid),
        f"mid total-density cv={mid_cv:.2f}%, mean|rho_q|={mid_q_abs:.4f} e/nm^3",
        "mid cv<=10% and mean|rho_q|<=0.10 e/nm^3",
    ))

    xc, qc, qa, qt = parse_charge_log(charge_log)
    n = len(xc)
    tail = max(5, n // 2)
    qc_t = qc[-tail:]
    qa_t = qa[-tail:]
    qt_t = qt[-tail:]
    qtotal_abs = float(np.mean(np.abs(qt_t)))

    h = tail // 2
    drift_qc = abs(float(np.mean(qc_t[h:]) - np.mean(qc_t[:h]))) if h > 1 else float("nan")
    drift_qa = abs(float(np.mean(qa_t[h:]) - np.mean(qa_t[:h]))) if h > 1 else float("nan")
    ok_cpm = (qtotal_abs <= 0.10) and (drift_qc <= 0.10) and (drift_qa <= 0.10)
    checks.append(CheckResult(
        "CPM charge convergence",
        pass_warn(ok_cpm),
        f"mean|Q_total|={qtotal_abs:.4f} e, dQ_c={drift_qc:.4f} e, dQ_a={drift_qa:.4f} e",
        "mean|Q_total|<=0.10 e and half-tail drift <=0.10 e",
    ))

    with out.open("w") as f:
        f.write("Equilibration Check Report\n")
        f.write("==========================\n")
        f.write(f"bulk_log: {bulk_log}\n")
        f.write(f"interface_log: {int_log}\n")
        f.write(f"charge_log: {charge_log}\n")
        f.write(f"distribution_csv: {dist_csv}\n\n")
        for c in checks:
            f.write(f"[{c.status}] {c.name}\n")
            f.write(f"  value: {c.value}\n")
            f.write(f"  criterion: {c.criterion}\n\n")

    for c in checks:
        print(f"[{c.status}] {c.name}: {c.value}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
