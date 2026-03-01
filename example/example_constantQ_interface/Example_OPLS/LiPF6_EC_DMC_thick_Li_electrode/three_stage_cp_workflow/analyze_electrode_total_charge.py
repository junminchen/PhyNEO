#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path


def parse_charge_timeseries(path: Path):
    rows = []
    with path.open() as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 4:
                continue
            step = int(parts[0])
            q_left = float(parts[1])
            q_right = float(parts[2])
            q_total = float(parts[3])
            rows.append((step, q_left, q_right, q_total))
    if not rows:
        raise RuntimeError(f"No valid rows in {path}")
    return rows


def mean(vals):
    return sum(vals) / len(vals)


def std(vals):
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    var = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
    return math.sqrt(var)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize electrode total charge time series")
    parser.add_argument("--input", default="outputs/electrode_total_charge_timeseries.dat")
    parser.add_argument("--output", default="outputs/electrode_total_charge_summary.txt")
    args = parser.parse_args()

    in_path = Path(args.input).resolve()
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = parse_charge_timeseries(in_path)
    steps = [r[0] for r in rows]
    q_left = [r[1] for r in rows]
    q_right = [r[2] for r in rows]
    q_total = [r[3] for r in rows]

    drift = q_total[-1] - q_total[0]

    text = "\n".join(
        [
            f"n_frames: {len(rows)}",
            f"step_start: {steps[0]}",
            f"step_end: {steps[-1]}",
            f"Q_left_mean_e: {mean(q_left):.10f}",
            f"Q_left_std_e: {std(q_left):.10f}",
            f"Q_right_mean_e: {mean(q_right):.10f}",
            f"Q_right_std_e: {std(q_right):.10f}",
            f"Q_total_mean_e: {mean(q_total):.10f}",
            f"Q_total_std_e: {std(q_total):.10f}",
            f"Q_total_drift_e: {drift:.10f}",
        ]
    ) + "\n"

    out_path.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
