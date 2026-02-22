#!/usr/bin/env python3
"""
Batch launcher: runs run_single.py for each solvent as a subprocess.
Each solvent runs independently so CUDA crashes don't kill the whole batch.

Usage: conda activate mpid && python batch_density.py
"""
import subprocess
import sys
import os
import csv
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"
PYTHON = sys.executable

SOLVENTS = ["EC", "DEC", "DMC", "PC", "FEC", "DME", "PS", "SL", "EMC"]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for solvent in SOLVENTS:
        csv_path = OUTPUT_DIR / f"{solvent.lower()}_density.csv"
        if csv_path.exists():
            print(f"[SKIP] {solvent}: result already exists", flush=True)
            continue

        print(f"\n{'='*60}", flush=True)
        print(f"[START] {solvent}", flush=True)
        print(f"{'='*60}", flush=True)

        result = subprocess.run(
            [PYTHON, "-u", str(SCRIPT_DIR / "run_single.py"), solvent],
            timeout=1800,
            cwd=str(SCRIPT_DIR),
        )

        if result.returncode != 0:
            print(f"[FAIL] {solvent} exited with code {result.returncode}", flush=True)
        else:
            print(f"[DONE] {solvent}", flush=True)

    # Merge all individual CSVs into one
    print(f"\n{'='*60}", flush=True)
    print("Merging results...", flush=True)
    merged_path = OUTPUT_DIR / "density_results.csv"
    header_written = False
    with open(merged_path, "w", newline="") as out_f:
        writer = csv.writer(out_f)
        for solvent in SOLVENTS:
            csv_path = OUTPUT_DIR / f"{solvent.lower()}_density.csv"
            if not csv_path.exists():
                print(f"  {solvent}: NO RESULT (failed)", flush=True)
                continue
            with open(csv_path) as in_f:
                reader = csv.reader(in_f)
                header = next(reader)
                if not header_written:
                    writer.writerow(header)
                    header_written = True
                for row in reader:
                    writer.writerow(row)
                    print(f"  {solvent}: {row}", flush=True)

    print(f"\nMerged CSV: {merged_path}", flush=True)


if __name__ == "__main__":
    main()
