#!/usr/bin/env bash
set -euo pipefail

# Stage 4: analyses + equilibration checks
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo "[04] Interface distribution"
python analyze_interface_distribution.py --between-electrodes-only --electrode_margin 1.0 --interface_width 5.0 --nblocks 5

echo "[04] Interfacial capacitance"
python analyze_interfacial_capacitance.py --skip 2 --nblocks 5

echo "[04] Electrode charge maps (last frame)"
python visualize_last_frame_electrode_charge.py --xy-bins 40

echo "[04] Equilibration quality checks"
python workflow/04_analysis/check_equilibration.py

echo "[04] Done. Outputs in results/"
