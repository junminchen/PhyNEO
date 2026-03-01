#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python run_three_stage_cp_workflow.py "$@"
python analyze_electrode_total_charge.py --input outputs/electrode_total_charge_timeseries.dat --output outputs/electrode_total_charge_summary.txt
