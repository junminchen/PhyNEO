#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PYTHON_BIN="${PYTHON_BIN:-python}"
CSV_PATH="${1:-all_formulations_homolumo.csv}"
OUT_DIR="${2:-analysis_reports}"

"$PYTHON_BIN" "$SCRIPT_DIR/analyze_solvation_results.py" --csv "$CSV_PATH" --outdir "$OUT_DIR"

