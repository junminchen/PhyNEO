#!/bin/bash

set -u

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PYTHON_BIN="${PYTHON_BIN:-python}"
FOLDER_PATTERN="${FOLDER_PATTERN:-newer*}"
OUTDIR="${OUTDIR:-shell_dynamics_reports}"

CATION_SELECTION="${CATION_SELECTION:-resname LI}"
SOLVENT_SELECTION="${SOLVENT_SELECTION:-(resname EC EMC DMC FEC DEC PC) and (name O* or type O*)}"

START="${START:-0}"
STOP="${STOP:-}"
STEP="${STEP:-1}"
RDF_NBINS="${RDF_NBINS:-250}"
DT_PS="${DT_PS:-}"

CMD=(
    "$PYTHON_BIN" "$SCRIPT_DIR/analyze_solvation_shell_dynamics.py"
    --folder-pattern "$FOLDER_PATTERN"
    --cation-selection "$CATION_SELECTION"
    --solvent-selection "$SOLVENT_SELECTION"
    --start "$START"
    --step "$STEP"
    --rdf-nbins "$RDF_NBINS"
    --outdir "$OUTDIR"
)

if [ -n "$STOP" ]; then
    CMD+=(--stop "$STOP")
fi

if [ -n "$DT_PS" ]; then
    CMD+=(--dt-ps "$DT_PS")
fi

if [ "$#" -gt 0 ]; then
    CMD+=("$@")
fi

echo "[Run] ${CMD[*]}"
"${CMD[@]}"
