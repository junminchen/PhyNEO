#!/bin/bash

set -u

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PYTHON_BIN="${PYTHON_BIN:-python}"
FOLDER_PATTERN="${FOLDER_PATTERN:-newer*}"
OUTDIR="${OUTDIR:-shell_dynamics_reports}"
TOP_PATTERNS="${TOP_PATTERNS:-solvent_salt.pdb,*.pdb,*.gro,*.prmtop,*.psf}"
TRAJ_PATTERNS="${TRAJ_PATTERNS:-transport_results/nvt.dcd,*.dcd,*.xtc,*.nc,*.trr}"

CATION_SELECTION="${CATION_SELECTION:-resname LI}"
SOLVENT_SELECTION="${SOLVENT_SELECTION:-(resname EC EMC DMC FEC DEC PC) and (name O* or type O*)}"
ANALYSIS_TARGET="${ANALYSIS_TARGET:-additive}"
ADDITIVE_SELECTION_TEMPLATE="${ADDITIVE_SELECTION_TEMPLATE:-resname {additive} and (name O* or type O*)}"
ADDITIVE_FALLBACK_SELECTION="${ADDITIVE_FALLBACK_SELECTION:-(resname EC EMC DMC FEC DEC PC) and (name O* or type O*)}"
ADDITIVE_MAP="${ADDITIVE_MAP:-}"
SKIP_NO_ADDITIVE="${SKIP_NO_ADDITIVE:-1}"

START="${START:-0}"
STOP="${STOP:-}"
STEP="${STEP:-1}"
RDF_NBINS="${RDF_NBINS:-250}"
DT_PS="${DT_PS:-}"

CMD=(
    "$PYTHON_BIN" "$SCRIPT_DIR/analyze_solvation_shell_dynamics.py"
    --folder-pattern "$FOLDER_PATTERN"
    --top-patterns "$TOP_PATTERNS"
    --traj-patterns "$TRAJ_PATTERNS"
    --analysis-target "$ANALYSIS_TARGET"
    --cation-selection "$CATION_SELECTION"
    --solvent-selection "$SOLVENT_SELECTION"
    --additive-selection-template "$ADDITIVE_SELECTION_TEMPLATE"
    --additive-fallback-selection "$ADDITIVE_FALLBACK_SELECTION"
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

if [ -n "$ADDITIVE_MAP" ]; then
    CMD+=(--additive-map "$ADDITIVE_MAP")
fi

if [ "$SKIP_NO_ADDITIVE" = "1" ]; then
    CMD+=(--skip-no-additive)
fi

if [ "$#" -gt 0 ]; then
    CMD+=("$@")
fi

echo "[Run] ${CMD[*]}"
"${CMD[@]}"
