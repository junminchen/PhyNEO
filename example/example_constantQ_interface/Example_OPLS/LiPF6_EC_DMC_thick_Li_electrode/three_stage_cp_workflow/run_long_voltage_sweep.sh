#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PLATFORM="${PLATFORM:-CUDA}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NEUTRAL_STEPS="${NEUTRAL_STEPS:-100000}"
ACTIVATION_STEPS="${ACTIVATION_STEPS:-500000}"
PRODUCTION_STEPS="${PRODUCTION_STEPS:-5000000}"
REPORT_INTERVAL="${REPORT_INTERVAL:-5000}"
BULK_LOG="${BULK_LOG:-}"

VOLTS=(0.0 2.0 4.0)

echo "[INFO] start_time=$(date '+%F %T')"
echo "[INFO] platform=${PLATFORM}"
echo "[INFO] python_bin=${PYTHON_BIN}"
echo "[INFO] neutral_steps=${NEUTRAL_STEPS} activation_steps=${ACTIVATION_STEPS} production_steps=${PRODUCTION_STEPS} report_interval=${REPORT_INTERVAL}"
if [ -n "$BULK_LOG" ]; then
  echo "[INFO] bulk_log=${BULK_LOG}"
fi

for v in "${VOLTS[@]}"; do
  tag="V${v//./p}"
  outdir="sweep_${tag}"
  mkdir -p "$outdir"

  echo "[INFO] running voltage=${v}V output=${outdir} at $(date '+%F %T')"
  extra_bulk_args=()
  if [ -n "$BULK_LOG" ]; then
    extra_bulk_args+=(--bulk-log "$BULK_LOG")
  fi
  PYTHONUNBUFFERED=1 "$PYTHON_BIN" run_three_stage_cp_workflow.py \
    --platform "$PLATFORM" \
    --delta-phi-v "$v" \
    --neutral-relax-steps "$NEUTRAL_STEPS" \
    --cp-activation-steps "$ACTIVATION_STEPS" \
    --production-steps "$PRODUCTION_STEPS" \
    --report-interval "$REPORT_INTERVAL" \
    "${extra_bulk_args[@]}" \
    --output-dir "$outdir/outputs"

  PYTHONUNBUFFERED=1 "$PYTHON_BIN" analyze_electrode_total_charge.py \
    --input "$outdir/outputs/electrode_total_charge_timeseries.dat" \
    --output "$outdir/outputs/electrode_total_charge_summary.txt"

  echo "[INFO] completed voltage=${v}V at $(date '+%F %T')"
done

echo "[INFO] all_done at $(date '+%F %T')"
