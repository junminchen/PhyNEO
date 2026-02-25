#!/usr/bin/env bash
set -euo pipefail

# Stage 1: bulk equilibration in sibling minimal OPLS example
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BULK_DIR="$ROOT/../LiPF6_EC_DMC_minimal"

cd "$BULK_DIR"
echo "[01] Bulk: Packmol"
bash run_packmol.sh

echo "[01] Bulk: NPT+production (run_md_opls.py)"
python run_md_opls.py

echo "[01] Bulk: NVT continuation"
python "$ROOT/workflow/01_bulk_equil/run_nvt_from_npt.py" --bulk-dir "../LiPF6_EC_DMC_minimal" --steps 10000 --report-interval 500

echo "[01] Done. Outputs in: $BULK_DIR"
