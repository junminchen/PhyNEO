#!/usr/bin/env bash
set -euo pipefail

# Stage 3: constant-potential MD
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo "[03] Run CPF MD"
python run_openmm84_inert_electrode.py

echo "[03] Done. Outputs: npt_inert.log traj_inert.dcd electrode_charges.log"
