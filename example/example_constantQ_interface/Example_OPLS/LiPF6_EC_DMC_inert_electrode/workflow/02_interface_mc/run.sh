#!/usr/bin/env bash
set -euo pipefail

# Stage 2: build interface + MC gap equilibration
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo "[02] Build slab with Packmol"
bash run_packmol.sh

echo "[02] Add inert electrodes"
python assemble_inert_electrode_system.py

echo "[02] MC gap equilibration"
python mc_gap_equilibrate.py

echo "[02] Done. Key output: start_with_electrodes_mc.pdb"
