#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python render_packmol.py
packmol < packmol.inp
echo "Wrote electrolyte_start.pdb"
