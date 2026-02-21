#!/usr/bin/env bash
set -euo pipefail

# Run from this folder so relative pdb/xml in config work as-is.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

PYTHONPATH="$REPO_ROOT" /opt/anaconda3/envs/mpid84/bin/python "$REPO_ROOT/phyneo_openmm/phyneo_protocol.py" \
  --config ./config_packmol_bulk_ec_transport.json
