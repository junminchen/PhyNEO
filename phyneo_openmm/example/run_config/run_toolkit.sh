#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

PYTHONPATH="$REPO_ROOT" /opt/anaconda3/envs/mpid84/bin/python ./run_toolkit_adapter.py \
  --config ./config_packmol_bulk_ec_transport.json
