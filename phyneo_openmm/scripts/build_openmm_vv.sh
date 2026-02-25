#!/usr/bin/env bash
set -euo pipefail

# Legacy wrapper -> unified installer
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENMM_DIR="${1:-/usr/local/openmm}"
if [[ $# -gt 0 ]]; then shift; fi

exec "${SCRIPT_DIR}/install_phyneo_openmm.sh" \
  --mode build-openmm-vv \
  --legacy-source-dir "$(pwd)" \
  --openmm-prefix "${OPENMM_DIR}" \
  "$@"
