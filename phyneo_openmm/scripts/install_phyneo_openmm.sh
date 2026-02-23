#!/usr/bin/env bash
set -euo pipefail

# Complete installer for phyneo_openmm (current simplified runtime tree).
# Features:
# 1) Install OpenMM 8.4 (conda) or reuse existing OpenMM prefix
# 2) Build/install local MPIDOpenMMPlugin (CPU/CUDA)
# 3) Install Python runtime deps used by current workflow
# 4) Run smoke checks (OpenMM + mpidplugin + load_phyneo_system)
#
# Example:
#   bash phyneo_openmm/scripts/install_phyneo_openmm.sh
#   bash phyneo_openmm/scripts/install_phyneo_openmm.sh --env-name mpid84 --mpid-mode cpu
#   bash phyneo_openmm/scripts/install_phyneo_openmm.sh --install-openmm skip --openmm-prefix "$CONDA_PREFIX" --python-exec "$CONDA_PREFIX/bin/python"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"           # .../phyneo_openmm
REPO_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"            # .../PhyNEO
MPID_PLUGIN_SRC_DEFAULT="${ROOT_DIR}/MPIDOpenMMPlugin"

ENV_NAME="mpid84"
OPENMM_VERSION="8.4.0"
INSTALL_OPENMM="conda"   # conda|skip
MPID_MODE="cpu"          # cpu|cuda
OPENMM_PREFIX=""
PYTHON_EXEC=""
MPID_PLUGIN_SRC="${MPID_PLUGIN_SRC_DEFAULT}"
INSTALL_PY_DEPS="yes"
RUN_CHECK="yes"

OS_NAME="$(uname -s)"

usage() {
  cat <<'EOF'
Usage:
  install_phyneo_openmm.sh [options]

Options:
  --env-name NAME             Conda environment name (default: mpid84)
  --openmm-version VER        OpenMM version for conda install (default: 8.4.0)
  --install-openmm MODE       conda|skip (default: conda)
  --mpid-mode MODE            cpu|cuda (default: cpu)
  --openmm-prefix PATH        OpenMM install prefix (required for --install-openmm skip)
  --python-exec PATH          Python executable used for install/check
  --mpid-plugin-src PATH      Path to MPIDOpenMMPlugin source
  --install-py-deps yes|no    Install python deps for current workflow (default: yes)
  --run-check yes|no          Run smoke checks at end (default: yes)
  -h, --help                  Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      ENV_NAME="$2"; shift 2;;
    --openmm-version)
      OPENMM_VERSION="$2"; shift 2;;
    --install-openmm)
      INSTALL_OPENMM="$2"; shift 2;;
    --mpid-mode)
      MPID_MODE="$2"; shift 2;;
    --openmm-prefix)
      OPENMM_PREFIX="$2"; shift 2;;
    --python-exec)
      PYTHON_EXEC="$2"; shift 2;;
    --mpid-plugin-src)
      MPID_PLUGIN_SRC="$2"; shift 2;;
    --install-py-deps)
      INSTALL_PY_DEPS="$2"; shift 2;;
    --run-check)
      RUN_CHECK="$2"; shift 2;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage
      exit 1;;
  esac
done

for v in INSTALL_OPENMM MPID_MODE INSTALL_PY_DEPS RUN_CHECK; do
  : "${!v}"
done

if [[ "${INSTALL_OPENMM}" != "conda" && "${INSTALL_OPENMM}" != "skip" ]]; then
  echo "[ERROR] --install-openmm must be conda|skip" >&2
  exit 1
fi
if [[ "${MPID_MODE}" != "cpu" && "${MPID_MODE}" != "cuda" ]]; then
  echo "[ERROR] --mpid-mode must be cpu|cuda" >&2
  exit 1
fi
if [[ "${INSTALL_PY_DEPS}" != "yes" && "${INSTALL_PY_DEPS}" != "no" ]]; then
  echo "[ERROR] --install-py-deps must be yes|no" >&2
  exit 1
fi
if [[ "${RUN_CHECK}" != "yes" && "${RUN_CHECK}" != "no" ]]; then
  echo "[ERROR] --run-check must be yes|no" >&2
  exit 1
fi
if [[ ! -d "${MPID_PLUGIN_SRC}" ]]; then
  echo "[ERROR] MPID plugin source not found: ${MPID_PLUGIN_SRC}" >&2
  exit 1
fi

require_conda() {
  if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] conda not found. Install Miniconda/Anaconda first." >&2
    exit 1
  fi
}

env_exists() {
  conda env list | awk '{print $1}' | grep -xq "$1"
}

get_nproc() {
  if command -v nproc >/dev/null 2>&1; then
    nproc
  elif command -v sysctl >/dev/null 2>&1; then
    sysctl -n hw.ncpu
  else
    echo 4
  fi
}

install_openmm_conda() {
  require_conda
  local pkgs=(
    "python=3.11"
    "openmm=${OPENMM_VERSION}"
    "cmake"
    "make"
    "swig"
    "pip"
    "numpy"
  )
  if [[ "${MPID_MODE}" == "cuda" ]]; then
    pkgs+=("cuda-nvcc")
  fi

  if env_exists "${ENV_NAME}"; then
    echo "[INFO] Conda env already exists: ${ENV_NAME}"
  else
    echo "[INFO] Creating conda env: ${ENV_NAME}"
    conda create -y -n "${ENV_NAME}" -c conda-forge "${pkgs[@]}"
  fi

  OPENMM_PREFIX="$(conda env list | grep -w "${ENV_NAME}" | head -n 1 | awk '{print $NF}')"
  if [[ -z "${PYTHON_EXEC}" ]]; then
    PYTHON_EXEC="${OPENMM_PREFIX}/bin/python"
  fi
}

resolve_skip_mode() {
  if [[ -z "${OPENMM_PREFIX}" ]]; then
    echo "[ERROR] --install-openmm skip requires --openmm-prefix PATH" >&2
    exit 1
  fi
  if [[ ! -d "${OPENMM_PREFIX}" ]]; then
    echo "[ERROR] --openmm-prefix not found: ${OPENMM_PREFIX}" >&2
    exit 1
  fi
  if [[ -z "${PYTHON_EXEC}" ]]; then
    PYTHON_EXEC="$(command -v python3 || true)"
  fi
}

install_python_deps() {
  if [[ "${INSTALL_PY_DEPS}" != "yes" ]]; then
    echo "[INFO] Skip python deps install"
    return
  fi
  echo "[INFO] Installing python deps into: ${PYTHON_EXEC}"
  "${PYTHON_EXEC}" -m pip install --upgrade pip
  "${PYTHON_EXEC}" -m pip install pandas scipy
}

build_mpid_plugin() {
  if [[ ! -x "${PYTHON_EXEC}" ]]; then
    echo "[ERROR] Python executable not found/executable: ${PYTHON_EXEC}" >&2
    exit 1
  fi
  local cuda_flag="OFF"
  if [[ "${MPID_MODE}" == "cuda" ]]; then
    cuda_flag="ON"
    if [[ "${OS_NAME}" != "Linux" ]]; then
      echo "[ERROR] CUDA mode expected on Linux only in this installer." >&2
      exit 1
    fi
  fi

  local build_tag
  build_tag="$(echo "${OPENMM_VERSION}" | tr -d '.')"
  local build_dir="${MPID_PLUGIN_SRC}/build-openmm${build_tag}-${OS_NAME,,}-${MPID_MODE}"

  echo "[INFO] Building MPIDOpenMMPlugin"
  echo "       source: ${MPID_PLUGIN_SRC}"
  echo "       build : ${build_dir}"
  echo "       prefix: ${OPENMM_PREFIX}"
  echo "       python: ${PYTHON_EXEC}"

  cmake -S "${MPID_PLUGIN_SRC}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${OPENMM_PREFIX}" \
    -DOPENMM_DIR="${OPENMM_PREFIX}" \
    -DMPID_BUILD_CUDA_LIB="${cuda_flag}" \
    -DMPID_BUILD_PYTHON_WRAPPERS=ON \
    -DPYTHON_EXECUTABLE="${PYTHON_EXEC}" \
    -DSWIG_EXECUTABLE="${OPENMM_PREFIX}/bin/swig"

  cmake --build "${build_dir}" -j "$(get_nproc)"
  cmake --install "${build_dir}"
  cmake --build "${build_dir}" --target PythonInstall
}

run_smoke_check() {
  local plugin_dir="${OPENMM_PREFIX}/lib/plugins"
  local run_prefix=(env "OPENMM_PLUGIN_DIR=${plugin_dir}" "${PYTHON_EXEC}")

  echo "[INFO] Smoke check 1/3: import openmm + mpidplugin"
  "${run_prefix[@]}" - <<'PY'
import openmm
import openmm.app
import openmm.unit
import mpidplugin
print('OpenMM:', openmm.__version__)
print('mpidplugin import ok')
PY

  echo "[INFO] Smoke check 2/3: import phyneo_openmm toolkit adapter"
  "${run_prefix[@]}" - <<PY
import sys
sys.path.append('${REPO_ROOT}')
from phyneo_openmm.toolkit.protocol import create_protocol_from_config
print('toolkit import ok:', callable(create_protocol_from_config))
PY

  echo "[INFO] Smoke check 3/3: load_phyneo_system on example PDB/XML"
  "${run_prefix[@]}" - <<PY
import sys
sys.path.append('${REPO_ROOT}')
from pathlib import Path
from phyneo_openmm.phyneo_protocol import load_phyneo_system

root = Path('${ROOT_DIR}/example/run_config')
pdb = root / 'bulk_ec_packmol.pdb'
xml = root / 'caff_5_mpid_slater_bond_hcp.xml'
if not xml.exists():
    xml = root / 'caff_5_mpid_slater_bond.xml'

loaded = load_phyneo_system(
    pdb_path=str(pdb),
    xml_path=str(xml),
    nonbonded_method='NoCutoff',
    constraints='none',
    use_mpid_scale_exclusions=True,
    platform='Reference',
    verbose=True,
)
print('particles:', loaded['system'].getNumParticles())
print('mpid_scales_applied:', loaded['mpid_scales_applied'])
PY
}

echo "[INFO] OS               : ${OS_NAME}"
echo "[INFO] install-openmm   : ${INSTALL_OPENMM}"
echo "[INFO] mpid-mode        : ${MPID_MODE}"
echo "[INFO] mpid plugin src  : ${MPID_PLUGIN_SRC}"

case "${INSTALL_OPENMM}" in
  conda)
    install_openmm_conda
    ;;
  skip)
    resolve_skip_mode
    ;;
esac

install_python_deps
build_mpid_plugin

if [[ "${RUN_CHECK}" == "yes" ]]; then
  run_smoke_check
fi

echo
echo "[DONE] phyneo_openmm install finished."
echo "[INFO] OPENMM_PREFIX=${OPENMM_PREFIX}"
echo "[INFO] OPENMM_PLUGIN_DIR=${OPENMM_PREFIX}/lib/plugins"
echo "[INFO] PYTHONPATH add: ${REPO_ROOT}"
if [[ "${INSTALL_OPENMM}" == "conda" ]]; then
  echo "[INFO] Activate env: conda activate ${ENV_NAME}"
fi
