#!/usr/bin/env bash
set -euo pipefail

# Unified installer for phyneo_openmm stack.
# Supports:
# 1) OpenMM installation by conda (recommended)
# 2) OpenMM installation via package/polff/submodules/openmm/install.sh (Linux)
# 3) MPIDOpenMMPlugin build/install (Linux/macOS; CPU or CUDA)
#
# Examples:
#   bash PhyNEO-OpenMM/scripts/install_phyneo_openmm.sh
#   bash PhyNEO-OpenMM/scripts/install_phyneo_openmm.sh --env-name mpid84 --install-openmm conda --mpid-mode cpu
#   bash PhyNEO-OpenMM/scripts/install_phyneo_openmm.sh --install-openmm submodule --openmm-prefix /usr/local/openmm --mpid-mode cuda

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_SUBMODULE_OPENMM_DIR="$(cd "${ROOT_DIR}/.." && pwd)/package/polff/submodules/openmm"
DEFAULT_MPID_PLUGIN_SRC="${ROOT_DIR}/phyneo_openmm/MPIDOpenMMPlugin"

ENV_NAME="mpid84"
OPENMM_VERSION="8.4.0"
INSTALL_OPENMM="conda"       # conda|submodule|skip
MPID_MODE="cpu"              # cpu|cuda
OPENMM_PREFIX=""
PYTHON_EXEC=""
SUBMODULE_OPENMM_DIR="${DEFAULT_SUBMODULE_OPENMM_DIR}"
MPID_PLUGIN_SRC="${DEFAULT_MPID_PLUGIN_SRC}"
RUN_CHECK="yes"

OS_NAME="$(uname -s)"

usage() {
  cat <<'EOF'
Usage:
  install_phyneo_openmm.sh [options]

Options:
  --env-name NAME                 Conda environment name (default: mpid84)
  --openmm-version VER            OpenMM version for conda install (default: 8.4.0)
  --install-openmm MODE           conda|submodule|skip (default: conda)
  --mpid-mode MODE                cpu|cuda (default: cpu)
  --openmm-prefix PATH            OpenMM install prefix (used by submodule/skip mode)
  --python-exec PATH              Python executable used for MPID wrapper install/check
  --submodule-openmm-dir PATH     Path to package/polff/submodules/openmm
  --mpid-plugin-src PATH          Path to MPIDOpenMMPlugin source
  --run-check yes|no              Run import smoke check at end (default: yes)
  -h, --help                      Show this help
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
    --submodule-openmm-dir)
      SUBMODULE_OPENMM_DIR="$2"; shift 2;;
    --mpid-plugin-src)
      MPID_PLUGIN_SRC="$2"; shift 2;;
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

if [[ "${INSTALL_OPENMM}" != "conda" && "${INSTALL_OPENMM}" != "submodule" && "${INSTALL_OPENMM}" != "skip" ]]; then
  echo "[ERROR] --install-openmm must be one of: conda, submodule, skip" >&2
  exit 1
fi
if [[ "${MPID_MODE}" != "cpu" && "${MPID_MODE}" != "cuda" ]]; then
  echo "[ERROR] --mpid-mode must be one of: cpu, cuda" >&2
  exit 1
fi
if [[ "${RUN_CHECK}" != "yes" && "${RUN_CHECK}" != "no" ]]; then
  echo "[ERROR] --run-check must be yes or no" >&2
  exit 1
fi

if [[ ! -d "${MPID_PLUGIN_SRC}" ]]; then
  echo "[ERROR] MPID plugin source not found: ${MPID_PLUGIN_SRC}" >&2
  exit 1
fi

get_nproc() {
  if command -v nproc >/dev/null 2>&1; then
    nproc
  elif command -v sysctl >/dev/null 2>&1; then
    sysctl -n hw.ncpu
  else
    echo 4
  fi
}

require_conda() {
  if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] conda not found. Install Miniconda/Anaconda first." >&2
    exit 1
  fi
}

env_exists() {
  conda env list | awk '{print $1}' | grep -xq "$1"
}

install_openmm_conda() {
  require_conda
  local conda_packages=(
    "python=3.11"
    "openmm=${OPENMM_VERSION}"
    "cmake"
    "make"
    "swig"
    "numpy"
    "pip"
  )
  if [[ "${MPID_MODE}" == "cuda" ]]; then
    conda_packages+=("cudatoolkit")
  fi

  if env_exists "${ENV_NAME}"; then
    echo "[INFO] Conda env already exists: ${ENV_NAME}"
  else
    echo "[INFO] Creating conda env: ${ENV_NAME}"
    conda create -y -n "${ENV_NAME}" -c conda-forge "${conda_packages[@]}"
  fi

  OPENMM_PREFIX="$(conda run -n "${ENV_NAME}" python -c 'import os; print(os.environ["CONDA_PREFIX"])' | tail -n 1)"
  if [[ -z "${PYTHON_EXEC}" ]]; then
    PYTHON_EXEC="${OPENMM_PREFIX}/bin/python"
  fi
}

install_openmm_submodule() {
  if [[ "${OS_NAME}" != "Linux" ]]; then
    echo "[ERROR] --install-openmm submodule currently only supports Linux." >&2
    exit 1
  fi
  if [[ ! -d "${SUBMODULE_OPENMM_DIR}" ]]; then
    echo "[ERROR] Submodule openmm dir not found: ${SUBMODULE_OPENMM_DIR}" >&2
    exit 1
  fi
  if [[ ! -x "${SUBMODULE_OPENMM_DIR}/install.sh" ]]; then
    echo "[ERROR] Missing executable script: ${SUBMODULE_OPENMM_DIR}/install.sh" >&2
    exit 1
  fi
  if [[ ! -f "${SUBMODULE_OPENMM_DIR}/build_openmm.sh" || ! -f "${SUBMODULE_OPENMM_DIR}/build_openmm_vv.sh" ]]; then
    echo "[ERROR] Missing build scripts under: ${SUBMODULE_OPENMM_DIR}" >&2
    exit 1
  fi

  if [[ -z "${OPENMM_PREFIX}" ]]; then
    OPENMM_PREFIX="/usr/local/openmm"
  fi
  echo "[INFO] Running submodule installer (includes build_openmm.sh + build_openmm_vv.sh):"
  echo "       ${SUBMODULE_OPENMM_DIR}/install.sh ${OPENMM_PREFIX}"
  (
    cd "${SUBMODULE_OPENMM_DIR}"
    bash ./install.sh "${OPENMM_PREFIX}"
  )

  if [[ -z "${PYTHON_EXEC}" ]]; then
    PYTHON_EXEC="$(command -v python3 || true)"
  fi
}

resolve_openmm_skip() {
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

build_mpid_plugin() {
  local cuda_flag="OFF"
  if [[ "${MPID_MODE}" == "cuda" ]]; then
    cuda_flag="ON"
    if [[ "${OS_NAME}" != "Linux" ]]; then
      echo "[ERROR] CUDA mode is only expected on Linux in this installer." >&2
      exit 1
    fi
  fi

  if [[ ! -x "${PYTHON_EXEC}" ]]; then
    echo "[ERROR] Python executable not found/executable: ${PYTHON_EXEC}" >&2
    exit 1
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
    -DPYTHON_EXECUTABLE="${PYTHON_EXEC}"

  cmake --build "${build_dir}" -j "$(get_nproc)"
  cmake --install "${build_dir}"
  cmake --build "${build_dir}" --target PythonInstall
}

run_smoke_check() {
  local plugin_dir="${OPENMM_PREFIX}/lib/plugins"
  local pyver site_guess
  pyver="$("${PYTHON_EXEC}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  site_guess="${OPENMM_PREFIX}/lib/python${pyver}/site-packages"

  echo "[INFO] Running import smoke check"
  if [[ "${INSTALL_OPENMM}" == "conda" ]]; then
    conda run -n "${ENV_NAME}" env OPENMM_PLUGIN_DIR="${plugin_dir}" python -c \
      "import openmm, openmm.app, openmm.unit; import mpidplugin; print('OpenMM:', openmm.__version__)"
  else
    env OPENMM_PLUGIN_DIR="${plugin_dir}" PYTHONPATH="${site_guess}:${PYTHONPATH:-}" "${PYTHON_EXEC}" -c \
      "import openmm, openmm.app, openmm.unit; import mpidplugin; print('OpenMM:', openmm.__version__)"
  fi
}

echo "[INFO] OS               : ${OS_NAME}"
echo "[INFO] install-openmm   : ${INSTALL_OPENMM}"
echo "[INFO] mpid-mode        : ${MPID_MODE}"
echo "[INFO] submodule dir    : ${SUBMODULE_OPENMM_DIR}"
echo "[INFO] mpid plugin src  : ${MPID_PLUGIN_SRC}"

case "${INSTALL_OPENMM}" in
  conda)
    install_openmm_conda
    ;;
  submodule)
    install_openmm_submodule
    ;;
  skip)
    resolve_openmm_skip
    ;;
esac

build_mpid_plugin

if [[ "${RUN_CHECK}" == "yes" ]]; then
  run_smoke_check
fi

echo
echo "[DONE] phyneo_openmm install finished."
echo "[INFO] OPENMM_PREFIX=${OPENMM_PREFIX}"
echo "[INFO] OPENMM_PLUGIN_DIR=${OPENMM_PREFIX}/lib/plugins"
if [[ "${INSTALL_OPENMM}" == "conda" ]]; then
  echo "[INFO] Activate env:"
  echo "       conda activate ${ENV_NAME}"
  echo "       export OPENMM_PLUGIN_DIR=${OPENMM_PREFIX}/lib/plugins"
else
  echo "[INFO] Set environment before running Python:"
  echo "       export OPENMM_PLUGIN_DIR=${OPENMM_PREFIX}/lib/plugins"
fi
