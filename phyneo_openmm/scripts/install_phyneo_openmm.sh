#!/usr/bin/env bash
set -euo pipefail

# Unified installer for phyneo_openmm/scripts.
# Modes:
#   1) phyneo (default): install OpenMM + build MPID plugin + smoke checks
#   2) build-openmm: legacy openmm source build (from a source directory)
#   3) build-openmm-vv: legacy openmm-velocityVerlet source build
#   4) install-openmm-stack: clone/build/install openmm + openmm-velocityVerlet

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"   # .../phyneo_openmm
REPO_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"    # .../PhyNEO
MPID_PLUGIN_SRC_DEFAULT="${ROOT_DIR}/MPIDOpenMMPlugin"

MODE="phyneo"   # phyneo|build-openmm|build-openmm-vv|install-openmm-stack

ENV_NAME="mpid84"
OPENMM_VERSION="8.4.0"
INSTALL_OPENMM="conda"   # conda|skip
MPID_MODE="cpu"          # cpu|cuda
OPENMM_PREFIX=""
PYTHON_EXEC=""
MPID_PLUGIN_SRC="${MPID_PLUGIN_SRC_DEFAULT}"
INSTALL_PY_DEPS="yes"
RUN_CHECK="yes"

LEGACY_SOURCE_DIR=""
LEGACY_WORK_DIR="${SCRIPT_DIR}/_legacy_build"
LEGACY_INSTALL_SYSTEM_DEPS="no"

OS_NAME="$(uname -s)"

usage() {
  cat <<'EOF'
Usage:
  install_phyneo_openmm.sh [options]

General:
  --mode MODE                 phyneo|build-openmm|build-openmm-vv|install-openmm-stack
  --python-exec PATH          Python executable used for build/install/check
  --openmm-prefix PATH        OpenMM install prefix (required in some modes)

Mode=phyneo options:
  --env-name NAME             Conda env name (default: mpid84)
  --openmm-version VER        OpenMM version (default: 8.4.0)
  --install-openmm MODE       conda|skip (default: conda)
  --mpid-mode MODE            cpu|cuda (default: cpu)
  --mpid-plugin-src PATH      MPIDOpenMMPlugin source dir
  --install-py-deps yes|no    Install pandas/scipy (default: yes)
  --run-check yes|no          Run smoke checks (default: yes)

Legacy mode options:
  --legacy-source-dir PATH    Source dir with CMakeLists.txt for build-openmm/build-openmm-vv
  --legacy-work-dir PATH      Working dir for install-openmm-stack clone/build
  --legacy-install-system-deps yes|no  apt install build deps on Linux (default: no)

Examples:
  bash install_phyneo_openmm.sh
  bash install_phyneo_openmm.sh --mode phyneo --env-name mpid84 --mpid-mode cpu
  bash install_phyneo_openmm.sh --mode phyneo --install-openmm skip --openmm-prefix "$CONDA_PREFIX" --python-exec "$CONDA_PREFIX/bin/python"

  # legacy-like build from current openmm source dir
  bash install_phyneo_openmm.sh --mode build-openmm --legacy-source-dir "$PWD" --openmm-prefix /usr/local/openmm

  # legacy-like build from current openmm-velocityVerlet source dir
  bash install_phyneo_openmm.sh --mode build-openmm-vv --legacy-source-dir "$PWD" --openmm-prefix /usr/local/openmm

  # clone & install both openmm and openmm-vv
  bash install_phyneo_openmm.sh --mode install-openmm-stack --openmm-prefix /usr/local/openmm
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"; shift 2;;
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
    --legacy-source-dir)
      LEGACY_SOURCE_DIR="$2"; shift 2;;
    --legacy-work-dir)
      LEGACY_WORK_DIR="$2"; shift 2;;
    --legacy-install-system-deps)
      LEGACY_INSTALL_SYSTEM_DEPS="$2"; shift 2;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage
      exit 1;;
  esac
done

fail() { echo "[ERROR] $*" >&2; exit 1; }
info() { echo "[INFO] $*"; }

require_conda() {
  command -v conda >/dev/null 2>&1 || fail "conda not found"
}

env_exists() {
  conda env list | awk '{print $1}' | grep -xq "$1"
}

get_nproc() {
  if command -v nproc >/dev/null 2>&1; then nproc
  elif command -v sysctl >/dev/null 2>&1; then sysctl -n hw.ncpu
  else echo 4; fi
}

resolve_python_exec() {
  if [[ -z "${PYTHON_EXEC}" ]]; then
    PYTHON_EXEC="$(command -v python3 || true)"
  fi
  [[ -x "${PYTHON_EXEC}" ]] || fail "Python executable not found: ${PYTHON_EXEC}"
}

detect_abi_flag() {
  local py="${PYTHON_EXEC}"
  if "${py}" -c "import torch" >/dev/null 2>&1; then
    local abi
    abi="$(${py} -c "import torch; print(int(bool(torch._C._GLIBCXX_USE_CXX11_ABI)))")"
    echo "${abi}"
  else
    echo 0
  fi
}

check_openmm_prefix() {
  [[ -n "${OPENMM_PREFIX}" ]] || fail "--openmm-prefix is required"
  [[ "${OPENMM_PREFIX}" = /*/openmm ]] || info "openmm-prefix does not end with /openmm (allowed)"
}

install_openmm_conda() {
  require_conda
  local pkgs=("python=3.11" "openmm=${OPENMM_VERSION}" "cmake" "make" "swig" "pip" "numpy")
  [[ "${MPID_MODE}" == "cuda" ]] && pkgs+=("cudatoolkit")

  if env_exists "${ENV_NAME}"; then
    info "Conda env already exists: ${ENV_NAME}"
  else
    info "Creating conda env: ${ENV_NAME}"
    conda create -y -n "${ENV_NAME}" -c conda-forge "${pkgs[@]}"
  fi

  OPENMM_PREFIX="$(conda run -n "${ENV_NAME}" python -c 'import os; print(os.environ["CONDA_PREFIX"])' | tail -n 1)"
  [[ -n "${PYTHON_EXEC}" ]] || PYTHON_EXEC="${OPENMM_PREFIX}/bin/python"
}

resolve_skip_mode() {
  check_openmm_prefix
  [[ -d "${OPENMM_PREFIX}" ]] || fail "openmm-prefix not found: ${OPENMM_PREFIX}"
  resolve_python_exec
}

install_python_deps() {
  [[ "${INSTALL_PY_DEPS}" == "yes" ]] || { info "Skip python deps install"; return; }
  info "Installing python deps using ${PYTHON_EXEC}"
  "${PYTHON_EXEC}" -m pip install --upgrade pip
  "${PYTHON_EXEC}" -m pip install pandas scipy
}

build_mpid_plugin() {
  [[ -d "${MPID_PLUGIN_SRC}" ]] || fail "MPID plugin source not found: ${MPID_PLUGIN_SRC}"
  resolve_python_exec

  local cuda_flag="OFF"
  if [[ "${MPID_MODE}" == "cuda" ]]; then
    [[ "${OS_NAME}" == "Linux" ]] || fail "CUDA mode is only supported on Linux in this installer"
    cuda_flag="ON"
  fi

  local build_tag
  build_tag="$(echo "${OPENMM_VERSION}" | tr -d '.')"
  local build_dir="${MPID_PLUGIN_SRC}/build-openmm${build_tag}-${OS_NAME,,}-${MPID_MODE}"

  info "Building MPIDOpenMMPlugin"
  info "source=${MPID_PLUGIN_SRC}"
  info "build=${build_dir}"
  info "prefix=${OPENMM_PREFIX}"

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

run_phyneo_smoke_check() {
  local plugin_dir="${OPENMM_PREFIX}/lib/plugins"
  local run_prefix=(env "OPENMM_PLUGIN_DIR=${plugin_dir}" "PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}" "${PYTHON_EXEC}")

  info "Smoke check 1/3: openmm + mpidplugin"
  "${run_prefix[@]}" - <<'PY'
import openmm
import openmm.app
import openmm.unit
import mpidplugin
print('OpenMM:', openmm.__version__)
print('mpidplugin import ok')
PY

  info "Smoke check 2/3: toolkit adapter import"
  "${run_prefix[@]}" - <<'PY'
from phyneo_openmm.toolkit.protocol import create_protocol_from_config
print('toolkit import ok:', callable(create_protocol_from_config))
PY

  info "Smoke check 3/3: load_phyneo_system"
  "${run_prefix[@]}" - <<'PY'
from pathlib import Path
from phyneo_openmm.phyneo_protocol import load_phyneo_system

root = Path('/Users/jeremychen/Desktop/Project/project_electrolyte/OpenMM_PhyNEO/PhyNEO/phyneo_openmm/example/run_config')
pdb = root / 'bulk_ec_packmol.pdb'
xml = root / 'caff_5_mpid_slater_bond_hcp.xml'
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

legacy_install_system_deps_if_needed() {
  if [[ "${LEGACY_INSTALL_SYSTEM_DEPS}" != "yes" ]]; then
    info "Skip system deps install (set --legacy-install-system-deps yes to enable)"
    return
  fi
  if [[ "${OS_NAME}" != "Linux" ]]; then
    info "legacy system deps auto-install only supported on Linux; skipping"
    return
  fi
  if command -v apt >/dev/null 2>&1; then
    info "Installing legacy build deps via apt"
    sudo apt update
    sudo apt install -y cmake swig doxygen python3-venv build-essential git
  else
    info "apt not found; install build deps manually"
  fi
}

legacy_build_openmm() {
  resolve_python_exec
  check_openmm_prefix
  local src="${LEGACY_SOURCE_DIR:-$(pwd)}"
  [[ -f "${src}/CMakeLists.txt" ]] || fail "Invalid source dir (missing CMakeLists.txt): ${src}"

  legacy_install_system_deps_if_needed

  local abi
  abi="$(detect_abi_flag)"
  info "Legacy build-openmm source=${src} abi=${abi} prefix=${OPENMM_PREFIX}"

  local output_dir="${src}/output"
  local build_dir="${src}/build"
  rm -rf "${output_dir}" "${build_dir}"
  mkdir -p "${output_dir}" "${build_dir}"

  (
    cd "${build_dir}"
    cmake -DBUILD_TESTING=OFF \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX="${OPENMM_PREFIX}" \
      -DCMAKE_CXX_FLAGS:STRING="-D_GLIBCXX_USE_CXX11_ABI=${abi}" \
      -DCMAKE_CXX_FLAGS_RELEASE:STRING="-O2 -DNDEBUG" \
      -DCMAKE_C_FLAGS_RELEASE:STRING="-O2 -DNDEBUG" \
      -DOPENMM_BUILD_AMOEBA_CUDA_LIB=ON \
      -DOPENMM_BUILD_AMOEBA_OPENCL_LIB=OFF \
      -DOPENMM_BUILD_AMOEBA_PLUGIN=ON \
      -DOPENMM_BUILD_COMMON=OFF \
      -DOPENMM_BUILD_CPU_LIB=ON \
      -DOPENMM_BUILD_CUDA_LIB=ON \
      -DOPENMM_BUILD_CUDA_TESTS=OFF \
      -DOPENMM_BUILD_DRUDE_CUDA_LIB=ON \
      -DOPENMM_BUILD_DRUDE_OPENCL_LIB=OFF \
      -DOPENMM_BUILD_DRUDE_PLUGIN=ON \
      -DOPENMM_BUILD_EXAMPLES=ON \
      -DOPENMM_BUILD_OPENCL_LIB=OFF \
      -DOPENMM_BUILD_PME_PLUGIN=ON \
      -DOPENMM_BUILD_PYTHON_WRAPPERS=ON \
      -DOPENMM_BUILD_RPMD_CUDA_LIB=ON \
      -DOPENMM_BUILD_RPMD_OPENCL_LIB=OFF \
      -DOPENMM_BUILD_RPMD_PLUGIN=ON \
      -DOPENMM_BUILD_SHARED_LIB=ON \
      -DOPENMM_BUILD_STATIC_LIB=OFF \
      -DOPENMM_GENERATE_API_DOCS=OFF \
      -DPYTHON_EXECUTABLE="${PYTHON_EXEC}" \
      "${src}"
    cmake --build . -j "$(get_nproc)"
    cmake --install .

    export OPENMM_INCLUDE_PATH="${OPENMM_PREFIX}/include"
    export OPENMM_LIB_PATH="${OPENMM_PREFIX}/lib"
    export CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=${abi}"

    cat << 'EOF' > python/pyproject.toml
[build-system]
requires = ["setuptools", "wheel", "Cython>=3.0,<4", "oldest-supported-numpy"]
build-backend = "setuptools.build_meta"
EOF
    (
      cd python
      "${PYTHON_EXEC}" -m pip install build
      "${PYTHON_EXEC}" -m build --wheel .
      cp dist/*.whl "${output_dir}/" || true
    )
  )

  tar -C "${OPENMM_PREFIX}/.." -czf "${output_dir}/openmm.tar.gz" "$(basename "${OPENMM_PREFIX}")"
  info "Legacy build-openmm done. Artifacts in ${output_dir}"
}

legacy_build_openmm_vv() {
  resolve_python_exec
  check_openmm_prefix
  local src="${LEGACY_SOURCE_DIR:-$(pwd)}"
  [[ -f "${src}/CMakeLists.txt" ]] || fail "Invalid source dir (missing CMakeLists.txt): ${src}"

  legacy_install_system_deps_if_needed

  local abi
  abi="$(detect_abi_flag)"
  info "Legacy build-openmm-vv source=${src} abi=${abi} openmm_dir=${OPENMM_PREFIX}"

  local output_dir="${src}/output"
  local build_dir="${src}/build"
  rm -rf "${output_dir}" "${build_dir}"
  mkdir -p "${output_dir}" "${build_dir}"

  (
    cd "${build_dir}"
    cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX="${output_dir}/openmm_vv" \
      -DVELOCITYVERLET_BUILD_CUDA_LIB=ON \
      -DCMAKE_CXX_FLAGS:STRING="-D_GLIBCXX_USE_CXX11_ABI=${abi}" \
      -DOPENMM_DIR="${OPENMM_PREFIX}" \
      -DPYTHON_EXECUTABLE="${PYTHON_EXEC}" \
      "${src}"
    cmake --build . -j "$(get_nproc)"
    cmake --install .
    cmake --build . --target PythonInstall || true
    (
      cd python
      "${PYTHON_EXEC}" -m pip install build
      "${PYTHON_EXEC}" -m build --wheel .
      cp dist/*.whl "${output_dir}/" || true
    )
  )

  tar -C "${output_dir}/openmm_vv" -czf "${output_dir}/openmm_vv.tar.gz" .
  info "Legacy build-openmm-vv done. Artifacts in ${output_dir}"
}

legacy_install_openmm_stack() {
  resolve_python_exec
  check_openmm_prefix
  mkdir -p "${LEGACY_WORK_DIR}"
  local work
  work="$(cd "${LEGACY_WORK_DIR}" && pwd)"
  info "Legacy install stack in ${work}"

  (
    cd "${work}"
    rm -rf openmm openmm-velocityVerlet

    git clone --branch 8.4.0 --single-branch https://github.com/openmm/openmm.git
    LEGACY_SOURCE_DIR="${work}/openmm" legacy_build_openmm

    git clone https://github.com/z-gong/openmm-velocityVerlet.git
    LEGACY_SOURCE_DIR="${work}/openmm-velocityVerlet" legacy_build_openmm_vv

    if compgen -G "${work}/openmm-velocityVerlet/output/*.whl" > /dev/null; then
      "${PYTHON_EXEC}" -m pip install --no-cache-dir --force-reinstall ${work}/openmm-velocityVerlet/output/*.whl
    fi
  )

  info "Legacy install-openmm-stack done"
}

run_phyneo_mode() {
  [[ "${INSTALL_OPENMM}" == "conda" || "${INSTALL_OPENMM}" == "skip" ]] || fail "--install-openmm must be conda|skip"
  [[ "${MPID_MODE}" == "cpu" || "${MPID_MODE}" == "cuda" ]] || fail "--mpid-mode must be cpu|cuda"
  [[ "${INSTALL_PY_DEPS}" == "yes" || "${INSTALL_PY_DEPS}" == "no" ]] || fail "--install-py-deps must be yes|no"
  [[ "${RUN_CHECK}" == "yes" || "${RUN_CHECK}" == "no" ]] || fail "--run-check must be yes|no"

  info "OS=${OS_NAME}"
  info "mode=phyneo"
  info "install-openmm=${INSTALL_OPENMM}"
  info "mpid-mode=${MPID_MODE}"

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
    run_phyneo_smoke_check
  fi

  echo
  echo "[DONE] phyneo install finished"
  echo "[INFO] OPENMM_PREFIX=${OPENMM_PREFIX}"
  echo "[INFO] OPENMM_PLUGIN_DIR=${OPENMM_PREFIX}/lib/plugins"
  echo "[INFO] PYTHONPATH add: ${REPO_ROOT}"
  [[ "${INSTALL_OPENMM}" == "conda" ]] && echo "[INFO] Activate env: conda activate ${ENV_NAME}"
}

case "${MODE}" in
  phyneo)
    run_phyneo_mode
    ;;
  build-openmm)
    legacy_build_openmm
    ;;
  build-openmm-vv)
    legacy_build_openmm_vv
    ;;
  install-openmm-stack)
    legacy_install_openmm_stack
    ;;
  *)
    fail "Unknown --mode: ${MODE}"
    ;;
esac
