#!/bin/bash

set -u

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
FOLDER_PATTERN="${FOLDER_PATTERN:-newer*}"
LOG_DIR="${LOG_DIR:-gpu4pyscf_logs}"
MAX_STRUCTURES_PER_CATEGORY="${MAX_STRUCTURES_PER_CATEGORY:-100}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DO_BENCH="${DO_BENCH:-0}"                  # 1: run one-structure SCF benchmark
BENCH_FOLDERS_LIMIT="${BENCH_FOLDERS_LIMIT:-1}"

count_selected_structures() {
    local folder="$1"
    local base="$folder/classified_structures"
    local total=0

    if [ ! -d "$base" ]; then
        echo 0
        return
    fi

    while IFS= read -r cat_dir; do
        [ -d "$cat_dir" ] || continue
        local n
        n=$(find "$cat_dir" -maxdepth 1 -type f -name "*.xyz" | wc -l | tr -d ' ')
        if [ "$n" -gt "$MAX_STRUCTURES_PER_CATEGORY" ]; then
            n="$MAX_STRUCTURES_PER_CATEGORY"
        fi
        total=$((total + n))
    done < <(find "$base" -mindepth 1 -maxdepth 1 -type d | sort)

    echo "$total"
}

find_log_for_folder() {
    local folder_name="$1"
    local newest
    newest=$(ls -1t "$LOG_DIR/${folder_name}.gpu"*.log 2>/dev/null | head -n 1 || true)
    echo "$newest"
}

progress_done_from_log() {
    local log_file="$1"
    if [ ! -f "$log_file" ]; then
        echo 0
        return
    fi
    local count
    count=$(grep -c "Progress:" "$log_file" 2>/dev/null || true)
    echo "${count:-0}"
}

status_from_log() {
    local log_file="$1"
    if [ ! -f "$log_file" ]; then
        echo "PENDING"
        return
    fi
    if grep -q "Finished calculation for" "$log_file"; then
        echo "DONE"
        return
    fi
    if grep -q "Traceback (most recent call last)" "$log_file"; then
        echo "ERROR"
        return
    fi
    echo "RUNNING"
}

print_header() {
    echo "============================================================"
    echo "GPU4PySCF Performance Debug Report"
    echo "Time: $(date '+%F %T %Z')"
    echo "Host: $(hostname)"
    echo "Workdir: $(pwd)"
    echo "Pattern: $FOLDER_PATTERN | Logs: $LOG_DIR"
    echo "============================================================"
}

print_system_info() {
    echo
    echo "[1] System & GPU"
    uname -a
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index,name,driver_version,pstate,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw,power.limit,clocks.sm --format=csv,noheader,nounits
    else
        echo "nvidia-smi: not found"
    fi
}

print_python_info() {
    echo
    echo "[2] Python Environment"
    echo "python bin: $(command -v "$PYTHON_BIN" || echo "$PYTHON_BIN (not found in PATH)")"
    "$PYTHON_BIN" - <<'PY'
import importlib
import platform
import sys

print(f"python_version: {sys.version.split()[0]}")
print(f"platform: {platform.platform()}")

mods = ["numpy", "cupy", "pyscf", "gpu4pyscf"]
for m in mods:
    try:
        mod = importlib.import_module(m)
        ver = getattr(mod, "__version__", "unknown")
        path = getattr(mod, "__file__", "unknown")
        print(f"{m}: version={ver} path={path}")
    except Exception as e:
        print(f"{m}: NOT_AVAILABLE ({e})")

try:
    import cupy as cp
    ndev = cp.cuda.runtime.getDeviceCount()
    print(f"cupy_device_count: {ndev}")
    for i in range(ndev):
        p = cp.cuda.runtime.getDeviceProperties(i)
        name = p.get("name", b"")
        if isinstance(name, bytes):
            name = name.decode(errors="ignore")
        print(f"cupy_device_{i}: {name}")
except Exception as e:
    print(f"cupy_device_query: failed ({e})")
PY
}

print_folder_and_log_stats() {
    echo
    echo "[3] Folder Progress & Throughput (from logs)"
    mapfile -t DIRS < <(find . -maxdepth 1 -type d -name "$FOLDER_PATTERN" | sort)
    if [ "${#DIRS[@]}" -eq 0 ]; then
        echo "No folders matching '$FOLDER_PATTERN'"
        return
    fi

    printf "%-24s %8s %8s %8s %10s %12s %s\n" "Folder" "Need" "Done" "Remain" "Status" "Throughput" "Log"
    printf "%-24s %8s %8s %8s %10s %12s %s\n" "------------------------" "--------" "--------" "--------" "----------" "------------" "---"

    local sum_need=0
    local sum_done=0

    for folder in "${DIRS[@]}"; do
        [ -d "$folder/classified_structures" ] || continue
        local folder_name need log_file done status remain throughput last_prog elapsed_min
        folder_name="$(basename "$folder")"
        need="$(count_selected_structures "$folder")"
        log_file="$(find_log_for_folder "$folder_name")"
        done="$(progress_done_from_log "$log_file")"
        status="$(status_from_log "$log_file")"

        if [ "$need" -gt 0 ] && [ "$done" -gt "$need" ]; then
            done="$need"
        fi
        if [ "$status" = "DONE" ]; then
            done="$need"
        fi

        remain=$((need - done))
        if [ "$remain" -lt 0 ]; then
            remain=0
        fi

        throughput="-"
        if [ -n "${log_file:-}" ] && [ -f "$log_file" ]; then
            last_prog="$(grep "Progress:" "$log_file" | tail -n 1 || true)"
            elapsed_min="$(echo "$last_prog" | sed -n 's/.*Elapsed:[[:space:]]*\([0-9.]\+\)[[:space:]]*min.*/\1/p')"
            if [ -n "$elapsed_min" ]; then
                throughput=$("$PYTHON_BIN" - <<PY
done = float("$done")
elapsed_min = float("$elapsed_min")
if elapsed_min > 0:
    print(f"{done/elapsed_min:.2f} stru/min")
else:
    print("-")
PY
)
            fi
        fi

        local log_name="-"
        if [ -n "${log_file:-}" ]; then
            log_name="$(basename "$log_file")"
        fi

        printf "%-24s %8d %8d %8d %10s %12s %s\n" "$folder_name" "$need" "$done" "$remain" "$status" "$throughput" "$log_name"
        sum_need=$((sum_need + need))
        sum_done=$((sum_done + done))
    done

    local sum_remain=$((sum_need - sum_done))
    if [ "$sum_remain" -lt 0 ]; then
        sum_remain=0
    fi
    echo "Summary: done=${sum_done}/${sum_need} remain=${sum_remain}"
}

run_single_structure_bench() {
    echo
    echo "[4] Optional SCF Micro Benchmark"
    if [ "$DO_BENCH" != "1" ]; then
        echo "Skipped. To enable: DO_BENCH=1 ./debug_gpu4pyscf_perf.sh"
        return
    fi

    mapfile -t DIRS < <(find . -maxdepth 1 -type d -name "$FOLDER_PATTERN" | sort | head -n "$BENCH_FOLDERS_LIMIT")
    if [ "${#DIRS[@]}" -eq 0 ]; then
        echo "No folders found for benchmark."
        return
    fi

    for folder in "${DIRS[@]}"; do
        [ -d "$folder/classified_structures" ] || continue
        xyz="$(find "$folder/classified_structures" -type f -name '*.xyz' | head -n 1 || true)"
        if [ -z "${xyz:-}" ]; then
            echo "No xyz file in $(basename "$folder"), skip."
            continue
        fi
        echo "Benchmark file: $xyz"
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PYTHON_BIN" - <<PY
import time
import numpy as np
from pyscf import gto
from gpu4pyscf import dft

xyz_path = r"$xyz"

with open(xyz_path, "r") as f:
    lines = f.readlines()
n_atoms = int(lines[0].strip())
coords = []
for line in lines[2:2+n_atoms]:
    p = line.split()
    coords.append([p[0], (float(p[1]), float(p[2]), float(p[3]))])

mol = gto.Mole()
mol.atom = coords
mol.basis = "6-311++G(d,p)"
mol.charge = 0
mol.spin = 0
mol.verbose = 0
mol.build()

mf = dft.RKS(mol, xc="B3LYP").density_fit()
t0 = time.time()
e = mf.kernel()
t1 = time.time()

mo = mf.mo_energy
occ = mf.mo_occ
ha_to_ev = 27.211386
homo = float("nan")
lumo = float("nan")
if np.any(occ > 0):
    homo = mo[np.where(occ > 0)[0][-1]] * ha_to_ev
if np.any(occ == 0):
    lumo = mo[np.where(occ == 0)[0][0]] * ha_to_ev
print(f"SCF done: energy={e:.8f} Ha time={(t1-t0):.2f} s homo={homo:.4f} lumo={lumo:.4f}")
PY
    done
}

print_tips() {
    cat <<'EOF'

[5] Slow-Run Checklist
1) Check nvidia-smi utilization/pstate: long-term GPU util near 0% often means CPU/I/O bottleneck.
2) Compare "Throughput stru/min" across folders to find outliers.
3) If one machine is much slower, compare driver/cuda/cupy/gpu4pyscf versions from section [2].
4) Confirm CUDA visibility: run with explicit binding, e.g. CUDA_VISIBLE_DEVICES=0.
5) If needed, lower basis or reduce MAX_STRUCTURES_PER_CATEGORY for quick A/B tests.
EOF
}

print_header
print_system_info
print_python_info
print_folder_and_log_stats
run_single_structure_bench
print_tips
