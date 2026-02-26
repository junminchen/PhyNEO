#!/bin/bash

set -u

# Multi-GPU batch runner for run_gpu4pyscf.py
# Default: scan newer* folders and dispatch one folder per free GPU.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
FOLDER_PATTERN="${FOLDER_PATTERN:-newer*}"
GPU_IDS_STR="${GPU_IDS:-0,1,2}"          # e.g. "0,1,2"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_DIR="${LOG_DIR:-gpu4pyscf_logs}"

IFS=',' read -r -a GPU_IDS <<< "$GPU_IDS_STR"
if [ "${#GPU_IDS[@]}" -eq 0 ]; then
    echo "[Error] No GPU IDs configured. Set GPU_IDS, e.g. GPU_IDS=0,1,2"
    exit 1
fi

mkdir -p "$LOG_DIR"

mapfile -t DIRS < <(find . -maxdepth 1 -type d -name "$FOLDER_PATTERN" | sort)
if [ "${#DIRS[@]}" -eq 0 ]; then
    echo "[Error] No folders matching '$FOLDER_PATTERN' in $(pwd)"
    exit 1
fi

declare -a FREE_GPUS=("${GPU_IDS[@]}")
declare -A PID_TO_GPU
declare -A PID_TO_DIR
success_count=0
fail_count=0

launch_job() {
    local folder="$1"
    local gpu="$2"
    local folder_name
    folder_name="$(basename "$folder")"
    local log_file="$LOG_DIR/${folder_name}.gpu${gpu}.log"

    echo "[Launch] $folder_name on GPU $gpu (log: $log_file)"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" "$SCRIPT_DIR/run_gpu4pyscf.py" "$folder" > "$log_file" 2>&1 &
    local pid=$!
    PID_TO_GPU["$pid"]="$gpu"
    PID_TO_DIR["$pid"]="$folder_name"
}

reap_one() {
    local finished_pid=""
    local exit_code=0
    if wait -n -p finished_pid; then
        exit_code=0
    else
        exit_code=$?
    fi

    local gpu="${PID_TO_GPU[$finished_pid]}"
    local folder_name="${PID_TO_DIR[$finished_pid]}"
    unset PID_TO_GPU["$finished_pid"]
    unset PID_TO_DIR["$finished_pid"]
    FREE_GPUS+=("$gpu")

    if [ "$exit_code" -eq 0 ]; then
        success_count=$((success_count + 1))
        echo "[Done] $folder_name on GPU $gpu"
    else
        fail_count=$((fail_count + 1))
        echo "[Fail] $folder_name on GPU $gpu (exit=$exit_code)"
    fi
}

echo "========================================="
echo "   Multi-GPU GPU4PySCF Batch Runner      "
echo "   GPUs: ${GPU_IDS[*]}"
echo "   Pattern: $FOLDER_PATTERN"
echo "   Folders: ${#DIRS[@]}"
echo "========================================="

for folder in "${DIRS[@]}"; do
    if [ ! -d "$folder/classified_structures" ]; then
        echo "[Skip] $(basename "$folder"): no classified_structures"
        continue
    fi

    while [ "${#FREE_GPUS[@]}" -eq 0 ]; do
        reap_one
    done

    gpu="${FREE_GPUS[0]}"
    FREE_GPUS=("${FREE_GPUS[@]:1}")
    launch_job "$folder" "$gpu"
done

while [ "${#PID_TO_GPU[@]}" -gt 0 ]; do
    reap_one
done

echo "========================================="
echo "Completed. success=$success_count fail=$fail_count"
echo "Logs in: $LOG_DIR"
echo "========================================="

if [ "$fail_count" -gt 0 ]; then
    exit 1
fi

