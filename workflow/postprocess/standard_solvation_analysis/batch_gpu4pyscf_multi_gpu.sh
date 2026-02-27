#!/bin/bash

set -u

# Multi-GPU batch runner for run_gpu4pyscf.py
# Default: scan newer* folders and dispatch one folder per free GPU.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
FOLDER_PATTERN="${FOLDER_PATTERN:-newer*}"
GPU_IDS_STR="${GPU_IDS:-0,1,2}"          # e.g. "0,1,2"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_DIR="${LOG_DIR:-gpu4pyscf_logs}"
MAX_STRUCTURES_PER_CATEGORY="${MAX_STRUCTURES_PER_CATEGORY:-100}"
FORCE_RERUN="${FORCE_RERUN:-0}"

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

declare -a RUNNABLE_DIRS=()
declare -A PID_TO_GPU
declare -A PID_TO_DIR
declare -A DIR_SELECTED_STRUCTS
declare -A DIR_LOG_FILE
declare -A DIR_ASSIGNED_GPU
declare -A DIR_LAUNCHED
declare -A GPU_LOAD
declare -A GPU_JOB_COUNT
success_count=0
fail_count=0
completed_count=0
launched_count=0
total_jobs=0
total_structures_selected=0
skipped_completed_count=0

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

count_done_from_csv() {
    local folder="$1"
    local folder_name
    folder_name="$(basename "$folder")"
    local csv_file="$folder/results/${folder_name}_homolumo.csv"

    if [ ! -f "$csv_file" ]; then
        echo 0
        return
    fi

    awk -F',' '
        NR <= 1 { next }
        NF >= 3 {
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", $3)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2)
            if ($2 != "") seen[$3 "|" $2] = 1
        }
        END { print length(seen) + 0 }
    ' "$csv_file"
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

render_bar() {
    local done="$1"
    local total="$2"
    local width=20
    local fill=0
    local i
    local bar=""

    if [ "$total" -gt 0 ]; then
        fill=$((done * width / total))
    else
        fill="$width"
    fi

    for ((i = 0; i < width; i++)); do
        if [ "$i" -lt "$fill" ]; then
            bar+="#"
        else
            bar+="-"
        fi
    done
    echo "$bar"
}

print_folder_progress_snapshot() {
    local sum_done=0
    local sum_need=0
    local done_folders=0
    local runnable="${#RUNNABLE_DIRS[@]}"

    echo "[Folder Progress]"
    for folder in "${RUNNABLE_DIRS[@]}"; do
        local folder_name
        folder_name="$(basename "$folder")"
        local need="${DIR_SELECTED_STRUCTS[$folder_name]:-0}"
        local log_file="${DIR_LOG_FILE[$folder_name]:-}"
        local assigned_gpu="${DIR_ASSIGNED_GPU[$folder_name]:-?}"
        local done=0
        local status="PENDING"

        if [ -n "$log_file" ]; then
            done="$(progress_done_from_log "$log_file")"
            status="$(status_from_log "$log_file")"
        fi

        if [ "$need" -gt 0 ] && [ "$done" -gt "$need" ]; then
            done="$need"
        fi
        if [ "$status" = "DONE" ]; then
            done="$need"
            done_folders=$((done_folders + 1))
        fi

        local pct=100
        if [ "$need" -gt 0 ]; then
            pct=$((100 * done / need))
        fi
        local bar
        bar="$(render_bar "$done" "$need")"

        sum_done=$((sum_done + done))
        sum_need=$((sum_need + need))

        printf "  - %-20s gpu=%-3s %5d/%-5d [%s] %3d%% %s\n" "$folder_name" "$assigned_gpu" "$done" "$need" "$bar" "$pct" "$status"
    done

    local sum_pct=100
    if [ "$sum_need" -gt 0 ]; then
        sum_pct=$((100 * sum_done / sum_need))
    fi
    echo "  Summary: folders_done=${done_folders}/${runnable} structures_done=${sum_done}/${sum_need} (${sum_pct}%)"
}

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
    DIR_LOG_FILE["$folder_name"]="$log_file"
    launched_count=$((launched_count + 1))
    echo "[Queue ] launched=$launched_count/$total_jobs completed=$completed_count/$total_jobs active=${#PID_TO_GPU[@]}"
    print_folder_progress_snapshot
}

pick_gpu_with_min_load() {
    local best_gpu="${GPU_IDS[0]}"
    local best_load="${GPU_LOAD[$best_gpu]:-0}"
    local gpu
    for gpu in "${GPU_IDS[@]}"; do
        local cur_load="${GPU_LOAD[$gpu]:-0}"
        if [ "$cur_load" -lt "$best_load" ]; then
            best_gpu="$gpu"
            best_load="$cur_load"
        fi
    done
    echo "$best_gpu"
}

assign_jobs_to_gpus() {
    local lines=()
    local folder
    for folder in "${RUNNABLE_DIRS[@]}"; do
        local folder_name
        folder_name="$(basename "$folder")"
        local need="${DIR_SELECTED_STRUCTS[$folder_name]:-0}"
        lines+=("${need}|${folder}")
    done

    local sorted_lines=()
    mapfile -t sorted_lines < <(printf "%s\n" "${lines[@]}" | sort -t '|' -k1,1nr -k2,2)

    local line
    for line in "${sorted_lines[@]}"; do
        local need folder gpu folder_name
        IFS='|' read -r need folder <<< "$line"
        gpu="$(pick_gpu_with_min_load)"
        folder_name="$(basename "$folder")"
        DIR_ASSIGNED_GPU["$folder_name"]="$gpu"
        GPU_LOAD["$gpu"]=$(( ${GPU_LOAD[$gpu]:-0} + need ))
        GPU_JOB_COUNT["$gpu"]=$(( ${GPU_JOB_COUNT[$gpu]:-0} + 1 ))
    done
}

launch_next_for_gpu() {
    local gpu="$1"
    local folder
    for folder in "${RUNNABLE_DIRS[@]}"; do
        local folder_name
        folder_name="$(basename "$folder")"
        if [ "${DIR_ASSIGNED_GPU[$folder_name]:-}" != "$gpu" ]; then
            continue
        fi
        if [ "${DIR_LAUNCHED[$folder_name]:-0}" -eq 1 ]; then
            continue
        fi
        DIR_LAUNCHED["$folder_name"]=1
        launch_job "$folder" "$gpu"
        return 0
    done
    return 1
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
    completed_count=$((completed_count + 1))

    if [ "$exit_code" -eq 0 ]; then
        success_count=$((success_count + 1))
        echo "[Done] $folder_name on GPU $gpu"
    else
        fail_count=$((fail_count + 1))
        echo "[Fail] $folder_name on GPU $gpu (exit=$exit_code)"
    fi
    echo "[Prog ] completed=$completed_count/$total_jobs success=$success_count fail=$fail_count active=${#PID_TO_GPU[@]}"
    print_folder_progress_snapshot

    launch_next_for_gpu "$gpu" || true
}

echo "========================================="
echo "   Multi-GPU GPU4PySCF Batch Runner      "
echo "   GPUs: ${GPU_IDS[*]}"
echo "   Pattern: $FOLDER_PATTERN"
echo "   Folders: ${#DIRS[@]}"
echo "   FORCE_RERUN: $FORCE_RERUN"
echo "========================================="

for folder in "${DIRS[@]}"; do
    if [ -d "$folder/classified_structures" ]; then
        selected="$(count_selected_structures "$folder")"
        done_count="$(count_done_from_csv "$folder")"
        folder_name="$(basename "$folder")"
        if [ "$FORCE_RERUN" != "1" ] && [ "$selected" -gt 0 ] && [ "$done_count" -ge "$selected" ]; then
            skipped_completed_count=$((skipped_completed_count + 1))
            echo "[Skip] $folder_name: already completed ($done_count/$selected, set FORCE_RERUN=1 to rerun)"
            continue
        fi
        RUNNABLE_DIRS+=("$folder")
        DIR_SELECTED_STRUCTS["$folder_name"]="$selected"
        total_structures_selected=$((total_structures_selected + selected))
        total_jobs=$((total_jobs + 1))
    else
        echo "[Skip] $(basename "$folder"): no classified_structures"
    fi
done

if [ "$total_jobs" -eq 0 ]; then
    echo "[Error] No folders with classified_structures found."
    exit 1
fi
echo "[Info ] total runnable jobs: $total_jobs"
echo "[Info ] total selected structures: $total_structures_selected (per-category cap=$MAX_STRUCTURES_PER_CATEGORY)"
if [ "$skipped_completed_count" -gt 0 ]; then
    echo "[Info ] skipped completed jobs: $skipped_completed_count (use FORCE_RERUN=1 to include)"
fi
assign_jobs_to_gpus
echo "[Plan ] static GPU assignment (balanced by selected structures):"
for gpu in "${GPU_IDS[@]}"; do
    echo "        - GPU $gpu: jobs=${GPU_JOB_COUNT[$gpu]:-0} structures=${GPU_LOAD[$gpu]:-0}"
done
echo "[Info ] per-folder selected structures:"
for folder in "${RUNNABLE_DIRS[@]}"; do
    folder_name="$(basename "$folder")"
    echo "        - $folder_name: ${DIR_SELECTED_STRUCTS[$folder_name]} (gpu=${DIR_ASSIGNED_GPU[$folder_name]})"
done

for gpu in "${GPU_IDS[@]}"; do
    launch_next_for_gpu "$gpu" || true
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
