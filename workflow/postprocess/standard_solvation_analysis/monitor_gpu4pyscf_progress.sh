#!/bin/bash

set -u

FOLDER_PATTERN="${FOLDER_PATTERN:-newer*}"
LOG_DIR="${LOG_DIR:-gpu4pyscf_logs}"
MAX_STRUCTURES_PER_CATEGORY="${MAX_STRUCTURES_PER_CATEGORY:-100}"

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

progress_done_from_log() {
    local log_file="$1"
    if [ ! -f "$log_file" ]; then
        echo 0
        return
    fi
    grep -c "Progress:" "$log_file" 2>/dev/null || echo 0
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

find_log_for_folder() {
    local folder_name="$1"
    local newest
    newest=$(ls -1t "$LOG_DIR/${folder_name}.gpu"*.log 2>/dev/null | head -n 1 || true)
    echo "$newest"
}

mapfile -t DIRS < <(find . -maxdepth 1 -type d -name "$FOLDER_PATTERN" | sort)
if [ "${#DIRS[@]}" -eq 0 ]; then
    echo "[Error] No folders matching '$FOLDER_PATTERN' in $(pwd)"
    exit 1
fi

printf "%-24s %10s %10s %9s %10s %s\n" "Folder" "Need" "Done" "Progress" "Status" "Log"
printf "%-24s %10s %10s %9s %10s %s\n" "------------------------" "----------" "----------" "---------" "----------" "---"

sum_need=0
sum_done=0
runnable=0
done_folders=0

for folder in "${DIRS[@]}"; do
    folder_name="$(basename "$folder")"
    if [ ! -d "$folder/classified_structures" ]; then
        continue
    fi

    runnable=$((runnable + 1))
    need="$(count_selected_structures "$folder")"
    log_file="$(find_log_for_folder "$folder_name")"
    done="$(progress_done_from_log "$log_file")"
    status="$(status_from_log "$log_file")"

    if [ "$need" -gt 0 ] && [ "$done" -gt "$need" ]; then
        done="$need"
    fi
    if [ "$status" = "DONE" ]; then
        done_folders=$((done_folders + 1))
        done="$need"
    fi

    sum_need=$((sum_need + need))
    sum_done=$((sum_done + done))

    if [ "$need" -gt 0 ]; then
        pct=$((100 * done / need))
    else
        pct=100
    fi

    log_name="-"
    if [ -n "${log_file:-}" ]; then
        log_name="$(basename "$log_file")"
    fi

    printf "%-24s %10d %10d %8d%% %10s %s\n" "$folder_name" "$need" "$done" "$pct" "$status" "$log_name"
done

if [ "$sum_need" -gt 0 ]; then
    sum_pct=$((100 * sum_done / sum_need))
else
    sum_pct=100
fi

echo
echo "Summary: folders_done=${done_folders}/${runnable} structures_done=${sum_done}/${sum_need} (${sum_pct}%)"
echo "Tips: run with 'watch -n 30 ./monitor_gpu4pyscf_progress.sh'"
