#!/bin/bash

set -u

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <report_A.txt> <report_B.txt> [label_A] [label_B]"
    echo "Example: $0 h20_report.txt a800_report.txt H20 A800"
    exit 1
fi

REPORT_A="$1"
REPORT_B="$2"
LABEL_A="${3:-ReportA}"
LABEL_B="${4:-ReportB}"

if [ ! -f "$REPORT_A" ]; then
    echo "[Error] File not found: $REPORT_A"
    exit 1
fi
if [ ! -f "$REPORT_B" ]; then
    echo "[Error] File not found: $REPORT_B"
    exit 1
fi

extract_scalar() {
    local file="$1"
    local key="$2"
    grep -E "^${key}:" "$file" | tail -n 1 | sed -E "s/^${key}:[[:space:]]*//" || true
}

extract_gpu_line() {
    local file="$1"
    # nvidia-smi csv line starts with numeric index
    grep -E '^[[:space:]]*[0-9]+,' "$file" | head -n 1 || true
}

extract_summary_done_need() {
    local file="$1"
    local line
    line=$(grep -E '^Summary: done=[0-9]+/[0-9]+' "$file" | tail -n 1 || true)
    if [ -z "$line" ]; then
        echo "0 0"
        return
    fi
    local done need
    done=$(echo "$line" | sed -n 's/^Summary: done=\([0-9]\+\)\/\([0-9]\+\).*/\1/p')
    need=$(echo "$line" | sed -n 's/^Summary: done=\([0-9]\+\)\/\([0-9]\+\).*/\2/p')
    echo "${done:-0} ${need:-0}"
}

extract_mean_throughput() {
    local file="$1"
    awk '
      /stru\/min/ {
        for (i=1; i<=NF; i++) {
          if ($i ~ /stru\/min/) {
            v=$(i-1)
            gsub(/[^0-9.]/, "", v)
            if (v != "") { sum += v; n += 1 }
          }
        }
      }
      END {
        if (n > 0) printf "%.4f", sum / n;
        else printf "0";
      }
    ' "$file"
}

echo "============================================================"
echo "GPU4PySCF Cross-Machine Compare"
echo "A: $LABEL_A ($REPORT_A)"
echo "B: $LABEL_B ($REPORT_B)"
echo "============================================================"

py_a=$(extract_scalar "$REPORT_A" "python_version")
py_b=$(extract_scalar "$REPORT_B" "python_version")
cupy_a=$(extract_scalar "$REPORT_A" "cupy")
cupy_b=$(extract_scalar "$REPORT_B" "cupy")
pyscf_a=$(extract_scalar "$REPORT_A" "pyscf")
pyscf_b=$(extract_scalar "$REPORT_B" "pyscf")
g4p_a=$(extract_scalar "$REPORT_A" "gpu4pyscf")
g4p_b=$(extract_scalar "$REPORT_B" "gpu4pyscf")
gpu_a=$(extract_gpu_line "$REPORT_A")
gpu_b=$(extract_gpu_line "$REPORT_B")

read -r done_a need_a <<< "$(extract_summary_done_need "$REPORT_A")"
read -r done_b need_b <<< "$(extract_summary_done_need "$REPORT_B")"
thr_a=$(extract_mean_throughput "$REPORT_A")
thr_b=$(extract_mean_throughput "$REPORT_B")

echo
echo "[Environment]"
printf "%-12s %s\n" "$LABEL_A" "python=$py_a"
printf "%-12s %s\n" "$LABEL_B" "python=$py_b"
printf "%-12s %s\n" "$LABEL_A" "cupy=$cupy_a"
printf "%-12s %s\n" "$LABEL_B" "cupy=$cupy_b"
printf "%-12s %s\n" "$LABEL_A" "pyscf=$pyscf_a"
printf "%-12s %s\n" "$LABEL_B" "pyscf=$pyscf_b"
printf "%-12s %s\n" "$LABEL_A" "gpu4pyscf=$g4p_a"
printf "%-12s %s\n" "$LABEL_B" "gpu4pyscf=$g4p_b"

echo
echo "[GPU Snapshot]"
printf "%-12s %s\n" "$LABEL_A" "${gpu_a:--}"
printf "%-12s %s\n" "$LABEL_B" "${gpu_b:--}"

echo
echo "[Progress/Throughput]"
printf "%-12s done=%s/%s mean_throughput=%s stru/min\n" "$LABEL_A" "$done_a" "$need_a" "$thr_a"
printf "%-12s done=%s/%s mean_throughput=%s stru/min\n" "$LABEL_B" "$done_b" "$need_b" "$thr_b"

speedup=$(python - <<PY
a = float("$thr_a")
b = float("$thr_b")
if a > 0 and b > 0:
    print(f"{a/b:.3f}")
else:
    print("0")
PY
)

echo
echo "[Conclusion]"
if [ "$speedup" = "0" ]; then
    echo "Insufficient throughput data (need 'stru/min' in both reports)."
else
    python - <<PY
label_a = "$LABEL_A"
label_b = "$LABEL_B"
s = float("$speedup")
if s >= 1.1:
    print(f"{label_a} is faster than {label_b}: {s:.2f}x")
elif s <= 0.9:
    print(f"{label_a} is slower than {label_b}: {1/s:.2f}x (inverse)")
else:
    print(f"Performance is similar: {label_a} vs {label_b} ({s:.2f}x)")
PY
fi

echo
echo "[Hints]"
echo "1) If versions differ (cupy/pyscf/gpu4pyscf), align them first."
echo "2) If GPU line shows low util or low clocks, check power cap/thermal throttling."
echo "3) If throughput differs only in some folders, inspect molecule size mix in those folders."
