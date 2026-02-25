#!/bin/bash

# ================= 批量处理脚本 (Standard Solvation Workflow) =================
# 1. 自动定位脚本所在目录，支持在任意位置运行。
# 2. 遍历当前工作目录下所有 test* 开头的文件夹。
# 3. 调用 classify_solvation_env.py 提取溶剂化结构。
# 4. 调用 run_gpu4pyscf.py 计算 HOMO-LUMO。
# =========================================================================

# 获取脚本所在目录，以便无论在哪里运行都能找到 python 文件
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# 匹配模式，可以根据需要修改，例如 "test*" 或 "run_*"
FOLDER_PATTERN="test*"

echo "========================================="
echo "   Batch Solvation Structure Analysis    "
echo "   Workflow Location: $SCRIPT_DIR        "
echo "========================================="

# 获取符合条件的文件夹列表
DIRS=$(ls -d $FOLDER_PATTERN 2>/dev/null)

if [ -z "$DIRS" ]; then
    echo "[Error] No folders matching '$FOLDER_PATTERN' found in current directory."
    echo "Current directory: $(pwd)"
    exit 1
fi

echo "Found folders:"
echo "$DIRS"
echo "-----------------------------------------"

# 阶段 1: 结构提取与分类
echo ">>> Phase 1: Extracting Structures..."
for dir in $DIRS; do
    if [ -d "$dir" ]; then
        echo "Processing folder: $dir"
        python "$SCRIPT_DIR/classify_solvation_env.py" "$dir"
        echo ""
    fi
done

echo "-----------------------------------------"
echo "Phase 1 Complete."
echo "-----------------------------------------"

# 阶段 2: 量子化学计算
echo ">>> Phase 2: Running GPU4PySCF Calculations..."
for dir in $DIRS; do
    # 只有当存在提取出的结构目录时才进行计算
    if [ -d "$dir/classified_structures" ]; then
        echo "Calculating for folder: $dir"
        python "$SCRIPT_DIR/run_gpu4pyscf.py" "$dir"
        echo ""
    else
        echo "Skipping $dir (No 'classified_structures' found)"
    fi
done

echo "========================================="
echo "All tasks finished."
echo "Results saved to: all_formulations_homolumo.csv"
echo "========================================="
