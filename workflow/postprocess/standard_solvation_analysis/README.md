# Standard Electrolyte Solvation Analysis Workflow

这个工作流用于标准化地从 MD 轨迹中提取 Li+ 溶剂化结构，并进行量子化学（HOMO-LUMO）计算。

## 目录结构
```text
standard_solvation_analysis/
├── batch_analysis.sh           # 主运行脚本（入口）
├── classify_solvation_env.py   # 结构提取与分类脚本
└── run_gpu4pyscf.py            # GPU 加速量子化学计算脚本
```

## 使用方法

### 1. 配置
在开始之前，请根据您的项目情况修改 `classify_solvation_env.py` 文件顶部的配置区：
- **ANIONS**: 确保列表中包含了您体系中所有的阴离子（如 `["PF6", "TFSI"]`）。
- **ADDITIVE_MAP**: 这是一个字典，用于将文件夹名称映射到添加剂名称。例如：
  ```python
  ADDITIVE_MAP = {
      "test_fec_1.0M": "FEC",
      "test_vc_1.2M": "VC",
      "test_base_electrolyte": "NONE"
  }
  ```

### 2. 运行
您可以在包含数据文件夹（如 `test_fec_1.0M`, `test_vc_1.2M` 等）的任何目录下运行此工作流。

**步骤：**
1. 进入您的数据目录：
   ```bash
   cd /path/to/your/simulation_data
   ```
2. 调用 `batch_analysis.sh`（使用绝对路径或相对路径）：
   ```bash
   # 假设工作流文件夹在上一级目录
   ../standard_solvation_analysis/batch_analysis.sh
   ```

### 3. 输出结果
- **结构提取**: 在每个数据文件夹内生成 `classified_structures/` 目录和对应的 `.tar.gz` 压缩包。
- **计算结果**: 在您当前运行脚本的目录下生成 `all_formulations_homolumo.csv`，包含所有配方的 HOMO/LUMO 能量及能隙数据。

## 依赖环境
- Python 3.8+
- MDAnalysis
- NumPy, Pandas
- GPU4PySCF (及 PySCF)
