# Standard Solvation Analysis Workflow

用于批量分析电解液 MD 轨迹中的 Li+ 第一溶剂化结构，并对提取簇执行 GPU-DFT 计算（HOMO/LUMO）。

## 1. 功能概览

本流程分两步：

1. 结构提取与分类（`classify_solvation_env.py`）
   - 从 `solvent_salt.pdb + transport_results/nvt.dcd` 中按间隔采样。
   - 以每个 Li+ 为中心，提取第一壳层簇结构。
   - 按是否含阴离子（SSIP/CIP/AGG）与是否含添加剂（with_add/no_add）分类。
2. 量化计算（`run_gpu4pyscf.py`）
   - 读取上一步导出的 `.xyz`。
   - 构建 PySCF 分子对象并调用 GPU4PySCF 进行 DFT。
   - 统计并写出 HOMO/LUMO/Gap 到总表 CSV。

批处理入口脚本：`batch_analysis.sh`。

## 2. 目录结构

```text
standard_solvation_analysis/
├── batch_analysis.sh           # 批处理入口（先提取，再量化）
├── classify_solvation_env.py   # Li+ 壳层结构提取与分类
├── run_gpu4pyscf.py            # GPU4PySCF 计算 HOMO/LUMO
├── ana_openmm_traj_rdf.py      # RDF/CN 分析辅助脚本（可选）
└── README.md
```

## 3. 输入数据约定

在你执行 `batch_analysis.sh` 的当前目录下，应存在若干 `newer*` 文件夹。每个文件夹内至少包含：

```text
newer_xxx/
├── solvent_salt.pdb
└── transport_results/
    └── nvt.dcd
```

## 4. 依赖环境

- Python 3.8+
- MDAnalysis
- NumPy
- PySCF
- GPU4PySCF

建议先在可用 GPU 环境中测试 `import gpu4pyscf` 是否成功。

## 5. 使用前配置

请先修改 `classify_solvation_env.py` 顶部配置区：

1. `ANIONS`
   - 填入体系中所有阴离子残基名，例如：`["PF6", "TFSI", "FSI"]`。
2. `ADDITIVE_MAP`
   - 将文件夹名映射到添加剂残基名，例如：

```python
ADDITIVE_MAP = {
    "test_fec_1.0M": "FEC",
    "test_vc_1.2M": "VC",
    "test_no_add": "NONE",
}
```

若某个文件夹不在 `ADDITIVE_MAP` 中，脚本会自动读取该文件夹下的 `topol.top`，在 `[ molecules ]` 段里选择“以 `A` 开头且数量最少”的分子名作为添加剂；若未找到 `A*` 分子则记为 `NONE`。

3. 采样和截断参数
   - `CUTOFF`：第一壳层距离阈值（单位 Angstrom）。
   - `INTERVAL`：轨迹采样间隔。

## 6. 运行方式

在包含 `newer*` 数据目录的路径下执行：

```bash
/path/to/standard_solvation_analysis/batch_analysis.sh
```

若你有多张 GPU 并希望仅并行执行量化阶段，可使用：

```bash
/path/to/standard_solvation_analysis/batch_gpu4pyscf_multi_gpu.sh
```

可选环境变量：
- `GPU_IDS`：GPU 编号列表，默认 `0,1,2`
- `FOLDER_PATTERN`：目录匹配，默认 `newer*`
- `PYTHON_BIN`：Python 解释器，默认 `python`
- `LOG_DIR`：日志目录，默认 `gpu4pyscf_logs`

脚本会自动：

1. 遍历所有 `newer*` 文件夹；
2. 运行 `classify_solvation_env.py`；
3. 对成功提取的结构运行 `run_gpu4pyscf.py`。

## 7. 输出结果

### 7.1 每个 `newer*` 文件夹内

- `classified_structures/`
  - `SSIP_no_add/`
  - `SSIP_with_add/`
  - `CIP_no_add/`
  - `CIP_with_add/`
  - `AGG_no_add/`
  - `AGG_with_add/`
- `classified_structures.tar.gz`

### 7.2 运行目录下

- `all_formulations_homolumo.csv`
  - 字段：`Folder, Filename, Category, Additive, Charge, HOMO(eV), LUMO(eV), Gap(eV)`

## 8. 可选脚本：RDF 分析

`ana_openmm_traj_rdf.py` 用于额外的 RDF/CN 分析与绘图，不是批处理主流程的一部分。

## 9. 常见问题

1. 提示找不到 PDB 或 DCD
   - 检查每个 `newer*` 目录结构是否符合输入约定。
2. 提示 `gpu4pyscf or pyscf not installed`
   - 确认环境中已安装 PySCF 与 GPU4PySCF，且 Python 路径一致。
3. 没有发现 `newer*` 文件夹
   - 请在数据目录中运行脚本，或修改 `batch_analysis.sh` 中 `FOLDER_PATTERN`。
