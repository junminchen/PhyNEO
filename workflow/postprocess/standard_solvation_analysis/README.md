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
   - 统计并写出 HOMO/LUMO/Gap 到每个配方目录下的独立 CSV（`results/*.csv`）。

批处理入口脚本：`batch_analysis.sh`。

## 2. 目录结构

```text
standard_solvation_analysis/
├── batch_analysis.sh                  # 批处理入口（先提取，再量化）
├── batch_gpu4pyscf_multi_gpu.sh       # 多 GPU 并行量化
├── monitor_gpu4pyscf_progress.sh      # 按文件夹进度监看（done/need + 进度条）
├── debug_gpu4pyscf_perf.sh            # GPU4PySCF 性能诊断（H20/A800等）
├── compare_gpu4pyscf_reports.sh       # 对比两份 debug 报告
├── classify_solvation_env.py          # Li+ 壳层结构提取与分类
├── run_gpu4pyscf.py                   # GPU4PySCF 计算 HOMO/LUMO（写入每个目录的results）
├── analyze_solvation_results.py       # HOMO/LUMO 可视化与跨配方比较
├── run_visual_analysis.sh             # 可视化分析入口
├── analyze_solvation_shell_dynamics.py# 壳层动力学分析（停留时间/第二壳层）
├── run_shell_dynamics_analysis.sh     # 壳层动力学分析入口
├── ana_openmm_traj_rdf.py             # RDF/CN 分析辅助脚本（可选）
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
   - `MAX_OUTPUT_PER_CATEGORY`：每个分类目录最多导出结构数（默认 100）。

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

运行时会在终端打印全局进度（`completed/total`、`success/fail`、`active`）。
并打印每个文件夹的 `done/need` 进度条快照。

脚本会自动：

1. 遍历所有 `newer*` 文件夹；
2. 运行 `classify_solvation_env.py`；
3. 对成功提取的结构运行 `run_gpu4pyscf.py`。

实时监看建议：

```bash
watch -n 20 /path/to/standard_solvation_analysis/monitor_gpu4pyscf_progress.sh
```

性能诊断（慢任务排查）：

```bash
/path/to/standard_solvation_analysis/debug_gpu4pyscf_perf.sh
```

可选微基准：

```bash
DO_BENCH=1 CUDA_VISIBLE_DEVICES=0 \
/path/to/standard_solvation_analysis/debug_gpu4pyscf_perf.sh
```

跨机器报告对比（例如 H20 vs A800）：

```bash
/path/to/standard_solvation_analysis/compare_gpu4pyscf_reports.sh \
  h20_report.txt a800_report.txt H20 A800
```

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

### 7.2 每个 `newer*` 文件夹内

- `results/<folder>_homolumo.csv`
  - 字段：`Folder, Filename, Category, Additive, Charge, HOMO(eV), LUMO(eV), Gap(eV)`

### 7.3 运行目录下

- `gpu4pyscf_logs/*.log`（多 GPU 运行日志）

## 8. 可选脚本：RDF 分析

`ana_openmm_traj_rdf.py` 用于额外的 RDF/CN 分析与绘图，不是批处理主流程的一部分。

## 9. 可选脚本：可视化与初步统计分析

完成量化计算后，可直接生成“单配方图 + 跨配方对比图 + 初步分析报告”：

```bash
/path/to/standard_solvation_analysis/run_visual_analysis.sh \
  /path/to/your_merged_results.csv analysis_reports
```

主要输出（默认在 `analysis_reports/`）：
- `summary_by_formulation.csv`
- `preliminary_analysis.md`
- `compare_mean_gap_by_formulation.png`
- `compare_gap_distribution_by_formulation.png`
- `compare_category_composition_heatmap.png`
- `per_formulation/*_overview.png`

说明：当前 `run_gpu4pyscf.py` 默认按文件夹分散写出 `results/*.csv`。  
如果要做全局可视化，请先把多个 `results/*.csv` 合并成一个总表后再传给 `run_visual_analysis.sh`。

## 10. 可选脚本：溶剂化壳层动力学对比分析

可直接比较不同配方在第一壳层停留时间和第二壳层占据情况，并输出图表：

```bash
/path/to/standard_solvation_analysis/run_shell_dynamics_analysis.sh
```

常用变量：
- `FOLDER_PATTERN`：默认 `newer*`
- `CATION_SELECTION`：默认 `resname LI`
- `SOLVENT_SELECTION`：默认 `(resname EC EMC DMC FEC DEC PC) and (name O* or type O*)`
- `OUTDIR`：默认 `shell_dynamics_reports`

说明（添加剂模式关键更新）：
- 当 `ANALYSIS_TARGET=additive`（默认）时，脚本会按“添加剂分子”为单位，在全轨迹逐帧跟踪其与任意 Li+ 的最小距离。
- 每个添加剂分子在每一帧被判定为：壳外 / 第一壳层 / 第二壳层，并分别统计第一壳层与第二壳层的连续停留事件（residence events）。
- 该口径适用于“添加剂数量很少（如 2-3 个）”的体系，避免仅看瞬时配位导致的统计偏差。

示例：

```bash
FOLDER_PATTERN="newer*" \
CATION_SELECTION="resname LI" \
SOLVENT_SELECTION="(resname EC EMC DMC FEC DEC PC) and (name O* or type O*)" \
OUTDIR="shell_dynamics_reports" \
/path/to/standard_solvation_analysis/run_shell_dynamics_analysis.sh
```

主要输出：
- `shell_dynamics_reports/shell_dynamics_summary.csv`
- `shell_dynamics_reports/all_residence_events.csv`
- `shell_dynamics_reports/shell_dynamics_report.md`
- `shell_dynamics_reports/compare_*.png`
- `shell_dynamics_reports/per_formulation/*_rdf_shells.png`
- `shell_dynamics_reports/per_formulation/*_residence_hist.png`
- `shell_dynamics_reports/per_formulation/*_residence_events.csv`

`shell_dynamics_summary.csv` 新增（或重点关注）字段：
- `mean_first_shell_residence_ps`, `median_first_shell_residence_ps`
- `mean_second_shell_residence_ps`, `median_second_shell_residence_ps`
- `n_first_shell_events`, `n_second_shell_events`
- `state_fraction_first_shell`, `state_fraction_second_shell`
- `n_target_residues`

新增对比图：
- `compare_mean_residence_time_by_shell.png`
- `compare_residence_time_violin_by_shell.png`

## 11. 常见问题

1. 提示找不到 PDB 或 DCD
   - 检查每个 `newer*` 目录结构是否符合输入约定。
2. 提示 `gpu4pyscf or pyscf not installed`
   - 确认环境中已安装 PySCF 与 GPU4PySCF，且 Python 路径一致。
3. 没有发现 `newer*` 文件夹
   - 请在数据目录中运行脚本，或修改 `batch_analysis.sh` 中 `FOLDER_PATTERN`。

## 12. 配方评价标准（统一口径）

为避免只看单一指标，建议把本流程输出分为三组指标联合评价。

### 12.1 结构组成指标（来自分类结果）

- 数据来源：`classified_structures/*`（或后续汇总表中的 `Category` 字段）
- 关键指标：
  - `SSIP/CIP/AGG` 比例（`AGG` 过高通常不利）
  - `with_add/no_add` 比例（观察添加剂是否真实进入第一壳层）
- 判据建议：
  - 优先选择 `AGG` 占比更低的配方；
  - `SSIP` 与 `CIP` 处于合理平衡，而非单一极端。

### 12.2 电子结构指标（来自 HOMO/LUMO）

- 数据来源：`results/*_homolumo.csv`，或 `analysis_reports/summary_by_formulation.csv`
- 关键指标：
  - `mean_gap`（均值）
  - `std_gap`（离散度）
  - `n_structures`（样本量）
- 判据建议：
  - 在当前脚本口径下（`analyze_solvation_results.py`）按 `mean_gap` 升序比较；
  - 若 `mean_gap` 接近，优先 `std_gap` 更小且 `n_structures` 更大的配方。

### 12.3 壳层动力学指标（来自壳层分析）

- 数据来源：`shell_dynamics_reports/shell_dynamics_summary.csv`
- 关键指标：
  - `mean_residence_ps`（第一壳层平均停留时间）
  - `second_shell_presence_frac`（第二壳层出现分数）
  - `mean_first_cn` / `mean_second_cn`（平均配位数）
- 判据建议：
  - `mean_residence_ps` 不能过低（壳层过于松散）；
  - `second_shell_presence_frac` 不宜过高（过度拥挤）；
  - `mean_first_cn` 建议在合理窗口（常见经验值约 3.5-5.0，需按体系校准）。

## 13. 综合评分与“最合理配方”判定

建议使用加权评分而不是单一排序，默认权重可设为：

- 电子结构分：`35%`
- 壳层动力学分：`30%`
- 结构组成分：`25%`
- 统计稳健分（样本量/覆盖度）：`10%`

可用如下公式（归一化到 `0-100`）：

```text
TotalScore = 0.35 * S_gap + 0.30 * S_dyn + 0.25 * S_struct + 0.10 * S_robust
```

其中：
- `S_gap`：`mean_gap` 越优、`std_gap` 越小得分越高；
- `S_dyn`：`mean_residence_ps` 适中偏高、`second_shell_presence_frac` 更低得分更高；
- `S_struct`：`AGG` 更低、`SSIP/CIP` 分布更合理得分更高；
- `S_robust`：样本量更大、类别覆盖更完整得分更高。

最终按 `TotalScore` 由高到低排序，第一名即“综合最合理配方”。

## 14. 最小输入要求（用于最终排序）

若需要直接给出最终配方排名，至少提供：

1. `analysis_reports/summary_by_formulation.csv`
2. `shell_dynamics_reports/shell_dynamics_summary.csv`

若希望把 `SSIP/CIP/AGG` 显式纳入评分，建议额外提供合并后的全局明细
（含 `Folder, Category` 字段），例如 `all_formulations_homolumo.csv`。
