# LiPF6_EC_DMC_thick_Li_electrode 开发与排障报告

## 1. 任务目标
对 `LiPF6_EC_DMC_thick_Li_electrode` 执行完整模拟流程，排查模拟是否存在问题，并得到可解释的双电层（EDL）结果。

## 2. 环境与执行结论
- 正确运行环境：`conda` 的 `mpid` 环境（含 `openmm 8.4` 与 `packmol`）。
- 初始失败原因：
  - 在 `base` 环境运行时 `packmol: command not found`。
  - `mpid` 环境中分析脚本缺失 `matplotlib`，导致 `analyze_interface_distribution.py` 报错。
- 处理：
  - 切换为 `conda run -n mpid ...` 执行。
  - 安装 `matplotlib` 后分析流程可完整运行。

## 3. 发现的关键建模问题
### 问题 A：界面体系使用各向同性 NPT 导致体积发散
- 现象：原始脚本中 `MonteCarloBarostat` 默认开启，`npt_thick_li.log` 体积持续增长（`272 -> 563 nm^3`），密度持续下降。
- 影响：夹层界面体系被非物理稀释，EDL 统计失真。

### 问题 B：厚电极在分析中使用“全层平均 z”当界面
- 现象：`analyze_interface_distribution.py` 使用厚电极所有原子均值作为电极位置。
- 影响：对于多层电极，界面窗口偏移，吸附积分会系统偏差。

### 问题 C：电极原子在 MD 中可运动，导致电极漂移/PBC 穿越
- 现象：不固定电极时，轨迹中出现电极跨周期边界，界面参考面不稳定。
- 影响：EDL 参考系漂移，分布解释困难。

## 4. 已实施代码修正
### 4.1 `run_openmm84_thick_li_electrode.py`
- 新增 `md.use_barostat` 开关；本案例可切换 NVT/NPT。
- 在日志输出中增加 `Running ensemble: ...`。
- 将电极原子质量设为 `0 dalton`，固定电极几何位置。

### 4.2 `config.json`
- 在 `md` 中新增：`"use_barostat": false`（本界面体系默认走 NVT）。

### 4.3 `analyze_interface_distribution.py`
- 电极界面位置改为“表面层”而非“厚电极层均值”：
  - cathode: `max(z[cathode_atoms])`
  - anode: `min(z[anode_atoms])`

### 4.4 `mc_gap_equilibrate.py`
- gap 定义由均值平面改为表面平面（与厚电极物理含义一致）。

## 5. 实际运行流程（修正后）
- 长程 CPF MD：
  - `--pdb start_with_electrodes.pdb`
  - `--equil-steps 20000`
  - `--prod-steps 100000`
  - `--report-interval 1000`
  - 平台：CUDA
  - 系综：NVT（固定盒长）
- 后处理：
  - `analyze_interface_distribution.py --between-electrodes-only --electrode_margin 1.0 --interface_width 5.0 --nblocks 5`
  - `analyze_interfacial_capacitance.py --skip 2 --nblocks 5`
  - `visualize_last_frame_electrode_charge.py --xy-bins 40`

## 6. 最终结果摘要（当前可复现）
### 6.1 几何与采样
- `frames_used = 120`
- `z_cathode = 16.000 A`
- `z_anode = 74.000 A`
- 分析区间：`[17, 73) A`（界面定位稳定）

### 6.2 电极电荷与守恒
- `Q_cath_mean = +7.09996 e`
- `Q_anode_mean = -7.15116 e`
- `Q_total_mean = -0.05120 e`（总电荷守恒稳定）

### 6.3 界面电容
- `C_interfacial_avg = 0.018851 F/m^2 = 1.885 uF/cm^2`
- block std: `0.0167 uF/cm^2`

### 6.4 EDL 分布特征（5A 界面窗口积分）
- 阴极（正极）界面：
  - `PF6` 明显富集：`0.03829 1/nm^2`
  - 界面净电荷：`-0.03747 e/nm^2`
- 阳极（负极）界面：
  - `Li` 富集较弱：`0.00028 1/nm^2`
  - `PF6` ~ `0`

结论：双电层方向性正确（阴极吸附阴离子、阳极排斥阴离子），但阳极侧 Li 富集仍偏弱，属于当前组成/采样长度下的“弱 EDL”状态。

## 7. 当前产出文件
- 主脚本与配置修改：
  - `run_openmm84_thick_li_electrode.py`
  - `config.json`
  - `analyze_interface_distribution.py`
  - `mc_gap_equilibrate.py`
- 结果文件：
  - `results/interface_distribution.csv`
  - `results/interface_distribution.png`
  - `results/interface_distribution_charge.png`
  - `results/interface_distribution_summary.txt`
  - `results/interfacial_capacitance_summary.txt`
  - `results/last_frame_electrode_charge_maps.png`
  - `results/last_frame_electrode_atom_charges.csv`

## 8. 后续建议（若要更“强”的双电层）
1. 做电压扫描（如 `0.5/1.0/1.5/2.0/3.0 V`）提取微分电容，而不只单点 `2.0 V`。
2. 延长生产段（例如 `>= 300000 steps`）提高 Li 在阳极界面统计显著性。
3. 对 `z_liq_min/z_liq_max` 与 `interface_width` 做敏感性分析，确认界面窗口定义对积分量的影响。
