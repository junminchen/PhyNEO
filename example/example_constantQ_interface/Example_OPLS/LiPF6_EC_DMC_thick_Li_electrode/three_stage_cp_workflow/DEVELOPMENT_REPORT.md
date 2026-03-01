# DEVELOPMENT REPORT - Three-Stage CP Workflow

## 目标
在独立新目录中实现以下流程：
1. 初态平衡（Neutral Relaxation）：CP 关闭，z-only NPT，收敛 Lz 与电解液密度。
2. CP 激活（CP-Activation）：设定 `DeltaPhi = Phi_R - Phi_L`，固定 Lz 的 NVT。
3. 生产采样（Production）：持续记录 CPF 原子电荷，并统计电极总电荷时间变化。

## 开发记录

### 2026-02-28 / Step 1 - 现有流程梳理
- 检查当前目录已有脚本：
  - `run_xy_npt_pick_nvt_production.py`
  - `run_openmm84_thick_li_electrode.py`
  - `fullfill_geometry_test/run_aniso_npt_density_test.py`
- 结论：现有流程具备 XY-NPT + NVT(CP) 基础，但未按“z-only NPT + CP 激活分段 + 独立原子电荷流水”封装。

### 2026-02-28 / Step 2 - 新目录初始化
- 新建目录：`three_stage_cp_workflow/`
- 创建独立配置：`config_three_stage.json`
- 设计为不修改原目录主流程，避免回归风险。

### 2026-02-28 / Step 3 - 三阶段主脚本实现
- 文件：`run_three_stage_cp_workflow.py`
- 已实现内容：
  - 阶段一：
    - 不添加 CPF（等价 CP 关闭）。
    - 使用 `MonteCarloAnisotropicBarostat(..., scaleX=False, scaleY=False, scaleZ=True)`。
    - 采样 `Lz` 和夹层电解液密度，尾段均值作为固定 `Lz`。
  - 阶段二：
    - 在 NVT 中添加 CPF。
    - 电势设置满足：`DeltaPhi = Phi_R - Phi_L`。
  - 阶段三：
    - 延续 NVT + CPF 生产采样。
    - 实时记录：
      - `electrode_atom_charges.csv`（逐帧逐原子电荷）
      - `electrode_total_charge_timeseries.dat`（总电荷时间序列）

### 2026-02-28 / Step 4 - 统计脚本与运行脚本
- 新增 `analyze_electrode_total_charge.py`：
  - 输出 `Q_left/Q_right/Q_total` 的均值、标准差、漂移。
- 新增 `run_three_stage.sh`：
  - 串联主流程与总电荷统计。

### 2026-02-28 / Step 5 - 文档补全
- 新增 `README.md`：
  - 三阶段定义、运行命令、参数和输出说明。
- 当前文档状态：完成基础版本，可直接用于交接与复现。

### 2026-02-28 / Step 6 - 健壮性与静态检查
- 增加 `config_three_stage.json` 的 `cell` 字段。
- 主脚本新增 fallback：若输入 PDB 缺失周期盒，则从 `config.cell` 自动设置。
- 完成语法检查：`python -m py_compile` 通过。

### 2026-03-01 / Step 7 - 增加 z-number-density 输出
- 新增阶段三 z 分箱统计：
  - 基于电解液残基（`LiA/PF6/ECA/DMC`）质心沿 z 的分布；
  - 输出总数密度与分组数密度（单位 `nm^-3`）。
- 输出文件：`stage3_production/z_number_density_profile.csv`。

## 设计决策说明
1. 使用 `z-only NPT` 而非各向同性 NPT
- 符合夹层界面体系目标，仅调节层间方向（z）密度。
- 避免 x/y 面内尺度变化干扰电极横向结构。

2. 阶段二与阶段三均使用 NVT
- 与需求一致：固定阶段一得到的平均 `Lz`。
- 将“电荷重分布”和“生产采样”逻辑分离，便于后处理对比。

3. 电势差定义显式化
- 输入使用 `DeltaPhi = Phi_R - Phi_L`。
- 实际施加采用中心分解，便于与不同参考电势体系兼容。

## 当前完成度
- [x] 新目录创建
- [x] 三阶段主流程脚本
- [x] CPF 原子电荷实时记录
- [x] 电极总电荷时间序列统计
- [x] README 与开发文档
- [ ] 长程实际产线参数运行（用户可按资源启动）
- [ ] 结果图像化（如需要可新增绘图脚本）

### 2026-03-01 / Step 8 - 阶段一改为准密度匹配
- 问题复盘：`z-only NPT` 在当前固定电极夹层体系中出现盒长发散，导致夹层密度非物理下降。
- 调整方案：
  - 阶段一改为 `XY-NPT, Z-fixed`（`MonteCarloMembraneBarostat`）。
  - 从 bulk `npt.log` 读取 target density。
  - 在阶段一每个采样点计算夹层密度，选取与 target 误差最小的 frame 作为 `stage1_fixed_lz_start.pdb`。
- 新增参数：`--bulk-log`（CLI）与 `bulk_reference.npt_log`（config）。

### 2026-03-01 / Step 9 - 密度优先运行模式
- 根据“先看密度，不先通压”的需求，新增 `--density-only` 运行开关。
- 行为：执行完 Stage1（准密度匹配）后直接停止，不进入 CP-Activation 与 Production。
- 目的：先稳定获得可信初态，再进入电压扫描，减少密度偏差对 EDL 解释的干扰。

## 下一步建议
1. 先用短程步数 smoke test（CPU），确认输出格式。
2. 再切换 CUDA 做正式步数。
3. 根据 `electrode_total_charge_summary.txt` 检查 `Q_total` 漂移是否可接受。
