# Three-Stage Constant-Potential Workflow (LiPF6/EC/DMC + Thick Li Electrodes)

这个目录实现了你指定的三阶段流程，并与当前工程已有建模约定保持一致。

## 三阶段定义

1. 阶段一：`Neutral Relaxation`
- `ConstantPotentialForce` 关闭（不添加 CPF）。
- 系综：`XY-NPT, Z-fixed`，使用 `MonteCarloMembraneBarostat`。
- 目标：以 bulk `npt.log` 的尾段平均密度作为 target，在线计算夹层电解液密度并选取最接近 target 的帧（准密度起点）。

2. 阶段二：`CP-Activation`
- 开启 CPF，并设置电势差：`DeltaPhi = Phi_R - Phi_L`。
- 系综：`NVT`，盒长固定为阶段一尾段平均 `Lz`。
- 目标：电荷在锂表面重分布，形成电双层。

3. 阶段三：`Production`
- 延续阶段二的 `NVT + CPF` 设置。
- 数据采样：
  - 记录 CPF 的实时原子电荷（电极原子逐帧输出）。
  - 记录并统计电极总电荷时间序列（`Q_left`, `Q_right`, `Q_total`）。
  - 输出 `z-number-density`（沿 z 方向的数密度分布，单位 `nm^-3`）。

## 目录文件

- `config_three_stage.json`：三阶段参数配置
- `run_three_stage_cp_workflow.py`：主流程脚本
- `analyze_electrode_total_charge.py`：总电荷时间序列统计
- `run_three_stage.sh`：一键运行（主流程 + 统计）
- `DEVELOPMENT_REPORT.md`：开发过程记录

## 快速运行

在当前目录执行：

```bash
cd three_stage_cp_workflow
bash run_three_stage.sh --platform CUDA
```

如果先做短程测试：

```bash
cd three_stage_cp_workflow
python run_three_stage_cp_workflow.py \
  --platform CPU \
  --neutral-relax-steps 5000 \
  --cp-activation-steps 5000 \
  --production-steps 10000 \
  --report-interval 500
python analyze_electrode_total_charge.py
```

## 关键参数

- `--bulk-log`：bulk NPT 日志路径，用于阶段一 target density
- `--delta-phi-v`：施加电势差，定义为 `Phi_R - Phi_L`
- `--phi-center-v`：电势中心值
- 实际施加关系：
  - `Phi_L = Phi_center - DeltaPhi/2`
  - `Phi_R = Phi_center + DeltaPhi/2`

默认在 `config_three_stage.json` 中：
- `delta_phi_v = 2.0`
- `phi_center_v = 0.0`

## 输出说明

默认输出目录为 `outputs/`：

- `stage1_neutral_relaxation/neutral_relax.log`
- `stage1_neutral_relaxation/neutral_relax.dcd`
- `stage1_neutral_relaxation/stage1_summary.json`
- `stage1_neutral_relaxation/stage1_fixed_lz_start.pdb`
- `stage2_cp_activation/cp_activation.log`
- `stage2_cp_activation/cp_activation.dcd`
- `stage3_production/production.log`
- `stage3_production/production.dcd`
- `stage3_production/production_final.pdb`
- `electrode_atom_charges.csv`（实时电极原子电荷）
- `electrode_total_charge_timeseries.dat`
- `electrode_total_charge_summary.txt`
- `stage3_production/z_number_density_profile.csv`
- `cp_setup_summary.txt`

## 与原目录关系

本目录复用了原工程的输入和力场：
- 输入结构：`../start_with_electrodes.pdb`
- 电极定义：`../electrode_residues.xml`, `../electrode_ff.xml`
- OPLS：`../../opls_salt.xml`, `../../opls_solvent.xml`
