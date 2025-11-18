# MSD-based Conductivity and Diffusion Coefficient Calculator

[中文版本在下方 | Chinese version below]

## English Version

### Overview

This tool calculates ionic conductivity and diffusion coefficients from molecular dynamics (MD) trajectories using the Mean Square Displacement (MSD) method. It is specifically designed for analyzing electrolyte systems and can calculate:

1. **Cation diffusion coefficient (D+)** - Movement rate of positive ions
2. **Anion diffusion coefficient (D-)** - Movement rate of negative ions  
3. **Ionic conductivity (σ)** - Overall electrical conductivity of the electrolyte

### Theory

#### Mean Square Displacement (MSD)

The MSD measures how much particles have moved from their initial positions over time:

```
MSD(t) = ⟨|r(t) - r(0)|²⟩
```

For 3D diffusion, the Einstein relation relates MSD to the diffusion coefficient:

```
MSD(t) = 6Dt
```

Therefore: **D = slope / 6** from a linear fit of MSD vs. time.

#### Nernst-Einstein Equation

The ionic conductivity is calculated from diffusion coefficients using:

```
σ = (c × NA × e² / (kB × T)) × (D+ + D-)
```

Where:
- σ = ionic conductivity (S/m)
- c = concentration (mol/m³)
- NA = Avogadro's number (6.022×10²³ mol⁻¹)
- e = elementary charge (1.602×10⁻¹⁹ C)
- kB = Boltzmann constant (1.381×10⁻²³ J/K)
- T = temperature (K)
- D+, D- = cation and anion diffusion coefficients (m²/s)

### Installation

Required Python packages:
```bash
pip install numpy scipy matplotlib MDAnalysis
# Optional for better plots:
pip install scienceplots
```

All dependencies should already be satisfied if you have installed the PhyNEO requirements.

### Usage

#### Basic Usage

```bash
python calculate_msd_conductivity.py \
    --pdb init.pdb \
    --traj trajectory.xyz \
    --cation "name Li" \
    --anion "name PF6" \
    --conc 1.0
```

#### With Plotting

```bash
python calculate_msd_conductivity.py \
    --pdb nvt_init.pdb \
    --traj nvt_ti.pos_0.1.xyz \
    --cation "name Li01" \
    --anion "name P02" \
    --conc 1.2 \
    --temp 298 \
    --plot
```

#### Advanced Options

```bash
python calculate_msd_conductivity.py \
    --pdb init.pdb \
    --traj trajectory.xyz \
    --cation "resname LiA" \
    --anion "resname FSI" \
    --conc 1.5 \
    --temp 338 \
    --dt 0.1 \
    --start-frame 1000 \
    --fit-start 0.3 \
    --fit-end 0.7 \
    --plot \
    --output-prefix my_analysis_
```

### Parameters

#### Required Parameters

- `--pdb` : PDB topology file
- `--traj` : Trajectory file (.xyz format from i-Pi)
- `--cation` : MDAnalysis selection string for cations (e.g., `"name Li"`, `"resname LiA"`)
- `--anion` : MDAnalysis selection string for anions (e.g., `"name PF6"`, `"resname FSI"`)
- `--conc` : Electrolyte concentration in mol/L

#### Optional Parameters

- `--temp` : Temperature in Kelvin (default: 298)
- `--dt` : Timestep in picoseconds (default: 0.1)
- `--start-frame` : First frame for analysis, to skip equilibration (default: 500)
- `--fit-start` : Starting fraction of trajectory for linear fit (default: 0.2)
- `--fit-end` : Ending fraction of trajectory for linear fit (default: 0.8)
- `--plot` : Generate MSD plots (flag)
- `--output-prefix` : Prefix for output files (default: '')

### Selection Syntax

The selection strings use [MDAnalysis selection syntax](https://docs.mdanalysis.org/stable/documentation_pages/selections.html):

- By atom name: `"name Li"`, `"name P02"`
- By residue name: `"resname LiA"`, `"resname FSI"`, `"resname PF6"`
- By element: `"element Li"`, `"element P"`
- Combined: `"resname LiA and name Li01"`

### Output Files

1. **diffusion_conductivity_results.txt** - Summary of all results
2. **msd_[selection].png** - MSD plot for each ion type (if `--plot` is used)

### Example Output

```
======================================================================
MSD-based Conductivity and Diffusion Coefficient Calculator
======================================================================

Input files:
  Topology: nvt_init.pdb
  Trajectory: nvt_ti.pos_0.1.xyz

Parameters:
  Cation selection: name Li01
  Anion selection: name P02
  Concentration: 1.0 mol/L
  Temperature: 298 K
  Timestep: 0.1 ps
  Start frame: 500
  Fit range: 20.0% - 80.0% of trajectory

----------------------------------------------------------------------
Loading trajectory...
----------------------------------------------------------------------
Loaded 10000 frames
Total simulation time: 1000.00 ps

----------------------------------------------------------------------
Calculating CATION diffusion coefficient...
----------------------------------------------------------------------
  Selected 64 atoms with selection: name Li01
  Diffusion coefficient: 2.450000e-11 m²/s (R² = 0.9987)
  MSD plot saved to: msd_name_Li01.png

----------------------------------------------------------------------
Calculating ANION diffusion coefficient...
----------------------------------------------------------------------
  Selected 64 atoms with selection: name P02
  Diffusion coefficient: 1.820000e-11 m²/s (R² = 0.9982)
  MSD plot saved to: msd_name_P02.png

----------------------------------------------------------------------
Calculating ionic conductivity...
----------------------------------------------------------------------

Conductivity: 2.7845 S/m (27.8450 mS/cm)

----------------------------------------------------------------------
Results Summary
----------------------------------------------------------------------
Results saved to: diffusion_conductivity_results.txt

======================================================================
Calculation completed successfully!
======================================================================
```

### Tips and Best Practices

1. **Equilibration**: Always skip the equilibration period using `--start-frame`. Typically 500-1000 frames are sufficient.

2. **Linear Fit Region**: The default fit range (20%-80%) usually works well. Adjust if:
   - Early trajectory shows non-linear behavior → increase `--fit-start`
   - Late trajectory has poor statistics → decrease `--fit-end`

3. **Check R² values**: R² > 0.99 indicates good linear fit. Lower values suggest:
   - Insufficient sampling time
   - Need to adjust fit region
   - System not equilibrated

4. **Trajectory Length**: Longer trajectories give better statistics. Aim for:
   - Minimum: 100 ps
   - Good: 1 ns
   - Excellent: 10 ns or more

5. **Concentration**: Must match your simulation box. Calculate as:
   ```
   c (mol/L) = (N_ions / N_A) / V (L)
   ```
   where N_ions is number of cation-anion pairs.

### Troubleshooting

**Problem**: "No atoms selected with: [selection]"
- **Solution**: Check atom/residue names in your PDB file. Use `grep` or visualization software.

**Problem**: R² value is low (< 0.95)
- **Solution**: Increase trajectory length, adjust fit range, or check equilibration.

**Problem**: Conductivity seems unrealistic
- **Solution**: Verify concentration calculation and temperature settings.

**Problem**: "ImportError: No module named MDAnalysis"
- **Solution**: Install required packages: `pip install MDAnalysis`

---

## 中文版本

### 概述

该工具使用均方位移（MSD）方法从分子动力学（MD）轨迹计算离子电导率和扩散系数。专门用于分析电解质体系，可以计算：

1. **阳离子扩散系数 (D+)** - 正离子的移动速率
2. **阴离子扩散系数 (D-)** - 负离子的移动速率
3. **离子电导率 (σ)** - 电解质的整体电导性能

### 理论基础

#### 均方位移（MSD）

MSD 测量粒子随时间从初始位置移动的距离：

```
MSD(t) = ⟨|r(t) - r(0)|²⟩
```

对于三维扩散，Einstein 关系将 MSD 与扩散系数联系起来：

```
MSD(t) = 6Dt
```

因此：从 MSD 对时间的线性拟合得到 **D = 斜率 / 6**。

#### Nernst-Einstein 方程

使用扩散系数计算离子电导率：

```
σ = (c × NA × e² / (kB × T)) × (D+ + D-)
```

其中：
- σ = 离子电导率 (S/m)
- c = 浓度 (mol/m³)
- NA = 阿伏伽德罗常数 (6.022×10²³ mol⁻¹)
- e = 基本电荷 (1.602×10⁻¹⁹ C)
- kB = 玻尔兹曼常数 (1.381×10⁻²³ J/K)
- T = 温度 (K)
- D+, D- = 阳离子和阴离子扩散系数 (m²/s)

### 安装

需要的 Python 包：
```bash
pip install numpy scipy matplotlib MDAnalysis
# 可选，用于更好的图表：
pip install scienceplots
```

如果已安装 PhyNEO 的依赖项，所有依赖都应该满足。

### 使用方法

#### 基本用法

```bash
python calculate_msd_conductivity.py \
    --pdb init.pdb \
    --traj trajectory.xyz \
    --cation "name Li" \
    --anion "name PF6" \
    --conc 1.0
```

#### 带绘图功能

```bash
python calculate_msd_conductivity.py \
    --pdb nvt_init.pdb \
    --traj nvt_ti.pos_0.1.xyz \
    --cation "name Li01" \
    --anion "name P02" \
    --conc 1.2 \
    --temp 298 \
    --plot
```

#### 高级选项

```bash
python calculate_msd_conductivity.py \
    --pdb init.pdb \
    --traj trajectory.xyz \
    --cation "resname LiA" \
    --anion "resname FSI" \
    --conc 1.5 \
    --temp 338 \
    --dt 0.1 \
    --start-frame 1000 \
    --fit-start 0.3 \
    --fit-end 0.7 \
    --plot \
    --output-prefix my_analysis_
```

### 参数说明

#### 必需参数

- `--pdb` : PDB 拓扑文件
- `--traj` : 轨迹文件（来自 i-Pi 的 .xyz 格式）
- `--cation` : 阳离子的 MDAnalysis 选择字符串（例如 `"name Li"`, `"resname LiA"`）
- `--anion` : 阴离子的 MDAnalysis 选择字符串（例如 `"name PF6"`, `"resname FSI"`）
- `--conc` : 电解质浓度，单位 mol/L

#### 可选参数

- `--temp` : 温度，单位 Kelvin（默认：298）
- `--dt` : 时间步长，单位皮秒（默认：0.1）
- `--start-frame` : 分析起始帧，用于跳过平衡过程（默认：500）
- `--fit-start` : 线性拟合起始位置的轨迹分数（默认：0.2）
- `--fit-end` : 线性拟合结束位置的轨迹分数（默认：0.8）
- `--plot` : 生成 MSD 图（标志）
- `--output-prefix` : 输出文件前缀（默认：''）

### 选择语法

选择字符串使用 [MDAnalysis 选择语法](https://docs.mdanalysis.org/stable/documentation_pages/selections.html)：

- 按原子名称：`"name Li"`, `"name P02"`
- 按残基名称：`"resname LiA"`, `"resname FSI"`, `"resname PF6"`
- 按元素：`"element Li"`, `"element P"`
- 组合：`"resname LiA and name Li01"`

### 输出文件

1. **diffusion_conductivity_results.txt** - 所有结果的摘要
2. **msd_[selection].png** - 每种离子类型的 MSD 图（如果使用 `--plot`）

### 输出示例

```
======================================================================
基于 MSD 的电导率和扩散系数计算器
======================================================================

输入文件：
  拓扑文件：nvt_init.pdb
  轨迹文件：nvt_ti.pos_0.1.xyz

参数：
  阳离子选择：name Li01
  阴离子选择：name P02
  浓度：1.0 mol/L
  温度：298 K
  时间步长：0.1 ps
  起始帧：500
  拟合范围：轨迹的 20.0% - 80.0%

----------------------------------------------------------------------
加载轨迹中...
----------------------------------------------------------------------
已加载 10000 帧
总模拟时间：1000.00 ps

----------------------------------------------------------------------
计算阳离子扩散系数...
----------------------------------------------------------------------
  选中 64 个原子，选择条件：name Li01
  扩散系数：2.450000e-11 m²/s (R² = 0.9987)
  MSD 图已保存至：msd_name_Li01.png

----------------------------------------------------------------------
计算阴离子扩散系数...
----------------------------------------------------------------------
  选中 64 个原子，选择条件：name P02
  扩散系数：1.820000e-11 m²/s (R² = 0.9982)
  MSD 图已保存至：msd_name_P02.png

----------------------------------------------------------------------
计算离子电导率...
----------------------------------------------------------------------

电导率：2.7845 S/m (27.8450 mS/cm)

----------------------------------------------------------------------
结果摘要
----------------------------------------------------------------------
结果已保存至：diffusion_conductivity_results.txt

======================================================================
计算成功完成！
======================================================================
```

### 最佳实践建议

1. **平衡过程**：始终使用 `--start-frame` 跳过平衡期。通常 500-1000 帧就足够了。

2. **线性拟合区域**：默认拟合范围（20%-80%）通常效果很好。如需调整：
   - 早期轨迹显示非线性行为 → 增加 `--fit-start`
   - 后期轨迹统计性较差 → 减少 `--fit-end`

3. **检查 R² 值**：R² > 0.99 表示良好的线性拟合。较低的值表明：
   - 采样时间不足
   - 需要调整拟合区域
   - 系统未充分平衡

4. **轨迹长度**：更长的轨迹能提供更好的统计。建议：
   - 最小值：100 ps
   - 良好：1 ns
   - 优秀：10 ns 或更长

5. **浓度**：必须与模拟盒子匹配。计算方式：
   ```
   c (mol/L) = (N_ions / N_A) / V (L)
   ```
   其中 N_ions 是阳离子-阴离子对的数量。

### 故障排除

**问题**："No atoms selected with: [selection]"
- **解决方案**：检查 PDB 文件中的原子/残基名称。使用 `grep` 或可视化软件。

**问题**：R² 值低（< 0.95）
- **解决方案**：增加轨迹长度、调整拟合范围或检查平衡状态。

**问题**：电导率似乎不合理
- **解决方案**：验证浓度计算和温度设置。

**问题**："ImportError: No module named MDAnalysis"
- **解决方案**：安装所需包：`pip install MDAnalysis`

### 引用

如果您在研究中使用此工具，请引用 PhyNEO：

```
CHEN, Junmin; YU, Kuang. PhyNEO: A Neural-Network-Enhanced Physics-Driven 
Force Field Development Workflow for Bulk Organic Molecule and Polymer 
Simulations. Journal of Chemical Theory and Computation, 2023, 20.1: 253-265. 
DOI: 10.1021/acs.jctc.3c01045
```

### 联系方式

如有问题或建议，请访问：https://github.com/junminchen/PhyNEO
