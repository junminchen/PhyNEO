# MSD 电导率计算器使用示例

## 快速开始

这个工具可以从分子动力学轨迹中计算：
- **阳离子扩散系数** (D+)
- **阴离子扩散系数** (D-)
- **离子电导率** (σ)

## 示例 1: 基本使用

```bash
python calculate_msd_conductivity.py \
    --pdb nvt_init.pdb \
    --traj nvt_ti.pos_0.1.xyz \
    --cation "name Li01" \
    --anion "name P02" \
    --conc 1.0
```

这将计算 Li-PF6 体系的扩散系数和电导率。

## 示例 2: 带图表输出

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

添加 `--plot` 参数会生成 MSD 曲线图。

## 示例 3: Li-FSI 电解质

```bash
python calculate_msd_conductivity.py \
    --pdb init.pdb \
    --traj trajectory.xyz \
    --cation "resname LiA" \
    --anion "resname FSI" \
    --conc 1.5 \
    --plot
```

## 示例 4: 高级选项（高温体系）

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
    --output-prefix high_temp_
```

## 如何选择原子

使用 MDAnalysis 选择语法：

### 按原子名称
```bash
--cation "name Li"
--anion "name P02"
```

### 按残基名称
```bash
--cation "resname LiA"
--anion "resname FSI"
```

### 组合选择
```bash
--cation "resname LiA and name Li01"
```

## 输出文件

1. **diffusion_conductivity_results.txt** - 计算结果摘要
2. **msd_name_Li01.png** - 阳离子 MSD 图（如果使用 --plot）
3. **msd_name_P02.png** - 阴离子 MSD 图（如果使用 --plot）

## 结果示例

```
======================================================================
Results Summary
======================================================================
Input Parameters:
  Topology file: nvt_init.pdb
  Trajectory file: nvt_ti.pos_0.1.xyz
  Cation selection: name Li01
  Anion selection: name P02
  Concentration: 1.0 mol/L
  Temperature: 298 K

Results:
  Cation diffusion coefficient: 2.450000e-11 m²/s (R² = 0.9987)
  Anion diffusion coefficient:  1.820000e-11 m²/s (R² = 0.9982)
  Ionic conductivity: 2.7845 S/m (27.8450 mS/cm)
======================================================================
```

## 注意事项

1. **浓度计算**：确保浓度与模拟盒子匹配
   ```
   浓度 (mol/L) = (离子对数量 / 阿伏伽德罗常数) / 体积 (L)
   ```

2. **平衡时间**：使用 `--start-frame` 跳过平衡阶段（通常 500-1000 帧）

3. **轨迹长度**：更长的轨迹提供更好的统计
   - 最小：100 ps
   - 推荐：1 ns
   - 最佳：10 ns 或更长

4. **拟合质量**：检查 R² 值
   - R² > 0.99：优秀
   - R² > 0.95：良好
   - R² < 0.95：需要更长轨迹或调整拟合区域

## 常见问题

**Q: 如何找到我的原子/残基名称？**
```bash
grep ATOM nvt_init.pdb | head -20
```

**Q: 计算的电导率如何转换单位？**
- S/m → mS/cm: 乘以 10
- 例如：2.7845 S/m = 27.845 mS/cm

**Q: 为什么我的 R² 值很低？**
- 轨迹可能太短
- 系统可能未充分平衡
- 尝试调整 `--fit-start` 和 `--fit-end` 参数

## 获取帮助

```bash
python calculate_msd_conductivity.py --help
```

查看完整文档：
```bash
cat README_MSD_Conductivity.md
```
