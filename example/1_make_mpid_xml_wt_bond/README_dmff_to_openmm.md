# DMFF XML -> OpenMM XML 转换说明

本目录已提供一套统一流程，将 DMFF 力场中的关键段转换为 OpenMM 可直接读取的 XML 结构，并可与基础力场（如 OPLS）自动合并。

## 1. 一步完成（推荐）

```bash
python convert_dmff_to_openmm.py pipeline \
  --dmff phyneo_ecl.xml \
  --base merged_opls.xml \
  --out phyneo_ecl_openmm.xml
```

可选参数：
- `--converted-out phyneo_ecl_converted.xml`：保留中间转换文件。
- `--zero-nonbonded-charges`：将 `<NonbondedForce>` 的原子电荷置零（避免与 MPID 静电重复计入）。
- `--strict`：开启严格校验（默认关闭；遇到对称原子局部坐标映射时可能报歧义）。

## 2. 分两步执行

### Step A: DMFF -> Converted

```bash
python convert_dmff_to_openmm.py convert --dmff phyneo_ecl.xml --out phyneo_ecl_z.xml
```

### Step B: 合并到基础力场

```bash
python convert_dmff_to_openmm.py merge \
  --base merged_opls.xml \
  --converted phyneo_ecl_z.xml \
  --out phyneo_ecl_z_b.xml
```

## 3. 兼容旧脚本

旧脚本仍可使用，但已改为调用新逻辑：

```bash
python 0_modify_admp_to_mpid.py --input phyneo_ecl.xml --output phyneo_ecl_z.xml
python 2_map_opls_atype_to_mpid.py --a merged_opls.xml --b phyneo_ecl_z.xml --out phyneo_ecl_z_b.xml
```

## 4. 当前实现覆盖

- `ADMPPmeForce` -> `MPIDForce`（`Atom` -> `Multipole`，`Polarize` 保留）
- `Slater*` + `ADMPDispPmeForce` -> `CustomNonbondedForce` 的按 type 参数表
- 根据 `(ResidueName, AtomName)` 自动建立 base/converted 的 type 映射
- 自动重映射 `Multipole` 局部坐标中的 `kx/kz/ky` 类型引用
