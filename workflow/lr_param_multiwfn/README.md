# Auto-Multipol 开发与使用文档 (PySCF 版)

`Auto-Multipol` 是一个基于 Python 的自动化工具，用于串联 **PySCF/ORCA** 和 **Multiwfn** 软件，实现从分子结构一键提取原子电荷、原子偶极矩和极化率的功能。

## 1. 环境准备

### 1.1 软件依赖
- **PySCF**: (推荐) 电子结构计算 Python 库。
- **ORCA**: (可选) 电子结构计算程序。
- **Multiwfn**: 波函数分析工具。请确保 `Multiwfn` 可执行文件在系统路径中。
- **Python 3.8+**: 运行脚本及 PySCF 所需。

### 1.2 安装 PySCF
```bash
pip install pyscf
```

## 2. 使用方法

### 2.1 本地一键运行 (PySCF + Multiwfn)
```bash
python auto_multipol.py -i molecule.xyz --driver pyscf --run-local
```

### 2.2 远程计算模式 (解耦模式)
如果你需要在远程集群计算波函数，而在本地分析：
1. **本地生成脚本**:
   ```bash
   python auto_multipol.py -i mols_dir/ --driver pyscf --gen-only --output remote_jobs/
   ```
2. **将 `remote_jobs/` 上传至集群并运行**生成的 `run_pyscf_*.py` 脚本。
3. **计算完成后，将生成的 `.wfn` 文件拉回本地**对应的文件夹中。
4. **本地执行分析**:
   ```bash
   python auto_multipol.py -i mols_dir/ --output remote_jobs/
   ```
   (脚本会自动识别已存在的 `.wfn` 文件并启动 Multiwfn 分析)

### 2.3 高级选项
- `-b, --basis`: 基组（默认: `aug-cc-pVTZ`）。
- `-f, --functional`: 泛函（默认: `PBE0`）。
- `--gpu`: **启用 GPU 加速**。调用 `gpu4pyscf` 库进行 DFT 计算，大幅提升中大规模分子的计算速度。需预先安装 `gpu4pyscf`。

## 3. GPU 加速配置 (可选)
如果您的机器配有 NVIDIA GPU，建议安装 `gpu4pyscf` 以获得数倍至数十倍的加速：
```bash
pip install gpu4pyscf
```
然后在运行脚本时添加 `--gpu` 参数：
```bash
python auto_multipol.py -i molecule.xyz --gpu
```

## 3. 输出说明
- `results.json`: 结构化的原子参数。
- `results.csv`: 可视化表格。
- `output/mol_name/`: 包含中间波函数、计算日志及 Multiwfn 解析日志。

## 4. 后续力场参数化
提取得到的 `results.json/csv` 数据可直接参考 `/home/jmchen/project/PhyNEO/workflow/lr_param` 中的脚本进行格式转换，用于构建极化力场。
建议配合 `convert_mom_to_xml.py` 等工具将提取的偶极矩和极化率转化为 `dmff_forcefield.xml` 格式。
