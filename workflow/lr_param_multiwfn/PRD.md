这是一份为您全新升级的 **产品需求文档 (PRD)**。将底层引擎替换为 **PySCF + Horton** 后，这个工具实现了真正的**“纯 Python 化”**，彻底告别了繁琐的文件读写和正则解析。它不仅运行速度极快，而且代码架构将极其优雅。

---

# 产品需求文档 (PRD)：Auto-Multipol (PySCF+Horton 版)

## 1. 产品概述 (Product Overview)
### 1.1 背景 (Background)
获取用于极化力场和长程相互作用（静电、极化、色散）的高精度单分子参数，传统上严重依赖 CamCASP 这样计算昂贵且配置复杂的软件。
通过结合 **PySCF**（顶级开源纯 Python 量子化学库）与 **Horton**（专注于波函数分析与高阶空间划分的 Python 库，特别是其强大的 MBIS 划分），我们可以构建一个**全内存计算、零中间文件 I/O** 的自动化流水线。该流水线利用分子静态电子密度和有效体积模型，能够以极高的效率估算出媲美 CamCASP 级别的分布式多极矩、原子极化率和色散系数。

### 1.2 目标 (Objectives)
开发一个基于纯 Python 生态的命令行工具/依赖库 `Auto-Multipol`，实现从分子坐标输入到量子化学计算，再到 MBIS 空间划分与极化率推导的**端到端全自动参数提取**。

## 2. 用户痛点与使用场景 (User Personas & Scenarios)
*   **目标用户**：极化力场（如 AMOEBA）开发者、分子动力学专家、高通量材料筛选研究员。
*   **使用场景**：用户拥有大量小分子或中等分子的坐标库。希望通过一条简单的 Python 命令或脚本，直接并行跑完所有分子的 PBE0/aug-cc-pVTZ 计算，并立刻获得每个原子的电荷、偶极矩、四极矩、极化率 ($\alpha_i$) 和原子间色散系数 ($C_6$)，直接输出为 JSON 供后续力场拟合程序使用。

## 3. 功能需求 (Functional Requirements)

### 3.1 模块一：PySCF 量子化学核心模块 (Quantum Chemistry Engine)
*   **分子对象构建**：利用 `pyscf.gto` 直接解析 `.xyz` 或 `.mol2` 文件构建内存中的分子对象。
*   **波函数与密度计算**：
    *   调用 `pyscf.dft` 执行高精度密度泛函计算（默认 PBE0，支持自定义）。
    *   配置弥散基组（如 `aug-cc-pVTZ`）并自动处理线性相关性问题。
    *   提取收敛后的全电子密度矩阵 (Density Matrix)。
*   **整体属性计算 (可选)**：调用 PySCF 内置的响应模块，计算分子的总极化率张量，作为后续原子极化率加和的基准参考。

### 3.2 模块二：Horton 空间划分模块 (Density Partitioning)
*   **数据无缝对接**：将 PySCF 生成的分子对象和电子密度，直接在内存中传递给 Horton（或其现代生态如 IOData/Grid）。
*   **MBIS 划分 (Minimal Basis Iterative Stockholder)**：
    *   执行 MBIS 迭代划分（目前替代 ISA 的极佳方案，专门针对极化力场优化）。
    *   提取极高精度的分布式多极矩：**原子电荷 (Charge)、局部偶极矩 (Dipole)、局部四极矩 (Quadrupole)**。
    *   **核心参数提取**：计算并提取每个原子在分子环境中的**有效体积 (Effective Atomic Volume, $V_i^{\text{eff}}$)**。

### 3.3 模块三：长程参数推导模块 (Polarizability & Dispersion Derivation)
*   **原子极化率推导 ($\alpha_i$)**：
    *   内置自由原子的参考极化率 ($\alpha_i^{\text{free}}$) 和参考体积 ($V_i^{\text{free}}$) 数据库。
    *   采用 **Tkatchenko-Scheffler (TS) 比例模型** 或相似的有效体积模型，计算分子中每个原子的极化率：
        $\alpha_i^{\text{mol}} = \alpha_i^{\text{free}} \times \frac{V_i^{\text{eff}}}{V_i^{\text{free}}}$
*   **色散系数推导 ($C_6^{ij}$)**：
    *   基于计算出的原子极化率，结合 Casimir-Polder 积分近似（或 TS 模型的组合规则），自动推导出任意两原子之间的 $C_6$ 色散系数。

### 3.4 模块四：数据结构化与序列化 (Data Output)
*   **无文本解析**：由于全是 Python 对象，完全告别正则表达式抓取。
*   生成层级清晰的字典结构，并序列化为 `.json` 或 `.csv`，包含：
    *   原子索引、元素
    *   Multipoles ($q, \mu_x, \mu_y, \mu_z, \Theta_{xx}, \dots$)
    *   Polarizability ($\alpha_i$)
    *   C6 Matrix (各原子间的色散系数矩阵)

## 4. 非功能需求 (Non-Functional Requirements)
*   **环境依赖**：
    *   完全去中心化：**不需要**安装任何外部独立编译的量化软件（如 Gaussian, ORCA, CamCASP）。
    *   依赖清单：`Python 3.8+`, `pyscf`, `horton` (或 `iodata`/`grid`), `numpy`。
*   **执行性能**：全流程在单个 Python 进程内完成，避免了反复读写几十 MB 的波函数文件，I/O 延迟降至零，极其适合高通量集群计算。

## 5. 交互设计 (Command Line Interface Design)

**基础执行命令示例：**
```bash
python auto_multipol.py molecule.xyz --basis aug-cc-pVTZ --functional PBE0 --output params.json
```

**Python API 调用设计（更推荐的高级用法）：**
```python
from auto_multipol import MoleculeAnalyzer

analyzer = MoleculeAnalyzer(coords="molecule.xyz", basis="aug-cc-pVTZ", xc="pbe0")
results = analyzer.run_mbis_pipeline()

print(results.atomic_charges)
print(results.atomic_polarizabilities)
results.to_json("params.json")
```

## 6. 与 CamCASP 的终极对比（必读声明）

采用此架构前，必须向最终用户（或力场拟合者）明确以下物理模型的本质区别：

| 特性 | CamCASP (ISA-Pol) | Auto-Multipol (PySCF+Horton/MBIS) |
| :--- | :--- | :--- |
| **底层理论** | 严格求解耦合微扰 (CP) 响应方程 | 电子密度体积划分 + 比例经验模型 (如 TS) |
| **分布式多极矩** | ISA (Iterated Stockholder Atoms) | MBIS (Minimal Basis Iterative Stockholder) |
| **极化率类型** | 包含非局域 (Non-local) 极化响应张量 | 仅限于局域 (Local) 的各向同性极化率近似 |
| **色散系数 ($C_6$)** | 虚频动态极化率积分 (极其严格) | 基于极化率的半经验公式推导 |
| **计算速度** | 极慢 (数小时~数天/单分子) | 极快 (主要取决于 PySCF 的 SCF 收敛，几分钟) |
| **易用性/自动化** | 极低，易报错，依赖多语言古老代码 | 极高，纯现代 Python 堆栈，全内存流转 |

**总结：**
`Auto-Multipol (PySCF+Horton)` 放弃了 CamCASP 中极其严苛且昂贵的**非局域响应积分**，转而采用现代力场（如部分 AMOEBA 参数化）广泛认可的 **“MBIS 密度划分 $\rightarrow$ 有效体积 $\rightarrow$ 极化率与色散”** 的物理逻辑。这是一种以极小精度代价换取成百上千倍速度提升的完美工程妥协，是高通量极化力场开发的绝佳利器。