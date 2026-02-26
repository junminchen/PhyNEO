import os
import sys
import tarfile
import numpy as np
import MDAnalysis as mda

# ==================== 用户配置区 ====================
# 1. 阴离子列表：只要在其中的残基都会被识别为阴离子
#    (请在此处添加您体系中所有的阴离子残基名称)
ANIONS = ["PF6", "TFSI", "FSI", "BF4", "DFOB", "BOB"]

# 2. 文件夹与添加剂的映射表
#    格式: "文件夹名": "添加剂残基名"
#    如果某个文件夹没有添加剂，可以设为 "NONE" 或其他标识
ADDITIVE_MAP = {
    "test_fec_1.0M": "FEC",
    "test_vc_1.2M": "VC",
    "test_dtd_0.8M": "DTD",
    "test_no_add": "NONE",
    # 在此处继续添加您的文件夹映射...
}

# 3. 其他参数
CATION_RES = "LI"
CUTOFF = 3.0       # 第一溶剂化壳层截断半径 (Angstrom)
INTERVAL = 50      # 采样间隔 (越小采样的结构越多)
# ===================================================

def infer_additive_from_topol(target_dir):
    """Infer additive from topol.top: pick A* molecule with the smallest count."""
    topol_file = os.path.join(target_dir, "topol.top")
    if not os.path.exists(topol_file):
        return "UNKNOWN"

    molecules = []
    in_molecules_block = False
    try:
        with open(topol_file, "r") as f:
            for raw_line in f:
                line = raw_line.split(";", 1)[0].strip()
                if not line:
                    continue
                if line.startswith("[") and line.endswith("]"):
                    block = line[1:-1].strip().lower()
                    in_molecules_block = (block == "molecules")
                    continue
                if not in_molecules_block:
                    continue
                if line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                name = parts[0]
                try:
                    count = int(parts[1])
                except ValueError:
                    continue
                molecules.append((name, count))
    except Exception:
        return "UNKNOWN"

    a_molecules = [(name, count) for name, count in molecules if name.startswith("A")]
    if not a_molecules:
        return "NONE"
    a_molecules.sort(key=lambda x: (x[1], x[0]))
    return a_molecules[0][0]

def extract_and_classify(target_dir):
    # 标准化路径并获取文件夹名
    target_dir = target_dir.rstrip('/')
    folder_name = os.path.basename(target_dir)
    
    # 路径检查
    pdb_file = os.path.join(target_dir, "solvent_salt.pdb")
    trj_file = os.path.join(target_dir, "transport_results/nvt.dcd")
    
    if not os.path.exists(pdb_file) or not os.path.exists(trj_file):
        print(f"[Warning] Skipping {folder_name}: PDB or DCD file not found.")
        return

    # 获取当前文件夹对应的添加剂名称
    # 优先用手工映射；若缺失则自动从 topol.top 推断
    current_additive = ADDITIVE_MAP.get(folder_name)
    if current_additive is None:
        current_additive = infer_additive_from_topol(target_dir)
    print(f"Processing: {folder_name} | Additive: {current_additive}")

    # 加载轨迹
    try:
        u = mda.Universe(pdb_file, trj_file)
    except Exception as e:
        print(f"[Error] Failed to load universe for {folder_name}: {e}")
        return

    citations = u.select_atoms(f"resname {CATION_RES}")
    output_dir = os.path.join(target_dir, "classified_structures")

    # 创建输出目录
    categories = [
        "SSIP_no_add",
        "SSIP_with_add",
        "CIP_no_add",
        "CIP_with_add",
        "AGG_no_add",
        "AGG_with_add",
    ]
    for cat in categories:
        os.makedirs(os.path.join(output_dir, cat), exist_ok=True)

    count_stats = {cat: 0 for cat in categories}

    # 遍历轨迹
    for ts in u.trajectory[::INTERVAL]:
        for li in citations:
            # 选区：Li 及其周围 CUTOFF 范围内的原子
            shell = u.select_atoms(f"around {CUTOFF} (index {li.index})", updating=True)
            res_in_shell = shell.residues
            
            # 1. 统计阴离子数量 (支持多种阴离子混合)
            n_anions = 0
            for anion_res in ANIONS:
                n_anions += len(res_in_shell[res_in_shell.resnames == anion_res])
            
            # 2. 统计阳离子数量 (中心Li + 壳层内可能的其他Li)
            n_cations = 1 + len(res_in_shell[res_in_shell.resnames == CATION_RES])
            
            # 3. 判断是否有指定添加剂
            has_additive = False
            if current_additive != "NONE" and current_additive != "UNKNOWN":
                if current_additive in res_in_shell.resnames:
                    has_additive = True
            
            # 4. 分类（与 ana_openmm_traj_rdf.py 口径一致）
            # SSIP: 阴离子数 = 0
            # CIP:  阴离子数 = 1
            # AGG:  阴离子数 >= 2
            if n_anions == 0:
                cat_base = "SSIP"
            elif n_anions == 1:
                cat_base = "CIP"
            else:
                cat_base = "AGG"
            
            cat_suffix = "_with_add" if has_additive else "_no_add"
            category = cat_base + cat_suffix
            
            count_stats[category] += 1

            # 5. 提取簇结构 (Li + 第一壳层)
            cluster = (li.residue.atoms + res_in_shell.atoms).unique
            # 简单周期性处理：无键信息时 unwrap 可能失败，回退到中心平移
            try:
                cluster.unwrap(compound="residues", reference="cog")
                positions = cluster.positions
            except Exception:
                positions = cluster.positions.copy()
                box = u.dimensions[:3]
                center = positions[0]
                positions -= center
                positions -= box * np.round(positions / box)
            
            # 6. 保存 XYZ
            # 文件名包含帧号和原子ID以防重名
            fname = f"frame{ts.frame}_li{li.resid}.xyz"
            save_path = os.path.join(output_dir, category, fname)
            
            with open(save_path, "w") as f:
                f.write(f"{len(cluster)}\n")
                # 注释行关键信息：用于后续电荷计算和归档
                f.write(
                    f"n_cations={n_cations} n_anions={n_anions} "
                    f"additive={current_additive} category={category} folder={folder_name}\n"
                )
                for idx, atom in enumerate(cluster):
                    x, y, z = positions[idx]
                    symbol = atom.type.strip() if atom.type else atom.name.strip()
                    f.write(f"{symbol:<3} {x:>10.5f} {y:>10.5f} {z:>10.5f}\n")

    # 打印统计
    print(f"  -> Stats for {folder_name}: {count_stats}")

    # 打包
    archive_name = os.path.join(target_dir, "classified_structures.tar.gz")
    print(f"  -> Archiving to {archive_name}...")
    with tarfile.open(archive_name, "w:gz") as tar:
        tar.add(output_dir, arcname=os.path.basename(output_dir))
    print("  -> Done.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python classify_solvation_env.py <folder_path>")
        sys.exit(1)
    
    target_folder = sys.argv[1]
    extract_and_classify(target_folder)
