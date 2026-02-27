import os
import sys
import glob
import time
import csv
from pathlib import Path
import numpy as np

# 尝试导入 PySCF 相关库
try:
    from pyscf import gto
    from gpu4pyscf import dft
except ImportError:
    print("[Error] gpu4pyscf or pyscf not installed. Please install them to run this script.")
    sys.exit(1)

# ==================== 计算参数 ====================
BASIS_SET = "6-311++G(d,p)"
FUNCTIONAL = "B3LYP"
OUTPUT_DIR_NAME = "results"
OUTPUT_CSV_TEMPLATE = "{folder}_homolumo.csv"
MAX_STRUCTURES = 100
# =================================================


def load_completed_keys(csv_path):
    completed = set()
    if not csv_path.exists():
        return completed

    try:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                category = (row.get("Category") or "").strip()
                filename = (row.get("Filename") or "").strip()
                if filename:
                    completed.add((category, filename))
    except Exception as e:
        print(f"[Warning] Failed to read existing CSV {csv_path}: {e}")
    return completed

def run_quantum_calculation(target_dir):
    target_dir = target_dir.rstrip('/')
    folder_tag = Path(target_dir).resolve().name
    output_dir = Path(target_dir) / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / OUTPUT_CSV_TEMPLATE.format(folder=folder_tag)

    # 搜索该目录下所有分类后的 xyz 文件（每个分类最多取 MAX_STRUCTURES 个）
    base_search_path = os.path.join(target_dir, "classified_structures")
    category_dirs = sorted(glob.glob(os.path.join(base_search_path, "*")))
    xyz_entries = []
    total_found = 0

    for cat_dir in category_dirs:
        if not os.path.isdir(cat_dir):
            continue
        cat_files = sorted(glob.glob(os.path.join(cat_dir, "*.xyz")))
        if not cat_files:
            continue
        total_found += len(cat_files)
        if len(cat_files) > MAX_STRUCTURES:
            print(
                f"Category {os.path.basename(cat_dir)} has {len(cat_files)} structures. "
                f"Capping to first {MAX_STRUCTURES}."
            )
            cat_files = cat_files[:MAX_STRUCTURES]
        category_name = os.path.basename(cat_dir)
        for xyz_path in cat_files:
            xyz_entries.append((category_name, xyz_path))
    
    if not xyz_entries:
        print(f"[Warning] No XYZ files found in {base_search_path}")
        return

    print(
        f"Found {total_found} structures in {target_dir}. "
        f"Selected {len(xyz_entries)} structures after per-category cap."
    )

    # 如果目标 CSV 不存在，先写表头
    if not output_csv.exists():
        with open(output_csv, "w") as f:
            f.write("Folder,Filename,Category,Additive,Charge,HOMO(eV),LUMO(eV),Gap(eV)\n")
    completed_keys = load_completed_keys(output_csv)
    pending_entries = []
    for category_name, xyz_path in xyz_entries:
        filename = os.path.basename(xyz_path)
        if (category_name, filename) in completed_keys:
            continue
        pending_entries.append((category_name, xyz_path))

    print(f"Writing results to: {output_csv}")
    print(
        f"Resume check: completed={len(completed_keys)} "
        f"pending={len(pending_entries)} total_selected={len(xyz_entries)}"
    )
    if not pending_entries:
        print(f"All selected structures already completed for {target_dir}.")
        print(f"Finished calculation for {target_dir}")
        return

    start_time = time.time()
    for i, (category_name, xyz_path) in enumerate(pending_entries):
        try:
            # 1. 解析 XYZ 文件
            with open(xyz_path, 'r') as f:
                lines = f.readlines()
                
            # 从第一行获取原子数
            n_atoms = int(lines[0].strip())
            
            # 从第二行注释解析元数据 (n_cations, n_anions, additive, etc.)
            comment_line = lines[1].strip()
            # 解析 key=value 对
            meta = {}
            for part in comment_line.split():
                if '=' in part:
                    k, v = part.split('=', 1)
                    meta[k] = v
            
            # 计算电荷: (阳离子数 * +1) + (阴离子数 * -1)
            n_cat = int(meta.get('n_cations', 1))
            n_ani = int(meta.get('n_anions', 0))
            charge = n_cat - n_ani
            
            category = meta.get('category', category_name if category_name else 'unknown')
            additive = meta.get('additive', 'unknown')
            folder_name = meta.get('folder', os.path.basename(target_dir))

            # 提取坐标
            atom_coords = []
            # 从第3行开始读取原子坐标
            for line in lines[2 : 2 + n_atoms]:
                parts = line.split()
                # 格式: AtomType X Y Z
                symbol = parts[0]
                coords = (float(parts[1]), float(parts[2]), float(parts[3]))
                atom_coords.append([symbol, coords])

            # 2. 构建 PySCF 分子对象
            mol = gto.Mole()
            mol.atom = atom_coords
            mol.basis = BASIS_SET
            mol.charge = charge
            mol.spin = 0  # 默认闭壳层单重态
            mol.verbose = 0
            mol.build()

            # 3. 运行 GPU DFT 计算
            # density_fit() 是加速关键
            mf = dft.RKS(mol, xc=FUNCTIONAL).density_fit()
            mf.run()

            # 4. 提取轨道能量 (Hartree -> eV)
            ha_to_ev = 27.211386
            mo_energies = mf.mo_energy
            mo_occ = mf.mo_occ
            
            # 获取 HOMO (最高占据) 和 LUMO (最低未占据)
            # mo_occ > 0 的最后一个是 HOMO
            # mo_occ == 0 的第一个是 LUMO
            if np.any(mo_occ > 0):
                homo_idx = np.where(mo_occ > 0)[0][-1]
                homo_ev = mo_energies[homo_idx] * ha_to_ev
            else:
                homo_ev = np.nan

            if np.any(mo_occ == 0):
                lumo_idx = np.where(mo_occ == 0)[0][0]
                lumo_ev = mo_energies[lumo_idx] * ha_to_ev
            else:
                lumo_ev = np.nan
            
            gap_ev = lumo_ev - homo_ev if (not np.isnan(homo_ev) and not np.isnan(lumo_ev)) else np.nan

            # 5. 写入结果
            fname = os.path.basename(xyz_path)
            with open(output_csv, "a") as f:
                f.write(
                    f"{folder_name},{fname},{category},{additive},{charge},"
                    f"{homo_ev:.4f},{lumo_ev:.4f},{gap_ev:.4f}\n"
                )
            
            # 实时进度信息（百分比 + ETA）
            done = i + 1
            total = len(pending_entries)
            elapsed = time.time() - start_time
            avg = elapsed / done if done > 0 else 0.0
            eta = avg * (total - done)
            pct = 100.0 * done / total if total > 0 else 100.0
            print(
                f"  Progress: {done}/{total} ({pct:6.2f}%) | "
                f"Elapsed: {elapsed/60:7.2f} min | ETA: {eta/60:7.2f} min",
                flush=True,
            )

            # 清理对象释放显存
            del mf
            del mol

        except Exception as e:
            print(f"  [Error] Failed to process {xyz_path}: {e}")
            continue

    print(f"Finished calculation for {target_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_gpu4pyscf.py <folder_path>")
        sys.exit(1)
    
    target_folder = sys.argv[1]
    run_quantum_calculation(target_folder)
