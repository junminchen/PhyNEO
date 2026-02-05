import numpy as np
from ase import Atoms
from ase.lattice.cubic import FaceCenteredCubic
from ase.io import write

def create_porous_bridge(
    box_size_x=40.0, 
    box_size_y=40.0, 
    bridge_length_z=40.0, 
    target_porosity=0.40,  # 目标孔隙率 (0.0 - 1.0)
    num_pores_x=2,         # X方向孔的数量
    num_pores_y=2,         # Y方向孔的数量
    wall_atom='Ar'         # 构成墙壁的原子 (建议用惰性原子)
):
    # 1. 基础检查
    n_pores = num_pores_x * num_pores_y
    area_total = box_size_x * box_size_y
    
    # 2. 计算所需的孔径半径 (R)
    # Area_pores = Porosity * Area_total
    # N * pi * R^2 = Porosity * Lx * Ly
    required_radius = np.sqrt((target_porosity * area_total) / (n_pores * np.pi))
    
    # 检查是否物理上可行 (防止孔太大导致墙壁消失)
    spacing_x = box_size_x / num_pores_x
    spacing_y = box_size_y / num_pores_y
    max_radius = min(spacing_x, spacing_y) / 2.0
    
    print(f"--- Configuration ---")
    print(f"Box Size: {box_size_x} x {box_size_y} A^2")
    print(f"Target Porosity: {target_porosity*100}%")
    print(f"Number of Pores: {num_pores_x} x {num_pores_y} = {n_pores}")
    print(f"Calculated Pore Radius: {required_radius:.2f} A")
    
    if required_radius >= max_radius:
        print(f"WARNING: Porosity too high! Pores will overlap. Max feasible radius is {max_radius:.2f} A")
    
    # 3. 生成实心基底 (Solid Block)
    # 使用FCC晶格保证高密度，防止离子穿墙
    # latticeconstant 设小一点可以让墙壁更致密
    wall = FaceCenteredCubic(symbol=wall_atom, latticeconstant=3.0, size=(15, 15, 15))
    
    # 强制调整 Box 尺寸
    wall.set_cell([box_size_x, box_size_y, bridge_length_z])
    wall.center()
    
    # 4. 挖孔逻辑 (Drilling the holes)
    pos = wall.get_positions()
    atom_indices_to_keep = []
    
    # 定义孔心的网格位置
    # 比如 box=40, 2个孔，孔心应该在 10 和 30
    centers_x = np.linspace(box_size_x/(2*num_pores_x), box_size_x - box_size_x/(2*num_pores_x), num_pores_x)
    centers_y = np.linspace(box_size_y/(2*num_pores_y), box_size_y - box_size_y/(2*num_pores_y), num_pores_y)
    
    # 生成所有孔心的坐标列表
    pore_centers = []
    for cx in centers_x:
        for cy in centers_y:
            pore_centers.append((cx, cy))
            
    # 遍历所有原子，判断是否在任何一个孔内
    for i, p in enumerate(pos):
        x, y = p[0], p[1]
        keep_atom = True
        
        for (cx, cy) in pore_centers:
            # 计算到该孔心的距离 (考虑周期性边界条件是个好习惯，但这里如果是刚性墙通常不需要跨边界挖孔)
            dist_sq = (x - cx)**2 + (y - cy)**2
            if dist_sq < required_radius**2:
                keep_atom = False # 在孔里，删掉
                break
        
        if keep_atom:
            atom_indices_to_keep.append(i)
            
    porous_bridge = wall[atom_indices_to_keep]
    
    return porous_bridge

# --- 执行生成 ---
# 设定你想要的参数
my_bridge = create_porous_bridge(
    box_size_x=40.0,
    box_size_y=40.0,
    bridge_length_z=50.0,  # 盐桥厚度
    target_porosity=0.30,  # 30% 孔隙率
    num_pores_x=2,         # 2x2 = 4个孔
    num_pores_y=2
)

# 导出文件，之后可以与电解液合并
write('porous_bridge_p30.gro', my_bridge) 
print(f"Generated bridge with {len(my_bridge)} atoms.")