from ase.io import read
from ase.visualize.plot import plot_atoms
import matplotlib.pyplot as plt

# Path to the PDB file
file_path = "stage1_fixed_lz_start.pdb"

def save_vis(atoms, rotation, filename, title):
    fig, ax = plt.subplots(figsize=(10, 8))
    # 设置白色背景
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # 绘制原子
    plot_atoms(atoms, ax, radii=0.5, rotation=rotation)
    
    # 移除坐标轴
    ax.set_axis_off()
    
    # 保存图片，确保背景为白色且不透明
    plt.savefig(filename, bbox_inches='tight', dpi=300, facecolor='white', transparent=False)
    plt.close(fig)
    print(f"Saved: {filename} ({title})")

try:
    atoms = read(file_path)

    # 1. 正对着三明治结构的图 (通常是侧视图，看层状堆叠)
    # 假设 Z 轴是堆叠方向，我们从 X 轴方向看过去 (90x, 0y, 0z)
    save_vis(atoms, rotation='90x,0y,0z', filename='vis_sandwich_side.png', title='Side View (Sandwich)')

    # 2. 之前的 45 度视角图
    save_vis(atoms, rotation='45x,45y,0z', filename='vis_isometric.png', title='Isometric View')

    print("\n可视化任务完成！已生成两张白底图片。")

except Exception as e:
    print(f"An error occurred: {e}")
