#!/usr/bin/env python
# imports
import MDAnalysis as mda
import MDAnalysis.analysis.msd as msd
from MDAnalysis.analysis.msd import EinsteinMSD
from scipy.stats import linregress
import numpy as np
import scienceplots
import matplotlib.pyplot as plt
import mdtraj as md
import os 
from MDAnalysis.transformations import unwrap
import glob 

plt.style.use(['science','no-latex', 'nature'])


paths = ['Li_FSI_1.11_DMC']#, 'lipf6_313_svr', 'lipf6_338_svr']
target_folder = 'res_msd_ipi'
os.makedirs(target_folder, exist_ok=True)
for path in paths:  
    # Load the trajectory for the current index
    traj_filename = path + '/simulation_nve.pos_0.xyz' 
    top_filename = path + '/output_final.pdb'

    u = mda.Universe(top_filename, traj_filename,dt=2)
    u1 = mda.Universe(top_filename)
    u.dimensions = u1.dimensions
    ag = u.atoms  # or a more specific atom group from the selection
    u.trajectory.add_transformations(unwrap(ag))
    MSD = EinsteinMSD(u, select='name Li01', msd_type='xyz', fft=True)
    MSD.run()
    msd =  MSD.results.timeseries/100 # convert to nm^2
    frames = MSD.frames
    timestep = 0.001 
    times = frames*timestep
    # 保存数据到文件（单位：纳米²）
    target_file = f'{target_folder}/msd_{path}'
    np.savetxt(target_file, np.column_stack((times, msd)), header='Time MSD (nm^2)')


def load_msd_data(filename):
    data = np.loadtxt(filename)
    return data[:, 0], data[:, 1]  # 假设第一列是时间，第二列是MSD

def calculate_diffusion_coefficient(time, msd, start_time, end_time):
    # Convert start and end time in ps to frame index
    start_frame = int(start_time/timestep)
    end_frame = int(end_time/timestep)
    # Fit a line to the MSD in the specified time range
    fit = np.polyfit(time[start_frame:end_frame], msd[start_frame:end_frame], 1)
    # Calculate diffusion coefficient from the slope (fit[0]), divided by 6 for 3D diffusion
    diff_coeff = fit[0] / 6.0  # in nm^2/ps
    # Convert from nm^2/ps to m^2/s
    diff_coeff *= 1e-6
    return diff_coeff

fig, ax = plt.subplots(1, 1)

fileslist = glob.glob(f'{target_folder}/msd_*')
fileslist.sort()
for target_file in fileslist:
    path = target_file.split('/')[-1]
    # read MSD file
    time_na, msd_na = load_msd_data(target_file)
    plt.plot(time_na, msd_na, label=f'{path}')

    x1 = 13
    x2 = 33

    # Calculate diffusion coefficients for Li and PF6 ions
    D_na = calculate_diffusion_coefficient(time_na, msd_na, x1, x2)
    print(D_na)

    # plt.plot((time_na[x1],msd_na[x1]),(time_na[x2],msd_na[x2]), label='MSD_fit')
    point1 = (x1,msd_na[int(x1/timestep)])
    point2 = (x2,msd_na[int(x2/timestep)])
    # 计算两点连线的斜率
    plt.plot([point1[0], point2[0]], [point1[1], point2[1]], marker='o', linestyle='--', color='k')  # marker='o'表示在点的位置绘制圆圈标记
    ax.text(0.5, 0.5, 'Diff. coeff.: %.3e m^2/s'%(D_na),
        verticalalignment='center', horizontalalignment='center',
        transform=ax.transAxes)

plt.xlabel('Time (ps)')
plt.ylabel('MSD (nm^2)')
plt.title('Mean MSD vs. Time')
plt.grid(True)
plt.legend()
# Show the plot
plt.savefig(f'{target_folder}/mda_msd_{path}.png', dpi=600, bbox_inches = 'tight')
# plt.show()
