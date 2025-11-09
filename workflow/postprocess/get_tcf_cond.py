#!/usr/bin/env python
from openmm.app import *
from openmm import *
from openmm.unit import *
import random
import numpy as np
import mdtraj as md
import sys
import matplotlib.pyplot as plt
import scienceplots
import os
from multiprocessing import Pool

plt.style.use(['science','no-latex', 'nature'])
random.seed(10)

KB = 1.380649e-23
NA = 6.0221408e23
T = 298 # temperature
e_charge = 1.602e-19
atomic_unit_velocity_to_MperS = 4.5710289e-7 # 2.18769126364e6
atomic_unit_velocity_to_MperS = 10e6 # 2.18769126364e6

# 计算电流自相关函数
def autocorrelation(x):
    x = np.asarray(x, dtype=float)
    result = np.correlate(x, x, mode='full')
    result = result[result.size // 2:]
    # 计算平均自相关值
    n = len(x)
    normalization = np.arange(n, 0, -1, dtype=float)  # 使用浮点数数组
    result /= normalization
    return result

def calculate_current_chunk(na_velocities_chunk, cl_velocities_chunk, na_charge, cl_charge, e_charge, dt):
    current = np.zeros((len(na_velocities_chunk), 3))
    for i in range(len(na_velocities_chunk)):
        current[i] = np.sum(na_charge * e_charge * na_velocities_chunk[i], axis=0) + \
                     np.sum(cl_charge * e_charge * cl_velocities_chunk[i], axis=0)
    return current

def save_cj_file_parallel(traj, top_file, res_name1, res_name2, timestep):
    tr = md.load(top_file)
    na_indices = tr.topology.select(res_name1)
    cl_indices = tr.topology.select(res_name2)
    na_charge = +1  # Na+ ionic charge
    cl_charge = -1  # Cl- ionic charge

    nout = 1
    dt = timestep * 0.001
    nsteps = traj.n_frames

    # 使用多进程来并行计算电流
    na_velocities = []
    cl_velocities = []
    for step in range(nsteps):
        if step % nout == 0:
            velocities = traj.xyz[step]
            na_velocities.append(velocities[na_indices])
            cl_velocities.append(velocities[cl_indices])

    # 将Na+和Cl-离子的速度分成多个块，以便并行处理
    chunk_size = 1000  # 可以根据你的硬件和数据大小调整这个值
    na_velocities_chunks = [na_velocities[i:i + chunk_size] for i in range(0, len(na_velocities), chunk_size)]
    cl_velocities_chunks = [cl_velocities[i:i + chunk_size] for i in range(0, len(cl_velocities), chunk_size)]

    with Pool(processes=8) as pool:  # 使用4个进程，这个数字可以根据你的CPU核心数调整
        current_chunks = pool.starmap(
            calculate_current_chunk,
            [(na_velocities_chunk, cl_velocities_chunk, na_charge, cl_charge, e_charge, dt) for na_velocities_chunk, cl_velocities_chunk in zip(na_velocities_chunks, cl_velocities_chunks)]
        )

    # 将所有子进程计算的电流合并
    current = np.vstack(current_chunks)
    current *= atomic_unit_velocity_to_MperS
    C_J_x = autocorrelation(current[:, 0])  # 计算 x 方向的自相关函数
    C_J_y = autocorrelation(current[:, 1])  # 计算 y 方向的自相关函数
    C_J_z = autocorrelation(current[:, 2])  # 计算 z 方向的自相关函数
    # 将三个方向的结果相加，并除以3
    C_J = (C_J_x + C_J_y + C_J_z) / 3
    # 保存电流数据到txt文件
    np.savetxt(f'res/C_J.txt', C_J)

def save_cj_file(traj, top_file, res_name1, res_name2, timestep):
    traj = md.load(traj_file, top=top_file)
    tr = md.load(top_file)
    na_indices = tr.topology.select(res_name1)
    cl_indices = tr.topology.select(res_name2)
    na_charge = +1  # Na+ ionic charge
    cl_charge = -1  # Cl- ionic charge

    nout = 1
    dt = timestep
    nsteps = traj.n_frames

    # for frame_index, i in enumerate([traj]):
    na_velocities = []
    cl_velocities = []
    for step in range(nsteps):
        if step % nout == 0:
            velocities = traj.xyz[step]
            # state = simulation.context.getState(getVelocities=True)
            # velocities = state.getVelocities(asNumpy=True)
            # 提取并保存Na+和Cl-离子的速度
            na_velocities.append(velocities[na_indices])
            cl_velocities.append(velocities[cl_indices])
    
    # 计算电流
    na_v = np.array(na_velocities)
    cl_v = np.array(cl_velocities)
    current = np.zeros((len(na_v), 3))
    for i in range(len(na_v)):
        # units from nm/ps to m/s
        # current[i] = np.sum(na_v[i], axis=0) #+ np.sum(cl_charge * e_charge * cl_v[i] * 1000, axis=0) 
        # current[i] = np.sum(na_charge * e_charge * na_v[i] * 1000, axis=0) + np.sum(cl_charge * e_charge * cl_v[i] * 1000, axis=0)
        current[i] = np.sum(na_charge * e_charge * na_v[i] * atomic_unit_velocity_to_MperS, axis=0) + \
                    np.sum(cl_charge * e_charge * cl_v[i] * atomic_unit_velocity_to_MperS, axis=0)
    # current *= atomic_unit_velocity_to_MperS
    C_J_x = autocorrelation(current[:, 0])  # 计算 x 方向的自相关函数
    C_J_y = autocorrelation(current[:, 1])  # 计算 y 方向的自相关函数
    C_J_z = autocorrelation(current[:, 2])  # 计算 z 方向的自相关函数
    # 将三个方向的结果相加，并除以3
    C_J = (C_J_x + C_J_y + C_J_z) / 3
    # 保存电流数据到txt文件
    np.savetxt(f'res/C_J.txt', C_J)

def plot_tcf(cj_file):
    fig, ax = plt.subplots(1, 1)
    #calculate conductivity
    # 初始化用于存储所有数据的列表
    all_C_J = []
    # 读取并添加每个文件的数据
    for frame_index in range(1):
        filename = cj_file
        C_J_data = np.loadtxt(filename)
        all_C_J.append(C_J_data)
    # 将列表转换为NumPy数组
    all_C_J_array = np.array(all_C_J)
    # 计算平均C_J
    average_C_J = np.mean(all_C_J_array, axis=0)
    # 绘制图表
    plt.plot(average_C_J)
    plt.xlabel('Time Step')
    plt.ylabel('Average TCF of Current')
    plt.title('Average TCF Over Time')
    plt.grid(True)
    # plt.show()
    plt.savefig(f"res/res_tcf.png", dpi=300, bbox_inches='tight')
    return average_C_J

def plot_t_intg(x1, x2, average_C_J, top_file, dt):
    tr = md.load(top_file)
    volume = tr.unitcell_volumes.mean() * 1e-27

    fig, ax = plt.subplots(1, 1)

    # time_step 是您的模拟中每个数据点的时间间隔（以秒为单位）
    time_step = 1e-15 * dt
    # 计算每个时间点的累积积分
    cumulative_integral = np.cumsum(average_C_J) * time_step
    # 生成对应的时间点数组
    time_points = np.arange(len(average_C_J)) * time_step

    x = time_points

    sigma = (1 / (KB * T * volume)) * np.mean(cumulative_integral[x1:x2]) # S/m 
    sigma *= 10 # 1 mS/cm = 100 mS/m =0.1 S/m
    print('Conductivity:', sigma)
    ax.axvspan(time_points[x1], time_points[x2], alpha=0.1)
    ax.text(0.5, 0.5, 'Conductivity: %.3e mS/cm'%(sigma),
        verticalalignment='center', horizontalalignment='center',
        transform=ax.transAxes)
    
    # 绘制累积积分随时间的变化
    plt.plot(time_points, cumulative_integral)
    plt.xlabel('Time (s)')
    plt.ylabel('Cumulative Integral')
    plt.title('Cumulative Integral of Current Autocorrelation Function')
    plt.grid(True)
    plt.savefig(f"res/res_time_intg.png", dpi=300, bbox_inches = 'tight')

if __name__ == '__main__':
    os.makedirs('res', exist_ok=True)
    # traj_file = 'simulation_nve.vel_0.xyz'
    traj_file = 'svr.vel_1.xyz'

    top_file = 'model.pdb'
    traj = md.load(traj_file, top=top_file)
    # tr = md.load(top_file)

    res_name1 = 'resname LiA'
    res_name2 = 'resname FSI and element N'
    #res_name2 = 'resname PF6 and element P'
    
    #dt = 100 # fs 
    dt = 1 # fs 
    dt = 1 # fs 
    infile = f'res/C_J.txt'
    res_dir = os.path.dirname(infile) 
    if not os.path.exists(res_dir):
        save_cj_file(traj, top_file, res_name1, res_name2, dt)
        # save_cj_file_parallel(traj, top_file, res_name1, res_name2, dt)

    average_C_J = plot_tcf(infile)

    x1 = 4000 # start step
    x2 = 6000 # end step
    #x1, x2 = sys.argv[1:]
     
    plot_t_intg(int(x1), int(x2), average_C_J, top_file, dt)

