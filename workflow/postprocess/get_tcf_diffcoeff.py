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

plt.style.use(['science','no-latex', 'nature'])
random.seed(10)

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


def save_cj_file(traj, top_file, res_name1, res_name2, timestep, outfile1, outfile2):
    # traj = md.load(traj_file, top=top_file)
    na_indices = traj.topology.select(res_name1)
    cl_indices = traj.topology.select(res_name2)
    num_anion = int(len(cl_indices)/len(na_indices))
    v_shape = [len(na_indices), num_anion, 3]
    
    na_charge = 0  # Na+ ionic charge

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
            add_vel = np.mean(velocities[cl_indices].reshape(v_shape),axis=1)
            cl_velocities.append(add_vel)
    # 计算电流
    na_v = np.array(na_velocities)
    cl_v = np.array(cl_velocities)
    current = np.zeros((len(na_v), 3))
    current_ = np.zeros((len(na_v), 3))

    for i in range(len(na_v)):
        # units from nm/ps to m/s
        current[i] = np.sum(na_v[i], axis=0) #+ np.sum(cl_charge * e_charge * cl_v[i] * 1000, axis=0) 
        current_[i] = np.sum(cl_v[i], axis=0)
        # current[i] = np.sum(na_charge * e_charge * na_v[i] * 1000, axis=0) #+ np.sum(cl_charge * e_charge * cl_v[i] * 1000, axis=0)

    atomic_unit_velocity_to_MperS = 2.18769126364e6
    current *= atomic_unit_velocity_to_MperS
    current_ *= atomic_unit_velocity_to_MperS

    C_J_x = autocorrelation(current[:, 0])  # 计算 x 方向的自相关函数
    C_J_y = autocorrelation(current[:, 1])  # 计算 y 方向的自相关函数
    C_J_z = autocorrelation(current[:, 2])  # 计算 z 方向的自相关函数
    # 将三个方向的结果相加，并除以3
    C_J = (C_J_x + C_J_y + C_J_z) / 3
    # 保存电流数据到txt文件
    np.savetxt(f'res_msd/{outfile1}', C_J)

    C_J_x = autocorrelation(current_[:, 0])  # 计算 x 方向的自相关函数
    C_J_y = autocorrelation(current_[:, 1])  # 计算 y 方向的自相关函数
    C_J_z = autocorrelation(current_[:, 2])  # 计算 z 方向的自相关函数
    # 将三个方向的结果相加，并除以3
    C_J = (C_J_x + C_J_y + C_J_z) / 3
    # 保存电流数据到txt文件
    np.savetxt(f'res_msd/{outfile2}', C_J)

def plot_tcf(cj_file, outfile):
    fig, ax = plt.subplots(1, 1)
    #calculate conductivity
    # 初始化用于存储所有数据的列表
    all_C_J = []
    # 读取并添加每个文件的数据
    for frame_index in range(1):
        filename = cj_file
        C_J_data = np.loadtxt(f'res_msd/{filename}')
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
    plt.savefig(f"res_msd/{outfile}", dpi=300, bbox_inches = 'tight')
    return average_C_J

def plot_t_intg(x1, x2, average_C_J, outfile):
    fig, ax = plt.subplots(1, 1)

    # time_step 是您的模拟中每个数据点的时间间隔（以秒为单位）
    time_step = 1e-15
    # 计算每个时间点的累积积分
    cumulative_integral = np.cumsum(average_C_J) * time_step
    # 生成对应的时间点数组
    time_points = np.arange(len(average_C_J)) * time_step

    x = time_points

    #calculate diffusion coeff
    D = 1/3 * np.mean(cumulative_integral[x1:x2])
    print('Diff. coeff.:', D)
    ax.axvspan(time_points[x1], time_points[x2], alpha=0.1)
    ax.text(0.5, 0.5, 'Diff. coeff.: %.3e m^2/s'%(D),
        verticalalignment='center', horizontalalignment='center',
        transform=ax.transAxes)
    # 绘制累积积分随时间的变化
    plt.plot(time_points, cumulative_integral)
    plt.xlabel('Time (s)')
    plt.ylabel('Cumulative Integral')
    plt.title('Cumulative Integral of Current Autocorrelation Function')
    plt.grid(True)
    plt.savefig(f"res_msd/{outfile}", dpi=300, bbox_inches = 'tight')

if __name__ == '__main__':
    os.makedirs('res_msd', exist_ok=True)
    traj_file = 'ti.vel_0.xyz'
    top_file = 'model.pdb'
    traj = md.load(traj_file, top=top_file)

    res_name1 = 'resname LiA'
    res_name2 = 'resname FSI'
    res_name2 = 'resname PF6'
    timestep = 0.001
    timestep = 0.05
    outfile1 = 'C_J_cation.txt'
    outfile2 = 'C_J_anion.txt'
    save_cj_file(traj, top_file, res_name1, res_name2, timestep, outfile1, outfile2)

    x1 = 100000
    x2 = 150000

    x1 = 2000
    x2 = 3000

    average_C_J = plot_tcf(outfile1, 'res_tcf_cation.png')
    plot_t_intg(x1, x2, average_C_J, 'res_time_intg_cation.png')

    average_C_J = plot_tcf(outfile2, 'res_tcf_anion.png')
    plot_t_intg(x1, x2, average_C_J, 'res_time_intg_anion.png')
