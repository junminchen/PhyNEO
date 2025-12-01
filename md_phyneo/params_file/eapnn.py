#!/usr/bin/env python

import os
import glob
import json
from functools import partial
from typing import Tuple, Dict, List, Any

import numpy as np
np.seterr(divide='ignore', invalid='ignore')

import jax
import jax.numpy as jnp
from jax import jit, value_and_grad, vmap
import jax.nn.initializers
from jax import config

from flax import linen as nn
from flax.training import train_state
import optax

from dmff.api import Hamiltonian
from dmff.common import nblist
from dmff.utils import jit_condition, regularize_pairs, pair_buffer_scales
from dmff.admp.pairwise import distribute_scalar
from dmff.admp.spatial import pbc_shift

import openmm
from openmm import *
from openmm.app import *
from openmm.unit import *

import pickle
import MDAnalysis as mda
from ase.io import read, write


# Return data needed for fitting
def get_data(data, arr):
    dimer_test = [key for key in data if key.split('_')[-2] in arr and key.split('_')[-1] in arr]
    return dimer_test

def get_topology_neighbors(pdb_file, connectivity=2, max_neighbors=18, max_n_atoms=None):
    mol = mda.Universe(pdb_file)
    n_atoms = len(mol.atoms)
    if max_n_atoms is None:
        max_n_atoms = n_atoms
    if max_neighbors is None:
        max_neighbors = np.max([len(fragments) for fragments in mol.atoms.fragments])
    
    indices = np.full((max_n_atoms, max_neighbors), -1, dtype=np.int32)  # 初始化为无效索引-1
    mask = np.zeros((max_n_atoms, max_neighbors), dtype=np.int32)
    
    try:
        has_bonds = len(mol.bonds) > 0
    except AttributeError:
        has_bonds = False
    
    if has_bonds:
        adj_matrix = np.zeros((n_atoms, n_atoms), dtype=bool)
        for bond in mol.bonds:
            i, j = bond.atoms[0].index, bond.atoms[1].index
            adj_matrix[i, j] = adj_matrix[j, i] = True
        
        # 计算拓扑距离（邻接矩阵幂次）
        adj_matrix_initial = np.copy(adj_matrix)
        adj_matrix_odd = np.copy(adj_matrix)
        adj_matrix_self_even = np.copy(adj_matrix)
        for _ in range(connectivity - 1):
            adj_matrix_self_even = np.dot(adj_matrix_self_even, adj_matrix_self_even)
            adj_matrix = adj_matrix_odd | adj_matrix_self_even
            adj_matrix_odd = np.dot(adj_matrix_self_even, adj_matrix_initial)        
        
        for i in range(n_atoms):
            # 获取所有邻居并排除自身
            neighbors = np.where(adj_matrix[i])[0]
            neighbors = neighbors[neighbors != i]  # 关键：排除自身索引
            
            n_real_neighbors = min(len(neighbors), max_neighbors)
            indices[i, :n_real_neighbors] = neighbors[:n_real_neighbors]
            mask[i, :n_real_neighbors] = 1  # 真实邻居标记为1
            
            # 填充无效索引（-1）而非自身
            if len(neighbors) < max_neighbors:
                indices[i, len(neighbors):] = -1
                mask[i, len(neighbors):] = 0  # 填充位置mask为0
    
    # 单原子体系或填充超过n_atoms的情况（已通过初始化为-1处理，无自身索引）
    
    topo_nblist_data = np.array(indices, dtype=np.int32)
    neighbor_mask = np.array(mask, dtype=np.int32)
    
    return topo_nblist_data, neighbor_mask

@jit_condition(static_argnums=())
@partial(jax.vmap, in_axes=(0, None, None), out_axes=(0, 0, 0, 0))
def get_environment_atoms(pairs, topo_nblist, topo_mask):
    # j_centers = pairs[:, 0]  # 中心原子j的索引（形状：(n_pairs,)）
    # k_centers = pairs[:, 1]  # 中心原子k的索引（形状：(n_pairs,)）
    j_centers = pairs[0]  # 中心原子j的索引（形状：(n_pairs,)）
    k_centers = pairs[1]  # 中心原子k的索引（形状：(n_pairs,)）
    
    # 安全索引：使用jnp.take替代原生数组索引
    j_neighbors = jnp.take(topo_nblist, j_centers, axis=0)  # (n_pairs, max_neighbors)
    k_neighbors = jnp.take(topo_nblist, k_centers, axis=0)  # (n_pairs, max_neighbors)

    # 过滤无效索引（-1 表示无效）
    valid_j = j_neighbors != -1
    valid_k = k_neighbors != -1

    # 创建掩码：排除邻居等于j_centers或k_centers的情况
    mask_j = (j_neighbors != j_centers) & (j_neighbors != k_centers) & valid_j
    mask_k = (k_neighbors != j_centers) & (k_neighbors != k_centers) & valid_k

    topo_mask_j = jnp.take(topo_mask, j_centers, axis=0)  # (n_pairs, max_neighbors)
    topo_mask_k = jnp.take(topo_mask, k_centers, axis=0)  # (n_pairs, max_neighbors)

    # 结合原始topo_mask和自定义掩码
    valid_mask_j = topo_mask_j & mask_j  # 布尔数组按位与
    valid_mask_k = topo_mask_k & mask_k  # 布尔数组按位与

    return j_neighbors, k_neighbors, valid_mask_j, valid_mask_k


import torch
from torch.utils.data import Dataset, DataLoader

class MoleculeTorchDataset(Dataset):
    def __init__(self, ase_structures):
        # self.max_atoms = max(len(structure) for structure in ase_structures)
        self.max_atoms = 60  # Set a fixed maximum number of atoms
        self.z_index = [1, 3, 5, 6, 7, 8, 9, 11, 15, 16]
        
        # Pre-process all structures
        self.structures = ase_structures
        
    def __len__(self):
        return len(self.structures)
    
    def __getitem__(self, idx):
        structure = self.structures[idx]
        n_atoms = len(structure)
        
        # Convert all data to numpy arrays for PyTorch compatibility
        pos = np.pad(structure.get_positions(), 
                     ((0, self.max_atoms - n_atoms), (0, 0)), 
                     mode='constant', constant_values=0)
        
        box = np.array(structure.get_cell())
        
        atomic_nums = np.pad(structure.get_atomic_numbers(), 
                             (0, self.max_atoms - n_atoms), 
                             mode='constant', constant_values=0)
        
        energy = float(structure.get_potential_energy())
        sr_energy = float(structure.info['sr_energy'])
        distance = float(structure.info['distance'])
        
        mask = np.pad(np.ones(n_atoms), 
                      (0, self.max_atoms - n_atoms), 
                      mode='constant', constant_values=0)

        mol_ID = np.pad(structure.get_array('molID'), 
                        (0, self.max_atoms - n_atoms), 
                        mode='constant', constant_values=10000)
        
        # Convert to numpy arrays first for PyTorch compatibility
        pairs = np.array(structure.info['pairs'])

        # topo_mask = np.array(structure.info['topo_mask'])
        # topo_nblist = np.array(structure.info['topo_nblist'])
        # 提取原始的topo_mask和topo_nblist
        orig_topo_mask = np.array(structure.info['topo_mask'])  # 形状: [n_atoms, max_neighbors]
        orig_topo_nblist = np.array(structure.info['topo_nblist'])  # 形状: [n_atoms, max_neighbors]
        
        # 对topo_mask进行padding (二维)
        # 先填充行(原子数不足max_atoms的情况)
        topo_mask = np.pad(orig_topo_mask, 
                        ((0, self.max_atoms - n_atoms), (0, 0)), 
                        mode='constant', constant_values=0)
        
        # 对topo_nblist进行padding (二维)
        # 先填充行，再处理超出max_atoms的邻居索引(设为-1表示无效)
        topo_nblist = np.pad(orig_topo_nblist, 
                            ((0, self.max_atoms - n_atoms), (0, 0)), 
                            mode='constant', constant_values=-1)
        
        # 将超出max_atoms的邻居索引设为-1
        # 注意：如果orig_topo_nblist中已经有超出max_atoms的索引，需要额外处理
        topo_nblist[topo_nblist >= self.max_atoms] = -1

        atypes = np.pad(
            np.array([self.z_index.index(i) for i in structure.get_atomic_numbers()]),
            (0, self.max_atoms - n_atoms),
            mode='constant',
            constant_values=len(self.z_index)  # 用z_index长度作为占位符（有效索引范围外的第一个值）
        )
        
        return {
            'pos': pos.astype(np.float32),
            'box': box.astype(np.float32),
            'atomic_numbers': atomic_nums.astype(np.int32),
            'energy': energy,
            'sr_energy': sr_energy,
            'mask': mask.astype(np.int32),
            'molID': mol_ID.astype(np.int32),
            'pairs': pairs.astype(np.int32),
            'atypes': atypes.astype(np.int32),
            'distance': distance,
            'topo_mask': topo_mask.astype(np.int32),
            'topo_nblist': topo_nblist.astype(np.int32),
        }


# 辅助函数（用户提供）
def parameter_shapes(params):
    return jax.tree_util.tree_map(lambda p: p.shape, params)

@jit_condition(static_argnums=())
@partial(jax.vmap, in_axes=(0, None), out_axes=0)
def cutoff_cosine(distances, cutoff):
    """余弦截止函数（二阶可微）"""
    x = distances / cutoff
    return jnp.where(x < 1, 0.5 * (jnp.cos(jnp.pi * x) + 1), 0.0)
zindex = [1.0, 3.0, 5.0, 6.0, 7.0, 8.0, 9.0, 11.0, 15.0, 16.0]

charge_to_index = { 0.0  : len(zindex),
                    1.0  : 0,
                    3.0  : 1,
                    5.0  : 2,
                    6.0  : 3,
                    7.0  : 4,
                    8.0  : 5,
                    9.0  : 6,
                    11.0 : 7,
                    15.0 : 8,
                    16.0 : 9,
                   }

def int_to_onehot(labels, num_classes: int):
    charges = jnp.array(list(charge_to_index.keys()))
    indices = jnp.array(list(charge_to_index.values()))
    zindex = jnp.take(indices, jnp.searchsorted(charges, labels))
    return jax.nn.one_hot(zindex, num_classes)

class EAPNNForce(nn.Module):
    n_atype: int
    rc: float
    n_atoms: int
    acsf_nmu: int
    apsf_nmu: int
    acsf_eta: float
    apsf_eta: float
    use_pbc: bool = True  # 新增参数，默认使用周期性边界条件

    def setup(self):
        self.feature_extractor = FeatureExtractor(
            n_atoms=self.n_atoms,
            n_atype=self.n_atype, 
            rc=self.rc, 
            acsf_nmu=self.acsf_nmu,
            apsf_nmu=self.apsf_nmu,
            acsf_eta_init=self.acsf_eta,
            apsf_eta_init=self.apsf_eta,
            use_pbc=self.use_pbc  # 新增参数，默认使用周期性边界条件
        )
        self.neural_network = NeuralNetwork()

    def __call__(self, pos, box, pairs, topo_nblist, topo_mask, mol_ID, atype_indices):
        features, dr_norm, buffer_scales = self.feature_extractor(pos, box, pairs, topo_nblist, topo_mask, mol_ID, atype_indices)
        pair_energies = self.neural_network(features, dr_norm, buffer_scales)
        # Apply mask to ensure padded "ghost" pairs contribute exactly 0.0 to the total energy
        masked_energies = pair_energies * buffer_scales
        return jnp.sum(masked_energies)

    def get_features(self, pos, box, pairs, topo_nblist, topo_mask, mol_ID, atype_indices):
        return self.feature_extractor(pos, box, pairs, topo_nblist, topo_mask, mol_ID, atype_indices)

    def get_energy(self, features, dr_norm, buffer_scales):
        pair_energies = self.neural_network(features, dr_norm, buffer_scales)
        # Apply mask to ensure padded "ghost" pairs contribute exactly 0.0 to the total energy
        masked_energies = pair_energies * buffer_scales
        return jnp.sum(masked_energies)

class FeatureExtractor(nn.Module):
    n_atoms: int
    n_atype: int  # 元素种类数（如10种）
    rc: float
    acsf_nmu: int = 20  # ACSF的μ数量
    apsf_nmu: int = 10  # APSF的μ数量
    # 移除固定的acsf_eta和apsf_eta，改为可学习参数的初始值
    acsf_eta_init: float = 100.0
    apsf_eta_init: float = 25.0
    use_pbc: bool = True
    acsf_dim: int = 64  # ACSF特征降维后的维度
    apsf_dim: int = 32  # APSF特征降维后的维度
    def setup(self):
        # 1. ACSF参数：按元素分类的μ和η
        # acsf_mus: (n_atype, acsf_nmu)，初始化为元素无关的均匀分布，再让模型学习
        # self.acsf_mus = self.param(
        #     'acsf_mus',
        #     jax.nn.initializers.uniform(0.0, self.rc),  # 初始在[0, rc]范围内
        #     (self.n_atype, self.acsf_nmu)
        # )
        # 将原初始化器替换为自定义范围的均匀分布
        self.acsf_mus = self.param(
            'acsf_mus',
            # 自定义初始化函数：生成 [minval, maxval] 范围内的均匀分布
            lambda key, shape, dtype=jnp.float32: jax.random.uniform(
                key, shape, dtype, minval=0.0, maxval=self.rc
            ),
            (self.n_atype, self.acsf_nmu)
        )
        # acsf_eta: (n_atype, 1)，控制高斯宽度，用softplus确保正性
        self.acsf_eta = self.param(
            'acsf_eta',
            jax.nn.initializers.constant(jnp.log(jnp.exp(self.acsf_eta_init) - 1)),  # softplus逆初始化
            (self.n_atype, 1)
        )

        # 2. APSF参数：按元素分类的μ和η
        # apsf_mus: (n_atype, apsf_nmu)，角度范围[-1, 1]（对应0~π弧度的cos值）
        self.apsf_mus = self.param(
            'apsf_mus',
            # 自定义初始化函数：生成 [-1.0, 1.0] 范围内的均匀分布
            lambda key, shape, dtype=jnp.float32: jax.random.uniform(
                key, shape, dtype, minval=-1.0, maxval=1.0
            ),
            (self.n_atype, self.apsf_nmu)
        )
        # apsf_eta: (n_atype, 1)，确保正性
        self.apsf_eta = self.param(
            'apsf_eta',
            jax.nn.initializers.constant(jnp.log(jnp.exp(self.apsf_eta_init) - 1)),
            (self.n_atype, 1)
        )
        # 新增：定义降维用的Dense子模块（关键修复）
        self.acsf_dense = nn.Dense(self.acsf_dim)  # ACSF特征降维层
        self.apsf_dense = nn.Dense(self.apsf_dim)  # APSF特征降维层

    def get_init_orb_coeff(self, key, n_atype, n_gto):
        rs = self.rs  # 已初始化的 rs 参数
        coeff = jnp.zeros((n_atype, n_gto))
        for i in range(n_atype):
            for j in range(n_gto):
                # 相邻轨道间距 Δr_s = rs[i,j+1] - rs[i,j]
                if j < n_gto - 1:
                    dr = rs[i, j + 1] - rs[i, j]
                    coeff = coeff.at[i, j].set(1.0 / dr)
                else:
                    coeff = coeff.at[i, j].set(1.0)
        return coeff

    # def compute_atomcenter_features(self, pos, box, topo_nblist, topo_mask, atype_indices, acsf_mus, acsf_eta):
    #     """直接计算所有原子的环境特征"""
    #     # 获取环境原子位置 [n_atoms, max_neighbors, 3]
    #     r_center = pos  # [n_atoms, 3]
    #     r_env = pos[topo_nblist]  # [n_atoms, max_neighbors, 3]
        
    #     # 计算相对位置和距离
    #     dr = r_env - r_center[:, None, :]  # [n_atoms, max_neighbors, 3]
    #     box_inv = jnp.linalg.inv(box)
    #     dr = pbc_shift(dr, box, box_inv)  # 如果使用PBC，这里需要传入box和box_inv
    #     dr_norm = jnp.linalg.norm(dr+1e-10, axis=2)  # [n_atoms, max_neighbors]
        
    #     # 计算截断函数
    #     f_cut = cutoff_cosine(dr_norm, self.rc) * topo_mask  # [n_atoms, max_neighbors]
        
    #     # 计算径向基函数
    #     exp_term = jnp.exp(-acsf_eta * jnp.square(dr_norm[..., None] - acsf_mus))  # [n_atoms, max_neighbors, n_mu]
    #     G_raw = exp_term * f_cut[..., None]  # [n_atoms, max_neighbors, n_mu]
        
    #     # 按原子类型累积特征
    #     type_one_hot = (atype_indices[topo_nblist][..., None] == jnp.arange(self.n_atype))  # [n_atoms, max_neighbors, n_atype]
        
    #     # 一次性计算所有特征
    #     G = jnp.einsum('ijk,ijl->ikl', G_raw, type_one_hot)  # [n_atoms, n_mu, n_atype]
        
    #     return G
    def compute_atomcenter_features(self, pos, box, topo_nblist, topo_mask, atype_indices):
        """直接计算所有原子的环境特征，使用类内部定义的ACSF超参数"""
        # 获取环境原子位置 [n_atoms, max_neighbors, 3]
        r_center = pos  # [n_atoms, 3]
        r_env = pos[topo_nblist]  # [n_atoms, max_neighbors, 3]
        
        # 计算相对位置和距离
        dr = r_env - r_center[:, None, :]  # [n_atoms, max_neighbors, 3]
        if self.use_pbc:  # 仅在启用PBC时进行周期性校正
            box_inv = jnp.linalg.inv(box)
            dr = pbc_shift(dr, box, box_inv)
        # Safe distance calculation: add epsilon inside sqrt for gradient stability
        dr_norm = jnp.sqrt(jnp.sum(dr ** 2, axis=2) + 1e-10)  # [n_atoms, max_neighbors]
        
        # 计算截断函数（结合拓扑掩码过滤无效邻居）
        f_cut = cutoff_cosine(dr_norm, self.rc) * topo_mask  # [n_atoms, max_neighbors]
        
        # 获取当前原子类型的ACSF参数（使用softplus确保eta为正）
        acsf_eta = nn.softplus(self.acsf_eta)  # [n_atype, 1]，转换为正值
        acsf_mus = self.acsf_mus  # [n_atype, acsf_nmu]，从类参数获取
        
        # 按中心原子类型索引获取对应的mus和eta
        center_atype = atype_indices[:, None]  # [n_atoms, 1]
        selected_mus = acsf_mus[center_atype]  # [n_atoms, 1, acsf_nmu]
        selected_eta = acsf_eta[center_atype]  # [n_atoms, 1, 1]
        
        # 计算径向基函数（按中心原子类型的参数）
        exp_term = jnp.exp(-selected_eta * jnp.square(dr_norm[..., None] - selected_mus))  # [n_atoms, max_neighbors, acsf_nmu]
        G_raw = exp_term * f_cut[..., None]  # [n_atoms, max_neighbors, acsf_nmu]
        
        # 按环境原子类型累积特征
        env_atype = atype_indices[topo_nblist]  # [n_atoms, max_neighbors]
        type_one_hot = jax.nn.one_hot(env_atype, self.n_atype)  # [n_atoms, max_neighbors, n_atype]
        
        # 汇总特征：按环境原子类型求和
        G = jnp.einsum('ijk,ijl->ikl', G_raw, type_one_hot)  # [n_atoms, acsf_nmu, n_atype]
        
        return G
    
    def compute_atompair_features(self, cos_gamma_i, cos_gamma_j, j_list, k_list, j_mask, k_mask,
                                buffer_nblist_inter_rc, atype_indices):
        """计算原子对的角度特征（APSF），使用类内部定义的可学习参数"""
        # 获取APSF参数并确保eta为正（通过softplus转换）
        apsf_eta = nn.softplus(self.apsf_eta)  # [n_atype, 1]，确保正值
        apsf_mus = self.apsf_mus  # [n_atype, apsf_nmu]，从类参数获取

        # 获取中心原子对的类型索引（i和j分别为一对原子）
        # 假设pairs隐含在j_list/k_list的第一维，取每个pair的中心原子类型
        pair_atype_i = atype_indices[j_list[:, 0]]  # [n_pairs, ]，i原子类型
        pair_atype_j = atype_indices[k_list[:, 0]]  # [n_pairs, ]，j原子类型

        # 按中心原子类型选择对应的APSF参数
        selected_mus_i = apsf_mus[pair_atype_i]  # [n_pairs, apsf_nmu]
        selected_mus_j = apsf_mus[pair_atype_j]  # [n_pairs, apsf_nmu]
        selected_eta_i = apsf_eta[pair_atype_i]  # [n_pairs, 1]
        selected_eta_j = apsf_eta[pair_atype_j]  # [n_pairs, 1]

        # 计算角度基函数（按中心原子类型的参数）
        angle_features_i = jnp.exp(-selected_eta_i[:, None] * jnp.square(cos_gamma_i[..., None] - selected_mus_i[:, None, :]))
        angle_features_j = jnp.exp(-selected_eta_j[:, None] * jnp.square(cos_gamma_j[..., None] - selected_mus_j[:, None, :]))
        # 形状：[n_pairs, max_neighbors, apsf_nmu]

        # 环境原子类型的独热编码（简化实现）
        env_atype_i = atype_indices[j_list]  # [n_pairs, max_neighbors]
        env_atype_j = atype_indices[k_list]  # [n_pairs, max_neighbors]
        type_one_hot_i = jax.nn.one_hot(env_atype_i, self.n_atype)  # [n_pairs, max_neighbors, n_atype]
        type_one_hot_j = jax.nn.one_hot(env_atype_j, self.n_atype)  # [n_pairs, max_neighbors, n_atype]

        # 应用掩码过滤无效邻居
        masked_features_i = angle_features_i * j_mask[..., None]  # [n_pairs, max_neighbors, apsf_nmu]
        masked_features_j = angle_features_j * k_mask[..., None]  # [n_pairs, max_neighbors, apsf_nmu]

        # 按环境原子类型汇总特征
        G_i = jnp.einsum('ijk,ijl->ikl', masked_features_i, type_one_hot_i)  # [n_pairs, apsf_nmu, n_atype]
        G_j = jnp.einsum('ijk,ijl->ikl', masked_features_j, type_one_hot_j)  # [n_pairs, apsf_nmu, n_atype]

        # 对称平均并应用分子间相互作用掩码
        G = (G_i + G_j) * 0.5 * buffer_nblist_inter_rc[:, None, None]  # 广播掩码到特征维度

        return G

    def __call__(self, pos, box, pairs, topo_nblist, topo_mask, mol_ID, atype_indices):
        """计算分子体系的特征，整合合ACSF和APSF特征并返回"""
        # 使用类内部定义的APSF和ACSF参数
        apsf_mus = self.apsf_mus
        apsf_eta = nn.softplus(self.apsf_eta)  # 确保eta为正
        acsf_mus = self.acsf_mus
        acsf_eta = nn.softplus(self.acsf_eta)  # 确保eta为正

        # 处理键级标度（保持原有逻辑）
        m_scales = jnp.array([0., 0., 0., 0., 0., 1.])
        pairs = pairs.at[:, :2].set(regularize_pairs(pairs[:, :2]))  # 规范化原子对顺序
        nbonds = pairs[:, 2]
        mscales = distribute_scalar(m_scales, nbonds - 1)
        pairs = pairs[:, :2]  # 保留原子对索引

        # 计算缓冲标度和距离相关特征
        buffer_scales = pair_buffer_scales(pairs)
        mscales = mscales * buffer_scales  # 结合缓冲标度的键级标度

        # 处理PBC（根据use_pbc参数决定是否应用周期性校正）
        if self.use_pbc:
            box = box.reshape(3, 3) if box.ndim == 1 else box  # 确保box形状正确
            box_inv = jnp.linalg.inv(box)
            rij = pos[pairs[:, 1]] - pos[pairs[:, 0]]
            rij = pbc_shift(rij, box, box_inv)  # 应用周期性位移
        else:
            rij = pos[pairs[:, 1]] - pos[pairs[:, 0]]  # 直接计算相对位置（无PBC）
            box_inv = jnp.eye(3)  # 无效化box_inv，避免下游错误

        dr_norm = jnp.linalg.norm(rij, axis=1)  # 原子对距离

        # 区分分子内/分子间相互作用
        same_mol = mol_ID[pairs[:, 0]] == mol_ID[pairs[:, 1]]
        buffer_inter = jnp.where(same_mol, 0.0, 1.0)  # 分子间标记
        buffer_intra = jnp.where(same_mol, 1.0, 0.0)  # 分子内标记

        # 余弦截断函数（距离超过rc时特征为0）
        cutoff = 0.5 * (1 + jnp.cos(jnp.pi * dr_norm / self.rc))
        cutoff = jnp.where(dr_norm <= self.rc, cutoff, 0.0)

        # 结合缓冲标度和截断函数的掩码
        buffer_nblist_inter = buffer_inter * buffer_scales
        buffer_nblist_intra = buffer_intra * buffer_scales
        buffer_nblist_inter_rc = buffer_nblist_inter * cutoff

        # 获取环境原子信息（邻居索引和有效性掩码）
        j_list, k_list, j_mask, k_mask = get_environment_atoms(pairs, topo_nblist, topo_mask)

        # 计算i原子的环境角度特征（cos(gamma_i)）
        valid_j_mask = j_mask[..., None]  # 扩展维度用于广播
        rj_env = jnp.where(valid_j_mask, pos[j_list], 0.0)  # 无效邻居位置设为0
        rj_X = rj_env - pos[pairs[:, 0]][:, None, :]  # 环境原子相对i的位置
        if self.use_pbc:
            rj_X = pbc_shift(rj_X, box, box_inv)  # 应用PBC校正
        # Safe norm: add epsilon inside sqrt for gradient stability
        norm_rj_X = jnp.sqrt(jnp.sum(rj_X ** 2, axis=2, keepdims=True) + 1e-10)
        rj_X_norm = rj_X / norm_rj_X  # 单位向量
        rij_unit = rij / jnp.sqrt(jnp.sum(rij ** 2, axis=1, keepdims=True) + 1e-10)  # 原子对单位向量
        cos_gamma_i = jnp.einsum('aji,ai->aj', rj_X_norm, rij_unit) * j_mask  # 点积计算余弦值

        # 计算j原子的环境角度特征（cos(gamma_j)）
        valid_k_mask = k_mask[..., None]
        rk_env = jnp.where(valid_k_mask, pos[k_list], 0.0)
        rk_X = rk_env - pos[pairs[:, 1]][:, None, :]  # 环境原子相对j的位置
        if self.use_pbc:
            rk_X = pbc_shift(rk_X, box, box_inv)  # 应用PBC校正
        # Safe norm: add epsilon inside sqrt for gradient stability
        norm_rk_X = jnp.sqrt(jnp.sum(rk_X ** 2, axis=2, keepdims=True) + 1e-10)
        rk_X_norm = rk_X / norm_rk_X
        rji_unit = -rij_unit  # 反向单位向量
        cos_gamma_j = jnp.einsum('aji,ai->aj', rk_X_norm, rji_unit) * k_mask

        # 计算原子对角度特征（APSF）
        atompair_features = self.compute_atompair_features(
            cos_gamma_i, cos_gamma_j, j_list, k_list, j_mask, k_mask,
            buffer_nblist_inter_rc, atype_indices
        )

        # 计算原子中心特征（ACSF）
        atom_features = self.compute_atomcenter_features(
            pos, box, topo_nblist, topo_mask, atype_indices
        )

        # 对称处理原子对的中心特征
        atom_features_i = atom_features[pairs[:, 0]]
        atom_features_j = atom_features[pairs[:, 1]]
        atom_features = (atom_features_i + atom_features_j) * 0.5  # 对称平均

        # 处理原子类型的独热编码特征
        elem_indices = jnp.array(zindex)[atype_indices]
        j_atype = elem_indices[pairs[:, 0]]  # i原子类型
        k_atype = elem_indices[pairs[:, 1]]  # j原子类型


        # j_onehot = jnp.concatenate([j_atype.reshape(-1, 1), int_to_onehot(j_atype, len(zindex) + 1)], axis=1)
        # k_onehot = jnp.concatenate([k_atype.reshape(-1, 1), int_to_onehot(k_atype, len(zindex) + 1)], axis=1)
        # 优化后：仅保留独热编码
        j_onehot = int_to_onehot(j_atype, len(zindex) + 1)  # 假设int_to_onehot输出独热向量
        k_onehot = int_to_onehot(k_atype, len(zindex) + 1)
        atype_onehot = jnp.concatenate([j_onehot, k_onehot], axis=1)

        # 展平特征并拼接
        atom_features = atom_features.reshape(atom_features.shape[0], -1)
        atompair_features = atompair_features.reshape(atompair_features.shape[0], -1)
        # apsf_features = jnp.concatenate([atom_features, atompair_features, atype_onehot], axis=1)
        distance_features = jnp.stack([
            dr_norm,
            1.0 / (dr_norm + 1e-10),  # 避免除零
            cutoff  # 之前计算的截断函数值
        ], axis=1)  # 形状：[n_pairs, 3]


        # 示例：简单缩放（可根据训练数据统计调整）
        atom_features = atom_features / jnp.max(jnp.abs(atom_features))  # 缩放至[-1,1]
        atompair_features = atompair_features / jnp.max(jnp.abs(atompair_features))


        # 展平特征并降维（使用setup中定义的子模块）
        atom_features = atom_features.reshape(atom_features.shape[0], -1)
        atom_features = self.acsf_dense(atom_features)  # 使用预定义的Dense层
        atom_features = nn.relu(atom_features)

        atompair_features = atompair_features.reshape(atompair_features.shape[0], -1)
        atompair_features = self.apsf_dense(atompair_features)  # 使用预定义的Dense层
        atompair_features = nn.relu(atompair_features)

        # 拼接所有特征
        apsf_features = jnp.concatenate([atom_features, atompair_features, atype_onehot], axis=1)
        # apsf_features = jnp.concatenate([
        #     atom_features, 
        #     atompair_features, 
        #     atype_onehot,
        #     distance_features
        # ], axis=1)
        
        
        return apsf_features, dr_norm, buffer_nblist_inter_rc




@jax.jit
def calculate_center_of_mass(positions, masses):
    total_mass = jnp.sum(masses)
    center_of_mass = jnp.sum(positions * masses[:, None], axis=0) / total_mass
    return center_of_mass

@jax.jit
def calculate_distance_between_centers_of_mass(pos1, masses1, pos2, masses2):
    com1 = calculate_center_of_mass(pos1, masses1)
    com2 = calculate_center_of_mass(pos2, masses2)
    distance = jnp.linalg.norm(com2 - com1)
    return distance    


@jax.jit
def calculate_distance_between_centers_of_mass_vmap(pos1, masses1, pos2, masses2):
    """Vectorized version that calculates distances between centers of mass for batches of positions"""
    return jax.vmap(calculate_distance_between_centers_of_mass, in_axes=(0, None, 0, None))(pos1, masses1, pos2, masses2)



class NeuralNetwork(nn.Module):
    dense_nodes: int = 20
    
    @nn.compact
    def __call__(self, combined, dr_norm, buffer_nblist_inter):
        x = combined
        for _ in range(2):
            x = nn.Dense(self.dense_nodes)(x)
            x = nn.LayerNorm()(x)
            x = nn.relu(x)
        out_AB = nn.Dense(1)(x)
        # Return per-pair energies without summing; mask is applied in EAPNNForce.__call__
        return out_AB.squeeze(-1) 
    
if __name__ == "__main__":
    rc = 6.0
    connectivity = 3
    max_neighbors = 10

    # ff_xml = 'phyneo_ecl_wt_peo_modified_all.xml'
    ff_xml = 'output.2.ABC.solvents.pospenalty.25.LiNa.AexAes.xml'
    pdb = 'dimer_000_DEC_DEC_1.pdb'
    outfile = 'dataset_eapnn/data_all.xyz'

    # 设置力场
    mol = PDBFile(pdb)
    pos = jnp.array(mol.positions._value) * 10
    box = jnp.array(mol.topology.getPeriodicBoxVectors()._value) * 10
    # BOX = jnp.eye(3) * 50

    H = Hamiltonian(ff_xml)
    pots = H.createPotential(mol.topology, nonbondedCutoff=rc*angstrom, nonbondedMethod=CutoffPeriodic, ethresh=1e-4)

    # nbl = nblist.NeighborList(box, 8.0, pots.meta['cov_map'])
    # nbl.capacity_multiplier = 800
    # pairs = nbl.allocate(pos, box)

    nbl = nblist.NoCutoffNeighborList(pots.meta['cov_map'], padding=True)
    nbl.capacity_multiplier = 1000
    nbl.allocate(pos, box)
    pairs = nbl.pairs

    mol_ID = []
    for atom in mol.topology.atoms():
        mol_ID.append(atom.residue.index)
    mol_ID = jnp.array(mol_ID)

    atom_elements = []
    for atom in mol.topology.atoms():
        atom_elements.append(atom.element.atomic_number)
    z_atomnum = jnp.array(atom_elements)

    zindex = [1.0, 3.0, 5.0, 6.0, 7.0, 8.0, 9.0, 11.0, 15.0, 16.0]
    n_atype = len(zindex) + 1
    z_atomnum_list = [float(num) for num in np.array(z_atomnum)]
    zindex_dict = {float(num): i for i, num in enumerate(zindex)}
    atype_indices = jnp.array([zindex_dict.get(num, -1) for num in z_atomnum_list])
    n_atoms = len(pos)
    # 获取拓扑邻居
    topo_nblist, topo_mask = get_topology_neighbors(pdb, connectivity=connectivity, max_neighbors=max_neighbors, max_n_atoms=None)

    model = EAPNNForce(
        n_atoms=n_atoms, 
        n_atype=n_atype, 
        rc=rc,  
        acsf_nmu=43,
        apsf_nmu=21,
        acsf_eta=100,
        apsf_eta=50,
        use_pbc=False,  # 关键修改
    )

    key = jax.random.PRNGKey(0)

    params_init = model.init(key, pos, box, pairs, topo_nblist, topo_mask, mol_ID, atype_indices)
    start_time = time.time()

    params = params_init
    features, dr_norm, buffer_scales = model.apply(params, pos, box, pairs, topo_nblist, topo_mask, mol_ID, atype_indices, method=model.get_features)
    energy = model.apply(params, pos, box, pairs, topo_nblist, topo_mask, mol_ID, atype_indices)
    print(energy)
    end_time = time.time()
    print(f"time cost: {end_time - start_time}s")

    num_runs = 10
    total_time = 0
    for _ in range(num_runs):
        start_time = time.time()
        energy = model.apply(params, pos, box, pairs, topo_nblist, topo_mask, mol_ID, atype_indices)
        end_time = time.time()
        total_time += (end_time - start_time)

    average_time = total_time / num_runs
    print(f"平均计算耗时: {average_time:.4f} 秒")

    data_ase = outfile
    ase_structures = read(data_ase, ':')
    # 提前构建二聚体类型到PDB路径的映射（同之前优化）
    dimer_file_map = {}

    for pdb_path in glob.glob("dimer_bank/*.pdb"):
        filename = os.path.basename(pdb_path)
        parts = filename.split('_')
        monomer_A, monomer_B = parts[-2], parts[-1].split('.')[0]  # 提取单体名称并去除文件扩展名
        dimer_file_map[f"{monomer_A}_{monomer_B}"] = pdb_path
        dimer_file_map[f"{monomer_B}_{monomer_A}"] = pdb_path

    # 填充缓存（在循环外一次性处理所有唯一二聚体）
    unique_dimers = set(structure.info['Comp'].split(':')[0].split('(')[0] + '_' + 
                        structure.info['Comp'].split(':')[1].split('(')[0] 
                        for structure in ase_structures)
    
    nblist_cache = {}
    for dimer in unique_dimers:
        monomer_A, monomer_B = dimer.split('_')
        if dimer not in dimer_file_map:
            continue  # 处理缺失文件（可选）
        
        pdb_path = dimer_file_map[dimer]
        mol = PDBFile(pdb_path)
        box = jnp.eye(3) * 50
        H = Hamiltonian(ff_xml)
        pots = H.createPotential(
            mol.topology,
            nonbondedCutoff=rc*angstrom,
            nonbondedMethod=CutoffPeriodic,
            ethresh=1e-4
        )
        cov_map = pots.meta['cov_map']
        
        # 计算邻居列表（JAX可加速部分）
        pos_dummy = jnp.array(mol.positions._value)  # 虚拟位置用于初始化
        nbl = nblist.NoCutoffNeighborList(pots.meta['cov_map'], padding=True)
        nbl.capacity_multiplier = 800
        pairs = nbl.allocate(pos_dummy, box)   
             
        # 计算拓扑邻居（仅需一次）
        topo_nblist, topo_mask = get_topology_neighbors(pdb_path, connectivity=connectivity, max_neighbors=max_neighbors, max_n_atoms=None)
        
        # 存入缓存
        nblist_cache[dimer] = (pairs, topo_nblist, topo_mask)

    # Print detailed analysis
    print(f"\nDATASET ANALYSIS:")
    print(f"Total structures: {len(ase_structures)}")
    print(f"Found PDB files for: {len(unique_dimers)} dimer types")

    import random
    random.seed(1234)
    random.shuffle(ase_structures)
    train_structures = ase_structures[:int(0.9*len(ase_structures))]
    test_structures = ase_structures[int(0.9*len(ase_structures)):]
    write('test_structures.xyz', test_structures)

    for structure in train_structures:
        comp = structure.info['Comp']
        monomer_A, monomer_B = comp.split(':')
        monomer_A = monomer_A.split('(')[0]
        monomer_B = monomer_B.split('(')[0]
        key = f"{monomer_A}_{monomer_B}"
        
        if key not in nblist_cache:
            raise KeyError(f"Cache miss for dimer type: {key}")
        
        # 从缓存中解包数据
        pairs, topo_nblist, topo_mask = nblist_cache[key]
        
        # 更新邻居列表的位置（使用真实结构的位置）
        # pos = jnp.array(structure.get_positions())
        # nbl = nblist.NoCutoffNeighborList(pairs.cov_map)  # 假设nblist可更新位置
        # nbl.pairs = pairs.update_positions(pos)  # 伪代码，需根据实际库实现调整
        
        structure.info['pairs'] = pairs
        structure.info['topo_nblist'] = topo_nblist
        structure.info['topo_mask'] = topo_mask

    for structure in test_structures:
        comp = structure.info['Comp']
        monomer_A, monomer_B = comp.split(':')
        monomer_A = monomer_A.split('(')[0]
        monomer_B = monomer_B.split('(')[0]
        key = f"{monomer_A}_{monomer_B}"
        
        if key not in nblist_cache:
            raise KeyError(f"Cache miss for dimer type: {key}")
        
        # 从缓存中解包数据
        pairs, topo_nblist, topo_mask = nblist_cache[key]
        
        # 更新邻居列表的位置（使用真实结构的位置）
        # pos = jnp.array(structure.get_positions())
        # nbl = nblist.NoCutoffNeighborList(pairs.cov_map)  # 假设nblist可更新位置
        # nbl.pairs = pairs.update_positions(pos)  # 伪代码，需根据实际库实现调整
        
        structure.info['pairs'] = pairs
        structure.info['topo_nblist'] = topo_nblist
        structure.info['topo_mask'] = topo_mask
            
    # Function to convert PyTorch batch to JAX arrays
    def torch_batch_to_jax(batch):
        jax_batch = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                jax_batch[k] = jnp.array(v.numpy())
            else:
                # Handle lists of tensors if needed
                jax_batch[k] = jnp.array(v)
        return jax_batch

    # Create datasets and dataloaders
    train_dataset = MoleculeTorchDataset(train_structures)
    test_dataset = MoleculeTorchDataset(test_structures)
    
    # print(train_dataset[1]['topo_mask'].shape)
    # Use PyTorch DataLoader with num_workers for parallel loading
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        drop_last=True
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=100,
        shuffle=False,
        drop_last=False
    )

    try:
        batch = next(iter(train_dataloader))
        print("Successfully loaded a batch:")
        for key, value in batch.items():
            print(f"{key}: shape {jnp.array(value).shape}")
    except Exception as e:
        print(f"Error occurred: {e}")


    def create_train_state(model, learning_rate, key):
        params = model.init(key, 
                        jnp.array(batch['pos'][0]), 
                        jnp.array(batch['box'][0]), 
                        jnp.array(batch['pairs'][0]), 
                        jnp.array(batch['topo_nblist'][0]),
                        jnp.array(batch['topo_mask'][0]),
                        jnp.array(batch['molID'][0]),
                        jnp.array(batch['atypes'][0]),)
        tx = optax.adamw(learning_rate)
        return train_state.TrainState.create(
            apply_fn=model.apply, params=params, tx=tx)

    @jax.jit
    def train_step(state, batch):
        def loss_fn(params):
            pred_delta = jax.vmap(state.apply_fn, in_axes=(None, 0, 0, 0, 0, 0, 0, 0))(
                params, 
                batch['pos'], 
                batch['box'], 
                batch['pairs'], 
                batch['topo_nblist'],
                batch['topo_mask'],
                batch['molID'],
                batch['atypes'],
            )            
            true_delta = batch['energy']
            
            # Use optax.huber_loss for better stability (delta=10.0)
            return jnp.mean(optax.huber_loss(pred_delta, true_delta, delta=10.0))
        
        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    key = jax.random.PRNGKey(0)
    model = EAPNNForce(
        n_atoms=n_atoms, 
        n_atype=n_atype, 
        rc=rc,  
        acsf_nmu=43,
        apsf_nmu=21,
        acsf_eta=100,
        apsf_eta=50,
        use_pbc=False,  # 关键修改
    )

    learning_rate = 1e-3

    state = create_train_state(model, learning_rate, key)


    def print_training_progress(
        epoch, num_epochs, loss, loss_history_train, loss_history_test,
        true_energies, predicted_energies, base_energies, distances,
        epoch_time, total_time, figure_size=(13, 4), english_font=None):
        """
        Draw training progress plots with three subplots: loss curve, prediction comparison, and error distribution
        """
        if english_font is None:
            english_font = fm.FontProperties(family='DejaVu Sans', size=12)
        
        # 转换系数
        KJ_TO_KCAL = 0.239006

        # 转换单位
        true_energies = true_energies * KJ_TO_KCAL
        predicted_energies = predicted_energies * KJ_TO_KCAL
        base_energies = base_energies * KJ_TO_KCAL
        loss_history_train = [l * KJ_TO_KCAL for l in loss_history_train]
        loss_history_test = [l * KJ_TO_KCAL for l in loss_history_test]
        loss = loss * KJ_TO_KCAL

        # 计算误差指标
        rmse = np.sqrt(np.mean((true_energies - predicted_energies)**2))
        rmse_base = np.sqrt(np.mean((true_energies - base_energies)**2))
        prediction_errors = predicted_energies - true_energies
        base_errors = base_energies - true_energies
        mean_error = np.mean(prediction_errors)
        std_error = np.std(prediction_errors)

        # 计算平均epoch时间
        avg_epoch_time = total_time / (epoch + 1)

        print(f"Epoch {epoch}/{num_epochs}, Loss: {loss:.4f}, RMSE: {rmse:.4f}, " +
            f"Epoch time: {epoch_time:.2f}s, Avg epoch time: {avg_epoch_time:.2f}s")


    def plot_training_progress(
        epoch, num_epochs, loss, loss_history_train, loss_history_test,
        true_energies, predicted_energies, base_energies, distances,
        epoch_time, total_time, figure_size=(13, 4), english_font=None):
        """
        Draw training progress plots with three subplots: loss curve, prediction comparison, and error distribution
        """
        if english_font is None:
            english_font = fm.FontProperties(family='DejaVu Sans', size=12)
        
        # 转换系数
        KJ_TO_KCAL = 0.239006

        # 转换单位
        true_energies = true_energies * KJ_TO_KCAL
        predicted_energies = predicted_energies * KJ_TO_KCAL
        base_energies = base_energies * KJ_TO_KCAL
        loss_history_train = [l * KJ_TO_KCAL for l in loss_history_train]
        loss_history_test = [l * KJ_TO_KCAL for l in loss_history_test]
        loss = loss * KJ_TO_KCAL

        # 计算误差指标
        rmse = np.sqrt(np.mean((true_energies - predicted_energies)**2))
        rmse_base = np.sqrt(np.mean((true_energies - base_energies)**2))
        prediction_errors = predicted_energies - true_energies
        base_errors = base_energies - true_energies
        mean_error = np.mean(prediction_errors)
        std_error = np.std(prediction_errors)

        # 计算平均epoch时间
        avg_epoch_time = total_time / (epoch + 1)

        # 创建图表
        fig, axs = plt.subplots(1, 3, figsize=figure_size)

        # 1. 损失下降曲线
        ax1 = axs[0]
        ax1.plot(range(1, len(loss_history_train) + 1), loss_history_train, marker='s', color='tomato',
                linewidth=2, markersize=5, markeredgecolor='k', markeredgewidth=0.8, label='Training')
        ax1.set_xlabel('Epochs', fontproperties=english_font, labelpad=8)
        ax1.set_ylabel('Training Loss (kcal/mol)', color='tomato', fontproperties=english_font, labelpad=8)
        ax1.set_title(f"Training Avg Epoch time: {avg_epoch_time:.1f}s | Total: {total_time/60:.1f}min",
                    fontproperties=english_font, pad=10)
        ax1.tick_params(axis='y', labelcolor='tomato')
        ax1.tick_params(axis='both', which='major',
                    length=6, direction='out', width=1.2,
                    bottom=True, top=False, left=True, right=False)
        ax1.tick_params(axis='both', which='minor',
                    length=3, direction='out', width=1.0,
                    bottom=True, top=False, left=True, right=False)

        # 创建次要y轴
        ax2 = ax1.twinx()
        ax2.plot(range(1, len(loss_history_test) + 1), loss_history_test, marker='o', color='cornflowerblue',
                linewidth=2, markersize=5, markeredgecolor='k', markeredgewidth=0.8, label='Testing')
        ax2.set_ylabel('Testing Loss (kcal/mol)', color='cornflowerblue', fontproperties=english_font, labelpad=8)
        ax2.tick_params(axis='y', labelcolor='cornflowerblue')
        ax2.tick_params(axis='both', which='major',
                    length=6, direction='out', width=1.2,
                    bottom=True, top=False, left=False, right=True)
        ax2.tick_params(axis='both', which='minor',
                    length=3, direction='out', width=1.0,
                    bottom=True, top=False, left=False, right=True)

        # 添加次要刻度
        for ax in [ax1, ax2]:
            ax.yaxis.set_major_locator(ticker.MaxNLocator(6))
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax1.xaxis.set_major_locator(ticker.MaxNLocator(6))
        ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))

        # 添加网格和图例
        ax1.grid(True, linestyle='--', alpha=0.7)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', prop=english_font)

        if loss_history_test:
            ax1.text(0.05, 0.95, f'Current loss: {loss_history_test[-1]:.4f}',
                    transform=ax1.transAxes, fontproperties=english_font, verticalalignment='top')

        # 2. 预测值与真实值的对角线图
        ax = axs[1]
        sc2 = ax.scatter(true_energies, base_energies, alpha=0.4,
                        label='w/o corr', color='lightgray', edgecolor='k', linewidth=0.5)
        sc1 = ax.scatter(true_energies, predicted_energies, alpha=0.7,
                        label='w/ corr', color='tab:blue', edgecolor='k', linewidth=0.5)

        min_val = min(np.min(true_energies), np.min(predicted_energies))
        max_val = max(np.max(true_energies), np.max(predicted_energies))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

        ax.set_xlabel("True Energy (kcal/mol)", fontproperties=english_font, labelpad=8)
        ax.set_ylabel("Predicted Energy (kcal/mol)", fontproperties=english_font, labelpad=8)
        ax.set_title("Parity Plot: Predicted vs True Energy", fontproperties=english_font, pad=10)

        ax.tick_params(axis='both', which='major',
                    length=6, direction='out', width=1.2,
                    bottom=True, top=False, left=True, right=False)
        ax.tick_params(axis='both', which='minor',
                    length=3, direction='out', width=1.0,
                    bottom=True, top=False, left=True, right=False)

        ax.yaxis.set_major_locator(ticker.MaxNLocator(6))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.xaxis.set_major_locator(ticker.MaxNLocator(6))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))

        ax.text(0.05, 0.95, f'RMSE w/ corr: {rmse:.4f}',
            transform=ax.transAxes, fontproperties=english_font, verticalalignment='top')
        ax.text(0.05, 0.90, f'RMSE w/o corr: {rmse_base:.4f}',
            transform=ax.transAxes, fontproperties=english_font, verticalalignment='top')
        ax.legend(loc='lower right', prop=english_font)

        # 3. 预测误差与距离的关系图
        ax = axs[2]
        sc_base = ax.scatter(distances, base_errors, alpha=0.4,
                            color='red', edgecolor='darkred', linewidth=0.5,
                            marker='^', label='w/o corr')
        sc = ax.scatter(distances, prediction_errors, alpha=0.7,
                    c=np.abs(prediction_errors), cmap='Blues',
                    edgecolor='k', linewidth=0.5, marker='o', label='w/ corr')
        ax.axhline(y=0, color='r', linestyle='--', lw=2)

        ax.set_xlabel("Distance", fontproperties=english_font, labelpad=8)
        ax.set_ylabel("Prediction Error (kcal/mol)", fontproperties=english_font, labelpad=8)
        ax.set_title("Error Distribution vs Distance", fontproperties=english_font, pad=10)

        ax.tick_params(axis='both', which='major',
                    length=6, direction='out', width=1.2,
                    bottom=True, top=False, left=True, right=False)
        ax.tick_params(axis='both', which='minor',
                    length=3, direction='out', width=1.0,
                    bottom=True, top=False, left=True, right=False)

        ax.yaxis.set_major_locator(ticker.MaxNLocator(6))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.xaxis.set_major_locator(ticker.MaxNLocator(6))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))

        ax.legend(loc='lower right', prop=english_font)
        ax.text(0.05, 0.95, f'Mean error: {mean_error:.4f}',
            transform=ax.transAxes, fontproperties=english_font, verticalalignment='top')
        ax.text(0.05, 0.90, f'Std error: {std_error:.4f}',
            transform=ax.transAxes, fontproperties=english_font, verticalalignment='top')

        # 添加颜色条
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('|Error| Magnitude (kcal/mol)', fontproperties=english_font)
        cbar.ax.tick_params(length=6, width=1.2)
        cbar.ax.tick_params(axis='y', which='major',
                        length=6, direction='in', width=1.2,
                        left=False, right=True)
        cbar.ax.tick_params(axis='y', which='minor',
                        length=3, direction='in', width=1.0,
                        left=False, right=True)
        cbar.ax.yaxis.set_major_locator(ticker.MaxNLocator(6))
        cbar.ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

        # 设置边框样式
        for ax in axs:
            for spine in ax.spines.values():
                spine.set_linewidth(1.2)
            ax.spines[['right', 'top']].set_visible(True)
        for spine in cbar.ax.spines.values():
            spine.set_linewidth(1.2)

        plt.tight_layout(pad=1.2)


        return fig, axs
    
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np
    import time
    import pickle
    from IPython.display import clear_output
    import matplotlib.font_manager as fm

    # 设置matplotlib参数
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'mathtext.fontset': 'custom',
        'mathtext.rm': 'DejaVu Sans',
        'mathtext.it': 'DejaVu Sans:italic',
        'mathtext.bf': 'DejaVu Sans:bold',
        'mathtext.default': 'rm',
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'figure.max_open_warning': 50,  # 提高警告阈值
    })

    # 设置字体
    english_font = fm.FontProperties(family='DejaVu Sans', size=12)

    # 设置图表尺寸
    figure_size = (13, 4)

    # 训练参数
    num_epochs = 1001
    batch_size = 32

    # 存储每个epoch的损失
    loss_history_train = []
    loss_history_test = []

    # plt.ion()  # 开启交互模式
    start_time = time.time()
    fig = None  # 初始化图形变量

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # 训练批次
        train_losses = []
        for batch in train_dataloader:
            # 转换批次并训练
            batch = torch_batch_to_jax(batch)         
            state, batch_loss = train_step(state, batch)
            train_losses.append(batch_loss)
        
        # 计算 epoch 平均损失
        avg_train_loss = np.mean(train_losses)
        print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_train_loss:.4f}")

        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - start_time
            
        # 每10个epoch评估并可视化
        if epoch % 10 == 0:
            # 记录训练损失
            loss_history_train.append(avg_train_loss)
            
            true_energies = []
            predicted_energies = []
            base_energies = []
            distances = []
            
            # 在测试集上评估模型
            for batch in test_dataloader:
                batch = torch_batch_to_jax(batch)
                pred_energies = jax.vmap(model.apply, in_axes=(None, 0, 0, 0, 0, 0, 0, 0))(
                    state.params, 
                    batch['pos'], 
                    batch['box'], 
                    batch['pairs'], 
                    batch['topo_nblist'],
                    batch['topo_mask'],
                    batch['molID'],
                    batch['atypes'],
                )
                true_energies.append(batch['energy'] + batch['sr_energy'])
                predicted_energies.append(pred_energies + batch['sr_energy'])
                base_energies.append(batch['sr_energy'])
                distances.append(batch['distance'])

            # 合并数组
            true_energies = np.concatenate(true_energies) 
            predicted_energies = np.concatenate(predicted_energies)
            base_energies = np.concatenate(base_energies)
            distances = np.concatenate(distances)
            
            # 计算误差指标
            rmse = np.sqrt(np.mean((true_energies - predicted_energies)**2))
            rmse_base = np.sqrt(np.mean((true_energies - base_energies)**2))
            mae = np.average(np.absolute(true_energies - predicted_energies))
            loss_history_test.append(mae)

            # 清除之前的输出
            clear_output(wait=True)
            print_training_progress(
                epoch=epoch,
                num_epochs=num_epochs,
                loss=avg_train_loss,  # 使用平均损失
                loss_history_train=loss_history_train,
                loss_history_test=loss_history_test,
                true_energies=true_energies,
                predicted_energies=predicted_energies,
                base_energies=base_energies,
                distances=distances,
                epoch_time=epoch_time,
                total_time=total_time,
                figure_size=figure_size,
                english_font=english_font
            )

            # 保存进度
            if epoch % 100 == 0:
                fig, axs = plot_training_progress(
                    epoch=epoch,
                    num_epochs=num_epochs,
                    loss=avg_train_loss,  # 使用平均损失
                    loss_history_train=loss_history_train,
                    loss_history_test=loss_history_test,
                    true_energies=true_energies,
                    predicted_energies=predicted_energies,
                    base_energies=base_energies,
                    distances=distances,
                    epoch_time=epoch_time,
                    total_time=total_time,
                    figure_size=figure_size,
                    english_font=english_font
                )

                plt.savefig(f'final_learnable_training_progress_epoch_{epoch}.png', dpi=300, bbox_inches='tight', 
                        facecolor='white', pad_inches=0.1)
                final_params = state.params
                with open('model_params.pickle','wb') as ofile:
                    pickle.dump(final_params, ofile)            

    # 训练结束处理
    plt.ioff()  # 关闭交互模式
    if fig is not None:
        plt.close(fig)  # 确保最后一个图形被关闭

    # 最终保存
    final_params = state.params
    with open('final_learnable_model_params.pickle', 'wb') as ofile:
        pickle.dump(final_params, ofile)
    print(f"Training completed. Total time: {total_time:.2f} seconds")
