#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import time
import scipy
import sys
import pickle
import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm
from IPython.display import clear_output

import jax
import jax.numpy as jnp
from jax import jit, value_and_grad, vmap, random
from jax import config
from functools import partial
# config.update("jax_enable_x64", True)  # 如需更高精度可开启
config.update("jax_debug_nans", True)

from flax import linen as nn
from flax.training import train_state
import optax

from dmff.api import Hamiltonian
from dmff.utils import jit_condition, regularize_pairs, pair_buffer_scales
from dmff.admp.pairwise import distribute_scalar, distribute_v3
from dmff.admp.spatial import pbc_shift
from dmff.common import nblist

from openmm import *
from openmm.app import *
from openmm.unit import *

import MDAnalysis as mda
from ase.io import read, write
import mdtraj as md
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, List, Any
from scipy.sparse import csr_matrix


class MoleculeTorchDataset(Dataset):
    def __init__(self, ase_structures):
        # self.max_atoms = max(len(structure) for structure in ase_structures)
        self.max_atoms = 60  # Set a fixed maximum number of atoms
        self.z_index = [1, 3, 5, 6, 7, 8, 9, 11, 15, 16]
        
        # Pre-process all structures
        self.structures = ase_structures
        # 验证力数据是否存在
        for idx, struct in enumerate(ase_structures):
            if not hasattr(struct, 'get_forces') or struct.get_forces() is None:
                raise ValueError(f"结构 {idx} 缺少力数据！请确保ASE结构包含forces属性")

        
    def __len__(self):
        return len(self.structures)
    
    def __getitem__(self, idx):
        structure = self.structures[idx]
        n_atoms = len(structure)
        
        # Convert all data to numpy arrays for PyTorch compatibility
        pos = np.pad(structure.get_positions(), 
                     ((0, self.max_atoms - n_atoms), (0, 0)), 
                     mode='constant', constant_values=0)
        # 力（核心新增）
        forces = np.pad(
            structure.get_forces(),
            ((0, self.max_atoms - n_atoms), (0, 0)), 
            mode='constant', constant_values=0
        )        
        box = np.array(structure.get_cell())
        
        atomic_nums = np.pad(structure.get_atomic_numbers(), 
                             (0, self.max_atoms - n_atoms), 
                             mode='constant', constant_values=0)
        
        energy = float(structure.get_potential_energy())
        sr_energy = float(structure.info['sr_energy'])
        distance = float(structure.info['distance'])
        
        # 原子掩码（过滤无效力）
        atom_mask = np.pad(
            np.ones(n_atoms), 
            (0, self.max_atoms - n_atoms), 
            mode='constant', constant_values=0
        )
        mol_ID = np.pad(structure.get_array('molID'), 
                        (0, self.max_atoms - n_atoms), 
                        mode='constant', constant_values=10000)
        

        pairs = np.array(structure.info['pairs'])
        valid_mask = np.array(structure.info['valid_mask'])
        orig_topo_mask = np.array(structure.info['topo_mask'])
        orig_topo_nblist = np.array(structure.info['topo_nblist'])
        
        # Topology数据padding
        topo_mask = np.pad(
            orig_topo_mask, 
            ((0, self.max_atoms - n_atoms), (0, 0)), 
            mode='constant', constant_values=0
        )
        topo_nblist = np.pad(
            orig_topo_nblist, 
            ((0, self.max_atoms - n_atoms), (0, 0)), 
            mode='constant', constant_values=-1
        )
        topo_nblist[topo_nblist >= self.max_atoms] = -1

        atypes = np.pad(
            np.array([self.z_index.index(i) for i in structure.get_atomic_numbers()]), 
            (0, self.max_atoms - n_atoms), 
            mode='constant', constant_values=10000
        )
        
        
        return {
            'pos': pos.astype(np.float32),
            'forces': forces.astype(np.float32),
            'box': box.astype(np.float32),
            'atomic_numbers': atomic_nums.astype(np.int32),
            'energy': energy,
            'sr_energy': sr_energy,
            'atom_mask': atom_mask.astype(np.float32),

            'molID': mol_ID.astype(np.int32),
            'pairs': pairs.astype(np.int32),
            'valid_mask': valid_mask.astype(np.int32),
            'atypes': atypes.astype(np.int32),
            'distance': distance,
            'topo_mask': topo_mask.astype(np.int32),
            'topo_nblist': topo_nblist.astype(np.int32),
        }


# --------------------------
# 1. 基础工具函数
# --------------------------
def get_data(data, arr):
    dimer_test = [key for key in data if key.split('_')[-2] in arr and key.split('_')[-1] in arr]
    return dimer_test

def get_topology_neighbors(pdb_file, connectivity=2, max_neighbors=18, max_n_atoms=None):
    mol = mda.Universe(pdb_file)
    n_atoms = len(mol.atoms)
    if max_n_atoms is None:
        max_n_atoms = n_atoms

    indices = np.full((max_n_atoms, max_neighbors), -1, dtype=np.int32)
    mask = np.zeros((max_n_atoms, max_neighbors), dtype=np.int32)

    try:
        has_bonds = len(mol.bonds) > 0
    except AttributeError:
        has_bonds = False

    if has_bonds:
        # 构建稀疏邻接矩阵
        row_idx, col_idx = [], []
        for bond in mol.bonds:
            i, j = bond.atoms[0].index, bond.atoms[1].index
            row_idx.extend([i, j])
            col_idx.extend([j, i])
        data = np.ones(len(row_idx), dtype=bool)
        adj_init = csr_matrix((data, (row_idx, col_idx)), shape=(n_atoms, n_atoms), dtype=bool)

        # 完全复刻奇偶幂逻辑
        adj_matrix_odd = adj_init.copy()
        adj_matrix_self_even = adj_init.copy()
        adj_matrix = adj_matrix_odd.copy()

        for _ in range(connectivity - 1):
            adj_matrix_self_even = (adj_matrix_self_even @ adj_matrix_self_even).astype(bool)
            adj_matrix = (adj_matrix_odd + adj_matrix_self_even).astype(bool)
            adj_matrix_odd = (adj_matrix_self_even @ adj_init).astype(bool)

        # 提取邻居
        for i in range(n_atoms):
            neighbors = adj_matrix[i].indices
            neighbors = neighbors[neighbors != i]
            n_real = min(len(neighbors), max_neighbors)
            indices[i, :n_real] = neighbors[:n_real]
            mask[i, :n_real] = 1

    return indices, mask
    
@jit_condition(static_argnums=())
@partial(vmap, in_axes=(0, None, None), out_axes=(0, 0, 0, 0))
def get_environment_atoms(pairs, topo_nblist, topo_mask):
    j_centers = pairs[0]
    k_centers = pairs[1]
    
    j_neighbors = jnp.take(topo_nblist, j_centers, axis=0)
    k_neighbors = jnp.take(topo_nblist, k_centers, axis=0)

    valid_j = j_neighbors != -1
    valid_k = k_neighbors != -1

    mask_j = (j_neighbors != j_centers) & (j_neighbors != k_centers) & valid_j
    mask_k = (k_neighbors != j_centers) & (k_neighbors != k_centers) & valid_k

    topo_mask_j = jnp.take(topo_mask, j_centers, axis=0)
    topo_mask_k = jnp.take(topo_mask, k_centers, axis=0)

    valid_mask_j = topo_mask_j & mask_j
    valid_mask_k = topo_mask_k & mask_k

    return j_neighbors, k_neighbors, valid_mask_j, valid_mask_k

@jit_condition(static_argnums=())
@partial(vmap, in_axes=(0, None), out_axes=0)
def cutoff_cosine(distances, cutoff):
    x = distances / cutoff
    return jnp.where(x < 1, 0.5 * (jnp.cos(jnp.pi * x) + 1), 0.0)

def parameter_shapes(params):
    return jax.tree_util.tree_map(lambda p: p.shape, params)


# --------------------------
# 核心函数：筛选原子对 + 固定长度填充 + 整数掩码
# --------------------------
def filter_and_pad_pairs(pairs, atype_indices, max_pairs=30):
    """
    筛选含Li/Na的原子对，填充到固定长度，并生成整数类型的有效掩码（1=有效，0=无效）
    
    参数：
        pairs: jnp.ndarray，原始原子对，shape [n_raw_pairs, 3]（前两列原子索引，第三列键数）
        atype_indices: jnp.ndarray，每个原子的类型索引，shape [n_atoms,]
        max_pairs: int，固定原子对数量
    
    返回：
        padded_pairs: jnp.ndarray，填充后的原子对，shape [max_pairs, 3]
        valid_mask_int: jnp.ndarray，整数有效掩码（1=有效，0=无效），shape [max_pairs,]
    """
    # 1. 筛选含Li/Na的目标原子对
    pair_atype_i = atype_indices[pairs[:, 0]]  # 原子对第一个原子的类型索引
    pair_atype_j = atype_indices[pairs[:, 1]]  # 原子对第二个原子的类型索引
    is_target_i = jnp.isin(pair_atype_i, TARGET_ATYPE_INDICES)
    is_target_j = jnp.isin(pair_atype_j, TARGET_ATYPE_INDICES)
    target_pair_mask = is_target_i | is_target_j  # 布尔掩码：True=目标对
    filtered_pairs = pairs[target_pair_mask]  # 筛选后的有效原子对
    n_target_pairs = filtered_pairs.shape[0]

    # 2. 填充到固定长度（用[-1,-1,-1]表示无效原子对）
    if n_target_pairs < max_pairs:
        # 生成填充用的无效原子对
        pad_pairs = jnp.full((max_pairs - n_target_pairs, 3), -1, dtype=pairs.dtype)
        padded_pairs = jnp.concatenate([filtered_pairs, pad_pairs], axis=0)
        
        # 生成整数掩码：前n_target_pairs个为1（有效），其余为0（无效）
        valid_mask_int = jnp.concatenate([
            jnp.ones(n_target_pairs, dtype=jnp.int32),  # 有效位置→1
            jnp.zeros(max_pairs - n_target_pairs, dtype=jnp.int32)  # 填充位置→0
        ])
    else:
        # 有效原子对超过max_pairs，截断到固定长度
        # padded_pairs = filtered_pairs[:max_pairs]
        padded_pairs = filtered_pairs
        valid_mask_int = jnp.ones(len(filtered_pairs), dtype=jnp.int32)  # 截断后全为有效→1
        
    return padded_pairs, valid_mask_int


# --------------------------
# 3. EAPNN模型（包含力计算）
# --------------------------
zindex = [1.0, 3.0, 5.0, 6.0, 7.0, 8.0, 9.0, 11.0, 15.0, 16.0]
charge_to_index = {
    0.0: 100000, 1.0: 0, 3.0: 1, 5.0: 2, 6.0: 3,
    7.0: 4, 8.0: 5, 9.0: 6, 11.0: 7, 15.0: 8, 16.0: 9
}

zindex = [1.0, 3.0, 5.0, 6.0, 7.0, 8.0, 9.0, 11.0, 15.0, 16.0]
LI_ATOMIC_NUM = 3.0    # Li的原子序数
NA_ATOMIC_NUM = 11.0   # Na的原子序数
# 获取Li和Na在zindex中的索引（用于后续筛选）
li_index = zindex.index(LI_ATOMIC_NUM) if LI_ATOMIC_NUM in zindex else -1
na_index = zindex.index(NA_ATOMIC_NUM) if NA_ATOMIC_NUM in zindex else -1
# 目标原子类型索引（Li和Na）
TARGET_ATYPE_INDICES = jnp.array([li_index, na_index])

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
            acsf_eta=self.acsf_eta,
            apsf_eta=self.apsf_eta,
            use_pbc=self.use_pbc  # 新增参数，默认使用周期性边界条件
        )
        self.neural_network = NeuralNetwork()

    def __call__(self, pos, box, pairs, valid_mask, topo_nblist, topo_mask, mol_ID, atype_indices):
        features, dr_norm, buffer_scales = self.feature_extractor(pos, box, pairs, valid_mask, topo_nblist, topo_mask, mol_ID, atype_indices)
        atomic_energies = self.neural_network(features, dr_norm, buffer_scales)
        return jnp.sum(atomic_energies)

    def get_features(self, pos, box, pairs, valid_mask, topo_nblist, topo_mask, mol_ID, atype_indices):
        return self.feature_extractor(pos, box, pairs, valid_mask, topo_nblist, topo_mask, mol_ID, atype_indices)

    def get_energy(self, features, dr_norm, buffer_scales):
        atomic_energies = self.neural_network(features, dr_norm, buffer_scales)
        return jnp.sum(atomic_energies)

    def predict_energy_force(self, params, pos, box, pairs, valid_mask, topo_nblist, topo_mask, mol_ID, atype_indices):
        """同时预测能量和力（力 = -dE/dpos）"""
        def energy_fn(pos):
            return self.apply(
                params, pos, box, pairs, valid_mask, topo_nblist, topo_mask, mol_ID, atype_indices
            )
        
        energy, grad_energy = value_and_grad(energy_fn)(pos)
        force = grad_energy  # 力的物理定义
        return energy, force

class FeatureExtractor(nn.Module):
    n_atoms: int
    n_atype: int
    rc: float
    acsf_nmu: int = 20
    apsf_nmu: int = 10
    acsf_eta: float = 100
    apsf_eta: float = 25
    use_pbc: bool = True  # 新增参数，默认使用周期性边界条件

    def setup(self):
        self.acsf_mus = jnp.linspace(0.0, 5.0, self.acsf_nmu)
        self.apsf_mus = jnp.linspace(-1.0, 1.0, self.apsf_nmu)

    def compute_atomcenter_features(self, pos, box, topo_nblist, topo_mask, atype_indices, acsf_mus, acsf_eta):
        """直接计算所有原子的环境特征"""
        # 获取环境原子位置 [n_atoms, max_neighbors, 3]
        r_center = pos  # [n_atoms, 3]
        r_env = pos[topo_nblist]  # [n_atoms, max_neighbors, 3]
        
        # 计算相对位置和距离
        dr = r_env - r_center[:, None, :]  # [n_atoms, max_neighbors, 3]
        box_inv = jnp.linalg.inv(box)
        dr = pbc_shift(dr, box, box_inv)  # 如果使用PBC，这里需要传入box和box_inv
        dr_norm = jnp.linalg.norm(dr+1e-10, axis=2)  # [n_atoms, max_neighbors]
        
        # 计算截断函数
        f_cut = cutoff_cosine(dr_norm, self.rc) * topo_mask  # [n_atoms, max_neighbors]
        
        # 计算径向基函数
        exp_term = jnp.exp(-acsf_eta * jnp.square(dr_norm[..., None] - acsf_mus))  # [n_atoms, max_neighbors, n_mu]
        G_raw = exp_term * f_cut[..., None]  # [n_atoms, max_neighbors, n_mu]
        
        # 按原子类型累积特征
        type_one_hot = (atype_indices[topo_nblist][..., None] == jnp.arange(self.n_atype))  # [n_atoms, max_neighbors, n_atype]
        
        # 一次性计算所有特征
        G = jnp.einsum('ijk,ijl->ikl', G_raw, type_one_hot)  # [n_atoms, n_mu, n_atype]
        
        return G
    
    def compute_atompair_features(self, cos_gamma_i, cos_gamma_j, j_list, k_list, j_mask, k_mask,
                                  buffer_nblist_inter_rc, atype_indices, apsf_mus, apsf_eta):
        # 计算 i 和 j 的角度特征
        angle_features_i = jnp.exp(-apsf_eta * jnp.square(cos_gamma_i[..., None] - apsf_mus))
        angle_features_j = jnp.exp(-apsf_eta * jnp.square(cos_gamma_j[..., None] - apsf_mus))

        # 创建type_one_hot [n_pairs, max_neighbors, n_atype]
        type_one_hot_i = (atype_indices[j_list][..., None] == jnp.arange(self.n_atype))
        type_one_hot_j = (atype_indices[k_list][..., None] == jnp.arange(self.n_atype))

        # 应用掩码
        masked_features_i = angle_features_i * j_mask[..., None]  # [n_pairs, max_neighbors, n_mu]
        masked_features_j = angle_features_j * k_mask[..., None]  # [n_pairs, max_neighbors, n_mu]

        # 一次性计算所有类型的贡献
        G_i = jnp.einsum('ijk,ijl->ikl', masked_features_i, type_one_hot_i)  # [n_pairs, n_mu, n_atype]
        G_j = jnp.einsum('ijk,ijl->ikl', masked_features_j, type_one_hot_j)  # [n_pairs, n_mu, n_atype]

        # 对称平均并应用分子间相互作用掩码
        G = (G_i + G_j) * 0.5 * buffer_nblist_inter_rc[:, None, None]

        return G

    def __call__(self, pos, box, pairs, valid_mask, topo_nblist, topo_mask, mol_ID, atype_indices):
        apsf_mus, apsf_eta = self.apsf_mus, self.apsf_eta
        acsf_mus, acsf_eta = self.acsf_mus, self.acsf_eta
        
        mScales = jnp.array([0., 0., 0., 0., 0., 1.])
        pairs = pairs.at[:, :2].set(regularize_pairs(pairs[:, :2]))
        nbonds = pairs[:, 2]
        mscales = distribute_scalar(mScales, nbonds - 1)
        pairs = pairs[:, :2]
        buffer_scales = pair_buffer_scales(pairs[:, :2]) * valid_mask
        mscales = mscales * buffer_scales

        box_inv = jnp.linalg.inv(box)
        ri = pos[pairs[:, 0]]
        rj = pos[pairs[:, 1]]

        rij = rj - ri
        rij = pbc_shift(rij, box, box_inv)

        dr_norm = jnp.linalg.norm(rij + 1e-10, axis=1)

        same_mol = mol_ID[pairs[:, 0]] == mol_ID[pairs[:, 1]]
        buffer_inter = jnp.where(same_mol, 0., 1.)
        buffer_intra = jnp.where(same_mol, 1., 0.)
        cutoff = 0.5 * (1 + jnp.cos(jnp.pi * dr_norm / self.rc))
        cutoff = jnp.where(dr_norm <= self.rc, cutoff, 0.0)

        buffer_nblist_inter = buffer_inter * buffer_scales
        buffer_nblist_intra = buffer_intra * buffer_scales
        buffer_nblist_inter_rc = buffer_nblist_inter * cutoff


        # 获取环境原子信息
        j_list, k_list, j_mask, k_mask = get_environment_atoms(pairs, topo_nblist, topo_mask)

        # 计算环境原子的位置和角度（两个方向）
        # i 的环境
        valid_j_mask = j_mask[..., None]  # (n_pairs, max_neighbors, 1)
        rj_env = jnp.where(valid_j_mask, pos[j_list], 0.0)  # 无效索引位置设为 0
        # rj_env = pos[j_list]
        rj_X = rj_env - ri[:, None, :]
        rj_X = pbc_shift(rj_X, box, box_inv)
        norm_rj_X = jnp.linalg.norm(rj_X + 1e-10, axis=2, keepdims=True)
        rj_X_norm = rj_X / norm_rj_X
        rij_unit = rij / (dr_norm[:, None] + 1e-10)
        cos_gamma_i = jnp.einsum('aji,ai->aj', rj_X_norm, rij_unit) * j_mask

        # j 的环境
        valid_k_mask = k_mask[..., None]  # (n_pairs, max_neighbors, 1)
        rk_env = jnp.where(valid_k_mask, pos[k_list], 0.0)  # 无效索引位置设为 0
        # rk_env = pos[k_list]
        rk_X = rk_env - rj[:, None, :]
        rk_X = pbc_shift(rk_X, box, box_inv)
        norm_rk_X = jnp.linalg.norm(rk_X + 1e-10, axis=2, keepdims=True)
        rk_X_norm = rk_X / norm_rk_X
        rji_unit = -rij_unit
        cos_gamma_j = jnp.einsum('aji,ai->aj', rk_X_norm, rji_unit) * k_mask

        # 计算原子对特征
        atompair_features = self.compute_atompair_features(cos_gamma_i, cos_gamma_j, j_list, k_list, j_mask, k_mask,
                                                           buffer_nblist_inter_rc, atype_indices, apsf_mus, apsf_eta)

        atom_features = self.compute_atomcenter_features(
            pos, box, topo_nblist, topo_mask, atype_indices, acsf_mus, acsf_eta)
        
        atom_features_i = atom_features[pairs[:, 0],]
        atom_features_j = atom_features[pairs[:, 1],]
        atom_features = (atom_features_i + atom_features_j) * 0.5

        # # 处理原子类型特征
        elem_indices = jnp.array(zindex)[atype_indices]
        j_atype = elem_indices[pairs[:,0]]  # j原子的类型
        k_atype = elem_indices[pairs[:,1]]  # k原子的类型
        
        # 为j和k原子分别创建one-hot编码
        j_onehot = jnp.concatenate([j_atype.reshape(-1,1), int_to_onehot(j_atype, 10)], axis=1)
        k_onehot = jnp.concatenate([k_atype.reshape(-1,1), int_to_onehot(k_atype, 10)], axis=1)
        
        # 合并j和k的类型特征
        atype_onehot = jnp.concatenate([j_onehot, k_onehot], axis=1)

        atom_features = atom_features.reshape(atom_features.shape[0], -1)
        atompair_features = atompair_features.reshape(atompair_features.shape[0], -1)
        apsf_features = jnp.concatenate((atom_features, atompair_features, atype_onehot), axis=1)

        return apsf_features, dr_norm, buffer_nblist_inter_rc    

class NeuralNetwork(nn.Module):
    dense_nodes: int = 64
    
    @nn.compact
    def __call__(self, combined, dr_norm, buffer_nblist_inter):
        x = combined
        for _ in range(3):
            x = nn.Dense(self.dense_nodes)(x)
            x = nn.LayerNorm()(x)
            x = nn.relu(x)
        out_AB = nn.Dense(1)(x)
        
        return jnp.sum(out_AB * buffer_nblist_inter[:,None]) 

if __name__ == "__main__":
    rc = 6.0
    connectivity = 4
    max_neighbors = 10
    acsf_nmu=20
    apsf_nmu=20
    acsf_eta=100
    apsf_eta=50 

    ff_xml = 'output.xml'
    pdb = 'dimer_062_Li_EC.pdb'

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
    n_atype = len(zindex)
    z_atomnum_list = [float(num) for num in np.array(z_atomnum)]
    zindex_dict = {float(num): i for i, num in enumerate(zindex)}
    atype_indices = jnp.array([zindex_dict.get(num, -1) for num in z_atomnum_list])
    n_atoms = len(pos)

    # 提取原子序数
    atomic_nums = jnp.array([atom.element.atomic_number for atom in mol.topology.atoms()], dtype=int)
    # 标记Li(3)和Na(11)原子
    target_mask = (atomic_nums == 3) | (atomic_nums == 11)
    target_indices = jnp.where(target_mask)[0]

    valid_pairs, valid_mask = filter_and_pad_pairs(pairs, atype_indices, max_pairs=len(target_indices)*100)

    # 获取拓扑邻居
    topo_nblist, topo_mask = get_topology_neighbors(pdb, connectivity=connectivity, max_neighbors=max_neighbors, max_n_atoms=None)

    model = EAPNNForce(
        n_atoms=n_atoms, 
        n_atype=n_atype, 
        rc=rc,  
        acsf_nmu=acsf_nmu,
        apsf_nmu=apsf_nmu,
        acsf_eta=acsf_eta,
        apsf_eta=apsf_eta,
        use_pbc=True,  # 关键修改
    )

    key = jax.random.PRNGKey(0)

    params_init = model.init(key, pos, box, valid_pairs, valid_mask, topo_nblist, topo_mask, mol_ID, atype_indices)
    start_time = time.time()

    params = params_init
    features, dr_norm, buffer_scales = model.apply(params, pos, box, valid_pairs, valid_mask, topo_nblist, topo_mask, mol_ID, atype_indices, method=model.get_features)
    energy = model.apply(params, pos, box, valid_pairs, valid_mask, topo_nblist, topo_mask, mol_ID, atype_indices)
    print(energy)
    end_time = time.time()
    print(f"time cost: {end_time - start_time}s")

    def dmff_calculator(pos, box, pairs, valid_pairs, valid_mask):
        E_nb_ml = model.apply(params, pos, box, valid_pairs, valid_mask, topo_nblist, topo_mask, mol_ID, atype_indices)
        return E_nb_ml
    calc_dmff = jit(value_and_grad(dmff_calculator,argnums=(0, 1)))
    # compile tot_force function
    energy, (grad, virial) = calc_dmff(pos, box, pairs, valid_pairs, valid_mask)
    print(energy, grad, virial)

    num_runs = 10
    total_time = 0
    for _ in range(num_runs):
        start_time = time.time()
        energy = calc_dmff(pos, box, pairs, valid_pairs, valid_mask)
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

        pairs, valid_mask = filter_and_pad_pairs(pairs, atype_indices, max_pairs=40)

        # 计算拓扑邻居（仅需一次）
        topo_nblist, topo_mask = get_topology_neighbors(pdb_path, connectivity=connectivity, max_neighbors=max_neighbors, max_n_atoms=None)
        
        # 存入缓存
        nblist_cache[dimer] = (pairs, valid_mask, topo_nblist, topo_mask)

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
        pairs, valid_mask, topo_nblist, topo_mask = nblist_cache[key]
        
        structure.info['pairs'] = pairs
        structure.info['valid_mask'] = valid_mask
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
        pairs, valid_mask, topo_nblist, topo_mask = nblist_cache[key]
        
        structure.info['pairs'] = pairs
        structure.info['valid_mask'] = valid_mask
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
        batch_size=128,
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
        
    batch = torch_batch_to_jax(batch)

    def predict_single(sample):
        energy_pred, force_pred = model.predict_energy_force(
            params,
            pos=sample['pos'],
            box=sample['box'],
            pairs=sample['pairs'],
            valid_mask=sample['valid_mask'],
            topo_nblist=sample['topo_nblist'],
            topo_mask=sample['topo_mask'],
            mol_ID=sample['molID'],
            atype_indices=sample['atypes']
        )
        return energy_pred, force_pred

    energy_pred, force_pred = vmap(predict_single)(batch)


    def create_train_state(model, learning_rate, key, init_batch):
        init_pos = init_batch['pos'][0]
        init_box = init_batch['box'][0]
        init_pairs = init_batch['pairs'][0]
        init_valid_mask = init_batch['valid_mask'][0]
        init_topo_nblist = init_batch['topo_nblist'][0]
        init_topo_mask = init_batch['topo_mask'][0]
        init_molID = init_batch['molID'][0]
        init_atypes = init_batch['atypes'][0]
        
        params = model.init(
            key, init_pos, init_box, init_pairs, init_valid_mask, init_topo_nblist, init_topo_mask, init_molID, init_atypes
        )
        # tx = optax.adamw(
        #     learning_rate=optax.exponential_decay(
        #         init_value=learning_rate,
        #         transition_steps=100,
        #         decay_rate=0.95
        #     )
        # )
        tx = optax.adam(learning_rate=learning_rate)
        return train_state.TrainState.create(
            apply_fn=model.apply, params=params, tx=tx
        )

    @jit
    def train_step(state, batch, force_weight=0):
    # def train_step(state, batch, force_weight=10):
        def loss_fn(params):
            # 批量预测能量和力
            def predict_single(sample):
                energy_pred, force_pred = model.predict_energy_force(
                    params,
                    pos=sample['pos'],
                    box=sample['box'],
                    pairs=sample['pairs'],
                    valid_mask=sample['valid_mask'],
                    topo_nblist=sample['topo_nblist'],
                    topo_mask=sample['topo_mask'],
                    mol_ID=sample['molID'],
                    atype_indices=sample['atypes']
                )
                return energy_pred, force_pred
            
            energy_pred, force_pred = vmap(predict_single)(batch)
            
            # 能量损失
            energy_true = batch['energy']
            energy_loss = jnp.mean((energy_pred - energy_true) ** 2)
            
            # 力损失（带掩码）
            force_true = batch['forces']
            atom_mask = batch['atom_mask'][..., None]
            force_error = (force_pred - force_true) * atom_mask
            force_loss = jnp.mean(force_error ** 2)
            
            # 总损失
            total_loss = energy_loss + force_weight * force_loss
            return total_loss, (energy_loss, force_loss)
        
        (total_loss, (energy_loss, force_loss)), grads = value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, total_loss, energy_loss, force_loss

    def evaluate(model, params, test_dataloader):
        energy_rmse_list = []
        force_rmse_list = []
        
        for batch in test_dataloader:
            batch = torch_batch_to_jax(batch)
            
            def predict_single(sample):
                energy_pred, force_pred = model.predict_energy_force(
                    params,
                    pos=sample['pos'],
                    box=sample['box'],
                    pairs=sample['pairs'],
                    valid_mask=sample['valid_mask'],
                    topo_nblist=sample['topo_nblist'],
                    topo_mask=sample['topo_mask'],
                    mol_ID=sample['molID'],
                    atype_indices=sample['atypes']
                )
                return energy_pred, force_pred
            
            energy_pred, force_pred = vmap(predict_single)(batch)
            
            # 能量RMSE
            energy_true = batch['energy']
            energy_rmse = jnp.sqrt(jnp.mean((energy_pred - energy_true) ** 2))
            energy_rmse_list.append(energy_rmse)
            
            # 力RMSE
            force_true = batch['forces']
            atom_mask = batch['atom_mask'][..., None]
            force_error = (force_pred - force_true) * atom_mask
            force_rmse = jnp.sqrt(jnp.mean(force_error ** 2))
            force_rmse_list.append(force_rmse)
        
        return {
            'energy_rmse': jnp.mean(jnp.array(energy_rmse_list)),
            'force_rmse': jnp.mean(jnp.array(force_rmse_list))
        }

    learning_rate = 1e-3
    # with open('final_model_params.pickle', 'rb') as ifile:
    #     params = pickle.load(ifile)
    # model.params = params
    state = create_train_state(model, learning_rate, jax.random.PRNGKey(0), batch)


    # --------------------------
    # 5. 可视化与主函数
    # --------------------------
    def setup_plot_style():
        plt.rcParams.update({
            'font.family': 'DejaVu Sans',
            'mathtext.fontset': 'custom',
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'figure.max_open_warning': 50,
            'figure.dpi': 100
        })
        return fm.FontProperties(family='DejaVu Sans', size=12)

    def plot_training_progress(epoch, num_epochs, train_metrics, test_metrics, total_time):
        english_font = setup_plot_style()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 损失曲线
        epochs = list(range(0, len(train_metrics['total_loss'])*10, 10))
        ax1.plot(epochs, train_metrics['total_loss'], 'r-', marker='o', label='Total Loss', markersize=4)
        ax1.plot(epochs, train_metrics['energy_loss'], 'b--', label='Energy Loss', linewidth=2)
        ax1.plot(epochs, train_metrics['force_loss'], 'g-.', label='Force Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontproperties=english_font)
        ax1.set_ylabel('Loss', fontproperties=english_font)
        ax1.set_title(f'Training Loss (Total Time: {total_time/60:.1f} min)', fontproperties=english_font)
        ax1.legend(prop=english_font)
        ax1.grid(alpha=0.3)
        
        # 测试集RMSE
        ax2_twin = ax2.twinx()
        ax2.plot(epochs, test_metrics['energy_rmse'], 'b-', marker='s', label='Energy RMSE', markersize=4)
        ax2_twin.plot(epochs, test_metrics['force_rmse'], 'g-', marker='^', label='Force RMSE', markersize=4)
        ax2.set_xlabel('Epoch', fontproperties=english_font)
        ax2.set_ylabel('Energy RMSE (kJ/mol)', color='b', fontproperties=english_font)
        ax2_twin.set_ylabel('Force RMSE (kJ/(mol·Å))', color='g', fontproperties=english_font)
        ax2.tick_params(axis='y', labelcolor='b')
        ax2_twin.tick_params(axis='y', labelcolor='g')
        ax2.set_title('Test Set RMSE', fontproperties=english_font)
        ax2.grid(alpha=0.3)
        
        # 合并图例
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', prop=english_font)
        
        plt.tight_layout()
        return fig
    # 训练参数

    # 训练循环
    print("开始训练...")
    start_time = time.time()
    train_metrics = {
        'total_loss': [], 'energy_loss': [], 'force_loss': []
    }
    test_metrics = {
        'energy_rmse': [], 'force_rmse': []
    }
    num_epochs = 5000
    for epoch in range(num_epochs):
        epoch_start = time.time()
        train_total_loss = []
        train_energy_loss = []
        train_force_loss = []
        
        # 训练批次
        for batch in train_dataloader:
            batch = torch_batch_to_jax(batch)
            state, total_loss, energy_loss, force_loss = train_step(
                state, batch)
            train_total_loss.append(total_loss.item())
            train_energy_loss.append(energy_loss.item())
            train_force_loss.append(force_loss.item())
        
        # 每10轮评估
        if epoch % 10 == 0:
            avg_total_loss = np.mean(train_total_loss)
            avg_energy_loss = np.mean(train_energy_loss)
            avg_force_loss = np.mean(train_force_loss)
            train_metrics['total_loss'].append(avg_total_loss)
            train_metrics['energy_loss'].append(avg_energy_loss)
            train_metrics['force_loss'].append(avg_force_loss)
            
            # 评估测试集
            test_metric = evaluate(model, state.params, test_dataloader)
            test_metrics['energy_rmse'].append(test_metric['energy_rmse'].item())
            test_metrics['force_rmse'].append(test_metric['force_rmse'].item())
            
            # 打印进度
            epoch_time = time.time() - epoch_start
            total_time = time.time() - start_time
            clear_output(wait=True)
            print(f"Epoch {epoch:4d}/{num_epochs} | "
                    f"Total Loss: {avg_total_loss:.4f} | "
                    f"Energy Loss: {avg_energy_loss:.4f} | "
                    f"Force Loss: {avg_force_loss:.4f} | "
                    f"Test Energy RMSE: {test_metric['energy_rmse']:.4f} | "
                    f"Test Force RMSE: {test_metric['force_rmse']:.4f} | "
                    f"Time: {epoch_time:.2f}s")
            
            # 绘制并保存进度图
            fig = plot_training_progress(epoch, num_epochs, train_metrics, test_metrics, total_time)
            plt.savefig(f"results/training_progress_epoch_{epoch}.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            # 保存模型参数
            os.makedirs("results", exist_ok=True)
            with open(f"results/model_params_epoch_{epoch}.pickle", 'wb') as f:
                pickle.dump(state.params, f)

    # 训练结束
    print("训练完成！")
    with open("results/final_model_params.pickle", 'wb') as f:
        pickle.dump(state.params, f)
    with open("results/training_metrics.pickle", 'wb') as f:
        pickle.dump({'train': train_metrics, 'test': test_metrics}, f)

