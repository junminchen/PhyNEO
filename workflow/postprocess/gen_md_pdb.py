#!/usr/bin/env python3
import os
import sys
import numpy as np
import MDAnalysis as mda
import re
from openmm.app import PDBFile


# get_pdb_init.py 的内容
def get_initial_from_trajectory(pdb_path, traj_path):
    path = '.'
    pdb = path + '/' + pdb_path
    traj = path + '/' + traj_path
    u = mda.Universe(pdb)
    u1 = mda.Universe(pdb, traj)
    last_frame = u1.trajectory[-1]
    box = last_frame.dimensions
    box[1] = box[0] # make the cubic box
    box[2] = box[0] # make the cubic box 
    u.trajectory[0].dimensions = box
    u.atoms.positions = last_frame.positions
    u.atoms.write(f'init.pdb')

# get_split_pdb.py 的内容
def split_pdb_by_residue(pdb_path, residue_names):
    u = mda.Universe(pdb_path)
    if u.atoms.select_atoms('resname PF6') or u.atoms.select_atoms('resname DFP') or u.atoms.select_atoms('resname BF4'):
        print('Yes, there are ABn molecules!')
        # 遍历原始PDB中的所有残基
        for residue in u.residues:
            # 检查残基名是否在我们的列表中
            if residue.resname in residue_names:
                # 如果是，将其添加到提取的宇宙中
                model_ABn = u.atoms.select_atoms(f"resname {residue.resname}")
                model_else = u.atoms.select_atoms(f"not resname {residue.resname}")
            else:
                continue
        model_ABn.write('init_extracted.pdb')
        model_else.write('init_remaining.pdb')
    else:
        print('No, there are not ABn molecules!')

# make_init_pdb.py 的内容
def create_initial_pdb(ifn, ofn):
    u = mda.Universe(ifn)
    a, b, c, alpha, beta, gamma = u.dimensions
    u.atoms[u.atoms.types == "LI"].masses = 6.941
    u.atoms[u.atoms.types == "LI"].masses = 22.989769  
    hydrogen_atoms = u.select_atoms("type LI")
    new_type = 'Li'
    for atom in hydrogen_atoms:
        atom.type = new_type
    hydrogen_atoms = u.select_atoms("type NA")
    new_type = 'Na'
    for atom in hydrogen_atoms:
        atom.type = new_type

    with open(ofn, 'w') as f: 
        print('TITLE cell{angstrom} positions{angstrom}', file=f) # everything in angstrom
        print('CRYST%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f P 1          1'%(a,b,c,alpha,beta,gamma), file=f)
        i_atom = 1
        for atom in u.atoms:
            if atom.mass > 1e-6: # not virtuals
                r = atom.position
                print('ATOM%7d%5s   1     1%12.3f%8.3f%8.3f  0.00  0.00            0'%(i_atom, atom.type, r[0], r[1], r[2]), file=f)
                i_atom += 1
        print('END', file=f)


# 整合后的脚本逻辑
if __name__ == '__main__':
    pdb_path = 'model.pdb'  # PDB文件路径
    traj_path = 'output.dcd'  # 轨迹文件路径
    residue_names = ['PF6', 'DFP', 'BF4']  # 需要提取的ABn型分子的残基名列表
    ifn = 'nvt_init.pdb'  # 输入文件名
    ofn = 'nvt_init_init.pdb' # 输出文件名以供i-pi识别位置元素等信息
    ifn = 'init.pdb'  # 输入文件名
    ofn = 'init_init.pdb' # 输出文件名以供i-pi识别位置元素等信息
    
    # 从轨迹文件中获取初始PDB文件
    #get_initial_from_trajectory(pdb_path, traj_path)
    
    # 创建初始PDB文件
    create_initial_pdb(ifn, ofn)

    # 根据残基名提取PDB文件
    #split_pdb_by_residue(ifn, residue_names)
