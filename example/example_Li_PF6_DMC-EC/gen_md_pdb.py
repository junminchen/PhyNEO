#!/usr/bin/env python3
import os
import sys
import numpy as np
import MDAnalysis as mda
import re
from openmm.app import PDBFile


# get_pdb_init.py
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

# get_split_pdb.py
def split_pdb_by_residue(pdb_path, residue_names):
    u = mda.Universe(pdb_path)
    if u.atoms.select_atoms('resname PF6') or u.atoms.select_atoms('resname DFP') or u.atoms.select_atoms('resname BF4'):
        print('Yes, there are ABn molecules!')
        for residue in u.residues:
            if residue.resname in residue_names:
                model_ABn = u.atoms.select_atoms(f"resname {residue.resname}")
                model_else = u.atoms.select_atoms(f"not resname {residue.resname}")
            else:
                continue
        model_ABn.write('init_extracted.pdb')
        model_else.write('init_remaining.pdb')
    else:
        print('No, there are not ABn molecules!')

# make_init_pdb.py
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


# -----------------------------
if __name__ == '__main__':
    pdb_path = 'model.pdb'  
    traj_path = 'output.dcd' 
    residue_names = ['PF6', 'DFP', 'BF4']  # ABn resid names
    ifn = 'init.pdb'  
    ofn = 'init_init.pdb' 
    
    # Get initial PDB file from trajectory
    # get_initial_from_trajectory(pdb_path, traj_path)
    
    # Create initial PDB file
    create_initial_pdb(ifn, ofn)

    # Extract PDB files based on residue names
    split_pdb_by_residue(ifn, residue_names)
