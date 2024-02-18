#!/usr/bin/env python
import shutil
import os 
import sys
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
from rdkit import Chem
from rdkit.Chem import AllChem
import pickle 

def replace_atype(pdb, start_atype):
    mol = Chem.MolFromPDBFile(pdb, sanitize=False)
    lst = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    
    # Create a dictionary to store element information for each atomic type
    atom_element_dict = {}
    
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        atom_type = str(lst[idx])
        atom_symbol = atom.GetSymbol()
        atom_element_dict[atom_type] = atom_symbol

    set_lst = list(set(lst))
    for idx, atom_type in enumerate(lst):
        atom_index = set_lst.index(atom_type) + start_atype
        lst[idx] = f"{atom_element_dict[str(atom_type)]}{atom_index}"

    result_dict = {}
    for idx, element in enumerate(lst):
        key = element
        if key in result_dict:
            result_dict[key] = np.append(result_dict[key], idx)
        else:
            result_dict[key] = np.array([idx], dtype=np.int32)
    
    return result_dict

start_index = 1
pdb = 'DMC.pdb'
dic_atypes = replace_atype(pdb, start_index)
data = {}
data['DMC'] = dic_atypes
with open('atype_data.pickle', 'wb') as ofile:
    pickle.dump(data, ofile)