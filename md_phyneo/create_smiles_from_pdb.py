import os
import numpy
import pickle
from ase import Atoms
from ase.io import write, read
from rdkit import Chem
import json
import glob
import sys
# conda activate bff

from rdkit import Chem


def pdb_to_mapped_smiles(pdb_file):
    if pdb_file.split('/')[-1] == 'Li.pdb':
        mapped_smiles = "[Li+:1]" 
    else:
        mol = Chem.MolFromPDBFile(pdb_file, removeHs=False, sanitize=False)
        for atom in mol.GetAtoms():
            print(f"Idx={atom.GetIdx()}, Symbol={atom.GetSymbol()}, Charge={atom.GetFormalCharge()}")
        smiles = Chem.MolToSmiles(mol)
        print("SMILES:", smiles)

        for i, atom in enumerate(mol.GetAtoms(), start=1):
            atom.SetAtomMapNum(i)
        mapped_smiles = Chem.MolToSmiles(mol)
    return mapped_smiles


pdb_folder = 'pdb_bank'

# pdb_files = glob.glob(pdb_folder+'/*.pdb')

pdb_files = [pdb_folder+f'/{sys.argv[1]}.pdb']

for pdbB_file in pdb_files:
    mapped_smiles_B = pdb_to_mapped_smiles(pdbB_file)
    print(mapped_smiles_B)
