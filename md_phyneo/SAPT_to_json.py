import os
import numpy as np
import pickle
from ase import Atoms
from ase.io import write, read
from rdkit import Chem
import json
from mol_bank import phyneo_name_mapped_smiles
import sys
# conda activate bff

def get_pairs(data, mon1, mon2):
    for key in data.keys():
        conf, num, monA, monB = key.split('_')
        if mon1 == monA and mon2 == monB:
            return key

pdb_folder = 'pdb_bank'


data_file = 'data.pickle'
with open(data_file,'rb') as ifile:
    data_tot = pickle.load(ifile)

# batch = '020'
# key = 'conf_000_DEC_DEC'
# key = 'conf_063_Li_EMC'
# key = 'conf_053_Li_FSI'
# key = 'conf_099_FSI_EMC'
# key = 'conf_051_Li_PF6'
# key = 'conf_054_Li_TFSI'
# key = 'conf_062_Li_EC'
mon1, mon2 = sys.argv[1:]
key = get_pairs(data_tot, mon1, mon2)
batch = '000'
data = data_tot[key][batch]

conf, num, monA, monB = key.split('_')
pdbA_file = f'{pdb_folder}/{monA}.pdb'
pdbB_file = f'{pdb_folder}/{monB}.pdb'

def pdb_to_mapped_smiles(pdb_file):
    if pdb_file.split('/')[-1] == 'Li.pdb':
        mapped_smiles = "[Li+:1]" 
    else:
        mol = Chem.MolFromPDBFile(pdb_file, removeHs=False)
        for atom in mol.GetAtoms():
            print(f"Idx={atom.GetIdx()}, Symbol={atom.GetSymbol()}, Charge={atom.GetFormalCharge()}")
        smiles = Chem.MolToSmiles(mol)
        print("SMILES:", smiles)

        for i, atom in enumerate(mol.GetAtoms(), start=1):
            atom.SetAtomMapNum(i)
        mapped_smiles = Chem.MolToSmiles(mol)
    return mapped_smiles

# mapped_smiles_A = pdb_to_mapped_smiles(pdbA_file)
# mapped_smiles_B = pdb_to_mapped_smiles(pdbB_file)
# print(mapped_smiles_B)
atomsA_raw = read(pdbA_file)
atomsB_raw = read(pdbB_file)


pdbA_frames = []
pdbB_frames = []
for ipt in range(len(data['tot_full'])):
    posA = data['posA'][ipt]
    posB = data['posB'][ipt].copy()

    atomsA = Atoms(symbols=atomsA_raw.get_chemical_symbols(),
                positions=atomsA_raw.get_positions())
    atomsB = Atoms(symbols=atomsB_raw.get_chemical_symbols(),
                positions=atomsB_raw.get_positions())


    atomsA.info["mapped_smiles"] = phyneo_name_mapped_smiles[monA]
    # atomsB.info["mapped_smiles"] = "[C:1]([F:2])([F:3])([F:4])[S:5](=[O:6])(=[O:7])[N-:8][S:9](=[O:10])(=[O:11])[C:12]([F:13])([F:14])[F:15]"
    
    # EC
    atomsB.info["mapped_smiles"] = phyneo_name_mapped_smiles[monB]


    # PF6
    # posB[[0, 1]] = posB[[1, 0]]
    # atomsB = Atoms(symbols=['P', 'F', 'F', 'F', 'F', 'F', 'F'],
    #             positions=atomsB_raw.get_positions())
    # atomsB.info["mapped_smiles"] = "[P-:1]([F:2])([F:3])([F:4])([F:5])([F:6])[F:7]"

    # FSI
    # atomsB.info["mapped_smiles"] = "[N-:1]([S:2](=[O:3])(=[O:4])[F:5])[S:6](=[O:7])(=[O:8])[F:9]"
    
    # EMC
    # atomsB.info["mapped_smiles"] = "[C:1]([C:2]([O:3][C:4](=[O:5])[O:6][C:7]([H:13])([H:14])[H:15])([H:11])[H:12])([H:8])([H:9])[H:10]"


    atomsA.positions = posA
    atomsB.positions = posB
    atomsA.pbc = [False, False, False]
    atomsB.pbc = [False, False, False]
    pdbA_frames.append(atomsA)
    pdbB_frames.append(atomsB)


os.makedirs('test', exist_ok=True)
output_folder = f'test/{monA}_{monB}'
os.makedirs(output_folder, exist_ok=True)

write(f'{output_folder}/{monA}_0.xyz', pdbA_frames)
write(f'{output_folder}/{monB}_1.xyz', pdbB_frames)



props = ['TOTAL', 'ELEC_PAULI', 'POLARIZATION', 'DISP', "CHARGE_TRANSFER"]

def convert_pickle_to_json(data):
    data_json = {}
    data_json['TOTAL'] = data['tot_full']
    data_json['POLARIZATION'] = data['pol'] + data['lr_pol']
    data_json['CHARGE TRANSFER'] = data['dhf']
    data_json['CLS ELEC'] = data['es'] + data['lr_es']
    data_json['DISP'] = data['disp'] + data['lr_disp']
    data_json['ELEC_PAULI'] = data['es'] + data['lr_es'] + data['ex']
    data_json['distance'] = data['shift']

    for key in data_json:
        data_json[key] = data_json[key]/4.184 #　kJ/mol to kcal/mol
        data_json[key] = data_json[key].tolist()
    return data_json

data_json = convert_pickle_to_json(data)

with open(f'{output_folder}/EDA.json','w',newline='\n') as f:
    json.dump(data_json, f,indent=2)
