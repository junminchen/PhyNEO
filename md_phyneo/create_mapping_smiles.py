from rdkit import Chem
import sys
# 1. 普通 SMILES
# smiles = "COCCOC"
smiles = sys.argv[1]
mol = Chem.MolFromSmiles(smiles)

# 2. 加显式氢
mol = Chem.AddHs(mol)

# 3. 给每个原子加 map number
for i, atom in enumerate(mol.GetAtoms()):
    atom.SetAtomMapNum(i+1)

# 4. 输出带映射的 SMILES
mapped_smiles = Chem.MolToSmiles(mol, canonical=True)
print(mapped_smiles)
