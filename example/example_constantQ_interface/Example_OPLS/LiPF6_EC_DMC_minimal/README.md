# Minimal OPLS Recipe: LiPF6-EC-DMC

This folder provides a minimal bulk electrolyte setup under OPLS force field files in `Example_OPLS`.

## Composition
- Li: 64
- PF6: 64
- EC: 256
- DMC: 512
- Cubic box: 55 x 55 x 55 A

## Files
- `config.json`: composition, box, force field, MD defaults
- `render_packmol.py`: generate `packmol.inp` from `config.json`
- `packmol.inp`: Packmol placement input
- `run_packmol.sh`: build `start.pdb` (requires `packmol`)
- `check_forcefield_mapping.py`: quick FF-template consistency check (no MD)
- `run_md_opls.py`: NPT equil + production script (CPU)

## Usage
1. Build Packmol input:
```bash
python render_packmol.py
```
2. Build initial box:
```bash
bash run_packmol.sh
```
3. Sanity check force-field mapping:
```bash
python check_forcefield_mapping.py
```
4. Run MD:
```bash
python run_md_opls.py
```

## Notes
- `EC.pdb` uses residue `ECA` and is compatible with `opls_solvent.xml`.
- `Li.pdb` uses residue `LiA` and is compatible with `opls_salt.xml`.
- This is a bulk liquid recipe (no electrode constraints applied here).
