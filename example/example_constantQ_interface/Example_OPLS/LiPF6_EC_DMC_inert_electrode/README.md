# OPLS LiPF6-EC-DMC with Inert Electrodes (OpenMM 8.4 CPF)

This workflow builds an electrolyte slab (with vacuum layers) and adds two inert electrode sheets, then runs native OpenMM `ConstantPotentialForce`.

Recommended staged workflow is under `workflow/`:
- `workflow/01_bulk_equil`
- `workflow/02_interface_mc`
- `workflow/03_cpm`
- `workflow/04_analysis`

## Files
- `config.json`: composition, box/slab/electrode/MD settings
- `render_packmol.py`, `packmol.inp`, `run_packmol.sh`: build `electrolyte_start.pdb`
- `assemble_inert_electrode_system.py`: create `start_with_electrodes.pdb`
- `run_openmm84_inert_electrode.py`: run NPT + CPF with charge logging
- `electrode_residues.xml`, `electrode_ff.xml`: inert electrode templates

## Build steps
1. Build electrolyte slab with Packmol:
```bash
bash run_packmol.sh
```

2. Add inert electrodes:
```bash
python assemble_inert_electrode_system.py
```

3. Optional MC pre-equilibration of plate gap (move anode and rescale electrolyte slab):
```bash
python mc_gap_equilibrate.py
```

4. Run MD (example):
```bash
python run_openmm84_inert_electrode.py --equil-steps 5000 --prod-steps 10000 --report-interval 500
```

5. Analyze interfacial distribution (z-profile + adsorption + block uncertainty):
```bash
python analyze_interface_distribution.py \
  --top start_with_electrodes_mc.pdb \
  --traj traj_inert.dcd \
  --config config.json \
  --between-electrodes-only \
  --electrode_margin 1.0 \
  --interface_width 5.0 \
  --nblocks 5
```

6. Analyze interfacial capacitance from constant-potential charge log:
```bash
python analyze_interfacial_capacitance.py \
  --config config.json \
  --charge-log electrode_charges.log \
  --skip 2 \
  --nblocks 5
```

## Outputs
- `npt_inert.log`
- `traj_inert.dcd`
- `electrode_charges.log`
- `final_inert.pdb`
- `start_with_electrodes_mc.pdb` (if MC step used)
- `mc_gap.log` (if MC step used)
- `results/interface_distribution.csv`
- `results/interface_distribution.png`
- `results/interface_distribution_charge.png`
- `results/interface_distribution_summary.txt`
- `results/interfacial_capacitance_summary.txt`

## Notes
- Electrolyte uses: `../opls_salt.xml` + `../opls_solvent.xml`
- Electrode sheets are on chain 0 (cathode) and chain 1 (anode)
- Electrolyte is packed only in `z_liq_min_angstrom` to `z_liq_max_angstrom`, leaving vacuum layers above/below.
- `run_openmm84_inert_electrode.py` will prefer `start_with_electrodes_mc.pdb` automatically if it exists.
