# OPLS LiPF6-EC-DMC with Thick Li Electrodes (OpenMM 8.4 CPF)

This example mirrors the inert-electrode workflow, but uses **multilayer lithium slabs** for both electrodes.

## Important model note
This is a non-reactive constant-potential demo for workflow testing. It does **not** include Li plating/SEI chemistry.

## Files
- `config.json`: composition, slab box, thick-electrode geometry, CPF and MD settings
- `render_packmol.py`, `run_packmol.sh`: build `electrolyte_start.pdb`
- `assemble_thick_li_electrode_system.py`: build multilayer Li slabs + electrolyte
- `mc_gap_equilibrate.py`: MC pre-equilibration of slab gap
- `run_openmm84_thick_li_electrode.py`: CPF MD
- `electrode_residues.xml`, `electrode_ff.xml`: Li slab residue/type definitions
- `analyze_interface_distribution.py`, `analyze_interfacial_capacitance.py`, `visualize_last_frame_electrode_charge.py`: post-analysis

## Quick run
```bash
conda activate mpid84
cd /Users/jeremychen/Desktop/Project/project_electrolyte/OpenMM_PhyNEO/PhyNEO/example/example_constantQ_interface/Example_OPLS/LiPF6_EC_DMC_thick_Li_electrode

bash run_packmol.sh
python assemble_thick_li_electrode_system.py
python mc_gap_equilibrate.py
python run_openmm84_thick_li_electrode.py

python analyze_interface_distribution.py --between-electrodes-only --electrode_margin 1.0 --interface_width 5.0 --nblocks 5
python analyze_interfacial_capacitance.py --skip 2 --nblocks 5
python visualize_last_frame_electrode_charge.py --xy-bins 40
```

## Geometry controls (in `config.json`)
- `electrode.layers_per_electrode`
- `electrode.interlayer_spacing_angstrom`
- `electrode.spacing_angstrom`
- `cell.z_cathode_surface_angstrom`
- `cell.z_anode_surface_angstrom`

## MC density matching controls (`config.json -> mc_gap`)
- `use_bulk_density_target`: read target density from bulk log tail (`bulk_density_log`)
- `initialize_to_target_gap`: affine initialize slab to target gap before MC refinement
- `target_gap_bias_k_kj_mol_per_A2`, `target_gap_pull`: steer MC around target gap
- `enforce_target_density`: fail fast if final slab density misses tolerance

## Default outputs
- `npt_thick_li.log`, `traj_thick_li.dcd`, `final_thick_li.pdb`
- `electrode_charges.log`
- `results/*`

## Density note
- `npt_thick_li.log` reports full-box density (includes vacuum), so it is not suitable for bulk-density matching.
- For pre-production density matching, use the slab density printed by `mc_gap_equilibrate.py` and `run_openmm84_thick_li_electrode.py`.
