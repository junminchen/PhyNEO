# Single-Layer Inert Graphene Electrode (Fixed-Charge) Example

This example runs LiPF6/EC/DMC between **single-layer graphene-like inert electrodes** with **fixed electrode charges** (not constant-potential).

## Files
- `config.json`: geometry, force field paths, MD settings, fixed charge magnitude
- `assemble_singlelayer_graphene_system.py`: build `start_fixedcharge_graphene.pdb`
- `run_fixedcharge_singlelayer_graphene.py`: fixed-charge MD
- `electrode_residues.xml`, `electrode_ff.xml`: inert graphene templates

## Quick start
```bash
conda run -n mpid python assemble_singlelayer_graphene_system.py
conda run -n mpid python run_fixedcharge_singlelayer_graphene.py
```

## Important settings
- `electrode.fixed_charge_per_atom_e`
  - cathode atoms use `+q`
  - anode atoms use `-q`
- `electrode.fix_electrode_positions`
  - `true` means graphene atoms are frozen (mass set to 0)

## Outputs
- `nvt_fixedcharge.log`
- `traj_fixedcharge.dcd`
- `electrode_fixed_charges.log`
- `final_fixedcharge.pdb`
