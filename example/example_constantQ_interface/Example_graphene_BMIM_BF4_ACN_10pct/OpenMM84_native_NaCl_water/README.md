# OpenMM 8.4 Native ConstantPotentialForce: NaCl(aq) + Inert Electrodes

This example creates a simple NaCl aqueous box (TIP3P water) between two inert electrode sheets and runs native OpenMM 8.4 `ConstantPotentialForce`.

## Files
- `build_nacl_water_box.py`: build initial `nacl_water_start.pdb`
- `run_openmm84_native_nacl_water.py`: run MD with native `ConstantPotentialForce`
- `electrode_residues.xml`, `electrode_ff.xml`: electrode templates/parameters

## 1) Build the box
```bash
conda run -n mpid84 python build_nacl_water_box.py \
  --output nacl_water_start.pdb \
  --ionic-strength-m 1.0 \
  --z-liq-min-nm 1.2 \
  --z-liq-max-nm 4.8
```

## 2) Run a smoke test
```bash
conda run -n mpid84 python run_openmm84_native_nacl_water.py \
  --pdb nacl_water_start.pdb \
  --platform CPU \
  --steps 500 \
  --report-interval 100 \
  --voltage-v 1.0
```

## Notes
- Water/ions use `amber14/tip3p.xml`.
- Electrode atoms are in chains `A` (cathode) and `B` (anode).
- This builder enforces a liquid slab and keeps vacuum layers outside it.
- Default geometry: cathode at 1.0 nm, anode at 5.0 nm, liquid in [1.2, 4.8] nm.
