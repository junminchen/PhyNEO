# Graphene + Lithium Electrolyte Example (Fixed Charge and MPIDForce)

This example mirrors the layout of `Example_graphene_BMIM_BF4_ACN_10pct` but is tailored for lithium battery electrolytes. It provides two runnable configurations:

- **FixedCharge**: uses the non-polarizable force field you provide.
- **MPIDForce**: uses the polarizable MPIDForce parameters you provide (no Drude oscillators).

## Preparation Workflow

1. **Place your files**  
   Put your electrolyte building-block PDB files and force field XML files in `forcefield/`. Expected defaults:
   - `li_fixed_charge.xml` and `li_fixed_charge_residues.xml`
   - `li_mpid.xml` and `li_mpid_residues.xml`
   - Any molecule fragments referenced by the Packmol template

2. **Packmol modeling**  
   Edit `packmol/packmol_li_electrolyte.inp` so the file names, molecule counts, and box dimensions match your system (graphene spacing, salt concentration, solvent ratio). Run:
   ```bash
   cd packmol
   packmol < packmol_li_electrolyte.inp
   ```
   This produces `packmol/li_electrolyte_box.pdb`, which both workflows use as their default starting structure.

3. **Pre-equilibrate (no applied voltage)**  
   Use Monte Carlo equilibration to relax the density:
   ```bash
   cd ../FixedCharge
   SIM_MODE=MC_equil FV_PLATFORM=Reference INPUT_PDB=../packmol/li_electrolyte_box.pdb python run_openMM.py
   ```
   Adjust `SIM_TIME_NS` or the platform as needed. The script writes `equilibrated.pdb` for the production step.

4. **Apply voltage and run interface MD**  
   Switch the same script to constant-voltage mode:
   ```bash
   SIM_MODE=Constant_V APPLIED_VOLTAGE=2.0 INPUT_PDB=equilibrated.pdb python run_openMM.py
   ```
   The resulting `FV_NVT.dcd` trajectory contains the interfacial solvation structure; `charges.dat` tracks electrode charges.

5. **Constant charge mode (恒电荷模式)**  
   Alternatively, run with fixed electrode charges instead of fixed voltage:
   ```bash
   SIM_MODE=Constant_Q CATHODE_CHARGE=1.0 INPUT_PDB=equilibrated.pdb python run_openMM.py
   ```
   This sets a uniform total charge of `CATHODE_CHARGE` (in elementary charge units) on the cathode and `-CATHODE_CHARGE` on the anode. The charges remain constant throughout the simulation. The trajectory is written to `FQ_NVT.dcd`.

6. **Polarizable MPIDForce variant**  
   After building the Packmol structure, run the MPIDForce workflow with your polarizable XML files:
   ```bash
   cd ../MPIDForce
   SIM_MODE=MC_equil FV_PLATFORM=Reference INPUT_PDB=../packmol/li_electrolyte_box.pdb python run_openMM.py
   SIM_MODE=Constant_V APPLIED_VOLTAGE=2.0 INPUT_PDB=equilibrated.pdb python run_openMM.py
   ```
   Ensure your MPIDForce plugin and parameters are installed; the script does not use Drude oscillators.

## Notes

- Graphene electrode force field files are reused from `Example_graphene_BMIM_BF4_ACN_10pct/graphene_ffdir/` by default. Override `GRAPHENE_FF_DIR` if you relocate or modify them (e.g., `export GRAPHENE_FF_DIR=/path/to/graphene_ffdir` in the same shell session or prefix the run command).
- The scripts automatically pick `equilibrated.pdb` when present; otherwise they fall back to the Packmol output.
- Adjust `cathode_index`/`anode_index` or the Packmol box to match your electrode placement. The defaults assume two graphene sheets separated along *z*; the cathode uses chain IDs 0 and 2, while the anode uses chain IDs 1 and 3 (the tuples list the chain indices that belong to each electrode).
