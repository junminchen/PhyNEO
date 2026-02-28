# Example_graphene_BMIM_BF4_ACN_10pct Workflow

This folder contains two simulation paths for constant-potential electrode/electrolyte simulations:

- Legacy fixed-voltage workflow (`simtk` APIs + custom Poisson solver)
- OpenMM 8.4 native `ConstantPotentialForce` workflow

## 1. What is `MC_equil` doing?

`MC_equil` in this project is a Monte Carlo barostat/electrode-spacing equilibration stage used before production fixed-voltage MD.

In practice, it is used to:

- relax initial packing stress between two electrodes,
- adjust effective confined density under interfacial constraints,
- reduce large energy/charge transients when switching to `Constant_V`.

So, it is not mathematically mandatory, but it is strongly recommended for stable startup.

## 2. Is MC equilibration required?

Short answer:

- Not strictly required.
- Usually recommended for this confined interface geometry.

If you pre-equilibrate density in bulk (no electrodes), that is fine as a pre-step, but after inserting into the electrode slit you should still do a short re-equilibration in the confined system (MC or short restrained MD), because interfacial layering changes local density/structure.

## 3. End-to-end workflow (legacy path)

### Step A: Prepare initial structure

Input structure for MC stage is:

- `MC_equilibrate/start.pdb`

Force-field dependencies:

- `graphene_ffdir/*.xml`
- SAPT files (`sapt.xml`, `sapt_residues.xml`) are downloaded/merged by script.

### Step B: MC density/interface equilibration

Script:

- `MC_equilibrate/run_openMM.py` with `simulation_type = "MC_equil"`

Main outputs:

- `MC_equilibrate/equil_MC.dcd`
- `MC_equilibrate/equilibrated_MC.pdb`
- `MC_equilibrate/start_drudes.pdb`

Typical cluster submit:

- `MC_equilibrate/run.pbs`

### Step C: Clean last MC snapshot for production restart

Script:

- `MC_equilibrate/pull_last_snapshot_clean.py`

Purpose:

- remove Drude particles from final snapshot,
- repair residue/atom naming issues from trajectory export,
- use `start.pdb` as template for selected electrode residues.

Use pattern:

```bash
cd MC_equilibrate
python pull_last_snapshot_clean.py > ../Applied_Voltage/equilibrated.pdb
```

### Step D: Constant-voltage production MD

Script:

- `Applied_Voltage/run_openMM.py` with `simulation_type = "Constant_V"`

Default setup in script:

- `Voltage = 2.0` V
- chain-index electrodes: cathode `(0,2)`, anode `(1,3)`

Main outputs:

- `Applied_Voltage/FV_NVT.dcd`
- `Applied_Voltage/charges.dat`
- `Applied_Voltage/state.chk` (restart checkpoint)

Typical cluster submit:

- `Applied_Voltage/run.pbs`

## 4. End-to-end workflow (OpenMM 8.4 native path)

There are two related native scripts:

- Minimal NaCl/water example: `OpenMM84_native_NaCl_water/*`
- Legacy-input-compatible native runner: `Applied_Voltage/run_openMM_84_native.py`

For this folder's production system, use:

- `Applied_Voltage/run_openMM_84_native.py`

Typical outputs:

- `Applied_Voltage/charges_openmm84.dat`
- `Applied_Voltage/state_openmm84.chk`
- optional trajectory if `--traj` is set.

## 5. Recommended practical options

### Option 1 (most robust)

`MC_equil` -> snapshot clean -> `Constant_V`

### Option 2 (faster but still safe)

bulk pre-equilibration -> build/interface insertion -> short confined re-equilibration -> `Constant_V`

### Option 3 (not recommended)

directly start `Constant_V` from unequilibrated confined structure.

