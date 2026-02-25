# MPID Simulation Workdir

This directory contains a standalone version of the PhyNEO MPID simulation for OpenMM.

## Contents
- `run_sim.py`: The main simulation script (CUDA/CPU).
- `phyneo_openmm/`: Local Python library containing the `phyneo_protocol` module.
- `ec_box_35A.pdb`: Input PDB structure.
- `caff_5_mpid_slater_bond.xml`: MPID force field parameters.
- `run.sh`: Helper shell script to run the simulation.

## Prerequisites
- A Conda environment with OpenMM and the MPID plugin installed.
- **Reminder**: You must activate your `mpid` environment before running:
  ```bash
  conda activate mpid
  ```
- The `mpidplugin` Python package available in your environment.

## Running the simulation
You can run the simulation directly using:
```bash
bash run.sh
```
Or manually:
```bash
export OPENMM_PLUGIN_DIR=$CONDA_PREFIX/lib/plugins
python run_sim.py
```
