# Installation Scripts

All installation flows are unified into one entry script:

- `install_phyneo_openmm.sh`

## Recommended Entry

```bash
bash /Users/jeremychen/Desktop/Project/project_electrolyte/OpenMM_PhyNEO/PhyNEO/phyneo_openmm/scripts/install_phyneo_openmm.sh --help
```

## Modes

- `--mode phyneo` (default): install/reuse OpenMM, build MPID plugin, install Python deps, run smoke checks.
- `--mode build-openmm`: legacy-style OpenMM source build.
- `--mode build-openmm-vv`: legacy-style openmm-velocityVerlet source build.
- `--mode install-openmm-stack`: clone/build/install openmm + openmm-velocityVerlet.

## Common Examples

```bash
# 1) Recommended phyneo install (OpenMM via conda + MPID plugin)
bash install_phyneo_openmm.sh --mode phyneo

# 2) Reuse existing OpenMM env/prefix
bash install_phyneo_openmm.sh \
  --mode phyneo \
  --install-openmm skip \
  --openmm-prefix "$CONDA_PREFIX" \
  --python-exec "$CONDA_PREFIX/bin/python"

# 3) Legacy-like build from current openmm source directory
bash install_phyneo_openmm.sh \
  --mode build-openmm \
  --legacy-source-dir "$PWD" \
  --openmm-prefix /usr/local/openmm

# 4) Legacy-like build from current openmm-velocityVerlet source directory
bash install_phyneo_openmm.sh \
  --mode build-openmm-vv \
  --legacy-source-dir "$PWD" \
  --openmm-prefix /usr/local/openmm

# 5) Clone + install openmm and openmm-velocityVerlet
bash install_phyneo_openmm.sh \
  --mode install-openmm-stack \
  --openmm-prefix /usr/local/openmm
```

## Compatibility Wrappers

These wrappers are preserved for old workflows and internally call `install_phyneo_openmm.sh`:

- `build_openmm.sh`
- `build_openmm_vv.sh`
- `install.sh`

## Reminder

If you only need the current PhyNEO workflow, use `--mode phyneo` only.
Legacy modes are for compatibility/migration and can require extra system dependencies.
