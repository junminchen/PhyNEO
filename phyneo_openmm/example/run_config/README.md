# Packmol Bulk EC + caff_5_mpid_slater_bond.xml

This folder is self-contained for `phyneo_openmm.phyneo_protocol`.

## Files
- `bulk_ec_packmol.pdb`: Packmol-style bulk EC box.
- `caff_5_mpid_slater_bond.xml`: PhyNEO OpenMM XML.
- `config_packmol_bulk_ec_transport.json`: transport protocol config (NPT/NVT/noneq).
- `run.sh`: direct launcher.

## Run
```bash
cd "$(pwd)"
bash ./run.sh
```

## Notes
- Paths in JSON are relative (`./bulk_ec_packmol.pdb`, `./caff_5_mpid_slater_bond.xml`).
- Default protocol type is `transport`.
- MPID scales are enabled with
  - `m_scales=[0,0,0,0,0]`
  - `p_scales=[0,0,0,0,0]`
  - `d_scales=[1,1,1,1,1]`
