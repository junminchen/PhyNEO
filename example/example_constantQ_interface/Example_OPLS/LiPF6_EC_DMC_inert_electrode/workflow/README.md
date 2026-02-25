# Workflow (Recommended Order)

1. `01_bulk_equil`: bulk LiPF6-EC-DMC equilibration (NPT -> NVT)
2. `02_interface_mc`: add inert electrodes and do MC gap pre-equilibration
3. `03_cpm`: run constant-potential MD (CPF)
4. `04_analysis`: run post-analysis and equilibration checks

Run each stage from this folder:

```bash
cd workflow/01_bulk_equil && bash run.sh
cd ../02_interface_mc && bash run.sh
cd ../03_cpm && bash run.sh
cd ../04_analysis && bash run.sh
```

Notes:
- Activate your runtime env first (e.g. `conda activate mpid84`).
- Stage 1 writes bulk outputs to sibling folder:
  `/Users/jeremychen/Desktop/Project/project_electrolyte/OpenMM_PhyNEO/PhyNEO/example/example_constantQ_interface/Example_OPLS/LiPF6_EC_DMC_minimal`
