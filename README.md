# PhyNEO

[![DOI: 10.1021/acs.jctc.3c01045](https://img.shields.io/badge/DOI-10.1021%2Facs.jctc.3c01045-blue)](https://pubs.acs.org/doi/10.1021/acs.jctc.3c01045)
 
## About PhyNEO

**PhyNEO** (**Phy**sics-driven and **N**eural-Network **E**nhanced **O**rganic and polymer Force Field) is a force field development workflow, based on [DMFF](https://github.com/deepmodeling/DMFF). PhyNEO features a hybrid approach that combines both the physics-driven and the data-driven methods and is able to generate a bulk potential with chemical accuracy using only quantum chemistry data of very small
clusters. Careful separations of long-/short-range interactions and nonbonding/bonding interactions are the key to the success of PhyNEO. By such a strategy, PhyNEO mitigate the limitations of pure data-driven methods in long-range interactions, thus largely increasing the data efficiency and the scalability of machine learning models.

### License and credits

The project PhyNEO is licensed under [GNU LGPL v3.0](LICENSE). If you use this code in any future publications, please cite this using `CHEN, Junmin; YU, Kuang. PhyNEO: A Neural-Network-Enhanced Physics-Driven Force Field Development Workflow for Bulk Organic Molecule and Polymer Simulations. Journal of Chemical Theory and Computation, 2023, 20.1: 253-265. DOI: 10.1021/acs.jctc.3c01045`

## User Guide

+ `md_example`: demos in papers interfaced i-Pi.

### Long-range Parameters

Dependency: 
`CAMCASP`: https://www-stone.ch.cam.ac.uk/programs.html
`Tinker (for the Local frame generator POLEDIT or you can do it by yourself)`: https://dasher.wustl.edu/tinker/

Firstly, we need to use CAMCASP to do TD-DFT and ISA-pol calculation. See it in 'conf.DMC'.
Then, we can use our scripts to generate XML force field file with the help of Tinker tool.
+ `worflow/lr_param/poledit`: Tinker files for constructing local frame definition to 'localframe'. 
```bash
./poledit DMC.xyz
```
Notes: JUST PRESS 'ENTER' TO LAST QUESTIONS!!! All we need is copy the local frame definition to 'localframe' file.
```txt
Local Frame Definition for Multipole Sites :

     Atom     Name      Axis Type     Z Axis  X Axis  Y Axis

       1      C         Z-then-X         2       3       0
       2      O         Z-then-X         3       1       0
       3      C         Bisector         2       5       0
       4      O         Z-Only           3       0       0
       5      O         Z-then-X         3       6       0
       6      C         Z-then-X         5       3       0
       7      H         Z-then-X         1       2       0
       8      H         Z-then-X         1       2       0
       9      H         Z-then-X         1       2       0
      10      H         Z-then-X         6       5       0
      11      H         Z-then-X         6       5       0
      12      H         Z-then-X         6       5       0
```
+ `worflow/lr_param/1_gen_atype.py`: generate atom type definition to 'atype_data.pickle'.
```bash
python 1_gen_atype.py
```
+ `worflow/lr_param/2_ff_gen.py`: generate PhyNEO forcefield by one step.
```bash
python 2_ff_gen.py > ff.xml
```

### Short-range Parameters

Dependency: 
`Molpro (or you can use Psi4)`: https://www.molpro.net

#### Data Preparation 

+ `worflow/sr_param/abinitio/run_md.py`: generate the trajectory of dimer by classical force field MD simulation
+ `worflow/sr_param/abinitio/geoms_op.py`: generate dimer scan Molpro input file to do SAPT calculation
+ `worflow/sr_param/abinitio/sapt/pack_data.py`: pack dimer scan data to 'data.pickle'

#### Training 

+ `worflow/sr_param/remove_lr.py`: separate the long-range interaction energy to fitting short-range parameters 
+ `worflow/sr_param/fit_basepair.py`: fit the short-range parameters by dimer scan data, and save it in 'params.pickle', then plot it in 'test_deomp3.png'.

### sub-Graph Neural Networks for Bonding interaction

To be continued.

## Support and Contribution

Please visit our repository on [GitHub](https://github.com/Jeremydream/PhyNEO) for the library source code. Any issues or bugs may be reported at our issue tracker. All contributions to PhyNEO are welcomed via pull requests!
