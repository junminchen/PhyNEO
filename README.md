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

`Tinker (for the Local frame generator POLEDIT or you can do it by yourself)`: https://dasher.wustl.edu/tinker/

+ `package`: files for constructing packages or images, such as conda recipe and docker files.
+ `tests`: unit tests.
+ `worfl`: DMFF python codes
+ `dmff/api`: source code of application programming interface of DMFF.
+ `dmff/admp`: source code of automatic differentiable multipolar polarizable (ADMP) force field module.
+ `dmff/classical`: source code of classical force field module.
+ `dmff/common`: source code of common functions, such as neighbor list.
+ `dmff/sgnn`: source of subgragh neural network force field model.
+ `dmff/eann`: source of embedded atom neural network force field model.
+ `dmff/generators`: source code of force generators.
+ `dmff/operators`: source code of operators.

### Short-range Parameters

#### Data Preparation 

#### Training 

+ `dmff/api`: source code of application programming interface of DMFF.

### sub-Graph Neural Networks for Bonding interaction


## Support and Contribution

Please visit our repository on [GitHub](https://github.com/Jeremydream/PhyNEO) for the library source code. Any issues or bugs may be reported at our issue tracker. All contributions to DMFF are welcomed via pull requests!
