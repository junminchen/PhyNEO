#About

This [OpenMM](https://openmm.org/) plugin implements polarizable multipole electrostatics, primarily aimed at
supporting the [MPID](https://doi.org/10.1063/1.4984113) formulation of the
CHARMM Drude force field.  However, the code is quite general and many features
can be compiled on-the-fly, allowing it to tailor features as required without
incurring a performance penalty.

Supported features include:

* Particle mesh Ewald electrostatics.
* Multipoles (up to octopoles).
* Induced dipoles, with a range of solvers to evaluate them.
* Isotropic or anisotropic polarizability.
