# Polarization

The induced dipoles are the solution to the $3N\times3N$ system of equations

$$
\boldsymbol{\mu}_\mathrm{ind} = \mathbf{T}^{-1} \mathbf{E}
$$

where the electric field due to fixed multipoles is $\mathbf{E}$ and the
coupling matrix

$$
\mathbf{T} = \boldsymbol{\alpha}^{-1} + \boldsymbol{\tau}
$$

is given by the inverse of the diagonal matrix of atomic polarizabilities and the
off-diagonal matrix that couples pairs of induced dipoles.  When generating the
permanent field $\mathbf{E}$, topologically excluded (1-2 and 1-3) interactions
are neglected, and any 1-4 interactions are scaled, if requested.  All pairs of
induced dipoles are allowed to interact, which is a key feature of the damping
procedure described below.

## Solvers

A few solvers are implemented for evaluating the induced dipoles, which we will
briefly discuss.  These can be selected by passing the appropriate string to
the `polarization` argument of `createSystem()`.

### Direct Solver

The simplest "direct" solver simply uses the approximation

$$
\mathbf{T} \approx \boldsymbol{\alpha}^{-1}
$$

which effectively removes the mutual interaction between induced dipoles.
Because $\boldsymbol{\alpha}$ is diagonal, it is trivially invertible and the
solution is fully analytic.  As a result, the energies and forces are
rigorously consistent and integration can be performed exactly as for
conventional fixed point charges.  However, this simplicity comes at the
expense of the quality of the description of the electrostatics.

### Mutual Solver

The "mutual" solver keeps the interaction between induced dipoles, making
$\mathbf{T}$ difficult to invert.  The induced dipole equations are solved
iteratively, with the convergence accelerated by the direct inversion of the
iterative subspace (DIIS) technique.  Because the coupling between dipoles is
present this is the most accurate method, but it is also the most costly due to
the need for multiple iterations to solve the equations.  While it might be
tempting for to lower the convergence criterion to save time, this will lead to
inconsistencies in the energies and forces, and runaway heating (or cooling) of
the microcanonical ensemble will result.  We recommend converging the equation
to at least $10^{-4}$ if using this solver; the convergence is controlled by
the `mutualInducedTargetEpsilon` argument to `createSystem()`, with a
default value of $10^{-5}$.

### Extrapolated Solver

Combining the strengths of both approaches, the $n$th order Optimized
Perturbation Theory [OPTn](http://dx.doi.org/10.1063/1.4964866) solver treats
the induced dipole coupling matrix as a small perturbation to the "direct"
model, leading to a power series expansion that can be truncated at any order.
Some empirical tuning yields the OPT$n$ family of methods.  At zeroth order,
the "direct" solution is obtained.  Successively increasing the order increases
the cost, with each order costing the equivalent of an iteration of the
"mutual" solver.  Unlike loosely converged "mutual" solutions, the OPT$n$
solutions have energies and forces that are rigorously consistent at all levels
of trunctions.  We recommend the third order "extrapolated" solver, OPT3, which
is the default solver in this plugin.  Other levels of OPT solver may be
selected by calling the `setExtrapolationCoefficients()` on the `MPIDForce`
object.

##Thole Damping

If a pair of polarizable particles get too close, they can strongly polarize
each other and at some distance this overpolarization will become infinite;
this is often refered to as the "polarization catastrophe".  Thole
[introduced](https://doi.org/10.1016/0301-0104(81)85176-2) a clever damping
that effectively gives the point dipoles a finite width and removes the
singularity responsible for terms blowing up at short range.  The density
ascribed to MPID's induced dipoles corresponds to Thole's $\rho_1$ choice (in
contrast to AMOEBA, which uses $\rho_2$):

$$
 \rho_1 = \frac{a^3}{8\pi}e^{-a u}
$$

where $a$ is a unitless width parameter and $u$ is a dimensionless distance.
The result of introducing this density is that interactions between a pair of
induced dipoles is scaled by a factor

$$
 1 - e^{-a u}(1 + au + \frac{(a u)^2}{2}).
$$

For [MPID](https://doi.org/10.1063/1.4984113), the dimensionless distance for a
pair of interacting atoms $i$ and $j$ is given by their separation, $R_{ij}$,
and the individual atomic polarizabilities $\alpha_i$ and $\alpha_j$ by

$$
 u = \frac{R_{ij}}{\alpha_i\alpha_j}.
$$

If the polarizability of and atom is anisotropic, the average of the three
diagonal tensor elements is used.  The dimensionless width parameter $a$ is
given a default value that is used for any interaction that is not excluded
(1-2, 1-3 and, if `coulomb14scale` is zero, 1-4 connected pairs); this default
value is controlled by the `defaultTholeWidth` argument to either the XML force
field file or `createSystem()`.  The excluded interactions do not contribute to
permanent-permenent moment interactions, or to forming the electric field
$\mathbf{E}$ used to define the induced dipoles.  For `mutual` and
`extrapolated` solvers, the induced dipoles are allowed to interact with each
other.  These interactions are not subject to excluded rules.  For pairs that
are not excluded due to topology, the `defaultTholeWidth` parameter is used to
define $a$.  For pairs that are excluded due to topology, we use the individual
Thole widths defined in the parameter file and define $a=a_i + a_j$.
