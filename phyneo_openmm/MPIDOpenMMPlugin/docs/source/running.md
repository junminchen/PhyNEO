# Running the Plugin

The following is a brief summary of the [simple
examples](https://github.com/andysim/MPIDOpenMMPlugin/tree/master/examples)
included in the repository.  The usage of the plugin follows [standard OpenMM
practices](http://docs.openmm.org/latest/userguide/index.htmlr)] so that would
be a good place to start for anybody unfamiliar with OpenMM.

## Specifying the Parameters

### Units

OpenMM uses the following units:

- Time: $ps$
- Energy: $kJ/mol$
- Distance: $nm$
- Mass: $amu$
- Temperature: $K$
- Charge: $e$
- Angle: $rad$

This choice ensures that forces obtained from $F=ma$ are consistent with those
obtained from $F=-\frac{\mathrm{d}U}{\mathrm{d}R}$ without the need for scale
factors.

## The XML file

Parameters should be specified using a standard [OpenMM XML
file](http://docs.openmm.org/latest/userguide/application.html#writing-the-xml-file).
As an illustrative example, here is the full specification of the MPID
implementation of the [SWM6](https://doi.org/10.1063/1.4774577) water model:

``` xml
<ForceField>
 <AtomTypes>
  <Type name="OT" class="OW" element="O" mass="15.999"/>
  <Type name="HT" class="HW" element="H" mass="1.008"/>
 </AtomTypes>
 <Residues>
  <Residue name="HOH">
   <Atom name="H1" type="HT"/>
   <Atom name="H2" type="HT"/>
   <Atom name="O" type="OT"/>
   <Bond from="0" to="2"/>
   <Bond from="1" to="2"/>
 </Residue>
 </Residues>
 <HarmonicBondForce>
  <Bond class1="OW" class2="HW" length="0.09572" k="376560"/>
 </HarmonicBondForce>
 <HarmonicAngleForce>
  <Angle class1="HW" class2="OW" class3="HW" angle="1.82421813418" k="460.24"/>
 </HarmonicAngleForce>
 <NonbondedForce coulomb14scale="0.833333" lj14scale="0.5">
  <Atom type="OT" charge="-0.0" sigma="0.31983264" epsilon="0.677808"/>
  <Atom type="HT" charge="0.0" sigma="1" epsilon="0"/>
 </NonbondedForce>
 <MPIDForce coulomb14scale="0.833333" defaultTholeWidth="8.0">
   <Multipole type="OT" kz="-HT" kx="-HT"
             c0="-1.0614"
             dX="0.0" dY="0.0"  dZ="-0.023671684"
             qXX="0.000150963" qXY="0.0" qYY="0.00008707" qXZ="0.0" qYZ="0.0" qZZ="-0.000238034"
             oXXX="0.0" oXXY="0.0" oXYY="0.0" oYYY="0.0" oXXZ="0.000000426" oXYZ="0.0" oYYZ="0.000000853" oXZZ="0.0" oYZZ="0.0" oZZZ="-0.000001279"
             />
   <Multipole type="HT" kz="OT" kx="HT"
             c0="0.5307"
             dX="0.0" dY="0.0"  dZ="0.0"
             qXX="0.0" qXY="0.0" qYY="0.0" qXZ="0.0" qYZ="0.0" qZZ="0.0"
             oXXX="0.0" oXXY="0.0" oXYY="0.0" oYYY="0.0" oXXZ="0.0" oXYZ="0.0" oYYZ="0.0" oXZZ="0.0" oYZZ="0.0" oZZZ="0.0"
             />
   <Polarize type="OT" polarizabilityXX="0.00088" polarizabilityYY="0.00088" polarizabilityZZ="0.00088" thole="8.0"/>
   <Polarize type="HT" polarizabilityXX="0.000" polarizabilityYY="0.000" polarizabilityZZ="0.000" thole="0.0"/>
 </MPIDForce>
</ForceField>
```

Most of this file sets up bonded terms, which are well described in the OpenMM
documentation linked above.  When defining the `NonbondedForce`, note that the
charges are defined as zero; this is because all electrostatic terms are to be
handled by the `MPIDForce`.  The `MPIDForce` does not handle Lennard-Jones (LJ)
terms, so these are still processed by the `NonbondedForce`, which can also use
particle mesh Ewald (PME) to compute the LJ terms effectively without cutoffs.
The section defining the `MPIDForce` terms looks like this:

``` xml
 <MPIDForce coulomb14scale="0.833333" defaultTholeWidth="8.0">
   <Multipole type="OT" kz="-HT" kx="-HT"
             c0="-1.0614"
             dX="0.0" dY="0.0"  dZ="-0.023671684"
             qXX="0.000150963" qXY="0.0" qYY="0.00008707" qXZ="0.0" qYZ="0.0" qZZ="-0.000238034"
             oXXX="0.0" oXXY="0.0" oXYY="0.0" oYYY="0.0" oXXZ="0.000000426" oXYZ="0.0" oYYZ="0.000000853" oXZZ="0.0" oYZZ="0.0" oZZZ="-0.000001279"
             />
   <Multipole type="HT" kz="OT" kx="HT"
             c0="0.5307"
             dX="0.0" dY="0.0"  dZ="0.0"
             qXX="0.0" qXY="0.0" qYY="0.0" qXZ="0.0" qYZ="0.0" qZZ="0.0"
             oXXX="0.0" oXXY="0.0" oXYY="0.0" oYYY="0.0" oXXZ="0.0" oXYZ="0.0" oYYZ="0.0" oXZZ="0.0" oYZZ="0.0" oZZZ="0.0"
             />
   <Polarize type="OT" polarizabilityXX="0.00088" polarizabilityYY="0.00088" polarizabilityZZ="0.00088" thole="8.0"/>
   <Polarize type="HT" polarizabilityXX="0.000" polarizabilityYY="0.000" polarizabilityZZ="0.000" thole="0.0"/>
 </MPIDForce>
```

The global `coulomb14scale` parameter controls the scale factor applied to
interactions that are topologically three bonds apart, and defaults to 1 (the
correct value for MPID) if omitted.  The `defaultTholeWidth` parameter is the
[Thole damping] (technical.md#thole-damping) parameter that's applied to
particles whose interaction is not neglected due to topological reasons.

Multipoles are defined by providing orientation rules, defining their
orientation with respect to "anchor" atoms within the same system and the
syntax follows the [conventions
used](https://pubs.acs.org/doi/10.1021/ct200304d) in the AMOEBA force field.
For example, `Multipole type="OT" kz="-HT" kx="-HT"` defines a multipole on the
`OT` atom, whose z direction is defined as the bisector of the two `HT` atoms
directly connected to it, with the $x$ direction defined by the plane
containing both `HT`.  The `Multipole type="HT" kz="OT" kx="HT"` similarly
defines a multipole on `HT`, whose local `z` axis is defined by its bond to the
`OT` atom; the $x$ axis is then defined by the plane containing the other `HT`
atom.

The `c0` entry defines the charge (in $e$), while `dX` defines the $x$
component of the dipole in $e/nm$, _etc._.  Multipoles up to octopoles are
supported, and any values omitted are assumed to be zero.

The `Polarize` tag is optional and is used to define a given atom as
polarizable.  The $xx$, $yy$ and $zz$ elements of the polarizability tensor
should be specified individually, in the axis system used to define the
multipoles described above.  If all three components of the polarizability
tensor are equal, the polarizability is isotropic and the orientation rules are
irrelevant.  The Thole damping parameter, [detailed
here](technical.md#thole-damping) is a unitless parameter used to dampen
topologically excluded interactions between pairs of induced dipoles if the
chosen solver considers them.

## Creating a System

With the appropriate XML-formatted force field in hand, setting up a simulation
follows the [standard OpenMM
approach](http://docs.openmm.org/latest/userguide/application.html#a-first-example).
The `createSystem()` function from the `ForceField` class, which is used to
build the system obeys all of the [usual
arguments](http://docs.openmm.org/development/api-python/generated/openmm.app.forcefield.ForceField.html#openmm.app.forcefield.ForceField.createSystem)
for controlling constraint algorithms, hydrogen mass repartitioning,
cutoffs, _etc._.  For `MPIDForce`, the `defaultTholeWidth` and
`coulomb14scale` arguments may be provided, overriding any values that may be
present in the XML parameter file described above.  The `polarization` argument
is used to control the [polarization solver](technical.md#solvers).
