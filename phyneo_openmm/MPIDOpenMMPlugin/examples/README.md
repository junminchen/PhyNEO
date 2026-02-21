Illustrative Examples
================
This folder contains some simple snippets to show how the MPID plugin can be
used.

waterbox
------------
This is a ~31 Å waterbox, holding 996 water molecules.  It is set up to be able
to run the standard TIP3P model, as well as the analytic MPID variants of the
SWM4 and SWM6 Drude water models.

water_dimer
----------------
This is a simple water dimer, whose intermolecular distance can be varied to
directly study the difference between the Drude and MPID models.

ethane_water_charge_only
---------------------------------
A single ethane molecule in a 35 Å box containing 1383 water molecules.  The
initial PDB file was constructed using `openmm-setup` and the XML parameters
were extracted from the CHARMM36 parameter file that ships with OpenMM, with
arbitrarily chosen polarizability settings assigned to the heavy atoms. The
`run_ethane.py` example runs a short trajectory of this system, while the
`alchemy_example.py` file demonstrates how to make alchemical changes to the
ethane molecule to allow it to disappear in the water box.
