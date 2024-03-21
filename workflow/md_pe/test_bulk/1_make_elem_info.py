#!/usr/bin/env python
import sys
import numpy as np
import MDAnalysis as mda
import MDAnalysis.transformations as trans

u = mda.Universe(sys.argv[1])
ts = u.trajectory[0]
ts.triclinic_dimensions = np.eye(3) * 50

# add bonds
if not hasattr(u, 'bonds'):
    bonds = mda.topology.guessers.guess_bonds(u.atoms, u.atoms.positions)
    u.add_TopologyAttr('bonds', bonds)

if not hasattr(u, 'elements'):
    atomtype = []
    chainIDs = []
    charge = []
    for i in range(len(u.atoms)):
        atomtype.append(mda.topology.guessers.guess_atom_type(u.atoms[i].name))
        charge.append(mda.topology.guessers.guess_atom_charge(u.atoms[i].name))
        chainIDs.append('1')
    u.add_TopologyAttr('elements', atomtype)
    u.add_TopologyAttr('chainIDs', chainIDs)
    u.add_TopologyAttr('charge', charge)    

ag = u.atoms
# transforms = trans.center_in_box(ag, wrap=False)
# ag.universe.trajectory.add_transformations(transforms)

ag.write(sys.argv[1])