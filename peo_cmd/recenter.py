#!/usr/bin/env python
import sys
import MDAnalysis as mda
import numpy as np

u = mda.Universe(sys.argv[1])
r_com = u.atoms.center_of_mass()

box = u.trajectory.ts.triclinic_dimensions
# u.atoms.positions += (np.sum(box, axis=0)/2 - r_com)
u.atoms.positions -= r_com

u.atoms.write('out.pdb')
