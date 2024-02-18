#!/usr/bin/env python
import sys
import numpy as np
import MDAnalysis as mda

u = mda.Universe('pe6.pdb')

dimer = mda.Merge(u.atoms, u.atoms)
n_atoms_mon = len(u.atoms)
print(n_atoms_mon)

positions = dimer.atoms.positions
positions[0:n_atoms_mon] -= np.array([0.0, 2.5, 0.0])
positions[n_atoms_mon:2*n_atoms_mon] += np.array([0.0, 2.5, 0.0])
positions += np.array([15.0, 15.0, 15.0])
dimer.atoms.positions = positions

dimer.dimensions = u.dimensions
dimer.atoms.write('pe6_dimer.pdb')
