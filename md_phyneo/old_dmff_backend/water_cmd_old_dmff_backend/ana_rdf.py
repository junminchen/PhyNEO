#!/usr/bin/env python
import sys
import MDAnalysis as mda
from MDAnalysis.analysis import rdf
import matplotlib.pyplot as plt
import numpy as np
from MDAnalysis.topology.guessers import guess_atom_element
from MDAnalysis.topology.guessers import guess_types
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis as HBA
import MDAnalysis.transformations as trans
from MDAnalysis.analysis import distances

font={'family':'sans-serif',
    'style':'normal',
    'weight':'normal',
    'size':10,
    'color':'grey'
}

def get_dmff_uni():
    pdb_file = 'density_0.03338.pdb'
    traj_file = 'simulation.pos_0.xyz'
    u = mda.Universe(pdb_file,traj_file,dt=0.0005) #load files
    # read the box information
    cell = []
    fo = open(traj_file,'r')
    for line in fo.readlines():
        if line.startswith('# CELL(abcABC):'):
            words = line.split("   ")
            cell.append([words[1],words[2],words[3]])
    cell = np.array(cell)
    for iframe in range(len(u.trajectory)):
        ts = u.trajectory[iframe]
        box = cell[iframe]
        ts.dimensions = np.array([box[0], box[1], box[2], 90., 90., 90.])
    return u



plt.rcParams['figure.figsize'] = (5, 4)
oxygen = "name O"
hydrogen = "name O"

# hydrogen = u.select_atoms('name H12')
# molecules = hydrogen.residues.segments
# intra_atoms = []
# for molecule in molecules:
#     intra_atoms.extend(molecule.selet_atoms('name O03'))

u = np.loadtxt('Soper_2000_298_OO')
exp_r = u.T[0]
exp_rdf = u.T[1]
plt.plot(exp_r,exp_rdf,label='Exp.')

u = np.loadtxt('classicalMD_OO.xvg')
exp_r = u.T[0]
exp_rdf = u.T[1]
plt.plot(exp_r,exp_rdf,label="Lan's Paper")

u = get_dmff_uni()
oxy = u.select_atoms(oxygen) # get oxygens
hyd = u.select_atoms(hydrogen) # get hydrogens
O_O = rdf.InterRDF(oxy,hyd,nbins=500,range=(2.0,8.0),verbose=True)
O_O.run(start=100)
plt.plot(O_O.results.bins, O_O.results.rdf,label="New Engine")

ax = plt.gca()

plt.xlim([2.0,8.0])
plt.tick_params(axis='both',direction='in',labelsize=12)
plt.xlabel('Radius (Angstrom)',fontsize=12)
plt.ylabel('Inter-Radial Distribution (H-O)',fontsize=12)
plt.legend(fontsize=12,frameon=False)
plt.savefig('rdf.png', dpi=300, bbox_inches = 'tight')
plt.show()