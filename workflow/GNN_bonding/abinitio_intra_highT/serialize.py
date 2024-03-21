#!/usr/bin/env python
import sys
import numpy as np
import pickle
import glob

target = sys.argv[1]
shift = -826881.1793333929

target = target.strip('/')
coms = glob.glob(target + '/*.com')
coms.sort()
outs = glob.glob(target + '/*.out')
outs.sort()

data = {}
data['elements'] = []
data['positions'] = []
data['energies'] = []

n_data = len(outs)

for i_data, (com, out) in enumerate(zip(coms, outs)):
    # read positions
    with open(com, 'r') as ifile:
        iread = 0
        pos = []
        elements = []
        for line in ifile:
            if 'geometry=' in line:
                iread = 1
                continue
            if iread and '}' in line:
                iread = 0
                break
            if iread:
                words = line.split(',')
                xyz = [float(w) for w in words[3:]]
                elements.append(words[1])
                pos.append(xyz)
    data['positions'].append(pos)
    with open(out, 'r') as ifile:
        for line in ifile:
            if line.startswith(' DF-MP2/USERDEF energy='):
                E = float(line.split('=')[-1]) * 2625.5002 # in kJ/mol
                break
    data['energies'].append(E-shift)

data['positions'] = np.array(data['positions'])
data['energies'] = np.array(data['energies'])
data['elements'] = np.array([elements])

with open(target + '.pickle', 'wb') as ofile:
    pickle.dump(data, ofile)
