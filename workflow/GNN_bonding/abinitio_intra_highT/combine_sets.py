#!/usr/bin/env python
import sys
import numpy as np
import pickle
import glob

ifiles = sys.argv[1:]

with open(ifiles[0], 'rb') as f:
    data = pickle.load(f)

for ifn in ifiles[1:]:
    with open(ifn, 'rb') as f:
        new_data = pickle.load(f)
        assert(np.all(new_data['elements'] == data['elements']))
        data['positions'] = np.vstack((data['positions'], new_data['positions']))
        data['energies'] = np.concatenate((data['energies'], new_data['energies']))

with open('set_combined.pickle', 'wb') as f:
    pickle.dump(data, f)
