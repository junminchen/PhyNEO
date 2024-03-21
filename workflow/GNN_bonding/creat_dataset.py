#!/usr/bin/env python
import sys
import numpy as np
import pickle
import glob

ifiles = glob.glob('abinitio_intra/set*')
ifiles2 = glob.glob('abinitio_intra_highT/set*')
ifiles3 = glob.glob('abinitio_intra_pe16/set*')

# training
with open(ifiles[0], 'rb') as f:
    data = pickle.load(f)
for ifn in ifiles[1:9]:
    with open(ifn, 'rb') as f:
        new_data = pickle.load(f)
        assert(np.all(new_data['elements'] == data['elements']))
        data['positions'] = np.vstack((data['positions'], new_data['positions']))
        data['energies'] = np.concatenate((data['energies'], new_data['energies']))
for ifn in ifiles2[1:9]:
    with open(ifn, 'rb') as f:
        new_data = pickle.load(f)
        assert(np.all(new_data['elements'] == data['elements']))
        data['positions'] = np.vstack((data['positions'], new_data['positions']))
        data['energies'] = np.concatenate((data['energies'], new_data['energies']))
with open('dataset_train.pickle', 'wb') as f:
    pickle.dump(data, f)
    
# testing
with open(ifiles[9], 'rb') as f:
    data = pickle.load(f)
for ifn in [ifiles2[9]]:
    with open(ifn, 'rb') as f:
        new_data = pickle.load(f)
        assert(np.all(new_data['elements'] == data['elements']))
        data['positions'] = np.vstack((data['positions'], new_data['positions']))
        data['energies'] = np.concatenate((data['energies'], new_data['energies']))
with open('dataset_test.pickle', 'wb') as f:
    pickle.dump(data, f)

# testing pe16
with open(ifiles3[0], 'rb') as f:
    data = pickle.load(f)
with open('dataset_test_pe16.pickle', 'wb') as f:
    pickle.dump(data, f)