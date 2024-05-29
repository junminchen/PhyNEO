import pickle
import numpy as np
import jax.numpy as jnp
import sys 
import glob 

filelist = glob.glob('*.pickle')
for file in filelist:
    # file = 'params.pickle'
    filename = file.split('.')[0]
    with open(file, 'rb') as ifile:
        data = pickle.load(ifile)

    jnp.save(f'{filename}.npy', data)
    _ = jnp.load(f'{filename}.npy', allow_pickle=True)
    data = _.item()
