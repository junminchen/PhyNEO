import pickle
import numpy as np
import jax.numpy as jnp
import sys 
import glob 

filelist = glob.glob('*.npy')
for file in filelist:
    # file = 'params.pickle'
    filename = file.split('.')[0]
    _ = jnp.load(f'{filename}.npy', allow_pickle=True)
    data = _.item()
    with open(f'_{filename}.pickle', 'wb') as ofile:
        pickle.dump(data, ofile)