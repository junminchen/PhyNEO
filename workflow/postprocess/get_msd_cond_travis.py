#!/usr/bin/env python
import numpy as np
import sys
import glob
import mdtraj as md
import MDAnalysis as mda
import csv


def read_msd_file(travis_file):
    out = travis_file
    iread = ''
    with open(out, 'r') as ifile:
        for line in ifile:
            words = line.split()
            if 'Saving result' in line:
                words_ = line.split('_')
                if 'Li' in words_:
                    iread = 'D+'
                else: 
                    iread = 'D-'
                    continue
                continue
            if iread == 'D+' and 'Diffusion coefficient' in line:
                D_cation = words[-2]
            if iread == 'D-' and 'Diffusion coefficient' in line:
                D_anion = words[-2]
    # print(D_cation, D_anion)
    return float(D_cation), float(D_anion)

def read_properties(file_path):
    properties = {'Viscosity': {}, 'Conductivity': {}}
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            species = row['Formula']
            properties['Viscosity'][species] = float(row['Viscosity'])
            properties['Conductivity'][species] = float(row['Conductivity'])
    return properties

def load_msd_data(filename):
    data = np.loadtxt(filename, delimiter=';')
    return data[:, 0], data[:, 1]  # 假设第一列是时间，第二列是MSD

def plot_msd(file_path):
    import scienceplots
    import matplotlib.pyplot as plt
    plt.style.use(['science','no-latex', 'nature'])

    fig, ax = plt.subplots(figsize=(5,5*0.618))

    msd_files = glob.glob(f'{file_path}/msd*')
    msd_files.sort()
    for file in msd_files:
        time_na, msd_na = load_msd_data(file)
        plt.plot(time_na, msd_na, label=f"{file.split('/')[-1]}")

    plt.xlabel('Time (ps)')
    plt.ylabel('MSD (nm^2)')
    plt.title('Mean MSD vs. Time')
    plt.grid(True)
    plt.legend()
    # Show the plot
    plt.savefig(f'res_msd/msd_{file_path}.png', dpi=600, bbox_inches = 'tight')
    plt.close(fig)
    # plt.show()

KB = 1.380649e-23
NA = 6.0221408e23
T = 298 # temperature
e_charge = 1.602e-19

paths = glob.glob('*/svr.out')
paths.sort()
exp = read_properties('/data/run01/scw6851/junmin/md/ref_bamboo/exp.csv')
# paths = ['Li_FSI_3.78_EC']
# paths = ['Li_FSI_1.12_DMC-EC_51-49', 'Li_FSI_3.74_DMC-EC_51-49']
# paths = ['Li_FSI_3.74_DMC-EC_51-49']

# paths = ['Li_FSI_1.11_DMC']

for path in paths:
    path = path.split('/')[0]
    # print(path)
    # plot_msd(path)
    cation = path.split('_')[0]
    anion = path.split('_')[1]

    tr = mda.Universe(f'{path}/init.pdb')
    volume = mda.lib.mdamath.box_volume(tr.dimensions) * 1e-30
    na_atoms = tr.select_atoms(f'resname {cation}A')
    n_na = len(na_atoms.fragments)
    cl_atoms = tr.select_atoms(f'resname {anion}')
    n_cl = len(cl_atoms.fragments)

    # print(n_na, n_cl)
    travis_file = f'{path}/travis.log'    
    # travis_file = f'{path}/travis.nve.log'
    D_na, D_cl = read_msd_file(travis_file)
    # D_na, D_cl = 2.2061859374753937e-11, 3.8056530661188375e-11 # Li_FSI_3.78_EC
    # D_na, D_cl = 0.548e-10, 0.550e-10  # Li_FSI_3.74_DMC-EC_51-49

    length = np.power(volume, 1/3)
    FSC = 2.837298 * KB * T / (6 * np.pi * exp['Viscosity'][path] * 1e-3 * length)
    D_na += FSC
    D_cl += FSC

    sigma = (e_charge**2 / (KB * T * volume)) * (n_na * D_na + n_cl * D_cl)
    print(path, D_na, D_cl, sigma)
