#!/usr/bin/env python
import sys
import numpy as np
import MDAnalysis as mda
import copy
import os

def get_dimer(pdb, trj, iframe=-1):
    u = mda.Universe(pdb, trj)
    n_atoms = len(u.atoms)
    monA = u.atoms[:n_atoms//2]
    monB = u.atoms[n_atoms//2:]
    return u, monA, monB

def gen_molpro_output(u, monA, monB, template_fn, ac_data, ofn=None, label=None):
    if ofn is None:
        ofile = sys.stdout
    else:
        ofile = open(ofn, 'w')
    if label is not None:
        print('! %s'%label, file=ofile)
    indices_A = np.array(range(1, len(monA)+1))
    indices_B = np.array(range(len(monA)+1, len(monA)+len(monB)+1))
    n_atoms_tot = len(monA) + len(monB)
    r_midbond = u.atoms.center_of_mass()
    with open(template_fn) as ifile:
        iread = 0
        for line in ifile:
            if iread == 0 and 'geometry=' in line:
                iread = 1
                print(line, end='', file=ofile)
                continue
            if iread == 1 and '}' in line:
                i_atom = 1
                for a in monA + monB:
                    r = a.position
                    print('%d,%s,,%.8f,%.8f,%.8f'%(i_atom, a.type, r[0], r[1], r[2]), file=ofile)
                    i_atom += 1
                print('%d,He,,%.8f,%.8f,%.8f'%(i_atom, r_midbond[0], r_midbond[1], r_midbond[2]), file=ofile)
                print(line, end='', file=ofile)
                iread = 0
                continue
            elif iread == 1:
                continue
            if iread == 0 and '!monomer A' in line:
                print(line, end='', file=ofile)
                iread = 2
                continue
            elif iread == 0 and '!monomer B' in line:
                print(line, end='', file=ofile)
                iread = 3
                continue
            elif iread == 0 and '!dimer' in line:
                print(line, end='', file=ofile)
                iread = 4
                continue
            if iread == 2:
                print('dummy', end='', file=ofile)
                for i in indices_B:
                    print(',%d'%i, end='', file=ofile)
                print(',%d'%(n_atoms_tot+1), file=ofile)
                print('', file=ofile)
                iread = 0
                continue
            elif iread == 3:
                print('dummy', end='', file=ofile)
                for i in indices_A:
                    print(',%d'%i, end='', file=ofile)
                print(',%d'%(n_atoms_tot+1), file=ofile)
                print('', file=ofile)
                iread = 0
                continue
            elif iread == 4:
                print('dummy,%d'%(n_atoms_tot+1), file=ofile)
                iread = 0
                continue
            if 'ip_A=' in line:
                print('ip_A=%.6f'%ac_data[0, 0], file=ofile)
                continue
            elif 'eps_homo_pbe0_A=' in line:
                print('eps_homo_PBE0_A=%.6f'%ac_data[0, 1], file=ofile)
                continue
            elif 'ip_B=' in line:
                print('ip_B=%.6f'%ac_data[1, 0], file=ofile)
                continue
            elif 'eps_homo_pbe0_B=' in line:
                print('eps_homo_PBE0_B=%.6f'%ac_data[1, 1], file=ofile)
                continue
            print(line, end='', file=ofile)
    ofile.close()
    return

# shift along the center of mass direction
def find_closest_distance(u, monA, monB):
    pos1 = monA.positions
    pos2 = monB.positions
    n_atoms1 = len(pos1)
    n_atoms2 = len(pos2)
    min_i = -1
    min_j = -1
    min_dr = 10000
    for i in range(n_atoms1):
        r1 = pos1[i]
        for j in range(n_atoms2):
            r2 = pos2[j]
            if np.linalg.norm(r1-r2) < min_dr:
                min_dr = np.linalg.norm(r1-r2)
                min_i = i
                min_j = n_atoms1 + j
    return min_i, min_j, min_dr

def gen_scan(u, monA, monB):
    dr = 1
    r_min = 1.4
    r_max = 6.2
    # r_min = 1.6
    # r_max = 6.5
    n_atoms1 = len(monA)
    n_atoms2 = len(monB)
    i, j, min_dr = find_closest_distance(u, monA, monB)
    dr_com = monB.center_of_mass() - monA.center_of_mass()
    dn_com = dr_com / np.linalg.norm(dr_com)
    indices = [0]
    positions = [u.atoms.positions]
    pos0 = copy.deepcopy(u.atoms.positions)
    i = 0
    di = 0.1
    while min_dr > r_min:
        i -= di
        pos = copy.deepcopy(pos0)
        pos[n_atoms1:] += dn_com * dr * i
        u.atoms.positions = pos
        _, _, min_dr = find_closest_distance(u, monA, monB)
        if min_dr < r_min:
            break
    i_min = i + di
    i = 0
    while min_dr < r_max:
        i += di
        pos = copy.deepcopy(pos0)
        pos[n_atoms1:] += dn_com * dr * i
        u.atoms.positions = pos
        _, _, min_dr = find_closest_distance(u, monA, monB)
        if min_dr > r_max:
            break
    i_max = i - di
    i_switch1 = i_min + (i_max-i_min)/6
    i_switch2 = i_min + (i_max-i_min)*3/6
    indices = list(np.arange(i_min, i_switch1, (i_switch1-i_min)/4)) \
            + list(np.arange(i_switch1, i_switch2, (i_switch2-i_switch1)/4)) \
            + list(np.arange(i_switch2, i_max, (i_max-i_switch2)/4))
    positions = []
    for i in indices:
        pos = copy.deepcopy(pos0)
        pos[n_atoms1:] += dn_com * dr * i
        positions.append(pos)
    # while min_dr > r_min:
    #     pos = copy.deepcopy(pos0)
    #     pos[n_atoms1:] += dn_com * dr * i
    #     u.atoms.positions = pos
    #     _, _, min_dr = find_closest_distance(u, monA, monB)
    #     if min_dr > r_min:
    #         indices = [i] + indices
    #         positions = [pos] + positions
    #     i -= 1
    # i = 1
    # while min_dr < r_max:
    #     pos = copy.deepcopy(pos0)
    #     pos[n_atoms1:] += dn_com * dr * i
    #     u.atoms.positions = pos
    #     _, _, min_dr = find_closest_distance(u, monA, monB)
    #     if min_dr < r_max:
    #         indices.append(i)
    #         positions.append(pos)
    #     i += 1
    return indices, positions
    
def gen_gjf(u, ofn=None):
    if ofn is None:
        ofile = sys.stdout
    else:
        ofile = open(ofn, 'w')
    print("# HF/6-31G(d)\n\ntitle\n\n0 1", file=ofile)
    for atom in u.atoms:
        elem = atom.type
        r = atom.position
        print('%3s%15.8f%15.8f%15.8f'%(elem, r[0], r[1], r[2]), file=ofile)
    print('', file=ofile)
    return

def padding(i):
    s = '%d'%i
    while len(s) < 3:
        s = '0' + s
    return s

def clean_folder(i_frame, maindir):
    folder = maindir + '/' + padding(i_frame)
    if os.path.isdir(folder):
        os.system('rm -r %s'%folder)
    os.system('mkdir %s'%folder)
    return folder

if __name__ == '__main__':
    # pick a frame and scan
    # i_frame = np.random.randint(1000) #int(sys.argv[1])
    i_frame = int(sys.argv[1])
    u, monA, monB = get_dimer('pe6_dimer.pdb', 'output.pdb')
    if i_frame == -1: # -1 means scan the dimer geometry provided in command line argument
        i_frame = 0
        u0 = mda.Universe(sys.argv[2])
        pos_mon = u0.atoms.positions
        shift = np.array([5.0, 0, 0])
        u.atoms.positions = np.vstack((pos_mon, pos_mon + shift))
    else:   # otherwise pick a frame in the trajectory output.pdb
        ts = u.trajectory[i_frame]
    u.dimensions = np.array([30, 30, 30, 90, 90, 90])
    # IP, and HOMO energy for each monomer, at extended minimum geometries
    # ac_data = np.array([[0.220086, -0.139377], [0.220086, -0.139377]])
    ac_data = np.array([[0.396796, -0.318417], [0.396796, -0.318417]])
    indices, positions = gen_scan(u, monA, monB)
    n_data = len(indices)
    folder_gjf = clean_folder(i_frame, 'gjfs')
    folder_sapt = clean_folder(i_frame, 'sapt')
    folder_mp2 = clean_folder(i_frame, 'mp2')
    folder_pdb = clean_folder(i_frame, 'pdb')
    for i_data in range(n_data):
        pos = positions[i_data]
        u.atoms.positions = pos

        # generate gjf files for visualization
        ofn = folder_gjf + '/' + padding(i_data) + '.gjf'
        gen_gjf(u, ofn)

        # write pdb
        ofn = folder_pdb + '/' + padding(i_data) + '.pdb'
        u.atoms.write(ofn)

        # generate sapt file
        ofn = folder_sapt + '/' + padding(i_data) + '.com'
        gen_molpro_output(u, monA, monB, 'sapt_template.com', ac_data, ofn=ofn, label='shift= %.6f'%indices[i_data])

        # # generate mp2 file
        ofn = folder_mp2 + '/' + padding(i_data) + '.com'
        gen_molpro_output(u, monA, monB, 'mp2_template.com', ac_data, ofn=ofn, label='shift = %.6f'%indices[i_data])
