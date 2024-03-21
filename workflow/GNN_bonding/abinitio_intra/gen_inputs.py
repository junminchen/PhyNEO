#!/usr/bin/env python
import sys
import numpy as np
import MDAnalysis as mda


def gen_com(atypes, positions, out=None):
    n_atoms = len(positions)
    if out is not None:
        ofile = open(out, 'w')
    else:
        ofile = sys.stdout
    with open(fn_template, 'r') as f:
        iread = 0
        for line in f:
            if 'geometry' in line:
                iread = 1
                print(line, file=ofile, end='')
                for i_atom in range(n_atoms):
                    r = positions[i_atom]
                    print('%d,%s,,%.6f,%.6f,%.6f'%(i_atom+1, atypes[i_atom], r[0], r[1], r[2]), file=ofile)
                continue
            if iread == 1 and '}' in line:
                iread = 0
                print(line, file=ofile, end='')
                continue
            if iread == 0:
                print(line, file=ofile, end='')
                continue

def padding(i):
    s = '%s'%i
    while len(s) < 4:
        s = '0' + s
    return s

if __name__ == '__main__':
    # i_frame = 0 # sys.argv[1]
    u = mda.Universe('pe8.pdb', 'traj.dcd')
    atypes = u.atoms.types
    fn_template = 'mp2_template.com'

    for i_frame in range(9000,10000):
        ts = u.trajectory[i_frame]
        tag = padding(i_frame)
        out = 'set009/' + tag + '.com'
        positions = ts.positions
        gen_com(atypes, positions, out=out)
