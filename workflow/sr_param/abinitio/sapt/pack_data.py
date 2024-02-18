#!/usr/bin/env python
import sys
import numpy as np
import pickle
import argparse
import glob
import re

mH2kj = 2.6255

def sfloat(string):
    s = re.sub('[Dd]', 'e', string)
    return float(s)

def parse_res(out):
    iread = ''
    with open(out, 'r') as ifile:
        res = {}
        pos = []
        elems = []
        dummy_lines = []
        for line in ifile:
            words = line.split()
            if iread == '' and 'geometry={' in line:
                iread = 'pos'
                continue
            if iread == 'pos' and '}' in line:
                iread = ''
                continue
            if iread == '' and 'IMW Results' in line:
                iread= 'res'
                continue
            if iread == 'res' and 'E1tot+E2tot  ' in line:
                res['tot'] = sfloat(words[-1]) # take value in kJ/mol
                continue
            if 'dummy' in line:
                dummy_lines.append(line)
            if iread == 'pos':
                words = line.split(',')
                pos.append([sfloat(w) for w in words[-3:]])
                elems.append(words[1])
            if iread == 'res':
                if 'E1pol ' in line:
                    res['es'] = sfloat(words[-1])
                elif 'E1exch ' in line:
                    res['ex'] = sfloat(words[-1])
                elif 'E2ind ' in line:
                    res['pol'] = sfloat(words[-1])
                elif 'E2ind-exch ' in line:
                    res['pol'] += sfloat(words[-1])
                elif 'E2disp ' in line:
                    res['disp'] = sfloat(words[-1])
                elif 'E2disp-exch ' in line:
                    res['disp'] += sfloat(words[-1])
            if 'SETTING DELTA_HF       =' in line:
                res['dhf'] = sfloat(words[3]) * mH2kj
            if 'SETTING EINT_DFTSAPT   =' in line:
                res['tot'] = sfloat(words[3]) * mH2kj
    dummy_atoms = [int(w) for w in dummy_lines[0].split(',')[1:]]
    dimer_indices = list(range(len(pos)))
    for i in dummy_atoms:
        dimer_indices.remove(i-1)

    monA_indices = list(range(len(pos)))
    dummy_atoms = [int(w) for w in dummy_lines[-2].split(',')[1:]]
    for i in dummy_atoms:
        monA_indices.remove(i-1)
    monB_indices = list(range(len(pos)))
    dummy_atoms = [int(w) for w in dummy_lines[-1].split(',')[1:]]
    for i in dummy_atoms:
        monB_indices.remove(i-1)

    pos = np.array(pos)
    elems = np.array(elems)

    pos_A = pos[monA_indices]
    pos_B = pos[monB_indices]
    elems_A = elems[monA_indices]
    elems_B = elems[monB_indices]

    return elems_A, elems_B, pos_A, pos_B, res

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pack data')
    
    parser.add_argument('--scanid', action='store', dest='scanid', help='Scan IDs to be packed. Default 000', default='xxx')
    parser.add_argument('--comps', action='store', dest='comps', help='Energy components to be packed, default is all (es, ex, pol, disp, dhf, tot)', default='es,ex,pol,disp,dhf,tot')
    parser.add_argument('--out', action='store', dest='out', help='Output file name (without .pickle extension, default data)', default='data')

    args = parser.parse_args()

    if args.scanid == 'xxx':
        scanids = glob.glob('[0-9][0-9][0-9]')
    else:
        scanids = args.scanid.split(',')
    comps = args.comps.split(',')

    data = {}

    for sid in scanids:
        print('Scan ID:', sid)
        outs = glob.glob(sid+'/[0-9][0-9][0-9].out')
        outs.sort()
        res_scan = {}
        pos_A_scan = []
        pos_B_scan = []
        for out in outs:
            print('%20s'%out)
            elem_A, elem_B, pos_A, pos_B, res = parse_res(out)
            for k in res:
                if k not in res_scan:
                    res_scan[k] = [res[k]]
                else:
                    res_scan[k].append(res[k])
            pos_A_scan.append(pos_A)
            pos_B_scan.append(pos_B)
        res_scan['posA'] = np.array(pos_A_scan)
        res_scan['posB'] = np.array(pos_B_scan)
        for k in res_scan.keys():
            res_scan[k] = np.array(res_scan[k])
        data[sid] = res_scan

    print('Dump to %s'%(args.out + '.pickle'))
    with open(args.out + '.pickle', 'wb') as ofile:
        pickle.dump(data, ofile)
