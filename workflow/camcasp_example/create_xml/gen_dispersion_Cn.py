#!/usr/bin/env python
import sys
import numpy as np
import re

dic_atypes = {}
atypes = []
with open('atypes.dat', 'r') as ifile:
    for line in ifile:
        words = line.split()
        atype = words[1]
        atypes.append(atype)
        dic_atypes[atype] = [int(i) for i in words[2:]]

dic_params = {}

with open(sys.argv[1], 'r') as ifile:
    iread = 0
    for line in ifile:
        words = line.split()
        if line.startswith('!'):
            continue
        if 'End' in line:
            iread = 0
            continue
        if iread == 0 and re.match('[A-Za-z]+[0-9]+', words[0]) is not None:
            iread = 1
            pair = (words[0], words[1])
            continue
        if iread == 1:
            C6 = float(words[3]) # in Ha/Bohr^6
            C8 = float(words[5])
            C10 = float(words[7])
            dic_params[pair] = np.array([C6, C8, C10])
            continue

# print(dic_params)
n_atypes = len(atypes)

def lookup_param(pair, dic_params):
    if pair in dic_params:
        return dic_params[pair]
    elif pair[::-1] in dic_params:
        return dic_params[pair[::-1]]
    else:
        return None

# check combination rule
# for iat1 in range(n_atypes):
#     at1 = atypes[iat1]
#     for iat2 in range(iat1, n_atypes):
#         at2 = atypes[iat2]
#         Caa = lookup_param((at1, at1), dic_params)
#         Cbb = lookup_param((at2, at2), dic_params)
#         Cab = lookup_param((at1, at2), dic_params)
#         print(np.sqrt(Caa*Cbb)/Cab)

dic_atomic_params = {}

au2kjmol = 2625.5002
au2nm = 0.0529177249
# ionization potential to determin B
atomic_ip = { 'H': 1312. /au2kjmol, # kj/mol to Hartree
              'O': 1314. /au2kjmol,
              'C': 1086. /au2kjmol}

for atype in atypes: # convert from au to openmm unit
    dic_atomic_params[atype] = {}
    dic_atomic_params[atype]["C6"] = dic_params[(atype, atype)][0] * au2kjmol * au2nm**6
    dic_atomic_params[atype]["C8"] = dic_params[(atype, atype)][1] * au2kjmol * au2nm**8
    dic_atomic_params[atype]["C10"] = dic_params[(atype, atype)][2] * au2kjmol * au2nm**10
    dic_atomic_params[atype]["B"] = 2 * np.sqrt(2*atomic_ip[atype[0]]) / au2nm

print("""<ADMPDispPmeForce mScale12="0.00" mScale13="0.00" mScale14="0.00" mScale15="0.00" mScale16="0.00" >""")
for iat, atype in enumerate(atypes):
    C6 = dic_atomic_params[atype]["C6"]
    C8 = dic_atomic_params[atype]["C8"]
    C10 = dic_atomic_params[atype]["C10"]
    B = dic_atomic_params[atype]["B"]
    print('<Atom type="%s" A="fill" B="%.8f" Q="fill" C6="%.6e" C8="%.6e" C10="%.6e"/>'%(atype, B, C6, C8, C10))
print("""</ADMPDispPmeForce>""")

