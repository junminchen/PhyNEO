#!/usr/bin/env python
import sys
import numpy as np
import re
import MDAnalysis as mda

############################
# the molecule
############################

# list_elems = ['C', 'H', 'H', 'O', 'C', 'H', 'H', 'H', 'C', 'H', 'H', 'O', 'H'] 
# list_atypes = ['C1', 'H1', 'H1','O1', 'C2', 'H2', 'H2', 'H2', 'C3', 'H3', 'H3', 'O2', 'H4'] 
# atypes = ['C1', 'H1', 'O1', 'C2', 'H2', 'C3', 'H3', 'O2', 'H4']

dic_atypes = {}
atypes = []
with open('atypes.dat', 'r') as ifile:
    for line in ifile:
        words = line.split()
        atype = words[1]
        atypes.append(atype)
        dic_atypes[atype] = [int(i) for i in words[2:]]

u = mda.Universe('occocco.pdb')
list_elems = list(u.atoms.elements)

# elements = set(list_elems)
# dic_atypes = {}
# for atype in atypes:
#     dic_atypes[atype] = np.where(np.array(list_atypes)==atype)[0]

list_alabels = []
for ia, a in enumerate(list_elems):
    def pad(i):
        s = '%d'%i
        while len(s) < 2:
            s = '0' + s
        return s
    list_alabels.append(a + pad(ia))


################################
# the electrostatic moments
################################
list_mom_defs = []
list_mom_attr = []
tmp_file = sys.argv[1]
# tmp_file = 'tmp'
with open(tmp_file, 'r') as f:
    pattern_attr = '[a-zA-Z0-9]+="[0-9+-.a-zA-Z]+"'
    for line in f:
        mom_defs = {}
        mom_attr = []
        if 'Atom' not in line:
            continue
        res = re.findall(pattern_attr, line)
        for s in res:
            key, val = s.split('=')
            mom_attr.append(key)
            mom_defs[key] = val
        list_mom_defs.append(mom_defs)
        list_mom_attr.append(mom_attr)

attr_to_dmff = {
'type':'type',
'kz':'kz',
'kx':'kx',
'c0':'c0',
'd1':'dX',
'd2':'dY',
'd3':'dZ',
'q11':'qXX',
'q21':'qXY',
'q22':'qYY',
'q31':'qXZ',
'q32':'qYZ',
'q33':'qZZ',
}

for atype in atypes:
    indices = dic_atypes[atype]
    i0 = indices[0]
    mom_defs_atype = {}
    mom_attr_atype = list_mom_attr[i0]
    for attr in mom_attr_atype:
        res_search = re.search('"([0-9+-.e]+)"', list_mom_defs[i0][attr])
        if res_search is None:
            mom_defs_atype[attr] = list_mom_defs[i0][attr]
        else:
            vals = []
            for i in indices:
                vals.append(float(re.search('"([0-9+-.e]+)"', list_mom_defs[i][attr]).group(1)))
            # print(attr, vals)
            mom_defs_atype[attr] = '"%.8f"'%np.average(vals)
    s = '<Atom'
    for attr in mom_attr_atype:
        if attr == 'type':
            mom_defs_atype[attr] = '"%s"'%atype
        s = s + ' ' + attr_to_dmff[attr] + '=' + mom_defs_atype[attr]
    s += ' />'
    print(s)

# with open('atypes.dat', 'w') as ofile:
#     for iat, atype in enumerate(atypes):
#         print('%3d%5s'%(iat+1, atype), end='', file=ofile)
#         for i in dic_atypes[atype]:
#             print('%4d'%i, end='', file=ofile)
#         print(file=ofile)
