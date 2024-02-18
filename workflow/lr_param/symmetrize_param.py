#!/usr/bin/env python
import sys
import numpy as np
import re
import pickle
import MDAnalysis as mda

############################
# the molecule
############################

# dic_atypes = {}
# atypes = []
# with open('atypes.dat', 'r') as ifile:
#     for line in ifile:
#         words = line.split()
#         atype = words[0]
#         atypes.append(atype)
#         dic_atypes[atype] = [int(i) for i in words[2:]]



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

################################
# the electrostatic moments
################################
def process(dic_atypes,atypes,kzkxky,tmpf):
    list_mom_defs = []
    list_mom_attr = []
    pattern_attr = '[a-zA-Z0-9]+="[0-9+-.a-zA-Z]+"'
    for line in tmpf:
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

    Qf={}
    multipolef=[]
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
            if attr == 'kz':
                mom_defs_atype[attr] = '"%s"'%kzkxky[atype][0]
            if attr == 'kx':
                mom_defs_atype[attr] = '"%s"'%kzkxky[atype][1]
            s = s + ' ' + attr_to_dmff[attr] + '=' + mom_defs_atype[attr]
        Qf[atype]=mom_defs_atype['c0'].strip('"')
        s += ' />'
        multipolef.append(s)
    return Qf,multipolef

def main():
    with open('tmp_new','r') as f:
        tmpf = f.readlines()
    with open('atype_data.pickle','rb') as i:
        dict = pickle.load(i)['EC']
    dic_atypes = {}
    key_lst = list(dict)
    for i in range(len(key_lst)):
        dic_atypes[str(i+1)] = dict[key_lst[i]]
    atypes=list(dic_atypes)
    kzkxky = {}
    with open('kzkxky_file', 'r') as ifile:
        for line in ifile:
            words = line.split()
            atype = words[0]
            kzkxky[atype] = [int(i) for i in words[1:]]
    Qdic, multipolef=process(dic_atypes,atypes,kzkxky,tmpf)
    f=open('2_multipole','w')
    for line in multipolef:
        print(line,file=f)
    Qf=open('Q_file','w')
    for key in list(Qdic):
        print(key,Qdic[key],file=Qf)

if __name__ == "__main__":
    main()