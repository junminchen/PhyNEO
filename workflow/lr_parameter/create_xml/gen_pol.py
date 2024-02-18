#!/usr/bin/env python
import sys
import re
import numpy as np

dic_atypes = {}
atypes = []
with open('atypes.dat', 'r') as ifile:
    for line in ifile:
        words = line.split()
        atype = words[1]
        atypes.append(atype)
        dic_atypes[atype] = [int(i) for i in words[2:]]

# typically input should be XXX_ref_wt3_L2iso_000.out
pol_iso = {}
with open(sys.argv[1], 'r') as ifile:
    iread = 0
    for line in ifile:
        words = line.split()
        if "Parameter values" in line:
            iread = 1
            continue
        if iread == 1 and re.match('[0-9]+', words[0]) is None:
            iread = 0
            continue
        elif iread:
            val = float(words[1])
            atype_orient = words[2]
            i_atom = int(atype_orient.split('_')[0][1:]) - 1
            for atype in atypes:
                if i_atom in dic_atypes[atype]:
                    break
            if '1_iso_A' in atype_orient: # dipol pol
                pol_iso[atype] = val * (0.0529177**3) # convert from au to nm^3

for itype, atype in enumerate(atypes):
    print('<Polarize type="%s" polarizabilityXX="%.4e" polarizabilityYY="%.4e" polarizabilityZZ="%.4e" thole="0.33"/>'%(atype, pol_iso[atype], pol_iso[atype], pol_iso[atype]))
# for itype, atype in enumerate(atypes):
#     print('<Polarize type="%d" polarizability="%.4e" thole="0.33" pgrp1="1" />'%(itype+1, pol_iso[atype]))


