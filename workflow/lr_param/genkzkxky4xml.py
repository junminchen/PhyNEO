#!/usr/bin/env python3
import numpy as np
import pickle

# datdict = {}

# with open('atypes.dat', 'r') as file:
#     for line in file:
#         line = line.strip()
#         if line:
#             columns = line.split()
#             key = int(columns[0])
#             values = [int(c) for c in columns[2:]]
#             datdict[key] = values
#print(datdict)

ZThenX = 0
Bisector = 1
ZBisect = 2
ThreeFold = 3
Zonly = 4
NoAxisType = 5
LastAxisTypeIndex = 6

def AtomType(index,datdict):
    for key, values in datdict.items():
        if index in values: 
            return key
    return 'NAN'

def getkxml(datdict,axisfile):
    axis_types = axisfile[:,0]
    axis_indices = axisfile[:,1:]
    z_atoms = axis_indices[:, 0]
    x_atoms = axis_indices[:, 1]
    y_atoms = axis_indices[:, 2]
    checklist = []
    kxml = {}
    for index in range(len(axis_types)):
        atomtype = AtomType(index,datdict)
        if atomtype in checklist:
            continue
        else:
            checklist.append(atomtype)
            axisType = axis_types[index]
            if axisType == ZThenX:
                kz = AtomType(z_atoms[index],datdict)
                kx = AtomType(x_atoms[index],datdict)
                ky = 0
            if axisType == NoAxisType:
                kz = 0
                kx = 0
                ky = 0
            if axisType == Zonly:
                kz = AtomType(z_atoms[index],datdict)
                kx = 0
                ky = 0            
            if axisType == Bisector:
                kz = '-'+AtomType(z_atoms[index],datdict)
                kx = AtomType(x_atoms[index],datdict)
                ky = 0
            if axisType == ZBisect:
                kz = AtomType(z_atoms[index],datdict)
                kx = '-'+AtomType(x_atoms[index],datdict) 
                ky = '-'+AtomType(y_atoms[index],datdict) 
            if axisType == ThreeFold:
                kz = '-'+AtomType(z_atoms[index],datdict)
                kx = '-'+AtomType(x_atoms[index],datdict)
                ky = '-'+AtomType(y_atoms[index],datdict)
            kxml[atomtype]=[kz,kx,ky]
                       
    return kxml


def main():
    # print('#type kz kx ky')
    axisfile = np.loadtxt('axes',dtype=int)
    with open('atype_data.pickle','rb') as i:
        dict = pickle.load(i)['EC']
    datdict = {}
    key_lst = list(dict)
    for i in range(len(key_lst)):
        datdict[str(i+1)] = dict[key_lst[i]]
    fk = open('kzkxky_file', 'w')
    kxml = getkxml(datdict,axisfile)
    for i in list(kxml):
        print(i,kxml[i][0],kxml[i][1],kxml[i][2],file=fk)

    #np.savetxt('textkxml',kxml)

if __name__ == "__main__":
    main()