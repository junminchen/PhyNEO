#!/usr/bin/env python
import sys
import os
import numpy as np
import re
import MDAnalysis as mda
import xml.etree.ElementTree as ET
from xml.dom import minidom
from rdkit import Chem
from rdkit.Chem import AllChem
import networkx as nx
from itertools import combinations_with_replacement
import pickle
import genkzkxky4xml
import reformat_mom
import symmetrize_param
import convert_mom_to_xml

au2kjmol = 2625.5002
au2nm = 0.0529177249
# ionization potential to determin B
atomic_ip = { 'H': 1312. /au2kjmol, # kj/mol to Hartree
              'O': 1314. /au2kjmol,
              'C': 1086. /au2kjmol,
              'B': 801.  /au2kjmol,
              'F': 1681. /au2kjmol,
              'S': 1000. /au2kjmol,
              'N': 1402. /au2kjmol,
              'P': 1012. /au2kjmol,
              'Li': 520. /au2kjmol,
              'Na': 496. /au2kjmol}

classes = {'C': 'CT', \
           'H': 'HC', \
           'O': 'OS', \
           'S': 'SU', \
           'N': 'NH', \
           'P': 'PD', \
           'F': 'FI', \
           'B': 'BE', \
           'Li': 'LiV',\
           'Na': 'NaN'}

masses  = {'C': '12.0107', \
           'H': '1.00784', \
           'O': '15.999', \
           'B': '9.01218', \
           'F': '18.998403', \
           'N': '14.0067', \
           'S': '32.06', \
           'P': '30.97376', \
           'Li': '6.941',\
           'Na': '22.98977'}

# 函数：将PDB文件中的原子类型替换为整数
def read_atypesdict(picklefile,molname,start):
    with open(picklefile,'rb') as ifile:
        dict = pickle.load(ifile)[molname]
    atype_dict = {}
    key_lst = list(dict)
    typemap = {}
    for i in range(len(key_lst)):
        atype_dict[str(start+i+1)] = dict[key_lst[i]]
        typemap[key_lst[i]]=str(start+i+1)

    return atype_dict, list(atype_dict), typemap

# 函数：将CONECT记录转换为键信息
def convert_conect_to_bond(pdb_file):
    with open(pdb_file, 'r') as file:
        pdb_lines = file.readlines()

    graph = nx.Graph()
    
    for line in pdb_lines:
        if line.startswith('CONECT'):
            atoms = line.split()[1:]
            from_atom = atoms[0]
            to_atoms = atoms[1:]

            for to_atom in to_atoms:
                graph.add_edge(from_atom, to_atom)

    xml_lines = []
    
    for from_atom, to_atom in graph.edges():
        from_atom = str(int(from_atom) - 1)
        to_atom = str(int(to_atom) - 1)
        xml_line = f'<Bond from="{from_atom}" to="{to_atom}" />'
        xml_lines.append(xml_line)

    return xml_lines

def AtomType(index, dic_atypes):
    for key, values in dic_atypes.items():
        if index in values:
            return key
    return 'NAN'
 
# 函数：创建力场（forcefield）
def create_forcefield(dic_atypes, atypes, typemap, axesf, pdb_file, momf, pol_file, disp_file, elements, atoms, resname):
    # dic_atypes0, atypes0 = replace_atype(pdb_file)
    # print(dic_atypes0)
    # print(dic_atypes)
    # atypes = list(dic_atypes)
        
    # 创建 <forcefield> 元素
    forcefield = ET.Element('forcefield')
    
    checklist = []
    # 创建 <AtomTypes> 元素
    atomtypes = ET.SubElement(forcefield, 'AtomTypes')
    for ia in range(len(atoms)):
        attrib = {}
        # attrib['name'] = '%d'%(ia+1)
        attrib['name'] = AtomType(ia, dic_atypes)
        if attrib['name'] in checklist:
            continue
        else:
            checklist.append(attrib['name']) 
            attrib['class'] = classes[elements[ia]]
            attrib['element'] = elements[ia]
            attrib['mass'] = masses[elements[ia]]
            a = ET.Element('Type', attrib=attrib)
            atomtypes.append(a)

    # 创建 <Residues> 元素
    residues = ET.SubElement(forcefield, 'Residues')
    residue = ET.Element('Residue', attrib={'name': resname})
    residues.append(residue)

    for ia in range(len(atoms)):
        name = atoms[ia]
        # atype = '%d'%(ia + 1)
        atype = AtomType(ia, dic_atypes)
        residue.append(ET.Element('Atom', attrib={'name':name, 'type':atype}))

    # 创建 <Atom> 元素和添加键信息
    xml_lines = convert_conect_to_bond(pdb_file)
    for xml_line in xml_lines:
        residue.append(ET.fromstring(xml_line))

    node = ET.SubElement(forcefield, 'ADMPPmeForce', attrib={'lmax':'2','mScale12': '0.00','mScale13': '0.00','mScale14': '0.00','mScale15': '0.00','mScale16': '0.00',\
                                                                        'pScale12': '0.00','pScale13': '0.00','pScale14': '0.00','pScale15': '0.00','pScale16': '0.00',\
                                                                        'dScale12': '1.00','dScale13': '1.00','dScale14': '1.00','dScale15': '1.00','dScale16': '1.00'})
    
    tmpf=convert_mom_to_xml.mainprocess(momf,axesf)
    kzkxky = genkzkxky4xml.getkxml(dic_atypes,axesf)
    Qdic, multipolef=symmetrize_param.process(dic_atypes,atypes,kzkxky,tmpf)

    for line in multipolef:
        if len(line.split()) < 1:
            continue
        node.append(ET.fromstring(line))

    # typically input should be XXX_ref_wt3_L2iso_000.out
    pol_iso = {}
    with open(pol_file, 'r') as ifile:
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
                i_atom = int(atype_orient.split('_')[0][1:])
                for atype in atypes:
                    if i_atom in dic_atypes[atype]:
                        break
                if '1_iso_A' in atype_orient: # dipol pol
                    pol_iso[atype] = val * (0.0529177**3) # convert from au to nm^3


    # 创建 <Polarize> 元素
    for itype, atype in enumerate(atypes):
        node.append(ET.fromstring('<Polarize type="%s" polarizabilityXX="%.4e" polarizabilityYY="%.4e" polarizabilityZZ="%.4e" thole="0.33"/>'%(atype, pol_iso[atype], pol_iso[atype], pol_iso[atype])))
  

    dic_params = {}

    with open(disp_file, 'r') as ifile:
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
                pair = (typemap[words[0]], typemap[words[1]])
                continue
            if iread == 1:
                C6 = float(words[3]) # in Ha/Bohr^6
                C8 = float(words[5])
                C10 = float(words[7])
                dic_params[pair] = np.array([C6, C8, C10])
                continue

    # Qdic = {}
    # with open('Q_file', 'r') as ifile:
    #     for line in ifile:
    #         words = line.split()
    #         Qdic[words[0]] = words[1]

    dic_atomic_params = {}
    for atype in atypes: # convert from au to openmm unit
        dic_atomic_params[atype] = {}
        dic_atomic_params[atype]["C6"] = dic_params[(atype, atype)][0] * au2kjmol * au2nm**6       
        dic_atomic_params[atype]["C8"] = dic_params[(atype, atype)][1] * au2kjmol * au2nm**8       
        dic_atomic_params[atype]["C10"] = dic_params[(atype, atype)][2] * au2kjmol * au2nm**10     
        # dic_atomic_params[atype]["B"] = 2 * np.sqrt(2*atomic_ip[atype[0]]) / au2nm
        elem = elements[dic_atypes[atype][0]]
        dic_atomic_params[atype]["B"] = 2 * np.sqrt(2*atomic_ip[elem]) / au2nm

    nodedisp = ET.SubElement(forcefield, 'ADMPDispPmeForce', attrib={'mScale12':'0.00','mScale13':'0.00', 'mScale14':'0.00', 'mScale15':'0.00', 'mScale16':'0.00' })
    nodees = ET.SubElement(forcefield, 'SlaterExForce', attrib={'mScale12':'0.00','mScale13':'0.00', 'mScale14':'0.00', 'mScale15':'0.00', 'mScale16':'0.00' })
    nodesres = ET.SubElement(forcefield, 'SlaterSrEsForce', attrib={'mScale12':'0.00','mScale13':'0.00', 'mScale14':'0.00', 'mScale15':'0.00', 'mScale16':'0.00' })
    nodesrpol = ET.SubElement(forcefield, 'SlaterSrPolForce', attrib={'mScale12':'0.00','mScale13':'0.00', 'mScale14':'0.00', 'mScale15':'0.00', 'mScale16':'0.00' })
    nodesrdisp = ET.SubElement(forcefield, 'SlaterSrDispForce', attrib={'mScale12':'0.00','mScale13':'0.00', 'mScale14':'0.00', 'mScale15':'0.00', 'mScale16':'0.00' })
    nodedhf = ET.SubElement(forcefield, 'SlaterDhfForce', attrib={'mScale12':'0.00','mScale13':'0.00', 'mScale14':'0.00', 'mScale15':'0.00', 'mScale16':'0.00' })
    nodeqqtt = ET.SubElement(forcefield, 'QqTtDampingForce', attrib={'mScale12':'0.00','mScale13':'0.00', 'mScale14':'0.00', 'mScale15':'0.00', 'mScale16':'0.00' })
    nodesltdmp = ET.SubElement(forcefield, 'SlaterDampingForce', attrib={'mScale12':'0.00','mScale13':'0.00', 'mScale14':'0.00', 'mScale15':'0.00', 'mScale16':'0.00' })


    for iat, atype in enumerate(atypes):
        C6 = dic_atomic_params[atype]["C6"]
        C8 = dic_atomic_params[atype]["C8"]
        C10 = dic_atomic_params[atype]["C10"]
        B = dic_atomic_params[atype]["B"]
        Q = float(Qdic[atype])
        # typename = map_name[atype]
        # Q = float(Qdic[typename])
        
        nodedisp.append(ET.fromstring('<Atom type="%s" C6="%.6e" C8="%.6e" C10="%.6e"/>'%(atype, C6, C8, C10)))
        nodees.append(ET.fromstring('<Atom type="%s" A="1.0" B="%.8f"/>'%(atype, B)))
        nodesres.append(ET.fromstring('<Atom type="%s" A="1.0" B="%.8f" Q="%.8f"/>'%(atype, B, Q)))
        nodesrpol.append(ET.fromstring('<Atom type="%s" A="1.0" B="%.8f"/>'%(atype, B)))
        nodesrdisp.append(ET.fromstring('<Atom type="%s" A="1.0" B="%.8f"/>'%(atype, B)))
        nodedhf.append(ET.fromstring('<Atom type="%s" A="1.0" B="%.8f"/>'%(atype, B)))
        nodeqqtt.append(ET.fromstring('<Atom type="%s" B="%.8f" Q="%.8f"/>'%(atype, B, Q)))
        nodesltdmp.append(ET.fromstring('<Atom type="%s" B="%.8f" C6="%.6e" C8="%.6e" C10="%.6e"/>'%(atype, B, C6, C8, C10)))
    
    print(minidom.parseString(ET.tostring(forcefield)).toprettyxml(indent="  "))

# 主函数
def main():
    molname = 'DMC'
    resname = 'DMC'
    pdb_file = molname + '.pdb'
    dirpath = './'
    camcasp_file = 'conf.'+molname
    mom_file = dirpath + camcasp_file + '/input/OUT/input_ISA-GRID.mom'
    pol_file = dirpath + camcasp_file + '/input/input_ref_wt3_L2iso_000.out'
    disp_file = dirpath + camcasp_file + '/input/input_ref_wt3_L2iso_Cniso.pot'
    start=10
    dic_atypes, atypes, typemap = read_atypesdict('atype_data.pickle', molname, start)
    # mom = 'reform.mom'
    axes = 'axes'
    axesf = np.array(np.loadtxt(axes,dtype=int),ndmin=2)
    # tmp = 'tmp_new'
    # multipole = '2_multipole'

    # os.system('python reformat_mom.py %s > %s'%(mom_file, mom))
    # os.system('python convert_mom_to_xml.py %s %s > %s'%(mom, axes, tmp))
    # os.system('python symmetrize_param.py %s > %s'%(tmp, multipole))
 
    atoms = []
    positions = []
    elements = []

    momf=reformat_mom.subprocess(mom_file)
    for line in momf:
        if 'Type' in line:
            words = line.split()
            atoms.append(words[0])
            elements.append(re.search('[A-Za-z]+', words[0]).group(0))
            r = np.array([float(w) for w in words[1:4]])
            positions.append(r)
    atoms = np.array(atoms)
    positions = np.array(positions)
    elements = np.array(elements)
    
    create_forcefield(dic_atypes, atypes, typemap, axesf, pdb_file, momf, pol_file, disp_file, elements, atoms, resname)

if __name__ == "__main__":
    main()
