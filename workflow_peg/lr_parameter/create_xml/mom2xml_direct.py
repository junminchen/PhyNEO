#!/usr/bin/env python
import sys
import numpy as np
import xml.etree.ElementTree as ET
import re
import copy
import xml.etree as etree
import os
import xml.dom.minidom as minidom

mom = sys.argv[1]
axes = sys.argv[2]
tmp = 'tmp'

os.system('./convert_mom_to_xml.py %s %s > %s'%(mom, axes, tmp))

forcefield = ET.Element('forcefield')

# atom list and bonding
atoms = []
positions = []
elements = []
with open(mom, 'r') as f:
    for line in f:
        if 'Type' in line:
            words = line.split()
            atoms.append(words[0])
            elements.append(re.search('[A-Za-z]+', words[0]).group(0))
            r = np.array([float(w) for w in words[1:4]])
            positions.append(r)
atoms = np.array(atoms)
positions = np.array(positions)
elements = np.array(elements)
bond_thresh = { ('C', 'C'): 1.75, \
                ('C', 'H'): 1.25, \
                ('H', 'H'): 1.20, \
                ('O', 'C'): 1.75, \
                ('O', 'H'): 1.20, \
        }
bond_thresh_complement = {}
for k in bond_thresh.keys():
    if k[::-1] not in bond_thresh:
        bond_thresh_complement[k[::-1]] = bond_thresh[k]
for k in bond_thresh_complement.keys():
    bond_thresh[k] = bond_thresh_complement[k]

n_atoms = len(atoms)

classes = {'C': 'CT', \
           'H': 'HC', \
           'O': 'OS', \
        }
masses  = {'C': '12.0107', \
           'H': '1.00784', \
           'O': '15.999', \
        }

# atoms, each atom has its own type
# atomtypes = ET.SubElement(forcefield, 'AtomTypes')
# for ia in range(n_atoms):
#     attrib = {}
#     attrib['name'] = '%d'%(ia+1)
#     attrib['class'] = classes[elements[ia]]
#     attrib['element'] = elements[ia]
#     attrib['mass'] = masses[elements[ia]]
#     a = ET.Element('Type', attrib=attrib)
#     atomtypes.append(a)

# residues = ET.SubElement(forcefield, 'Residues')
# residue = ET.Element('Residue', attrib={'name': 'TER'})
# residues.append(residue)
# for ia in range(n_atoms):
#     name = atoms[ia]
#     atype = '%d'%(ia + 1)
#     residue.append(ET.Element('Atom', attrib={'name':name, 'type':atype}))


# add by junmin 2022-02-22
node = ET.SubElement(forcefield, 'ADMPPmeForce', attrib={'lmax':'2','mScale12': '0.00','mScale13': '0.00','mScale14': '0.00','mScale15': '0.00','mScale16': '0.00',\
                                                                    'pScale12': '0.00','pScale13': '0.00','pScale14': '0.00','pScale15': '0.00','pScale16': '0.00',\
                                                                    'dScale12': '1.00','dScale13': '1.00','dScale14': '1.00','dScale15': '1.00','dScale16': '1.00'})
with open(tmp, 'r') as f:
    for line in f:
        if len(line.split()) < 1:
            continue
        node.append(ET.fromstring(line))
# os.system('rm %s'%tmp)

print(minidom.parseString(ET.tostring(forcefield)).toprettyxml(indent="  "))
