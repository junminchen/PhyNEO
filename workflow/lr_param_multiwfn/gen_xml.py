
import json
import numpy as np
from pathlib import Path

# Constants for unit conversion
BOHR_TO_NM = 0.05291772109
HARTREE_TO_KJ_MOL = 2625.500

def rotate_to_local_frame(coords, atom_idx, z_idx, x_idx, multipoles):
    """
    Rotate multipoles from global Cartesian to local frame.
    Following the AMOEBA/OpenMM convention.
    """
    ri = coords[atom_idx]
    rj = coords[z_idx]
    rk = coords[x_idx]
    
    # Z-axis: from i to j
    z = rj - ri
    z /= np.linalg.norm(z)
    
    # X-axis: projection of k-i onto plane perpendicular to z
    x = rk - ri
    x = x - np.dot(x, z) * z
    x /= np.linalg.norm(x)
    
    # Y-axis
    y = np.cross(z, x)
    
    # Rotation matrix (columns are local axes in global frame)
    R = np.vstack([x, y, z]).T
    # We want global to local, so we use R.T
    R_inv = R.T
    
    # Dipole rotation
    d_global = np.array(multipoles['dipole'])
    d_local = R_inv @ d_global
    
    # Quadrupole rotation (Q_local = R_inv @ Q_global @ R)
    q_global = np.array(multipoles['quadrupole'])
    q_local = R_inv @ q_global @ R
    
    return d_local, q_local

def gen_xml():
    with open('results_high.json', 'r') as f:
        data = json.load(f)
    
    # Read coordinates from XYZ (simplified)
    coords = []
    with open('EC.xyz', 'r') as f:
        lines = f.readlines()[2:]
        for line in lines:
            if not line.strip(): continue
            parts = line.split()
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    coords = np.array(coords) / 0.529177 # convert Angstrom to Bohr for consistency with MBIS

    # Define Types for EC (C3H4O3)
    # 0: Carbonyl Carbon (C) -> Type 11
    # 1: Carbonyl Oxygen (O) -> Type 12
    # 2, 3: Ring Oxygen (O) -> Type 13
    # 4, 5: Ring Carbon (C) -> Type 14
    # 6-9: Hydrogen (H) -> Type 15
    
    atom_to_type = {
        0: 11, 1: 12, 2: 13, 3: 13, 4: 14, 5: 14, 
        6: 15, 7: 15, 8: 15, 9: 15
    }
    
    # Local frames (kz, kx) - using indices
    # Carbonyl C: kz=O_carb, kx=O_ring
    # Carbonyl O: kz=C_carb, kx=O_ring
    # Ring O: kz=C_carb, kx=C_ring
    # Ring C: kz=O_ring, kx=C_ring_other
    # Hydrogen: kz=C_parent, kx=O_ring
    
    frames = {
        0: (1, 2),  # C0: kz=1, kx=2
        1: (0, 2),  # O1: kz=0, kx=2
        2: (0, 4),  # O2: kz=0, kx=4
        3: (0, 5),  # O3: kz=0, kx=5
        4: (2, 5),  # C4: kz=2, kx=5
        5: (3, 4),  # C5: kz=3, kx=4
        6: (4, 2),  # H6: kz=4, kx=2
        7: (4, 2),  # H7: kz=4, kx=2
        8: (5, 3),  # H8: kz=5, kx=3
        9: (5, 3),  # H9: kz=5, kx=3
    }

    type_data = {}
    
    for i, atom in enumerate(data['atoms']):
        t = atom_to_type[i]
        kz, kx = frames[i]
        
        d_local, q_local = rotate_to_local_frame(coords, i, kz, kx, atom)
        
        if t not in type_data:
            type_data[t] = []
        
        # Convert units to nm and e
        # Multipoles are in Bohr and e
        atom_res = {
            'c0': atom['charge'],
            'dX': d_local[0] * BOHR_TO_NM,
            'dY': d_local[1] * BOHR_TO_NM,
            'dZ': d_local[2] * BOHR_TO_NM,
            'qXX': q_local[0, 0] * (BOHR_TO_NM**2),
            'qXY': q_local[0, 1] * (BOHR_TO_NM**2),
            'qYY': q_local[1, 1] * (BOHR_TO_NM**2),
            'qXZ': q_local[0, 2] * (BOHR_TO_NM**2),
            'qYZ': q_local[1, 2] * (BOHR_TO_NM**2),
            'qZZ': q_local[2, 2] * (BOHR_TO_NM**2),
            'alpha': atom['alpha_iso'] * (BOHR_TO_NM**3),
            'c6': atom['c6_ii'] * HARTREE_TO_KJ_MOL * (BOHR_TO_NM**6),
            'element': atom['element']
        }
        type_data[t].append(atom_res)

    # Average by type
    final_types = {}
    for t, instances in type_data.items():
        avg = {}
        for k in instances[0].keys():
            if isinstance(instances[0][k], str):
                avg[k] = instances[0][k]
            else:
                avg[k] = np.mean([ins[k] for ins in instances])
        
        # Estimate C8 and C10 (simple scaling)
        # C8 ~ C6 * (RvdW)^2, C10 ~ C8 * (RvdW)^2
        # Use approximate RvdW in nm: C=0.17, O=0.15, H=0.12
        rvdw = {'C': 0.17, 'O': 0.15, 'H': 0.12}[avg['element']]
        avg['c8'] = avg['c6'] * (rvdw**2)
        avg['c10'] = avg['c8'] * (rvdw**2)
        
        # Slater B parameter (approx 35 nm^-1 based on example)
        avg['B'] = 35.0
        
        final_types[t] = avg

    # Map back to type-based frames (types of neighbors)
    type_frames = {
        11: (12, 13),
        12: (11, 13),
        13: (11, 14),
        14: (13, 14),
        15: (14, 13)
    }

    # Generate XML
    xml = '<?xml version="1.0" ?>\n<forcefield>\n'
    
    # AtomTypes
    xml += '  <AtomTypes>\n'
    masses = {'C': 12.011, 'O': 15.999, 'H': 1.008}
    classes = {11: 'C_carb', 12: 'O_carb', 13: 'O_ring', 14: 'C_ring', 15: 'H'}
    for t in sorted(final_types.keys()):
        elem = final_types[t]['element']
        xml += f'    <Type name="{t}" class="{classes[t]}" element="{elem}" mass="{masses[elem]}"/>\n'
    xml += '  </AtomTypes>\n'
    
    # Residues (simplified EC)
    xml += '  <Residues>\n    <Residue name="EC">\n'
    atom_names = {0: 'C1', 1: 'O1', 2: 'O2', 3: 'O3', 4: 'C2', 5: 'C3', 6: 'H1', 7: 'H2', 8: 'H3', 9: 'H4'}
    for i in range(10):
        xml += f'      <Atom name="{atom_names[i]}" type="{atom_to_type[i]}"/>\n'
    # Bonds could be added here if needed
    xml += '    </Residue>\n  </Residues>\n'

    # ADMPPmeForce
    xml += '  <ADMPPmeForce lmax="2" mScale12="0.00" mScale13="0.00" mScale14="0.00" mScale15="0.00" mScale16="0.00" pScale12="0.00" pScale13="0.00" pScale14="0.00" pScale15="0.00" pScale16="0.00" dScale12="1.00" dScale13="1.00" dScale14="1.00" dScale15="1.00" dScale16="1.00">\n'
    for t in sorted(final_types.keys()):
        d = final_types[t]
        kz, kx = type_frames[t]
        xml += f'    <Atom type="{t}" kz="{kz}" kx="{kx}" c0="{d["c0"]:.8f}" dX="{d["dX"]:.8f}" dY="{d["dY"]:.8f}" dZ="{d["dZ"]:.8f}" qXX="{d["qXX"]:.8f}" qXY="{d["qXY"]:.8f}" qYY="{d["qYY"]:.8f}" qXZ="{d["qXZ"]:.8f}" qYZ="{d["qYZ"]:.8f}" qZZ="{d["qZZ"]:.8f}"/>\n'
    for t in sorted(final_types.keys()):
        d = final_types[t]
        xml += f'    <Polarize type="{t}" polarizabilityXX="{d["alpha"]:.4e}" polarizabilityYY="{d["alpha"]:.4e}" polarizabilityZZ="{d["alpha"]:.4e}" thole="0.33"/>\n'
    xml += '  </ADMPPmeForce>\n'

    # ADMPDispPmeForce
    xml += '  <ADMPDispPmeForce mScale12="0.00" mScale13="0.00" mScale14="0.00" mScale15="0.00" mScale16="0.00">\n'
    for t in sorted(final_types.keys()):
        d = final_types[t]
        xml += f'    <Atom type="{t}" C6="{d["c6"]:.4e}" C8="{d["c8"]:.4e}" C10="{d["c10"]:.4e}"/>\n'
    xml += '  </ADMPDispPmeForce>\n'

    # Slater Forces (using B=35 as estimate)
    for force in ['SlaterExForce', 'SlaterSrEsForce', 'SlaterSrPolForce', 'SlaterSrDispForce', 'SlaterDhfForce']:
        xml += f'  <{force} mScale12="0.00" mScale13="0.00" mScale14="0.00" mScale15="0.00" mScale16="0.00">\n'
        for t in sorted(final_types.keys()):
            d = final_types[t]
            extra = f' Q="{d["c0"]:.8f}"' if force == 'SlaterSrEsForce' else ''
            xml += f'    <Atom type="{t}" A="1.0" B="{d["B"]:.8f}"{extra}/>\n'
        xml += f'  </{force}>\n'

    # Damping Forces
    xml += '  <QqTtDampingForce mScale12="0.00" mScale13="0.00" mScale14="0.00" mScale15="0.00" mScale16="0.00">\n'
    for t in sorted(final_types.keys()):
        d = final_types[t]
        xml += f'    <Atom type="{t}" B="{d["B"]:.8f}" Q="{d["c0"]:.8f}"/>\n'
    xml += '  </QqTtDampingForce>\n'

    xml += '  <SlaterDampingForce mScale12="0.00" mScale13="0.00" mScale14="0.00" mScale15="0.00" mScale16="0.00">\n'
    for t in sorted(final_types.keys()):
        d = final_types[t]
        xml += f'    <Atom type="{t}" B="{d["B"]:.8f}" C6="{d["c6"]:.4e}" C8="{d["c8"]:.4e}" C10="{d["c10"]:.4e}"/>\n'
    xml += '  </SlaterDampingForce>\n'

    xml += '</forcefield>\n'
    
    with open('ec_forcefield.xml', 'w') as f:
        f.write(xml)
    print("Force field generated: ec_forcefield.xml")

if __name__ == "__main__":
    gen_xml()
