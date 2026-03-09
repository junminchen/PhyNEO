import json
import numpy as np
from local_frame_tool import AmoebaFrameConverter, cartesian_to_tinker_quadrupole

BOHR_TO_NM = 0.05291772109
HARTREE_TO_KJ_MOL = 2625.500

def reprocess_ec():
    with open('results_high.json', 'r') as f:
        data = json.load(f)
    
    coords = []
    with open('EC.xyz', 'r') as f:
        lines = f.readlines()[2:]
        for line in lines:
            if not line.strip(): continue
            parts = line.split()
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    
    coords_bohr = np.array(coords) / 0.529177
    converter = AmoebaFrameConverter(coords_bohr)
    
    frame_rules = {
        0: (0, 1, 2, 3, "Z-Bisect"),
        1: (1, 0, 2, None, "Z-then-X"),
        2: (2, 0, 4, None, "Bisector"),
        3: (3, 0, 5, None, "Bisector"),
        4: (4, 2, 5, None, "Z-then-X"),
        5: (5, 3, 4, None, "Z-then-X"),
        6: (6, 4, 2, None, "Z-then-X"),
        7: (7, 4, 2, None, "Z-then-X"),
        8: (8, 5, 3, None, "Z-then-X"),
        9: (9, 5, 3, None, "Z-then-X"),
    }

    atom_results = []
    for i, atom in enumerate(data['atoms']):
        rule = frame_rules[i]
        center, z, x, l, mode = rule
        d_loc, q_loc = converter.rotate_multipoles(center, z, x, l, mode=mode,
                                                 dipole=atom['dipole'], quadrupole=atom['quadrupole'])
        tinker_q = cartesian_to_tinker_quadrupole(q_loc)
        res = {
            'c0': atom['charge'],
            'dX': d_loc[0] * BOHR_TO_NM, 'dY': d_loc[1] * BOHR_TO_NM, 'dZ': d_loc[2] * BOHR_TO_NM,
            'qXX': tinker_q['qXX'] * (BOHR_TO_NM**2), 'qXY': tinker_q['qXY'] * (BOHR_TO_NM**2),
            'qYY': tinker_q['qYY'] * (BOHR_TO_NM**2), 'qXZ': tinker_q['qXZ'] * (BOHR_TO_NM**2),
            'qYZ': tinker_q['qYZ'] * (BOHR_TO_NM**2), 'qZZ': tinker_q['qZZ'] * (BOHR_TO_NM**2),
            'alpha': atom['alpha_iso'] * (BOHR_TO_NM**3),
            'c6': atom['c6_ii'] * HARTREE_TO_KJ_MOL * (BOHR_TO_NM**6),
            'element': atom['element']
        }
        atom_results.append(res)

    type_map = {0: 11, 1: 12, 2: 13, 3: 13, 4: 14, 5: 14, 6: 15, 7: 15, 8: 15, 9: 15}
    averaged_types = {}
    for i, res in enumerate(atom_results):
        t = type_map[i]
        if t not in averaged_types: averaged_types[t] = []
        averaged_types[t].append(res)
    
    final_types = {}
    for t, instances in averaged_types.items():
        avg = {}
        for k in instances[0].keys():
            if isinstance(instances[0][k], (int, float)):
                avg[k] = np.mean([ins[k] for ins in instances])
            else: avg[k] = instances[0][k]
        rvdw = {'C': 0.17, 'O': 0.15, 'H': 0.12}[avg['element']]
        avg['c8'] = avg['c6'] * (rvdw**2); avg['c10'] = avg['c8'] * (rvdw**2); avg['B'] = 35.0
        final_types[t] = avg

    def generate_ff_xml(filename, force_tag):
        classes = {11:'C_carb', 12:'O_carb', 13:'O_ring', 14:'C_ring', 15:'H'}
        masses = {'C': 12.011, 'O': 15.999, 'H': 1.008}
        xml = '<?xml version="1.0" ?>\n<forcefield>\n  <AtomTypes>\n'
        for t in sorted(final_types.keys()):
            elem = final_types[t]['element']
            xml += f'    <Type name="{t}" class="{classes[t]}" element="{elem}" mass="{masses[elem]}"/>\n'
        xml += '  </AtomTypes>\n  <Residues>\n    <Residue name="EC">\n'
        names = {0:'C1', 1:'O1', 2:'O2', 3:'O3', 4:'C2', 5:'C3', 6:'H1', 7:'H2', 8:'H3', 9:'H4'}
        for i in range(10): xml += f'      <Atom name="{names[i]}" type="{type_map[i]}"/>\n'
        xml += '    </Residue>\n  </Residues>\n'
        
        # Scaling settings
        xml += f'  <{force_tag} lmax="2" mScale12="0.00" mScale13="0.00" mScale14="0.00" mScale15="0.00" mScale16="0.00" pScale12="0.00" pScale13="0.00" pScale14="0.00" pScale15="0.00" pScale16="0.00" dScale12="1.00" dScale13="1.00" dScale14="1.00" dScale15="1.00" dScale16="1.00">\n'
        type_frames = {11: (12, 13), 12: (11, 13), 13: (11, 14), 14: (13, 14), 15: (14, 13)}
        for t in sorted(final_types.keys()):
            d = final_types[t]; kz, kx = type_frames[t]
            xml += f'    <Atom type="{t}" kz="{kz}" kx="{kx}" c0="{d["c0"]:.8f}" dX="{d["dX"]:.8f}" dY="{d["dY"]:.8f}" dZ="{d["dZ"]:.8f}" qXX="{d["qXX"]:.8f}" qXY="{d["qXY"]:.8f}" qYY="{d["qYY"]:.8f}" qXZ="{d["qXZ"]:.8f}" qYZ="{d["qYZ"]:.8f}" qZZ="{d["qZZ"]:.8f}"/>\n'
        for t in sorted(final_types.keys()):
            d = final_types[t]
            xml += f'    <Polarize type="{t}" polarizabilityXX="{d["alpha"]:.4e}" polarizabilityYY="{d["alpha"]:.4e}" polarizabilityZZ="{d["alpha"]:.4e}" thole="0.33"/>\n'
        xml += f'  </{force_tag}>\n'
        
        xml += '  <ADMPDispPmeForce mScale12="1.00" mScale13="1.00" mScale14="1.00" mScale15="1.00" mScale16="1.00">\n'
        for t in sorted(final_types.keys()):
            d = final_types[t]
            xml += f'    <Atom type="{t}" C6="{d["c6"]:.4e}" C8="{d["c8"]:.4e}" C10="{d["c10"]:.4e}"/>\n'
        xml += '  </ADMPDispPmeForce>\n'

        forces = ['SlaterExForce', 'SlaterSrEsForce', 'SlaterSrPolForce', 'SlaterSrDispForce', 'SlaterDhfForce']
        for force in forces:
            xml += f'  <{force} mScale12="1.00" mScale13="1.00" mScale14="1.00" mScale15="1.00" mScale16="1.00">\n'
            for t in sorted(final_types.keys()):
                d = final_types[t]; extra = f' Q="{d["c0"]:.8f}"' if force == 'SlaterSrEsForce' else ''
                xml += f'    <Atom type="{t}" A="1.0" B="{d["B"]:.8f}"{extra}/>\n'
            xml += f'  </{force}>\n'

        xml += '  <QqTtDampingForce mScale12="1.00" mScale13="1.00" mScale14="1.00" mScale15="1.00" mScale16="1.00">\n'
        for t in sorted(final_types.keys()):
            d = final_types[t]
            xml += f'    <Atom type="{t}" B="{d["B"]:.8f}" Q="{d["c0"]:.8f}"/>\n'
        xml += '  </QqTtDampingForce>\n'

        xml += '  <SlaterDampingForce mScale12="1.00" mScale13="1.00" mScale14="1.00" mScale15="1.00" mScale16="1.00">\n'
        for t in sorted(final_types.keys()):
            d = final_types[t]
            xml += f'    <Atom type="{t}" B="{d["B"]:.8f}" C6="{d["c6"]:.4e}" C8="{d["c8"]:.4e}" C10="{d["c10"]:.4e}"/>\n'
        xml += '  </SlaterDampingForce>\n</forcefield>\n'
        
        with open(filename, 'w') as f: f.write(xml)

    generate_ff_xml('ec_dmff_ff.xml', 'ADMPPmeForce')
    generate_ff_xml('ec_mpid_ff.xml', 'MPIDPmeForce')

if __name__ == "__main__":
    reprocess_ec()
