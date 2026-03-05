import openmm as mm
import openmm.app as app
import openmm.unit as unit
import numpy as np
import pandas as pd
import csv
import struct
from pathlib import Path

class MinimalDCDReader:
    def __init__(self, filename):
        self.file = open(filename, 'rb')
        self._read_header()

    def _read_block(self):
        len_buf = self.file.read(4)
        if not len_buf: return None
        length = struct.unpack('i', len_buf)[0]
        data = self.file.read(length)
        end_len = struct.unpack('i', self.file.read(4))[0]
        if length != end_len:
            raise ValueError(f"Block length mismatch: {length} != {end_len}")
        return data

    def _read_header(self):
        # 1. Header block
        header = self._read_block()
        if header[:4] != b'CORD':
            raise ValueError("Not a CORD DCD file")
        self.n_frames = struct.unpack('i', header[4:8])[0]
        
        # 2. Title block
        self._read_block()
        
        # 3. NATOMS block
        natoms_data = self._read_block()
        self.n_atoms = struct.unpack('i', natoms_data)[0]

    def iterframes(self):
        for _ in range(self.n_frames):
            try:
                # Optional Unit Cell block (48 bytes)
                len_buf = self.file.read(4)
                if not len_buf: break
                length = struct.unpack('i', len_buf)[0]
                if length == 48:
                    # Skip unit cell data
                    self.file.read(48 + 4)
                else:
                    # Not a unit cell block, rewind length bytes
                    self.file.seek(-4, 1)
                
                # Read X, Y, Z coordinates (each is a block)
                x_data = self._read_block()
                x = struct.unpack(f'{self.n_atoms}f', x_data)
                
                y_data = self._read_block()
                y = struct.unpack(f'{self.n_atoms}f', y_data)
                
                z_data = self._read_block()
                z = struct.unpack(f'{self.n_atoms}f', z_data)
                
                yield np.stack([x, y, z], axis=1)
            except Exception as e:
                print(f"Iterator stopped: {e}")
                break

    def close(self):
        self.file.close()

def postprocess_density():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    
    here = Path('.').resolve()
    out_dir = Path(args.output_dir).resolve()
    pdb_path = out_dir / 'stage1_neutral_relaxation/stage1_fixed_lz_start.pdb'
    dcd_path = out_dir / 'stage3_production/production.dcd'
    out_csv = out_dir / 'stage3_production/z_number_density_profile.csv'
    
    if not dcd_path.exists():
        print(f"Skipping: {dcd_path} not found.")
        return
    
    pdb = app.PDBFile(str(pdb_path))
    topology = pdb.topology
    box = topology.getPeriodicBoxVectors()
    lx_nm = box[0][0].value_in_unit(unit.nanometer)
    ly_nm = box[1][1].value_in_unit(unit.nanometer)
    lz_nm = box[2][2].value_in_unit(unit.nanometer)
    area_nm2 = lx_nm * ly_nm
    lz_a = lz_nm * 10.0
    
    res_names = ['DMC', 'ECA', 'LiA', 'PF6']
    # Define mapping for specific atoms
    # We use lists of atom names because PF6 has 6 F atoms
    specific_atoms_map = {
        'ECA_O_carbonyl': ('ECA', ['O00']),
        'DMC_O_carbonyl': ('DMC', ['O03']),
        'PF6_F': ('PF6', ['F01', 'F03', 'F04', 'F05', 'F06', 'F07']),
        'LiA_Li': ('LiA', ['Li01'])
    }
    
    res_data = {}
    for resname in res_names:
        res_list = [r for r in topology.residues() if r.name == resname]
        res_info = []
        for r in res_list:
            indices = [a.index for a in r.atoms()]
            masses = [a.element.mass.value_in_unit(unit.dalton) if a.element else 1.0 for a in r.atoms()]
            res_info.append((indices, masses))
        res_data[resname] = res_info

    # Pre-collect specific atom indices
    specific_atom_indices = {name: [] for name in specific_atoms_map}
    for res in topology.residues():
        rn = res.name.strip()
        for sp_name, (target_res, target_atom_names) in specific_atoms_map.items():
            if rn == target_res:
                for atom in res.atoms():
                    if atom.name in target_atom_names:
                        specific_atom_indices[sp_name].append(atom.index)

    z_density_bins = 180
    res_com_counts = {name: np.zeros(z_density_bins) for name in res_names}
    atom_counts = {name: np.zeros(z_density_bins) for name in specific_atom_indices}
    
    reader = MinimalDCDReader(str(dcd_path))
    n_frames = 0
    print(f"Processing {reader.n_frames} frames from DCD...")
    for pos in reader.iterframes():
        n_frames += 1
        for resname, info_list in res_data.items():
            for indices, masses in info_list:
                mass_sum = sum(masses)
                z = 0.0
                for idx, m in zip(indices, masses):
                    z += pos[idx][2] * m
                z /= mass_sum
                z_wrapped = z % lz_a
                bin_id = int(z_wrapped / lz_a * z_density_bins)
                res_com_counts[resname][min(bin_id, z_density_bins-1)] += 1
        
        for sp_name, indices in specific_atom_indices.items():
            for idx in indices:
                z = pos[idx][2]
                z_wrapped = z % lz_a
                bin_id = int(z_wrapped / lz_a * z_density_bins)
                atom_counts[sp_name][min(bin_id, z_density_bins-1)] += 1
    reader.close()

    all_names = res_names + sorted(specific_atom_indices.keys())
    dz_nm = lz_nm / float(z_density_bins)
    with out_csv.open("w", newline="") as fz:
        writer = csv.writer(fz)
        writer.writerow(["z_center_angstrom", "number_density_total_nm^-3"] + [f"number_density_{n}_nm^-3" for n in all_names])
        total_counts = np.zeros(z_density_bins)
        for n in res_names:
            total_counts += res_com_counts[n]
        for i in range(z_density_bins):
            z_center_a = (i + 0.5) * lz_a / float(z_density_bins)
            total_rho = total_counts[i] / (n_frames * area_nm2 * dz_nm)
            row = [f"{z_center_a:.6f}", f"{total_rho:.10f}"]
            for n in res_names:
                rho = res_com_counts[n][i] / (n_frames * area_nm2 * dz_nm)
                row.append(f"{rho:.10f}")
            for n in sorted(specific_atom_indices.keys()):
                rho = atom_counts[n][i] / (n_frames * area_nm2 * dz_nm)
                row.append(f"{rho:.10f}")
            writer.writerow(row)
    print(f"Updated {out_csv} with {n_frames} frames.")

if __name__ == "__main__":
    postprocess_density()
