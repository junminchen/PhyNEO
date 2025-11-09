import numpy as np
import MDAnalysis as mda
def read_xyz_trajectory(file_path):
    """
    Reads an XYZ trajectory file and extracts positions, elements, and cell parameters.

    Parameters:
        file_path (str): Path to the XYZ file.

    Returns:
        list: A list of dictionaries, each containing 'pos', 'elem', and 'cell'.
    """
    def parse_cell_parameters(cell_line):
        """
        Parse the cell parameters from the comment line.

        Parameters:
            cell_line (str): The comment line containing cell parameters.

        Returns:
            np.ndarray: A 3x3 matrix representing the cell vectors.
        """
        # Remove leading '#' if present
        cell_line = cell_line.lstrip('#').strip()
        parts = cell_line.split()
        if "CELL(abcABC):" in parts:
            a, b, c = map(float, parts[1:4])
            alpha, beta, gamma = map(float, parts[4:7])

            # Calculate the cell matrix
            cell = np.zeros(6)
            cell[0] = a
            cell[1] = b
            cell[2] = c 
            cell[3] = alpha
            cell[4] = beta
            cell[5] = gamma

            return cell
        return None

    trajectory = []

    with open(file_path, 'r') as file:
        while True:
            try:
                # Read the number of atoms
                num_atoms = int(file.readline().strip())

                # Read the comment line (may contain cell parameters)
                comment_line = file.readline().strip()
                cell = parse_cell_parameters(comment_line)

                # Read atomic data
                pos = []
                elem = []
                for _ in range(num_atoms):
                    line = file.readline().strip().split()
                    elem.append(line[0])
                    pos.append([float(x) for x in line[1:4]])

                trajectory.append({
                    'pos': np.array(pos),
                    'elem': elem,
                    'cell': cell
                })
            except ValueError:
                # End of file
                break

    return trajectory

# Example usage
# trajectory = read_xyz_trajectory("path_to_xyz_file.xyz")
# print(trajectory)
trajectory = read_xyz_trajectory("ti.pos_0.xyz")
pdb = 'init.pdb'
u = mda.Universe(pdb)
box = trajectory[-1]['cell']
u.trajectory[0].dimensions = box
u.atoms.positions = trajectory[-1]['pos']

u.atoms.write('nvt_init.pdb')
