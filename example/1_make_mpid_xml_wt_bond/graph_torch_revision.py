import sys
from functools import partial
from itertools import permutations, product
from typing import Dict, List

import torch
import torch.nn.functional as F
try:
    import mdtraj as md
except ImportError:
    pass
import numpy as np

'''
This module works on building graphs based on molecular topology
'''

ATYPE_INDEX = {
    'H': 0, 'He': 1,
    'Li': 2, 'Be': 3,
    'B': 4, 'C': 5, 'N': 6, 'O': 7, 'F': 8, 'Ne': 9,
    'Na': 10, 'Mg': 11,
    'Al': 12, 'Si': 13, 'P': 14, 'S': 15, 'Cl': 16, 'Ar': 17,
    'K': 18, 'Ca': 19
}
N_ATYPES = len(ATYPE_INDEX.keys())

# used to compute equilibrium bond lengths
COVALENT_RADIUS = {
    'H': 0.31, 'He': 0.28,
    'Li': 1.28, 'Be': 0.96,
    'B': 0.84, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57, 'Ne': 0.58,
    'Na': 1.66, 'Mg': 1.41,
    'Al': 1.21, 'Si': 1.11, 'P': 1.10, 'S': 1.05, 'Cl': 1.02, 'Ar': 1.06,
    'K': 2.03, 'Ca': 1.76 
}

# scaling parameters for feature calculations
FSCALE_BOND = 10.0
FSCALE_ANGLE = 5.0

MAX_VALENCE = 4
MAX_ANGLES_PER_SITE = MAX_VALENCE * (MAX_VALENCE - 1) // 2
MAX_DIHEDS_PER_BOND = (MAX_VALENCE - 1)**2

# dimension of bond features
DIM_BOND_FEATURES_GEOM = {
    'bonds': 2 * MAX_VALENCE - 1,
    'angles0': MAX_VALENCE * (MAX_VALENCE - 1) // 2,
    'angles1': MAX_VALENCE * (MAX_VALENCE - 1) // 2,
    'diheds': (MAX_VALENCE - 1)**2
}
DIM_BOND_FEATURES_GEOM_TOT = np.sum(
    [DIM_BOND_FEATURES_GEOM[k] for k in DIM_BOND_FEATURES_GEOM.keys()])
DIM_BOND_FEATURES_ATYPES = MAX_VALENCE * 2 * N_ATYPES


def pbc_shift_torch(dr, box, box_inv):
    if box is None:
        return dr
    
    is_batched = dr.dim() == 3
    if is_batched:
        # print(dr.shape, box_inv.shape, box.shape)
        ds = torch.einsum('bni,ij->bnj', dr, box_inv)
        ds = ds - torch.floor(ds + 0.5)
        dr_shifted = torch.einsum('bni,ij->bnj', ds, box)
    else:
        ds = torch.matmul(dr, box_inv.T)
        ds = ds - torch.floor(ds + 0.5)
        dr_shifted = torch.matmul(ds, box)
    return dr_shifted


def distribute_scalar_torch(scalar_values, indices):
    """Gather scalar values by index, mirroring distribute_scalar while supporting batching."""
    squeeze_batch = False
    if scalar_values.dim() == 1:
        scalar_values = scalar_values.unsqueeze(0)
        squeeze_batch = True
    elif scalar_values.dim() != 2:
        raise ValueError('scalar_values must be 1D or 2D tensor.')

    batch_size, n_values = scalar_values.shape
    indices_clamped = torch.clamp(indices, min=0, max=n_values - 1)
    valid_mask = indices >= 0

    expand_shape = (batch_size, ) + tuple(indices.shape[:-1]) + (n_values, )
    values_expanded = scalar_values.view(batch_size, *([1] * (indices.dim() - 1)), n_values)
    values_expanded = values_expanded.expand(expand_shape)

    gather_index = indices_clamped.unsqueeze(0).expand(batch_size, *indices.shape)
    gathered = torch.gather(values_expanded, dim=-1, index=gather_index)
    gathered = gathered.to(torch.float32)
    gathered = gathered * valid_mask.unsqueeze(0).to(gathered.dtype)

    if squeeze_batch:
        gathered = gathered.squeeze(0)
    return gathered


def distribute_v3_torch(vectors, indices):
    """Gather vector values while supporting batching and padded indices."""
    if not torch.is_tensor(indices):
        indices = torch.as_tensor(indices, dtype=torch.long, device=vectors.device)
    else:
        indices = indices.to(device=vectors.device, dtype=torch.long)

    squeeze_batch = False
    if vectors.dim() == 2:
        vectors = vectors.unsqueeze(0)
        squeeze_batch = True
    elif vectors.dim() != 3:
        raise ValueError('vectors must be 2D or 3D tensor.')

    batch_size, n_vectors, vec_dim = vectors.shape
    if n_vectors == 0:
        empty = torch.zeros((batch_size, *indices.shape, vec_dim), dtype=vectors.dtype, device=vectors.device)
        return empty.squeeze(0) if squeeze_batch else empty

    indices_shape = indices.shape
    if indices.numel() == 0:
        empty = torch.zeros((batch_size, *indices_shape, vec_dim), dtype=vectors.dtype, device=vectors.device)
        return empty.squeeze(0) if squeeze_batch else empty

    indices_flat = indices.reshape(-1)
    invalid_mask = indices_flat < 0
    indices_clamped = indices_flat.clone()
    indices_clamped = torch.clamp(indices_clamped, min=0, max=n_vectors - 1)

    gathered = vectors[:, indices_clamped]
    if invalid_mask.any():
        gathered[:, invalid_mask, :] = 0.0

    gathered = gathered.view(batch_size, *indices_shape, vec_dim)
    if squeeze_batch:
        gathered = gathered.squeeze(0)
    return gathered


class TopGraph:

    def __init__(self, list_atom_elems, bonds, positions=None, box=None, device='cpu'):
        self.device = torch.device(device)
        self.list_atom_elems = list_atom_elems
        self.bonds = torch.tensor(bonds, dtype=torch.long, device=self.device)
        self.n_atoms = len(list_atom_elems)
        if positions is not None:
            if isinstance(positions, torch.Tensor):
                self.positions = positions.clone().detach().to(dtype=torch.float32, device=self.device)
            else:
                self.positions = torch.tensor(positions, dtype=torch.float32, device=self.device)
        else:
            self.positions = None
        # self._build_connectivity()
        # self._get_valences()
        self._build_connectivity_torch()
        self._get_valences_torch()
        self.set_internal_coords_indices()
        if box is not None:
            if isinstance(box, torch.Tensor):
                self.box = box.clone().detach().to(dtype=torch.float32, device=self.device)
            else:
                self.box = torch.tensor(box, dtype=torch.float32, device=self.device)
            self.box_inv = torch.linalg.inv(self.box)
        else:
            self.box = None
            self.box_inv = None
        return


    def _build_connectivity_torch(self):
        """Build connectivity map as a PyTorch tensor."""
        self.connectivity = torch.zeros((self.n_atoms, self.n_atoms), dtype=torch.int, device=self.device)
        if self.bonds.numel() > 0:
            row, col = self.bonds.T
            self.connectivity[row, col] = 1
            self.connectivity[col, row] = 1

    def _build_connectivity(self):
        """Alias to match the JAX API naming."""
        self._build_connectivity_torch()

    def _get_valences_torch(self):
        """Get valences from the connectivity tensor."""
        self.valences = torch.sum(self.connectivity, dim=1)

    def typify_atom(self, i, depth=0, excl=None):
        """Recursive atom typification mirroring the JAX implementation."""
        if depth == 0:
            return self.list_atom_elems[i]

        neighbors = torch.nonzero(self.connectivity[i] == 1, as_tuple=False).flatten().tolist()
        atype_neighbors = []
        for j in neighbors:
            if excl is not None and j == excl:
                continue
            atype_neighbors.append(self.typify_atom(int(j), depth=depth - 1, excl=i))
        atype_neighbors.sort()
        if not atype_neighbors:
            return self.list_atom_elems[i]
        return f"{self.list_atom_elems[i]}-({','.join(atype_neighbors)})"

    def typify_all_atoms(self, depth=0):
        """Typify every atom in the graph."""
        self.atom_types = np.array([self.typify_atom(i, depth=depth) for i in range(self.n_atoms)], dtype="object")
        return self.atom_types

    def typify_subgraph(self, i):
        """Typify atoms within a specific subgraph."""
        if not hasattr(self, 'subgraphs'):
            raise AttributeError('No subgraphs have been generated yet.')
        self.subgraphs[i].typify_all_atoms(depth=(2 * self.nn + 4))
        return

    def typify_all_subgraphs(self):
        """Typify atoms for all existing subgraphs."""
        if not hasattr(self, 'subgraphs'):
            raise AttributeError('No subgraphs have been generated yet.')
        for i in range(self.n_subgraphs):
            self.typify_subgraph(i)
        return

    def get_all_subgraphs(self,
                          nn,
                          type_center='bond',
                          typify=True,
                          id_chiral=True):
        """Construct subgraphs following the JAX implementation."""
        if type_center not in {'bond', 'atom'}:
            raise ValueError("type_center must be 'bond' or 'atom'.")

        if type_center == 'atom':
            centers = range(self.n_atoms)
        else:
            centers = range(len(self.bonds))

        self.subgraphs = [TopSubGraph(self, idx, nn, type_center) for idx in centers]
        self.nn = nn
        self.n_subgraphs = len(self.subgraphs)

        # if typify:
        #     for g in self.subgraphs:
        #         g.typify_all_atoms(depth=(2 * nn + 4))
        #     if id_chiral:
        #         for g in self.subgraphs:
        #             g._add_chirality_labels()
        #             g.get_canonical_orders_wt_permutation_grps()

        # if getattr(self, 'positions', None) is not None:
        #     self._update_subgraph_positions()

        return 

    def set_box(self, box):
        '''
        Set the box information in the class

        Parameters
        ----------
        box: array
            3 * 3: the box array, pbc vectors arranged in rows
        '''
        if isinstance(box, torch.Tensor):
            self.box = box.clone().detach().to(dtype=torch.float32, device=self.device)
        else:
            self.box = torch.tensor(box, dtype=torch.float32, device=self.device)
        self.box_inv = torch.linalg.inv(self.box)
        if hasattr(self, 'subgraphs'):
            self._propagate_attr('box')
            self._propagate_attr('box_inv')
        return

    def set_positions(self, positions, update_subgraph=True):
        if isinstance(positions, torch.Tensor):
            self.positions = positions.clone().detach().to(dtype=torch.float32, device=self.device)
        else:
            self.positions = torch.tensor(positions, dtype=torch.float32, device=self.device)
        if update_subgraph:
            self._update_subgraph_positions()
        return

    def _propagate_attr(self, attr):
        for ig in range(self.n_subgraphs):
            setattr(self.subgraphs[ig], attr, getattr(self, attr))
        return
    
    def _add_chirality_labels(self, verbose=False):
        """Add chirality labels to distinguish equivalent hydrogens."""
        atom_types_copy = self.atom_types.copy()

        for i in range(self.n_atoms):
            if self.positions is None:
                continue

            neighbors_t = torch.nonzero(self.connectivity[i] == 1, as_tuple=False).flatten()
            if neighbors_t.numel() != 4:
                continue
            neighbors = neighbors_t.cpu().numpy()

            labels = atom_types_copy[neighbors]
            flags = np.array([np.sum(labels == label) for label in labels])

            if np.sum(flags) != 6:
                continue

            identical_mask = flags == 2
            identical_indices = np.where(identical_mask)[0]
            different_indices = np.where(~identical_mask)[0]

            if len(identical_indices) != 2 or len(different_indices) != 2:
                continue

            j = int(neighbors[identical_indices[0]])
            k = int(neighbors[identical_indices[1]])
            l = int(neighbors[different_indices[0]])
            m = int(neighbors[different_indices[1]])

            if atom_types_copy[j].endswith(('R', 'L')) or atom_types_copy[k].endswith(('R', 'L')):
                continue

            tl, tm = atom_types_copy[l], atom_types_copy[m]
            if tl > tm:
                l, m = m, l

            idx_tensor = torch.tensor([i, j, l, m], dtype=torch.long, device=self.device)
            ri, rj, rl, rm = self.positions[idx_tensor]

            v_ij = pbc_shift_torch(rj - ri, self.box, self.box_inv)
            v_il = pbc_shift_torch(rl - ri, self.box, self.box_inv)
            v_im = pbc_shift_torch(rm - ri, self.box, self.box_inv)

            signed_volume = torch.dot(v_ij, torch.linalg.cross(v_il, v_im))

            if signed_volume > 0:
                self.atom_types[j] = self.atom_types[j] + 'R'
                self.atom_types[k] = self.atom_types[k] + 'L'
            else:
                self.atom_types[j] = self.atom_types[j] + 'L'
                self.atom_types[k] = self.atom_types[k] + 'R'
        return

    def _process_unique_subgraphs(self, id_chiral=True):
        """Cache computations for unique subgraph types, including chirality."""
        unique_subgraph_cache = {}
        subgraph_type_keys = []

        # 1. First pass: Typify all subgraphs and add chirality labels 
        for idx, g in enumerate(self.subgraphs):
            g.typify_all_atoms(depth=(2 * self.nn + 4))
            if id_chiral:
                g._add_chirality_labels()
            # key = tuple(sorted(g.atom_types))
            # key = tuple(g.atom_types)
            key = idx
            subgraph_type_keys.append(key)

        # 2. Second pass: Process each unique key once
        for key in set(subgraph_type_keys):
            if key not in unique_subgraph_cache:
                rep_idx = subgraph_type_keys.index(key)
                g = self.subgraphs[rep_idx]
                g.get_canonical_orders_wt_permutation_grps()
                g.prepare_graph_feature_calc()
                
                # Store results in cache
                cached_feature_atypes = {
                    kb: tensor.clone().detach().to(self.device)
                    for kb, tensor in g.feature_atypes.items()
                }
                cached_feature_indices = {
                    kb: {kf: idx_tensor.clone().detach().to(self.device)
                         for kf, idx_tensor in tensors.items()}
                    for kb, tensors in g.feature_indices.items()
                }
                nb_connect = getattr(g, 'nb_connect', None)
                cached_nb_connect = None
                if nb_connect is not None:
                    cached_nb_connect = {
                        kb: tensor.clone().detach().to(self.device)
                        for kb, tensor in nb_connect.items()
                    }

                unique_subgraph_cache[key] = {
                    'canonical_orders': np.array(g.canonical_orders, copy=True),
                    'maps_canonical_orders': np.array(g.maps_canonical_orders, copy=True),
                    'n_permutations': g.n_permutations,
                    'feature_atypes': cached_feature_atypes,
                    'feature_indices': cached_feature_indices,
                    'weights': g.weights.clone().detach().to(self.device),
                    'nb_connect': cached_nb_connect,
                    'n_sym_perm': g.n_sym_perm
                }

        # 3. Final pass: Assign cached results to all subgraphs
        for i, g in enumerate(self.subgraphs):
            key = subgraph_type_keys[i]
            cached_data = unique_subgraph_cache[key]
            g.canonical_orders = np.array(cached_data['canonical_orders'], copy=True)
            g.maps_canonical_orders = np.array(cached_data['maps_canonical_orders'], copy=True)
            g.n_permutations = cached_data['n_permutations']
            g.feature_atypes = {
                kb: tensor.clone().detach().to(self.device)
                for kb, tensor in cached_data['feature_atypes'].items()
            }
            g.feature_indices = {
                kb: {kf: idx_tensor.clone().detach().to(self.device)
                     for kf, idx_tensor in tensors.items()}
                for kb, tensors in cached_data['feature_indices'].items()
            }
            if cached_data['nb_connect'] is not None:
                g.nb_connect = {
                    kb: tensor.clone().detach().to(self.device)
                    for kb, tensor in cached_data['nb_connect'].items()
                }
            else:
                g.nb_connect = None
            g.weights = cached_data['weights'].clone().detach().to(self.device)
            g.n_sym_perm = cached_data['n_sym_perm']
        return

    def _update_subgraph_positions(self):
        """Propagate parent positions to each subgraph."""
        if self.positions is None:
            return
        for g in self.subgraphs:
            map_indices = torch.as_tensor(g.map_sub2parent, dtype=torch.long, device=self.device)
            g.positions = distribute_v3_torch(self.positions, map_indices)
        return


    def set_internal_coords_indices(self):
        """Fully vectorized and PyTorch-based version of IC indices generation."""
        # --- Bonds ---
        self.bonds = torch.tensor(self.bonds, dtype=torch.long, device=self.device)
        if self.bonds.numel() > 0:
            a0, a1 = self.bonds.T
            at0 = [self.list_atom_elems[i] for i in a0.cpu().numpy()]
            at1 = [self.list_atom_elems[i] for i in a1.cpu().numpy()]
            r0 = torch.tensor([COVALENT_RADIUS[e] for e in at0], device=self.device)
            r1 = torch.tensor([COVALENT_RADIUS[e] for e in at1], device=self.device)
            self.b0 = r0 + r1
        else:
            self.b0 = torch.empty(0, device=self.device)
        self.n_bonds = len(self.bonds)
        self.bond_map = {tuple(sorted(bond.tolist())): i for i, bond in enumerate(self.bonds)}

        # --- Angles ---
        angles = []
        for i in range(self.n_atoms):
            neighbors = self.connectivity[i].nonzero(as_tuple=False).flatten()
            if len(neighbors) >= 2:
                j_atoms, k_atoms = torch.combinations(neighbors, r=2).T
                i_atoms = torch.full_like(j_atoms, i)
                angles.append(torch.stack([j_atoms, i_atoms, k_atoms], dim=1))
        self.angles = torch.cat(angles, dim=0) if angles else torch.empty((0, 3), dtype=torch.long, device=self.device)
        self.n_angles = len(self.angles)
        self.angle_map = {tuple(ang.tolist()): i for i, ang in enumerate(self.angles)}
        self.angle_map.update({tuple(ang.tolist()[::-1]): i for i, ang in enumerate(self.angles)})
        
        # --- cos_a0 ---
        if self.n_angles > 0:
            center_atoms = self.angles[:, 1]
            valences = self.valences[center_atoms]
            atom_elems_np = np.array(self.list_atom_elems)[center_atoms.cpu().numpy()]
            
            cos_a0 = torch.zeros(self.n_angles, device=self.device, dtype=torch.float32)
            
            cos_a0[valences == 4] = np.cos(109.45 / 180 * np.pi)
            cos_a0[valences == 3] = np.cos(120.00 / 180 * np.pi)
            cos_a0[valences == 2] = np.cos(np.pi)
            
            mask_v3_N = (valences == 3) & torch.from_numpy(atom_elems_np == 'N').to(self.device)
            cos_a0[mask_v3_N] = np.cos(107. / 180 * np.pi)
            
            mask_v2_O_S = (valences == 2) & torch.from_numpy((atom_elems_np == 'O') | (atom_elems_np == 'S')).to(self.device)
            cos_a0[mask_v2_O_S] = np.cos(104.45 / 180 * np.pi)
            
            mask_v2_N = (valences == 2) & torch.from_numpy(atom_elems_np == 'N').to(self.device)
            cos_a0[mask_v2_N] = np.cos(120. / 180 * np.pi)
            
            self.cos_a0 = cos_a0
        else:
            self.cos_a0 = torch.empty(0, device=self.device)

        # --- Dihedrals ---
        diheds = []
        if self.n_bonds > 0:
            for j, k in self.bonds:
                neighbors_j = self.connectivity[j].nonzero(as_tuple=False).flatten()
                neighbors_k = self.connectivity[k].nonzero(as_tuple=False).flatten()
                i_atoms = neighbors_j[neighbors_j != k]
                l_atoms = neighbors_k[neighbors_k != j]
                if len(i_atoms) > 0 and len(l_atoms) > 0:
                    i_grid, l_grid = torch.cartesian_prod(i_atoms, l_atoms).T
                    j_grid = torch.full_like(i_grid, j.item())
                    k_grid = torch.full_like(i_grid, k.item())
                    diheds.append(torch.stack([i_grid, j_grid, k_grid, l_grid], dim=1))

        self.diheds = torch.cat(diheds, dim=0) if diheds else torch.empty((0, 4), dtype=torch.long, device=self.device)
        self.n_diheds = len(self.diheds)
        self.dihed_map = {tuple(d.tolist()): i for i, d in enumerate(self.diheds)}
        self.dihed_map.update({tuple(d.tolist()[::-1]): i for i, d in enumerate(self.diheds)})
        
        def calc_internal_coords_features(positions, box):
            is_batched = positions.dim() == 3
            # bonds
            a0 = self.bonds[:, 0]
            a1 = self.bonds[:, 1]
            p0 = distribute_v3_torch(positions, a0)
            p1 = distribute_v3_torch(positions, a1)
            
            box_inv = torch.linalg.inv(box) if box is not None else None

            if box is None:
                dp = p1 - p0
            else:
                dp = pbc_shift_torch(p1 - p0, box, box_inv)
            b = torch.sqrt(torch.sum(dp**2, dim=-1))
            fb = FSCALE_BOND * (b - self.b0)

            # angles
            a0 = self.angles[:, 0]
            a1 = self.angles[:, 1]
            a2 = self.angles[:, 2]
            p0 = distribute_v3_torch(positions, a0)
            p1 = distribute_v3_torch(positions, a1)
            p2 = distribute_v3_torch(positions, a2)
            if box is None:
                v1 = p0 - p1
                v2 = p2 - p1
            else:
                v1 = pbc_shift_torch(p0 - p1, box, box_inv)
                v2 = pbc_shift_torch(p2 - p1, box, box_inv)
            v1_norm = torch.sqrt(torch.sum(v1**2, dim=-1))
            v2_norm = torch.sqrt(torch.sum(v2**2, dim=-1))
            v1_norm = torch.clamp(v1_norm, min=1e-6)
            v2_norm = torch.clamp(v2_norm, min=1e-6)
            
            v1_u = v1 / v1_norm.unsqueeze(-1)
            v2_u = v2 / v2_norm.unsqueeze(-1)
            cos_a = torch.sum(v1_u * v2_u, dim=-1)
            cos_a = torch.clamp(cos_a, min=-1.0, max=1.0)
            fa = FSCALE_ANGLE * (cos_a - self.cos_a0)

            # diheds
            a0 = self.diheds[:, 0]
            a1 = self.diheds[:, 1]
            a2 = self.diheds[:, 2]
            a3 = self.diheds[:, 3]
            p0 = distribute_v3_torch(positions, a0)
            p1 = distribute_v3_torch(positions, a1)
            p2 = distribute_v3_torch(positions, a2)
            p3 = distribute_v3_torch(positions, a3)
            if box is None:
                v1 = p1 - p0
                v2 = p2 - p1
                v3 = p3 - p2
            else:
                v1 = pbc_shift_torch(p1 - p0, box, box_inv)
                v2 = pbc_shift_torch(p2 - p1, box, box_inv)
                v3 = pbc_shift_torch(p3 - p2, box, box_inv)

            if is_batched:
                c1 = torch.cross(v1, v2, dim=-1)
                c2 = torch.cross(v2, v3, dim=-1)
                c3 = torch.cross(c1, c2, dim=-1)
            else:
                c1 = torch.cross(v1, v2)
                c2 = torch.cross(v2, v3)
                c3 = torch.cross(c1, c2)

            c1_norm = torch.sqrt(torch.sum(c1**2, dim=-1))
            c2_norm = torch.sqrt(torch.sum(c2**2, dim=-1))
            c1_norm = torch.clamp(c1_norm, min=1e-6)
            c2_norm = torch.clamp(c2_norm, min=1e-6)
            
            c1_u = c1 / c1_norm.unsqueeze(-1)
            c2_u = c2 / c2_norm.unsqueeze(-1)
            
            fd = torch.sum(c1_u * c2_u, dim=-1)
            fd = torch.clamp(fd, min=-1.0, max=1.0)
            
            return fb, fa, fd

        self.calc_internal_coords_features = calc_internal_coords_features

        return

    def prepare_subgraph_feature_calc(self):
        r'''
        Preparing the feature calculation.
        Specifically, find out the indices mapping between feature elements and ICs.

        After preparing the variables in all subgraphs, we stack all subgraphs along the first axis.
        After stacking, each row represents a fixed-order subgraph calculation.
        The total number of rows: Ntot = \sum_g N_p(g), with N_p(g) being the permutation number of subgraph g.
        Get these variables ready:
        (kb = ['center', 'nb_bonds_0', 'nb_bonds_1'])
        (kf = ['bonds', 'angles0', 'angles1', 'diheds'])
        feature_atypes: (Ntot, 2*MAX_VALENCE-1, DIM_BOND_FEATURES_ATYPES)
        feature_indices[kf]: (Ntot, 2*MAX_VALENCE-1, DIM_BOND_FEATURES_GEOM[kf])
        nb_connect[kb]: (Ntot, MAX_VALENCE-1)
        self.n_features: dimensionality of bond features

        Also setup the following function:
        self.calc_subgraph_features: 
            pos (Na*3), box (3*3) -> features (Ntot*7*n_features)
                The calculator for the Graph features.
        '''
        # for g in self.subgraphs:
        #     g.prepare_graph_feature_calc()
        self.n_features_atypes = DIM_BOND_FEATURES_ATYPES
        self.n_features_geom = DIM_BOND_FEATURES_GEOM_TOT
        self.n_features = self.n_features_atypes + self.n_features_geom

        # concatenate permutations
        self.feature_atypes = {}
        self.feature_indices = {}
        if self.nn == 0:
            bond_groups = ['center']
        else:
            bond_groups = ['center', 'nb_bonds_0', 'nb_bonds_1']
        feature_groups = ['bonds', 'angles0', 'angles1', 'diheds']
        for kb in bond_groups:
            self.feature_atypes[kb] = torch.cat(
                [g.feature_atypes[kb].clone().detach().to(self.device) for g in self.subgraphs])
            self.feature_indices[kb] = {}
            for kf in feature_groups:
                self.feature_indices[kb][kf] = torch.cat(
                    [g.feature_indices[kb][kf].clone().detach().to(self.device) for g in self.subgraphs])
        self.weights = torch.cat([g.weights.clone().detach().to(self.device) for g in self.subgraphs])
        if self.nn == 1:
            self.nb_connect = {}
            for kb in ['nb_bonds_0', 'nb_bonds_1']:
                self.nb_connect[kb] = torch.cat([
                    torch.tile(g.nb_connect[kb].clone().detach().to(self.device), (g.n_sym_perm, 1))
                    for g in self.subgraphs
                ])
        self.map_subgraph_perm = torch.cat([
            torch.full((self.subgraphs[ig].n_sym_perm,), ig, dtype=torch.long, device=self.device)
            for ig in range(self.n_subgraphs)
        ])

        print(f"self.feature_atypes['center']: {self.feature_atypes['center']}")
        print(f"self.feature_atypes['nb_bonds_0']: {self.feature_atypes['nb_bonds_0']}")
        print(f"self.feature_atypes['nb_bonds_1']: {self.feature_atypes['nb_bonds_1']}")
        
        # concatenate bond groups
        if self.nn == 0:
            self.feature_atypes = self.feature_atypes['center']
        elif self.nn == 1:
            self.feature_atypes = torch.cat([
                self.feature_atypes['center'],
                self.feature_atypes['nb_bonds_0'],
                self.feature_atypes['nb_bonds_1']
            ], dim=1)

        
        feature_indices = {}
        for kf in feature_groups:
            if self.nn == 0:
                feature_indices[kf] = self.feature_indices['center'][kf]
            elif self.nn == 1:
                feature_indices[kf] = torch.cat([
                    self.feature_indices['center'][kf],
                    self.feature_indices['nb_bonds_0'][kf],
                    self.feature_indices['nb_bonds_1'][kf]
                ], dim=1)
        self.feature_indices = feature_indices
        if self.nn == 1:
            self.nb_connect = torch.cat(
                [self.nb_connect['nb_bonds_0'], self.nb_connect['nb_bonds_1']],
                dim=1)

        # set up the feature calculation function
        def _get_features(fb, fa, fd, f_atypes, indices_bonds, indices_angles0,
                          indices_angles1, indices_diheds):
            is_batched = fb.dim() == 2

            f_bonds = distribute_scalar_torch(fb, indices_bonds)
            f_angles0 = distribute_scalar_torch(fa, indices_angles0)
            f_angles1 = distribute_scalar_torch(fa, indices_angles1)
            f_diheds = distribute_scalar_torch(fd, indices_diheds)

            if is_batched:
                batch_size = fb.shape[0]
                f_atypes_expanded = f_atypes.unsqueeze(0).expand(batch_size, -1, -1, -1)
                features = torch.cat(
                    (f_atypes_expanded, f_bonds, f_angles0, f_angles1, f_diheds), dim=-1)
            else:
                features = torch.cat(
                    (f_atypes, f_bonds, f_angles0, f_angles1, f_diheds), dim=2)
            return features

        def calc_subgraph_features(positions, box):
            fb, fa, fd = self.calc_internal_coords_features(positions, box)
            # print(fb)
            # print(fa)
            # print(fd)
            # print(self.feature_indices)
            features = _get_features(fb, fa, fd, self.feature_atypes,
                                     self.feature_indices['bonds'],
                                     self.feature_indices['angles0'],
                                     self.feature_indices['angles1'],
                                     self.feature_indices['diheds'])
            return features

        self.calc_subgraph_features = calc_subgraph_features
        return

    def write_xyz(self, file=None):
        '''
        Write the xyz file of the molecule
        '''
        if file is None:
            file = sys.stdout
        print(self.n_atoms, file=file)
        print('Generated by dmff.sgnn', file=file)
        for i in range(self.n_atoms):
            elem = self.list_atom_elems[i]
            x, y, z = self.positions[i].cpu().numpy()
            print('%3s %12.6f %12.6f %12.6f' % (elem, x, y, z), file=file)
        return


class TopSubGraph(TopGraph):

    def __init__(self, graph, i_center, nn, type_center='bond'):
        '''
        Find a subgraph within the graph, centered on a certain bond/atom
        The size of the subgraph is determined by nn (# of neighbour searches around the center)
        i_center defines the center, could be a bond, could be an atom
        '''
        self.device = graph.device
        self.list_atom_elems = []
        self.bonds = []
        self.positions = []
        self.valences = []
        self.map_sub2parent = [] 
        self.map_parent2sub = {}
        self.parent = graph
        self.box = graph.box
        self.box_inv = graph.box_inv
        self.nn = nn
        n_atoms = 0
        if type_center == 'atom':
            self.map_sub2parent.append(i_center)
            self.map_parent2sub[i_center] = n_atoms
            n_atoms += 1
            self.list_atom_elems.append(graph.list_atom_elems[i_center])
            self.valences.append(graph.valences[i_center].item() if isinstance(graph.valences[i_center], torch.Tensor) else graph.valences[i_center])
        elif type_center == 'bond':
            b0 = graph.bonds[i_center]
            for i in b0:
                i_item = i.item() if isinstance(i, torch.Tensor) else i
                self.map_sub2parent.append(i_item)
                self.map_parent2sub[i_item] = n_atoms
                n_atoms += 1
                self.list_atom_elems.append(graph.list_atom_elems[i_item])
                self.valences.append(graph.valences[i_item].item() if isinstance(graph.valences[i_item], torch.Tensor) else graph.valences[i_item])
            # the first bond of the subgraph is always (0, 1), the central bond
            self.bonds.append([0, 1])
        self.n_atoms = n_atoms

        for n in range(nn + 1):
            self.add_neighbors()
        self._build_connectivity()

        self.map_sub2parent.append(-1)  
        self.map_sub2parent = np.array(self.map_sub2parent)
        if graph.positions is not None:
            valid_indices = [idx for idx in self.map_sub2parent[:-1] if idx >= 0]
            if len(valid_indices) > 0:
                self.positions = graph.positions[torch.tensor(valid_indices, device=self.device)]
            else:
                self.positions = torch.zeros((self.n_atoms, 3), device=self.device)
        else:
            self.positions = torch.zeros((self.n_atoms, 3), device=self.device)

        return

    # search one more layer of neighbours
    def add_neighbors(self):
        atoms_in_subgraph = list(self.map_parent2sub.keys())
        n_atoms = self.n_atoms
        for b in self.parent.bonds:
            b_list = b.tolist() if isinstance(b, torch.Tensor) else list(b)
            flags = [i not in atoms_in_subgraph for i in b_list]
            if sum(flags) == 1:
                i_old_idx = 0 if not flags[0] else 1
                i_new_idx = 1 if not flags[0] else 0
                i_old = b_list[i_old_idx]
                i_new = b_list[i_new_idx]
                self.list_atom_elems.append(self.parent.list_atom_elems[i_new])
                # Don't append to positions here - it will be set after all neighbors are added
                valence_val = self.parent.valences[i_new]
                self.valences.append(valence_val.item() if isinstance(valence_val, torch.Tensor) else valence_val)
                self.map_sub2parent.append(i_new)
                self.map_parent2sub[i_new] = n_atoms
                bond = sorted([n_atoms, self.map_parent2sub[i_old]])
                self.bonds.append(bond)
                n_atoms += 1
        self.n_atoms = n_atoms
        return

    def _build_connectivity(self):
        """Construct connectivity and valence tensors for the subgraph."""
        self.connectivity = torch.zeros((self.n_atoms, self.n_atoms), dtype=torch.int, device=self.device)
        for bond in self.bonds:
            if isinstance(bond, torch.Tensor):
                i, j = bond.tolist()
            else:
                i, j = bond
            self.connectivity[i, j] = 1
            self.connectivity[j, i] = 1
        self.valences = torch.sum(self.connectivity, dim=1)
        return

    def get_canonical_orders_wt_permutation_grps(self):
        """Mirror the numpy implementation to ensure identical canonical ordering."""
        if len(self.bonds) == 0 or list(self.bonds[0]) != [0, 1]:
            raise RuntimeError("get_canonical_orders_wt_permutation_grps currently supports bond-centered subgraphs only.")

        if self.atom_types[0] == self.atom_types[1]:
            orders = [np.array([0, 1], dtype=int), np.array([1, 0], dtype=int)]
        else:
            if self.atom_types[0] < self.atom_types[1]:
                orders = [np.array([0, 1], dtype=int)]
            else:
                orders = [np.array([1, 0], dtype=int)]

        def permute_using_atypes(indices, atypes):
            if len(indices) == 0:
                return [indices]
            set_atypes = sorted(set(atypes))
            permutation_grps = []
            for t in set_atypes:
                mask = atypes == t
                permutation_grps.append(indices[mask])
            seg_permutations = [list(permutations(seg)) for seg in permutation_grps]
            pfull = []
            for p in product(*seg_permutations):
                pfull.append(np.concatenate(p))
            return pfull

        def extend_orders(current_orders):
            n_order = len(current_orders)
            for _ in range(n_order):
                order = current_orders.pop(0)
                seg_permutations = []
                for i in order:
                    js = np.where(self.connectivity[i].cpu().numpy() == 1)[0]
                    js = js[[j not in order for j in js]]
                    if len(js) == 0:
                        continue
                    atypes = np.array(self.atom_types)[js]
                    new_orders = permute_using_atypes(js, atypes)
                    seg_permutations.append(new_orders)
                if len(seg_permutations) == 0:
                    current_orders.append(order)
                    continue
                for p in product(*seg_permutations):
                    current_orders.append(np.concatenate((order, np.concatenate(p))))
            return current_orders

        for _ in range(self.nn + 1):
            orders = extend_orders(orders)

        canonical_orders = np.array(orders, dtype=int)
        maps_canonical_orders = []
        for order in canonical_orders:
            map_order = np.zeros(self.n_atoms, dtype=int)
            for ii, i in enumerate(order):
                map_order[i] = ii
            maps_canonical_orders.append(map_order)
        maps_canonical_orders = np.array(maps_canonical_orders, dtype=int)

        self.canonical_orders = canonical_orders
        self.maps_canonical_orders = maps_canonical_orders
        self.n_permutations = len(canonical_orders)

        return

    def prepare_bond_feature_atypes(self, bond, map_order: torch.Tensor) -> torch.Tensor:
        """Generates atom type one-hot features for a bond."""

        if not isinstance(map_order, torch.Tensor):
            map_order = torch.as_tensor(map_order, dtype=torch.long, device=self.device)
        else:
            map_order = map_order.to(self.device)

        def sort_by_order_torch(indices: torch.Tensor, order_map: torch.Tensor) -> torch.Tensor:
            if indices.numel() == 0:
                return indices
            if not isinstance(order_map, torch.Tensor):
                order_map = torch.as_tensor(order_map, dtype=torch.long, device=indices.device)
            else:
                order_map = order_map.to(indices.device)
            _, sort_indices = torch.sort(order_map[indices])
            return indices[sort_indices]

        if not isinstance(bond, torch.Tensor):
            bond = torch.tensor(bond, dtype=torch.long, device=self.device)

        indices_atoms_center = sort_by_order_torch(bond, map_order)
        i, j = indices_atoms_center[0].item(), indices_atoms_center[1].item()

        fi = F.one_hot(
            torch.tensor(ATYPE_INDEX[self.list_atom_elems[i]], device=self.device),
            num_classes=N_ATYPES,
        ).float()
        fj = F.one_hot(
            torch.tensor(ATYPE_INDEX[self.list_atom_elems[j]], device=self.device),
            num_classes=N_ATYPES,
        ).float()

        indices_n0 = self.connectivity[i].nonzero(as_tuple=False).flatten()
        indices_n1 = self.connectivity[j].nonzero(as_tuple=False).flatten()

        indices_n0 = indices_n0[indices_n0 != j]
        indices_n1 = indices_n1[indices_n1 != i]

        indices_n0 = sort_by_order_torch(indices_n0, map_order)
        indices_n1 = sort_by_order_torch(indices_n1, map_order)

        f_n0 = torch.zeros(N_ATYPES * (MAX_VALENCE - 1), dtype=torch.float32, device=self.device)
        if indices_n0.numel() > 0:
            neighbor_elems_n0 = [ATYPE_INDEX[self.list_atom_elems[idx.item()]] for idx in indices_n0]
            one_hot_n0 = F.one_hot(
                torch.tensor(neighbor_elems_n0, device=self.device),
                num_classes=N_ATYPES,
            ).float()
            f_n0[: len(indices_n0) * N_ATYPES] = one_hot_n0.flatten()

        f_n1 = torch.zeros(N_ATYPES * (MAX_VALENCE - 1), dtype=torch.float32, device=self.device)
        if indices_n1.numel() > 0:
            neighbor_elems_n1 = [ATYPE_INDEX[self.list_atom_elems[idx.item()]] for idx in indices_n1]
            one_hot_n1 = F.one_hot(
                torch.tensor(neighbor_elems_n1, device=self.device),
                num_classes=N_ATYPES,
            ).float()
            f_n1[: len(indices_n1) * N_ATYPES] = one_hot_n1.flatten()

        return torch.cat((fi, fj, f_n0, f_n1)).to(self.device)

    def prepare_bond_feature_calc_indices(self, bond, map_order: torch.Tensor, verbose=False) -> Dict[str, torch.Tensor]:
        """
        Prepares geometric feature indices for a given bond, fully in PyTorch.
        bond can be a list, numpy array, or torch tensor.
        """
        if not isinstance(map_order, torch.Tensor):
            map_order = torch.as_tensor(map_order, dtype=torch.long, device=self.device)
        else:
            map_order = map_order.to(self.device)
        # Convert bond to torch tensor if needed
        if not isinstance(bond, torch.Tensor):
            bond = torch.tensor(bond, dtype=torch.long, device=self.device)
        
        
        def sort_by_order_torch(indices: torch.Tensor, order_map: torch.Tensor) -> torch.Tensor:
            if indices.numel() == 0:
                return indices
            if not isinstance(order_map, torch.Tensor):
                order_map = torch.as_tensor(order_map, dtype=torch.long, device=indices.device)
            else:
                order_map = order_map.to(indices.device)
            _ , sort_indices = torch.sort(order_map[indices])
            return indices[sort_indices]
        
        indices = {}
        G = self.parent
        
        # --- Atom and Neighbor Setup ---
        indices_atoms_center = sort_by_order_torch(bond, map_order)
        i, j = indices_atoms_center[0].item(), indices_atoms_center[1].item()
        
        indices_n0 = self.connectivity[i].nonzero(as_tuple=False).flatten()
        indices_n1 = self.connectivity[j].nonzero(as_tuple=False).flatten()
        indices_n0 = sort_by_order_torch(indices_n0[indices_n0 != j], map_order)
        indices_n1 = sort_by_order_torch(indices_n1[indices_n1 != i], map_order)
        
        # --- Padding  ---
        indices_atoms_n0 = torch.full((MAX_VALENCE - 1,), -1, dtype=torch.long, device=self.device)
        indices_atoms_n1 = torch.full((MAX_VALENCE - 1,), -1, dtype=torch.long, device=self.device)
        indices_atoms_n0[:len(indices_n0)] = indices_n0
        indices_atoms_n1[:len(indices_n1)] = indices_n1

        # --- Relevant Bonds ---
        bond_indices_to_lookup = torch.cat([
            indices_atoms_center.unsqueeze(0),
            torch.stack([torch.full_like(indices_atoms_n0, i), indices_atoms_n0], dim=1),
            torch.stack([torch.full_like(indices_atoms_n1, j), indices_atoms_n1], dim=1)
        ])
        
        bonds_list = []
        for b in bond_indices_to_lookup:
            if b[1] < 0:
                bonds_list.append(-1)
                continue
            
            p = tuple(sorted((self.map_sub2parent[b[0].item()], self.map_sub2parent[b[1].item()])))
            bonds_list.append(G.bond_map.get(p, -1))
        indices['bonds'] = torch.tensor(bonds_list, dtype=torch.long, device=self.device)

        # --- Relevant Angles ---
        set_0 = torch.cat([indices_atoms_center[1].unsqueeze(0), indices_atoms_n0])
        set_1 = torch.cat([indices_atoms_center[0].unsqueeze(0), indices_atoms_n1])
        
        
        combos_0 = torch.combinations(set_0, r=2)
        combos_1 = torch.combinations(set_1, r=2)
        
        angles0_to_lookup = torch.stack([combos_0[:, 0], torch.full_like(combos_0[:, 0], i), combos_0[:, 1]], dim=1)
        angles1_to_lookup = torch.stack([combos_1[:, 0], torch.full_like(combos_1[:, 0], j), combos_1[:, 1]], dim=1)

        angles0_list, angles1_list = [], []
        for a in angles0_to_lookup:
            if a[0] < 0 or a[2] < 0:
                angles0_list.append(-1)
                continue
            p = tuple(self.map_sub2parent[idx.item()] for idx in a)
            angles0_list.append(G.angle_map.get(p, G.angle_map.get(tuple(reversed(p)), -1)))
            
        for a in angles1_to_lookup:
            if a[0] < 0 or a[2] < 0:
                angles1_list.append(-1)
                continue
            p = tuple(self.map_sub2parent[idx.item()] for idx in a)
            angles1_list.append(G.angle_map.get(p, G.angle_map.get(tuple(reversed(p)), -1)))
            
        indices['angles0'] = torch.tensor(angles0_list, dtype=torch.long, device=self.device)
        indices['angles1'] = torch.tensor(angles1_list, dtype=torch.long, device=self.device)

        # --- Relevant Dihedrals  ---
        dihedrals_list = []
        parent_diheds = G.diheds
        if isinstance(parent_diheds, np.ndarray):
            parent_diheds = torch.from_numpy(parent_diheds)
        parent_diheds = parent_diheds.to(self.device)
        for idx_i in indices_atoms_n0:
            i_val = idx_i.item()
            for idx_l in indices_atoms_n1:
                l_val = idx_l.item()
                if i_val < 0 or l_val < 0 or parent_diheds.numel() == 0:
                    dihedrals_list.append(-1)
                    continue

                p = torch.tensor([
                    self.map_sub2parent[i_val],
                    self.map_sub2parent[i],
                    self.map_sub2parent[j],
                    self.map_sub2parent[l_val]
                ], dtype=torch.long, device=self.device)

                match = torch.nonzero(
                    torch.all(parent_diheds == p, dim=1)
                    | torch.all(parent_diheds == torch.flip(p, dims=(0,)), dim=1),
                    as_tuple=False
                )

                if match.numel() == 0:
                    dihedrals_list.append(-1)
                else:
                    dihedrals_list.append(match[0, 0].item())

        indices['diheds'] = torch.tensor(dihedrals_list, dtype=torch.long, device=self.device)

        return indices

    def prepare_graph_feature_calc(self):
        '''
        Prepare the variables that are needed in feature calculations.
        So far, we assume self.nn <= 1, so it is either only the central bond, or the central bond + its closest neighbor bonds
        The closest neighbor bonds are grouped into two groups: (nb_bonds_0) and (nb_bonds_1)
        The first group of bonds are attached to the first atom of the central bond
        The second group of bonds are attached to the second atom of the central bond
        So there are three bond groups: center (1bond), nb_bonds_0 (max 3 bonds), and nb_bonds_1 (max 3 bonds)
        In principle, it's not necessary to dinstinguish nb_bonds_0 and nb_bonds_1. Such division is merely a historical legacy.

        The following variables are set after the execution of this function

        Output: 
            self.feature_atypes:
                Dictionary with bond groups (['center', 'nb_bonds_0', 'nb_bonds_1']) as keywords
                'center': this group contains only one bond: the central bond
                'nb_bonds_0': this group contains the neighbor bonds attached to the first atoms
                'nb_bonds_1': this group contains the neighbor bonds attached to the second atoms
                feature_atypes['...'] is a (n_sym_perm, n_bonds, n_bond_features_atype) array, stores the atype features
                of the bond group. Atype features describes the atomtyping information of the graph, thus is bascially constant
                during the simulation.
            self.feature_indices:
                Nested dictionary with bond groups (['center', 'nb_bonds_0', 'nb_bonds_1']) as the first keyword
                and geometric feature types (['bonds', 'angles0', 'angles1', 'diheds']) as the second keyword
                It stores all the relevant IC indices
                Dimensionalities (when MAX_VALENCE=4):
                feature_indices['center']['bonds']: (n_sym_perm, 1, 7)
                feature_indices['center']['angles0']: (n_sym_perm, 1, 6)
                feature_indices['center']['angles1']: (n_sym_perm, 1, 6)
                feature_indices['center']['diheds']: (n_sym_perm, 1, 9)
                feature_indices['nb_bonds_x']['bonds']: (n_sym_perm, 3, 7)
                feature_indices['nb_bonds_x']['angles0']: (n_sym_perm, 3, 6)
                feature_indices['nb_bonds_x']['angles1']: (n_sym_perm, 3, 6)
                feature_indices['nb_bonds_x']['diheds']: (n_sym_perm, 3, 9)
            self.nb_connect:
                Dictionary with keywords: ['nb_bonds_0', 'nb_bonds_1']
                Describes how many neighbor bonds the central bond has. E.g., if there are only 2 neighbor bonds attached to 
                the first atom, then:
                self.nb_connect['nb_bonds_0'] = torch.tensor([1., 1., 0.])

        '''
        self.n_bond_features_atypes = DIM_BOND_FEATURES_ATYPES
        self.n_bond_features_geom = DIM_BOND_FEATURES_GEOM_TOT
        self.n_bond_features = self.n_bond_features_atypes + self.n_bond_features_geom

        center_bond = self.bonds[0]
        i, j = center_bond[0], center_bond[1]

        # ---  Find neighboring bonds---
        nb_bonds_0, nb_bonds_1 = [], []
        if self.nn == 1:
            parent_bonds = self.parent.bonds
            map_sub2parent_i, map_sub2parent_j = self.map_sub2parent[i], self.map_sub2parent[j]
            for b0_b1 in parent_bonds:
                b0 = b0_b1[0].item() if isinstance(b0_b1[0], torch.Tensor) else b0_b1[0]
                b1 = b0_b1[1].item() if isinstance(b0_b1[1], torch.Tensor) else b0_b1[1]
                if b0 == map_sub2parent_i and b1 != map_sub2parent_j and b1 in self.map_parent2sub:
                    nb_bonds_0.append([i, self.map_parent2sub[b1]])
                elif b1 == map_sub2parent_i and b0 != map_sub2parent_j and b0 in self.map_parent2sub:
                    nb_bonds_0.append([i, self.map_parent2sub[b0]])
                elif b0 == map_sub2parent_j and b1 != map_sub2parent_i and b1 in self.map_parent2sub:
                    nb_bonds_1.append([j, self.map_parent2sub[b1]])
                elif b1 == map_sub2parent_j and b0 != map_sub2parent_i and b0 in self.map_parent2sub:
                    nb_bonds_1.append([j, self.map_parent2sub[b0]])

        # --- Step 2: Calculate features once ---
        all_perms_indices = []
        all_perms_atypes = []
        for map_order in self.maps_canonical_orders:
            perm_indices = {'center': self.prepare_bond_feature_calc_indices(center_bond, map_order)}
            perm_atypes = {'center': self.prepare_bond_feature_atypes(center_bond, map_order)}
            if self.nn == 1:
                perm_indices['nb_bonds_0'] = [self.prepare_bond_feature_calc_indices(b, map_order) for b in nb_bonds_0]
                perm_atypes['nb_bonds_0'] = [self.prepare_bond_feature_atypes(b, map_order) for b in nb_bonds_0]
                perm_indices['nb_bonds_1'] = [self.prepare_bond_feature_calc_indices(b, map_order) for b in nb_bonds_1]
                perm_atypes['nb_bonds_1'] = [self.prepare_bond_feature_atypes(b, map_order) for b in nb_bonds_1]
            all_perms_indices.append(perm_indices)
            all_perms_atypes.append(perm_atypes)

        self.n_sym_perm = self.n_permutations
        if self.n_sym_perm == 0:
            self.feature_indices = {'center': {k: torch.empty(0) for k in DIM_BOND_FEATURES_GEOM.keys()}}
            self.feature_atypes = {'center': torch.empty(0)}
            self.weights = torch.empty(0, device=self.device)
            return
        self.weights = torch.full((self.n_sym_perm,), 1.0 / self.n_sym_perm, dtype=torch.float32, device=self.device)

        feature_indices = {}
        feature_atypes = {}
        bond_groups = ['center'] + (['nb_bonds_0', 'nb_bonds_1'] if self.nn == 1 else [])
        nb_map = {'center': 1, 'nb_bonds_0': MAX_VALENCE - 1, 'nb_bonds_1': MAX_VALENCE - 1}

        for kb in bond_groups:
            atype_arrays = torch.zeros(
                (self.n_sym_perm, nb_map[kb], DIM_BOND_FEATURES_ATYPES),
                dtype=torch.float32,
                device=self.device,
            )
            for ip, perm_atypes in enumerate(all_perms_atypes):
                if kb == 'center':
                    atype_arrays[ip, 0, :] = perm_atypes[kb].to(torch.float32)
                else:
                    num_bonds = len(perm_atypes.get(kb, []))
                    for ib in range(num_bonds):
                        atype_arrays[ip, ib, :] = perm_atypes[kb][ib].to(torch.float32)
            feature_atypes[kb] = atype_arrays

            feature_indices[kb] = {}
            for kf, dim in DIM_BOND_FEATURES_GEOM.items():
                index_arrays = -torch.ones(
                    (self.n_sym_perm, nb_map[kb], dim),
                    dtype=torch.long,
                    device=self.device,
                )
                for ip, perm_indices in enumerate(all_perms_indices):
                    if kb == 'center':
                        index_arrays[ip, 0, :] = perm_indices[kb][kf]
                    else:
                        num_bonds = len(perm_indices.get(kb, []))
                        for ib in range(num_bonds):
                            index_arrays[ip, ib, :] = perm_indices[kb][ib][kf]
                feature_indices[kb][kf] = index_arrays

        if self.nn == 1:
            self.nb_connect = {}
            for kb in ['nb_bonds_0', 'nb_bonds_1']:
                connect_mask = torch.zeros(MAX_VALENCE - 1, dtype=torch.float32, device=self.device)
                max_bonds = max(len(perm_indices.get(kb, [])) for perm_indices in all_perms_indices)
                if max_bonds > 0:
                    connect_mask[:max_bonds] = 1.0
                self.nb_connect[kb] = connect_mask

        self.feature_indices = feature_indices
        self.feature_atypes = feature_atypes
        
        return


def sort_by_order(ilist, map_order):
    return np.array(ilist)[np.argsort([map_order[i] for i in ilist])]


def from_pdb(pdb, device='cpu'):
    device = torch.device(device)
    mol = md.load(pdb)
    bonds = []
    for bond in mol.top.bonds:
        bonds.append(np.sort(np.array((bond.atom1.index, bond.atom2.index))))
    bonds = np.array(bonds)
    list_atom_elems = np.array([a.element.symbol for a in mol.top.atoms])
    positions = torch.tensor(mol.xyz[0] * 10, dtype=torch.float32, device=device)
    if mol.unitcell_vectors is None:
        box = None
    else:
        box = torch.tensor(mol.unitcell_vectors[0] * 10, dtype=torch.float32, device=device)
    return TopGraph(list_atom_elems, bonds, positions=positions, box=box, device=device)
