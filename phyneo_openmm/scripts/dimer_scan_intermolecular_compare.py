#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit

try:
    import mpidplugin  # noqa: F401
except Exception:
    mpidplugin = None

try:
    import jax.numpy as jnp
    from dmff import Hamiltonian
    from dmff.common import nblist
except Exception:
    Hamiltonian = None


DEFAULT_M_SCALES = [0.0, 0.0, 0.0, 0.0, 0.0]
DEFAULT_P_SCALES = [0.0, 0.0, 0.0, 0.0, 0.0]
DEFAULT_D_SCALES = [1.0, 1.0, 1.0, 1.0, 1.0]


def residue_atom_indices(topology, residue_index):
    res = list(topology.residues())[residue_index]
    return [a.index for a in res.atoms()]


def center_of_mass_nm(topology, pos_nm, atom_indices):
    masses = []
    coords = []
    atoms = list(topology.atoms())
    for idx in atom_indices:
        masses.append(atoms[idx].element.mass.value_in_unit(unit.dalton))
        coords.append(pos_nm[idx])
    masses = np.array(masses, dtype=float)
    coords = np.array(coords, dtype=float)
    return (coords * masses[:, None]).sum(axis=0) / masses.sum()


def nearest_atom_distance_nm(pos_nm, idx_a, idx_b):
    pa = np.asarray(pos_nm[idx_a, :], dtype=float)
    pb = np.asarray(pos_nm[idx_b, :], dtype=float)
    d = pa[:, None, :] - pb[None, :, :]
    return float(np.sqrt(np.min(np.sum(d * d, axis=2))))


def map_targets_to_shift_by_minatom(pos0_nm, idx_a, idx_b, u, target_rs_nm, dr_min_nm=-2.0, dr_max_nm=2.0, ngrid=20001):
    pa = np.asarray(pos0_nm[idx_a, :], dtype=float)
    pb = np.asarray(pos0_nm[idx_b, :], dtype=float)
    v0 = pb[None, :, :] - pa[:, None, :]
    dot = np.tensordot(v0, u, axes=([2], [0]))  # (na, nb)
    c = np.sum(v0 * v0, axis=2)  # (na, nb)

    dr_grid = np.linspace(dr_min_nm, dr_max_nm, ngrid)
    # dist2(dr) = dr^2 + 2*dot*dr + c
    dist2 = dr_grid[:, None, None] * dr_grid[:, None, None] + 2.0 * dr_grid[:, None, None] * dot[None, :, :] + c[None, :, :]
    min_dist = np.sqrt(np.maximum(np.min(dist2, axis=(1, 2)), 0.0))

    mn = float(min_dist.min())
    mx = float(min_dist.max())
    dr_out = []
    actual_out = []
    for t in target_rs_nm:
        tt = float(t)
        if tt < mn or tt > mx:
            raise ValueError(f"Target min-atom distance {tt:.4f} nm is outside reachable range [{mn:.4f}, {mx:.4f}] nm")
        i = int(np.argmin(np.abs(min_dist - tt)))
        dr_out.append(float(dr_grid[i]))
        actual_out.append(float(min_dist[i]))
    return np.array(dr_out, dtype=float), np.array(actual_out, dtype=float)


def _bonded_shells(topology, max_depth=5):
    atoms = list(topology.atoms())
    n = len(atoms)
    adj = [set() for _ in range(n)]
    for bond in topology.bonds():
        i = bond[0].index
        j = bond[1].index
        adj[i].add(j)
        adj[j].add(i)

    shells_all = []
    for i in range(n):
        visited = {i}
        frontier = {i}
        shells = {d: set() for d in range(1, max_depth + 1)}
        for d in range(1, max_depth + 1):
            nxt = set()
            for u in frontier:
                nxt |= (adj[u] - visited)
            shells[d] = nxt
            visited |= nxt
            frontier = nxt
        shells_all.append(shells)
    return shells_all


def apply_dmff_like_intra_exclusions(system, topology, m_scales=None, p_scales=None, d_scales=None):
    if mpidplugin is None:
        return False

    mpid_force = None
    for i in range(system.getNumForces()):
        f = system.getForce(i)
        if mpidplugin.MPIDForce.isinstance(f):
            mpid_force = mpidplugin.MPIDForce.cast(f)
            break
    if mpid_force is None:
        return False

    m_scales = list(DEFAULT_M_SCALES if m_scales is None else m_scales)
    p_scales = list(DEFAULT_P_SCALES if p_scales is None else p_scales)
    d_scales = list(DEFAULT_D_SCALES if d_scales is None else d_scales)
    if not (len(m_scales) == len(p_scales) == len(d_scales) == 5):
        raise ValueError("m_scales/p_scales/d_scales must each have 5 values")

    shells_all = _bonded_shells(topology, max_depth=5)
    residue_atoms = [[atom.index for atom in residue.atoms()] for residue in topology.residues()]
    atom_to_residue = {a: r for r, atoms in enumerate(residue_atoms) for a in atoms}
    covalent15 = getattr(mpid_force, "Covalent15", 3)

    for atom_index in range(mpid_force.getNumMultipoles()):
        shells = shells_all[atom_index]
        residue_index = atom_to_residue[atom_index]
        full_intra = tuple(sorted([a for a in residue_atoms[residue_index] if a != atom_index]))

        if float(m_scales[4]) == 0.0:
            c12, c13, c14, c15 = full_intra, tuple(), tuple(), tuple()
        else:
            c12 = tuple(sorted(shells[1])) if float(m_scales[0]) == 0.0 else tuple()
            c13 = tuple(sorted(shells[2])) if float(m_scales[1]) == 0.0 else tuple()
            c14 = tuple(sorted(shells[3])) if float(m_scales[2]) == 0.0 else tuple()
            c15 = tuple(sorted(shells[4])) if float(m_scales[3]) == 0.0 else tuple()
        mpid_force.setCovalentMap(atom_index, mpid_force.Covalent12, c12)
        mpid_force.setCovalentMap(atom_index, mpid_force.Covalent13, c13)
        mpid_force.setCovalentMap(atom_index, mpid_force.Covalent14, c14)
        mpid_force.setCovalentMap(atom_index, covalent15, c15)

        if float(p_scales[4]) == 0.0:
            p12, p13, p14 = full_intra, tuple(), tuple()
        else:
            p12 = tuple(sorted(shells[1])) if float(p_scales[0]) == 0.0 else tuple()
            p13 = tuple(sorted(shells[2])) if float(p_scales[1]) == 0.0 else tuple()
            p14 = tuple(sorted(shells[3])) if float(p_scales[2]) == 0.0 else tuple()
        mpid_force.setCovalentMap(atom_index, mpid_force.PolarizationCovalent11, (atom_index,))
        mpid_force.setCovalentMap(atom_index, mpid_force.PolarizationCovalent12, p12)
        mpid_force.setCovalentMap(atom_index, mpid_force.PolarizationCovalent13, p13)
        mpid_force.setCovalentMap(atom_index, mpid_force.PolarizationCovalent14, p14)

    return True


def extract_single_residue_model(pdb, residue_index):
    model = app.Modeller(pdb.topology, pdb.positions)
    to_delete = [atom for atom in model.topology.atoms() if atom.residue.index != residue_index]
    model.delete(to_delete)
    return model


class OpenMMEnergyModel:
    def __init__(self, topology, positions, xml_path, platform, mpid_intra_exclusion):
        self.ff = app.ForceField(str(xml_path))
        self.system = self.ff.createSystem(
            topology,
            nonbondedMethod=app.NoCutoff,
            constraints=None,
            removeCMMotion=False,
        )
        if mpid_intra_exclusion == "scales":
            apply_dmff_like_intra_exclusions(
                self.system,
                topology,
                DEFAULT_M_SCALES,
                DEFAULT_P_SCALES,
                DEFAULT_D_SCALES,
            )

        for i in range(self.system.getNumForces()):
            self.system.getForce(i).setForceGroup(i)
        self.context = mm.Context(
            self.system,
            mm.VerletIntegrator(0.001),
            mm.Platform.getPlatformByName(platform),
        )
        self.context.setPositions(positions)

    def eval(self, positions):
        self.context.setPositions(positions)
        total = self.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        terms = {}
        for i in range(self.system.getNumForces()):
            f = self.system.getForce(i)
            name = f.getName() if hasattr(f, "getName") else type(f).__name__
            e = self.context.getState(getEnergy=True, groups=1 << i).getPotentialEnergy().value_in_unit(
                unit.kilojoule_per_mole
            )
            terms[name] = terms.get(name, 0.0) + float(e)
        return float(total), terms


class DMFFEnergyModel:
    def __init__(self, topology, positions, xml_path, cutoff_nm):
        if Hamiltonian is None:
            raise RuntimeError("DMFF/JAX is not available")
        self.h = Hamiltonian(str(xml_path))
        self.pot = self.h.createPotential(topology, nonbondedMethod=app.NoCutoff)
        self.params = self.h.getParameters()
        self.cutoff_nm = float(cutoff_nm)
        self.box = np.diag([10.0, 10.0, 10.0])
        self.term_names = list(self.pot.dmff_potentials.keys())
        self.term_funcs = {name: self.pot.getPotentialFunc([name]) for name in self.term_names}
        self.positions = positions

    def eval(self, pos_nm):
        pos_jnp = jnp.array(pos_nm)
        nb = nblist.NeighborList(self.box, self.cutoff_nm, self.pot.meta["cov_map"])
        nb.allocate(pos_jnp)
        pairs = nb.pairs
        terms = {}
        for name, func in self.term_funcs.items():
            terms[name] = float(func(pos_jnp, self.box, pairs, self.params))
        total = float(sum(terms.values()))
        return total, terms


def interaction_from_terms(dimer_total, dimer_terms, a_total, a_terms, b_total, b_terms):
    inter_terms = {}
    keys = set(dimer_terms.keys()) | set(a_terms.keys()) | set(b_terms.keys())
    for k in keys:
        inter_terms[k] = float(dimer_terms.get(k, 0.0) - a_terms.get(k, 0.0) - b_terms.get(k, 0.0))
    inter_total = float(dimer_total - a_total - b_total)
    return inter_total, inter_terms


def main():
    parser = argparse.ArgumentParser(description="Dimer intermolecular scan: OpenMM vs DMFF")
    parser.add_argument("--pdb", required=True)
    parser.add_argument("--openmm-xml", required=True)
    parser.add_argument("--dmff-xml", required=True)
    parser.add_argument("--residue-a", type=int, default=0)
    parser.add_argument("--residue-b", type=int, default=1)
    parser.add_argument("--r-min", type=float, default=2.5)
    parser.add_argument("--r-max", type=float, default=8.0)
    parser.add_argument("--r-step", type=float, default=0.1)
    parser.add_argument("--distance-unit", choices=["nm", "angstrom"], default="angstrom")
    parser.add_argument("--distance-metric", choices=["com", "minatom"], default="com")
    parser.add_argument("--platform", choices=["Reference", "CPU", "OpenCL"], default="CPU")
    parser.add_argument("--cutoff-nm", type=float, default=1.5)
    parser.add_argument("--mpid-intra-exclusion", choices=["none", "scales"], default="scales")
    parser.add_argument("--output", default="dimer_scan_intermolecular_compare.csv")
    parser.add_argument("--plot", default=None)
    args = parser.parse_args()

    pdb = app.PDBFile(str(args.pdb))
    pos0_nm = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)

    # Build dimer and monomer models (OpenMM).
    omm_dimer = OpenMMEnergyModel(pdb.topology, pdb.positions, args.openmm_xml, args.platform, args.mpid_intra_exclusion)
    mono_a = extract_single_residue_model(pdb, args.residue_a)
    mono_b = extract_single_residue_model(pdb, args.residue_b)
    omm_a = OpenMMEnergyModel(mono_a.topology, mono_a.positions, args.openmm_xml, args.platform, args.mpid_intra_exclusion)
    omm_b = OpenMMEnergyModel(mono_b.topology, mono_b.positions, args.openmm_xml, args.platform, args.mpid_intra_exclusion)

    omm_a_total, omm_a_terms = omm_a.eval(mono_a.positions)
    omm_b_total, omm_b_terms = omm_b.eval(mono_b.positions)

    # Build dimer and monomer models (DMFF).
    dmff_dimer = DMFFEnergyModel(pdb.topology, pdb.positions, args.dmff_xml, args.cutoff_nm)
    dmff_a = DMFFEnergyModel(mono_a.topology, mono_a.positions, args.dmff_xml, args.cutoff_nm)
    dmff_b = DMFFEnergyModel(mono_b.topology, mono_b.positions, args.dmff_xml, args.cutoff_nm)

    dmff_a_total, dmff_a_terms = dmff_a.eval(np.array(mono_a.positions.value_in_unit(unit.nanometer), dtype=float))
    dmff_b_total, dmff_b_terms = dmff_b.eval(np.array(mono_b.positions.value_in_unit(unit.nanometer), dtype=float))

    idx_a = residue_atom_indices(pdb.topology, args.residue_a)
    idx_b = residue_atom_indices(pdb.topology, args.residue_b)
    com_a = center_of_mass_nm(pdb.topology, pos0_nm, idx_a)
    com_b = center_of_mass_nm(pdb.topology, pos0_nm, idx_b)
    u = com_b - com_a
    r0 = float(np.linalg.norm(u))
    if r0 < 1e-8:
        raise ValueError("Initial COM distance is zero")
    u = u / r0

    rs_input = np.arange(args.r_min, args.r_max + 0.5 * args.r_step, args.r_step)
    to_nm = 0.1 if args.distance_unit == "angstrom" else 1.0
    rs_nm = rs_input * to_nm
    if args.distance_metric == "minatom":
        drs_nm, actual_rs_nm = map_targets_to_shift_by_minatom(pos0_nm, idx_a, idx_b, u, rs_nm)
    else:
        drs_nm = rs_nm - r0
        actual_rs_nm = rs_nm

    rows = []
    for r_in_target, r_nm_target, dr in zip(rs_input, rs_nm, drs_nm):
        pos_nm = np.array(pos0_nm, copy=True)
        pos_nm[idx_b, :] = pos_nm[idx_b, :] + dr * u[None, :]
        if args.distance_metric == "minatom":
            r_nm_actual = nearest_atom_distance_nm(pos_nm, idx_a, idx_b)
        else:
            r_nm_actual = float(np.linalg.norm(center_of_mass_nm(pdb.topology, pos_nm, idx_b) - center_of_mass_nm(pdb.topology, pos_nm, idx_a)))
        r_in_actual = r_nm_actual / to_nm

        omm_d_total, omm_d_terms = omm_dimer.eval(pos_nm * unit.nanometer)
        omm_inter_total, omm_inter_terms = interaction_from_terms(
            omm_d_total, omm_d_terms, omm_a_total, omm_a_terms, omm_b_total, omm_b_terms
        )

        dmff_d_total, dmff_d_terms = dmff_dimer.eval(pos_nm)
        dmff_inter_total, dmff_inter_terms = interaction_from_terms(
            dmff_d_total, dmff_d_terms, dmff_a_total, dmff_a_terms, dmff_b_total, dmff_b_terms
        )

        row = {
            f"r_{args.distance_unit}": float(r_in_actual),
            "r_nm": float(r_nm_actual),
            f"r_target_{args.distance_unit}": float(r_in_target),
            "r_target_nm": float(r_nm_target),
            "openmm_inter_total_kjmol": float(omm_inter_total),
            "openmm_inter_MPIDForce_kjmol": float(omm_inter_terms.get("MPIDForce", 0.0)),
            "openmm_inter_CustomNonbondedForce_kjmol": float(omm_inter_terms.get("CustomNonbondedForce", 0.0)),
            "dmff_inter_total_kjmol": float(dmff_inter_total),
            "dmff_inter_ADMPPmeForce_kjmol": float(dmff_inter_terms.get("ADMPPmeForce", 0.0)),
            "dmff_inter_ADMPDispPmeForce_kjmol": float(dmff_inter_terms.get("ADMPDispPmeForce", 0.0)),
        }
        rows.append(row)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        f"r_{args.distance_unit}",
        "r_nm",
        f"r_target_{args.distance_unit}",
        "r_target_nm",
        "openmm_inter_total_kjmol",
        "openmm_inter_MPIDForce_kjmol",
        "openmm_inter_CustomNonbondedForce_kjmol",
        "dmff_inter_total_kjmol",
        "dmff_inter_ADMPPmeForce_kjmol",
        "dmff_inter_ADMPDispPmeForce_kjmol",
    ]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    plot_path = Path(args.plot) if args.plot else out.with_suffix(".png")
    x = np.array([r[f"r_{args.distance_unit}"] for r in rows], dtype=float)
    plt.figure(figsize=(7.2, 5.0), dpi=150)
    plt.plot(x, np.array([r["openmm_inter_total_kjmol"] for r in rows], dtype=float), label="OpenMM inter total", linewidth=2.0)
    plt.plot(x, np.array([r["openmm_inter_MPIDForce_kjmol"] for r in rows], dtype=float), label="OpenMM MPID inter", linewidth=1.7)
    plt.plot(
        x,
        np.array([r["openmm_inter_CustomNonbondedForce_kjmol"] for r in rows], dtype=float),
        label="OpenMM CNB inter",
        linewidth=1.7,
    )
    plt.plot(x, np.array([r["dmff_inter_total_kjmol"] for r in rows], dtype=float), label="DMFF inter total", linewidth=2.0)
    plt.xlabel(f"Distance ({'A' if args.distance_unit == 'angstrom' else 'nm'})")
    plt.ylabel("Intermolecular Energy (kJ/mol)")
    plt.title("Dimer Intermolecular Scan: OpenMM vs DMFF")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    print(f"Wrote {len(rows)} scan points to {out}")
    print(f"Wrote plot to {plot_path}")


if __name__ == "__main__":
    main()
