#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from typing import List, Tuple

import openmm.app as app
import openmm.unit as unit
from openmm import Vec3


def build_electrodes(nx: int, ny: int, spacing_nm: float, z_cath_nm: float, z_anod_nm: float, lx_nm: float, ly_nm: float):
    top = app.Topology()
    chain_c = top.addChain('A')
    chain_a = top.addChain('B')

    positions = []
    x0 = 0.5 * (lx_nm - (nx - 1) * spacing_nm)
    y0 = 0.5 * (ly_nm - (ny - 1) * spacing_nm)

    for ix in range(nx):
        for iy in range(ny):
            x = x0 + ix * spacing_nm
            y = y0 + iy * spacing_nm
            rc = top.addResidue('CAT', chain_c)
            top.addAtom('CA', app.element.carbon, rc)
            positions.append(Vec3(x, y, z_cath_nm))

    for ix in range(nx):
        for iy in range(ny):
            x = x0 + ix * spacing_nm
            y = y0 + iy * spacing_nm
            ra = top.addResidue('ANO', chain_a)
            top.addAtom('AN', app.element.nitrogen, ra)
            positions.append(Vec3(x, y, z_anod_nm))

    return top, positions


def residue_com_z_nm(modeller: app.Modeller, residue: app.topology.Residue) -> float:
    pos = modeller.positions
    indices = [a.index for a in residue.atoms()]
    return sum(pos[i].value_in_unit(unit.nanometer)[2] for i in indices) / float(len(indices))


def apply_liquid_slab_and_neutrality(
    modeller: app.Modeller,
    z_liq_min_nm: float,
    z_liq_max_nm: float,
) -> Tuple[int, int, int]:
    """
    Keep electrode residues always; keep HOH/NA/CL only if COM-z inside [z_liq_min_nm, z_liq_max_nm].
    Then enforce Na/Cl count balance by deleting extra ions from the majority species.
    """
    to_delete = []
    na_res: List[app.topology.Residue] = []
    cl_res: List[app.topology.Residue] = []

    for res in modeller.topology.residues():
        name = res.name.strip()
        if name in ("CAT", "ANO"):
            continue
        if name not in ("HOH", "NA", "CL"):
            to_delete.extend(list(res.atoms()))
            continue
        zc = residue_com_z_nm(modeller, res)
        if zc < z_liq_min_nm or zc > z_liq_max_nm:
            to_delete.extend(list(res.atoms()))
        else:
            if name == "NA":
                na_res.append(res)
            elif name == "CL":
                cl_res.append(res)

    if to_delete:
        modeller.delete(to_delete)

    # Re-collect ions after first delete (indices changed).
    na_res = [r for r in modeller.topology.residues() if r.name.strip() == "NA"]
    cl_res = [r for r in modeller.topology.residues() if r.name.strip() == "CL"]
    d = len(na_res) - len(cl_res)
    if d > 0:
        # Too many Na: remove extra Na furthest from slab center first.
        zc = 0.5 * (z_liq_min_nm + z_liq_max_nm)
        ranked = sorted(na_res, key=lambda r: abs(residue_com_z_nm(modeller, r) - zc), reverse=True)
        extra_atoms = []
        for r in ranked[:d]:
            extra_atoms.extend(list(r.atoms()))
        modeller.delete(extra_atoms)
    elif d < 0:
        # Too many Cl: remove extra Cl furthest from slab center first.
        zc = 0.5 * (z_liq_min_nm + z_liq_max_nm)
        ranked = sorted(cl_res, key=lambda r: abs(residue_com_z_nm(modeller, r) - zc), reverse=True)
        extra_atoms = []
        for r in ranked[: (-d)]:
            extra_atoms.extend(list(r.atoms()))
        modeller.delete(extra_atoms)

    n_water = sum(1 for r in modeller.topology.residues() if r.name.strip() == "HOH")
    n_na = sum(1 for r in modeller.topology.residues() if r.name.strip() == "NA")
    n_cl = sum(1 for r in modeller.topology.residues() if r.name.strip() == "CL")
    return n_water, n_na, n_cl


def main():
    parser = argparse.ArgumentParser(description='Build graphene-like electrode + NaCl(aq) box with TIP3P water.')
    parser.add_argument('--output', default='nacl_water_start.pdb')
    parser.add_argument('--nx', type=int, default=0, help='If <=0 and --fill-xy is set, auto-compute from box and spacing.')
    parser.add_argument('--ny', type=int, default=0, help='If <=0 and --fill-xy is set, auto-compute from box and spacing.')
    parser.add_argument('--spacing-nm', type=float, default=0.45)
    parser.add_argument('--fill-xy', action='store_true', default=True, help='Auto-fill electrode lattice across x/y.')
    parser.add_argument('--no-fill-xy', action='store_false', dest='fill_xy', help='Disable auto-fill and use explicit nx/ny.')
    parser.add_argument('--edge-margin-nm', type=float, default=0.15, help='Edge margin for auto-filled electrode lattice.')
    parser.add_argument('--box-x-nm', type=float, default=3.0)
    parser.add_argument('--box-y-nm', type=float, default=3.0)
    parser.add_argument('--box-z-nm', type=float, default=6.0)
    parser.add_argument('--z-cathode-nm', type=float, default=1.0)
    parser.add_argument('--z-anode-nm', type=float, default=5.0)
    parser.add_argument('--z-liq-min-nm', type=float, default=1.2, help='Lower bound of liquid slab (nm).')
    parser.add_argument('--z-liq-max-nm', type=float, default=4.8, help='Upper bound of liquid slab (nm).')
    parser.add_argument('--ionic-strength-m', type=float, default=1.0)
    args = parser.parse_args()

    if not (0.0 < args.z_cathode_nm < args.z_anode_nm < args.box_z_nm):
        raise ValueError('Need 0 < z_cathode < z_anode < box_z')
    if not (0.0 < args.z_liq_min_nm < args.z_liq_max_nm < args.box_z_nm):
        raise ValueError('Need 0 < z_liq_min < z_liq_max < box_z')
    if not (args.z_cathode_nm < args.z_liq_min_nm < args.z_liq_max_nm < args.z_anode_nm):
        raise ValueError('Need z_cathode < z_liq_min < z_liq_max < z_anode')

    nx = int(args.nx)
    ny = int(args.ny)
    if args.fill_xy:
        usable_x = max(args.box_x_nm - 2.0 * args.edge_margin_nm, args.spacing_nm)
        usable_y = max(args.box_y_nm - 2.0 * args.edge_margin_nm, args.spacing_nm)
        if nx <= 0:
            nx = int(usable_x / args.spacing_nm) + 1
        if ny <= 0:
            ny = int(usable_y / args.spacing_nm) + 1
    if nx < 1 or ny < 1:
        raise ValueError('nx and ny must be >= 1')

    top, pos = build_electrodes(
        nx,
        ny,
        args.spacing_nm,
        args.z_cathode_nm,
        args.z_anode_nm,
        args.box_x_nm,
        args.box_y_nm,
    )

    modeller = app.Modeller(top, unit.Quantity(pos, unit.nanometer))
    ff = app.ForceField('amber14/tip3p.xml', 'electrode_residues.xml', 'electrode_ff.xml')
    modeller.addSolvent(
        ff,
        model='tip3p',
        boxSize=Vec3(args.box_x_nm, args.box_y_nm, args.box_z_nm) * unit.nanometer,
        ionicStrength=args.ionic_strength_m * unit.molar,
        neutralize=False,
    )
    n_water, n_na, n_cl = apply_liquid_slab_and_neutrality(modeller, args.z_liq_min_nm, args.z_liq_max_nm)

    with open(args.output, 'w') as f:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, f)

    counts = Counter(res.name for res in modeller.topology.residues())
    print(f'wrote {args.output}')
    print(f'electrode lattice: nx={nx} ny={ny} spacing={args.spacing_nm:.3f} nm')
    print('residue counts:', dict(counts))
    print(f'liquid slab nm: [{args.z_liq_min_nm:.3f}, {args.z_liq_max_nm:.3f}]')
    print(f'kept solvent/ions: HOH={n_water} NA={n_na} CL={n_cl}')


if __name__ == '__main__':
    main()
