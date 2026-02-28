#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import openmm.app as app
import openmm.unit as unit
from openmm import Vec3


def make_single_layer_graphene_electrodes(
    box_x: float,
    box_y: float,
    z_cathode: float,
    z_anode: float,
    spacing: float,
    margin: float,
):
    nx = int((box_x - 2.0 * margin) / spacing) + 1
    ny = int((box_y - 2.0 * margin) / spacing) + 1

    top = app.Topology()
    chain_c = top.addChain("A")
    chain_a = top.addChain("B")

    pos = []
    x0 = 0.5 * (box_x - (nx - 1) * spacing)
    y0 = 0.5 * (box_y - (ny - 1) * spacing)

    for ix in range(nx):
        for iy in range(ny):
            x = x0 + ix * spacing
            y = y0 + iy * spacing
            rc = top.addResidue("CAT", chain_c)
            top.addAtom("CG", app.element.carbon, rc)
            pos.append(Vec3(x, y, z_cathode))

    for ix in range(nx):
        for iy in range(ny):
            x = x0 + ix * spacing
            y = y0 + iy * spacing
            ra = top.addResidue("ANO", chain_a)
            top.addAtom("CG", app.element.carbon, ra)
            pos.append(Vec3(x, y, z_anode))

    return top, unit.Quantity(pos, unit.angstrom), nx, ny


def main() -> None:
    here = Path(__file__).resolve().parent
    cfg = json.loads((here / "config.json").read_text())
    cell = cfg["cell"]
    ele = cfg["electrode"]

    electrolyte_pdb = (here / cfg["input"]["electrolyte_pdb"]).resolve()
    if not electrolyte_pdb.exists():
        raise FileNotFoundError(f"Electrolyte PDB not found: {electrolyte_pdb}")

    topo_e, pos_e, nx, ny = make_single_layer_graphene_electrodes(
        float(cell["box_x_angstrom"]),
        float(cell["box_y_angstrom"]),
        float(cell["z_cathode_angstrom"]),
        float(cell["z_anode_angstrom"]),
        float(ele["spacing_angstrom"]),
        float(ele["edge_margin_angstrom"]),
    )

    modeller = app.Modeller(topo_e, pos_e)

    pdb_liq = app.PDBFile(str(electrolyte_pdb))
    modeller.add(pdb_liq.topology, pdb_liq.positions)
    modeller.topology.setPeriodicBoxVectors(None)

    out = here / "start_fixedcharge_graphene.pdb"
    with out.open("w") as f:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, f)

    counts = Counter(r.name for r in modeller.topology.residues())
    print(f"Wrote {out}")
    print(f"Graphene grid: nx={nx}, ny={ny}, atoms per sheet={nx*ny}")
    print(f"Residues: {dict(counts)}")


if __name__ == "__main__":
    main()
