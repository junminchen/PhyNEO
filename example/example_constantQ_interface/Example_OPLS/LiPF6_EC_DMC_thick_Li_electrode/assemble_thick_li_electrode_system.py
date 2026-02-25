#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import openmm.app as app
import openmm.unit as unit
from openmm import Vec3


def make_multilayer_li_electrode_topology(
    box_x: float,
    box_y: float,
    zc_surface: float,
    za_surface: float,
    spacing: float,
    margin: float,
    nlayer: int,
    dl: float,
):
    nx = int((box_x - 2.0 * margin) / spacing) + 1
    ny = int((box_y - 2.0 * margin) / spacing) + 1

    top = app.Topology()
    chain_c = top.addChain("A")
    chain_a = top.addChain("B")

    pos = []
    x0 = 0.5 * (box_x - (nx - 1) * spacing)
    y0 = 0.5 * (box_y - (ny - 1) * spacing)

    # Cathode slab: layer 0 is surface facing electrolyte, deeper layers go to smaller z.
    for layer in range(nlayer):
        zc = zc_surface - layer * dl
        for ix in range(nx):
            for iy in range(ny):
                x = x0 + ix * spacing
                y = y0 + iy * spacing
                rc = top.addResidue("CAT", chain_c)
                top.addAtom("LI", app.element.lithium, rc)
                pos.append(Vec3(x, y, zc))

    # Anode slab: layer 0 is surface facing electrolyte, deeper layers go to larger z.
    for layer in range(nlayer):
        za = za_surface + layer * dl
        for ix in range(nx):
            for iy in range(ny):
                x = x0 + ix * spacing
                y = y0 + iy * spacing
                ra = top.addResidue("ANO", chain_a)
                top.addAtom("LI", app.element.lithium, ra)
                pos.append(Vec3(x, y, za))

    return top, unit.Quantity(pos, unit.angstrom), nx, ny


def main() -> None:
    here = Path(__file__).resolve().parent
    cfg = json.loads((here / "config.json").read_text())
    cell = cfg["cell"]
    ele = cfg["electrode"]

    ele_pdb = here / "electrolyte_start.pdb"
    if not ele_pdb.exists():
        raise FileNotFoundError("electrolyte_start.pdb not found. Run run_packmol.sh first.")

    topo_e, pos_e, nx, ny = make_multilayer_li_electrode_topology(
        float(cell["box_x_angstrom"]),
        float(cell["box_y_angstrom"]),
        float(cell["z_cathode_surface_angstrom"]),
        float(cell["z_anode_surface_angstrom"]),
        float(ele["spacing_angstrom"]),
        float(ele["edge_margin_angstrom"]),
        int(ele["layers_per_electrode"]),
        float(ele["interlayer_spacing_angstrom"]),
    )

    modeller = app.Modeller(topo_e, pos_e)

    pdb_liq = app.PDBFile(str(ele_pdb))
    modeller.add(pdb_liq.topology, pdb_liq.positions)

    modeller.topology.setPeriodicBoxVectors(None)

    out = here / "start_with_electrodes.pdb"
    with out.open("w") as f:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, f)

    counts = Counter(r.name for r in modeller.topology.residues())
    nlayer = int(ele["layers_per_electrode"])
    print(f"Wrote {out}")
    print(f"Li slab grid: nx={nx}, ny={ny}, layers={nlayer}, atoms per slab={nx*ny*nlayer}")
    print(f"Residues: {dict(counts)}")


if __name__ == "__main__":
    main()
