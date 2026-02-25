#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

COMP_TO_PDB = {
    "Li": "Li.pdb",
    "PF6": "PF6.pdb",
    "EC": "EC.pdb",
    "DMC": "DMC.pdb",
}


def main() -> None:
    here = Path(__file__).resolve().parent
    cfg = json.loads((here / "config.json").read_text())
    comp = cfg["composition"]
    cell = cfg["cell"]
    tol = float(cfg["packmol"]["tolerance_angstrom"])

    lines = [
        "# Electrolyte-only packing in liquid slab",
        f"tolerance {tol}",
        "filetype pdb",
        "output electrolyte_start.pdb",
        "",
    ]

    x0, y0, z0 = 0.0, 0.0, float(cell["z_liq_min_angstrom"])
    x1, y1, z1 = float(cell["box_x_angstrom"]), float(cell["box_y_angstrom"]), float(cell["z_liq_max_angstrom"])

    for key in ("Li", "PF6", "EC", "DMC"):
        n = int(comp.get(key, 0))
        if n <= 0:
            continue
        lines.extend([
            f"structure ../pdb_bank/{COMP_TO_PDB[key]}",
            f"  number {n}",
            f"  inside box {x0:.3f} {y0:.3f} {z0:.3f} {x1:.3f} {y1:.3f} {z1:.3f}",
            "end structure",
            "",
        ])

    out = here / "packmol.inp"
    out.write_text("\n".join(lines))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
