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
    box = cfg["box_angstrom"]
    tol = cfg.get("packmol_tolerance_angstrom", 2.0)

    lines = [
        "# Auto-generated from config.json",
        f"tolerance {tol}",
        "filetype pdb",
        "output start.pdb",
        "",
    ]
    for name in ("Li", "PF6", "EC", "DMC"):
        n = int(comp.get(name, 0))
        if n <= 0:
            continue
        pdb = COMP_TO_PDB[name]
        lines.extend(
            [
                f"structure ../pdb_bank/{pdb}",
                f"  number {n}",
                f"  inside box 0.0 0.0 0.0 {box[0]} {box[1]} {box[2]}",
                "end structure",
                "",
            ]
        )

    out = here / "packmol.inp"
    out.write_text("\n".join(lines))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
