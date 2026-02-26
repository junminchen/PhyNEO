#!/usr/bin/env python3
"""Backward-compatible wrapper. Prefer using run_openmm.py."""

import argparse
import subprocess
import sys


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compatibility wrapper for run_openmm.py")
    parser.add_argument("xml", nargs="?", help="OpenMM forcefield XML")
    parser.add_argument("--pdb", default="EC.pdb")
    parser.add_argument("--method", choices=["nocutoff", "pme"], default="nocutoff")
    parser.add_argument("--cutoff-angstrom", type=float, default=8.0)
    parser.add_argument("--platform", default="Reference")
    args = parser.parse_args()

    if not args.xml:
        print("usage: run_mpid.py <xml> [--pdb EC.pdb]", file=sys.stderr)
        sys.exit(2)

    cmd = [
        sys.executable,
        "run_openmm.py",
        "--xml",
        args.xml,
        "--pdb",
        args.pdb,
        "--method",
        args.method,
        "--cutoff-angstrom",
        str(args.cutoff_angstrom),
        "--platform",
        args.platform,
    ]
    raise SystemExit(subprocess.call(cmd))
