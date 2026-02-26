#!/usr/bin/env python3
"""Backward-compatible wrapper for DMFF -> converted OpenMM XML."""

import argparse

from convert_dmff_to_openmm import convert_dmff_xml


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DMFF XML into MPID+CustomNonbonded XML")
    parser.add_argument("--input", default="phyneo_ecl.xml", help="Input DMFF XML")
    parser.add_argument("--output", default="phyneo_ecl_z.xml", help="Output converted XML")
    parser.add_argument("--no-strict", action="store_true", help="Allow missing parameters when possible")
    args = parser.parse_args()

    convert_dmff_xml(args.input, args.output, strict=not args.no_strict)
    print(f"[ok] converted: {args.input} -> {args.output}")
