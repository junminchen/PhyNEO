#!/usr/bin/env python3
"""Backward-compatible wrapper for merging converted XML into base XML."""

import argparse

from convert_dmff_to_openmm import merge_converted_into_base


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge converted MPID XML into a base OpenMM XML")
    parser.add_argument("--a", required=True, help="Base OpenMM XML (e.g., merged_opls.xml)")
    parser.add_argument("--b", required=True, help="Converted XML (e.g., phyneo_ecl_z.xml)")
    parser.add_argument("--out", required=True, help="Output merged XML")
    parser.add_argument("--strict", action="store_true", help="Fail on ambiguous mapping")
    parser.add_argument("--zero-nonbonded-charges", action="store_true", help="Set NonbondedForce atom charges to 0")
    args = parser.parse_args()

    stats = merge_converted_into_base(
        base_xml=args.a,
        converted_xml=args.b,
        output_xml=args.out,
        strict=args.strict,
        zero_charges=args.zero_nonbonded_charges,
    )
    print(f"[ok] merged: {args.a} + {args.b} -> {args.out}")
    print(f"[stats] {stats}")
