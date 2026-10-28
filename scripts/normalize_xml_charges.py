#!/usr/bin/env python3
"""Constrain the maintained DMFF XMLs and PE reference force fields to formal charge.

Only monopoles and their electrostatic damping copies are changed. The projection
minimizes the sum of squared changes over atoms in each distinct primitive
residue composition, with equivalent atom types tied. Redundant DME/TER templates
therefore do not change the result between copies of the PEO model. Both mixed and hydroxyl-ended PEG
chains are constrained, including the two-TEO-end topology in peg_cmd/init.pdb.
"""

import argparse
from collections import Counter
from decimal import Decimal, localcontext
from difflib import SequenceMatcher
from fractions import Fraction
import json
from math import gcd
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
ION_CHARGES = {"BF4": -1, "BOB": -1, "DFO": -1, "DFP": -1,
               "FSI": -1, "PF6": -1, "TFS": -1, "LiA": 1, "NaA": 1}


def constraints(root):
    residues = {r.get("name"): Counter(a.get("type") for a in r.findall("Atom"))
                for r in root.findall("Residues/Residue")}
    rows = []
    for name, counts in residues.items():
        if "TEO" in residues and name in ("TER", "TEO"):
            continue
        rows.append((name, counts, ION_CHARGES.get(name, 0)))
    if "TEO" in residues:
        rows.append(("TER+TEO", residues["TER"] + residues["TEO"], 0))
        rows.append(("2*TEO", Counter({t: 2*n for t, n in residues["TEO"].items()}), 0))
    weights, seen = Counter(), set()
    for counts in residues.values():
        divisor = 0
        for n in counts.values():
            divisor = gcd(divisor, n)
        primitive = tuple(sorted((t, n//divisor) for t, n in counts.items()))
        if primitive not in seen:
            weights.update(dict(primitive))
            seen.add(primitive)
    return rows, weights


def independent(rows, types):
    """Keep independent original constraints, rejecting conflicting duplicates."""
    basis, kept = [], []
    for name, counts, target in rows:
        row = [Fraction(counts[t]) for t in types] + [Fraction(target)]
        for pivot, reduced in basis:
            scale = row[pivot]
            row = [a - scale*b for a, b in zip(row, reduced)]
        pivot = next((i for i, x in enumerate(row[:-1]) if x), None)
        if pivot is None:
            if row[-1]:
                raise ValueError("Conflicting charge constraints: " + name)
            continue
        scale = row[pivot]
        basis.append((pivot, [x/scale for x in row]))
        kept.append((name, counts, target))
    return kept


def solve(matrix, rhs):
    augmented = [list(row) + [value] for row, value in zip(matrix, rhs)]
    for col in range(len(rhs)):
        pivot = next(i for i in range(col, len(rhs)) if augmented[i][col])
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        scale = augmented[col][col]
        augmented[col] = [x/scale for x in augmented[col]]
        for i in range(len(rhs)):
            if i != col:
                scale = augmented[i][col]
                augmented[i] = [a-scale*b for a, b in zip(augmented[i], augmented[col])]
    return [row[-1] for row in augmented]


def project(charges, rows, weights):
    types = list(weights)
    if set(types) - charges.keys():
        raise ValueError("Residue has atom types without charge parameters")
    active = independent(rows, types)
    a = [[Fraction(counts[t]) for t in types] for _, counts, _ in active]
    q = [charges[t] for t in types]
    residual = [sum(c*v for c, v in zip(row, q))-target
                for row, (_, _, target) in zip(a, active)]
    gram = [[sum(x*y/weights[t] for x, y, t in zip(left, right, types))
             for right in a] for left in a]
    multipliers = solve(gram, residual)
    result = dict(charges)
    for i, t in enumerate(types):
        result[t] = q[i] - sum(row[i]*v for row, v in zip(a, multipliers))/weights[t]
    for name, counts, target in rows:
        if sum(result[t]*n for t, n in counts.items()) != target:
            raise ValueError("Projection failed: " + name)
    return result


def decimal_string(value):
    with localcontext() as ctx:
        ctx.prec = 60
        number = Decimal(value.numerator)/Decimal(value.denominator)
        return format(number.quantize(Decimal("1e-20")), "f").rstrip("0").rstrip(".") or "0"


def replace_attributes(text, section, attribute, values):
    pattern = r"(<" + section + r"\b[^>]*>)(.*?)(</" + section + r">)"

    def replace_section(match):
        def replace_atom(atom_match):
            raw = atom_match.group()
            atom = ET.fromstring(raw)
            key = atom.get("type")
            old = atom.get(attribute)
            if old is None or key not in values or Fraction(old) == Fraction(values[key]):
                return raw
            return re.sub(r'(\b' + attribute + r'=\")[^\"]*(\")',
                          lambda m: m[1] + values[key] + m[2], raw)
        body = re.sub(r"<Atom\b[^>]*?/>", replace_atom, match[2])
        return match[1] + body + match[3]
    return re.sub(pattern, replace_section, text, flags=re.DOTALL)


def normalize(path):
    original = path.read_text()
    root = ET.fromstring(original)
    section = "ADMPPmeForce" if root.find("ADMPPmeForce") is not None else "NonbondedForce"
    attribute = "c0" if section == "ADMPPmeForce" else "charge"
    charges = {}
    for atom in root.findall(section + "/Atom"):
        key, value = atom.get("type"), Fraction(atom.get(attribute))
        if key in charges and charges[key] != value:
            raise ValueError("Conflicting local-frame monopoles for type " + key)
        charges[key] = value
    rows, weights = constraints(root)
    result = project(charges, rows, weights)
    values = {t: decimal_string(q) for t, q in result.items()}
    updated = replace_attributes(original, section, attribute, values)
    if section == "ADMPPmeForce":
        for damping in ("SlaterSrEsForce", "QqTtDampingForce"):
            updated = replace_attributes(updated, damping, "Q", values)
    # Retain formatting on untouched lines; remove inherited trailing spaces
    # only on lines whose charge values changed.
    old_lines, new_lines = original.splitlines(keepends=True), updated.splitlines(keepends=True)
    for tag, _, _, start, end in SequenceMatcher(None, old_lines, new_lines, autojunk=False).get_opcodes():
        if tag != "equal":
            for i in range(start, end):
                line = new_lines[i]
                new_lines[i] = line.rstrip(" \t\r\n") + ("\n" if line.endswith("\n") else "")
    updated = "".join(new_lines)
    # Validate the serialized values, not just the exact rational solution.
    for name, counts, target in rows:
        error = sum(Decimal(values[t])*n for t, n in counts.items()) - target
        if abs(error) > Decimal("1e-18"):
            raise ValueError("Serialized charge constraint failed: " + name)
    return updated, {"path": str(path.relative_to(ROOT)), "changed": updated != original,
                     "max_site_change_e": float(max(abs(result[t]-charges[t]) for t in charges)),
                     "constraints": [{"name": name, "before_e": float(sum(charges[t]*n for t, n in c.items())),
                                      "after_e": str(sum(Decimal(values[t])*n for t, n in c.items())), "target_e": target}
                                     for name, c, target in rows]}


def maintained_files():
    paths = subprocess.check_output(["git", "ls-files", "*.xml"], cwd=ROOT, text=True).splitlines()
    for name in paths:
        path = ROOT/name
        if path.is_symlink():
            continue
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            continue  # Historical MPID conversion examples are outside this scope.
        if root.find("ADMPPmeForce/Atom") is not None and root.find("Residues/Residue") is not None:
            yield path
        elif name.startswith("workflow/GNN_bonding/abinitio_intra") and path.name == "pe.xml":
            yield path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Apply corrections; default is a read-only check")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    reports = []
    for path in maintained_files():
        updated, report = normalize(path)
        if args.write and report["changed"]:
            path.write_text(updated)
        reports.append(report)
    if args.report:
        args.report.write_text(json.dumps(reports, indent=2) + "\n")
    changed = sum(r["changed"] for r in reports)
    print(f"{len(reports)} force fields checked; {changed} {'updated' if args.write else 'need normalization'}")
    return 0 if args.write or not changed else 1


if __name__ == "__main__":
    raise SystemExit(main())
