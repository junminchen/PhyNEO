#!/usr/bin/env python3
"""Convert DMFF XML into OpenMM-readable XML sections and optionally merge into a base FF.

This script provides three workflows:
1) convert:   DMFF XML -> converted XML containing MPIDForce + CustomNonbondedForce
2) merge:     base OpenMM XML + converted XML -> merged XML
3) pipeline:  run convert + merge in one command
"""

from __future__ import annotations

import argparse
import copy
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

TResidueAtom = Tuple[str, str]

CUSTOM_NB_ENERGY = (
    "A*K2*exp(-B*r)+(-138.93542)*exp(-B*r)*(1+B*r)*Q/r"
    " - (1-exp(-x)*(1+x+0.5*x^2+x^3/6+x^4/24+x^5/120+x^6/720)) * C6/(r^6)"
    " - (1-exp(-x)*(1+x+0.5*x^2+x^3/6+x^4/24+x^5/120+x^6/720+x^7/5040+x^8/40320)) * C8/(r^8)"
    " - (1-exp(-x)*(1+x+0.5*x^2+x^3/6+x^4/24+x^5/120+x^6/720+x^7/5040+x^8/40320+x^9/362880+x^10/3628800)) * C10/(r^10);"
    "x=B*r - (2*B^2*r+3*B)/(B^2*r^2+3*B*r+3)*r;"
    "K2=(Br^2)/3 + Br + 1;"
    "Br = B*r;"
    "B=sqrt(Bexp1*Bexp2);"
    "Q=Q1*Q2;"
    "C6=sqrt(C61*C62);"
    "C8=sqrt(C81*C82);"
    "C10=sqrt(C101*C102);"
    "A=Aex-Ael-Ain-Adh-Adi;"
    "Aex=(Aexch1*Aexch2);"
    "Ael=(Aelec1*Aelec2);"
    "Ain=(Aind1*Aind2);"
    "Adh=(Adhf1*Adhf2);"
    "Adi=(Adisp1*Adisp2)"
)

CUSTOM_NB_PARAMS = ["Aexch", "Aelec", "Aind", "Adhf", "Adisp", "Bexp", "Q", "C6", "C8", "C10", "C12"]


@dataclass(frozen=True)
class TypeMaps:
    a_to_b: Dict[str, str]
    b_to_a_unique: Dict[str, str]
    b_to_a_candidates: Dict[str, List[str]]


def parse_xml(path: str) -> ET.ElementTree:
    return ET.parse(path)


def write_xml(tree: ET.ElementTree, path: str) -> None:
    root = tree.getroot()
    if hasattr(ET, "indent"):
        ET.indent(root, space="  ")
    tree.write(path, xml_declaration=True, encoding="utf-8")


def require_section(root: ET.Element, name: str) -> ET.Element:
    sec = root.find(name)
    if sec is None:
        raise ValueError(f"Missing required section <{name}>")
    return sec


def copy_section(root_src: ET.Element, root_dst: ET.Element, section_name: str) -> ET.Element:
    src = require_section(root_src, section_name)
    new_sec = copy.deepcopy(src)
    root_dst.append(new_sec)
    return new_sec


def index_by_type(section: Optional[ET.Element], tag: str) -> Dict[str, ET.Element]:
    if section is None:
        return {}
    result: Dict[str, ET.Element] = {}
    for elem in section.findall(tag):
        type_id = elem.get("type")
        if type_id:
            result[type_id] = elem
    return result


def build_dmff_custom_nonbonded(root_dmff: ET.Element, root_out: ET.Element, *, strict: bool) -> None:
    custom_nb = ET.SubElement(root_out, "CustomNonbondedForce", {"bondCutoff": "3", "energy": CUSTOM_NB_ENERGY})
    for name in CUSTOM_NB_PARAMS:
        ET.SubElement(custom_nb, "PerParticleParameter", {"name": name})

    atom_types = require_section(root_dmff, "AtomTypes")
    all_types = [elem.get("name") for elem in atom_types.findall("Type") if elem.get("name")]

    slater_ex = index_by_type(require_section(root_dmff, "SlaterExForce"), "Atom")
    slater_sres = index_by_type(require_section(root_dmff, "SlaterSrEsForce"), "Atom")
    slater_srpol = index_by_type(require_section(root_dmff, "SlaterSrPolForce"), "Atom")
    slater_dhf = index_by_type(require_section(root_dmff, "SlaterDhfForce"), "Atom")
    slater_srdisp = index_by_type(require_section(root_dmff, "SlaterSrDispForce"), "Atom")
    admpp_disp = index_by_type(require_section(root_dmff, "ADMPDispPmeForce"), "Atom")

    missing: List[str] = []

    for type_id in all_types:
        if type_id not in slater_ex or type_id not in slater_sres or type_id not in slater_srpol:
            missing.append(type_id)
            continue
        if type_id not in slater_dhf or type_id not in slater_srdisp or type_id not in admpp_disp:
            missing.append(type_id)
            continue

        atom = ET.SubElement(custom_nb, "Atom", {"type": type_id})
        atom.set("Aexch", slater_ex[type_id].get("A", "0"))
        atom.set("Aelec", slater_sres[type_id].get("A", "0"))
        atom.set("Aind", slater_srpol[type_id].get("A", "0"))
        atom.set("Adhf", slater_dhf[type_id].get("A", "0"))
        atom.set("Adisp", slater_srdisp[type_id].get("A", "0"))
        atom.set("Bexp", slater_ex[type_id].get("B", "0"))
        atom.set("Q", slater_sres[type_id].get("Q", "0"))
        atom.set("C6", admpp_disp[type_id].get("C6", "0"))
        atom.set("C8", admpp_disp[type_id].get("C8", "0"))
        atom.set("C10", admpp_disp[type_id].get("C10", "0"))
        atom.set("C12", "0.0")

    if missing and strict:
        preview = ", ".join(missing[:12])
        suffix = "..." if len(missing) > 12 else ""
        raise ValueError(f"Missing CustomNonbonded parameters for {len(missing)} types: {preview}{suffix}")


def convert_dmff_xml(input_dmff: str, output_converted: str, *, strict: bool = True) -> None:
    tree_dmff = parse_xml(input_dmff)
    root_dmff = tree_dmff.getroot()

    root_out = ET.Element("ForceField")

    copy_section(root_dmff, root_out, "AtomTypes")
    copy_section(root_dmff, root_out, "Residues")

    admp_pme = require_section(root_dmff, "ADMPPmeForce")
    mpid_force = ET.SubElement(root_out, "MPIDForce")
    mpid_force.attrib.update(admp_pme.attrib)
    if "coulomb14scale" not in mpid_force.attrib:
        mpid_force.set("coulomb14scale", "0")

    for child in admp_pme:
        if child.tag == "Atom":
            out = ET.SubElement(mpid_force, "Multipole")
            out.attrib.update(child.attrib)
        elif child.tag == "Polarize":
            out = ET.SubElement(mpid_force, "Polarize")
            out.attrib.update(child.attrib)

    build_dmff_custom_nonbonded(root_dmff, root_out, strict=strict)
    write_xml(ET.ElementTree(root_out), output_converted)


def build_atom_map(xml_root: ET.Element) -> Dict[TResidueAtom, str]:
    atom_map: Dict[TResidueAtom, str] = {}
    residues = xml_root.find("Residues")
    if residues is None:
        return atom_map

    for residue in residues.findall("Residue"):
        res_name = residue.get("name")
        if not res_name:
            continue
        for atom in residue.findall("Atom"):
            atom_name = atom.get("name")
            atom_type = atom.get("type")
            if atom_name and atom_type:
                atom_map[(res_name, atom_name)] = atom_type
    return atom_map


def build_type_maps(map_a: Dict[TResidueAtom, str], map_b: Dict[TResidueAtom, str], *, strict: bool) -> TypeMaps:
    shared = sorted(set(map_a) & set(map_b))
    if not shared:
        raise ValueError("No shared (Residue, Atom) pairs between base XML and converted XML")

    a_to_b: Dict[str, str] = {}
    b_to_a_candidates: Dict[str, List[str]] = defaultdict(list)

    for key in shared:
        a_type = map_a[key]
        b_type = map_b[key]
        old = a_to_b.get(a_type)
        if old is not None and old != b_type:
            msg = f"Ambiguous mapping for A type {a_type}: {old} vs {b_type} from {key}"
            if strict:
                raise ValueError(msg)
            continue
        a_to_b[a_type] = b_type
        b_to_a_candidates[b_type].append(a_type)

    b_to_a_unique: Dict[str, str] = {}
    for b_type, a_types in b_to_a_candidates.items():
        unique = sorted(set(a_types))
        if len(unique) == 1:
            b_to_a_unique[b_type] = unique[0]
        b_to_a_candidates[b_type] = unique

    return TypeMaps(a_to_b=a_to_b, b_to_a_unique=b_to_a_unique, b_to_a_candidates=b_to_a_candidates)


def ensure_section(root: ET.Element, name: str) -> ET.Element:
    sec = root.find(name)
    if sec is None:
        sec = ET.SubElement(root, name)
    return sec


def remove_existing_section(root: ET.Element, name: str) -> None:
    old = root.find(name)
    if old is not None:
        root.remove(old)


def normalize_type_sort_key(type_id: str) -> Tuple[int, str]:
    digits = "".join(ch for ch in type_id if ch.isdigit())
    return (int(digits), type_id) if digits else (10**12, type_id)


def copy_clean_section_from_converted(root_base: ET.Element, root_converted: ET.Element, section: str) -> ET.Element:
    remove_existing_section(root_base, section)
    src = require_section(root_converted, section)
    new_sec = copy.deepcopy(src)
    for child in list(new_sec):
        if child.tag in {"Atom", "Multipole", "Polarize"}:
            new_sec.remove(child)
    root_base.append(new_sec)
    return new_sec


def remap_type_reference(value: Optional[str], maps: TypeMaps, *, strict: bool, context: str) -> Optional[str]:
    if value is None:
        return None

    raw = value.strip()
    if raw == "":
        return raw

    sign = ""
    token = raw
    if token.startswith("-"):
        sign = "-"
        token = token[1:]

    mapped = maps.b_to_a_unique.get(token)
    if mapped:
        return f"{sign}{mapped}"

    candidates = maps.b_to_a_candidates.get(token, [])
    if candidates:
        if strict:
            raise ValueError(f"Ambiguous local-frame remap for {context}: {raw} -> {candidates}")
        return f"{sign}{candidates[0]}"

    return raw


def append_custom_nonbonded_atoms(root_base: ET.Element, root_converted: ET.Element, maps: TypeMaps) -> int:
    sec_base = ensure_section(root_base, "CustomNonbondedForce")
    sec_converted = require_section(root_converted, "CustomNonbondedForce")

    params_b = index_by_type(sec_converted, "Atom")
    existing = {e.get("type") for e in sec_base.findall("Atom") if e.get("type")}

    inserted = 0
    for a_type, b_type in sorted(maps.a_to_b.items(), key=lambda kv: normalize_type_sort_key(kv[0])):
        if a_type in existing:
            continue
        src = params_b.get(b_type)
        if src is None:
            continue
        node = copy.deepcopy(src)
        node.set("type", a_type)
        sec_base.append(node)
        inserted += 1

    return inserted


def append_mpid_terms(root_base: ET.Element, root_converted: ET.Element, maps: TypeMaps, *, strict: bool) -> Tuple[int, int]:
    sec_base = ensure_section(root_base, "MPIDForce")
    sec_conv = require_section(root_converted, "MPIDForce")

    if "coulomb14scale" not in sec_base.attrib:
        sec_base.set("coulomb14scale", "0")

    conv_m = index_by_type(sec_conv, "Multipole")
    conv_p = index_by_type(sec_conv, "Polarize")

    existing_m = {e.get("type") for e in sec_base.findall("Multipole") if e.get("type")}
    existing_p = {e.get("type") for e in sec_base.findall("Polarize") if e.get("type")}

    add_m, add_p = 0, 0

    for a_type, b_type in sorted(maps.a_to_b.items(), key=lambda kv: normalize_type_sort_key(kv[0])):
        if a_type not in existing_m and b_type in conv_m:
            node = copy.deepcopy(conv_m[b_type])
            node.set("type", a_type)
            for attr in ("kz", "kx", "ky"):
                if attr in node.attrib:
                    node.set(
                        attr,
                        remap_type_reference(
                            node.get(attr),
                            maps,
                            strict=strict,
                            context=f"Multipole.{attr} for type {a_type}",
                        )
                        or "",
                    )
            sec_base.append(node)
            add_m += 1

        if a_type not in existing_p and b_type in conv_p:
            node = copy.deepcopy(conv_p[b_type])
            node.set("type", a_type)
            sec_base.append(node)
            add_p += 1

    return add_m, add_p


def zero_nonbonded_charges(root: ET.Element) -> int:
    sec = root.find("NonbondedForce")
    if sec is None:
        return 0
    count = 0
    for atom in sec.findall("Atom"):
        atom.set("charge", "0")
        count += 1
    return count


def merge_converted_into_base(
    base_xml: str,
    converted_xml: str,
    output_xml: str,
    *,
    strict: bool = True,
    zero_charges: bool = False,
) -> Dict[str, int]:
    tree_base = parse_xml(base_xml)
    tree_conv = parse_xml(converted_xml)
    root_base = tree_base.getroot()
    root_conv = tree_conv.getroot()

    copy_clean_section_from_converted(root_base, root_conv, "MPIDForce")
    copy_clean_section_from_converted(root_base, root_conv, "CustomNonbondedForce")

    maps = build_type_maps(build_atom_map(root_base), build_atom_map(root_conv), strict=strict)

    custom_added = append_custom_nonbonded_atoms(root_base, root_conv, maps)
    multipole_added, polarize_added = append_mpid_terms(root_base, root_conv, maps, strict=strict)
    zeroed = zero_nonbonded_charges(root_base) if zero_charges else 0

    write_xml(tree_base, output_xml)

    return {
        "mapped_types": len(maps.a_to_b),
        "custom_atoms_added": custom_added,
        "multipoles_added": multipole_added,
        "polarizes_added": polarize_added,
        "nonbonded_charges_zeroed": zeroed,
    }


def run_pipeline(
    dmff_xml: str,
    base_xml: str,
    output_xml: str,
    *,
    converted_out: Optional[str],
    strict: bool,
    zero_charges: bool,
) -> Dict[str, int]:
    converted_path = converted_out or str(Path(output_xml).with_suffix(".converted.xml"))
    convert_dmff_xml(dmff_xml, converted_path, strict=strict)
    return merge_converted_into_base(
        base_xml,
        converted_path,
        output_xml,
        strict=strict,
        zero_charges=zero_charges,
    )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="DMFF XML -> OpenMM XML converter/merger")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_convert = sub.add_parser("convert", help="Convert DMFF XML to OpenMM-style converted XML")
    p_convert.add_argument("--dmff", required=True, help="Input DMFF XML")
    p_convert.add_argument("--out", required=True, help="Converted output XML")
    p_convert.add_argument("--strict", action="store_true", help="Enable strict validation (fail on missing params)")

    p_merge = sub.add_parser("merge", help="Merge converted XML terms into base XML")
    p_merge.add_argument("--base", required=True, help="Base OpenMM XML")
    p_merge.add_argument("--converted", required=True, help="Converted XML from `convert`")
    p_merge.add_argument("--out", required=True, help="Merged output XML")
    p_merge.add_argument("--strict", action="store_true", help="Fail on ambiguous mapping")
    p_merge.add_argument("--zero-nonbonded-charges", action="store_true", help="Set <NonbondedForce>/<Atom charge=0>")

    p_pipeline = sub.add_parser("pipeline", help="One-step convert + merge")
    p_pipeline.add_argument("--dmff", required=True, help="Input DMFF XML")
    p_pipeline.add_argument("--base", required=True, help="Base OpenMM XML")
    p_pipeline.add_argument("--out", required=True, help="Merged output XML")
    p_pipeline.add_argument("--converted-out", help="Optional converted intermediate output XML")
    p_pipeline.add_argument("--strict", action="store_true", help="Fail on ambiguous mapping")
    p_pipeline.add_argument("--zero-nonbonded-charges", action="store_true", help="Set <NonbondedForce>/<Atom charge=0>")

    return ap


def main() -> None:
    args = build_parser().parse_args()
    strict = getattr(args, "strict", False)

    if args.cmd == "convert":
        convert_dmff_xml(args.dmff, args.out, strict=strict)
        print(f"[ok] converted: {args.dmff} -> {args.out}")
        return

    if args.cmd == "merge":
        stats = merge_converted_into_base(
            args.base,
            args.converted,
            args.out,
            strict=strict,
            zero_charges=args.zero_nonbonded_charges,
        )
        print(f"[ok] merged: {args.base} + {args.converted} -> {args.out}")
        print(f"[stats] {stats}")
        return

    stats = run_pipeline(
        args.dmff,
        args.base,
        args.out,
        converted_out=args.converted_out,
        strict=strict,
        zero_charges=args.zero_nonbonded_charges,
    )
    print(f"[ok] pipeline: {args.dmff} + {args.base} -> {args.out}")
    print(f"[stats] {stats}")


if __name__ == "__main__":
    main()
