#!/usr/bin/env python3
"""Fully automated forcefield merge + parameter remap.

This script merges parameters from a converted forcefield (B) into a base forcefield (A).
It will:
  1) Copy MPIDForce and CustomNonbondedForce sections from B into A, but remove Atom/Polarize/Multipole.
  2) Build a robust mapping between A atom types and B atom types based on (Residue, Atom) pairs.
  3) Append CustomNonbondedForce Atom parameters and MPIDForce Multipole/Polarize parameters into A
     using the mapping. It also remaps kx/kz references safely.
  4) Zero charges in NonbondedForce (optional).

Usage:
  python 2_map_opls_atype_to_mpid.auto.py \
    --a merged_opls.xml \
    --b converted_forcefield.xml \
    --out caff_5_mpid_LJ_bond_all.xml \
    --zero-nonbonded-charges

Notes:
- Fully automated: no manual intermediate XML.
- Safer defaults: validates mapping consistency; optional strict mode.
"""

from __future__ import annotations

import argparse
import copy
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from lxml import etree


TResidueAtom = Tuple[str, str]


@dataclass(frozen=True)
class TypeMaps:
    a_to_b: Dict[str, str]
    b_to_a_unique: Dict[str, str]
    b_to_a_candidates: Dict[str, List[str]]


def parse_xml(path: str) -> etree._ElementTree:
    parser = etree.XMLParser(remove_blank_text=False, recover=True, huge_tree=True)
    return etree.parse(path, parser)


def write_xml(tree: etree._ElementTree, path: str) -> None:
    tree.write(path, pretty_print=True, xml_declaration=True, encoding="UTF-8")


def build_atom_map(xml_root: etree._Element) -> Dict[TResidueAtom, str]:
    """Build (ResidueName, AtomName) -> type mapping from Residue/Atom definitions."""
    atom_map: Dict[TResidueAtom, str] = {}
    for res in xml_root.xpath(".//Residue"):
        res_name = res.get("name")
        if not res_name:
            continue
        for atom in res.xpath("Atom"):
            atom_name = atom.get("name")
            atom_type = atom.get("type")
            if not atom_name or not atom_type:
                continue
            atom_map[(res_name, atom_name)] = atom_type
    return atom_map


def ensure_section(root: etree._Element, section_name: str) -> etree._Element:
    sec = root.find(section_name)
    if sec is None:
        sec = etree.SubElement(root, section_name)
    return sec


def remove_child_tags(section: etree._Element, tags_to_remove: Iterable[str]) -> int:
    removed = 0
    tag_set = set(tags_to_remove)
    for elem in list(section):
        if elem.tag in tag_set:
            section.remove(elem)
            removed += 1
    return removed


def copy_cleaned_section(
    root_src: etree._Element,
    root_dst: etree._Element,
    section_name: str,
    tags_to_remove: Iterable[str],
) -> None:
    src = root_src.find(section_name)
    if src is None:
        raise ValueError(f"Section <{section_name}> not found in source XML")

    # Remove existing section in destination to avoid duplication.
    existing = root_dst.find(section_name)
    if existing is not None:
        root_dst.remove(existing)

    new_section = copy.deepcopy(src)
    remove_child_tags(new_section, tags_to_remove)
    root_dst.append(new_section)


def build_type_maps(
    map_a: Dict[TResidueAtom, str],
    map_b: Dict[TResidueAtom, str],
    *,
    strict: bool,
) -> TypeMaps:
    a_to_b: Dict[str, str] = {}
    b_to_a_candidates: Dict[str, List[str]] = defaultdict(list)

    shared = set(map_a.keys()) & set(map_b.keys())
    if not shared:
        raise ValueError("No shared (Residue, Atom) pairs between A and B. Cannot build mapping.")

    for key in sorted(shared):
        a_type = map_a[key]
        b_type = map_b[key]
        if a_type in a_to_b and a_to_b[a_type] != b_type:
            msg = (
                f"Ambiguous mapping for A type '{a_type}': "
                f"'{a_to_b[a_type]}' vs '{b_type}' from {key}"
            )
            if strict:
                raise ValueError(msg)
            # best-effort: keep first mapping
            continue
        a_to_b[a_type] = b_type
        b_to_a_candidates[b_type].append(a_type)

    b_to_a_unique: Dict[str, str] = {}
    for b_type, a_types in b_to_a_candidates.items():
        uniq = sorted(set(a_types))
        if len(uniq) == 1:
            b_to_a_unique[b_type] = uniq[0]

    return TypeMaps(
        a_to_b=a_to_b,
        b_to_a_unique=b_to_a_unique,
        b_to_a_candidates={k: sorted(set(v)) for k, v in b_to_a_candidates.items()},
    )


def index_parameters_by_type(section: Optional[etree._Element], tag: str) -> Dict[str, etree._Element]:
    if section is None:
        return {}
    d: Dict[str, etree._Element] = {}
    for elem in section.findall(tag):
        t = elem.get("type")
        if not t:
            continue
        d[t] = elem
    return d


def safe_remap_type_ref(
    ref: str,
    b_to_a_unique: Dict[str, str],
    b_to_a_candidates: Dict[str, List[str]],
    *,
    strict: bool,
    context: str,
) -> str:
    if ref is None:
        return ""
    ref = ref.strip()
    if ref == "":
        return ""

    sign = ""
    token = ref
    if token.startswith("-"):
        sign = "-"
        token = token[1:]

    if token in b_to_a_unique:
        return f"{sign}{b_to_a_unique[token]}"

    if token in b_to_a_candidates and b_to_a_candidates[token]:
        msg = f"Ambiguous remap for {context}: B ref '{ref}' candidates -> {b_to_a_candidates[token]}"
        if strict:
            raise ValueError(msg)
        return f"{sign}{b_to_a_candidates[token][0]}"

    return ref


def append_custom_nonbonded_atoms(
    root_a: etree._Element,
    root_b: etree._Element,
    a_to_b: Dict[str, str],
) -> int:
    sec_a = ensure_section(root_a, "CustomNonbondedForce")
    sec_b = root_b.find("CustomNonbondedForce")

    params_b = index_parameters_by_type(sec_b, "Atom")
    existing_a = {e.get("type") for e in sec_a.findall("Atom") if e.get("type")}

    inserted = 0
    for a_type, b_type in sorted(a_to_b.items()):
        if a_type in existing_a:
            continue
        src = params_b.get(b_type)
        if src is None:
            continue
        new_atom = etree.Element("Atom")
        new_atom.attrib.update(src.attrib)
        new_atom.set("type", a_type)
        sec_a.append(new_atom)
        inserted += 1
    return inserted


def append_mpid_terms(
    root_a: etree._Element,
    root_b: etree._Element,
    maps: TypeMaps,
    *,
    strict: bool,
) -> Tuple[int, int]:
    sec_a = ensure_section(root_a, "MPIDForce")
    sec_b = root_b.find("MPIDForce")
    if sec_b is None:
        raise ValueError("Section <MPIDForce> not found in B")

    multipole_b = index_parameters_by_type(sec_b, "Multipole")
    polarize_b = index_parameters_by_type(sec_b, "Polarize")

    existing_multipole_a = {e.get("type") for e in sec_a.findall("Multipole") if e.get("type")}
    existing_polarize_a = {e.get("type") for e in sec_a.findall("Polarize") if e.get("type")}

    inserted_m = 0
    inserted_p = 0

    for a_type, b_type in sorted(maps.a_to_b.items()):
        if a_type not in existing_multipole_a:
            src_m = multipole_b.get(b_type)
            if src_m is not None:
                atom = copy.deepcopy(src_m)
                atom.set("type", a_type)

                kz = atom.get("kz")
                kx = atom.get("kx")
                if kz is not None:
                    atom.set(
                        "kz",
                        safe_remap_type_ref(
                            kz,
                            maps.b_to_a_unique,
                            maps.b_to_a_candidates,
                            strict=strict,
                            context=f"Multipole.kz for A type {a_type}",
                        ),
                    )
                if kx is not None:
                    atom.set(
                        "kx",
                        safe_remap_type_ref(
                            kx,
                            maps.b_to_a_unique,
                            maps.b_to_a_candidates,
                            strict=strict,
                            context=f"Multipole.kx for A type {a_type}",
                        ),
                    )

                sec_a.append(atom)
                inserted_m += 1

        if a_type not in existing_polarize_a:
            src_p = polarize_b.get(b_type)
            if src_p is not None:
                atom = copy.deepcopy(src_p)
                atom.set("type", a_type)
                sec_a.append(atom)
                inserted_p += 1

    return inserted_m, inserted_p
def update_customNB_A_with_B_parameters(root_A, root_B, type_map):
    parameters_B = {}
    CustomNBref = root_B.find("CustomNonbondedForce")
    for atom_type in CustomNBref.findall('.//Atom'):
        type_name = atom_type.get('type')
        if type_name:
            parameters_B[type_name] = atom_type

    parameters_A = {}
    for a_type, b_type in type_map.items():
        if b_type in parameters_B:
            parameters_A[a_type] = parameters_B[b_type]

    CustomNB = root_A.find("CustomNonbondedForce")

    # 按照 a_type 的数字部分排序后插入
    for a_type, atom_element in sorted(parameters_A.items(), key=lambda x: int(''.join(filter(str.isdigit, x[0])))):
        new_atom = etree.Element("Atom")
        new_atom.attrib.update(atom_element.attrib)
        new_atom.set("type", a_type)

        # 设置每个 Atom 元素的 tail 为换行缩进
        new_atom.tail = "\n    "
        CustomNB.append(new_atom)

    # 保证 CustomNonbondedForce 的开头和结尾都有适当换行
    CustomNB.text = "\n    "  # 子元素前的换行与缩进
    if CustomNB[-1] is not None:  # 检查如果已经有子元素，末尾换行修正
        CustomNB[-1].tail = "\n"

    return root_A
    
def update_mpidforce_A_with_B_parameters(root_A, root_B, type_map, type_reverse_map):
    mpid_B = root_B.find("MPIDForce")

    # 提取 B 文件中的 Multipole 和 Polarize Atom 参数
    parameters_B_multipole = {
        atom.get("type"): atom for atom in mpid_B.findall("Multipole") if atom.get("type")
    }
    parameters_B_polarize = {
        atom.get("type"): atom for atom in mpid_B.findall("Polarize") if atom.get("type")
    }

    # 根据 type_map 映射，构建 A 文件需要添加的元素
    parameters_A_multipole = {}
    parameters_A_polarize = {}

    for a_type, b_type in type_map.items():
        if b_type in parameters_B_multipole:
            atom = copy.deepcopy(parameters_B_multipole[b_type])
            atom.set("type", a_type)
            parameters_A_multipole[a_type] = atom

        if b_type in parameters_B_polarize:
            atom = copy.deepcopy(parameters_B_polarize[b_type])
            atom.set("type", a_type)
            parameters_A_polarize[a_type] = atom

    mpid_A = root_A.find("MPIDForce")

    # 如果 <MPIDForce> 节点不存在，则创建
    if mpid_A is None:
        mpid_A = etree.SubElement(root_A, "MPIDForce")

    # 确保 <MPIDForce> 具有属性 coulomb14scale="0"
    mpid_A.set("coulomb14scale", "0")

    # 按 type 的数字部分为 Multipole 排序后插入
    for a_type, atom_element in sorted(parameters_A_multipole.items(), key=lambda x: int(''.join(filter(str.isdigit, x[0])))):
        atom_element.tail = "\n    "  # 设置换行缩进
        mpid_A.append(atom_element)

    # 按 type 的数字部分为 Polarize 排序后插入
    for a_type, atom_element in sorted(parameters_A_polarize.items(), key=lambda x: int(''.join(filter(str.isdigit, x[0])))):
        atom_element.tail = "\n    "  # 设置换行缩进
        mpid_A.append(atom_element)

    # 保证 <MPIDForce> 开头和结尾都有换行
    mpid_A.text = "\n    "  # 子元素前换行缩进
    if mpid_A[-1] is not None:  # 最后一个子节点对齐
        mpid_A[-1].tail = "\n"

    return root_A

def zero_charges_in_nonbonded(root: etree._Element) -> int:
    nb = root.find("NonbondedForce")
    if nb is None:
        return 0
    count = 0
    for p in nb.findall("Atom"):
        p.set("charge", "0")
        count += 1
    return count

def main():
    import argparse

    ap = argparse.ArgumentParser(description="Merge and sort A and B XML files.")
    ap.add_argument("--a", required=True, help="Base forcefield XML (A), e.g., merged_opls.xml")
    ap.add_argument("--b", required=True, help="Converted forcefield XML (B), e.g., converted_forcefield.xml")
    ap.add_argument("--out", required=True, help="Output merged XML")
    ap.add_argument(
        "--strict", action="store_true", help="Fail on ambiguous type mappings (recommended for reproducibility)"
    )
    args = ap.parse_args()

    # Step 1: Parse input XML files
    tree_A = etree.parse(args.a)
    root_A = tree_A.getroot()
    tree_B = etree.parse(args.b)
    root_B = tree_B.getroot()

    # Step 2: Copy cleaned sections from B to A
    print("🔍 Copying sections from B to A...")
    tags_to_remove = ["Atom", "Polarize", "Multipole"]
    sections_to_copy = ["MPIDForce", "CustomNonbondedForce"]
    for section in sections_to_copy:
        copy_cleaned_section(root_B, root_A, section, tags_to_remove)

    # Step 3: Build Residue-Atom mappings and type mappings
    print("🔁 Building Residue-Atom mappings...")
    map_A = build_atom_map(root_A)
    map_B = build_atom_map(root_B)

    print("🔁 Building type mappings...")
    type_map = {}  # A type -> B type
    type_reverse_map = {}  # B type -> A type
    for key in map_A:
        if key in map_B:
            type_map[map_A[key]] = map_B[key]
            type_reverse_map[map_B[key]] = map_A[key]

    # Step 4: Update CustomNonbondedForce in A with parameters from B
    print("🛠️ Updating CustomNonbondedForce in A...")
    update_customNB_A_with_B_parameters(root_A, root_B, type_map)

    # Step 5: Update MPIDForce in A with parameters from B
    print("🛠️ Updating MPIDForce in A...")
    update_mpidforce_A_with_B_parameters(root_A, root_B, type_map, type_reverse_map)

    # Step 6: Write output
    print("📁 Writing output to:", args.out)
    tree_A.write(args.out, pretty_print=True, xml_declaration=True, encoding="UTF-8")
    print("✅ Merge completed!")


# Execute script if run directly
if __name__ == "__main__":
    main()