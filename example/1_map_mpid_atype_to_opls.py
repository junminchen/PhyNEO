from lxml import etree
from xml.etree.ElementTree import tostring

def process_B_based_on_A(A_xml, B_xml, output_path):

    xml_A = etree.parse(A_xml)
    xml_B = etree.parse(B_xml)
    print("🔍 开始清理 B 文件中无用 Residue 和相关类型...")
    clean_B_by_A(xml_A, xml_B)

    print("\n🔁 开始替换 B 文件中 Residue 的 Atom type...")
    map_and_replace_types(A_xml, 'tmp.xml', output_path)

    print("\n✅ 全部处理完成")

def clean_B_by_A(xml_A, xml_B):
    root_A = xml_A.getroot()
    root_B = xml_B.getroot()

    # 1. 收集 A 文件中的 Residue 名称
    residues_A = {res.get("name") for res in root_A.xpath(".//Residue") if res.get("name")}

    # 2. 找出 B 文件中不在 A 中的 Residue，收集 type/class 并删除 Residue
    types_to_remove = set()
    residues_B = root_B.find("Residues")
    removed_residues = 0
    for res in list(residues_B.xpath("Residue")):
        res_name = res.get("name")
        if res_name not in residues_A:
            for atom in res.xpath("Atom"):
                t = atom.get("type")
                if t:
                    types_to_remove.add(t)
            residues_B.remove(res)
            removed_residues += 1

    print(f"🗑️ 删除 Residues 中 {removed_residues} 个 Residue")
    print(f"🧹 收集到待删除的 type/class: {sorted(types_to_remove)}")

    # 3. 删除 AtomTypes 中对应的 Type
    atomtypes = root_B.find("AtomTypes")
    removed_atomtypes = 0
    for t in list(atomtypes.xpath("Type")):
        name = t.get("name")
        if name in types_to_remove:
            atomtypes.remove(t)
            removed_atomtypes += 1
    print(f"🧹 删除 AtomTypes 中 {removed_atomtypes} 个条目")

    # 4. 自动识别力场部分并删除引用的 type/class
    force_sections = ["HarmonicBondForce", "HarmonicAngleForce", "PeriodicTorsionForce", "NonbondedForce"]
    attrs_to_check = ["class", "class1", "class2", "class3", "class4", "type", "type1", "type2"]

    for section_name in force_sections:
        section = root_B.find(section_name)
        if section is None:
            continue
        removed = 0
        for elem in list(section):
            classes = [elem.get(attr) for attr in attrs_to_check if elem.get(attr)]
            if any(c in types_to_remove for c in classes):
                section.remove(elem)
                removed += 1
        print(f"🧹 删除 {section_name} 中 {removed} 个条目")

    print("✅ 清理完成")

    # 5. 保存清理后的 B 文件为 tmp.xml
    xml_B.write("caff_1_opls.xml", pretty_print=True, encoding="utf-8", xml_declaration=True)
    print("📁 已保存清理后的 B 文件为 caff_1_opls.xml")
    return root_A, root_B

def map_and_replace_types(A_path, B_path, output_path):
    tree_A = etree.parse(A_path)
    root_A = tree_A.getroot()
    tree_B = etree.parse(B_path)
    root_B = tree_B.getroot()

    # 1. 构建 A 中 Residue 的 (residue, atom) → type 映射
    a_atom_map = {}
    for res in root_A.findall(".//Residue"):
        res_name = res.get("name")
        for atom in res.findall("Atom"):
            a_atom_map[(res_name, atom.get("name"))] = atom.get("type")

    # 2. 构建 B 中要替换的 type 映射表：B_type → A_type
    b_type_to_a_type = {}
    for res in root_B.findall(".//Residue"):
        res_name = res.get("name")
        for atom in res.findall("Atom"):
            key = (res_name, atom.get("name"))
            if key in a_atom_map:
                b_type_to_a_type[atom.get("type")] = a_atom_map[key]

    print("即将执行的替换映射：")
    for b, a in b_type_to_a_type.items():
        print(f"  {b}  →  {a}")

    # 3. 全局查找并替换所有 type/class 属性
    #    包含 class1, class2, class3, class4，用于 Angle/Torsion 里的 classN
    attrs_to_check = ["type", "class", "class1", "class2", "class3", "class4"]
    count = 0
    for elem in root_B.iter():
        for attr in attrs_to_check:
            old = elem.get(attr)
            if old in b_type_to_a_type:
                new = b_type_to_a_type[old]
                elem.set(attr, new)
                print(f"替换 <{elem.tag}> @{attr}: {old} → {new}")
                count += 1

    print(f"总共替换了 {count} 处 type/class 属性。")

    # 4. 保存
    tree_B.write(output_path,
                 encoding="utf-8",
                 xml_declaration=True,
                 pretty_print=True)
    print(f"修改后的 B 力场已保存至: {output_path}")


def copy_all_bond_related_terms(xml_B, xml_A):
    bond_related_tags = [
        "HarmonicBondForce",
        "HarmonicAngleForce",
        "PeriodicTorsionForce",
        "RBTorsionForce",
        "CustomBondForce",
        "CustomAngleForce",
        "CustomTorsionForce",
        "NonbondedForce"
    ]

    for tag in bond_related_tags:
        b_nodes = xml_B.findall(tag)
        if b_nodes:
            for b_node in b_nodes:
                b_str = tostring(b_node)
                # 检查是否已存在相同内容
                if not any(tostring(a_node) == b_str for a_node in xml_A.findall(tag)):
                    xml_A.append(b_node)
            print(f"已从 B 中复制 {len(b_nodes)} 个 {tag} 到 A（已去重）")
        else:
            print(f"B 中未找到 {tag}，跳过")


def sync_atomtype_class(xml_root):
    atom_types = xml_root.find("AtomTypes")
    if atom_types is None:
        print("未找到 AtomTypes 节点，跳过同步")
        return

    for type_elem in atom_types.findall("Type"):
        type_name = type_elem.get("name")
        type_elem.set("class", type_name)
    print("已将所有 AtomType 的 class 设置为与 type 相同")



def zero_charges_in_nonbonded(xml_root):
    """
    把 NonbondedForce 中所有 <Atom charge="..."/> 的 charge 设为 0
    """
    nb = xml_root.find("NonbondedForce")
    if nb is None:
        print("未找到 NonbondedForce，跳过 charge 置零")
        return
    count = 0
    for p in nb.findall("Atom"):
        p.set("charge", "0")
        count += 1
    print(f"已将 NonbondedForce 中 {count} 个 Atom 的 charge 设为 0")

if __name__ == "__main__":
    # A_xml = "converted_forcefield.xml"
    # B_xml = "opls_solvent_bond.xml"
    # output_xml = "updated_B_forcefield.xml"
    # map_and_replace_types(A_xml, B_xml, output_xml)

    # A_xml = "converted_forcefield.xml"
    # B_xml = "opls_solvent.xml"
    # output_xml = "tmp.xml"

    # process_B_based_on_A(A_xml, B_xml, output_xml)
    # # 合并所有键相关力场项
    # tree_A = etree.parse(A_xml)
    # tree_B = etree.parse(output_xml)

    # xml_A = tree_A.getroot()
    # xml_B = tree_B.getroot()

    # # 合并键相关项
    # copy_all_bond_related_terms(xml_B, xml_A)

    # # 同步 AtomTypes 的 class 属性
    # sync_atomtype_class(xml_A)


    # # NonbondedForce charge 置零
    # zero_charges_in_nonbonded(xml_A)
    # tree_A.write("caff_3_mpid_LJ_bond.xml", encoding="utf-8", xml_declaration=True, pretty_print=True)
    # print("最终合并完成，文件保存至: caff_3_mpid_LJ_bond.xml")
    # # 保存最终结果
    # tree_A.write("caff_5_mpid_slater_bond.xml", encoding="utf-8", xml_declaration=True, pretty_print=True)
    # print("最终合并完成，文件保存至: caff_5_mpid_slater_bond.xml")



    A_xml = "converted_forcefield.xml"
    B_xml = "opls_salt.xml"
    output_xml = "tmp.xml"

    process_B_based_on_A(A_xml, B_xml, output_xml)

    # 合并所有键相关力场项
    tree_A = etree.parse(A_xml)
    tree_B = etree.parse(output_xml)

    xml_A = tree_A.getroot()
    xml_B = tree_B.getroot()

    # 合并键相关项
    copy_all_bond_related_terms(xml_B, xml_A)

    # 同步 AtomTypes 的 class 属性
    sync_atomtype_class(xml_A)


    # NonbondedForce charge 置零
    zero_charges_in_nonbonded(xml_A)
    tree_A.write("caff_3_mpid_LJ_bond.salt.xml", encoding="utf-8", xml_declaration=True, pretty_print=True)
    print("最终合并完成，文件保存至: caff_3_mpid_LJ_bond.salt.xml")
    # 保存最终结果
    tree_A.write("caff_5_mpid_slater_bond.salt.xml", encoding="utf-8", xml_declaration=True, pretty_print=True)
    print("最终合并完成，文件保存至: caff_5_mpid_slater_bond.salt.xml")
