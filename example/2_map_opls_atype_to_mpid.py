from lxml import etree
import copy
from xml.dom import minidom
from collections import defaultdict
from copy import deepcopy
def find_many_to_one(d):
    value_to_keys = defaultdict(list)
    
    # 构建反向映射：值 → 键列表
    for key, value in d.items():
        value_to_keys[value].append(key)
    
    # 筛选出那些被多个键指向的值
    many_to_one = {value: keys for value, keys in value_to_keys.items() if len(keys) > 1}
    
    return many_to_one


def save_pretty_xml(root, output_path):
    rough_string = etree.tostring(root, encoding='utf-8', xml_declaration=True)
    reparsed = minidom.parseString(rough_string)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(reparsed.toprettyxml(indent="  "))

def build_atom_map(xml_root):
    """
    构建 (Residue, Atom) → type 映射表
    """
    atom_map = {}
    for res in xml_root.xpath(".//Residue"):
        res_name = res.get("name")
        for atom in res.xpath("Atom"):
            atom_name = atom.get("name")
            atom_type = atom.get("type")
            atom_map[(res_name, atom_name)] = atom_type
    return atom_map

def ensure_force_section(root, section_name):
    """
    确保 A 文件中存在指定力场部分，如果没有则创建
    """
    section = root.find(section_name)
    if section is None:
        section = etree.SubElement(root, section_name)
    return section
import copy

def insert_force_terms_by_index(target_root, terms, type_map, section_name):
    section = ensure_force_section(target_root, section_name)
    inserted = 0

    for i, term in enumerate(terms):
        # 深拷贝原始元素，避免断开解析树
        new_term = copy.deepcopy(term)
        for attr, val in new_term.attrib.items():
            new_val = type_map.get(val, val)
            new_term.set(attr, new_val)
        section.insert(i, new_term)
        inserted += 1

    print(f"✅ 插入 {inserted} 个条目到 <{section_name}>（按索引，保持结构）")


def remove_elements(section, tags_to_remove):
    """
    删除指定标签的元素
    """
    removed = 0
    for elem in list(section):
        if elem.tag in tags_to_remove:
            section.remove(elem)
            removed += 1
    return removed

def copy_cleaned_force_section(B_root, A_root, section_name, tags_to_remove):
    """
    从 B 中复制指定力场部分到 A，并删除指定子元素
    """
    source_section = B_root.find(section_name)
    if source_section is None:
        print(f"⚠️ B 文件中未找到 <{section_name}>，跳过")
        return

    # 深拷贝整个 section
    new_section = copy.deepcopy(source_section)

    # 删除指定标签
    removed_count = remove_elements(new_section, tags_to_remove)

    # 插入到 A 文件中
    A_root.append(new_section)
    print(f"✅ 已复制 <{section_name}>，并删除 {removed_count} 个元素：{tags_to_remove}")

def process_forcefield(A_path, B_path, output_path):
    tree_A = etree.parse(A_path)
    root_A = tree_A.getroot()

    tree_B = etree.parse(B_path)
    root_B = tree_B.getroot()

    tags_to_remove = ["Atom", "Polarize", "Multipole"]
    sections_to_copy = ["MPIDForce", "CustomNonbondedForce"]

    for section in sections_to_copy:
        copy_cleaned_force_section(root_B, root_A, section, tags_to_remove)

    # 保存并格式化
    xml_bytes = etree.tostring(tree_A, pretty_print=True, encoding="utf-8", xml_declaration=True)
    with open(output_path, "wb") as f:
        f.write(xml_bytes)

    print(f"\n📁 已保存合并后的 A 文件为: {output_path}")

def build_antisymmetric_dict(pairs):
    antisymmetric = {}
    for a, b in pairs:
        if b in antisymmetric and antisymmetric[b] == a:
            raise ValueError(f"Conflict: {b} → {a} already exists as {a} → {b}")
        antisymmetric[a] = b
    return antisymmetric



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
    A_xml = "merged_opls.xml"
    B_xml = "converted_forcefield.xml"
    output_path = "merged_cleaned.xml"

    process_forcefield(A_xml, B_xml, output_path)

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

        for a_type, atom_element in parameters_A.items():
            new_atom = etree.Element("Atom")
            new_atom.attrib.update(atom_element.attrib)
            new_atom.set("type", a_type)

            # 设置每个 Atom 元素的 tail 为换行缩进
            new_atom.tail = "\n    "
            CustomNB.append(new_atom)

        return root_A


    def update_mpidforce_A_with_B_parameters(root_A, root_B, type_map, type_reverse_map):
        # 提取 B 文件中的 Multipole 和 Polarize Atom 参数
        mpid_B = root_B.find("MPIDForce")

        parameters_B_multipole = {
            atom.get("type"): atom for atom in mpid_B.findall("Multipole") if atom.get("type")
        }
        parameters_B_polarize = {
            atom.get("type"): atom for atom in mpid_B.findall("Polarize") if atom.get("type")
        }

        # 根据 type_map 映射，构建 A 文件中需要添加的 Atom 元素
        parameters_A_multipole = {}
        parameters_A_polarize = {}
        # def remap_kx_kz(atom, type_reverse_map):
        #     kz = atom.attrib.get('kz', '')
        #     kx = atom.attrib.get('kx', '')

        #     if kz == '':
        #         return None  # 或者 return atom 不修改

        #     try:
        #         kz_val = int(kz)
        #         kx_val = int(kx) if kx != '' else 0
        #     except ValueError:
        #         return atom  # 非数字，跳过

        #     target_kz = type_reverse_map.get(str(abs(kz_val)), kz)
        #     target_kx = type_reverse_map.get(str(abs(kx_val)), kx)

        #     atom.attrib['kz'] = f"{'-' if kz_val < 0 else ''}{target_kz}"
        #     atom.attrib['kx'] = f"{'-' if kx_val < 0 else ''}{target_kx}"

        #     return atom

        # for a_type, b_type in type_map.items():
        #     atom = deepcopy(parameters_B_multipole[b_type])
        #     atom.set("type", a_type)
        #     atom = remap_kx_kz(atom, type_reverse_map)
        #     parameters_A_multipole[a_type] = atom
                 
        for a_type, b_type in type_map.items():
            # print(a_type, b_type)
            parameters_A_multipole[a_type] = deepcopy(parameters_B_multipole[b_type])

            kz = parameters_B_multipole[b_type].get('kz')
            kx = parameters_B_multipole[b_type].get('kx')
            print(b_type, kz, kx)
            if kz == '':
                print(kz, kx)
                continue
            elif int(kz) > 0:
                target_kz = type_reverse_map[str(abs(int(kz)))]
                parameters_A_multipole[a_type].set('kz', f'{target_kz}')
                # parameters_A_multipole[a_type].attrib['kz'] = f'{target_kz}'
                if kx == "":
                    continue
                elif int(kx) < 0:
                    target_kx = type_reverse_map[str(abs(int(kx)))]
                    parameters_A_multipole[a_type].set('kx', f'-{target_kx}')
                    # parameters_A_multipole[a_type].attrib['kx'] = f'-{target_kx}'
                elif int(kx) > 0:
                    target_kx = type_reverse_map[str(abs(int(kx)))]
                    parameters_A_multipole[a_type].set('kx', f'{target_kx}')
                    # parameters_A_multipole[a_type].attrib['kx'] = f'{target_kx}'
            elif int(kz) < 0:
                print(target_kz, target_kx)
                target_kz = type_reverse_map[str(abs(int(kz)))]
                target_kx = type_reverse_map[str(abs(int(kx)))]
                parameters_A_multipole[a_type].set('kz', f'-{target_kz}')
                parameters_A_multipole[a_type].set('kx', f'{target_kx}')
        for a_type, b_type in type_map.items():
            parameters_A_polarize[a_type] = parameters_B_polarize[b_type]

        # 找到 A 文件中的目标节点
        mpid_A = root_A.find("MPIDForce")
        # 添加 Multipole Atom 元素
        for a_type, atom_element in parameters_A_multipole.items():
            new_atom = etree.Element("Multipole")
            new_atom.attrib.update(atom_element.attrib)
            new_atom.set("type", a_type)
            new_atom.tail = "\n    "
            mpid_A.append(new_atom)

        # 添加 Polarize Atom 元素
        for a_type, atom_element in parameters_A_polarize.items():
            new_atom = etree.Element("Polarize")
            new_atom.attrib.update(atom_element.attrib)
            new_atom.set("type", a_type)
            new_atom.tail = "\n    "
            mpid_A.append(new_atom)

        return root_A

    # Load XML files
    # A_xml_path = "merged_opls.xml"
    A_xml_path = "merged_cleaned.xml"
    B_xml_path = "converted_forcefield.xml"
    updated_A_path = "merged_cleaned_append.xml"

    try:
        tree_A = etree.parse(A_xml_path)
        root_A = tree_A.getroot()

        tree_B = etree.parse(B_xml_path)
        root_B = tree_B.getroot()

        print("🔍 构建 Residue-Atom 映射...")
        map_A = build_atom_map(root_A)
        map_B = build_atom_map(root_B)

        print("🔁 构建类型映射表...")
        type_map = {}
        for key in map_A:
            if key in map_B:
                type_map[map_A[key]] = map_B[key]
        
        result = find_many_to_one(type_map)
        type_reverse_map = {}
        for key in map_B:
            if key in map_A:
                type_reverse_map[map_B[key]] = map_A[key]
        
        print("🛠️ 更新 A.xml 中的 AtomType 参数...")
        updated_root_A = update_customNB_A_with_B_parameters(root_A, root_B, type_map)
        updated_root_A = update_mpidforce_A_with_B_parameters(updated_root_A, root_B, type_map, type_reverse_map)

        tree_updated_A = etree.ElementTree(updated_root_A)
        # save_pretty_xml(updated_root_A, updated_A_path)
        tree_updated_A.write(updated_A_path, pretty_print=True, xml_declaration=True, encoding='UTF-8') 
        print(f"✅ 更新完成，保存路径: {updated_A_path}")

    except Exception as e:
        print(f"❌ 处理过程中发生错误: {e}")


    

    tree_A = etree.parse('merged_cleaned_append.manual.xml')
    xml_A = tree_A.getroot()

    # NonbondedForce charge 置零
    zero_charges_in_nonbonded(xml_A)
    tree_A.write("caff_5_mpid_LJ_bond_all.xml", encoding="utf-8", xml_declaration=True, pretty_print=True)
    print("最终合并完成，文件保存至: caff_5_mpid_LJ_bond_all.xml")
