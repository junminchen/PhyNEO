from lxml import etree
import os

def extract_residue_to_xml(source_xml_path, target_residue_names, output_xml_path):
    """
    从完整的力场 XML 中提取指定 Residue 及其相关的所有参数，生成独立的 XML 文件。
    
    Args:
        source_xml_path: 原始大 XML 路径 (e.g., phyneo_ecl.xml)
        target_residue_names: 需要提取的残基名称列表 (e.g., ["ECA", "DMC"])
        output_xml_path: 输出 XML 路径 (e.g., EC_DMC.xml)
    """
    # 1. 解析源 XML
    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(source_xml_path, parser)
    root = tree.getroot()
    
    new_root = etree.Element("ForceField")
    
    # -------------------------------------------------
    # 2. 提取 Residues 并收集所有使用的 atom type
    # -------------------------------------------------
    active_types = set()
    
    residues_node = root.find("Residues")
    new_residues_node = etree.SubElement(new_root, "Residues")
    
    found_residues = False
    if residues_node is not None:
        for res in residues_node.findall("Residue"):
            if res.get("name") in target_residue_names:
                found_residues = True
                # 复制该 Residue 节点
                new_residues_node.append(etree.fromstring(etree.tostring(res)))
                
                # 收集该 Residue 中所有 Atom 的 type
                for atom in res.findall("Atom"):
                    t = atom.get("type")
                    if t:
                        active_types.add(t)

    if not found_residues:
        print(f"Warning: None of the residues {target_residue_names} were found in {source_xml_path}")
        # 即使没找到也继续，以免完全生成空文件，但实际上应该检查逻辑
        # return 

    print(f"Found active types from {target_residue_names}: {active_types}")

    # -------------------------------------------------
    # 3. 提取 AtomTypes 并收集所有使用的 class
    # -------------------------------------------------
    active_classes = set()
    
    atomtypes_node = root.find("AtomTypes")
    new_atomtypes_node = etree.SubElement(new_root, "AtomTypes")
    
    if atomtypes_node is not None:
        for t in atomtypes_node.findall("Type"):
            name = t.get("name")
            if name in active_types:
                # 复制 Type 节点
                new_atomtypes_node.append(etree.fromstring(etree.tostring(t)))
                
                # 收集 class
                c = t.get("class")
                if c:
                    active_classes.add(c)

    print(f"Found active classes: {active_classes}")

    # -------------------------------------------------
    # 4. 过滤并复制其他所有 Force Section
    # -------------------------------------------------
    # 这一步会遍历 root 下除 Residues/AtomTypes 以外的所有子节点 (ForceSections)
    # 并保留所有属性中引用的 type 或 class 在 active 集合中的子元素
    
    for section in root:
        if section.tag in ["Residues", "AtomTypes"]:
            continue
            
        # 创建新的 Force 节点
        new_section = etree.SubElement(new_root, section.tag, section.attrib)
        
        count = 0
        for elem in section:
            # 检查该元素是否需要保留
            # 规则：
            # 1. 如果包含 type/type1/type2... 属性，所有出现的 type 必须都在 active_types 中
            # 2. 如果包含 class/class1/class2... 属性，所有出现的 class 必须都在 active_classes 中
            
            has_constraints = False
            
            # 获取所有属性
            attribs = elem.attrib
            
            # 检查 Types
            type_keys = [k for k in attribs.keys() if "type" in k.lower()]
            # 检查 Classes
            class_keys = [k for k in attribs.keys() if "class" in k.lower()]
            
            match_type = True
            if type_keys:
                has_constraints = True
                for k in type_keys:
                    if attribs[k] not in active_types:
                        match_type = False
                        break
            
            match_class = True
            if class_keys:
                has_constraints = True
                for k in class_keys:
                    if attribs[k] not in active_classes:
                        match_class = False
                        break
            
            # 只有当所有存在的约束都满足时才保留
            # 如果没有任何 type/class 约束，通常是全局参数或无需过滤的项，保留之
            if (not has_constraints) or (match_type and match_class):
                new_section.append(etree.fromstring(etree.tostring(elem)))
                count += 1
        
        # 如果该 Force section 没有任何子元素被保留，且它本身没有重要属性（通常不会），可以考虑移除
        if count == 0 and len(list(section)) > 0:
            # 如果原 section 有内容但过滤后为空，说明该力场项与选定残基无关，不添加
            new_root.remove(new_section)
        elif count > 0:
            print(f"Extracted {count} entries for {section.tag}")

    # -------------------------------------------------
    # 5. 格式化并保存
    # -------------------------------------------------
    import xml.dom.minidom
    rough_string = etree.tostring(new_root, encoding='utf-8')
    reparsed = xml.dom.minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="  ")
    
    # 移除多余空行
    clean_xml = '\n'.join([line for line in pretty_xml.split('\n') if line.strip()])
    
    with open(output_xml_path, "w", encoding="utf-8") as f:
        f.write(clean_xml)
    
    print(f"\n✅ Successfully extracted {target_residue_names} to {output_xml_path}")

if __name__ == "__main__":
    input_xml = "../caff_5_mpid_slater_bond.xml"
    # input_xml = "../example_Li_PF6_DMC-EC/phyneo_ecl.xml"
    
    # 示例 1: 提取单个 Residue (EC)
    # output_xml = "EC_extracted.xml"
    # residue_names = ["ECA"]
    
    # 示例 2: 提取多个 Residue (EC 和 DMC)
    output_xml = "EC_DMC_extracted.xml"
    residue_names = ["ECA", "DMC"]
    residue_names = ["ECA", "LiA"]
    
    if os.path.exists(input_xml):
        extract_residue_to_xml(input_xml, residue_names, output_xml)
    else:
        print(f"Error: Input file {input_xml} not found.")