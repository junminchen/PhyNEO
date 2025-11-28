from lxml import etree
import os

def merge_forcefields_generic(A_path, B_path, target_residues, output_path):
    """
    通用力场XML合并工具：将B中指定残基的力场参数转移到A，支持任意力场节点类型
    
    参数:
        A_path: 主文件路径（保留其结构，合并后输出）
        B_path: 待提取参数的文件路径
        target_residues: 需要匹配的残基名称列表（如["TER", "INT"]）
        output_path: 合并后文件保存路径
    """
    # --------------------------
    # 步骤1: 解析XML文件
    # --------------------------
    tree_A = etree.parse(A_path)
    root_A = tree_A.getroot()
    tree_B = etree.parse(B_path)
    root_B = tree_B.getroot()


    # --------------------------
    # 步骤2: 处理A的AtomTypes，使class与type（name）一致
    # --------------------------
    atom_types_A = root_A.find("AtomTypes")
    a_type_to_original_class = {}  # 记录A中type与原始class的映射（用于后续验证）
    for type_elem in atom_types_A.findall("Type"):
        type_name = type_elem.get("name")  # A的type（如"1"、"Li+"）
        original_class = type_elem.get("class")
        a_type_to_original_class[type_name] = original_class
        type_elem.set("class", type_name)  # class = type


    # --------------------------
    # 步骤3: 建立B的type到A的type的映射（通过原子名称匹配）
    # --------------------------
    # 提取A中目标残基的原子名称→type映射
    a_res_atom_map = {res: {} for res in target_residues}  # {残基: {原子名: A的type}}
    for res_name in target_residues:
        a_res = root_A.xpath(f'//Residue[@name="{res_name}"]')[0]
        for atom in a_res.findall("Atom"):
            a_res_atom_map[res_name][atom.get("name")] = atom.get("type")

    # 提取B中目标残基的原子名称→type映射
    b_res_atom_map = {res: {} for res in target_residues}  # {残基: {原子名: B的type}}
    for res_name in target_residues:
        b_res = root_B.xpath(f'//Residue[@name="{res_name}"]')[0]
        for atom in b_res.findall("Atom"):
            b_res_atom_map[res_name][atom.get("name")] = atom.get("type")

    # 建立B的type→A的type的映射（通过原子名称关联）
    b_type_to_a_type = {}
    for res_name in target_residues:
        for atom_name, b_type in b_res_atom_map[res_name].items():
            if atom_name in a_res_atom_map[res_name]:
                a_type = a_res_atom_map[res_name][atom_name]
                if b_type not in b_type_to_a_type:
                    b_type_to_a_type[b_type] = a_type


    # --------------------------
    # 步骤4: 建立B的class到A的type的映射
    # --------------------------
    # B的AtomTypes中type→class映射
    b_type_to_class = {
        elem.get("name"): elem.get("class") 
        for elem in root_B.find("AtomTypes").findall("Type")
    }

    # B的class→A的type映射（一个class可能对应多个A的type，取第一个）
    b_class_to_a_type = {}
    for b_type, a_type in b_type_to_a_type.items():
        b_class = b_type_to_class[b_type]
        if b_class not in b_class_to_a_type:
            b_class_to_a_type[b_class] = a_type  # 取第一个匹配的A的type


    # --------------------------
    # 步骤5: 处理B中所有力场节点，替换class为A的type
    # --------------------------
    def replace_class_attr(elem):
        """递归替换元素中所有class1/class2/class3/class4属性为A的type"""
        for attr in ["class1", "class2", "class3", "class4"]:
            if elem.get(attr) is not None:
                b_class = elem.get(attr)
                # 替换为A中对应的type（若B的class在映射中不存在则报错）
                if b_class not in b_class_to_a_type:
                    raise ValueError(f"B中class {b_class} 未找到对应的A的type")
                elem.set(attr, b_class_to_a_type[b_class])
        # 递归处理子元素
        for child in elem:
            replace_class_attr(child)

    # 提取B中所有力场节点（排除AtomTypes和Residues，其余均视为力场参数节点）
    b_force_nodes = []
    for child in root_B:
        if child.tag not in ["AtomTypes", "Residues"]:  # 仅排除结构节点
            b_force_nodes.append(child)

    # 处理所有力场节点，替换class属性
    for node in b_force_nodes:
        replace_class_attr(node)


    # --------------------------
    # 步骤6: 合并力场节点到A（替换原有同类型节点）
    # --------------------------
    for b_node in b_force_nodes:
        node_tag = b_node.tag  # 力场节点标签（如"CustomBondForce"）
        # 移除A中已有的同类型节点（避免重复）
        existing_a_nodes = root_A.findall(node_tag)
        for a_node in existing_a_nodes:
            root_A.remove(a_node)
        # 添加处理后的B节点到A
        root_A.append(b_node)


    # --------------------------
    # 步骤7: 保存合并后的文件
    # --------------------------
    tree_A.write(
        output_path,
        encoding="utf-8",
        xml_declaration=True,
        pretty_print=True
    )
    print(f"通用合并完成，文件保存至: {output_path}")

# --------------------------
# 执行示例
# --------------------------
if __name__ == "__main__":
    A_xml = "converted_forcefield.xml"  # 第一个XML文件路径
    # B_xml = "peo_opls_bond.xml"  # 第二个XML文件路径
    # target_res = ["TER", "INT"]  # 目标残基

    B_xml = "opls_solvent_bond.xml"  # 第二个XML文件路径
    target_res = ["ECA"]  # 目标残基
    output_xml = "merged_forcefield.xml"  # 输出文件路径

    merge_forcefields_generic(A_xml, B_xml, target_res, output_xml)
