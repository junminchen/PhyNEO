from lxml import etree
import xml.dom.minidom

def convert_forcefield(input_path, output_path):
    """
    将原力场XML文件转换为目标格式，保留所有ExternalBond信息
    
    参数:
        input_path: 原力场文件路径
        output_path: 转换后文件保存路径
    """
    # 1. 解析原XML文件
    tree = etree.parse(input_path)
    root = tree.getroot()
    
    # 2. 创建新的ForceField根节点
    new_root = etree.Element("ForceField")
    
    # 3. 处理AtomTypes部分
    atom_types = root.find("AtomTypes")
    new_atom_types = etree.SubElement(new_root, "AtomTypes")
    for type_elem in atom_types.findall("Type"):
        new_type = etree.SubElement(new_atom_types, "Type")
        new_type.set("name", type_elem.get("name"))
        new_type.set("class", type_elem.get("class"))
        new_type.set("element", type_elem.get("element"))
        new_type.set("mass", type_elem.get("mass"))
    
    # 4. 处理Residues部分（保留ExternalBond）
    residues = root.find("Residues")
    new_residues = etree.SubElement(new_root, "Residues")
    for residue in residues.findall("Residue"):
        new_res = etree.SubElement(new_residues, "Residue", {"name": residue.get("name")})
        
        # 复制原子信息
        for atom in residue.findall("Atom"):
            etree.SubElement(new_res, "Atom", {
                "name": atom.get("name"),
                "type": atom.get("type")
            })
        
        # 复制键信息
        for bond in residue.findall("Bond"):
            etree.SubElement(new_res, "Bond", {
                "from": bond.get("from"),
                "to": bond.get("to")
            })
        
        # 新增：复制ExternalBond信息（确保不丢失）
        for ext_bond in residue.findall("ExternalBond"):
            new_ext_bond = etree.SubElement(new_res, "ExternalBond")
            new_ext_bond.set("atomName", ext_bond.get("atomName"))
    
    # 5. 处理MPIDForce
    admppme = root.find("ADMPPmeForce")
    mpid_force = etree.SubElement(new_root, "MPIDForce")
    
    # 5.1 映射Multipole
    for atom in admppme.findall("Atom"):
        etree.SubElement(mpid_force, "Multipole", {
            "type": atom.get("type"),
            "kz": atom.get("kz"),
            "kx": atom.get("kx"),
            "c0": atom.get("c0"),
            "dX": atom.get("dX"),
            "dY": atom.get("dY"),
            "dZ": atom.get("dZ"),
            "qXX": atom.get("qXX"),
            "qXY": atom.get("qXY"),
            "qYY": atom.get("qYY"),
            "qXZ": atom.get("qXZ"),
            "qYZ": atom.get("qYZ"),
            "qZZ": atom.get("qZZ")

        })
    
    # 5.2 映射Polarize
    for polar in admppme.findall("Polarize"):
        etree.SubElement(mpid_force, "Polarize", {
            "type": polar.get("type"),
            "polarizabilityXX": polar.get("polarizabilityXX"),
            "polarizabilityYY": polar.get("polarizabilityYY"),
            "polarizabilityZZ": polar.get("polarizabilityZZ"),
            "thole": polar.get("thole")
        })
    
    # 6. 处理CustomNonbondedForce（按type定义）
    custom_nb = etree.SubElement(
        new_root, "CustomNonbondedForce",
        {"bondCutoff": "3", "energy": (
            "A*K2*exp(-B*r)+(-138.93542)*exp(-B*r)*(1+B*r)*Q/r - (1-exp(-x)*(1+x+0.5*x^2+x^3/6+x^4/24+x^5/120+x^6/720)) * C6/(r^6) - (1-exp(-x)*(1+x+0.5*x^2+x^3/6+x^4/24+x^5/120+x^6/720+x^7/5040+x^8/40320)) * C8/(r^8) - (1-exp(-x)*(1+x+0.5*x^2+x^3/6+x^4/24+x^5/120+x^6/720+x^7/5040+x^8/40320+x^9/362880+x^10/3628800)) * C10/(r^10);"
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
        )}
    )
    
    # 添加参数定义
    params = ["Aexch", "Aelec", "Aind", "Adhf", "Adisp", "Bexp", "Q", "C6", "C8", "C10", "C12"]
    for param in params:
        etree.SubElement(custom_nb, "PerParticleParameter", {"name": param})
    
    # 提取所有type
    all_types = [type_elem.get("name") for type_elem in new_atom_types.findall("Type")]
    
    # 按type提取参数
    slater_ex = {e.get("type"): e for e in root.find("SlaterExForce").findall("Atom")}
    slater_sres = {e.get("type"): e for e in root.find("SlaterSrEsForce").findall("Atom")}
    slater_srpol = {e.get("type"): e for e in root.find("SlaterSrPolForce").findall("Atom")}
    slater_dhf = {e.get("type"): e for e in root.find("SlaterDhfForce").findall("Atom")}
    slater_srdisp = {e.get("type"): e for e in root.find("SlaterSrDispForce").findall("Atom")}
    admpp_disp = {e.get("type"): e for e in root.find("ADMPDispPmeForce").findall("Atom")}
    
    # 为每个type创建参数节点
    for type_id in all_types:
        try:
            type_params = {
                "Aexch": slater_ex[type_id].get("A"),
                "Aelec": slater_sres[type_id].get("A"),
                "Aind": slater_srpol[type_id].get("A"),
                "Adhf": slater_dhf[type_id].get("A"),
                "Adisp": slater_srdisp[type_id].get("A"),
                "Bexp": slater_ex[type_id].get("B"),
                "Q": slater_sres[type_id].get("Q"),
                "C6": admpp_disp[type_id].get("C6"),
                "C8": admpp_disp[type_id].get("C8"),
                "C10": admpp_disp[type_id].get("C10"),
                "C12": "0.0"
            }
            
            atom_elem = etree.SubElement(custom_nb, "Atom", {"type": type_id})
            for key, value in type_params.items():
                atom_elem.set(key, value)
        
        except KeyError as e:
            raise ValueError(f"Type {type_id} 缺失参数: {e}")
    
    # 7. 格式化并保存
    rough_xml = etree.tostring(new_root, encoding="utf-8")
    pretty_xml = xml.dom.minidom.parseString(rough_xml).toprettyxml(indent="  ")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(pretty_xml)



def add_prefix_to_types_classes_and_names(xml_path, output_path, prefix="Z"):
    """
    为XML文件中所有type、class属性值，以及AtomTypes中Type节点的name属性值添加前缀
    
    参数:
        xml_path: 输入XML文件路径
        output_path: 输出XML文件路径
        prefix: 要添加的前缀（默认为"Z"）
    """
    # 解析XML文件
    tree = etree.parse(xml_path)
    root = tree.getroot()
    
    # 1. 处理AtomTypes中Type节点的name属性（单独处理，确保添加前缀）
    for type_elem in root.xpath("//AtomTypes/Type[@name]"):
        current_name = type_elem.get("name")
        type_elem.set("name", f"{prefix}{current_name}")
    
    # 2. 处理所有包含type属性的节点（如Atom、Multipole、Polarize等）
    for elem in root.xpath("//*[@type]"):
        current_type = elem.get("type")
        elem.set("type", f"{prefix}{current_type}")
    
    # 3. 处理所有包含class属性的节点（如Bond、Angle、Proper等）
    for elem in root.xpath("//*[@class] | //*[@class1] | //*[@class2] | //*[@class3] | //*[@class4]"):
        # 处理class1-class4等扩展属性
        for attr in ["class", "class1", "class2", "class3", "class4"]:
            if elem.get(attr) is not None:
                current_class = elem.get(attr)
                elem.set(attr, f"{prefix}{current_class}")
    
    # 保存修改后的XML
    tree.write(output_path, encoding="utf-8", xml_declaration=True, pretty_print=True)


if __name__ == "__main__":
    # convert_forcefield("peo.xml", "converted_forcefield.xml")
    # convert_forcefield("EC.xml", "converted_forcefield.xml")
    # convert_forcefield("EC.xml", "converted_forcefield.xml")
    # convert_forcefield("output.1.ABC.solvents.pospenalty.25.Aex.salts.xml", "converted_forcefield.xml")
    convert_forcefield("output.2.ABC.solvents.pospenalty.25.LiNa.AexAes.xml", "converted_forcefield.xml")

    # input_file = "../opls_bond/opls_salt.xml"  # 替换为你的输入文件路径
    # output_file = "../opls_bond/opls_salt_Z.xml"  # 替换为你的输出文件路径
    # add_prefix_to_types_classes_and_names(input_file, output_file, prefix="Z")
    # input_file = "../opls_bond/opls_solvent.xml"  # 替换为你的输入文件路径
    # output_file = "../opls_bond/opls_solvent_Z.xml"  # 替换为你的输出文件路径
    # add_prefix_to_types_classes_and_names(input_file, output_file, prefix="Z")



#  axisType = mm.AmoebaMultipoleForce.ZThenX 
#  if (kz == 0): 
#      axisType = mm.AmoebaMultipoleForce.NoAxisType 
#  if (kz != 0 and kx == 0): 
#      axisType = mm.AmoebaMultipoleForce.ZOnly 
#  if (kz < 0 or kx < 0): 
#      axisType = mm.AmoebaMultipoleForce.Bisector 
#  if (kx < 0 and ky < 0): 
#      axisType = mm.AmoebaMultipoleForce.ZBisect 
#  if (kz < 0 and kx < 0 and ky  < 0): 
#      axisType = mm.AmoebaMultipoleForce.ThreeFold 


#         kz = kIndices[0]
#         kzNegative = False
#         if kz.startswith('-'):
#             kz = kz[1:]
#             kzNegative = True

#         axisType = ZThenX
#         if (not kz):
#             axisType = NoAxisType
#         if (kz and not kx):
#             axisType = ZOnly
#         if (kz and kzNegative or kx and kxNegative):
#             axisType = Bisector
#         if (kx and kxNegative and ky and kyNegative):
#             axisType = ZBisect
#         if (kz and kzNegative and kx and kxNegative and ky and kyNegative):
#             axisType = ThreeFold

