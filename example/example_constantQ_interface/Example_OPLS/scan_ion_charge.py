import xml.etree.ElementTree as ET

scale = 0.7
tree = ET.parse("opls_salt.xml")   # 你的文件名
root = tree.getroot()

# 遍历所有 Residue 下的 Atom 标签
for atom in root.findall(".//Residue/Atom"):
    if "charge" in atom.attrib:
        old_charge = float(atom.attrib["charge"])
        new_charge = old_charge * scale
        atom.set("charge", f"{new_charge:.6f}")  # 保留6位小数

# 保存到新文件
tree.write("opls_salt_scaled.xml", encoding="utf-8", xml_declaration=True)
