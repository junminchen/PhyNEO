import xml.etree.ElementTree as ET
import sys

def zero_nonbonded_params(input_file, output_file):
    try:
        # 解析 XML 文件
        # 为了保留原始文件的注释和格式，通常使用 ET 可能会有些简化
        # 但对于修改属性来说，这是最标准的方法
        tree = ET.parse(input_file)
        root = tree.getroot()

        # 找到 NonbondedForce 节点
        # 注意：有些 XML 可能有命名空间，但在 OpenMM XML 中通常没有
        nonbonded_force = root.find(".//NonbondedForce")
        
        if nonbonded_force is None:
            print(f"Error: 在 {input_file} 中未找到 <NonbondedForce> 标签。")
            return

        cnt = 0
        # 遍历 NonbondedForce 下的所有 Atom 标签
        for atom in nonbonded_force.findall("Atom"):
            atom.set('charge', '0.000000')
            atom.set('sigma', '1.000000')
            atom.set('epsilon', '0.000000')
            cnt += 1

        # 保存修改后的文件
        tree.write(output_file, encoding='utf-8', xml_declaration=True)
        print(f"成功处理了 {cnt} 个原子。结果已保存至: {output_file}")

    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python zero_nonbonded.py <输入文件名> [输出文件名]")
        print("示例: python zero_nonbonded.py phyneo_ecl_z_b.xml phyneo_ecl_om.xml")
    else:
        file_in = sys.argv[1]
        file_out = sys.argv[2] if len(sys.argv) > 2 else "zeroed_" + file_in
        zero_nonbonded_params(file_in, file_out)