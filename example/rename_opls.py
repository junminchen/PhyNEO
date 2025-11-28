import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
from itertools import product
from copy import deepcopy
import numpy as np

TYPE_KEYS  = {"type", "type1", "type2", "type3", "type4"}
CLASS_KEYS = {"class", "class1", "class2", "class3", "class4"}

def _lname(tag):
    return tag.split('}')[-1]

def padding(i):
    """将数字转换为4位字符串"""
    return f"{i:03d}"

def indent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for e in elem:
            indent(e, level+1)
        if not e.tail or not e.tail.strip():
            e.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i

def find_first(root, name):
    for el in root.iter():
        if _lname(el.tag) == name:
            return el
    return None

def is_int_string(s):
    try:
        int(s)
        return True
    except Exception:
        return False

def process_forcefield(input_xml, output_xml, type_map_tsv=None):
    tree = ET.parse(input_xml)
    root = tree.getroot()

    atomtypes = find_first(root, "AtomTypes")
    residues  = find_first(root, "Residues")
    if atomtypes is None or residues is None:
        raise ValueError("必须包含 <AtomTypes> 和 <Residues> 节点")

    # 读取原 AtomTypes: old_type_name -> {element, mass, old_class(optional), sigma/epsilon(optional)}
    old_type_info = {}
    for t in atomtypes:
        if _lname(t.tag) != "Type":
            continue
        old_name = t.get("name")
        if not old_name:
            continue
        old_type_info[old_name] = {
            "element": t.get("element"),
            "mass": t.get("mass"),
            "class": t.get("class"),
            "sigma": t.get("sigma"),
            "epsilon": t.get("epsilon"),
        }

    # 先从原始 NonbondedForce 和 AtomTypes 收集 LJ 参数（旧 type/class -> (sigma, epsilon)）
    oldtype_LJ = {}
    oldclass_LJ = {}

    nb_orig = find_first(root, "NonbondedForce")
    if nb_orig is not None:
        for child in list(nb_orig):
            if _lname(child.tag) not in {"Atom", "Particle", "Type"}:
                continue
            t = child.get("type")
            c = child.get("class")
            sig = child.get("sigma")
            eps = child.get("epsilon")
            if sig is not None and eps is not None:
                try:
                    sigf = float(sig)
                    epsf = float(eps)
                except Exception:
                    continue
                if t and not is_int_string(t):
                    # 如果重复且冲突，可按需校验；此处直接覆盖为最后一次
                    oldtype_LJ[t] = (sigf, epsf)
                if c and not is_int_string(c):
                    oldclass_LJ[c] = (sigf, epsf)

    # 从 AtomTypes 兜底（如果 Type 节点上自带 sigma/epsilon）
    for old_name, info in old_type_info.items():
        sig = info.get("sigma")
        eps = info.get("epsilon")
        if sig and eps:
            try:
                sigf = float(sig); epsf = float(eps)
                oldtype_LJ.setdefault(old_name, (sigf, epsf))
                cls = info.get("class")
                if cls:
                    oldclass_LJ.setdefault(cls, (sigf, epsf))
            except Exception:
                pass

    # 第一轮：按 Residues 顺序分配新 type（数字），并在 Residues 里就地更新 atom 的 type/class
    new_type_counter = 1
    res_oldtype_to_newid = OrderedDict()
    old_type_to_newids = defaultdict(list)
    old_class_to_newids = defaultdict(list)
    type_charge_sets = defaultdict(set)
    new_type_map = OrderedDict()

    for ires, res in enumerate(residues):
        if _lname(res.tag) != "Residue":
            continue
        resname = res.get("name") or f"R{ires+1}"
        local_map = OrderedDict()  # old_type within this residue -> new_id

        for atom in list(res):
            if _lname(atom.tag) != "Atom":
                continue
            old_type = atom.get("type")
            if not old_type:
                continue

            if old_type not in local_map:
                new_id = f"99{padding(int(new_type_counter))}"
                # new_id = str(new_type_counter)
                new_type_counter += 1
                local_map[old_type] = new_id
                res_oldtype_to_newid[(resname, old_type)] = new_id
                new_type_map[new_id] = (resname, old_type)
                old_type_to_newids[old_type].append(new_id)

                old_cls = old_type_info.get(old_type, {}).get("class")
                if old_cls:
                    old_class_to_newids[old_cls].append(new_id)

            new_id = local_map[old_type]
            # 更新 Residues 内 atom 的 type 和 class 为数字
            atom.set("type", new_id)
            atom.set("class", new_id)

            # 收集电荷并移除 Residues 上的 charge
            q = atom.get("charge")
            if q is not None:
                try:
                    qval = float(q)
                    type_charge_sets[new_id].add(qval)
                except Exception:
                    pass
                del atom.attrib["charge"]
    
    print(new_type_map)
    # 基于 old_type / old_class 为每个 new_id 选取 LJ 参数
    new_id_LJ = {}
    missing_LJ_new_ids = []
    for new_id, (_resname, old_type) in new_type_map.items():
        lj = None
        if old_type in oldtype_LJ:
            lj = oldtype_LJ[old_type]
        else:
            old_cls = old_type_info.get(old_type, {}).get("class")
            if old_cls and old_cls in oldclass_LJ:
                lj = oldclass_LJ[old_cls]
        if lj is not None:
            new_id_LJ[new_id] = lj
        else:
            missing_LJ_new_ids.append((new_id, old_type, old_type_info.get(old_type, {}).get("class")))

    if missing_LJ_new_ids:
        detail = ", ".join([f"{nid}(old_type={ot}, class={oc})" for nid, ot, oc in missing_LJ_new_ids])
        raise ValueError(f"下列新类型缺少 sigma/epsilon 参数，无法重建 NonbondedForce: {detail}")

    # 第二轮：重建 AtomTypes（name=class=数字，沿用 element/mass）
    new_atomtypes = ET.Element("AtomTypes")
    for new_id, (_resname, old_type) in new_type_map.items():
        tt = ET.Element("Type")
        tt.set("name", new_id)
        tt.set("class", new_id)
        info = old_type_info.get(old_type, {})
        if info.get("element"):
            tt.set("element", info["element"])
        if info.get("mass"):
            tt.set("mass", info["mass"])
        new_atomtypes.append(tt)

    # 替换 AtomTypes
    parent = root
    for idx, child in enumerate(list(parent)):
        if _lname(child.tag) == "AtomTypes":
            parent.remove(child)
            parent.insert(idx, new_atomtypes)
            break
    else:
        parent.insert(0, new_atomtypes)

    # 第三轮：展开并替换其他 Force 段的旧 type/class 引用为数字（跳过 NonbondedForce）
    def expand_node_inplace(container):
        for node in list(container):
            expand_node_inplace(node)
            if _lname(node.tag) in {"AtomTypes", "Residues"}:
                continue
            if _lname(container.tag) == "NonbondedForce":
                continue

            replace_keys = []
            replace_value_lists = []
            for attr, val in list(node.attrib.items()):
                key = attr.lower()
                if key in TYPE_KEYS and val in old_type_to_newids:
                    replace_keys.append(attr)
                    replace_value_lists.append(old_type_to_newids[val])
                elif key in CLASS_KEYS and val in old_class_to_newids:
                    replace_keys.append(attr)
                    replace_value_lists.append(old_class_to_newids[val])

            if not replace_keys:
                continue

            combos = list(product(*replace_value_lists))
            insert_pos = list(container).index(node)
            for combo in combos:
                clone = deepcopy(node)
                for k, newv in zip(replace_keys, combo):
                    clone.set(k, newv)
                container.insert(insert_pos, clone)
                insert_pos += 1
            container.remove(node)

    for top in list(root):
        if _lname(top.tag) in {"AtomTypes", "Residues"}:
            continue
        expand_node_inplace(top)

    # 第四轮：重建/更新 NonbondedForce，写入 charge + sigma + epsilon
    nb = find_first(root, "NonbondedForce")
    if nb is None:
        nb = ET.SubElement(root, "NonbondedForce")

    for child in list(nb):
        if _lname(child.tag) == "UseAttributeFromResidue":
            nb.remove(child)
    # 清掉旧的以名字引用的条目（非数字的 type/class）
    for child in list(nb):
        if _lname(child.tag) in {"Atom", "Particle", "Type"}:
            tval = child.get("type")
            cval = child.get("class")
            if (tval and not is_int_string(tval)) or (cval and not is_int_string(cval)):
                nb.remove(child)

    # 读取现存的数字条目，避免重复
    existing = {}
    for child in nb:
        if _lname(child.tag) in {"Atom", "Particle", "Type"}:
            t = child.get("type") or child.get("class")
            if t and is_int_string(t):
                existing[t] = child

    # 写入/更新 charge + LJ
    for new_id, qset in type_charge_sets.items():
        if len(qset) == 0:
            # 没有 Residues 电荷信息；此处置零（也可选择跳过/报错）
            q = 0.0
        elif len(qset) > 1:
            q = float(np.mean(list(qset)))
            # 如需严格一致可改为 raise
        else:
            q = next(iter(qset))

        if new_id not in new_id_LJ:
            raise ValueError(f"new_id={new_id} 缺少 LJ 参数，逻辑不应到达此处。")

        sigma, epsilon = new_id_LJ[new_id]

        if new_id in existing:
            existing[new_id].set("charge", f"{q:.6f}")
            existing[new_id].set("sigma",  f"{sigma:.6f}")
            existing[new_id].set("epsilon",f"{epsilon:.6f}")
        else:
            el = ET.Element("Atom")
            el.set("type", new_id)
            el.set("charge", f"{q:.6f}")
            el.set("sigma",  f"{sigma:.6f}")
            el.set("epsilon",f"{epsilon:.6f}")
            nb.append(el)

    indent(root)
    tree.write(output_xml, encoding="utf-8", xml_declaration=True)

    if type_map_tsv:
        with open(type_map_tsv, "w", encoding="utf-8") as f:
            f.write("new_type_id\tresidue_name\told_type_name\n")
            for new_id, (resname, old_type) in new_type_map.items():
                f.write(f"{new_id}\t{resname}\t{old_type}\n")


def merge_forcefield_xml(file1, file2, output_file):
    tree1 = ET.parse(file1)
    tree2 = ET.parse(file2)
    root1 = tree1.getroot()
    root2 = tree2.getroot()

    # 合并 AtomTypes（允许相同 class）
    types1 = find_first(root1, "AtomTypes")
    types2 = find_first(root2, "AtomTypes")
    for t in types2:
        if _lname(t.tag) == "Type":
            types1.append(deepcopy(t))

    # 合并 Residues（按 residue name 去重）
    residues1 = find_first(root1, "Residues")
    residues2 = find_first(root2, "Residues")
    existing_resnames = set()
    for r in residues1:
        if _lname(r.tag) == "Residue":
            name = r.get("name")
            if name:
                existing_resnames.add(name)
    for r in residues2:
        if _lname(r.tag) == "Residue":
            name = r.get("name")
            if name and name not in existing_resnames:
                residues1.append(deepcopy(r))
                existing_resnames.add(name)

    # 合并其他 Force 段（除了 AtomTypes 和 Residues）
    tag_exclude = {"AtomTypes", "Residues"}
    for top2 in root2:
        lname2 = _lname(top2.tag)
        if lname2 in tag_exclude:
            continue
        match1 = None
        for top1 in root1:
            if _lname(top1.tag) == lname2:
                match1 = top1
                break
        if match1 is None:
            root1.append(deepcopy(top2))
        else:
            for child in top2:
                match1.append(deepcopy(child))

    # 缩进并输出
    indent(root1)
    tree1.write(output_file, encoding="utf-8", xml_declaration=True)


# 用法：
process_forcefield(
    input_xml="opls_salt.xml",
    output_xml="renamed.xml",
    type_map_tsv="type_map.tsv"
)


# 🎯 用法示例：
merge_forcefield_xml("renamed.xml", "opls_solvent.xml", "merged_opls.xml")
