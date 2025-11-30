from lxml import etree
import xml.dom.minidom
import os

def convert_forcefield(input_path, output_path):
    """
    Convert original forcefield XML to target MPID format, preserving all ExternalBond information.
    Completes the previously truncated CustomNonbondedForce energy expression.
    
    Args:
        input_path: Path to source forcefield file (e.g., EC_DMC_extracted.xml)
        output_path: Path to save converted file (e.g., EC_DMC_mpid.xml)
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found.")
        return

    print(f"Reading {input_path}...")
    # 1. Parse original XML file
    tree = etree.parse(input_path)
    root = tree.getroot()
    
    # 2. Create new ForceField root element
    new_root = etree.Element("ForceField")
    
    # 3. Process AtomTypes section
    atom_types = root.find("AtomTypes")
    new_atom_types = etree.SubElement(new_root, "AtomTypes")
    if atom_types is not None:
        for type_elem in atom_types.findall("Type"):
            new_type = etree.SubElement(new_atom_types, "Type")
            new_type.set("name", type_elem.get("name"))
            new_type.set("class", type_elem.get("class"))
            new_type.set("element", type_elem.get("element"))
            new_type.set("mass", type_elem.get("mass"))
    
    # 4. Process Residues section (preserve ExternalBond)
    residues = root.find("Residues")
    new_residues = etree.SubElement(new_root, "Residues")
    if residues is not None:
        for residue in residues.findall("Residue"):
            new_res = etree.SubElement(new_residues, "Residue", {"name": residue.get("name")})
            
            # Copy atom information
            for atom in residue.findall("Atom"):
                etree.SubElement(new_res, "Atom", {
                    "name": atom.get("name"),
                    "type": atom.get("type")
                })
            
            # Copy bond information
            for bond in residue.findall("Bond"):
                etree.SubElement(new_res, "Bond", {
                    "from": bond.get("from"),
                    "to": bond.get("to")
                })
            
            # New: Copy ExternalBond information (ensure not lost)
            for ext_bond in residue.findall("ExternalBond"):
                new_ext_bond = etree.SubElement(new_res, "ExternalBond")
                new_ext_bond.set("atomName", ext_bond.get("atomName"))
    
    # 5. Process MPIDForce
    admppme = root.find("ADMPPmeForce")
    if admppme is not None:
        mpid_force = etree.SubElement(new_root, "MPIDForce")
        
        # 5.1 Map Multipole
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
        
        # 5.2 Map Polarize
        for polar in admppme.findall("Polarize"):
            etree.SubElement(mpid_force, "Polarize", {
                "type": polar.get("type"),
                "polarizabilityXX": polar.get("polarizabilityXX"),
                "polarizabilityYY": polar.get("polarizabilityYY"),
                "polarizabilityZZ": polar.get("polarizabilityZZ"),
                "thole": polar.get("thole")
            })
    
    # 6. Process CustomNonbondedForce (defined by type)
    # The energy expression here completes C8 and C10 terms.
    # Tang-Toennies damping term expansion coefficients:
    # C6: 1, x, x^2/2, x^3/6, x^4/24, x^5/120, x^6/720
    # C8: ... + x^7/5040 + x^8/40320
    # C10: ... + x^9/362880 + x^10/3628800
    
    energy_expression = (
        "A*K2*exp(-B*r)+(-138.93542)*exp(-B*r)*(1+B*r)*Q/r "
        "- (1-exp(-x)*(1+x+0.5*x^2+x^3/6+x^4/24+x^5/120+x^6/720)) * C6/(r^6) "
        "- (1-exp(-x)*(1+x+0.5*x^2+x^3/6+x^4/24+x^5/120+x^6/720+x^7/5040+x^8/40320)) * C8/(r^8) "
        "- (1-exp(-x)*(1+x+0.5*x^2+x^3/6+x^4/24+x^5/120+x^6/720+x^7/5040+x^8/40320+x^9/362880+x^10/3628800)) * C10/(r^10);"
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

    custom_nb = etree.SubElement(
        new_root, "CustomNonbondedForce",
        {"bondCutoff": "3", "energy": energy_expression}
    )
    
    # Add parameter definitions
    params = ["Aexch", "Aelec", "Aind", "Adhf", "Adisp", "Bexp", "Q", "C6", "C8", "C10", "C12"]
    for param in params:
        etree.SubElement(custom_nb, "PerParticleParameter", {"name": param})
    
    # Extract all types
    all_types = []
    if new_atom_types is not None:
        all_types = [type_elem.get("name") for type_elem in new_atom_types.findall("Type")]
    
    # Try to get each Force node, return empty dict if not present
    def get_atoms_dict(force_tag):
        node = root.find(force_tag)
        if node is not None:
            return {e.get("type"): e for e in node.findall("Atom")}
        return {}

    slater_ex = get_atoms_dict("SlaterExForce")
    slater_sres = get_atoms_dict("SlaterSrEsForce")
    slater_srpol = get_atoms_dict("SlaterSrPolForce")
    slater_dhf = get_atoms_dict("SlaterDhfForce")
    slater_srdisp = get_atoms_dict("SlaterSrDispForce")
    admpp_disp = get_atoms_dict("ADMPDispPmeForce")
    
    # Create parameter nodes for each type
    for type_id in all_types:
        try:
            # Check if required parameters exist
            if type_id not in slater_ex:
                print(f"Warning: Type {type_id} missing in SlaterExForce, skipping or filling default 0.")
            
            type_params = {
                "Aexch": slater_ex.get(type_id, {}).get("A", "0"),
                "Aelec": slater_sres.get(type_id, {}).get("A", "0"),
                "Aind": slater_srpol.get(type_id, {}).get("A", "0"),
                "Adhf": slater_dhf.get(type_id, {}).get("A", "0"),
                "Adisp": slater_srdisp.get(type_id, {}).get("A", "0"),
                "Bexp": slater_ex.get(type_id, {}).get("B", "0"),
                "Q": slater_sres.get(type_id, {}).get("Q", "0"),
                "C6": admpp_disp.get(type_id, {}).get("C6", "0"),
                "C8": admpp_disp.get(type_id, {}).get("C8", "0"),
                "C10": admpp_disp.get(type_id, {}).get("C10", "0"),
                "C12": "0.0"
            }
            
            atom_elem = etree.SubElement(custom_nb, "Atom", {"type": type_id})
            for key, value in type_params.items():
                atom_elem.set(key, value)
        
        except KeyError as e:
            print(f"Error processing type {type_id}: {e}")
    
    # 7. Format and save
    rough_xml = etree.tostring(new_root, encoding="utf-8")
    reparsed = xml.dom.minidom.parseString(rough_xml)
    pretty_xml = reparsed.toprettyxml(indent="  ")
    
    # Remove extra blank lines
    clean_xml = '\n'.join([line for line in pretty_xml.split('\n') if line.strip()])
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(clean_xml)
    
    print(f"✅ Successfully converted to MPID format: {output_path}")


if __name__ == "__main__":
    # Default to processing the EC_DMC_extracted.xml generated in the previous step
    input_xml = "EC_Li_extracted.xml"
    output_xml = "EC_Li_mpid.xml"
    
    convert_forcefield(input_xml, output_xml)