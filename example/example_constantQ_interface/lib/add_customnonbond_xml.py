import lxml.etree as ET

# this subroutine adds atom class parameters to an existing CustomNonbondedForce in an xml file, assuming it's of the SAPT-FF functional form
#      xml_base              : Existing xml file to add parameters to
#      xml_param             : parameters to add, in xml format ..
#      xml_combine           : new xml file to be written with combined FF parameters
def add_CustomNonbondedForce_SAPTFF_parameters( xml_base , xml_param , xml_combine ):
    # xml tree of base .xml force field file 
    tree_base= ET.parse(xml_base)
    root_base = tree_base.getroot()
    customnonbond_base = root_base.find('CustomNonbondedForce')

    # xml tree of parameter "to add" xml file
    tree_param = ET.parse(xml_param)
    root_param = tree_param.getroot()

    # get parameters to add .., make a new etree Element for each atom
    customnonbond = root_param.find('CustomNonbondedForce')
    #loop over atoms in customnonbond section and pull parameters
    element_list = []
    for child in customnonbond:
        #print( child.attrib['class'] , child.attrib['Aexch'], child.attrib['Aelec'], child.attrib['Aind'], child.attrib['Adhf'], child.attrib['Bexp'], child.attrib['C6'], child.attrib['C8'], child.attrib['C10'], child.attrib['C12'] )
        # get parameters for this atom class
        atom_class = child.attrib['class'] ; Aexch = child.attrib['Aexch'] ; Aelec = child.attrib['Aelec'] ; Aind = child.attrib['Aind'] ; Adhf = child.attrib['Adhf']
        Bexp = child.attrib['Bexp'] ; C6 = child.attrib['C6'] ; C8 = child.attrib['C8'] ; C10 = child.attrib['C10'] ; C12 = child.attrib['C12']

        # make a new element for this atom class and add FF parameters as attributes
        element_new = ET.Element("Atom")
        element_new.attrib["class"] = atom_class
        element_new.attrib["Aexch"] = Aexch
        element_new.attrib["Aelec"] = Aelec
        element_new.attrib["Aind"] = Aind
        element_new.attrib["Adhf"] = Adhf
        element_new.attrib["Bexp"] = Bexp
        element_new.attrib["C6"] = C6
        element_new.attrib["C8"] = C8
        element_new.attrib["C10"] = C10
        element_new.attrib["C12"] = C12 
        element_new.tail = "\n"

        # add to existing xml file customnonbond section
        customnonbond_base.append( element_new )

    # now write new xml file
    tree_base.write( xml_combine)
     
