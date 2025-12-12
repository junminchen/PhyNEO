import sys
import os
from lxml import etree
import numpy as np
from openmm import *
from openmm.app import *
from openmm.unit import *

# Try to import mpidplugin
try:
    import mpidplugin
    HAS_MPID = True
except ImportError:
    print("Warning: mpidplugin not found. MPIDForce will not be available.")
    HAS_MPID = False

class OpenMMEnergyCalculator:
    def __init__(self, pdb_file, xml_file, bond_cutoff=0.9*nanometers, default_thole_width=8.0, coulomb_scale14=0.5):
        self.pdb_file = pdb_file
        self.xml_file = xml_file
        self.bond_cutoff = bond_cutoff
        self.default_thole_width = default_thole_width
        self.coulomb_scale14 = coulomb_scale14
        
        self.pdb = PDBFile(pdb_file)
        self.topology = self.pdb.topology
        self.positions = self.pdb.positions
        
        # Parse XML
        self.xml_root = etree.parse(xml_file).getroot()
        
        # Build System
        self.system = self._build_system()
        
        # Initialize Simulation
        self.integrator = VerletIntegrator(0.001*picoseconds)
        # self.platform = Platform.getPlatformByName('CUDA') # Switch to CUDA if available
        self.platform = Platform.getPlatformByName('Reference')
        
        self.simulation = Simulation(self.topology, self.system, self.integrator, self.platform)
        self.simulation.context.setPositions(self.positions)

    def _build_system(self):
        system = System()
        if self.topology.getPeriodicBoxVectors():
            system.setDefaultPeriodicBoxVectors(*self.topology.getPeriodicBoxVectors())
        
        for atom in self.topology.atoms():
            system.addParticle(atom.element.mass)

        # 1. CustomNonbondedForce
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
        
        custom_nb = CustomNonbondedForce(energy_expression)
        params_list = ["Aexch", "Aelec", "Aind", "Adhf", "Adisp", "Bexp", "Q", "C6", "C8", "C10"]
        for p in params_list:
            custom_nb.addPerParticleParameter(p)
            
        custom_nb.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
        custom_nb.setCutoffDistance(self.bond_cutoff)
        custom_nb.setForceGroup(1)

        # FIX: Convert bond objects to list of tuples (index1, index2)
        bonds_indices = [(b.atom1.index, b.atom2.index) for b in self.topology.bonds()]
        custom_nb.createExclusionsFromBonds(bonds_indices, 3)

        # 2. MPIDForce
        mpid_force = None
        if HAS_MPID:
            mpid_force = mpidplugin.MPIDForce()
            mpid_force.setNonbondedMethod(mpidplugin.MPIDForce.PME)
            mpid_force.setForceGroup(2)

        # Prepare Mappings
        atom_types_map = {}
        for res in self.xml_root.find("Residues").findall("Residue"):
            r_name = res.get("name")
            for atom in res.findall("Atom"):
                atom_types_map[(r_name, atom.get("name"))] = atom.get("type")

        def get_params_dict(tag_name):
            node = self.xml_root.find(tag_name)
            return {e.get("type"): e for e in node.findall("Atom")} if node is not None else {}

        def get_polarize_dict(tag_name):
            node = self.xml_root.find(tag_name)
            return {e.get("type"): e for e in node.findall("Polarize")} if node is not None else {}

        # Load parameter dictionaries
        slater_ex = get_params_dict("SlaterExForce")
        slater_sres = get_params_dict("SlaterSrEsForce")
        slater_srpol = get_params_dict("SlaterSrPolForce")
        slater_dhf = get_params_dict("SlaterDhfForce")
        slater_srdisp = get_params_dict("SlaterSrDispForce")
        admpp_disp = get_params_dict("ADMPDispPmeForce")
        
        mpid_atoms = get_params_dict("ADMPPmeForce") or get_params_dict("MPIDForce")
        mpid_polars = get_polarize_dict("ADMPPmeForce") or get_polarize_dict("MPIDForce")

        for atom in self.topology.atoms():
            res_name = atom.residue.name
            atom_name = atom.name
            type_id = atom_types_map.get((res_name, atom_name))
            
            if not type_id:
                print(f"Warning: Type not found for {res_name}-{atom_name}")
                custom_nb.addParticle([0.0]*10)
                if mpid_force:
                    mpid_force.addMultipole(0.0, [0.0]*3, [0.0]*6, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0)
                continue

            # Add CustomNonbonded params
            try:
                c_params = [
                    float(slater_ex[type_id].get("A")),
                    float(slater_sres[type_id].get("A")),
                    float(slater_srpol[type_id].get("A")),
                    float(slater_dhf[type_id].get("A")),
                    float(slater_srdisp[type_id].get("A")),
                    float(slater_ex[type_id].get("B")),
                    float(slater_sres[type_id].get("Q")),
                    float(admpp_disp[type_id].get("C6")),
                    float(admpp_disp[type_id].get("C8")),
                    float(admpp_disp[type_id].get("C10"))
                ]
                custom_nb.addParticle(c_params)
            except KeyError as e:
                raise ValueError(f"Missing CustomNonbonded params for {type_id}: {e}")

            # Add MPID params
            if mpid_force:
                try:
                    p_mpid = mpid_atoms[type_id]
                    p_polar = mpid_polars[type_id]

                    kz = int(p_mpid.get("kz", 0))
                    kx = int(p_mpid.get("kx", 0))
                    ky = int(p_mpid.get("ky", 0))

                    axis_type = mpidplugin.MPIDForce.ZThenX
                    if kz == 0: axis_type = mpidplugin.MPIDForce.NoAxisType
                    elif kz != 0 and kx == 0: axis_type = mpidplugin.MPIDForce.ZOnly
                    elif kz < 0 or kx < 0: axis_type = mpidplugin.MPIDForce.Bisector 
                    # Note: Simplified axis logic; expand if needed based on 0_modify_admp_to_mpid.py
                    
                    # Use default thole if not present
                    thole_val = float(p_polar.get("thole", self.default_thole_width))

                    mpid_force.addMultipole(
                        float(p_mpid.get("c0")),
                        [float(p_mpid.get("dX")), float(p_mpid.get("dY")), float(p_mpid.get("dZ"))],
                        [float(p_mpid.get("qXX")), float(p_mpid.get("qXY")), float(p_mpid.get("qYY")),
                         float(p_mpid.get("qXZ")), float(p_mpid.get("qYZ")), float(p_mpid.get("qZZ"))],
                        axis_type,
                        abs(kz)-1 if kz != 0 else 0,
                        abs(kx)-1 if kx != 0 else 1,
                        abs(ky)-1 if ky != 0 else 2,
                        thole_val,
                        float(p_polar.get("polarizabilityXX")),
                        float(p_polar.get("polarizabilityYY")),
                        float(p_polar.get("polarizabilityZZ"))
                    )
                except KeyError as e:
                    raise ValueError(f"Missing MPID params for {type_id}: {e}")

        system.addForce(custom_nb)
        if mpid_force:
            system.addForce(mpid_force)
            
        return system

    def compute_energy(self):
        state = self.simulation.context.getState(getEnergy=True)
        total_energy = state.getPotentialEnergy()
        print(f"Total Energy: {total_energy.value_in_unit(kilojoules_per_mole):.4f} kJ/mol")
        
        # Decompose
        groups = {1: "CustomNonbonded", 2: "MPIDForce"}
        for gid, name in groups.items():
            try:
                e = self.simulation.context.getState(getEnergy=True, groups={gid}).getPotentialEnergy()
                print(f"  {name}: {e.value_in_unit(kilojoules_per_mole):.4f} kJ/mol")
            except: pass
        return total_energy

    def minimize(self):
        print("\n--- Minimization ---")
        self.compute_energy()
        self.simulation.minimizeEnergy()
        print("Minimized:")
        self.compute_energy()
        
        PDBFile.writeFile(self.topology, 
                          self.simulation.context.getState(getPositions=True).getPositions(), 
                          open("minimized.pdb", 'w'))
        print("Saved minimized.pdb")

if __name__ == "__main__":
    pdb_arg = "dimer_bank/dimer_003_EC_EC.pdb"
    xml_arg = "ff_dmff_EC.xml" 
    
    # Simple argument parsing if run from command line
    if len(sys.argv) > 1: pdb_arg = sys.argv[1]
    if len(sys.argv) > 2: xml_arg = sys.argv[2]
    
    if os.path.exists(pdb_arg) and os.path.exists(xml_arg):
        calc = OpenMMEnergyCalculator(
            pdb_arg, 
            xml_arg, 
            bond_cutoff=0.9*nanometers, 
            default_thole_width=8.0,
            coulomb_scale14=0.5
        )
        calc.minimize()
    else:
        print(f"File not found: {pdb_arg} or {xml_arg}")