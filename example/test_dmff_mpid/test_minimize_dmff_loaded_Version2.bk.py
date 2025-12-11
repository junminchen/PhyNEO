import sys
import os
from lxml import etree
import numpy as np
from openmm import *
from openmm.app import *
from openmm.unit import *

# 尝试导入 mpidplugin
try:
    import mpidplugin
    HAS_MPID = True
except ImportError:
    print("Warning: mpidplugin not found. MPIDForce will not be available.")
    HAS_MPID = False

class OpenMMEnergyCalculator:
    def __init__(self, pdb_file, xml_file):
        self.pdb_file = pdb_file
        self.xml_file = xml_file
        self.pdb = PDBFile(pdb_file)
        self.topology = self.pdb.topology
        self.positions = self.pdb.positions
        
        # 解析 XML 获取参数
        self.xml_root = etree.parse(xml_file).getroot()
        
        # 构建 OpenMM System
        self.system = self._build_system()
        
        # 初始化 Simulation (Context)
        self.integrator = VerletIntegrator(0.001*picoseconds)
        self.platform = Platform.getPlatformByName('Reference')  # 精度较高，适合调试
        # self.platform = Platform.getPlatformByName('CUDA')     # 如果有 GPU
        
        self.simulation = Simulation(self.topology, self.system, self.integrator, self.platform)
        self.simulation.context.setPositions(self.positions)

    def _build_system(self):
        system = System()
        if self.topology.getPeriodicBoxVectors():
            system.setDefaultPeriodicBoxVectors(*self.topology.getPeriodicBoxVectors())
        
        # 添加粒子
        for atom in self.topology.atoms():
            system.addParticle(atom.element.mass)

        # 1. 创建 CustomNonbondedForce
        # 能量表达式来自 example/0_modify_admp_to_mpid.py
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
        # 添加每个粒子的参数
        params_list = ["Aexch", "Aelec", "Aind", "Adhf", "Adisp", "Bexp", "Q", "C6", "C8", "C10"]
        for p in params_list:
            custom_nb.addPerParticleParameter(p)
            
        custom_nb.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
        custom_nb.setCutoffDistance(1.2*nanometer)
        custom_nb.setForceGroup(1) # 设置 Force Group 方便后续查看分项能量

        # 2. 创建 MPIDForce
        mpid_force = None
        if HAS_MPID:
            mpid_force = mpidplugin.MPIDForce()
            mpid_force.setNonbondedMethod(mpidplugin.MPIDForce.PME)
            mpid_force.setForceGroup(2) # 设置 Force Group

        # 3. 准备参数映射 (Type -> Params)
        # 解析 Residues 获取 (ResName, AtomName) -> Type
        atom_types_map = {}
        for res in self.xml_root.find("Residues").findall("Residue"):
            r_name = res.get("name")
            for atom in res.findall("Atom"):
                atom_types_map[(r_name, atom.get("name"))] = atom.get("type")

        # 提取 XML 中的参数块
        def get_params_dict(tag_name):
            node = self.xml_root.find(tag_name)
            if node is None: return {}
            return {e.get("type"): e for e in node.findall("Atom")}

        def get_polarize_dict(tag_name):
            node = self.xml_root.find(tag_name)
            if node is None: return {}
            return {e.get("type"): e for e in node.findall("Polarize")}

        # CustomNonbondedForce 相关参数
        slater_ex = get_params_dict("SlaterExForce")
        slater_sres = get_params_dict("SlaterSrEsForce")
        slater_srpol = get_params_dict("SlaterSrPolForce")
        slater_dhf = get_params_dict("SlaterDhfForce")
        slater_srdisp = get_params_dict("SlaterSrDispForce")
        admpp_disp = get_params_dict("ADMPDispPmeForce")
        
        # MPIDForce 相关参数 (注意 XML 这里的标签可能也是 ADMPPmeForce，需根据你的实际 XML 调整)
        mpid_atoms = get_params_dict("ADMPPmeForce") 
        if not mpid_atoms: 
             mpid_atoms = get_params_dict("MPIDForce") # 尝试查找 MPIDForce 标签
        
        # Polarize 参数通常在同一个 Force 块下
        mpid_polars = get_polarize_dict("ADMPPmeForce")
        if not mpid_polars:
            mpid_polars = get_polarize_dict("MPIDForce")


        # 4. 遍历拓扑，填充参数
        for atom in self.topology.atoms():
            res_name = atom.residue.name
            atom_name = atom.name
            
            # 查找 Type
            type_id = atom_types_map.get((res_name, atom_name))
            if not type_id:
                # 尝试直接匹配 Type (有些 PDB 可能不标准)
                print(f"Warning: Type not found for {res_name}-{atom_name}, skipping force parameters.")
                custom_nb.addParticle([0.0]*10)
                if mpid_force:
                    mpid_force.addMultipole(0.0, [0.0]*3, [0.0]*6, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0)
                continue

            # 填充 CustomNonbondedForce
            try:
                # 注意：如果 XML 中缺少某些 Type 的参数，这里会报错，需要确保 XML 完整
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
                print(f"Error: Missing CustomNonbonded params for type {type_id}: {e}")
                raise

            # 填充 MPIDForce
            if mpid_force:
                try:
                    p_mpid = mpid_atoms[type_id]
                    p_polar = mpid_polars[type_id]

                    # 处理 Axis Type (简化版逻辑，参考 modify_admp_to_mpid.py)
                    kz = int(p_mpid.get("kz", 0))
                    kx = int(p_mpid.get("kx", 0))
                    ky = int(p_mpid.get("ky", 0)) # 如果有 ky

                    axis_type = mpidplugin.MPIDForce.ZThenX
                    if kz == 0: axis_type = mpidplugin.MPIDForce.NoAxisType
                    elif kz != 0 and kx == 0: axis_type = mpidplugin.MPIDForce.ZOnly
                    # ... 更多逻辑可根据需要补充 ...

                    mpid_force.addMultipole(
                        float(p_mpid.get("c0")),
                        [float(p_mpid.get("dX")), float(p_mpid.get("dY")), float(p_mpid.get("dZ"))],
                        [float(p_mpid.get("qXX")), float(p_mpid.get("qXY")), float(p_mpid.get("qYY")),
                         float(p_mpid.get("qXZ")), float(p_mpid.get("qYZ")), float(p_mpid.get("qZZ"))],
                        axis_type,
                        abs(kz)-1 if kz != 0 else 0,
                        abs(kx)-1 if kx != 0 else 1,
                        abs(ky)-1 if ky != 0 else 2,
                        float(p_polar.get("thole")),
                        float(p_polar.get("polarizabilityXX")),
                        float(p_polar.get("polarizabilityYY")),
                        float(p_polar.get("polarizabilityZZ"))
                    )
                except KeyError as e:
                    print(f"Error: Missing MPID params for type {type_id}: {e}")
                    raise

        system.addForce(custom_nb)
        if mpid_force:
            system.addForce(mpid_force)
            
        return system

    def compute_energy(self):
        """计算并打印能量详情"""
        state = self.simulation.context.getState(getEnergy=True)
        total_energy = state.getPotentialEnergy()
        print(f"\n{'='*30}")
        print(f"Total Potential Energy: {total_energy.value_in_unit(kilojoules_per_mole):.4f} kJ/mol")
        
        # 分项能量
        groups = {1: "CustomNonbonded (Slater/Disp)", 2: "MPIDForce (Elec/Pol)"}
        for g_id, g_name in groups.items():
            try:
                g_energy = self.simulation.context.getState(getEnergy=True, groups={g_id}).getPotentialEnergy()
                print(f"  - {g_name:<25}: {g_energy.value_in_unit(kilojoules_per_mole):.4f} kJ/mol")
            except Exception:
                pass
        print(f"{'='*30}\n")
        return total_energy

    def minimize(self, tolerance=10*kilojoules_per_mole/nanometer, maxIterations=0):
        """执行能量最小化"""
        print(f"Starting minimization (Tolerance: {tolerance})...")
        print("Energy before minimization:")
        self.compute_energy()
        
        self.simulation.minimizeEnergy(tolerance=tolerance, maxIterations=maxIterations)
        
        print("Energy after minimization:")
        self.compute_energy()
        
        # 保存新的 PDB
        output_pdb = "minimized.pdb"
        positions = self.simulation.context.getState(getPositions=True).getPositions()
        PDBFile.writeFile(self.topology, positions, open(output_pdb, 'w'))
        print(f"Minimized structure saved to {output_pdb}")

if __name__ == "__main__":
    # 示例用法
    pdb_path = "dimer_bank/dimer_003_EC_EC.pdb"  # 请替换为实际文件路径
    # xml_path = "modify_xml/mpid_EC_DMC_extracted.xml" # 请替换为实际文件路径
    xml_path = "ff_dmff_EC.xml" # 示例
    
    # 检查文件是否存在
    if not os.path.exists(pdb_path) or not os.path.exists(xml_path):
        print(f"Files not found: {pdb_path} or {xml_path}")
        print("Usage: python minimize_dmff_loaded.py <pdb> <xml>")
        if len(sys.argv) >= 3:
            pdb_path = sys.argv[1]
            xml_path = sys.argv[2]
        else:
            sys.exit(1)

    print(f"Loading calculator with PDB: {pdb_path}, XML: {xml_path}")
    calc = OpenMMEnergyCalculator(pdb_path, xml_path)
    
    # 1. 打印初始能量
    calc.compute_energy()
    
    # 2. 执行最小化
    # calc.minimize()