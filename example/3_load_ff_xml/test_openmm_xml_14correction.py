#!/usr/bin/env python
from openmm.app import Modeller
import openmm as mm 
import openmm.app as app
import openmm.unit as unit 
import numpy as np
import sys

# 尝试导入可选依赖
try:
    from dmff import Hamiltonian
    from dmff.common import nblist
    from jax import jit
    import jax.numpy as jnp
except ImportError:
    pass

def forcegroupify(system):
    forcegroups = {}
    for i in range(system.getNumForces()):
        force = system.getForce(i)
        force.setForceGroup(i)
        forcegroups[force] = i
    return forcegroups

def getEnergyDecomposition(context, forcegroups):
    energies = {}
    for f, i in forcegroups.items():
        energies[f] = context.getState(getEnergy=True, groups=2**i).getPotentialEnergy()
    return energies

def find_14_pairs(topology):
    """遍历拓扑寻找所有 1-4 原子对"""
    atoms = list(topology.atoms())
    bonds = list(topology.bonds())
    atom_neighbors = [set() for _ in range(topology.getNumAtoms())]
    for bond in bonds:
        atom_neighbors[bond.atom1.index].add(bond.atom2.index)
        atom_neighbors[bond.atom2.index].add(bond.atom1.index)

    pairs14 = set()
    for a1 in range(len(atoms)):
        for a2 in atom_neighbors[a1]:
            for a3 in atom_neighbors[a2]:
                if a3 == a1: continue
                for a4 in atom_neighbors[a3]:
                    if a4 == a2 or a4 == a1: continue
                    if a4 in atom_neighbors[a1]: continue 
                    if a1 < a4:
                        pairs14.add((a1, a4))
    return list(pairs14)

def create_14_correction_force(system, topology, scale_factor=1.0):
    """
    创建 1-4 修正力。
    由于 XML 中将力场拆分为了 Repulsion, Disp_Main, Disp_Damp 三部分，
    这里我们需要将它们重新组合，构建一个完整的 1-4 势能函数。
    """
    print("Creating CustomBondForce for 1-4 interactions...")
    
    atom_params = {} 

    # 1. 提取参数
    # 我们需要遍历所有 CustomNonbondedForce，收集所有原子的所有参数
    # 因为现在的参数分散在 3 个 Force 里
    forces_to_check = [f for f in system.getForces() if isinstance(f, mm.CustomNonbondedForce)]
    
    if not forces_to_check:
        print("Warning: No CustomNonbondedForces found. Skipping 1-4 generation.")
        return

    for force in forces_to_check:
        for i in range(system.getNumParticles()):
            if i not in atom_params: atom_params[i] = {}
            vals = force.getParticleParameters(i)
            for p_idx in range(force.getNumPerParticleParameters()):
                p_name = force.getPerParticleParameterName(p_idx)
                atom_params[i][p_name] = vals[p_idx]

    # 2. 构建 1-4 的能量公式
    # 目标公式 = (Repulsion + (Disp_Main + Disp_Damp)) * scale
    # 注意：Disp_Main 是 -C6/r^6，Disp_Damp 是 +exp()*C6/r^6
    # 它们相加正好是原始的带阻尼的色散公式: - (1-exp)*C6/r^6
    # 所以我们可以直接用原始的完整公式，这样最简单也最不易错。
    
    # Repulsion
    rep_expr = "(A*K2*exp(-B*r) + (-138.93542)*exp(-B*r)*(1+B*r)*Q/r)"
    
    # Full Dispersion (带阻尼的完整形式)
    disp_expr = """(
    - (1-exp(-x)*(1+x+0.5*x^2+x^3/6+x^4/24+x^5/120+x^6/720)) * C6/(r^6) 
    - (1-exp(-x)*(1+x+0.5*x^2+x^3/6+x^4/24+x^5/120+x^6/720+x^7/5040+x^8/40320)) * C8/(r^8) 
    - (1-exp(-x)*(1+x+0.5*x^2+x^3/6+x^4/24+x^5/120+x^6/720+x^7/5040+x^8/40320+x^9/362880+x^10/3628800)) * C10/(r^10)
    )"""
    
    # 公共定义
    common_definitions = """
    x=B*r - (2*B^2*r+3*B)/(B^2*r^2+3*B*r+3)*r;
    K2=(Br^2)/3 + Br + 1;
    Br = B*r;
    B=sqrt(Bexp1*Bexp2);
    Q=Q1*Q2;
    C6=sqrt(C61*C62);
    C8=sqrt(C81*C82);
    C10=sqrt(C101*C102);
    A=Aex-Ael-Ain-Adh-Adi;
    Aex=(Aexch1*Aexch2);
    Ael=(Aelec1*Aelec2);
    Ain=(Aind1*Aind2);
    Adh=(Adhf1*Adhf2);
    Adi=(Adisp1*Adisp2);
    """
    
    full_formula = f"({rep_expr} + {disp_expr}) * scale; " + common_definitions
    
    bond_force = mm.CustomBondForce(full_formula)
    
    # 3. 添加参数
    param_names = [
        "Bexp", "Q", "C6", "C8", "C10", 
        "Aexch", "Aelec", "Aind", "Adhf", "Adisp"
    ]
    
    for p in param_names:
        bond_force.addPerBondParameter(f"{p}1")
        bond_force.addPerBondParameter(f"{p}2")
    
    bond_force.addPerBondParameter("scale") 
    
    # 4. 添加 1-4 对
    pairs_14 = find_14_pairs(topology)
    print(f"Found {len(pairs_14)} 1-4 pairs.")
    
    count = 0
    for idx1, idx2 in pairs_14:
        p1 = atom_params[idx1]
        p2 = atom_params[idx2]
        bond_params = []
        for p in param_names:
            # 使用 .get(p, 0.0) 处理部分参数只在某些 Force 里存在的情况
            bond_params.append(p1.get(p, 0.0))
            bond_params.append(p2.get(p, 0.0))
        
        bond_params.append(scale_factor)
        bond_force.addBond(idx1, idx2, bond_params)
        count += 1
        
    bond_force.setName("Custom14Force")
    system.addForce(bond_force)
    print(f"Added {count} interactions to Custom14Force.")
    return bond_force

if __name__ == "__main__":

    print("Loading PDB and XML...")
    pdb = app.PDBFile("dimer_bank/dimer_003_EC_EC.pdb")
    
    # 务必确保 xml 已经按照要求拆分成 3 部分 (Repulsion, Disp_Main, Disp_Damp)
    ff = app.ForceField("ff_openmm_EC_14.xml")
    
    print("Creating System...")
    # 使用 CutoffPeriodic, 1.2nm 截断
    system = ff.createSystem(
        pdb.topology, 
        nonbondedMethod=app.CutoffPeriodic, 
        nonbondedCutoff=1.2*unit.nanometer,
        constraints=None, 
        removeCMMotion=False
    )

    print("Configuring Long Range Corrections...")
    for i, force in enumerate(system.getForces()):
        if isinstance(force, mm.CustomNonbondedForce):
            energy_expr = force.getEnergyFunction()
            
            # 逻辑：只有不含 exp 且含有 C6 的 Force (Dispersion Main) 开启 LRC
            if "exp" not in energy_expr and "C6" in energy_expr:
                print(f"  Force {i} (Dispersion Main): Enabling LRC (Standard 1/r^n)")
                force.setUseLongRangeCorrection(True)
            elif "exp" in energy_expr:
                print(f"  Force {i} (Short Range - Repulsion/Damping): Disabling LRC")
                force.setUseLongRangeCorrection(False)
            else:
                print(f"  Force {i}: Unknown Type, Disabling LRC")
                force.setUseLongRangeCorrection(False)

    # 添加 1-4 修正 (因为 CutoffPeriodic 会自动排除 1-4，所以要手动加回来)
    create_14_correction_force(system, pdb.topology, scale_factor=1.0)

    print("Running Simulation...")
    forcegroups = forcegroupify(system)
    integrator = mm.VerletIntegrator(0.1 * unit.femtoseconds)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("Reference"))
    
    context.setPositions(pdb.positions)
    if pdb.topology.getPeriodicBoxVectors():
        context.setPeriodicBoxVectors(*pdb.topology.getPeriodicBoxVectors())

    state = context.getState(getEnergy=True)
    energy = state.getPotentialEnergy()
    energies = getEnergyDecomposition(context, forcegroups)
    
    print("-" * 30)
    print(f"Total Potential Energy: {energy.value_in_unit(unit.kilojoules_per_mole):.4f} kJ/mol")
    print("-" * 30)
    for f, e in energies.items():
        # 尝试获取更有意义的名字
        name = f.getName()
        if isinstance(f, mm.CustomNonbondedForce):
             if "exp" not in f.getEnergyFunction() and "C6" in f.getEnergyFunction():
                 name += " (Dispersion Main LRC)"
             elif "exp" in f.getEnergyFunction() and "C6" in f.getEnergyFunction():
                 name += " (Dispersion Damping)"
             elif "exp" in f.getEnergyFunction() and "Q" in f.getEnergyFunction():
                 name += " (Repulsion/Elec)"
        
        print(f"{name}: {e.value_in_unit(unit.kilojoules_per_mole):.4f} kJ/mol")
    print("-" * 30)