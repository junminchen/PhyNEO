#!/usr/bin/env python3
"""
Run single-solvent NPT MD density calculation.
Usage: conda activate mpid && python run_single.py <SOLVENT_NAME>

Outputs density to output/<solvent>_density.csv
"""
import sys
import os
import csv
from pathlib import Path

# Import openmm BEFORE adding PhyNEO to sys.path (PhyNEO/openmm/ shadows it)
conda_prefix = os.environ.get('CONDA_PREFIX', '/home/am3-peichenzhong-group/miniconda3/envs/mpid')
os.environ['OPENMM_PLUGIN_DIR'] = os.path.join(conda_prefix, 'lib', 'plugins')

import openmm as mm
import openmm.app as app
import openmm.unit as unit
import mpidplugin

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from phyneo_openmm.phyneo_protocol import _bonded_shells  # noqa: F401

PDB_BANK = PROJECT_ROOT / "data" / "pdb_bank"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
XML_PATH = PROJECT_ROOT / "phyneo_openmm" / "example" / "run_config" / "caff_5_mpid_slater_bond.xml"

SOLVENTS = {
    "EC":  ("EC.pdb",  "ECA", 10, 88.06,  1.32),
    "DEC": ("DEC.pdb", "DEC", 18, 118.13, 0.975),
    "DMC": ("DMC.pdb", "DMC", 12, 90.08,  1.07),
    "PC":  ("PC.pdb",  "PCA", 13, 102.09, 1.20),
    "FEC": ("FEC.pdb", "FEC", 10, 106.05, 1.45),
    "DME": ("DME.pdb", "DME", 16, 90.12,  0.87),
    "PS":  ("PS.pdb",  "PSA", 13, 120.17, 1.39),
    "SL":  ("SL.pdb",  "SLA", 15, 120.17, 1.27),
    "EMC": ("EMC.pdb", "EMC", 15, 104.10, 1.01),
}

BOX_SIZE = 35.0
NA = 6.02214076e23


def parse_conect_bonds(pdb_path):
    bonds = set()
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("CONECT"):
                parts = line.split()
                ai = int(parts[1]) - 1
                for p in parts[2:]:
                    bonds.add(tuple(sorted((ai, int(p) - 1))))
    return list(bonds)


def setup_mpid_polarization(system, topology):
    """Set PolarizationCovalent maps on MPIDForce.
    Only sets PolarizationCovalent (not Cov12/13/14/15) to avoid CUDA kernel crash.
    The XML-generated Cov12/13/14 defaults are already correct."""
    mpid_force = None
    for i in range(system.getNumForces()):
        f = system.getForce(i)
        if mpidplugin.MPIDForce.isinstance(f):
            mpid_force = mpidplugin.MPIDForce.cast(f)
            break
    if mpid_force is None:
        return

    residue_atoms = [[atom.index for atom in residue.atoms()] for residue in topology.residues()]
    atom_to_residue = {a: r for r, atoms in enumerate(residue_atoms) for a in atoms}

    for atom_index in range(mpid_force.getNumMultipoles()):
        res_idx = atom_to_residue[atom_index]
        full_intra = tuple(sorted([a for a in residue_atoms[res_idx] if a != atom_index]))
        mpid_force.setCovalentMap(atom_index, mpid_force.PolarizationCovalent11, (atom_index,))
        mpid_force.setCovalentMap(atom_index, mpid_force.PolarizationCovalent12, full_intra)
        mpid_force.setCovalentMap(atom_index, mpid_force.PolarizationCovalent13, tuple())
        mpid_force.setCovalentMap(atom_index, mpid_force.PolarizationCovalent14, tuple())


def main():
    solvent_name = sys.argv[1].upper()
    if solvent_name not in SOLVENTS:
        print(f"Unknown solvent: {solvent_name}")
        sys.exit(1)

    pdb_file, resname, atoms_per_mol, mw, exp_density = SOLVENTS[solvent_name]
    box_pdb = OUTPUT_DIR / f"{solvent_name.lower()}_box_{int(BOX_SIZE)}A.pdb"
    nmol = int(round(exp_density * (BOX_SIZE * 1e-8) ** 3 * NA / mw))

    print(f"=== {solvent_name}: {nmol} mols, res={resname} ===", flush=True)

    # Packmol if needed
    if not box_pdb.exists():
        import subprocess
        inp = f"""tolerance 2.0
filetype pdb
output {box_pdb}

structure {PDB_BANK / pdb_file}
  number {nmol}
  inside box 1.0 1.0 1.0 {BOX_SIZE-1} {BOX_SIZE-1} {BOX_SIZE-1}
end structure
"""
        inp_file = OUTPUT_DIR / f"{solvent_name.lower()}_packmol.inp"
        with open(inp_file, "w") as f:
            f.write(inp)
        subprocess.run(["packmol"], stdin=open(inp_file), capture_output=True, timeout=300)
        if not box_pdb.exists():
            print("PACKMOL FAILED", flush=True)
            sys.exit(1)

    # Load PDB
    pdb = app.PDBFile(str(box_pdb))
    topology = pdb.topology
    positions = pdb.positions

    # Add bonds
    template_bonds = parse_conect_bonds(str(PDB_BANK / pdb_file))
    for residue in topology.residues():
        if residue.name == resname:
            atoms = list(residue.atoms())
            for i, j in template_bonds:
                if i < len(atoms) and j < len(atoms):
                    topology.addBond(atoms[i], atoms[j])

    print(f"Atoms: {topology.getNumAtoms()}, Bonds/mol: {len(template_bonds)}", flush=True)

    box_vectors = (mm.Vec3(BOX_SIZE, 0, 0), mm.Vec3(0, BOX_SIZE, 0), mm.Vec3(0, 0, BOX_SIZE)) * unit.angstroms
    topology.setPeriodicBoxVectors(box_vectors)

    # Create system
    ff = app.ForceField(str(XML_PATH))
    print("Creating system...", flush=True)
    system = ff.createSystem(topology, nonbondedMethod=app.PME,
                             nonbondedCutoff=0.9 * unit.nanometers, constraints=app.HBonds)

    m_scales = [0.0, 0.0, 0.0, 0.0, 1.0]
    setup_mpid_polarization(system, topology)
    system.addForce(mm.MonteCarloBarostat(1.0 * unit.bar, 300.0 * unit.kelvin, 25))

    # Simulation
    # Try CUDA first; if --cpu flag, use CPU platform
    use_cpu = "--cpu" in sys.argv
    if use_cpu:
        print("Using CPU platform (--cpu flag)...", flush=True)
        platform = mm.Platform.getPlatformByName('CPU')
    else:
        print("Setting up CUDA simulation...", flush=True)
        try:
            platform = mm.Platform.getPlatformByName('CUDA')
        except Exception:
            print("CUDA not available, falling back to CPU", flush=True)
            platform = mm.Platform.getPlatformByName('CPU')
    integrator = mm.LangevinMiddleIntegrator(300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.5 * unit.femtoseconds)
    sim = app.Simulation(topology, system, integrator, platform)
    sim.context.setPositions(positions)

    total_mass_amu = sum(system.getParticleMass(i).value_in_unit(unit.dalton)
                         for i in range(system.getNumParticles()))

    # Add StateDataReporter for trajectory logging
    report_interval = 1000
    sim.reporters.append(app.StateDataReporter(
        sys.stdout, report_interval, step=True, potentialEnergy=True,
        temperature=True, density=True, speed=True, time=True, elapsedTime=True
    ))

    # Minimize
    print("Minimizing (2000 iter)...", flush=True)
    sim.minimizeEnergy(maxIterations=2000)
    state = sim.context.getState(getEnergy=True)
    print(f"Energy after min: {state.getPotentialEnergy()}", flush=True)

    # Equilibration in chunks to catch failures
    equil_steps = 100000
    chunk = 10000
    print(f"Equilibrating ({equil_steps} steps in {chunk}-step chunks)...", flush=True)
    for i in range(0, equil_steps, chunk):
        sim.step(chunk)
        state = sim.context.getState(getEnergy=True)
        bv = state.getPeriodicBoxVectors()
        vol_nm3 = (bv[0][0].value_in_unit(unit.nanometer)
                    * bv[1][1].value_in_unit(unit.nanometer)
                    * bv[2][2].value_in_unit(unit.nanometer))
        pe = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        print(f"  Equil step {i+chunk}: PE={pe:.1f} kJ/mol, Vol={vol_nm3:.3f} nm³", flush=True)

    # Production
    prod_steps = 100000
    report_interval = 2000
    print(f"Production ({prod_steps} steps)...", flush=True)
    densities = []
    for step_i in range(prod_steps // report_interval):
        sim.step(report_interval)
        state = sim.context.getState(getEnergy=True)
        bv = state.getPeriodicBoxVectors()
        vol_nm3 = (bv[0][0].value_in_unit(unit.nanometer)
                    * bv[1][1].value_in_unit(unit.nanometer)
                    * bv[2][2].value_in_unit(unit.nanometer))
        vol_cm3 = vol_nm3 * 1e-21
        mass_g = total_mass_amu / NA
        density = mass_g / vol_cm3
        densities.append(density)

    avg_d = sum(densities) / len(densities)
    std_d = (sum((d - avg_d) ** 2 for d in densities) / len(densities)) ** 0.5

    # Write result
    csv_path = OUTPUT_DIR / f"{solvent_name.lower()}_density.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Solvent", "Residue", "MW_g_mol", "Exp_Density_g_cm3",
                          "N_molecules", "Sim_Density_g_cm3", "Sim_Density_Std"])
        writer.writerow([solvent_name, resname, mw, exp_density, nmol, f"{avg_d:.6f}", f"{std_d:.6f}"])

    print(f"\nRESULT: {solvent_name} exp={exp_density:.3f} sim={avg_d:.4f}±{std_d:.4f} g/cm³", flush=True)
    print(f"Saved to: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
