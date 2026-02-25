import sys
import os
from pathlib import Path

# Add current directory to sys.path to allow importing local phyneo_openmm
script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir))

import openmm as mm
import openmm.app as app
import openmm.unit as unit

# Ensure OPENMM_PLUGIN_DIR is set
plugin_dir = os.environ.get('OPENMM_PLUGIN_DIR')
if not plugin_dir:
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        plugin_dir = os.path.join(conda_prefix, 'lib', 'plugins')
        os.environ['OPENMM_PLUGIN_DIR'] = plugin_dir
    else:
        print("Warning: CONDA_PREFIX not found. Please activate your mpid environment.")

# Import MPID plugin (needed for XML loading)
try:
    import mpidplugin
except ImportError:
    print("Error: mpidplugin not found. Make sure you are in the correct mpid environment.")
    sys.exit(1)

from phyneo_openmm.phyneo_protocol import load_phyneo_system, apply_mpid_scale_exclusions, _bonded_shells

def run():
    # Paths to input files (local to workdir)
    pdb_path = str(script_dir / 'ec_box_35A.pdb')
    xml_path = str(script_dir / 'caff_5_mpid_slater_bond.xml')
    
    print(f"Loading base system from {pdb_path}...")
    pdb = app.PDBFile(pdb_path)
    topology = pdb.topology
    positions = pdb.positions
    
    # Manually add bonds for ECA residues (missing CONECT records in PDB)
    print("Adding missing bonds for ECA residues...")
    for residue in topology.residues():
        if residue.name == 'ECA':
            atoms = list(residue.atoms())
            if len(atoms) != 10:
                continue
            bonds_indices = [
                (0,1), (1,2), (1,5), (2,3), (3,4), (4,5),
                (3,6), (3,7), (4,8), (4,9)
            ]
            for i, j in bonds_indices:
                topology.addBond(atoms[i], atoms[j])

    print(f"Base system atoms: {topology.getNumAtoms()}")

    # Replicate manually to reach 10,240 atoms
    repl_factor = 1
    print(f"Replicating system {repl_factor}x{repl_factor}x{repl_factor} to 10,240 atoms...")
    modeller = app.Modeller(topology, positions)
    box = topology.getPeriodicBoxVectors()
    if box is None:
        print("Warning: PDB has no CRYST1 record. Setting box size to 35.0 A based on filename.")
        box_vectors = (mm.Vec3(35.0, 0.0, 0.0), mm.Vec3(0.0, 35.0, 0.0), mm.Vec3(0.0, 0.0, 35.0)) * unit.angstroms
        topology.setPeriodicBoxVectors(box_vectors)
        box = box_vectors
    vec_a, vec_b, vec_c = box[0], box[1], box[2]
    
    for i in range(repl_factor):
        for j in range(repl_factor):
            for k in range(repl_factor):
                if i == 0 and j == 0 and k == 0: continue
                offset = i*vec_a + j*vec_b + k*vec_c
                new_pos = [(p + offset) for p in positions]
                modeller.add(topology, new_pos)
    
    new_topology = modeller.getTopology()
    new_positions = modeller.getPositions()
    new_topology.setPeriodicBoxVectors((repl_factor*vec_a, repl_factor*vec_b, repl_factor*vec_c))
    print(f"New system atoms: {new_topology.getNumAtoms()}")

    # Create new system
    ff = app.ForceField(xml_path)
    print("Creating replicated system...")
    new_system = ff.createSystem(
        new_topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=0.9*unit.nanometers,
        constraints=app.HBonds
    )

    # Define m_scales for exclusion consistency (matching default in phyneo_protocol)
    m_scales = [0.0, 0.0, 0.0, 0.0, 1.0] 

    apply_mpid_scale_exclusions(new_system, new_topology, m_scales=m_scales)

    # Add Barostat for NPT ensemble
    print("Adding MonteCarloBarostat for NPT ensemble...")
    new_system.addForce(mm.MonteCarloBarostat(1.0*unit.bar, 300.0*unit.kelvin, 25))

    # CUDA-specific synchronization of exclusions
    nb_force = next((f for f in new_system.getForces() if isinstance(f, mm.NonbondedForce)), None)
    custom_forces_indices = [i for i, f in enumerate(new_system.getForces()) if "CustomNonbondedForce" in type(f).__name__]
    
    if custom_forces_indices:
        print(f"Rigidly synchronizing exclusions for CUDA compatibility...")
        final_exclusions = set()
        if nb_force:
            for i in range(nb_force.getNumExceptions()):
                p1, p2, q, sig, eps = nb_force.getExceptionParameters(i)
                final_exclusions.add(tuple(sorted((p1, p2))))
        
        for idx in custom_forces_indices:
            cf = new_system.getForce(idx)
            for j in range(cf.getNumExclusions()):
                final_exclusions.add(tuple(sorted(cf.getExclusionParticles(j))))
        
        # Add m_scales logic to the union
        shells_all = _bonded_shells(new_topology, max_depth=5)
        residue_atoms = [[atom.index for atom in residue.atoms()] for residue in new_topology.residues()]
        atom_to_residue = {a: r for r, atoms in enumerate(residue_atoms) for a in atoms}
        for i in range(new_topology.getNumAtoms()):
            res_idx = atom_to_residue[i]
            shells = shells_all[i]
            pairs = set()
            if float(m_scales[4]) == 0.0:
                pairs.update(residue_atoms[res_idx])
            else:
                if float(m_scales[0]) == 0.0: pairs.update(shells[1])
                if float(m_scales[1]) == 0.0: pairs.update(shells[2])
                if float(m_scales[2]) == 0.0: pairs.update(shells[3])
                if float(m_scales[3]) == 0.0: pairs.update(shells[4])
            for j in pairs:
                if i < j:
                    final_exclusions.add((i, j))

        # Update NonbondedForce Exceptions
        if nb_force:
            existing_pairs = {}
            for i in range(nb_force.getNumExceptions()):
                p1, p2, _, _, _ = nb_force.getExceptionParameters(i)
                pair = tuple(sorted((p1, p2)))
                existing_pairs[pair] = i

            for p1, p2 in final_exclusions:
                pair = tuple(sorted((p1, p2)))
                if pair in existing_pairs:
                    idx = existing_pairs[pair]
                    nb_force.setExceptionParameters(idx, p1, p2, 0.0, 0.1, 0.0)
                else:
                    nb_force.addException(p1, p2, 0.0, 0.1, 0.0)

        # Rebuild CustomNonbondedForces
        for idx in sorted(custom_forces_indices, reverse=True):
            old_cf = new_system.getForce(idx)
            new_cf = mm.CustomNonbondedForce(old_cf.getEnergyFunction())
            new_cf.setNonbondedMethod(old_cf.getNonbondedMethod())
            new_cf.setCutoffDistance(old_cf.getCutoffDistance())
            new_cf.setUseLongRangeCorrection(old_cf.getUseLongRangeCorrection())
            new_cf.setForceGroup(old_cf.getForceGroup())
            for i in range(old_cf.getNumPerParticleParameters()):
                new_cf.addPerParticleParameter(old_cf.getPerParticleParameterName(i))
            for i in range(old_cf.getNumGlobalParameters()):
                new_cf.addGlobalParameter(old_cf.getGlobalParameterName(i), old_cf.getGlobalParameterDefaultValue(i))
            for i in range(old_cf.getNumParticles()):
                new_cf.addParticle(old_cf.getParticleParameters(i))
            for p1, p2 in final_exclusions:
                new_cf.addExclusion(p1, p2)
            new_system.removeForce(idx)
            new_system.addForce(new_cf)

    # Assign force groups
    force_groups = {}
    for i, force in enumerate(new_system.getForces()):
        group_id = i % 32
        force.setForceGroup(group_id)
        force_groups[group_id] = force.getName() if force.getName() else type(force).__name__

    # Use CUDA platform
    try:
        platform = mm.Platform.getPlatformByName('CUDA')
    except Exception:
        print("CUDA platform not found, falling back to CPU.")
        platform = mm.Platform.getPlatformByName('CPU')

    integrator = mm.LangevinMiddleIntegrator(300.0*unit.kelvin, 1.0/unit.picosecond, 1.0*unit.femtoseconds)
    sim = app.Simulation(new_topology, new_system, integrator, platform)
    sim.context.setPositions(new_positions)
    
    def print_energy_breakdown(simulation, context_name="Energy Breakdown"):
        print(f"\n--- {context_name} ---")
        state = simulation.context.getState(getEnergy=True)
        total_energy = state.getPotentialEnergy()
        print(f"Total Potential Energy: {total_energy}")
        for group_id, force_name in force_groups.items():
            state = simulation.context.getState(getEnergy=True, groups=(1 << group_id))
            energy = state.getPotentialEnergy()
            print(f"Force Group {group_id} ({force_name}): {energy}")
        print("------------------------\n")

    print_energy_breakdown(sim, "Initial Energy")

    print("Minimizing energy...")
    sim.minimizeEnergy(maxIterations=100)
    
    num_steps = 1000
    report_interval = 100
    print(f"Starting simulation ({num_steps} steps) on {platform.getName()}...")
    sim.reporters.append(app.StateDataReporter(
        sys.stdout, report_interval, step=True, potentialEnergy=True, 
        temperature=True, density=True, speed=True
    ))
    
    sim.step(num_steps)
    print("Simulation finished.")

if __name__ == '__main__':
    run()
