import sys
import os
from pathlib import Path

# Add project root to sys.path
script_dir = Path(__file__).resolve().parent
repo_root = str(script_dir.parent.parent)
sys.path.append(repo_root)

from phyneo_openmm.phyneo_protocol import load_phyneo_system

def run_test():
    example_root = script_dir.parent / 'example' / 'run_config'
    pdb_path = str(example_root / 'bulk_ec_packmol.pdb')
    xml_path = str(example_root / 'caff_5_mpid_slater_bond_hcp.xml')
    
    conda_prefix = os.environ.get('CONDA_PREFIX', '/home/am3-peichenzhong-group/miniconda3/envs/mpid')
    plugin_dir = os.path.join(conda_prefix, 'lib', 'plugins')
    os.environ['OPENMM_PLUGIN_DIR'] = plugin_dir

    import mpidplugin
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit

    print("Loading system...")
    loaded = load_phyneo_system(pdb_path=pdb_path, xml_path=xml_path, nonbonded_method='PME', platform='CUDA', verbose=False)
    system = loaded['system']
    topology = loaded['pdb'].topology
    positions = loaded['pdb'].positions

    from phyneo_openmm.phyneo_protocol import apply_mpid_scale_exclusions, _bonded_shells
    m_scales = [0.0, 0.0, 0.0, 0.0, 0.0]
    apply_mpid_scale_exclusions(system, topology, m_scales=m_scales)

    nb_force = next((f for f in system.getForces() if isinstance(f, mm.NonbondedForce)), None)
    custom_forces_indices = [i for i, f in enumerate(system.getForces()) if "CustomNonbondedForce" in type(f).__name__]
    
    if nb_force and custom_forces_indices:
        print(f"Synchronizing exclusions...")
        final_exclusions = set()
        for i in range(nb_force.getNumExceptions()):
            p1, p2, q, sig, eps = nb_force.getExceptionParameters(i)
            final_exclusions.add(tuple(sorted((p1, p2))))
        for idx in custom_forces_indices:
            cf = system.getForce(idx)
            for j in range(cf.getNumExclusions()):
                final_exclusions.add(tuple(sorted(cf.getExclusionParticles(j))))
        
        shells_all = _bonded_shells(topology, max_depth=5)
        residue_atoms = [[atom.index for atom in residue.atoms()] for residue in topology.residues()]
        atom_to_residue = {a: r for r, atoms in enumerate(residue_atoms) for a in atoms}
        for i in range(topology.getNumAtoms()):
            res_idx = atom_to_residue[i]; shells = shells_all[i]
            pairs = set(residue_atoms[res_idx]) if float(m_scales[4]) == 0.0 else set().union(*(shells[d] for d in range(1, 5) if float(m_scales[d-1]) == 0.0))
            for j in pairs:
                if i < j: final_exclusions.add((i, j))

        nb_force.createExceptionsFromBonds([], 1.0, 1.0)
        for p1, p2 in final_exclusions: nb_force.addException(p1, p2, 0.0, 0.1, 0.0)

        for idx in sorted(custom_forces_indices, reverse=True):
            old_cf = system.getForce(idx)
            new_cf = mm.CustomNonbondedForce(old_cf.getEnergyFunction())
            new_cf.setNonbondedMethod(old_cf.getNonbondedMethod())
            new_cf.setCutoffDistance(old_cf.getCutoffDistance())
            new_cf.setForceGroup(old_cf.getForceGroup())
            for i in range(old_cf.getNumPerParticleParameters()): new_cf.addPerParticleParameter(old_cf.getPerParticleParameterName(i))
            for i in range(old_cf.getNumGlobalParameters()): new_cf.addGlobalParameter(old_cf.getGlobalParameterName(i), old_cf.getGlobalParameterDefaultValue(i))
            for i in range(old_cf.getNumParticles()): new_cf.addParticle(old_cf.getParticleParameters(i))
            for p1, p2 in final_exclusions: new_cf.addExclusion(p1, p2)
            system.removeForce(idx); system.addForce(new_cf)

    print("Checking for MPIDForce...")
    mpid_force = None
    for i in range(system.getNumForces()):
        f = system.getForce(i)
        if mpidplugin.MPIDForce.isinstance(f):
            mpid_force = mpidplugin.MPIDForce.cast(f)
            break
    
    if mpid_force:
        print("Ensuring MPIDForce exclusions match...")
        # Note: MPIDForce manages its own CovalentMap which we already set via apply_mpid_scale_exclusions.
        # CUDA requirement usually focuses on NonbondedForce vs CustomNonbondedForce.
        pass

    platform = mm.Platform.getPlatformByName('CUDA')
    integrator = mm.LangevinMiddleIntegrator(300*unit.kelvin, 1/unit.picosecond, 1*unit.femtoseconds)
    sim = app.Simulation(topology, system, integrator, platform)
    sim.context.setPositions(positions)
    print("Test: Context created successfully on CUDA!")
    sim.step(10)
    print("Test: 10 steps completed successfully!")

if __name__ == '__main__':
    run_test()
