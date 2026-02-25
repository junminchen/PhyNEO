from __future__ import print_function
import os
import sys
from os import path

sys.path.append("../../lib/")
from simtk.openmm.app import *  # noqa: F401,F403
from simtk.openmm import *  # noqa: F401,F403
from simtk.unit import *  # noqa: F401,F403
from MM_classes_FV import *
from Fixed_Voltage_routines import *


def _env_bool(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _pick_default_electrolyte_xml():
    candidates = [
        "phyneo_ecl_om.xml",
        "phyneo_ecl.xml",
        "../forcefield/li_fixed_charge.xml",
    ]
    for candidate in candidates:
        if path.exists(candidate):
            return candidate
    return candidates[0]


def _safe_reinitialize_context(simmd):
    state = simmd.context.getState(getPositions=True)
    positions = state.getPositions()
    simmd.context.reinitialize()
    simmd.context.setPositions(positions)


def _copy_custom_q_to_nonbonded(MMsys):
    custom = MMsys.customNonbondedForce
    if custom is None:
        return False
    q_index = None
    for i in range(custom.getNumPerParticleParameters()):
        if custom.getPerParticleParameterName(i) == "Q":
            q_index = i
            break
    if q_index is None:
        return False

    for i in range(custom.getNumParticles()):
        params = custom.getParticleParameters(i)
        q_i = float(params[q_index])
        _, sigma_i, epsilon_i = MMsys.nbondedForce.getParticleParameters(i)
        MMsys.nbondedForce.setParticleParameters(i, q_i, sigma_i, epsilon_i)
    return True


def _remove_force_classes(MMsys, class_names):
    removed = []
    for i in range(MMsys.system.getNumForces() - 1, -1, -1):
        force = MMsys.system.getForce(i)
        force_name = force.__class__.__name__
        if force_name in class_names:
            removed.append(force_name)
            MMsys.system.removeForce(i)
    return removed


def _set_electrode_fixed_charges(MMsys, cathode_total_charge):
    if MMsys.Cathode.Natoms == 0 or MMsys.Anode.Natoms == 0:
        raise ValueError("Both electrodes must have at least one atom for constant charge mode")

    cathode_q = cathode_total_charge / MMsys.Cathode.Natoms
    anode_q = -cathode_total_charge / MMsys.Anode.Natoms

    for atom in MMsys.Cathode.electrode_atoms:
        atom.charge = cathode_q
        MMsys.nbondedForce.setParticleParameters(atom.atom_index, cathode_q, 1.0, 0.0)
    for atom in MMsys.Anode.electrode_atoms:
        atom.charge = anode_q
        MMsys.nbondedForce.setParticleParameters(atom.atom_index, anode_q, 1.0, 0.0)

    MMsys.nbondedForce.updateParametersInContext(MMsys.simmd.context)
    print(
        "Constant_Q mode: Cathode total charge = {:.6f} e, Anode total charge = {:.6f} e".format(
            cathode_total_charge, -cathode_total_charge
        )
    )


simulation_time_ns = float(os.environ.get("SIM_TIME_NS", 1.0))
freq_charge_update_fs = int(os.environ.get("FREQ_CHARGE_UPDATE_FS", 50))
freq_traj_output_ps = int(os.environ.get("FREQ_TRAJ_OUTPUT_PS", 50))
freq_checkpoint_ps = int(os.environ.get("FREQ_CHECKPOINT_PS", 10))
write_charges = _env_bool("WRITE_CHARGES", True)

checkpoint = "state.chk"
charge_name = "charges.dat"

simulation_type = os.environ.get("SIM_MODE", "Constant_Q")
Voltage = float(os.environ.get("APPLIED_VOLTAGE", 0.0))
cathode_charge = float(os.environ.get("CATHODE_CHARGE", 0.0))

# Interaction model for interface:
# - full: keep original force decomposition from XML.
# - charge_only: remove MPID/CustomNonbonded, use Nonbonded charge-charge (Q copied from CustomNonbonded if available).
interface_model = os.environ.get("INTERFACE_MODEL", "charge_only")

cathode_index = (0, 2)
anode_index = (1, 3)

graphene_ffdir = os.environ.get(
    "GRAPHENE_FF_DIR", "../../Example_graphene_BMIM_BF4_ACN_10pct/graphene_ffdir/"
)
electrolyte_ff = os.environ.get("ELECTROLYTE_FF_XML", _pick_default_electrolyte_xml())
electrolyte_residues = os.environ.get("ELECTROLYTE_RESIDUE_XML", "")

equilibrated_pdb = os.environ.get("EQUILIBRATED_PDB", "equilibrated.pdb")
packmol_pdb = os.environ.get("PACKMOL_PDB", "../packmol/li_electrolyte_box.pdb")
input_pdb = os.environ.get("INPUT_PDB")
if input_pdb is None:
    input_pdb = equilibrated_pdb if path.exists(equilibrated_pdb) else packmol_pdb

residue_xml_list = [
    graphene_ffdir + "graph_residue_c.xml",
    graphene_ffdir + "graph_residue_n.xml",
]
if electrolyte_residues and path.exists(electrolyte_residues):
    residue_xml_list = [electrolyte_residues] + residue_xml_list

MMsys = MM_FixedVoltage(
    pdb_list=[input_pdb],
    residue_xml_list=residue_xml_list,
    ff_xml_list=[
        electrolyte_ff,
        graphene_ffdir + "graph.xml",
        graphene_ffdir + "graph_c_freeze.xml",
        graphene_ffdir + "graph_n_freeze.xml",
    ],
)

MMsys.set_periodic_residue(True)
platform_name = os.environ.get("FV_PLATFORM", "Reference")
MMsys.set_platform(platform_name)

MMsys.initialize_electrodes(
    Voltage,
    cathode_identifier=cathode_index,
    anode_identifier=anode_index,
    chain=True,
    exclude_element=("H",),
)
MMsys.initialize_electrolyte(Natom_cutoff=100)
MMsys.generate_exclusions(flag_SAPT_FF_exclusions=False)

if interface_model == "charge_only":
    copied_q = _copy_custom_q_to_nonbonded(MMsys)
    removed = _remove_force_classes(MMsys, {"MPIDForce", "CustomNonbondedForce"})
    _safe_reinitialize_context(MMsys.simmd)
    print("INTERFACE_MODEL=charge_only, copied_Q_to_nonbonded =", copied_q)
    print("Removed force classes:", removed if removed else "none")

use_checkpoint = _env_bool("USE_CHECKPOINT", False)
if use_checkpoint and path.exists(checkpoint):
    print("restarting simulation from checkpoint", checkpoint)
    MMsys.simmd.loadCheckpoint(checkpoint)

state = MMsys.simmd.context.getState(getEnergy=True, getForces=True, getPositions=True)
with open("start.pdb", "w") as handle:
    PDBFile.writeFile(MMsys.simmd.topology, state.getPositions(), handle)

append_trajectory = False
if simulation_type == "MC_equil":
    trajectory_file_name = "equil_MC.dcd"
elif simulation_type == "Constant_V":
    trajectory_file_name = "FV_NVT.dcd"
elif simulation_type == "Constant_Q":
    trajectory_file_name = "FQ_NVT.dcd"
else:
    trajectory_file_name = "traj.dcd"
if simulation_type in ("Constant_V", "Constant_Q") and path.exists(trajectory_file_name):
    append_trajectory = True

MMsys.set_trajectory_output(
    trajectory_file_name,
    freq_traj_output_ps * 1000,
    append_trajectory,
    checkpoint,
    freq_checkpoint_ps * 1000,
)

if simulation_type == "MC_equil":
    celldim = MMsys.simmd.topology.getUnitCellDimensions()
    MMsys.MC = MC_parameters(
        MMsys.temperature,
        celldim,
        electrode_move="Anode",
        pressure=1.0 * bar,
        barofreq=100,
        shiftscale=0.2,
    )
elif simulation_type == "Constant_Q":
    _set_electrode_fixed_charges(MMsys, cathode_charge)
elif simulation_type == "Constant_V":
    MMsys.Poisson_solver_fixed_voltage(
        Niterations=5, compute_intermediate_forces=True, print_flag=True
    )
else:
    print("simulation type not recognized:", simulation_type)
    sys.exit(1)

for _ in range(int(simulation_time_ns * 1000 / freq_traj_output_ps)):
    state = MMsys.simmd.context.getState(
        getEnergy=True, getForces=True, getVelocities=False, getPositions=True
    )
    print(str(state.getKineticEnergy()))
    print(str(state.getPotentialEnergy()))
    print("", flush=True)

    if simulation_type == "MC_equil":
        for _ in range(int(freq_traj_output_ps * 1000 / MMsys.MC.barofreq)):
            MMsys.MC_Barostat_step()
    elif simulation_type == "Constant_V":
        MMsys.Poisson_solver_fixed_voltage(
            Niterations=1, compute_intermediate_forces=True, print_flag=True
        )
        for _ in range(int(freq_traj_output_ps * 1000 / freq_charge_update_fs)):
            MMsys.Poisson_solver_fixed_voltage(
                Niterations=1, compute_intermediate_forces=True
            )
            MMsys.simmd.step(freq_charge_update_fs)
    elif simulation_type == "Constant_Q":
        MMsys.simmd.step(int(freq_traj_output_ps * 1000))

    if write_charges:
        with open(charge_name, "a") as charge_file:
            MMsys.write_electrode_charges(charge_file)

if simulation_type == "MC_equil":
    state = MMsys.simmd.context.getState(getPositions=True)
    with open("equilibrated.pdb", "w") as handle:
        PDBFile.writeFile(MMsys.simmd.topology, state.getPositions(), handle)

print("done!")
sys.exit(0)
