from __future__ import print_function
import copy
import os
import sys
import xml.etree.ElementTree as ET

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
        "../forcefield/li_mpid.xml",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def _collect_type_to_class(*xml_files):
    type_to_class = {}
    for xml_file in xml_files:
        if not os.path.exists(xml_file):
            continue
        root = ET.parse(xml_file).getroot()
        atom_types = root.find("AtomTypes")
        if atom_types is None:
            continue
        for t in atom_types.findall("Type"):
            t_name = t.attrib.get("name")
            t_class = t.attrib.get("class")
            if t_name and t_class:
                type_to_class[t_name] = t_class
    return type_to_class


def _collect_types_from_residue_xml(*residue_xml_files):
    type_names = set()
    for residue_xml in residue_xml_files:
        root = ET.parse(residue_xml).getroot()
        for atom in root.findall(".//Atom"):
            t_name = atom.attrib.get("type")
            if t_name:
                type_names.add(t_name)
    return type_names


def _patch_customnonbonded_for_electrode(root, graph_custom_xml, required_classes):
    custom = root.find("CustomNonbondedForce")
    if custom is None:
        raise RuntimeError("Input electrolyte xml has no <CustomNonbondedForce> section.")

    existing_classes = {
        x.attrib.get("class") for x in custom.findall("Atom") if x.attrib.get("class")
    }

    graph_root = ET.parse(graph_custom_xml).getroot()
    graph_custom = graph_root.find("CustomNonbondedForce")
    if graph_custom is None:
        raise RuntimeError("Graphene parameter xml has no <CustomNonbondedForce> section.")
    graph_class_map = {
        x.attrib.get("class"): x
        for x in graph_custom.findall("Atom")
        if x.attrib.get("class")
    }

    zero_template = {
        "Aexch": "0",
        "Aelec": "0",
        "Aind": "0",
        "Adhf": "0",
        "Adisp": "0",
        "Bexp": "100",
        "Q": "0",
        "C6": "0",
        "C8": "0",
        "C10": "0",
        "C12": "0",
    }

    inserted = 0
    for cls in sorted(required_classes):
        if cls in existing_classes:
            continue
        if cls in graph_class_map:
            atom = copy.deepcopy(graph_class_map[cls])
            for key, val in zero_template.items():
                if key not in atom.attrib:
                    atom.set(key, val)
            custom.append(atom)
        else:
            atom = ET.Element("Atom")
            atom.set("class", cls)
            for key, val in zero_template.items():
                atom.set(key, val)
            custom.append(atom)
        inserted += 1
    return inserted


def _patch_mpid_for_electrode_types(root, electrode_types, c0_default, alpha_default, thole_default):
    mpid = root.find("MPIDForce")
    if mpid is None:
        raise RuntimeError("Input electrolyte xml has no <MPIDForce> section.")

    if "coulomb14scale" not in mpid.attrib:
        mpid.set("coulomb14scale", "0")

    multipoles = {x.attrib.get("type") for x in mpid.findall("Multipole") if x.attrib.get("type")}
    polarizes = {x.attrib.get("type") for x in mpid.findall("Polarize") if x.attrib.get("type")}

    inserted_m = 0
    inserted_p = 0
    for t_name in sorted(electrode_types):
        if t_name not in multipoles:
            m = ET.Element("Multipole")
            m.set("type", t_name)
            m.set("kz", "")
            m.set("kx", "")
            m.set("c0", c0_default)
            m.set("dX", "0.00000000")
            m.set("dY", "0.00000000")
            m.set("dZ", "0.00000000")
            m.set("qXX", "0.00000000")
            m.set("qXY", "0.00000000")
            m.set("qYY", "0.00000000")
            m.set("qXZ", "0.00000000")
            m.set("qYZ", "0.00000000")
            m.set("qZZ", "0.00000000")
            mpid.append(m)
            inserted_m += 1

        if t_name not in polarizes:
            p = ET.Element("Polarize")
            p.set("type", t_name)
            p.set("polarizabilityXX", alpha_default)
            p.set("polarizabilityYY", alpha_default)
            p.set("polarizabilityZZ", alpha_default)
            p.set("thole", thole_default)
            mpid.append(p)
            inserted_p += 1
    return inserted_m, inserted_p


def _prepare_polarizable_interface_xml(
    base_xml,
    graph_custom_xml,
    graph_residue_c_xml,
    graph_residue_n_xml,
    graph_c_freeze_xml,
    graph_n_freeze_xml,
    out_xml,
    c0_default,
    alpha_default,
    thole_default,
):
    type_to_class = _collect_type_to_class(graph_c_freeze_xml, graph_n_freeze_xml)
    electrode_types = _collect_types_from_residue_xml(graph_residue_c_xml, graph_residue_n_xml)
    required_classes = {type_to_class[t] for t in electrode_types if t in type_to_class}

    tree = ET.parse(base_xml)
    root = tree.getroot()

    n_custom = _patch_customnonbonded_for_electrode(root, graph_custom_xml, required_classes)
    n_mpid_m, n_mpid_p = _patch_mpid_for_electrode_types(
        root, electrode_types, c0_default, alpha_default, thole_default
    )
    tree.write(out_xml, encoding="utf-8", xml_declaration=True)
    return n_custom, n_mpid_m, n_mpid_p


def _prepare_graph_xml_nonbonded_compatible(electrolyte_xml, graph_xml, out_graph_xml):
    elec_root = ET.parse(electrolyte_xml).getroot()
    elec_nb = elec_root.find("NonbondedForce")
    if elec_nb is None:
        raise RuntimeError("Electrolyte xml has no <NonbondedForce> section.")

    graph_tree = ET.parse(graph_xml)
    graph_root = graph_tree.getroot()
    graph_nb = graph_root.find("NonbondedForce")
    if graph_nb is None:
        raise RuntimeError("Graph xml has no <NonbondedForce> section.")

    for key in ("coulomb14scale", "lj14scale"):
        if key in elec_nb.attrib:
            graph_nb.set(key, elec_nb.attrib[key])
    graph_tree.write(out_graph_xml, encoding="utf-8", xml_declaration=True)


def _set_electrode_fixed_charges(MMsys, cathode_total_charge, sigma, epsilon):
    if MMsys.Cathode.Natoms == 0 or MMsys.Anode.Natoms == 0:
        raise ValueError("Both electrodes must have at least one atom for constant charge mode")

    cathode_q = cathode_total_charge / MMsys.Cathode.Natoms
    anode_q = -cathode_total_charge / MMsys.Anode.Natoms

    for atom in MMsys.Cathode.electrode_atoms:
        atom.charge = cathode_q
        MMsys.nbondedForce.setParticleParameters(atom.atom_index, cathode_q, sigma, epsilon)
    for atom in MMsys.Anode.electrode_atoms:
        atom.charge = anode_q
        MMsys.nbondedForce.setParticleParameters(atom.atom_index, anode_q, sigma, epsilon)

    MMsys.nbondedForce.updateParametersInContext(MMsys.simmd.context)
    print(
        "Constant_Q mode: Cathode total charge = {:.6f} e, Anode total charge = {:.6f} e".format(
            cathode_total_charge, -cathode_total_charge
        )
    )


def _parse_box_nm_from_env():
    raw = os.environ.get("PBC_BOX_NM")
    if raw:
        parts = [x.strip() for x in raw.split(",")]
        if len(parts) != 3:
            raise ValueError("PBC_BOX_NM must be 3 comma-separated floats, e.g. 6.0,6.0,12.0")
        return float(parts[0]), float(parts[1]), float(parts[2])

    x = os.environ.get("BOX_X_NM")
    y = os.environ.get("BOX_Y_NM")
    z = os.environ.get("BOX_Z_NM")
    if x and y and z:
        return float(x), float(y), float(z)
    return None


def _infer_box_nm_from_positions(MMsys, pad_xy_nm, pad_z_nm, vacuum_z_nm):
    state = MMsys.simmd.context.getState(getPositions=True)
    positions = state.getPositions()
    xmin = min(p[0]._value for p in positions)
    xmax = max(p[0]._value for p in positions)
    ymin = min(p[1]._value for p in positions)
    ymax = max(p[1]._value for p in positions)
    zmin = min(p[2]._value for p in positions)
    zmax = max(p[2]._value for p in positions)
    lx = (xmax - xmin) + pad_xy_nm
    ly = (ymax - ymin) + pad_xy_nm
    lz = (zmax - zmin) + pad_z_nm + vacuum_z_nm
    return lx, ly, lz


def _ensure_periodic_box(MMsys):
    box_vecs = MMsys.simmd.topology.getPeriodicBoxVectors()
    if box_vecs is not None:
        return box_vecs

    box_from_env = _parse_box_nm_from_env()
    pad_xy_nm = float(os.environ.get("BOX_PADDING_XY_NM", os.environ.get("BOX_PADDING_NM", "0.2")))
    pad_z_nm = float(os.environ.get("BOX_PADDING_Z_NM", os.environ.get("BOX_PADDING_NM", "0.2")))
    vacuum_z_nm = float(os.environ.get("VACUUM_LAYER_Z_NM", "2.0"))
    if box_from_env is None:
        lx, ly, lz = _infer_box_nm_from_positions(MMsys, pad_xy_nm, pad_z_nm, vacuum_z_nm)
        source = "inferred from positions + vacuum"
    else:
        lx, ly, lz = box_from_env
        source = "PBC_BOX_NM/BOX_*_NM"

    # OpenMM API uses nanometer as implicit length unit for raw Vec3 box vectors.
    a = Vec3(lx, 0.0, 0.0)
    b = Vec3(0.0, ly, 0.0)
    c = Vec3(0.0, 0.0, lz)

    MMsys.modeller.topology.setPeriodicBoxVectors((a, b, c))
    MMsys.simmd.context.setPeriodicBoxVectors(a, b, c)
    MMsys.system.setDefaultPeriodicBoxVectors(a, b, c)
    print(
        "Periodic box was missing; set to ({:.4f}, {:.4f}, {:.4f}) nm [{}]".format(lx, ly, lz, source)
    )
    return MMsys.modeller.topology.getPeriodicBoxVectors()


simulation_time_ns = float(os.environ.get("SIM_TIME_NS", 1.0))
freq_charge_update_fs = int(os.environ.get("FREQ_CHARGE_UPDATE_FS", 50))
freq_traj_output_ps = int(os.environ.get("FREQ_TRAJ_OUTPUT_PS", 50))
freq_checkpoint_ps = int(os.environ.get("FREQ_CHECKPOINT_PS", 10))
write_charges = _env_bool("WRITE_CHARGES", True)

checkpoint = "state.chk"
charge_name = "charges.dat"

simulation_type = os.environ.get("SIM_MODE", "Constant_Q")
minimize_before_md = _env_bool("MINIMIZE_BEFORE_MD", simulation_type == "MC_equil")
minimize_iters = int(os.environ.get("MINIMIZE_ITERS", "2000"))
minimize_tol_kjmol = float(os.environ.get("MINIMIZE_TOL_KJMOL", "10.0"))
Voltage = float(os.environ.get("APPLIED_VOLTAGE", 0.0))
cathode_charge = float(os.environ.get("CATHODE_CHARGE", 0.0))
cutoff_nm = float(os.environ.get("CUTOFF_NM", "0.8"))

graphene_ffdir = os.environ.get(
    "GRAPHENE_FF_DIR", "../../Example_graphene_BMIM_BF4_ACN_10pct/graphene_ffdir/"
)
electrolyte_ff = os.environ.get("ELECTROLYTE_FF_XML", _pick_default_electrolyte_xml())
prepared_ff = os.environ.get("PREPARED_ELECTROLYTE_XML", "phyneo_ecl_om_interface_polar.xml")
interface_model = os.environ.get("INTERFACE_MODEL", "polarizable_mpid")

# Inert electrode MPID defaults:
# c0 = 0: avoid static multipole on inert carbon in MPID channel.
# alpha ~ 1e-8: practically non-polarizable (numerically stable, avoids singular solve).
electrode_mpid_c0 = os.environ.get("ELECTRODE_MPID_C0", "0.00000000")
electrode_mpid_alpha = os.environ.get("ELECTRODE_MPID_ALPHA", "1.0e-8")
electrode_mpid_thole = os.environ.get("ELECTRODE_MPID_THOLE", "0.33")

# Keep LJ in NonbondedForce off by default; vdW is handled by CustomNonbondedForce.
electrode_nb_sigma = float(os.environ.get("ELECTRODE_NB_SIGMA", 1.0))
electrode_nb_epsilon = float(os.environ.get("ELECTRODE_NB_EPSILON", 0.0))

cathode_index = (0, 2)
anode_index = (1, 3)

equilibrated_pdb = os.environ.get("EQUILIBRATED_PDB", "equilibrated.pdb")
packmol_pdb = os.environ.get("PACKMOL_PDB", "../packmol/li_electrolyte_box.pdb")
input_pdb = os.environ.get("INPUT_PDB")
if input_pdb is None:
    input_pdb = equilibrated_pdb if os.path.exists(equilibrated_pdb) else packmol_pdb

graph_residue_c_xml = graphene_ffdir + "graph_residue_c.xml"
graph_residue_n_xml = graphene_ffdir + "graph_residue_n.xml"
graph_xml = graphene_ffdir + "graph.xml"
graph_xml_compatible = os.environ.get("GRAPHENE_XML_COMPAT", "graph_interface_compatible.xml")
graph_c_freeze_xml = graphene_ffdir + "graph_c_freeze.xml"
graph_n_freeze_xml = graphene_ffdir + "graph_n_freeze.xml"
graph_custom_xml = graphene_ffdir + "graph_customnonbonded.xml"

electrolyte_ff_runtime = electrolyte_ff
if interface_model == "polarizable_mpid":
    n_custom, n_mpid_m, n_mpid_p = _prepare_polarizable_interface_xml(
        base_xml=electrolyte_ff,
        graph_custom_xml=graph_custom_xml,
        graph_residue_c_xml=graph_residue_c_xml,
        graph_residue_n_xml=graph_residue_n_xml,
        graph_c_freeze_xml=graph_c_freeze_xml,
        graph_n_freeze_xml=graph_n_freeze_xml,
        out_xml=prepared_ff,
        c0_default=electrode_mpid_c0,
        alpha_default=electrode_mpid_alpha,
        thole_default=electrode_mpid_thole,
    )
    electrolyte_ff_runtime = prepared_ff
    print("INTERFACE_MODEL=polarizable_mpid")
    print("Prepared xml:", electrolyte_ff_runtime)
    print("Patched CustomNonbonded classes:", n_custom)
    print("Patched MPID Multipole types:", n_mpid_m)
    print("Patched MPID Polarize types:", n_mpid_p)
else:
    raise ValueError("run_openMM_polar.py only supports INTERFACE_MODEL=polarizable_mpid")

_prepare_graph_xml_nonbonded_compatible(
    electrolyte_ff_runtime, graph_xml, graph_xml_compatible
)

MMsys = MM_FixedVoltage(
    pdb_list=[input_pdb],
    residue_xml_list=[graph_residue_c_xml, graph_residue_n_xml],
    ff_xml_list=[
        electrolyte_ff_runtime,
        graph_xml_compatible,
        graph_c_freeze_xml,
        graph_n_freeze_xml,
    ],
    cutoff=cutoff_nm,
)

MMsys.set_periodic_residue(True)
platform_name = os.environ.get("FV_PLATFORM", "Reference")
try:
    MMsys.set_platform(platform_name)
except Exception as exc:
    if platform_name == "OpenCL":
        print("OpenCL init failed ({}), falling back to CPU".format(exc))
        MMsys.set_platform("CPU")
    else:
        raise
_ensure_periodic_box(MMsys)

MMsys.initialize_electrodes(
    Voltage,
    cathode_identifier=cathode_index,
    anode_identifier=anode_index,
    chain=True,
    exclude_element=("H",),
)
MMsys.initialize_electrolyte(Natom_cutoff=100)
MMsys.generate_exclusions(flag_SAPT_FF_exclusions=False)

use_checkpoint = _env_bool("USE_CHECKPOINT", False)
if use_checkpoint and os.path.exists(checkpoint):
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
if simulation_type in ("Constant_V", "Constant_Q") and os.path.exists(trajectory_file_name):
    append_trajectory = True

MMsys.set_trajectory_output(
    trajectory_file_name,
    freq_traj_output_ps * 1000,
    append_trajectory,
    checkpoint,
    freq_checkpoint_ps * 1000,
)

if simulation_type == "MC_equil":
    if minimize_before_md:
        print(
            "Minimizing before MC_equil: iters={}, tol={} kJ/mol".format(
                minimize_iters, minimize_tol_kjmol
            )
        )
        MMsys.simmd.minimizeEnergy(minimize_tol_kjmol * kilojoule_per_mole, minimize_iters)
        state_min = MMsys.simmd.context.getState(getPositions=True)
        with open("minimized.pdb", "w") as handle:
            PDBFile.writeFile(MMsys.simmd.topology, state_min.getPositions(), handle)

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
    _set_electrode_fixed_charges(MMsys, cathode_charge, electrode_nb_sigma, electrode_nb_epsilon)
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
