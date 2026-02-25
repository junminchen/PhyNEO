#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import openmm as mm
import openmm.app as app
import openmm.unit as unit

KJMOL_PER_E_PER_VOLT = 96.48533212331002


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenMM 8.4 native ConstantPotentialForce example with legacy input files."
    )
    parser.add_argument("--pdb", default="equilibrated.pdb")
    parser.add_argument("--residue-xml", nargs="+", default=["sapt_residues.xml", "../graphene_ffdir/graph_residue_c.xml", "../graphene_ffdir/graph_residue_n.xml"])
    parser.add_argument("--ff-xml", nargs="+", default=["sapt_add.xml", "../graphene_ffdir/graph.xml", "../graphene_ffdir/graph_c_freeze.xml", "../graphene_ffdir/graph_n_freeze.xml"])
    parser.add_argument("--sapt-base-xml", default="sapt.xml")
    parser.add_argument("--graph-custom-xml", default="../graphene_ffdir/graph_customnonbonded.xml")
    parser.add_argument("--sapt-combined-xml", default="sapt_add.xml")
    parser.add_argument("--skip-sapt-merge", action="store_true", help="Skip sapt.xml + graph_customnonbonded.xml merge.")
    parser.add_argument("--platform", default="OpenCL", choices=["Reference", "CPU", "OpenCL", "CUDA"])
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--friction-ps", type=float, default=1.0)
    parser.add_argument("--timestep-fs", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--report-interval", type=int, default=5000)
    parser.add_argument("--traj", default="")
    parser.add_argument("--charge-log", default="charges_openmm84.dat")
    parser.add_argument("--checkpoint", default="state_openmm84.chk")
    parser.add_argument("--checkpoint-interval", type=int, default=10000)
    parser.add_argument("--voltage-v", type=float, default=2.0, help="Applied voltage magnitude. Electrodes are set to +V and -V.")
    parser.add_argument("--cathode-chains", default="0,2")
    parser.add_argument("--anode-chains", default="1,3")
    parser.add_argument("--exclude-element", default="H")
    parser.add_argument("--gaussian-width-nm", type=float, default=0.02)
    parser.add_argument("--thomas-fermi-scale-invnm", type=float, default=0.0)
    parser.add_argument("--virtual-stride", type=int, default=10, help="Use every N-th real electrode atom to build virtual electrodes.")
    parser.add_argument("--electrode-mode", choices=["virtual", "direct"], default="virtual")
    parser.add_argument("--cutoff-nm", type=float, default=1.2)
    parser.add_argument("--cg-error-tol", type=float, default=1.0e-4)
    parser.add_argument("--use-charge-constraint", action="store_true")
    parser.add_argument("--charge-constraint-target-e", type=float, default=0.0)
    parser.add_argument("--minimize", action="store_true")
    parser.add_argument("--minimize-max-iter", type=int, default=200)
    return parser.parse_args()


def _parse_chain_indices(text: str) -> tuple[int, ...]:
    text = text.strip()
    if not text:
        return tuple()
    return tuple(int(x.strip()) for x in text.split(",") if x.strip())


def _collect_chain_atoms(topology: app.Topology, chain_indices: tuple[int, ...], exclude_element: set[str]) -> list[int]:
    out: list[int] = []
    for chain in topology.chains():
        if chain.index in chain_indices:
            for atom in chain.atoms():
                symbol = atom.element.symbol if atom.element is not None else ""
                if symbol not in exclude_element:
                    out.append(atom.index)
    return sorted(out)


def _require_force(system: mm.System, cls: type) -> mm.Force:
    for i in range(system.getNumForces()):
        f = system.getForce(i)
        if isinstance(f, cls):
            return f
    raise RuntimeError(f"Required force not found: {cls.__name__}")


def patch_missing_bonds_from_templates(topology: app.Topology, ff: app.ForceField) -> int:
    """Add missing intra-residue bonds from ff templates for PDBs without CONECT records."""
    existing = set()
    for a1, a2 in topology.bonds():
        i, j = a1.index, a2.index
        existing.add((i, j) if i < j else (j, i))

    added = 0
    for residue in topology.residues():
        atoms = list(residue.atoms())
        if len(atoms) < 2:
            continue
        has_intra = False
        for a1, a2 in topology.bonds():
            if a1.residue.index == residue.index and a2.residue.index == residue.index:
                has_intra = True
                break
        if has_intra:
            continue

        template = ff._templates.get(residue.name, None)  # type: ignore[attr-defined]
        if template is None:
            continue
        if len(template.atoms) != len(atoms):
            continue

        by_name = {}
        duplicate = False
        for atom in atoms:
            if atom.name in by_name:
                duplicate = True
                break
            by_name[atom.name] = atom
        if duplicate:
            continue

        mapped = []
        ok = True
        for t_atom in template.atoms:
            if t_atom.name not in by_name:
                ok = False
                break
            mapped.append(by_name[t_atom.name])
        if not ok:
            continue

        for i, j in template.bonds:
            ai = mapped[i]
            aj = mapped[j]
            key = (ai.index, aj.index) if ai.index < aj.index else (aj.index, ai.index)
            if key not in existing:
                topology.addBond(ai, aj)
                existing.add(key)
                added += 1
    return added


def merge_sapt_customnonbond(xml_base: str, xml_param: str, xml_out: str) -> None:
    tree_base = ET.parse(xml_base)
    root_base = tree_base.getroot()
    cnb_base = root_base.find("CustomNonbondedForce")
    if cnb_base is None:
        raise RuntimeError(f"CustomNonbondedForce not found in {xml_base}")

    tree_param = ET.parse(xml_param)
    root_param = tree_param.getroot()
    cnb_param = root_param.find("CustomNonbondedForce")
    if cnb_param is None:
        raise RuntimeError(f"CustomNonbondedForce not found in {xml_param}")

    for atom in cnb_param.findall("Atom"):
        cnb_base.append(atom)
    tree_base.write(xml_out)


def build_system(args: argparse.Namespace) -> tuple[app.Topology, list, mm.System, mm.ConstantPotentialForce]:
    if not args.skip_sapt_merge:
        merge_sapt_customnonbond(args.sapt_base_xml, args.graph_custom_xml, args.sapt_combined_xml)

    pdb = app.PDBFile(args.pdb)
    ff = app.ForceField(*args.residue_xml, *args.ff_xml)
    n_added = patch_missing_bonds_from_templates(pdb.topology, ff)
    if n_added > 0:
        print(f"[build] Added {n_added} missing intra-residue bonds from templates")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    try:
        modeller.addExtraParticles(ff)
        print("[build] Added extra particles from FF templates")
    except Exception as e:
        fallback = Path(args.pdb).with_name("start_drudes.pdb")
        if Path(args.pdb).name == "equilibrated.pdb" and fallback.exists():
            print(f"[build] addExtraParticles failed for {args.pdb}: {e}")
            print(f"[build] Fallback to {fallback.name}")
            pdb = app.PDBFile(str(fallback))
            patch_missing_bonds_from_templates(pdb.topology, ff)
            modeller = app.Modeller(pdb.topology, pdb.positions)
        else:
            print(f"[build] addExtraParticles skipped: {e}")
    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=args.cutoff_nm * unit.nanometer,
        constraints=None,
        removeCMMotion=False,
    )

    nb: mm.NonbondedForce = _require_force(system, mm.NonbondedForce)  # type: ignore[assignment]
    exclude = {x.strip() for x in args.exclude_element.split(",") if x.strip()}
    cathode_chains = _parse_chain_indices(args.cathode_chains)
    anode_chains = _parse_chain_indices(args.anode_chains)
    cathode_real_full = _collect_chain_atoms(modeller.topology, cathode_chains, exclude)
    anode_real_full = _collect_chain_atoms(modeller.topology, anode_chains, exclude)
    stride = max(1, int(args.virtual_stride))
    cathode_real = cathode_real_full[::stride]
    anode_real = anode_real_full[::stride]
    real_electrode_atom_set = set(cathode_real) | set(anode_real)
    if not cathode_real or not anode_real:
        raise RuntimeError("Electrode atom selection is empty. Check chain indices and exclude-element.")

    positions = list(modeller.positions)
    cathode_virtual: list[int] = []
    anode_virtual: list[int] = []
    if args.electrode_mode == "virtual":
        # Add virtual electrode particles (decoupled from LJ and bonded terms).
        custom_nbs = []
        for fi in range(system.getNumForces()):
            f = system.getForce(fi)
            if isinstance(f, mm.CustomNonbondedForce):
                custom_nbs.append(f)

        for i_real in cathode_real:
            idx = system.addParticle(0.0)
            cathode_virtual.append(idx)
            positions.append(positions[i_real])
            nb.addParticle(0.0 * unit.elementary_charge, 1.0 * unit.nanometer, 0.0 * unit.kilojoule_per_mole)
            for cnb in custom_nbs:
                cnb.addParticle([0.0] * cnb.getNumPerParticleParameters())

        for i_real in anode_real:
            idx = system.addParticle(0.0)
            anode_virtual.append(idx)
            positions.append(positions[i_real])
            nb.addParticle(0.0 * unit.elementary_charge, 1.0 * unit.nanometer, 0.0 * unit.kilojoule_per_mole)
            for cnb in custom_nbs:
                cnb.addParticle([0.0] * cnb.getNumPerParticleParameters())

    cpf = mm.ConstantPotentialForce()
    cpf.setCutoffDistance(float(args.cutoff_nm))
    cpf.setConstantPotentialMethod(mm.ConstantPotentialForce.CG)
    cpf.setCGErrorTolerance(float(args.cg_error_tol))
    cpf.setUseChargeConstraint(bool(args.use_charge_constraint))
    if args.use_charge_constraint:
        cpf.setChargeConstraintTarget(float(args.charge_constraint_target_e))

    # Move all Coulomb terms into CPF; in virtual mode, real electrode atoms are fixed neutral in CPF.
    original_q = []
    n0 = nb.getNumParticles()
    for i in range(n0):
        q, sig, eps = nb.getParticleParameters(i)
        q_e = q.value_in_unit(unit.elementary_charge)
        q_for_cpf = float(q_e)
        if args.electrode_mode == "virtual" and i in real_electrode_atom_set:
            q_for_cpf = 0.0
        cpf.addParticle(q_for_cpf)
        original_q.append(float(q_e))
        nb.setParticleParameters(i, 0.0 * unit.elementary_charge, sig, eps)

    for ex in range(nb.getNumExceptions()):
        p1, p2, qprod, sig, eps = nb.getExceptionParameters(ex)
        p1 = int(p1)
        p2 = int(p2)
        qprod_e2 = qprod.value_in_unit(unit.elementary_charge**2)
        if args.electrode_mode == "virtual" and (p1 in real_electrode_atom_set or p2 in real_electrode_atom_set):
            qprod_e2 = 0.0
        cpf.addException(p1, p2, float(qprod_e2))
        nb.setExceptionParameters(ex, p1, p2, 0.0 * unit.elementary_charge**2, sig, eps)

    # Keep input convention from the legacy script: applied magnitude V gives +V (cathode) and -V (anode).
    cathode_potential = float(args.voltage_v) * KJMOL_PER_E_PER_VOLT
    anode_potential = -float(args.voltage_v) * KJMOL_PER_E_PER_VOLT
    if args.electrode_mode == "virtual":
        cathode_set = set(cathode_virtual)
        anode_set = set(anode_virtual)
    else:
        cathode_set = set(cathode_real)
        anode_set = set(anode_real)
    cpf.addElectrode(cathode_set, cathode_potential, float(args.gaussian_width_nm), float(args.thomas_fermi_scale_invnm))
    cpf.addElectrode(anode_set, anode_potential, float(args.gaussian_width_nm), float(args.thomas_fermi_scale_invnm))

    system.addForce(cpf)
    for i in range(system.getNumForces()):
        system.getForce(i).setForceGroup(i)

    print("[build] NonbondedForce charges moved to ConstantPotentialForce")
    print(f"[build] CPF exceptions added={cpf.getNumExceptions()}")
    print(f"[build] Electrode mode: {args.electrode_mode}")
    print(f"[build] Real electrode atoms selected: {len(real_electrode_atom_set)} (stride={stride})")
    print(f"[build] CPF electrode atoms: cathode={len(cathode_set)} anode={len(anode_set)}")
    print(f"[build] Potential: cathode={cathode_potential:.6f} kJ/mol/e anode={anode_potential:.6f} kJ/mol/e")
    print(f"[build] Total initial fixed charge (from FF): {sum(original_q):.8f} e")
    return modeller.topology, positions, system, cpf


def print_energy_decomposition(sim: app.Simulation, system: mm.System, title: str) -> None:
    print(f"\n[{title}]")
    total = sim.context.getState(getEnergy=True).getPotentialEnergy()
    print(f"Total: {total}")
    for i in range(system.getNumForces()):
        f = system.getForce(i)
        e = sim.context.getState(getEnergy=True, groups=1 << i).getPotentialEnergy()
        print(f"{i:2d} {f.__class__.__name__:<30s} {e}")


def main() -> None:
    args = parse_args()
    topology, positions, system, cpf = build_system(args)

    integrator = mm.LangevinMiddleIntegrator(
        args.temperature_k * unit.kelvin,
        args.friction_ps / unit.picosecond,
        args.timestep_fs * unit.femtosecond,
    )
    platform = mm.Platform.getPlatformByName(args.platform)
    sim = app.Simulation(topology, system, integrator, platform)
    sim.context.setPositions(positions)

    if args.minimize:
        print("[run] Minimizing ...")
        sim.minimizeEnergy(maxIterations=args.minimize_max_iter)
    else:
        print("[run] Skip minimization")
    print_energy_decomposition(sim, system, "initial energy")

    if args.traj:
        sim.reporters.append(app.DCDReporter(args.traj, args.report_interval))
    sim.reporters.append(
        app.StateDataReporter(
            sys.stdout,
            args.report_interval,
            step=True,
            temperature=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True,
            volume=True,
            density=True,
            speed=True,
        )
    )
    sim.reporters.append(app.CheckpointReporter(args.checkpoint, args.checkpoint_interval))

    cathode_idx = 0
    anode_idx = 1
    with open(args.charge_log, "w") as fout:
        fout.write("# step Q_cathode(e) Q_anode(e) Q_total(e)\n")
        nblocks = args.steps // args.report_interval
        rem = args.steps % args.report_interval
        cath_particles = list(cpf.getElectrodeParameters(cathode_idx)[0])
        anode_particles = list(cpf.getElectrodeParameters(anode_idx)[0])
        for _ in range(nblocks):
            sim.step(args.report_interval)
            charges = list(cpf.getCharges(sim.context))
            q_cath = sum(float(charges[i].value_in_unit(unit.elementary_charge)) for i in cath_particles)
            q_anod = sum(float(charges[i].value_in_unit(unit.elementary_charge)) for i in anode_particles)
            q_tot = q_cath + q_anod
            step = sim.currentStep
            fout.write(f"{step} {q_cath:.10f} {q_anod:.10f} {q_tot:.10f}\n")
            fout.flush()
        if rem > 0:
            sim.step(rem)

    print_energy_decomposition(sim, system, "final energy")
    print("[run] done")


if __name__ == "__main__":
    main()
