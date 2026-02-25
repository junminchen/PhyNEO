#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys

import openmm as mm
import openmm.app as app
import openmm.unit as unit

KJMOL_PER_E_PER_VOLT = 96.48533212331002


def parse_chain_indices(text: str):
    return tuple(int(x.strip()) for x in text.split(',') if x.strip())


def collect_chain_atoms(topology: app.Topology, chain_indices, exclude_elements=('H',)):
    out = []
    ex = set(exclude_elements)
    for chain in topology.chains():
        if chain.index in chain_indices:
            for atom in chain.atoms():
                symbol = atom.element.symbol if atom.element is not None else ''
                if symbol not in ex:
                    out.append(atom.index)
    return sorted(out)


def main():
    parser = argparse.ArgumentParser(description='OpenMM 8.4 native ConstantPotentialForce: electrode + NaCl(aq).')
    parser.add_argument('--pdb', default='nacl_water_start.pdb')
    parser.add_argument('--platform', default='CPU', choices=['Reference', 'CPU', 'OpenCL', 'CUDA'])
    parser.add_argument('--temperature-k', type=float, default=300.0)
    parser.add_argument('--friction-ps', type=float, default=1.0)
    parser.add_argument('--timestep-fs', type=float, default=1.0)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--report-interval', type=int, default=200)
    parser.add_argument('--cutoff-nm', type=float, default=1.0)
    parser.add_argument('--voltage-v', type=float, default=1.0)
    parser.add_argument('--cathode-chains', default='0')
    parser.add_argument('--anode-chains', default='1')
    parser.add_argument('--gaussian-width-nm', type=float, default=0.2)
    parser.add_argument('--thomas-fermi-scale-invnm', type=float, default=5.0)
    parser.add_argument('--cg-error-tol', type=float, default=1e-3)
    parser.add_argument('--use-charge-constraint', action='store_true')
    parser.add_argument('--charge-constraint-target-e', type=float, default=0.0)
    parser.add_argument('--traj', default='nacl_water_native84.dcd')
    parser.add_argument('--charge-log', default='nacl_water_native84_charges.dat')
    parser.add_argument('--checkpoint', default='nacl_water_native84.chk')
    args = parser.parse_args()

    pdb = app.PDBFile(args.pdb)
    ff = app.ForceField('amber14/tip3p.xml', 'electrode_residues.xml', 'electrode_ff.xml')
    system = ff.createSystem(
        pdb.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=args.cutoff_nm * unit.nanometer,
        constraints=app.HBonds,
        rigidWater=True,
        removeCMMotion=False,
    )

    nb = next(f for f in (system.getForce(i) for i in range(system.getNumForces())) if isinstance(f, mm.NonbondedForce))

    cathode_atoms = collect_chain_atoms(pdb.topology, parse_chain_indices(args.cathode_chains))
    anode_atoms = collect_chain_atoms(pdb.topology, parse_chain_indices(args.anode_chains))
    cpf = mm.ConstantPotentialForce()
    cpf.setCutoffDistance(float(args.cutoff_nm))
    cpf.setConstantPotentialMethod(mm.ConstantPotentialForce.CG)
    cpf.setCGErrorTolerance(float(args.cg_error_tol))
    cpf.setUseChargeConstraint(bool(args.use_charge_constraint))
    if args.use_charge_constraint:
        cpf.setChargeConstraintTarget(float(args.charge_constraint_target_e))

    for i in range(system.getNumParticles()):
        q, sig, eps = nb.getParticleParameters(i)
        q_e = q.value_in_unit(unit.elementary_charge)
        cpf.addParticle(float(q_e))
        nb.setParticleParameters(i, 0.0 * unit.elementary_charge, sig, eps)

    for ex in range(nb.getNumExceptions()):
        p1, p2, qprod, sig, eps = nb.getExceptionParameters(ex)
        qprod_e2 = qprod.value_in_unit(unit.elementary_charge**2)
        cpf.addException(int(p1), int(p2), float(qprod_e2))
        nb.setExceptionParameters(ex, p1, p2, 0.0 * unit.elementary_charge**2, sig, eps)

    cath_pot = float(args.voltage_v) * KJMOL_PER_E_PER_VOLT
    ano_pot = -float(args.voltage_v) * KJMOL_PER_E_PER_VOLT
    cpf.addElectrode(set(cathode_atoms), cath_pot, float(args.gaussian_width_nm), float(args.thomas_fermi_scale_invnm))
    cpf.addElectrode(set(anode_atoms), ano_pot, float(args.gaussian_width_nm), float(args.thomas_fermi_scale_invnm))

    system.addForce(cpf)
    for i in range(system.getNumForces()):
        system.getForce(i).setForceGroup(i)

    integrator = mm.LangevinMiddleIntegrator(
        args.temperature_k * unit.kelvin,
        args.friction_ps / unit.picosecond,
        args.timestep_fs * unit.femtosecond,
    )
    platform = mm.Platform.getPlatformByName(args.platform)
    sim = app.Simulation(pdb.topology, system, integrator, platform)
    sim.context.setPositions(pdb.positions)

    print(f'cathode atoms: {len(cathode_atoms)}, anode atoms: {len(anode_atoms)}')
    print('initial energy:', sim.context.getState(getEnergy=True).getPotentialEnergy())

    sim.reporters.append(app.DCDReporter(args.traj, args.report_interval))
    sim.reporters.append(app.StateDataReporter(
        sys.stdout, args.report_interval,
        step=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True,
        temperature=True, density=True, speed=True,
    ))
    sim.reporters.append(app.CheckpointReporter(args.checkpoint, args.report_interval))

    cath_particles = list(cpf.getElectrodeParameters(0)[0])
    ano_particles = list(cpf.getElectrodeParameters(1)[0])
    with open(args.charge_log, 'w') as f:
        f.write('# step Q_cathode(e) Q_anode(e) Q_total(e)\n')
        blocks = args.steps // args.report_interval
        rem = args.steps % args.report_interval
        for _ in range(blocks):
            sim.step(args.report_interval)
            charges = list(cpf.getCharges(sim.context))
            q_c = sum(float(charges[i].value_in_unit(unit.elementary_charge)) for i in cath_particles)
            q_a = sum(float(charges[i].value_in_unit(unit.elementary_charge)) for i in ano_particles)
            f.write(f'{sim.currentStep} {q_c:.10f} {q_a:.10f} {(q_c+q_a):.10f}\n')
        if rem:
            sim.step(rem)

    print('final energy:', sim.context.getState(getEnergy=True).getPotentialEnergy())
    print('done')


if __name__ == '__main__':
    main()
