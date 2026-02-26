#!/usr/bin/env python3
import argparse

import jax.numpy as jnp
from dmff.api import Hamiltonian
from dmff.common import nblist
from openmm.app import CutoffPeriodic, NoCutoff, PDBFile
from openmm.unit import angstrom


class DMFFEnergyCalculator:
    def __init__(self, ff_file, pdb_file, method="cutoff", cutoff_angstrom=25.0, box_nm=6.0, step_pol=20):
        self.ff = ff_file
        self.pdb = PDBFile(pdb_file)
        self.positions = jnp.array(self.pdb.positions._value)

        if self.pdb.topology.getPeriodicBoxVectors() is not None:
            a, b, c = self.pdb.topology.getPeriodicBoxVectors()
            self.box = jnp.array([a._value, b._value, c._value])
        else:
            self.box = jnp.eye(3) * box_nm

        nb_method = CutoffPeriodic if method == "cutoff" else NoCutoff

        self.H = Hamiltonian(self.ff)
        self.potentials_obj = self.H.createPotential(
            self.pdb.topology,
            nonbondedCutoff=cutoff_angstrom * angstrom,
            nonbondedMethod=nb_method,
            ethresh=1e-4,
            step_pol=step_pol,
        )
        self.params = self.H.getParameters()

        rc_nm = cutoff_angstrom * 0.1
        self.nblist = nblist.NeighborList(self.box, rc_nm, self.potentials_obj.meta["cov_map"])
        self.nblist.allocate(self.positions)
        self.pairs = self.nblist.pairs
        self.pairs = self.pairs[self.pairs[:, 0] < self.pairs[:, 1]]

        self.potentials_mapping = {
            "espol": "ADMPPmeForce",
            "disp": "ADMPDispPmeForce",
            "ex": "SlaterExForce",
            "sr_es": "SlaterSrEsForce",
            "sr_pol": "SlaterSrPolForce",
            "sr_disp": "SlaterSrDispForce",
            "dhf": "SlaterDhfForce",
            "dmp_es": "QqTtDampingForce",
            "dmp_disp": "SlaterDampingForce",
        }

    def compute_components(self):
        energy_dict = {}
        for key, force_name in self.potentials_mapping.items():
            try:
                potential_func = self.potentials_obj.getPotentialFunc(force_name)
                energy_dict[key] = potential_func(self.positions, self.box, self.pairs, self.params)
            except Exception:
                continue
        return energy_dict

    def compute_total(self):
        return self.potentials_obj.getPotentialFunc()(self.positions, self.box, self.pairs, self.params)


def main():
    parser = argparse.ArgumentParser(description="Compute DMFF energy components")
    parser.add_argument("--xml", default="EC.xml", help="DMFF XML file")
    parser.add_argument("--pdb", default="EC.pdb", help="PDB file")
    parser.add_argument("--method", choices=["cutoff", "nocutoff"], default="cutoff")
    parser.add_argument("--cutoff-angstrom", type=float, default=25.0)
    parser.add_argument("--box-nm", type=float, default=6.0)
    parser.add_argument("--step-pol", type=int, default=20)
    args = parser.parse_args()

    calc = DMFFEnergyCalculator(
        ff_file=args.xml,
        pdb_file=args.pdb,
        method=args.method,
        cutoff_angstrom=args.cutoff_angstrom,
        box_nm=args.box_nm,
        step_pol=args.step_pol,
    )

    components = calc.compute_components()
    print("DMFF Energy Components:")
    for k, v in components.items():
        print(f"{k}: {v}")

    total = calc.compute_total()
    print(f"DMFF Total Energy: {total}")


if __name__ == "__main__":
    main()
