import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
from functools import partial
import numpy as np
from openmm import *
from openmm.unit import *
from openmm.app import *

from ase import Atoms
from ase.io import read, write
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

from ase import units
from dmff.api import Hamiltonian
from dmff.common import nblist
from intra import onebodyenergy

def get_atoms_box(atoms):
    box = atoms.get_cell() / 10.0
    if len(box.flatten()) == 3:
        box = jnp.diag(box)
    else:
        box = jnp.array(box)
    box_inv = jnp.linalg.inv(box)
    return box, box_inv # matrix in nanometers

class DMFFCalculator(Calculator):
    implemented_properties = ['energy', 'forces', 'stress']
    
    def __init__(self, pdb, ff_xml, rc, **kwargs):
        super().__init__(**kwargs)
        self.results = {}

        r_xml = 'residues.xml'
        Topology.loadBondDefinitions(r_xml)
        mol = PDBFile(pdb) 

        self.topology = mol.topology
        pos = jnp.array(mol.positions._value) 
        box = jnp.array(mol.topology.getPeriodicBoxVectors()._value)
        
        H = Hamiltonian(ff_xml)
        pots = H.createPotential(
            mol.topology, 
            nonbondedCutoff=rc*nanometers,
            nonbondedMethod=PME, ethresh=1e-4, step_pol=10
        )
        self.efunc = pots.getPotentialFunc()
        self.params = H.getParameters()
       
        self.nbl = nblist.NeighborList(box, rc, pots.meta['cov_map'])
        self.nbl.allocate(pos, box)
        pairs = self.nbl.pairs

        # G_ = from_pdb(pdb)
        # G_h2o = from_pdb(f'special_residue/HOH.pdb')
        # model_ = MolGNNForce(G_h2o, nn=0, max_valence=6)
        # with open('params_sgnn.pickle', 'rb') as ifile:
        #     params_bond_ = pickle.load(ifile)
        # model_.batch_forward = vmap(model_.forward, in_axes=(0, None, None), out_axes=(0))

        def dmff_calculator(pos, box, pairs):
            E = self.efunc(pos, box, pairs, self.params)
            # E_bond_ = jnp.sum(model_.batch_forward(pos_ABn*10, box*10, params_bond_))
            E_bond_ = onebodyenergy(pos*10, box*10)  # compute intramolecular energy 
            # return E
            return E+E_bond_

        self.calc_dmff = jit(value_and_grad(dmff_calculator, argnums=(0, 1)))

        # compile functions
        energy, (grad, dE_dB) = self.calc_dmff(pos, box, pairs)
        print(f"Initial energy: {energy}")
        print(f"Initial dE_dB: {dE_dB}")

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        positions = jnp.array(atoms.get_positions()) / 10.0 
        box, box_inv = get_atoms_box(atoms)
        volume = jnp.linalg.det(box)
        # spositions = jnp.matmul(box_inv, positions.T).T
        self.nbl.update(positions, box)
        pairs = self.nbl.pairs

        energy, (grad, dE_dB) = self.calc_dmff(positions, box, pairs)
        # virial = (jnp.matmul(-grad.T, spositions) - dE_dB) * box
        virial = positions.T @ (-grad) - box.T @ dE_dB

        self.results['energy'] = energy.item() * 0.010364  # kj/mol to eV
        self.results['forces'] = np.array(-grad) * 0.010364 / 10.0
        self.results['stress'] = -full_3x3_to_voigt_6_stress(virial/volume) * 0.010364 / 1000


def print_energy(atoms, ofile='energy_output.txt'):
    """Print energy information"""
    epot = atoms.get_potential_energy() / len(atoms)
    ekin = atoms.get_kinetic_energy() / len(atoms)
    volume = atoms.get_volume()
    total_mass = sum(atoms.get_masses())
    density = total_mass * 1.66053907e-24 / (volume * 1e-24)
    
    energy_str = (f'Epot = {epot:.4f}eV  Ekin = {ekin:.4f}eV (T={ekin/(1.5 * units.kB):.0f}K)  '
                 f'Etot = {epot + ekin:.4f}eV  Density = {density:.4f} g/cm³\n')
    
    with open(ofile, 'a') as f:
        f.write(energy_str)
    print(energy_str, end='')

