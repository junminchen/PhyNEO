#!/usr/bin/env python
import numpy as np

import jax
import jax.numpy as jnp
from jax import value_and_grad, vmap, jit

from openmm.app import PDBFile
from openmm.unit import angstrom
from openmm.app import CutoffPeriodic
from functools import partial
import pickle

from dmff.api import Hamiltonian
from dmff.utils import jit_condition
from dmff.common import nblist

import time 
import optax
import sys 

# from jax.config import config
# config.update("jax_debug_nans", True)
# config.update("jax_platform_name", "gpu")

def padding(i):
    s = '%d'%i
    while len(s) < 3:
        s = '0' + s
    return s

def params_convert(params):
    params_ex = {}
    params_sr_es = {}
    params_sr_pol = {}
    params_sr_disp = {}
    params_dhf = {}
    params_dmp_es = {}  # electrostatic damping
    params_dmp_disp = {} # dispersion damping
    for k in ['B', 'mScales']:
        params_ex[k] = params[k]
        params_sr_es[k] = params[k]
        params_sr_pol[k] = params[k]
        params_sr_disp[k] = params[k]
        params_dhf[k] = params[k]
        params_dmp_es[k] = params[k]
        params_dmp_disp[k] = params[k]
    params_ex['A'] = params['A_ex']
    params_sr_es['A'] = params['A_es']
    params_sr_pol['A'] = params['A_pol']
    params_sr_disp['A'] = params['A_disp']
    params_dhf['A'] = params['A_dhf']
    # damping parameters
    params_dmp_es['Q'] = params['Q']
    params_dmp_disp['C6'] = params['C6']
    params_dmp_disp['C8'] = params['C8']
    params_dmp_disp['C10'] = params['C10']
    p = {}
    p['SlaterExForce'] = params_ex
    p['SlaterSrEsForce'] = params_sr_es
    p['SlaterSrPolForce'] = params_sr_pol
    p['SlaterSrDispForce'] = params_sr_disp
    p['SlaterDhfForce'] = params_dhf
    p['QqTtDampingForce'] = params_dmp_es
    p['SlaterDampingForce'] = params_dmp_disp
    return p

class BasePairs:
    def __init__(self, ff, pdb, pdb_A, pdb_B):
        pdb = PDBFile(pdb)
        pdb_A = PDBFile(pdb_A)
        pdb_B = PDBFile(pdb_B)
        self.H = Hamiltonian(ff)
        self.pots = self.H.createPotential(pdb.topology, nonbondedCutoff=25*angstrom, nonbondedMethod=CutoffPeriodic, ethresh=1e-4)
        self.pots_A = self.H.createPotential(pdb_A.topology, nonbondedCutoff=25*angstrom, nonbondedMethod=CutoffPeriodic, ethresh=1e-4)
        self.pots_B = self.H.createPotential(pdb_B.topology, nonbondedCutoff=25*angstrom, nonbondedMethod=CutoffPeriodic, ethresh=1e-4)

        self.pos = jnp.array(pdb.positions._value) * 10
        self.pos_A = jnp.array(pdb_A.positions._value) * 10
        self.pos_B = jnp.array(pdb_B.positions._value) * 10

        self.box = jnp.array(pdb.topology.getPeriodicBoxVectors()._value) * 10
        self.rc = 25
        self.nblist = nblist.NeighborList(self.box, self.rc, self.pots.meta['cov_map'])
        self.nblist_A = nblist.NeighborList(self.box, self.rc, self.pots_A.meta['cov_map'])
        self.nblist_B = nblist.NeighborList(self.box, self.rc, self.pots_B.meta['cov_map'])
        self.nblist.allocate(self.pos)
        self.nblist_A.allocate(self.pos_A)
        self.nblist_B.allocate(self.pos_B)
        self.pairs = self.nblist.pairs
        self.pairs_A = self.nblist_A.pairs
        self.pairs_B = self.nblist_B.pairs
        self.pairs_AB = self.pairs[self.pairs[:, 0] < self.pairs[:, 1]]
        self.pairs_A = self.pairs_A[self.pairs_A[:, 0] < self.pairs_A[:, 1]]
        self.pairs_B = self.pairs_B[self.pairs_B[:, 0] < self.pairs_B[:, 1]]

        self.potentials_names = ['ex', 'sr_es', 'sr_pol', 'sr_disp', 'dhf', 'dmp_es', 'dmp_disp']
        self.potentials_mapping = {
            'ex': 'SlaterExForce',
            'sr_es': 'SlaterSrEsForce',
            'sr_pol': 'SlaterSrPolForce',
            'sr_disp': 'SlaterSrDispForce',
            'dhf': 'SlaterDhfForce',
            'dmp_es': 'QqTtDampingForce',
            'dmp_disp': 'SlaterDampingForce',
        }
        
        for potentials_name in self.potentials_names:
            print(potentials_name)
            setattr(self, f'pots_{potentials_name}', self.pots.dmff_potentials[self.potentials_mapping[potentials_name]])
            setattr(self, f'pots_{potentials_name}_A', self.pots_A.dmff_potentials[self.potentials_mapping[potentials_name]])
            setattr(self, f'pots_{potentials_name}_B', self.pots_B.dmff_potentials[self.potentials_mapping[potentials_name]])

    def cal_E(self, params, pos_A, pos_B):
        # get position array
        params = params_convert(params)
        pos_AB = jnp.concatenate([pos_A, pos_B], axis=0)
        box = self.box
        #####################
        # exchange repulsion
        #####################
        E_ex = self.pots_ex(pos_AB, box, self.pairs_AB, params)\
               - self.pots_ex_A(pos_A, box, self.pairs_A, params)\
               - self.pots_ex_B(pos_B, box, self.pairs_B, params)

        #######################
        # electrostatic
        #######################
        E_dmp_es = self.pots_dmp_es(pos_AB, box, self.pairs_AB, params) \
                    - self.pots_dmp_es_A(pos_A, box, self.pairs_A, params) \
                    - self.pots_dmp_es_B(pos_B, box, self.pairs_B, params)
        E_sr_es = self.pots_sr_es(pos_AB, box, self.pairs_AB, params) \
                - self.pots_sr_es_A(pos_A, box, self.pairs_A, params) \
                - self.pots_sr_es_B(pos_B, box, self.pairs_B, params)

        ###################################
        # polarization (induction) energy
        ###################################
        E_sr_pol = self.pots_sr_pol(pos_AB, box, self.pairs_AB, params) \
                    - self.pots_sr_pol_A(pos_A, box, self.pairs_A, params) \
                    - self.pots_sr_pol_B(pos_B, box, self.pairs_B, params)

        #############
        # dispersion
        #############
        E_dmp_disp = self.pots_dmp_disp(pos_AB, box, self.pairs_AB, params) \
                    - self.pots_dmp_disp_A(pos_A, box, self.pairs_A, params) \
                    - self.pots_dmp_disp_B(pos_B, box, self.pairs_B, params)
        E_sr_disp = self.pots_sr_disp(pos_AB, box, self.pairs_AB, params) \
                    - self.pots_sr_disp_A(pos_A, box, self.pairs_A, params) \
                    - self.pots_sr_disp_B(pos_B, box, self.pairs_B, params)

        ###########
        # dhf
        ###########
        E_dhf = self.pots_dhf(pos_AB, box, self.pairs_AB, params) \
                 - self.pots_dhf_A(pos_A, box, self.pairs_A, params) \
                 - self.pots_dhf_B(pos_B, box, self.pairs_B, params)

        E_es = E_dmp_es + E_sr_es
        E_pol = E_sr_pol
        E_disp = E_dmp_disp + E_sr_disp
        E_tot = E_ex + E_es + E_pol + E_disp + E_dhf
        return E_ex, E_es, E_pol, E_disp, E_dhf, E_tot 

@jit
def MSELoss(params, data):
    '''
    The weighted mean squared error loss function
    Conducted for each scan
    '''
    # batch = padding(batch)
    scan_res = data
    comps = ['ex', 'es', 'pol', 'disp', 'dhf', 'tot']
    weights_comps = jnp.array([0.1, 0.1, 0.5, 0.1, 0.1, 1.0])

    E_tot_full = scan_res['tot_full']
    kT = 2.494 # 300 K = 2.494 kJ/mol
    weights_pts = jnp.piecewise(E_tot_full, [E_tot_full<25, E_tot_full>=25], [lambda x: jnp.array(1.0), lambda x: jnp.exp(-(x-25)/kT)])
    npts = len(weights_pts)

    energies = {
            'ex': jnp.zeros(npts),
            'es': jnp.zeros(npts),
            'pol': jnp.zeros(npts),
            'disp': jnp.zeros(npts),
            'dhf': jnp.zeros(npts),
            'tot': jnp.zeros(npts)
            }

    E_ex, E_es, E_pol, E_disp, E_dhf, E_tot = cal_energy[key](params, scan_res['posA'], scan_res['posB'])
    
    for ipt in range(npts):
        energies['ex'] = energies['ex'].at[ipt].set(E_ex[ipt])
        energies['es'] = energies['es'].at[ipt].set(E_es[ipt])
        energies['pol'] = energies['pol'].at[ipt].set(E_pol[ipt])
        energies['disp'] = energies['disp'].at[ipt].set(E_disp[ipt])
        energies['dhf'] = energies['dhf'].at[ipt].set(E_dhf[ipt])
        energies['tot'] = energies['tot'].at[ipt].set(E_tot[ipt])

    errs = jnp.zeros(len(comps))
    for ic, c in enumerate(comps):
        dE = energies[c] - scan_res[c]
        mse = dE**2 * weights_pts / jnp.sum(weights_pts)
        errs = errs.at[ic].set(jnp.sum(mse))
    loss = jnp.sum(weights_comps * errs)
    return loss

if __name__ == '__main__':
    restart = None
    params0 = Hamiltonian('dmff_forcefield.xml').getParameters()
    comps = ['ex', 'es', 'pol', 'disp', 'dhf', 'tot']
    if restart is None:
        params = {}
        sr_forces = {
                'ex': 'SlaterExForce',
                'es': 'SlaterSrEsForce',
                'pol': 'SlaterSrPolForce',
                'disp': 'SlaterSrDispForce',
                'dhf': 'SlaterDhfForce',
                }
        for k in params0['ADMPPmeForce']:
            params[k] = params0['ADMPPmeForce'][k]
        for k in params0['ADMPDispPmeForce']:
            params[k] = params0['ADMPDispPmeForce'][k]
        for c in comps:
            if c == 'tot':
                continue
            force = sr_forces[c]
            for k in params0[sr_forces[c]]:
                if k == 'A':
                    params['A_'+c] = params0[sr_forces[c]][k]
                else:
                    params[k] = params0[sr_forces[c]][k]
        # a random initialization of A
        for c in comps:
            if c == 'tot':
                continue
            params['A_'+c] = jnp.array(np.random.random(params['A_'+c].shape))
        # specify charges for es damping
        params['Q'] = params0['QqTtDampingForce']['Q']
    else:
        with open(restart, 'rb') as ifile:
            params = pickle.load(ifile)

    with open('data_sr.pickle', 'rb') as ifile:
        data = pickle.load(ifile)

    pair_classes = [
        ("_pe_dimer", "pe6_dimer.pdb", "pe6.pdb", "pe6.pdb"),
        # Add other definition
    ]

    class_instances = {}
    # Loop to create subclasses and add them to the global namespace
    for pair_class_name, dimer_file, monomer_A_file, monomer_B_file in pair_classes:
        class_definition = f"""
class Pairs{pair_class_name}(BasePairs):
    def __init__(self):
        super().__init__('dmff_forcefield.xml', 'dimer/{dimer_file}', 'monomer/{monomer_A_file}', 'monomer/{monomer_B_file}')
        """
        exec(class_definition)

        # Instantiate the class and add it to the dictionary
        class_instances[f'Pairs{pair_class_name}'] = globals()[f'Pairs{pair_class_name}']()

    cal_energy = {}
    for class_name, class_instance in class_instances.items():
        cal_energy[class_name] = jit(vmap(class_instance.cal_E, in_axes=(None, 0, 0), out_axes=(0, 0, 0, 0, 0, 0)))

    
    batch = padding(0)
    cal_energy['Pairs_pe_dimer'](params, data['Pairs_pe_dimer']['000']['posA'], data['Pairs_pe_dimer']['000']['posB']) 
    # save the jit object 
    MSELoss_grad = {}
    for key in data.keys():
        MSELoss_grad[key] = jit(value_and_grad(MSELoss, argnums=(0)))
        err, gradients = MSELoss_grad[key](params, data[key][batch])
        print(key, err)

    # only optimize these parameters A/B
    def mask_fn(grads):
        for k in grads:
            if k.startswith('A_') or k == 'B':
                continue
            else:
                grads[k] = 0.0
        return grads

    # start to do optmization
    lr = 0.1
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)
    n_epochs = 50

    time0 = time.time()
    for i_epoch in range(n_epochs):
        print('***************')
        print(time.time() - time0) 
        data_keys = list(data.keys())
        np.random.shuffle(data_keys)
        #np.random.shuffle(data_key)
        for batch in range(0, 1000, 20):
            batch = padding(batch)
            for key0 in data_keys:
                loss, grad = MSELoss_grad[key0](params, data[key0][batch])
                print(loss)
                grad = mask_fn(grad)
                sys.stdout.flush()
                updates, opt_state = optimizer.update(grad, opt_state)
                params = optax.apply_updates(params, updates)
        with open('params.pickle', 'wb') as ofile:
           pickle.dump(params, ofile)

    energies = {'ex': [], 'es': [], 'pol': [], 'disp': [], 'dhf': [], 'tot': []}
    energies_ref = {'ex': [], 'es': [], 'pol': [], 'disp': [], 'dhf': [], 'tot': []}
    # keys = ['PairsCoccoDimer']\
    from tqdm import tqdm
    for key in tqdm(data.keys()):
        E = {}
        for sid in data[key].keys():
            scan_res = data[key][sid]
            E_tot_full = scan_res['tot_full']
            kT = 2.494 # 300 K = 2.494 kJ/mol
            weights_pts = jnp.piecewise(E_tot_full, [E_tot_full<25, E_tot_full>=25], [lambda x: jnp.array(1.0), lambda x: jnp.exp(-(x-25)/kT)])
            pos_A = jnp.array(scan_res['posA'])
            pos_B = jnp.array(scan_res['posB'])
            E['ex'], E['es'], E['pol'], E['disp'], E['dhf'], E['tot'] = cal_energy[key](params, pos_A, pos_B)
            E_ref = scan_res
            npts = len(E_ref)
            for ipt in range(npts):
                if weights_pts[ipt] > 1e-2:
                    for component in energies.keys():
                        energies[component].append(E[component][ipt])
                        energies_ref[component].append(E_ref[component][ipt])
    rmsd = []
    for component in energies:
        energies[component] = np.array(energies[component])
        energies_ref[component] = np.array(energies_ref[component])
        dE = energies[component] - energies_ref[component]
        rmsd.append(np.sqrt(np.average(dE**2)))
    print(rmsd)


    import matplotlib.pyplot as plt
    import scienceplots
    from math import ceil
    # Calculate the number of components
    num_components = len(energies)

    # Calculate the number of rows and columns for the subplots
    num_rows = 2
    num_cols = ceil(num_components / num_rows)

    # Create a figure with subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 6))
    plt.style.use(['science', 'no-latex'])

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Define colors
    colors = ['#82B0D2', '#BEB8DC', '#E7DAD2','#FFBE7A', '#8ECFC9', '#FA7F6F']

    # fig.text(0.5, 0.04, "Reference Energy (kJ/mol)", ha='center')
    # fig.text(0.04, 0.5, "Fitted Energy (kJ/mol)", va='center', rotation='vertical')

    # Iterate through the components and create subplots
    for i, component in enumerate(energies):
        ax = axes[i]
        x_data = energies_ref[component]
        a = np.min([np.min(x_data)])
        b = np.max([np.max(x_data)])
        ax.axis([a, b, a, b])
        ax.set_xlabel("Reference Energy (kJ/mol)")
        ax.set_ylabel(f"Fitted {component} Energy (kJ/mol)")
        ax.axline((0, 0), slope=1, linewidth=1.5, color="k", alpha=0.8)
        ax.scatter(energies_ref[component], energies[component], color=colors[i], s=10, alpha=0.6, edgecolors='None', label=component)
        ax.legend(fontsize=10, loc='upper left')
        ax.text(0.98, 0.05, f'RMSD = {rmsd[i]:.2f} (kJ/mol)',
                verticalalignment='bottom', horizontalalignment='right',
                transform=ax.transAxes)

    # Hide any empty subplots
    for i in range(num_components, num_rows * num_cols):
        fig.delaxes(axes[i])

    # Adjust subplot layout and spacing
    plt.tight_layout()
    plt.legend()
    # Save the figure
    plt.savefig('test_decomp3.png', dpi=300)
    # plt.show()

    # from tqdm import tqdm
    
    # energies = []
    # energies_ref = []
    # # keys = ['PairsCoccoDimer']
    # for key in tqdm(data.keys()):
    #     for sid in data[key].keys():
    #         scan_res = data[key][sid]
    #         E_tot_full = scan_res['tot_full']
    #         kT = 2.494 # 300 K = 2.494 kJ/mol
    #         weights_pts = jnp.piecewise(E_tot_full, [E_tot_full<25, E_tot_full>=25], [lambda x: jnp.array(1.0), lambda x: jnp.exp(-(x-25)/kT)])
    #         pos_A = jnp.array(scan_res['posA'])
    #         pos_B = jnp.array(scan_res['posB'])
    #         E_tot = cal_energy[key](params, pos_A, pos_B)[-1]
    #         E_ref = scan_res['tot']
    #         npts = len(E_ref)

    #         for ipt in range(npts):
    #             if weights_pts[ipt] > 1e-2:
    #                 energies.append(E_tot[ipt])
    #                 energies_ref.append(E_ref[ipt])

    # energies = np.array(energies)
    # energies_ref = np.array(energies_ref)

    # dE = energies - energies_ref
    # rmsd = np.sqrt(np.average(dE**2))
    # print(rmsd)

    # import matplotlib.pyplot as plt
    # import scienceplots
    # plt.figure(figsize=(7.5,5))
    # plt.style.use(['science','no-latex'])
    # a = np.min([np.min(energies),np.min(energies)])
    # b = np.max([np.max(energies_ref),np.max(energies_ref)])
    # plt.axis([a-10, b+1, a-10, b+1])    
    # plt.xlabel("Reference Energy (kJ/mol)")
    # plt.ylabel("Fitted Energy (kJ/mol)")
    # plt.axline((0, 0), slope=1, linewidth=1.5, color="k",alpha=0.8)
    # plt.scatter(energies_ref, energies, s=10, alpha=0.4, edgecolors='None')
    # ax = plt.gca()
    # ax.text(0.98, 0.05, '\nRMSD = %.2f (kJ/mol)'%(rmsd),
    #         verticalalignment='bottom', horizontalalignment='right',
    #         transform=ax.transAxes)
    # plt.savefig('test_300.png',dpi=300)