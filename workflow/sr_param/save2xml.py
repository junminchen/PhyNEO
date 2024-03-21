#!/usr/bin/env python
import xml.etree.ElementTree as ET
import pickle
from dmff.api import Hamiltonian
def params_convert(params):
    params_ex = {}
    params_sr_es = {}
    params_sr_pol = {}
    params_sr_disp = {}
    params_dhf = {}
    params_dmp_es = {}  # electrostatic damping
    params_dmp_disp = {} # dispersion damping
    for k in ['B']:
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

# get params or restart from fitted params
def get_params(restart, params0):
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
    return params


#paramsfile = '../params/params.formula.wts.pickle'
# paramsfile = '../params/params.solvents.anion.Li.Na.pickle'
# paramsfile = '../params/params.all.pickle'
# paramsfile = '../params/params.all.0.01.pickle'

paramsfile = 'params.pickle'
ff_file = 'dmff_forcefield.xml'
params0 = Hamiltonian(ff_file).getParameters()
params = get_params(paramsfile, params0)
force_values = params_convert(params)

# 读取XML文件
tree = ET.parse(ff_file)
root = tree.getroot()

for force_type, values in force_values.items():
    print(force_type)
    for elem in root.iter(force_type):
        for i, atom_elem in enumerate(elem.iter('Atom')):
            if 'A' in values:
                atom_elem.set('A', str(float(values['A'][i])))
            if 'B' in values:
                atom_elem.set('B', str(float(values['B'][i])))

save_model = f"output"
print(f'Dump to {save_model}.xml')
# 将修改后的XML树写入新的XML文件
tree.write(f'{save_model}.xml')
