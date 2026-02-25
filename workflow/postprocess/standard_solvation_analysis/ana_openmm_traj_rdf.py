# IO
import shutil
from pathlib import Path
from typing import List, Optional, Tuple, Dict

# md traj
import MDAnalysis as mda
from MDAnalysis import Universe
from MDAnalysis import transformations
from  MDAnalysis.analysis import rdf
import glob
import os

# plot
import seaborn as sns
import matplotlib.pyplot as plt

# data process
import palettable
import numpy as np
import pandas as pd
from sklearn import datasets
from pandas import Series,DataFrame
from scipy.integrate import simpson
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from collections import Counter

#General constant
kB=1.38064852e-23
e=1.60217662e-19
T=300.0
beta=1.0/(kB*T)

colors = [ 'r', 'b', 'g', 'c','k','orange']
mark=['*', '^', ',', 'o', 'v', '<', 'p', 'P', 'D', '3', '4', '8']


from typing import List, Union, Optional

#Define Function
def _find_first_shell_from_rdf(
    bins: np.ndarray,
    rdf: np.ndarray,
    smooth: bool = True,
    savgol_window: int = 11,
    savgol_poly: int = 2,
    prominence_frac: float = 0.02
) -> Tuple[float, int]:
    """Find rmin (first-shell cutoff) index based on first peak then the first minimum after it."""
    r = np.asarray(bins)
    g = np.asarray(rdf)
    if smooth and len(g) >= 5:
        win = int(savgol_window)
        if win % 2 == 0:
            win -= 1
        if win < 3:
            win = 3
        if win > len(g):
            win = len(g) if len(g) % 2 == 1 else len(g) - 1
        try:
            g_s = savgol_filter(g, win, savgol_poly)
        except Exception:
            g_s = g
    else:
        g_s = g

    peaks_max, _ = find_peaks(g_s)
    peaks_min, _ = find_peaks(-g_s, prominence=max(g_s) * prominence_frac if max(g_s) > 0 else 0.0)

    if peaks_max.size == 0:
        if peaks_min.size > 0:
            idx_min = int(peaks_min[0])
        else:
            idx_min = len(r) - 1
    else:
        first_peak_idx = int(peaks_max[0])
        mins_after = peaks_min[peaks_min > first_peak_idx]
        if mins_after.size > 0:
            idx_min = int(mins_after[0])
        else:
            idx_min = len(r) - 1

    return float(r[idx_min]), int(idx_min)


def get_rdf(
    u: Universe,
    resnames: List[str],
    names: List[str],
    cation: str = 'LI',
    res: Optional[str] = None,
    name: Optional[str] = None,
    start: int = 500,
    whether_to_plot: bool = True,
    figname: Optional[str] = None,
    legend_with_cn: bool = True,
    palette: Optional[List[str]] = None,
    linewidth: float = 1.6,
    alpha: float = 0.95,
    figsize: Tuple[float, float] = (6.0, 4.0),
    dpi: int = 300,
    show_grid: bool = True,
    show_rmin_lines: bool = True
) -> Dict[str, dict]:
    """
    Compute RDFs and cumulative CNs; optionally plot with improved aesthetics.

    New/changed plotting features:
    - Seaborn style and context for nicer fonts and grid.
    - Color palette usage (pass `palette` to override).
    - Thicker, smoother lines and clearer dashed CN curves.
    - CN values included in RDF legend entries when legend_with_cn=True.
    - Optional vertical dashed lines at first-shell cutoff (rmin) and subtle marker.
    - Legend placed outside to avoid overlap; crate boxed legend with smaller font.
    - Returns results dict (same structure as before) in all cases.

    Returns:
        results: dict keyed by descriptive label with fields:
          'bins','RDF','CN','N','box_volume','rmin','rmin_idx','CN_first_shell'
    """
    assert len(u.trajectory) != 0

    numb_res = len(resnames)
    assert len(names) == numb_res

    # plotting style
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.05)

    # prepare palette
    if palette is None:
        palette = sns.color_palette("tab10")  # default nice palette
    else:
        palette = list(palette)

    results: Dict[str, dict] = {}
    g_cation = u.select_atoms('resname {}'.format(cation))
    if res:
        if name:
            g = u.select_atoms('resname {} and name {}'.format(res, name))
        else:
            g = u.select_atoms('resname {}'.format(res))
    else:
        g = g_cation

    box_volume = float(np.prod(u.dimensions[:3])) if np.prod(u.dimensions[:3]) > 0 else 1.0

    for i in range(numb_res):
        g_res = u.select_atoms('resname {} and type {}'.format(resnames[i], names[i]))
        if len(g_res.atoms) == 0:
            print('CN:{} of {} is zero'.format(names[i], resnames[i]))
        else:
            RDF = __import__('MDAnalysis').analysis.rdf.InterRDF(g, g_res, nbins=200, range=(0, min(u.dimensions[:3]) / 2.0))
            RDF.run(start=start)

            N_particles = len(g_res.atoms)
            rho = float(N_particles) / box_volume if box_volume > 0 else 0.0

            cn = [0.0]
            bins = np.asarray(RDF.results.bins)
            rdf_vals = np.asarray(RDF.results.rdf)
            for j in range(1, len(rdf_vals)):
                cn_val = simpson(y=4.0 * np.pi * bins[:j]**2 * rdf_vals[:j] * rho, x=bins[:j])
                cn.append(float(cn_val))

            label = '{} - {} of {}'.format(res, names[i], resnames[i]) if res else '{} - {} of {}'.format(cation, names[i], resnames[i])
            results[label] = {
                'bins': list(bins),
                'RDF': list(rdf_vals),
                'CN': list(cn),
                'N': N_particles,
                'box_volume': box_volume
            }

            try:
                rmin, idx_min = _find_first_shell_from_rdf(bins, rdf_vals, smooth=True, savgol_window=11, savgol_poly=2, prominence_frac=0.02)
                cn_first_shell = float(cn[idx_min]) if idx_min < len(cn) else float(cn[-1])
            except Exception:
                rmin = float(bins[-1])
                idx_min = len(bins) - 1
                cn_first_shell = float(cn[-1])

            results[label]['rmin'] = float(rmin)
            results[label]['rmin_idx'] = int(idx_min)
            results[label]['CN_first_shell'] = float(cn_first_shell)

    # Plotting
    if whether_to_plot:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax2 = ax.twinx()

        # nice background and grid
        if show_grid:
            ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)

        handles = []
        handle_labels = []

        # Plot each series with palette cycling
        for idx, (k, v) in enumerate(results.items()):
            col = palette[idx % len(palette)]
            bins = np.asarray(v['bins'])
            rdf_vals = np.asarray(v['RDF'])
            cn_vals = np.asarray(v['CN'])

            cn_first = v.get('CN_first_shell', None)
            if legend_with_cn and cn_first is not None:
                rdf_label = f'{k} (CN={cn_first:.2f})'
            else:
                rdf_label = f'RDF: {k}'

            # smoother visual by plotting a small moving average for RDF (but keep raw for data)
            # use simple convolution if needed - keep simple for reproducibility
            ax_line, = ax.plot(bins, rdf_vals, color=col, linewidth=linewidth, alpha=alpha, label=rdf_label)
            cn_line, = ax2.plot(bins, cn_vals, color=col, linestyle='--', linewidth=max(1.0, linewidth-0.6), alpha=0.9)

            # optional vertical rmin line and marker on CN axis
            if show_rmin_lines and ('rmin' in v):
                rmin = v['rmin']
                cn_at_rmin = v['CN_first_shell']
                # vertical line on RDF axis (faint)
                ax.axvline(rmin, color=col, linestyle=':', linewidth=0.9, alpha=0.6)
                # marker on CN axis
                ax2.plot(rmin, cn_at_rmin, marker='x', color=col, markersize=6, markeredgewidth=1.6)

            handles.append(ax_line)
            handle_labels.append(rdf_label)

        # legend: put outside to the right to avoid overlapping curves
        if handles:
            # main legend (RDF lines with CN shown in label)
            leg = ax.legend(handles=handles, labels=handle_labels, loc='upper right',
                            borderaxespad=0., frameon=True, fontsize=9, title='RDF (first-shell CN)')
            leg.get_frame().set_edgecolor('gray')
            leg.get_frame().set_alpha(0.95)

        # Axis labels and limits
        ax.set_xlabel(r"Radius ($\AA$)", fontsize=10)
        ax.set_ylabel(r"g(r)", fontsize=10)
        ax2.set_ylabel("CN", fontsize=10)

        # reasonable default limits (can be adjusted by user after call)
        ax.set_xlim(0, 11)
        ax2.set_xlim(0, 11)
        # autoscale CN y to data with small margin
        cn_all = np.concatenate([np.asarray(v['CN']) for v in results.values()]) if results else np.array([0.0])
        cn_ymax = max(1.0, np.nanpercentile(cn_all, 99) * 1.15)
        ax.set_ylim(0, )
        ax2.set_ylim(0, 10)

        # title with concentration estimate (same as previous behavior)
        try:
            conc = round(len(g_cation) * 10000 / 6.022 / u.dimensions[0]**3, 2)
            title_m = f'{conc:.3f} M LiPF6'
        except Exception:
            title_m = ''
        if title_m:
            ax.set_title(title_m, fontsize=10)

        plt.tight_layout(rect=(0, 0, 0.88, 1.0))  # leave room for legend on right

        if figname:
            plt.savefig(figname, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

    return results

def get_atom_rdf(
    u:Universe, 
    name1: str,
    name2: str,
    distance: List[float]=[1.4, 2],
    start:int=500,
    whether_to_plot:bool=True):

    assert len(u.trajectory) != 0

    ga = u.select_atoms('name {}'.format(name1))
    gb = u.select_atoms('name {}'.format(name2))
    
    RDF = mda.analysis.rdf.InterRDF(ga, gb, nbins=200, range=distance)
    RDF.run(start=start)

    result = {'bins':list(RDF.results.bins), 'RDF':list(RDF.results.rdf)}

    if whether_to_plot:
        fig = plt.figure(figsize=(4.5,3.5))
        ax = fig.add_subplot(111)
        ax.plot(result['bins'], result['RDF'], color="orange", label='RDF: {}-{}'.format(name1, name2), alpha=0.8)

        ax.set_xlim(distance)
        ax.legend(loc="upper right")
        ax.set_xlabel(r"Radius ($\AA$)")
        ax.set_ylabel(r"g(r)")
        plt.tight_layout()
        plt.show()

    else:
        return result

vdw_radii = {
    'H': 1.20,
    'He': 1.40,

    'Li': 1.82,
    'Be': 1.53,
    'B': 1.92,
    'C': 1.70,
    'N': 1.55,
    'O': 1.52,
    'F': 1.47,
    'Ne': 1.54,

    'Na': 2.27,
    'Mg': 1.73,
    'Al': 1.84,
    'Si': 2.10,
    'P': 1.80,
    'S': 1.80,
    'Cl': 1.75,
    'Ar': 1.88,

    'K': 2.75,
    'Ca': 2.31,
    'Ga': 1.87,
    'Ge': 2.11,
    'As': 1.85,
    'Se': 1.90,
    'Br': 1.85,
    'Kr': 2.02,

    'Rb': 3.03,
    'Sr': 2.49,
    'In': 1.93,
    'Sn': 2.17,
    'Sb': 2.06,
    'Te': 2.06,
    'I': 1.98,
    'Xe': 2.16,
}

def get_coord_data(
    u:Universe, 
    resnames:List[str], 
    names:List[str], 
    cation:str='LI', 
    anion:Optional[str]=None,
    rcuts:Union[float, List[float]]=2.5,
    nres:Optional[int]=None,
    print_strs:bool=False,
    start:int=500,
    interval:int=10,
    max_dim: int=10,
    second_res:Optional[List[str]]=None,
    second_rcut:float=5):

    assert len(u.trajectory) != 0
    # u.atoms.guess_bonds(vdwradii=vdw_radii)
    
    numb_res = len(resnames)
    # assert len(names) == numb_res
    if isinstance(rcuts, list):
        assert len(rcuts) == numb_res
    elif isinstance(rcuts, (float,int)):
        rcuts = [rcuts]*numb_res

    if second_res:
        print(second_res)
        data = np.zeros([max_dim]*(numb_res+len(second_res)))
    else:
        data = np.zeros([max_dim]*numb_res)
    if print_strs:
        dirname = Path(u.filename).parent/"solv_strs"
        dirname.mkdir(exist_ok=True)
        
    if not anion:
        g_cation = u.select_atoms('resname {}'.format(cation))
        for ts in u.trajectory[start::interval]:
            for atom in g_cation.atoms:
                g_atom = mda.core.groups.AtomGroup([atom])
                g_sol_s = g_atom.copy()
                numb_coord_res = np.zeros((len(data.shape),), dtype=int)
                for i in range(numb_res):
                    # sol_atoms = u.select_atoms('resname {} and name {} and around {} group li'.format(resnames[i], names[i], rcuts[i]), li=g_atom)
                    sol_atoms = u.select_atoms('resname {} and around {} group li'.format(resnames[i], rcuts[i]), li=g_atom)
                    sol_res = sol_atoms.residues
                    ids_res = sol_res.resids
                    g_sol_s += sol_res.atoms
                    numb_coord_res[i] = len(ids_res)
                if second_res:   
                    for i in range(len(second_res)):
                        sol_atoms = u.select_atoms('resname {} and around {} group li'.format(second_res[i], second_rcut), li=g_atom)
                        sol_res = sol_atoms.residues
                        ids_res = sol_res.resids
                        g_sol_s += sol_res.atoms
                        numb_coord_res[numb_res+i] = len(ids_res)
                data[tuple(numb_coord_res)] += 1.0
                
                if print_strs:
                    sol_list = sorted(g_sol_s.residues.resnames.tolist())
                    numb_sol = [f"{sol_list.count(i)}{i}" for i in set(sol_list)]
                    if nres and len(sol_list)!=nres:
                        continue
                    # g_sol_s.unwrap(compound="residues", reference="cog")
                    pos = g_sol_s.positions
                    dr = u.dimensions[0]/2-pos[0]
                    pos += dr
                    pos = np.where(pos>0, pos, pos+u.dimensions[0])
                    pos = np.where(pos<u.dimensions[0], pos, pos-u.dimensions[0])
                    with open(f"{dirname}/frame{ts.frame}-{'-'.join(numb_sol)}-id{atom.resid}.xyz", "w") as f:
                        f.write(f"{len(g_sol_s)}\nframe {ts.frame}\n")
                        for idx, a in enumerate(g_sol_s):
                            f.write(f"{a.type:>6}{pos[idx][0]:>11.5f}{pos[idx][1]:>11.5f}{pos[idx][2]:>11.5f}\n")
        data = data/data.sum()*100

    else:
        g_anion = u.select_atoms('resname {}'.format(anion))
        for ts in u.trajectory[start::interval]:
            for residue in g_anion.residues:
                g_residue = residue.atoms.copy()
                g_sol_s = g_residue.copy()
                numb_coord_res = np.zeros((numb_res,), dtype=int)
                for i in range(numb_res):
                    # sol_atoms = u.select_atoms('resname {} and name {} and around {} group anion'.format(resnames[i], names[i], rcuts[i]), anion=g_residue)
                    sol_atoms = u.select_atoms('resname {} and around {} group anion'.format(resnames[i], rcuts[i]), anion=g_residue)
                    sol_res = sol_atoms.residues
                    ids_res = sol_res.resids
                    g_sol_s += sol_res.atoms
                    numb_coord_res[i] = len(ids_res)    
                data[tuple(numb_coord_res)] += 1.0
                if print_strs:
                    sol_list = g_sol_s.residues.resnames.tolist()
                    numb_sol = [f"{sol_list.count(i)}{i}" for i in set(sol_list)]
                    g_sol_s.unwrap(compound="residues", reference="cog")
                    pos = g_sol_s.positions
                    dr = u.dimensions[0]/2-pos[0]
                    pos += dr
                    pos = np.where(pos>0, pos, pos+u.dimensions[0])
                    pos = np.where(pos<u.dimensions[0], pos, pos-u.dimensions[0])
                    with open(f"{dirname}/frame{ts.frame}-{'-'.join(numb_sol)}-id{residue.resid}.xyz", "w") as f:
                        f.write(f"{len(g_sol_s)}\nframe {ts.frame}\n")
                        for idx, a in enumerate(g_sol_s):
                            f.write(f"{a.type:>6}{pos[idx][0]:>11.5f}{pos[idx][1]:>11.5f}{pos[idx][2]:>11.5f}\n")
        data = data/data.sum()*100

    return data


def get_ssip_cip_agg(data):

    ssip_cip_agg = [data[0,:].sum(), data[1,:].sum(), data[2:,:].sum()]

    return np.around(ssip_cip_agg, 3)



current_dir = "."
folder_list = glob.glob('test*')[:1]
paths = folder_list
paths.sort()
print(paths)
# second_res = ["BIDTD"]
for idx, folder in enumerate(sorted(paths)):
    # print(i)
    pdbf = f"{folder}/solvent_salt.pdb"
    trjf = f"{folder}/transport_results/nvt.dcd"
    print(idx, folder)
    
    if os.path.isdir(f"{folder}/solv_strs"):
        shutil.rmtree(f"{folder}/solv_strs")
    # os.makedirs(f"{i}/solv_strs",exist_ok=True)
    
    u = mda.Universe(pdbf, trjf, dt=0.002)
    resnames = list(set(u.residues.resnames))
    resnames.remove("LI")
    for jdx, i in enumerate(resnames):
        resnames[jdx] = i.replace("(","?").replace(")","?").replace("]","?")
    print(resnames)
    # RDF
    get_rdf(u, cation='LI', resnames=resnames, names=["O F N"]*len(resnames), start=1500, figname=f"{folder}/rdf.png")
    # Li+ solvation shell
    # get_coord_data(u, resnames=resnames, names=["O* N* F*"]*len(resnames), rcuts=2.5, interval=100, max_dim=7, print_strs=True)
    # second solvation shell
    # get_coord_data(u, resnames=resnames, names=["O* N* F*"]*len(resnames), rcuts=2.5, nres=6, interval=100, max_dim=10, print_strs=True, second_res=[second_res[idx]], second_rcut=5.5)
