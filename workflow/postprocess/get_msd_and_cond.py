#!/usr/bin/env python
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import MDAnalysis as mda
import MDAnalysis.transformations as trans
from MDAnalysis.analysis import msd
import scienceplots

# Set plotting style
plt.style.use(['science', 'no-latex'])

# Physical constants (SI units)
KB = 1.380649e-23  # Boltzmann constant (J/K)
NA = 6.02214076e23  # Avogadro's number (1/mol)
E_CHARGE = 1.602176634e-19  # Elementary charge (C)
EPS0 = 8.8541878128e-12  # Vacuum permittivity (F/m)

def get_dmff_uni(mol, pdb_file, traj_file):
    """Get MDAnalysis Universe object and set trajectory box information"""
    traj_file = os.path.join(mol, traj_file)
    u = mda.Universe(pdb_file, traj_file, dt=0.1)  # dt unit: ns (adjust according to your trajectory)
    #u = mda.Universe(pdb_file, traj_file, dt=0.1)  # dt unit: ns (adjust according to your trajectory)

    # Read and set box dimensions
    cell = []
    with open(traj_file, 'r') as fo:
        for line in fo.readlines():
            if line.startswith('# CELL(abcABC):'):
                words = line.split()
                cell.append([float(words[2]), float(words[3]), float(words[4])])  # Extract box dimensions (Å)
    
    # Apply box dimensions to each trajectory frame
    cell = np.array(cell)
    for iframe in range(len(u.trajectory)):
        ts = u.trajectory[iframe]
        if iframe < len(cell):  # Ensure no out-of-bounds error
            box = cell[iframe]
            # Convert to MDAnalysis format: [a, b, c, alpha, beta, gamma] (Å, degrees)
            ts.dimensions = np.array([box[0], box[1], box[2], 90., 90., 90.])
    
    return u

def calculate_diffusion_coefficient(u, selection, start_frame=500, plot_msd=False):
    """
    Calculate diffusion coefficient (m^2/s)
    selection: Atom selection string
    start_frame: First frame to analyze (skip equilibration)
    """
    # Select atoms
    atoms = u.select_atoms(selection)
    if len(atoms) == 0:
        raise ValueError(f"No atoms selected with: {selection}")
    
    # Calculate MSD (Mean Squared Displacement)
    MSD = msd.EinsteinMSD(u,
                  select=selection,
                  groupselections=None,
                  msd_type='xyz',  # 3D MSD
                  fft=True,        # Use FFT acceleration
                  start=start_frame)
    
    MSD.run()
    # average_msd = np.mean(MSD.results['msds_by_particle'], axis=1)

    # Extract results
    # msd_results = MSD.results.msd  # Units: Å^2
    # msd_results = MSD.results['msds_by_particle']
    msd_results = np.mean(MSD.results['msds_by_particle'],axis=1)
    times = MSD.times  # Units: ns (determined by dt)
    
    # Convert units: Å^2 -> m^2, ps -> s
    msd_m2 = msd_results * 1e-20  # 1 Å = 1e-10 m, 1 Å^2 = 1e-20 m^2
    times_s = times * 1e-12        # 1 ps = 1e-12 s
    
    # Linear fit to calculate diffusion coefficient (MSD = 6Dt)
    # Select linear region (typically middle portion of trajectory)
    fit_start = int(len(times_s) * 0.2)  # Skip initial portion
    fit_end = int(len(times_s) * 0.8)    # Skip final portion
    
    slope, intercept, r_value, p_value, std_err = linregress(
        times_s[fit_start:fit_end],
        msd_m2[fit_start:fit_end]
    )
    
    # Diffusion coefficient D = slope / 6
    diffusion_coeff = slope / 6
    
    # Plot MSD curve (optional)
    if plot_msd:
        plt.figure(figsize=(6, 4))
        plt.plot(times_s, msd_m2, label='MSD')
        plt.plot(times_s[fit_start:fit_end], 
                 intercept + slope * times_s[fit_start:fit_end], 
                 'r--', label=f'Fit: D = {diffusion_coeff:.2e} m²/s')
        plt.xlabel('Time (s)')
        plt.ylabel('MSD (m²)')
        plt.title(f'Mean Squared Displacement: {selection}')
        plt.legend()
        plt.savefig(f'msd_{selection.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Diffusion coefficient ({selection}): {diffusion_coeff:.2e} m²/s (R² = {r_value**2:.4f})")
    return diffusion_coeff

def calculate_conductivity(D_cation, D_anion, conc_mol_per_liter, temp_k=298):
    """
    Calculate conductivity (S/m) using Nernst-Einstein equation
    D_cation: Cation diffusion coefficient (m²/s)
    D_anion: Anion diffusion coefficient (m²/s)
    conc_mol_per_liter: Electrolyte concentration (mol/L)
    temp_k: Temperature (K)
    """
    # Convert concentration units: mol/L -> mol/m³ (1 mol/L = 1000 mol/m³)
    conc = conc_mol_per_liter * 1000
    
    # Calculate conductivity: σ = (N e² (D+ + D-)) / (kBT)
    conductivity = (conc * NA * E_CHARGE**2 * (D_cation + D_anion)) / (KB * temp_k)
    
    print(f"Conductivity: {conductivity:.4f} S/m")
    return conductivity

def main():
    # Input parameters
    path = '.'
    pdb_file = 'nvt_init.pdb'
    traj_file = 'nvt_ti.pos_0.1.xyz'
    cation_selection = "name Li01"  # Cation selection string (adjust for your topology)
    anion_selection = "name P02"    # Anion selection string (adjust for your topology)
    #concentration = 1.0             # Electrolyte concentration (mol/L)
    concentration = sys.argv[1]             # Electrolyte concentration (mol/L)
    temperature = 298               # Temperature (K)
    start_frame = 500               # First frame for analysis (skip equilibration)
    
    # Load trajectory data
    print("Loading trajectory file...")
    u = get_dmff_uni(path, pdb_file, traj_file)
    
    # Calculate cation diffusion coefficient
    print("\nCalculating cation diffusion coefficient...")
    D_cation = calculate_diffusion_coefficient(
        u, 
        cation_selection, 
        start_frame=start_frame, 
        plot_msd=True
    )
    
    # Calculate anion diffusion coefficient
    print("\nCalculating anion diffusion coefficient...")
    D_anion = calculate_diffusion_coefficient(
        u, 
        anion_selection, 
        start_frame=start_frame, 
        plot_msd=True
    )
    
    # Calculate conductivity
    print("\nCalculating conductivity...")
    conductivity = calculate_conductivity(D_cation, D_anion, concentration, temperature)
    
    # Save results to file
    with open('diffusion_conductivity_results.txt', 'w') as f:
        f.write(f"Cation diffusion coefficient: {D_cation:.6e} m²/s\n")
        f.write(f"Anion diffusion coefficient: {D_anion:.6e} m²/s\n")
        f.write(f"Electrolyte concentration: {concentration} mol/L\n")
        f.write(f"Temperature: {temperature} K\n")
        f.write(f"Conductivity: {conductivity:.6f} S/m\n")
    
    print("\nResults saved to diffusion_conductivity_results.txt")

if __name__ == "__main__":
    main()
