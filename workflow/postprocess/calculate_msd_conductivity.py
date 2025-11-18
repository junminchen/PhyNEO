#!/usr/bin/env python
"""
Calculate ionic conductivity and diffusion coefficients from MD trajectories using MSD method.

This script calculates:
1. Cation diffusion coefficient (D+)
2. Anion diffusion coefficient (D-)
3. Ionic conductivity (σ) using the Nernst-Einstein equation

Usage:
    python calculate_msd_conductivity.py --pdb init.pdb --traj trajectory.xyz \
           --cation "name Li" --anion "name PF6" --conc 1.0 --temp 298

Author: PhyNEO Contributors
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import MDAnalysis as mda
from MDAnalysis.analysis import msd

# Physical constants (SI units)
KB = 1.380649e-23        # Boltzmann constant (J/K)
NA = 6.02214076e23       # Avogadro's number (1/mol)
E_CHARGE = 1.602176634e-19  # Elementary charge (C)
EPS0 = 8.8541878128e-12  # Vacuum permittivity (F/m)


def get_universe_from_ipi(pdb_file, traj_file, dt=0.1):
    """
    Load trajectory from i-Pi output format and set box dimensions.
    
    Parameters:
    -----------
    pdb_file : str
        Path to PDB topology file
    traj_file : str
        Path to trajectory file (.xyz format from i-Pi)
    dt : float
        Timestep in picoseconds
        
    Returns:
    --------
    u : MDAnalysis.Universe
        Universe object with trajectory loaded
    """
    # Create Universe with trajectory
    u = mda.Universe(pdb_file, traj_file, dt=dt)
    
    # Read and set box dimensions from i-Pi trajectory
    cell = []
    with open(traj_file, 'r') as fo:
        for line in fo.readlines():
            if line.startswith('# CELL(abcABC):'):
                words = line.split()
                # Extract box dimensions (in Angstroms)
                cell.append([float(words[2]), float(words[3]), float(words[4])])
    
    # Apply box dimensions to each trajectory frame
    cell = np.array(cell)
    for iframe in range(len(u.trajectory)):
        ts = u.trajectory[iframe]
        if iframe < len(cell):
            box = cell[iframe]
            # MDAnalysis format: [a, b, c, alpha, beta, gamma] (Å, degrees)
            ts.dimensions = np.array([box[0], box[1], box[2], 90., 90., 90.])
    
    return u


def calculate_diffusion_coefficient(u, selection, start_frame=500, 
                                    fit_start_frac=0.2, fit_end_frac=0.8,
                                    plot_msd=False, output_prefix=''):
    """
    Calculate diffusion coefficient from MSD using Einstein relation.
    
    Parameters:
    -----------
    u : MDAnalysis.Universe
        Universe object with trajectory
    selection : str
        Atom selection string (e.g., "name Li")
    start_frame : int
        First frame to analyze (skip equilibration)
    fit_start_frac : float
        Starting fraction of trajectory for linear fit (0.0-1.0)
    fit_end_frac : float
        Ending fraction of trajectory for linear fit (0.0-1.0)
    plot_msd : bool
        Whether to plot MSD curve
    output_prefix : str
        Prefix for output files
        
    Returns:
    --------
    diffusion_coeff : float
        Diffusion coefficient in m²/s
    r_squared : float
        R² value of linear fit
    """
    # Select atoms
    atoms = u.select_atoms(selection)
    if len(atoms) == 0:
        raise ValueError(f"No atoms selected with: {selection}")
    
    print(f"  Selected {len(atoms)} atoms with selection: {selection}")
    
    # Calculate MSD using Einstein relation
    MSD = msd.EinsteinMSD(u,
                          select=selection,
                          msd_type='xyz',  # 3D MSD
                          fft=True,        # Use FFT acceleration
                          start=start_frame)
    
    MSD.run()
    
    # Extract results
    msd_results = np.mean(MSD.results['msds_by_particle'], axis=1)  # Å²
    times = MSD.times  # ps (determined by dt)
    
    # Convert units: Å² -> m², ps -> s
    msd_m2 = msd_results * 1e-20  # 1 Å² = 1e-20 m²
    times_s = times * 1e-12         # 1 ps = 1e-12 s
    
    # Linear fit to calculate diffusion coefficient (MSD = 6Dt)
    # Select linear region (typically middle portion of trajectory)
    fit_start = int(len(times_s) * fit_start_frac)
    fit_end = int(len(times_s) * fit_end_frac)
    
    slope, intercept, r_value, p_value, std_err = linregress(
        times_s[fit_start:fit_end],
        msd_m2[fit_start:fit_end]
    )
    
    # Diffusion coefficient D = slope / 6 (for 3D)
    diffusion_coeff = slope / 6
    r_squared = r_value ** 2
    
    # Plot MSD curve
    if plot_msd:
        try:
            import scienceplots
            plt.style.use(['science', 'no-latex', 'nature'])
        except:
            pass
        
        plt.figure(figsize=(6, 4))
        plt.plot(times_s * 1e12, msd_m2 * 1e20, 'b-', linewidth=1.5, label='MSD')
        plt.plot(times_s[fit_start:fit_end] * 1e12, 
                 (intercept + slope * times_s[fit_start:fit_end]) * 1e20,
                 'r--', linewidth=2, 
                 label=f'Fit: D = {diffusion_coeff:.2e} m²/s\nR² = {r_squared:.4f}')
        plt.xlabel('Time (ps)')
        plt.ylabel('MSD (Ų)')
        plt.title(f'Mean Squared Displacement\n{selection}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        filename = f'{output_prefix}msd_{selection.replace(" ", "_")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  MSD plot saved to: {filename}")
    
    print(f"  Diffusion coefficient: {diffusion_coeff:.6e} m²/s (R² = {r_squared:.4f})")
    return diffusion_coeff, r_squared


def calculate_conductivity(D_cation, D_anion, conc_mol_per_liter, 
                           z_cation=1, z_anion=-1, temp_k=298):
    """
    Calculate ionic conductivity using Nernst-Einstein equation.
    
    σ = (N * e² / (kB * T)) * (n+ * z+² * D+ + n- * z-² * D-)
    
    For 1:1 electrolyte with equal concentrations:
    σ = (c * NA * e² / (kB * T)) * (D+ + D-)
    
    Parameters:
    -----------
    D_cation : float
        Cation diffusion coefficient (m²/s)
    D_anion : float
        Anion diffusion coefficient (m²/s)
    conc_mol_per_liter : float
        Electrolyte concentration (mol/L)
    z_cation : int
        Cation charge number (default: +1)
    z_anion : int
        Anion charge number (default: -1)
    temp_k : float
        Temperature (K)
        
    Returns:
    --------
    conductivity : float
        Ionic conductivity in S/m
    """
    # Convert concentration: mol/L -> mol/m³ (1 mol/L = 1000 mol/m³)
    conc = conc_mol_per_liter * 1000
    
    # Nernst-Einstein equation for 1:1 electrolyte
    # σ = (N * e² * (D+ + D-)) / (kB * T)
    conductivity = (conc * NA * E_CHARGE**2 * (D_cation + D_anion)) / (KB * temp_k)
    
    print(f"\nConductivity: {conductivity:.4f} S/m ({conductivity*10:.4f} mS/cm)")
    return conductivity


def main():
    parser = argparse.ArgumentParser(
        description='Calculate ionic conductivity and diffusion coefficients from MD trajectories using MSD.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with i-Pi trajectory
  python calculate_msd_conductivity.py --pdb init.pdb --traj trajectory.xyz \\
         --cation "name Li" --anion "name PF6" --conc 1.0
  
  # With custom parameters
  python calculate_msd_conductivity.py --pdb init.pdb --traj trajectory.xyz \\
         --cation "name Li01" --anion "name P02" --conc 1.2 --temp 338 \\
         --dt 0.1 --start-frame 1000 --plot
  
  # Different ion types
  python calculate_msd_conductivity.py --pdb init.pdb --traj trajectory.xyz \\
         --cation "resname LiA" --anion "resname FSI" --conc 1.5 --plot

Notes:
  - The trajectory file should be in i-Pi .xyz format with box information
  - Concentration should be in mol/L
  - Temperature should be in Kelvin
  - Default timestep is 0.1 ps (adjust with --dt if different)
        """
    )
    
    # Required arguments
    parser.add_argument('--pdb', required=True, 
                        help='PDB topology file')
    parser.add_argument('--traj', required=True,
                        help='Trajectory file (.xyz format from i-Pi)')
    parser.add_argument('--cation', required=True,
                        help='Cation selection string (e.g., "name Li")')
    parser.add_argument('--anion', required=True,
                        help='Anion selection string (e.g., "name PF6")')
    parser.add_argument('--conc', type=float, required=True,
                        help='Electrolyte concentration (mol/L)')
    
    # Optional arguments
    parser.add_argument('--temp', type=float, default=298,
                        help='Temperature in Kelvin (default: 298)')
    parser.add_argument('--dt', type=float, default=0.1,
                        help='Timestep in picoseconds (default: 0.1)')
    parser.add_argument('--start-frame', type=int, default=500,
                        help='First frame for analysis (default: 500)')
    parser.add_argument('--fit-start', type=float, default=0.2,
                        help='Starting fraction for linear fit (default: 0.2)')
    parser.add_argument('--fit-end', type=float, default=0.8,
                        help='Ending fraction for linear fit (default: 0.8)')
    parser.add_argument('--plot', action='store_true',
                        help='Generate MSD plots')
    parser.add_argument('--output-prefix', type=str, default='',
                        help='Prefix for output files')
    
    args = parser.parse_args()
    
    # Print header
    print("="*70)
    print("MSD-based Conductivity and Diffusion Coefficient Calculator")
    print("="*70)
    print(f"\nInput files:")
    print(f"  Topology: {args.pdb}")
    print(f"  Trajectory: {args.traj}")
    print(f"\nParameters:")
    print(f"  Cation selection: {args.cation}")
    print(f"  Anion selection: {args.anion}")
    print(f"  Concentration: {args.conc} mol/L")
    print(f"  Temperature: {args.temp} K")
    print(f"  Timestep: {args.dt} ps")
    print(f"  Start frame: {args.start_frame}")
    print(f"  Fit range: {args.fit_start:.1%} - {args.fit_end:.1%} of trajectory")
    
    # Check if files exist
    if not os.path.exists(args.pdb):
        print(f"\nError: PDB file not found: {args.pdb}")
        sys.exit(1)
    if not os.path.exists(args.traj):
        print(f"\nError: Trajectory file not found: {args.traj}")
        sys.exit(1)
    
    # Load trajectory
    print("\n" + "-"*70)
    print("Loading trajectory...")
    print("-"*70)
    u = get_universe_from_ipi(args.pdb, args.traj, dt=args.dt)
    print(f"Loaded {len(u.trajectory)} frames")
    print(f"Total simulation time: {len(u.trajectory) * args.dt:.2f} ps")
    
    # Calculate cation diffusion coefficient
    print("\n" + "-"*70)
    print("Calculating CATION diffusion coefficient...")
    print("-"*70)
    D_cation, r2_cation = calculate_diffusion_coefficient(
        u, 
        args.cation, 
        start_frame=args.start_frame,
        fit_start_frac=args.fit_start,
        fit_end_frac=args.fit_end,
        plot_msd=args.plot,
        output_prefix=args.output_prefix
    )
    
    # Calculate anion diffusion coefficient
    print("\n" + "-"*70)
    print("Calculating ANION diffusion coefficient...")
    print("-"*70)
    D_anion, r2_anion = calculate_diffusion_coefficient(
        u, 
        args.anion, 
        start_frame=args.start_frame,
        fit_start_frac=args.fit_start,
        fit_end_frac=args.fit_end,
        plot_msd=args.plot,
        output_prefix=args.output_prefix
    )
    
    # Calculate conductivity
    print("\n" + "-"*70)
    print("Calculating ionic conductivity...")
    print("-"*70)
    conductivity = calculate_conductivity(
        D_cation, D_anion, args.conc, temp_k=args.temp
    )
    
    # Save results
    output_file = f'{args.output_prefix}diffusion_conductivity_results.txt'
    print("\n" + "-"*70)
    print("Results Summary")
    print("-"*70)
    with open(output_file, 'w') as f:
        lines = [
            "="*70,
            "MSD-based Conductivity and Diffusion Coefficient Results",
            "="*70,
            "",
            "Input Parameters:",
            f"  Topology file: {args.pdb}",
            f"  Trajectory file: {args.traj}",
            f"  Cation selection: {args.cation}",
            f"  Anion selection: {args.anion}",
            f"  Concentration: {args.conc} mol/L",
            f"  Temperature: {args.temp} K",
            f"  Timestep: {args.dt} ps",
            f"  Analysis start frame: {args.start_frame}",
            f"  Fit range: {args.fit_start:.1%} - {args.fit_end:.1%}",
            "",
            "Results:",
            f"  Cation diffusion coefficient: {D_cation:.6e} m²/s (R² = {r2_cation:.4f})",
            f"  Anion diffusion coefficient:  {D_anion:.6e} m²/s (R² = {r2_anion:.4f})",
            f"  Ionic conductivity: {conductivity:.6f} S/m ({conductivity*10:.6f} mS/cm)",
            "",
            "Notes:",
            "  - Diffusion coefficients calculated using Einstein relation: MSD = 6Dt",
            "  - Conductivity calculated using Nernst-Einstein equation",
            "  - Assumes 1:1 electrolyte with equal ion concentrations",
            "="*70
        ]
        
        for line in lines:
            print(line)
            f.write(line + '\n')
    
    print(f"\nResults saved to: {output_file}")
    print("\n" + "="*70)
    print("Calculation completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
