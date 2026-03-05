import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from pathlib import Path

def plot_charge_timeseries(out_dir):
    file_path = out_dir / 'electrode_total_charge_timeseries.dat'
    if not file_path.exists():
        print(f"{file_path} not found.")
        return

    # Use space separator and handle comment
    df = pd.read_csv(file_path, sep=r'\s+', comment='#', names=['step', 'Q_left_e', 'Q_right_e', 'Q_total_e'])
    plt.figure(figsize=(10, 6))
    plt.plot(df['step'], df['Q_left_e'], label='Q_left')
    plt.plot(df['step'], df['Q_right_e'], label='Q_right')
    plt.plot(df['step'], df['Q_total_e'], label='Q_total', linestyle='--')
    plt.xlabel('Step')
    plt.ylabel('Charge (e)')
    plt.title(f'Electrode Total Charge Timeseries - {out_dir.parent.name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(out_dir / 'charge_timeseries.png')
    plt.close()
    print(f"Saved {out_dir / 'charge_timeseries.png'}")

def plot_z_density(out_dir):
    file_path = out_dir / 'stage3_production/z_number_density_profile.csv'
    if not file_path.exists():
        print(f"{file_path} not found.")
        return

    df = pd.read_csv(file_path)
    
    # Create two subplots: one for COM, one for Specific Atoms
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Define which labels go to the bottom plot (Specific Atoms)
    specific_suffixes = ['_O_carbonyl', '_Li', '_F']
    
    # Plot Total and COM species in ax1
    for col in df.columns:
        if col == 'z_center_angstrom': continue
        label = col.replace('number_density_', '').replace('_nm^-3', '')
        
        # Determine if it is a specific atom or a residue COM
        is_specific = any(suffix in label for suffix in specific_suffixes)
        
        if not is_specific:
            # Residue COM (DMC, ECA, LiA, PF6, total)
            ax1.plot(df['z_center_angstrom'], df[col], label=label)
        else:
            # Specific atoms (ECA_O, DMC_O, LiA_Li, PF6_F) in ax2
            ax2.plot(df['z_center_angstrom'], df[col], label=label, linestyle='-')
            
    ax1.set_ylabel('Number Density (nm^-3)')
    ax1.set_title(f'Z Number Density Profile (Residue COM) - {out_dir.parent.name}')
    ax1.legend(loc='upper right', fontsize='small', ncol=2)
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('z (Angstrom)')
    ax2.set_ylabel('Number Density (nm^-3)')
    ax2.set_title('Z Number Density Profile (Specific Atoms: Carbonyl O, Li, F)')
    ax2.legend(loc='upper right', fontsize='small', ncol=2)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'z_density_profile.png')
    plt.close()
    print(f"Saved {out_dir / 'z_density_profile.png'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    out_dir = Path(args.output_dir).resolve()
    plot_charge_timeseries(out_dir)
    plot_z_density(out_dir)
