import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_sweep_comparison():
    here = Path('.').resolve()
    volts = ["3.0", "4.0", "5.0"]
    tags = ["V3p0", "V4p0", "V5p0"]
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    
    specific_atoms = ['LiA_Li', 'PF6_F', 'ECA_O_carbonyl', 'DMC_O_carbonyl']
    
    # Create subplots
    fig, axes = plt.subplots(len(specific_atoms), 1, figsize=(12, 16), sharex=True)
    
    for i, atom_name in enumerate(specific_atoms):
        ax = axes[i]
        col_name = f'number_density_{atom_name}_nm^-3'
        
        for tag, v_str, color in zip(tags, volts, colors):
            csv_path = here / f'sweep_{tag}/outputs/stage3_production/z_number_density_profile.csv'
            if not csv_path.exists():
                print(f"Skipping {csv_path} - not found.")
                continue
                
            df = pd.read_csv(csv_path)
            if col_name in df.columns:
                ax.plot(df['z_center_angstrom'], df[col_name], label=f'{v_str}V', color=color, alpha=0.8)
            else:
                print(f"Column {col_name} not found in {csv_path}")
                
        ax.set_ylabel('Number Density (nm^-3)')
        ax.set_title(f'Density Profile Comparison: {atom_name}')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
    axes[-1].set_xlabel('z (Angstrom)')
    plt.tight_layout()
    
    output_path = here / 'sweep_comparison_specific_atoms.png'
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")

if __name__ == "__main__":
    plot_sweep_comparison()
