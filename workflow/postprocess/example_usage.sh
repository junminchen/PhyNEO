#!/bin/bash

# Example usage script for MSD-based conductivity calculator
# This script demonstrates how to use calculate_msd_conductivity.py

echo "=================================================="
echo "Example Usage of MSD Conductivity Calculator"
echo "=================================================="
echo ""

# Example 1: Basic usage
echo "Example 1: Basic calculation"
echo "-----------------------------"
echo "python calculate_msd_conductivity.py \\"
echo "    --pdb nvt_init.pdb \\"
echo "    --traj nvt_ti.pos_0.1.xyz \\"
echo "    --cation \"name Li01\" \\"
echo "    --anion \"name P02\" \\"
echo "    --conc 1.0"
echo ""

# Example 2: With plotting
echo "Example 2: With MSD plots"
echo "-------------------------"
echo "python calculate_msd_conductivity.py \\"
echo "    --pdb nvt_init.pdb \\"
echo "    --traj nvt_ti.pos_0.1.xyz \\"
echo "    --cation \"name Li01\" \\"
echo "    --anion \"name P02\" \\"
echo "    --conc 1.2 \\"
echo "    --temp 298 \\"
echo "    --plot"
echo ""

# Example 3: Advanced options
echo "Example 3: Advanced options (custom fit range, high temperature)"
echo "----------------------------------------------------------------"
echo "python calculate_msd_conductivity.py \\"
echo "    --pdb init.pdb \\"
echo "    --traj trajectory.xyz \\"
echo "    --cation \"resname LiA\" \\"
echo "    --anion \"resname FSI\" \\"
echo "    --conc 1.5 \\"
echo "    --temp 338 \\"
echo "    --dt 0.1 \\"
echo "    --start-frame 1000 \\"
echo "    --fit-start 0.3 \\"
echo "    --fit-end 0.7 \\"
echo "    --plot \\"
echo "    --output-prefix my_system_"
echo ""

# Example 4: Different ion types
echo "Example 4: Li-PF6 electrolyte"
echo "-----------------------------"
echo "python calculate_msd_conductivity.py \\"
echo "    --pdb system.pdb \\"
echo "    --traj simulation.xyz \\"
echo "    --cation \"resname LiA and name Li\" \\"
echo "    --anion \"resname PF6\" \\"
echo "    --conc 1.0 \\"
echo "    --plot"
echo ""

echo "=================================================="
echo "For more help, run:"
echo "python calculate_msd_conductivity.py --help"
echo "=================================================="
