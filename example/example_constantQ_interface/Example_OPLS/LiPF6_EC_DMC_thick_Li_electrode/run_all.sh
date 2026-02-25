#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

bash run_packmol.sh
python assemble_thick_li_electrode_system.py
python mc_gap_equilibrate.py
python run_openmm84_thick_li_electrode.py
python analyze_interface_distribution.py --between-electrodes-only --electrode_margin 1.0 --interface_width 5.0 --nblocks 5
python analyze_interfacial_capacitance.py --skip 2 --nblocks 5
python visualize_last_frame_electrode_charge.py --xy-bins 40

echo "Done. See results/ and npt_thick_li.log"
