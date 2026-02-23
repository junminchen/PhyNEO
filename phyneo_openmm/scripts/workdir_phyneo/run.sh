#!/usr/bin/env bash

# Check if mpid environment is active
if [[ "$CONDA_DEFAULT_ENV" != "mpid" ]]; then
    echo "Reminder: Please activate your mpid environment before running this script."
    echo "Example: conda activate mpid"
    # exit 1 # Uncomment to enforce activation
fi

# Set plugin directory based on active conda environment
if [[ -n "$CONDA_PREFIX" ]]; then
    export OPENMM_PLUGIN_DIR="${CONDA_PREFIX}/lib/plugins"
fi

# Run simulation
python run_sim.py
