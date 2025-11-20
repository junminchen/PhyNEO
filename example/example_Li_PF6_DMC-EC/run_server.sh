#!/bin/bash
export OMP_NUM_THREADS=1
#cat input.xml | sed -e "s/<address> \([a-zA-Z_]\+\)/<address> \1_${SLURM_JOB_ID}/" > .tmp.xml
# i-pi simulation_lj.restart >& logfile &
#i-pi .tmp.xml >& logfile &
i-pi input.xml >& logfile &
#i-pi simulation.restart >& logfile &

# i-pi input_nvt.xml >& logfile &

wait
