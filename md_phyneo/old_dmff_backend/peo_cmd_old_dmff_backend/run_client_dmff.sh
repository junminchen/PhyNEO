#!/bin/bash
# create the right environment to run client: note client runs in python3
# while i-pi server runs in python2

# source ~/.bashrc
# conda activate dmff_dev

# module load gcc/8.3.0
# module load fftw/3.3.8/single-threads
# module load compiler/intel/ips2018/u1
# module load mkl/intel/ips2018/u1
# module load cuda/11.4

export OPENMM_CPU_THREADS=1
export OMP_NUM_THREADS=1

addr=unix_dmff
port=1234
socktype=unix

ffdir=ff_files_peo
python ./client_dmff.py init.pdb $ffdir/forcefield.xml $ffdir/params.pickle $ffdir/params_sgnn.pickle $ffdir/params_eann.pickle $addr $port $socktype > log_dmff

