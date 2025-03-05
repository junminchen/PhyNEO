#!/bin/bash
#SBATCH --export=ALL
#SBATCH --job-name=mdtest
#SBATCH -N 1 -n 1 -t 800:00:00 -c 1
#SBATCH --gres=gpu:1 -p rtx3090
#SBATCH -o out -e err

module load cuda/11.4
#export OPENMM_CPU_THREADS=1
#export OMP_NUM_THREADS=1

echo "***** start time *****"
date

python run_dmff_ase.py > log

echo "***** finish time *****"
date

sleep 1
 
