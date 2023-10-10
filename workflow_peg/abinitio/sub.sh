#!/bin/bash
#SBATCH --export=ALL
#SBATCH -o out -e err
#SBATCH -N 1 -n 1 -c 1 -t 24:00:00 --mem=10000mb --gres=gpu:1
#SBATCH --job-name=openmm_sample
#SBATCH -p gpu

python run_md.py > md.log
