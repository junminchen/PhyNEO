#!/bin/bash
#SBATCH --export=ALL
#SBATCH --job-name=localize
#SBATCH -N 1 -n 1 -t 24:00:00 -c 2
#SBATCH --tasks-per-node=1
#SBATCH --mem=30000mb

prefix=occocco
CCPROOT=/share/home/kuangy/compile/camcasp6.1
source $CCPROOT/env.sh
export CORES=$NCPUS
localize.py $prefix --isotropic 
#localize.py --help

