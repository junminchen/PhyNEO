#!/bin/bash
#SBATCH --export=ALL
#SBATCH --job-name=peo3oml
#SBATCH -N 1 -n 1 -t 800:00:00 -c 1
#SBATCH --gres=gpu:1 -p rtx3090
#SBATCH --mem=20000mb -o o -e e

export OPENMM_CPU_THREADS=1
export OMP_NUM_THREADS=1

scrdir=/tmp

# clean folder
rm $scrdir/ipi_unix_dmff_*
echo "***** start time *****"
date

cd  $SLURM_SUBMIT_DIR
# run server
#i-pi input.xml >& logfile & sleep 8
bash run_server.sh &
sleep 8

# check socket
ls -l $scrdir

# run client
bash run_client_dmff.sh & 
wait

echo "***** finish time *****"
date

sleep 1
 
