#!/bin/bash
#SBATCH --export=ALL
#SBATCH --job-name=LTFSI
#SBATCH -N 1 -n 1 -t 800:00:00 -c 1
#SBATCH --gres=gpu:1 -p gpu_4090
#SBATCH -o out -e err
export OPENMM_CPU_THREADS=1
export OMP_NUM_THREADS=1

#python get_pdb_init.py
python make_init_pdb.py bulk.pdb > init_init.pdb
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
 
