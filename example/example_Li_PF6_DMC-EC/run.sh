#!/bin/bash

#python gen_md_pdb.py
scrdir=/tmp

# clean folder
rm $scrdir/ipi_unix_dmff_*
echo "***** start time *****"
date

# run server
i-pi input.xml >& logfile & sleep 8
# bash run_server.sh &
# sleep 8

# check socket
ls -l $scrdir

# run client
addr=unix_dmff
port=1234
socktype=unix

# python /home/am3-peichenzhong-group/Documents/project/PhyNEO/example/example_Li_PF6_DMC-EC/client_dmff.py $addr $port $socktype > log_dmff
python client_dmff.py $addr $port $socktype > log_dmff


# bash run_client_dmff.sh & 
wait

echo "***** finish time *****"
date

sleep 1
 
