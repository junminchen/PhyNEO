#!/bin/bash
export OMP_NUM_THREADS=1
i-pi input.xml >& logfile &
#i-pi test_npt.restart >& logfile &
wait
