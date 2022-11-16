#!/bin/bash

sim="fgnet-1.0.0"
cd $sim 

# Record the total run time and any errors 
grep "Total Remote Usage" log/*.log > "log-time_$sim.txt"
grep "ERROR" out/*.out > "error-$sim.txt"
# grep -rnw 'out/' -e 'ERROR' > re-run-$sim-$step.txt # for re-running

# Clean up the err, log, out files 
rm err/*.err
rm log/*.log
rm out/*.out

cd ..   