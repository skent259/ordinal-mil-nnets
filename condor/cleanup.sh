#!/bin/bash

sim="fgnet-1.0.1"
cd $sim 

# Record the total run time and any errors 
grep "Total Remote Usage" log/*.log > "log-time_$sim.txt"


# Clean up the err, log, out files 
rm err/*.err
rm log/*.log
rm out/*.out

cd ..   