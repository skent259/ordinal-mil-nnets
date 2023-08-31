#!/bin/bash
# NOTE: do not run from condor/, run from ./

sim="bcnb-3.0.2"
cd condor/$sim 

# Record the total run time and any errors 
grep "Total Remote Usage" log/*.log > "log-time_$sim.txt"
grep "Memory (MB)" log/*.log > "log-memory_$sim.txt"

# Clean up the err, log, out files 
rm err/*.err
rm log/*.log
rm out/*.out

cd ../..
