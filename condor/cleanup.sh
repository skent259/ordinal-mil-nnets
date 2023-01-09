#!/bin/bash
# NOTE: do not run from condor/, run from ./

sim="afad-6.0.1"
cd condor/$sim 

# Record the total run time and any errors 
grep "Total Remote Usage" log/*.log > "log-time_$sim.txt"
grep "Memory (MB)" log/*.log > "log-memory_$sim.txt"

# Clean up the err, log, out files 
rm err/*.err
rm log/*.log
rm out/*.out

cd ../..