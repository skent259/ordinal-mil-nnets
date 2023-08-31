#!/bin/bash

# Prepare folder structure -----------------------------------------#
mkdir results
mkdir results/afad
mkdir results/amrev
mkdir results/bcnb
mkdir results/fgnet
mkdir results/imdb

# Run scripts for each experiment ----------------------------------#
# Suggest running in a high-throughput manner due to runtime, below 
# code provides an example only

# fgnet (1.0.X)
for ((i=0; i<600; i++)) 
do 
    python3 run.py -e experiment/params/experiment-fgnet-1.0.1.csv --i $i --output_dir results/fgnet/
done 

for ((i=0; i<300; i++)) 
do 
    python3 run.py -e experiment/params/experiment-fgnet-1.0.2.csv --i $i --output_dir results/fgnet/
done 

for ((i=0; i<300; i++)) 
do 
    python3 run.py -e experiment/params/experiment-fgnet-1.0.3.csv --i $i --output_dir results/fgnet/
done 

for ((i=0; i<1800; i++)) 
do 
    python3 run.py -e experiment/params/experiment-fgnet-1.0.4.csv --i $i --output_dir results/fgnet/
done 

# bcnb (3.0.X)
for ((i=0; i<240; i++)) 
do 
    python3 run.py -e experiment/params/experiment-bcnb-3.0.1.csv --i $i --output_dir results/bcnb/
done 

for ((i=0; i<360; i++)) 
do 
    python3 run.py -e experiment/params/experiment-bcnb-3.0.2.csv --i $i --output_dir results/bcnb/
done 

# amrev (4.0.X)
for ((i=0; i<1200; i++)) 
do 
    python3 run.py -e experiment/params/experiment-amrev-4.0.1.csv --i $i --output_dir results/amrev/
done 

# imdb (5.0.X)
for ((i=0; i<1200; i++)) 
do 
    python3 run.py -e experiment/params/experiment-imdb-5.0.1.csv --i $i --output_dir results/imdb/
done 

# afad (6.0.X)
for ((i=0; i<1200; i++)) 
do 
    python3 run.py -e experiment/params/experiment-afad-6.0.1.csv --i $i --output_dir results/afad/
done 

# After results are available, run the following:
# * analysis/01-primary-analysis.qmd
# * analysis/02-create-figures.R
# * analysis/03-create-tables.qmd