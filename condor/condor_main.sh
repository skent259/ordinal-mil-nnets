#!/bin/bash

NAME=$1
EXP_FILE=$2
I=$3
OUTPUT_DIR="results/$NAME/"

ENVNAME=ordinal-mil-nnets
ENVDIR=$ENVNAME

# Set-up Python package environment
cp /staging/spkent/ordinal-mil-nnets/$ENVNAME.tar.gz ./
export PATH
mkdir $ENVDIR
tar -xzf $ENVNAME.tar.gz -C $ENVDIR
. $ENVDIR/bin/activate

# Prepare folder structure
tar -xzf to-transfer-$NAME.tar.gz
mkdir results
mkdir results/$NAME

# run script
# python3 fgnet-1.0.0.py --i $1 --output_dir $OUTPUT_DIR
python3 condor_main.py -e $EXP_FILE --i $I --output_dir $OUTPUT_DIR

# zip up output
tar -czf $NAME-1.0.0_$1.tar.gz results/
mv $NAME-1.0.0_$1.tar.gz /staging/spkent/ordinal-mil-nnets/

find $OUTPUT_DIR -name "*.hdf5" -exec rm {} \;
tar -czf $NAME-1.0.0_sm_$1.tar.gz results/ 

# clean up
rm $ENVNAME.tar.gz
rm to-transfer-fgnet.tar.gz