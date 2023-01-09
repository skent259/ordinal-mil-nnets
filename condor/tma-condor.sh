#!/bin/bash

NAME=$1
VERSION=$2
I=$3
OUTPUT_DIR=$4
EXP_FILE="example-${NAME}-${VERSION}.csv"

ENVNAME=ordinal-mil-nnets
ENVDIR=$ENVNAME

# Set-up Python package environment
cp /staging/spkent/ordinal-mil-nnets/$ENVNAME.tar.gz ./
export PATH
mkdir $ENVDIR
tar -xzf $ENVNAME.tar.gz -C $ENVDIR
. $ENVDIR/bin/activate

# Prepare folder structure
cp /staging/spkent/ordinal-mil-nnets/to-transfer-${NAME}.tar.gz ./
tar -xzf to-transfer-${NAME}.tar.gz
mkdir results
mkdir results/${NAME}

export NLTK_Data="./nltk_data"

# if [ ${NAME} == "bcnb" ]; then
#     echo ~/
#     mkdir ~/.keras/
#     mkdir ~/.keras/models/
#     cp /staging/spkent/ordinal-mil-nnets/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5 ~/.keras/models/
# fi

# run script
python3 02-train-tma-7.0.1.py -e $EXP_FILE --i $I --output_dir $OUTPUT_DIR

# zip up output
tar -czf ${NAME}-${VERSION}_${I}.tar.gz results/
mv ${NAME}-${VERSION}_${I}.tar.gz /staging/spkent/ordinal-mil-nnets/

find $OUTPUT_DIR -name "*.hdf5" -exec rm {} \;
tar -czf ${NAME}-${VERSION}_sm_${I}.tar.gz results/ 

# clean up
rm $ENVNAME.tar.gz
rm to-transfer-${NAME}.tar.gz
