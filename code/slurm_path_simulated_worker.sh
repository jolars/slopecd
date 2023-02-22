#!/bin/bash

cat $0

# receive my worker number
export WRK_NB=$1
export scenario=$2
export solver=$3

# create worker-private subdirectory in $SNIC_TMP
export WRK_DIR=$SNIC_TMP/WRK_${WRK_NB}
mkdir $WRK_DIR

# create a variable to address the "job directory"
export JOB_DIR=$SLURM_SUBMIT_DIR/job_${WRK_NB}

# now copy the input data and program from there
cp -r ${SLURM_SUBMIT_DIR}/pyproject.toml \
    ${SLURM_SUBMIT_DIR}/setup.cfg \
    ${SLURM_SUBMIT_DIR}/slope \
    ${SLURM_SUBMIT_DIR}/expes \
    $WRK_DIR/

# change to the execution directory
cd $WRK_DIR

mkdir -p results/path_simulated

# load modules
module load foss/2022a
module load Python/3.10.4
module load SciPy-bundle/2022.05
module load R/4.2.1

# install modules
# pip install . --prefix=$HOME/local package_name

pip install -U .

# run the program
python -u expes/path_simulated.py $scenario $solver 

# rescue the results back to job directory
cp -r results/path_simulated/* ${SLURM_SUBMIT_DIR}/results/path_simulated/

# clean up the local disk and remove the worker-private directory
cd $SNIC_TMP

rm -rf WRK_${WRK_NB}
