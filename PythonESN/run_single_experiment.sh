#!/bin/bash

# NOTE: This file needs to be executable (i.e. chmod +x run_experiment.sh before attempting to run)

DATAFILE=$1
OPTCONFIG=$2
ESNCONFIG=$3
RUNS=$4

#DATAFILE=./data/NARMA
#OPTCONFIG=ridge_identity
#ESNCONFIG=esnconfig
#RUNS=30
for times in {6..10}
do
    for count in {2207..2207}
    do
        # Tune parameters. Note: the config file for the best parameters are saved at the location in $ESNCONFIG
        python -m scoop -n 4 ./genoptesn.py $count $DATAFILE $OPTCONFIG $ESNCONFIG --percent_dim
        #python ./genoptesn.py $count $DATAFILE $OPTCONFIG $ESNCONFIG --percent_dim

        # Run experiments with these parameters
        #python -m scoop -n 2 ./esn_experiment.py $DATAFILE $ESNCONFIG $RUNS
        python ./esn_run.py $times $count $DATAFILE $ESNCONFIG $RUNS
    done
done
