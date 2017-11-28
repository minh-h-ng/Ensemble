#!/bin/bash

###
# $1 => name of dataset (e.g. edgar_10_12).
# $2 => results_suffix (e.g. 1).

# Usage examples:
# (venv)$ ./run_ga.sh edgar_10_12 1
# (venv)$ ./run_ga.sh edgar_10_12 2
# (venv)$ ./run_ga.sh edgar_10_12 3
###

########### Variables ###########
# Top most directory of project
top_dir=$(git rev-parse --show-toplevel)

# Data directory
data_dir=${top_dir}"/PythonESN/data_backup/"

# Data set
data_set=$1

# Both hours_start and hours_end are inclusive
# and signifies the number of hours elapsed.
# Each output line of forecast.py (except header) signifies
# 1-hr of elapsed time
hours_start=1
hours_end=2207

# Directory path (relative to top_dir) for storing GA results
results_dir='/genetic_algorithm/'$1'/'

# File name prefix for storing GA results
results_prefix='Ensemble_GA_'

# File name suffix for multiple runs
results_suffix='_'$2

# Processors
cpus=$(cat /proc/cpuinfo | grep 'processor' | wc -l)

#################################

# Make sure results_dir exists
dir_path=${top_dir}${results_dir}
mkdir -p ${dir_path}

# Output path
result_path=${dir_path}${results_prefix}${data_set}${results_suffix}

# Execute script
python3 -m scoop -n ${cpus} evolve.py ${data_dir}${data_set} ${hours_start} ${hours_end} ${result_path}
