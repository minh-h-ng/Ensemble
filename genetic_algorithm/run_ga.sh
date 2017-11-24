#!/bin/bash

times=1

########### Variables ###########
# Top most directory of project
top_dir=$(git rev-parse --show-toplevel)

# Data
DATAFILE=${top_dir}"/PythonESN/data_backup/edgar_10_12"

# Both hours_start and hours_end are inclusive
# and signifies the number of hours elapsed.
# Each output line of forecast.py (except header) signifies
# 1-hr of elapsed time
HOURS_START=1
HOURS_END=2207

# Directory path (relative to top_dir) for storing GA results
results_dir='/genetic_algorithm/'

# File name for storing GA results
results_file='GA_results'

# Processors
cpus=$(cat /proc/cpuinfo | grep 'processor' | wc -l)

#################################

# Make sure results_dir exists
dir_path=${top_dir}${results_dir}
mkdir -p ${dir_path}

# Output path
result_path=${dir_path}${results_file}

# Execute script
python3 -m scoop -n ${cpus} evolve.py ${times} ${DATAFILE} ${HOURS_START} ${HOURS_END} ${result_path}
