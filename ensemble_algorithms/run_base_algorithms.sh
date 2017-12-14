#!/bin/bash

###
# $1 => name of dataset (e.g. edgar_10_12).

# Usage examples:
# (venv)$ ./run_base_algorithms.sh edgar_10_12
# (venv)$ ./run_base_algorithms.sh cran_10_12
# (venv)$ ./run_base_algorithms.sh kyoto_10_12
###

########### Variables ###########
# Top most directory of project
top_dir=$(git rev-parse --show-toplevel)

# Data directory
data_dir=${top_dir}"/processed/"

# Data set
data_set=$1

# Directory path (relative to top_dir) for storing results
results_dir='/PythonESN/data_backup/'
#################################

# Make sure results_dir exists
dir_path=${top_dir}${results_dir}
mkdir -p ${dir_path}

# Output path
result_path=${dir_path}${data_set}

# Execute script
python3 base.py ${data_dir}${data_set} ${result_path}