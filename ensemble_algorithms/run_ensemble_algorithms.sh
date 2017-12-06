#!/bin/bash

###
# $1 => name of dataset (e.g. edgar_10_12).
# $2 => results_suffix (e.g. 1).

# Usage examples:
# (venv)$ ./run_ensemble_algorithms.sh edgar_10_12 1
# (venv)$ ./run_ensemble_algorithms.sh edgar_10_12 2
# (venv)$ ./run_ensemble_algorithms.sh edgar_10_12 3
###

# Top most directory of project
top_dir=$(git rev-parse --show-toplevel)

# Data directory
data_dir=${top_dir}"/processed/"

# Data set
data_set=$1

# Directory path (relative to top_dir) for storing results
results_dir='/ensemble_algorithms/results/'$1'/'

# File name suffix for multiple runs
results_suffix='_'$2

# Make sure results_dir exists
dir_path=${top_dir}${results_dir}
mkdir -p ${dir_path}

data_file=${data_dir}${data_set}

run_ensemble_average() {
    # File name prefix for results
    results_prefix='Ensemble_Average_'

    # Output path
    result_path=${dir_path}${results_prefix}${data_set}${results_suffix}

    # Execute script
    python3 ensemble_average.py ${data_dir}${data_set} ${result_path}
}

run_ensemble_cross_validation() {
    # File name prefix for results
    results_prefix='Ensemble_CrossValidation_'

    # Output path
    result_path=${dir_path}${results_prefix}${data_set}${results_suffix}

    # Execute script
    python3 ensemble_cross_validation.py ${data_dir}${data_set} ${result_path}
}

run_ensemble_logistic_regression() {
    # File name prefix for results
    results_prefix='Ensemble_LogisticRegression_'

    # Output path
    result_path=${dir_path}${results_prefix}${data_set}${results_suffix}

    # Execute script
    python3 ensemble_logistic_regression.py ${data_dir}${data_set} ${result_path}
}

run_ensemble_bagging_regression() {
    # File name prefix for results
    results_prefix='Ensemble_BaggingRegression_'

    # Output path
    result_path=${dir_path}${results_prefix}${data_set}${results_suffix}

    # Execute script
    python3 ensemble_bagging_regression.py ${data_dir}${data_set} ${result_path}
}

#run_ensemble_average
run_ensemble_cross_validation
#run_ensemble_logistic_regression
#run_ensemble_bagging_regression