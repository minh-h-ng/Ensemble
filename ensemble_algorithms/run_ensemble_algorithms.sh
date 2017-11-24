#!/bin/bash

dataFile="/home/minh/PycharmProjects/Ensemble/PythonESN/data_backup/edgar_10_12"
baggingFile="/home/minh/PycharmProjects/Ensemble/ensemble_algorithms/bagging_results"

python ./ensemble_algorithms.py $dataFile $baggingFile