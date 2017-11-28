#!/bin/bash

dataFile="/home/minh/PycharmProjects/Ensemble/PythonESN/data_backup/edgar_10_12"
baggingFile="/home/minh/PycharmProjects/Ensemble/ensemble_algorithms/bagging_results"
crossFile="/home/minh/PycharmProjects/Ensemble/ensemble_algorithms/cross_validation_results"

#python ./ensemble_bagging.py $dataFile $baggingFile
python ./ensemble_cross_validation.py $dataFile $crossFile