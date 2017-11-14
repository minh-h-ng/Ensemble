#!/bin/bash

DATAFILE="/home/minh/PycharmProjects/Ensemble/PythonESN/data_backup/edgar"

for HOURS_ELAPSED in {1..1}
do
    python -m scoop -n 2 ./evolve.py $DATAFILE
done