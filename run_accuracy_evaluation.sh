#!/bin/bash

# Top most directory of project
top_dir=$(git rev-parse --show-toplevel)

python ./Final_Results.py ${top_dir}
python ./Percent_Results.py ${top_dir}
python ./Accuracy_Results.py ${top_dir}
python ./Cost_Projection.py ${top_dir}