#!/bin/bash

conda env create -f environment.yml
eval "$(conda shell.bash hook)"
conda activate hybrid_optimization
pip install --no-deps fdfdpy

python dask_runs.py
python meep_eval.py
python paper_plots.py
