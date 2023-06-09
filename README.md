# Inverse design of nanophotonic devices with structural integrity

This file contains instructions for reproducing the data and figures used in [Inverse design of nanophotonic devices with structural integrity](https://doi.org/10.1021/acsphotonics.0c00699 "DOI URL").

## Directory structure

```
.
├── README.md
├── environment.yml
├── data/
├── plots/
├── autograd_lib/
│   ├── __init__.py
│   └── filters.py
├── topopt/
│   ├── __init__.py
│   └── geom_util.py
├── combined_opt.py
├── problems.py
├── dask_runs.py
├── meep_eval.py
├── mpl_style.py
├── paper_plots.py
├── utilities.py
└── run_all.sh
```

## Requirements

This code was tested on Fedora 32 (kernel 5.7.10) on an Intel i7-8565U CPU as well as on a cluster running Debian 4.9 (kernel 4.9.0).
All packages were installed using Anaconda version 4.8.3 with Python 3.7.8.
Detailed package versions are included in the file `environment.yml`, which can be used to recreate the same Anaconda environment.

## Quick start

Environment creation as well as generation of all data and figures in the paper can be done in a single step by executing the bash script `./run_all.sh`.

## Installation

The following steps create a Python environment identical to the one used while developing this code.

First, install either Anaconda or Miniconda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html

Then, install all dependencies via `conda env create -f environment.yml`.
Afterwards, activate the environment using `conda activate hybrid_optimization` and install the FDFD solver via `pip install --no-deps fdfdpy`.

## Running

All data used in the paper can be generated by executing the file `dask_runs.py` which can be adapted to run on HPC clusters.
For doing so, please refer to the [Dask documentation](https://docs.dask.org/en/latest/setup/hpc.html).
Note that this performs 42 optimizations in total, which can take up to a day when run serially.
It is also possible to run single optimizations by directly executing `combined_opt.py`.

The mode converter example needs to be evaluated using [Meep](https://meep.readthedocs.io/en/latest/). To do this, run the file `meep_eval.py` _after_ `dask_runs.py`.

Finally, Figures 2 & 4 from the paper can be generated by running `paper_plots.py`.

## Citing

If you use this code or associated data for your research, please cite:

```
@article{augenstein2020inverse,
  title = {Inverse Design of Nanophotonic Devices with Structural Integrity},
  author = {Augenstein, Yannick and Rockstuhl, Carsten},
  year = 2020,
  journal = {ACS Photonics},
  volume = 7,
  number = 8,
  pages = {2190--2196},
  doi = {10.1021/acsphotonics.0c00699}
}
```
