#!/usr/bin/env python3
# vim: set fileencoding=utf-8

import os
import h5py
import numpy as np
from numbers import Number
from combined_opt import combined_opt, default_args
from problems import setup_problem_lens, setup_problem_mode_conv

from dask.distributed import Client, as_completed, Lock

client = Client(processes=False)


def write(fname, group, data):
    with h5py.File(fname, "a") as f:
        grp = f.create_group(group)
        for key, val in data.items():
            if isinstance(val, (str, dict)):
                grp.create_dataset(key, data=str(val))
            elif isinstance(val, Number):
                grp.create_dataset(key, data=val)
            elif isinstance(val, (Number, np.ndarray)):
                grp.create_dataset(key, data=val, compression="gzip")


def run(out_file):
    jobs = []
    for compliance_factor in np.linspace(0, 1, 21):
        args = default_args()
        args["max_eval"] = 1500
        args["method"] = "bfgs"
        args["run_name"] = "lens_resonance_penalty_multistart_newloads"
        args["sigma"] = 1.0
        args["force_symmetry_axis"] = 0
        args["binarization_factor"] = 3
        args["n2"] = 1.5
        args["compliance_factor"] = compliance_factor
        args["use_resonance_penalty"] = True
        jobs.append(args)
    futures = client.map(lambda x: combined_opt(x, setup_problem_lens(args)), jobs)

    for future, result in as_completed(futures, with_results=True):
        with Lock():
            write(out_file, future.key, result)

    jobs = []
    for compliance_factor in np.linspace(0, 1, 21):
        args = default_args()
        args["max_eval"] = 1500
        args["method"] = "bfgs"
        args["run_name"] = "mode_converter_multistart_abs2_newgeom_lowbin"
        args["sigma"] = 1.0
        args["force_symmetry_axis"] = -1
        args["binarization_factor"] = 1
        args["n2"] = 1.5
        args["compliance_factor"] = compliance_factor
        args["use_resonance_penalty"] = False
        jobs.append(args)
    futures = client.map(lambda x: combined_opt(x, setup_problem_mode_conv(args)), jobs)

    for future, result in as_completed(futures, with_results=True):
        with Lock():
            write(out_file, future.key, result)


if __name__ == "__main__":
    datadir = "./data/"
    filename = "20200130_193502_hybrid_optimization.h5"
    path = datadir + filename
    run(path)
