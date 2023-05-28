#!/usr/bin/env python3
# vim: set fileencoding=utf-8

import os
import json
import h5py
import pickle
import pandas as pd
import numpy as np
import meep as mp

from topopt import geom_util


class HDFBrowser:
    def __init__(self, fname, mode="r"):
        self.fname = fname
        self.mode = mode

    def to_dict(self):
        f = h5py.File(self.fname, "r")
        odict = {}
        for key in f.keys():
            grp = f[key]
            grp_dict = {}
            for gkey, gval in grp.items():
                if gkey == "options":
                    grp_dict[gkey] = eval(str(gval[()]))
                elif gkey == "timestamp":
                    ts = str(gval[()]).replace("_", " ")
                    grp_dict[gkey] = pd.Timestamp(ts)
                elif gval.dtype == object:
                    grp_dict[gkey] = str(gval[()])
                else:
                    v = np.array(gval)
                    if v.size == 1:
                        grp_dict[gkey] = v.item()
                    else:
                        grp_dict[gkey] = np.array(gval)
            odict[key] = grp_dict
        f.close()
        return odict

    def to_df(self, merge=True):
        runs = []
        for run in self.to_dict().values():
            if merge:
                run.update(run["options"])
                run.pop("options", None)
            runs.append(run)
        return pd.DataFrame(runs).set_index("timestamp").sort_index()

    def __str__(self):
        f = h5py.File(self.fname, "r")
        summary = f"HDF5 file {self.fname} with {len(f.keys())} groups:\n"
        groups = "\n".join([f"  {key}" for key in f])
        f.close()
        return summary + groups


def make_rho_transforms(args, mats):
    # define rho transformations for photonic optimization
    rho2filt = geom_util.make_rho2filt_sp(args["sigma"], mats["design_roi"])
    filt2proj = geom_util.make_filt2proj(args["proj_k"], 0.5, mats["design_roi"])
    proj2eps = geom_util.make_proj2eps(args["n1"] ** 2, args["n2"] ** 2)
    rho2eps = geom_util.compose(proj2eps, filt2proj, rho2filt)

    # define rho transformations for structural optimization
    extract_submat = geom_util.make_submat(mats["design_roi"])
    rho2filt_yng = geom_util.make_rho2filt_sp(
        args["sigma"], np.ones((mats["nelx"], mats["nely"]))
    )
    if args.get("use_same_transforms", False):
        filt_yng2proj = geom_util.make_filt2proj(
            args["proj_k"], 0.5, np.ones((mats["nelx"], mats["nely"]))
        )
        proj2young = geom_util.make_proj2eps(args["young_min"], args["young_max"])
        rho2young = geom_util.compose(
            proj2young, filt_yng2proj, rho2filt_yng, extract_submat
        )
    else:
        proj2young = geom_util.make_proj2eps(
            args["young_min"], args["young_max"], args["penalty"]
        )
        rho2young = geom_util.compose(proj2young, rho2filt_yng, extract_submat)

    return rho2eps, rho2young


def get_epsilon(df, symmetry_cond):
    rho2eps, _ = make_rho_transforms(df, df)
    eps = rho2eps(np.reshape(df["rho_final"], df["design_roi"].shape))
    if symmetry_cond == "force_symmetry_axis":
        if df["force_symmetry_axis"] >= 0:
            eps = geom_util.make_symmetric(eps, axis=df["force_symmetry_axis"])
    else:
        if df["force_symmetry"]:
            eps = np.vstack(
                [eps[: eps.shape[0] // 2], np.flipud(eps[: eps.shape[0] // 2])]
            )
    return eps


def eps_from_pickle(run_name):
    with open(f"./data/{run_name}_epsilon.pkl", "rb") as f:
        eps = pickle.load(f)
    return eps


def eps_to_pickle(infile, run_name):
    data = HDFBrowser(infile)
    df = data.to_df()
    df = df[df["run_name"] == run_name]
    epsilon = {}
    for idx, run in df.iterrows():
        eps = get_epsilon(run, "force_symmetry_axis")
        epsilon.update({np.around(run["compliance_factor"], 2): eps})
    with open(f"./data/{run_name}_epsilon.pkl", "wb") as f:
        pickle.dump(epsilon, f, protocol=pickle.HIGHEST_PROTOCOL)


def get_mode_coeffs(eps, sim_res):
    resolution = sim_res
    dpml = 0.5
    lcen = 1
    dx, dy = 12, 8
    modes = np.arange(1, 6)

    fcen = 1 / lcen
    df = 0.2 * fcen
    cell = mp.Vector3(dx, dy)
    pml = [mp.PML(dpml)]
    mode_in_vol = mp.Volume(
        center=mp.Vector3(dpml + 0.5, dy / 2), size=mp.Vector3(0, dy - 2 * dpml)
    )
    mode_out_vol = mp.Volume(
        center=mp.Vector3(dx - dpml - 0.5, dy / 2), size=mp.Vector3(0, dy - 2 * dpml)
    )
    mode_in_reg = mp.FluxRegion(volume=mode_in_vol, direction=mp.X)
    mode_out_reg = mp.FluxRegion(volume=mode_out_vol, direction=mp.X)

    sources = [
        mp.EigenModeSource(
            src=mp.GaussianSource(frequency=fcen, fwidth=df),
            center=mp.Vector3(dpml, dy / 2),
            size=mp.Vector3(0, dy - 2 * dpml),
            direction=mp.X,
            eig_parity=mp.ODD_Z,
            eig_match_freq=True,
            eig_band=1,
        )
    ]

    # run one simulation with a straight waveguide to compute the power of the source
    straight_waveguide = np.tile(eps[0, :].reshape(1, -1).T, eps.shape[1]).T
    sim = mp.Simulation(
        cell_size=cell,
        default_material=np.ascontiguousarray(straight_waveguide),
        boundary_layers=pml,
        resolution=resolution,
        sources=sources,
        geometry_center=cell / 2,
    )

    refl_flux = sim.add_flux(fcen, 0, 1, mode_in_reg)

    sim.run(
        until_after_sources=mp.stop_when_fields_decayed(
            10, mp.Ez, mp.Vector3(dx - dpml, dy / 2), decay_by=1e-9
        )
    )

    wvg_in_flux = mp.get_fluxes(refl_flux)
    wvg_in_flux_data = sim.get_flux_data(refl_flux)

    sim.reset_meep()

    # now do the actual simulation
    sim = mp.Simulation(
        cell_size=cell,
        default_material=eps,
        boundary_layers=pml,
        resolution=resolution,
        sources=sources,
        geometry_center=cell / 2,
    )

    refl_flux = sim.add_flux(fcen, 0, 1, mode_in_reg)
    sim.load_minus_flux_data(refl_flux, wvg_in_flux_data)
    tran_flux = sim.add_flux(fcen, 0, 1, mode_out_reg)

    sim.run(
        until_after_sources=mp.stop_when_fields_decayed(
            10, mp.Ez, mp.Vector3(dx - dpml, dy / 2), decay_by=1e-9
        )
    )

    res = sim.get_eigenmode_coefficients(tran_flux, modes, eig_parity=mp.ODD_Z)
    t_coeffs = res.alpha

    results = {}
    for mode_idx in range(len(modes)):
        t_mode = np.abs(t_coeffs[mode_idx, 0, 0]) ** 2 / wvg_in_flux[0]
        results.update({f"mode_{mode_idx + 1}": t_mode})

    r_flux = mp.get_fluxes(refl_flux)
    t_flux = mp.get_fluxes(tran_flux)
    refl = -r_flux[0] / wvg_in_flux[0]
    tran = t_flux[0] / wvg_in_flux[0]
    results.update(
        {
            "poynting_trans": tran,
            "poynting_refl": refl,
            "poynting_loss": 1 - refl - tran,
        }
    )

    return results


def write_mode_coeffs(infile, run_name):
    if not os.path.isfile(f"./data/{run_name}_epsilon.pkl"):
        eps_to_pickle(infile, run_name)
    eps = eps_from_pickle(run_name)

    res_table = {}
    for key, val in sorted(eps.items()):
        res_table.update({key: get_mode_coeffs(eps[key], sim_res=30)})

    with open(f"./data/{run_name}_mode_coeffs.json", "w") as f:
        json.dump(res_table, f, indent=4)


def load_mode_coeffs(run_name):
    with open(f"./data/{run_name}_mode_coeffs.json", "r") as f:
        data = json.load(f)
    return data


def main(infile, run_name):
    if not os.path.isfile(f"./data/{run_name}_mode_coeffs.json"):
        write_mode_coeffs(infile, run_name)
    data = load_mode_coeffs(run_name)

    x, y1, y2, y3, y4 = [], [], [], [], []
    for cfac, mode_dict in data.items():
        x.append(float(cfac))
        y1.append(mode_dict["mode_1"])
        y2.append(mode_dict["mode_2"])
        y3.append(mode_dict["mode_3"])
        y4.append(mode_dict["mode_4"])


if __name__ == "__main__":
    main(
        "./data/20200130_193502_hybrid_optimization.h5",
        "mode_converter_multistart_abs2_newgeom_lowbin",
    )
