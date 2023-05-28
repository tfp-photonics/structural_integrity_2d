#!/usr/bin/env python3
# vim: set fileencoding=utf-8

import os
import json
from string import ascii_lowercase
import numpy as np
import pandas as pd
from scipy.constants import c

import mpl_style
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes

from utilities import HDFBrowser
from topopt import geom_util
from combined_opt import (
    make_rho_transforms,
    make_maxwell_solver,
    make_structural_solver,
    compliance,
)


def make_paper_data(args):
    data = HDFBrowser(args["data"])
    df = data.to_df()
    run = df.loc[df["run_name"] == args["run"]]
    run = run.set_index("compliance_factor").sort_index()
    run.to_pickle(args["datafile"])


def get_field(df, symmetry_cond, eps=None, cut_pml=True):
    rho2eps, rho2young = make_rho_transforms(df, df)
    if eps is None:
        eps = rho2eps(np.reshape(df["rho_final"], df["design_roi"].shape))
    src = df["source"]
    pol = df["pol"]
    L0 = df["l0"]
    dpml = df["dpml"]
    res = df["res"]
    lcen = df["lcen"]
    n2 = df["n2"]
    sim_res = np.ceil(res / (lcen / n2))
    dl = 1 / sim_res
    omega = 2 * np.pi * c / (lcen * L0)
    npml = [int(np.ceil(dpml * sim_res)), int(np.ceil(dpml * sim_res))]
    if symmetry_cond == "force_symmetry_axis":
        if df["force_symmetry_axis"] >= 0:
            eps = geom_util.make_symmetric(eps, axis=df["force_symmetry_axis"])
    else:
        if df["force_symmetry"]:
            eps = np.vstack(
                [eps[: eps.shape[0] // 2], np.flipud(eps[: eps.shape[0] // 2])]
            )
    mode = {"pol": pol, "src": src}
    fdfd_solver = make_maxwell_solver(omega, mode, dl, npml, L0)
    ez = fdfd_solver(eps)
    if cut_pml:
        ez = ez[npml[0] // 2 : -npml[0] // 2, npml[1] // 2 : -npml[1] // 2]
    return ez


def get_epsilon(df, symmetry_cond, cut_pml=True):
    rho2eps, _ = make_rho_transforms(df, df)
    dpml = df["dpml"]
    res = df["res"]
    lcen = df["lcen"]
    n2 = df["n2"]
    sim_res = np.ceil(res / (lcen / n2))
    npml = [int(np.ceil(dpml * sim_res)), int(np.ceil(dpml * sim_res))]
    eps = rho2eps(np.reshape(df["rho_final"], df["design_roi"].shape))
    if symmetry_cond == "force_symmetry_axis":
        if df["force_symmetry_axis"] >= 0:
            eps = geom_util.make_symmetric(eps, axis=df["force_symmetry_axis"])
    else:
        if df["force_symmetry"]:
            eps = np.vstack(
                [eps[: eps.shape[0] // 2], np.flipud(eps[: eps.shape[0] // 2])]
            )
    if cut_pml:
        eps = eps[npml[0] // 2 : -npml[0] // 2, npml[1] // 2 : -npml[1] // 2]
    return eps


def get_young(df, symmetry_cond):
    _, rho2young = make_rho_transforms(df, df)
    young = rho2young(np.reshape(df["rho_final"], df["design_roi"].shape))
    if symmetry_cond == "force_symmetry_axis":
        if df["force_symmetry_axis"] >= 0:
            young = geom_util.make_symmetric(young, axis=df["force_symmetry_axis"])
    else:
        if df["force_symmetry"]:
            young = np.vstack(
                [young[: young.shape[0] // 2], np.flipud(young[: young.shape[0] // 2])]
            )
    return young


def get_compliance(df):
    ke = df["stiffness_matrix"]
    forces = df["forces"]
    freedofs = df["freedofs"]
    fixdofs = df["fixdofs"]
    fem_solver = make_structural_solver(ke, forces, freedofs, fixdofs)
    young = get_young(df, "force_symmetry")
    u = fem_solver(young)
    return compliance(young, u, ke)


def fom_and_results(args, cmap="inferno"):
    df = pd.read_pickle(args["datafile"])

    compliance_idx = [
        df.index.get_loc(cpl, method="nearest", tolerance=1e-3)
        for cpl in args["compliance_samples"]
    ]

    pts_em = []
    if "mode_data" in args:
        with open(args["mode_data"], "r") as f:
            data = json.load(f)
        for cfac, mode_dict in data.items():
            pts_em.append([float(cfac), mode_dict["mode_2"]])
    else:
        field_objs = []
        for cfac, run in df.iterrows():
            field_objs.append(
                np.sum(
                    np.abs(
                        get_field(run, args["symmetry_cond"], cut_pml=False)
                        * run["merit_roi"]
                    )
                )
            )
            pts_em.append([cfac, field_objs[-1] / field_objs[0]])
    pts_em = np.array(pts_em)
    if "mode_data" not in args:
        pts_em.T[1] /= np.linalg.norm(pts_em.T[1])

    pts_cpl = []
    for cfac, run in df.iterrows():
        pts_cpl.append([cfac, run["compliance"][-1]])
    pts_cpl = np.array(pts_cpl)
    pts_cpl.T[1][0] = np.nan
    pts_cpl.T[1] = 1 / pts_cpl.T[1]
    pts_cpl.T[1] /= np.linalg.norm(pts_cpl.T[1][1:])

    # plotting starts here

    fig = plt.figure(figsize=(7, 3.3), constrained_layout=True)
    widths = [1, 1, 0.8]
    heights = [1, 1]
    gs = fig.add_gridspec(2, 3, width_ratios=widths, height_ratios=heights)
    gs.tight_layout(fig, h_pad=0, w_pad=0, pad=0)

    fom_ax_em = fig.add_subplot(gs[:, 2])
    fom_ax_mech = fom_ax_em.twinx()
    field_ax = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]

    data = {"eps": [], "ez": [], "res": [], "npml": []}
    for cpl, ax in zip(compliance_idx, field_ax):
        dfi = df.iloc[cpl]
        dpml = dfi["dpml"]
        res = dfi["res"]
        lcen = dfi["lcen"]
        n2 = dfi["n2"]
        sim_res = np.ceil(res / (lcen / n2))
        npml = [int(np.ceil(dpml * sim_res)), int(np.ceil(dpml * sim_res))]
        data["ez"].append(get_field(dfi, args["symmetry_cond"]))
        data["eps"].append(get_epsilon(dfi, args["symmetry_cond"]))
        data["res"].append(sim_res)
        data["npml"].append(npml)

    abc = iter(ascii_lowercase)
    colors = iter(
        plt.cm.__dict__[cmap](np.linspace(0.2, 0.8, len(args["compliance_samples"])))
    )
    shapes = iter(Line2D.filled_markers)
    vmin = None
    vmax = None
    for cpl, ax, eps, ez, res in zip(
        compliance_idx, field_ax, data["eps"], data["ez"], data["res"]
    ):
        eps = np.ma.masked_where(eps < np.mean(eps), eps)
        if vmin is None and vmax is None:
            vmin = np.min(np.abs(ez)) ** 2
            vmax = np.max(np.abs(ez)) ** 2
        ez_im = ax.imshow(
            np.abs(ez.T) ** 2, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper"
        )
        ax.imshow(eps.T, cmap="gray", origin="upper", alpha=0.5)

        # add scale bar
        width = res
        line_x = 10
        line_y = eps.shape[1] - 30
        line = Line2D(
            [line_x, line_x + width], [line_y, line_y], linewidth=2, color="white"
        )
        ax.add_line(line)
        ax.text(
            line_x + width / 2,
            line_y + 15,
            r"\SI{1}{\micro\meter}",
            va="center",
            ha="center",
            color="white",
            fontsize=10,
        )

        if cpl == compliance_idx[0]:
            # add colorbars
            cb_ax = inset_axes(
                ax,
                width="35%",
                height="3%",
                bbox_to_anchor=(-0.04, -0.08, 1, 1),
                bbox_transform=ax.transAxes,
            )
            cb = fig.colorbar(ez_im, cax=cb_ax, orientation="horizontal")
            cb.set_ticks([])
            cb.outline.set_visible(True)
            cb_ax.text(
                0.5,
                3.2,
                r"$\abs{\vb*{E}_z}^2$",
                transform=cb_ax.transAxes,
                va="center",
                ha="center",
                color="white",
                fontsize=10,
            )
            cb_ax.text(
                -0.15,
                -1.7,
                # f"$\\num{{{vmin:.0e}}}$",
                f"$0$",
                transform=cb_ax.transAxes,
                va="center",
                ha="left",
                color="white",
                fontsize=9,
            )
            cb_ax.text(
                1.15,
                -1.7,
                # f"$\\num{{{vmax:.0e}}}$",
                f"$1$",
                transform=cb_ax.transAxes,
                va="center",
                ha="right",
                color="white",
                fontsize=9,
            )

        # add the markers
        color = next(colors)
        shape = next(shapes)
        ax.plot(
            25,
            25,
            marker=shape,
            c=color,
            markersize=7,
            markeredgecolor="white",
            markeredgewidth=0.5,
        )
        fom_ax_em.scatter(
            *pts_em[cpl].T, marker=shape, c=color, edgecolor=color, s=30, zorder=10
        )
        fom_ax_mech.scatter(
            *pts_cpl[cpl].T, marker=shape, c=color, edgecolor=color, s=30, zorder=10
        )
        ax.set_title(
            next(abc)
            + f"\\qquad~~\\scalebox{{0.8}}{{$\\omega_{{\\mathrm{{C}}}} = {pts_em[cpl][0]:.2f}$}}",
            loc="left",
            pad=2,
        )
        ax.axis("off")

    for inset_ax in args["inset_ax"]:
        ax = field_ax[inset_ax]
        ez = data["ez"][inset_ax]
        npml = data["npml"][inset_ax]
        ax_ins = zoomed_inset_axes(
            ax, zoom=1.1, bbox_to_anchor=(0, 0, 1.01, 0.31), bbox_transform=ax.transAxes
        )
        x1, x2, y1, y2 = (
            ez.shape[0] // 2 - 20,
            ez.shape[0] // 2 + 20,
            ez.shape[1] - 50,
            ez.shape[1] - 10,
        )
        ax_ins.imshow(
            np.abs(ez.T) ** 2,
            cmap=cmap,
            vmin=np.min(np.abs(ez[x1:x2, y1:y2]) ** 2),
            vmax=np.max(np.abs(ez[x1:x2, y1:y2]) ** 2),
        )
        ax_ins.set_xlim(x1, x2)
        ax_ins.set_ylim(y2, y1)
        ax_ins.set_xticks([])
        ax_ins.set_yticks([])
        for side in ax_ins.spines.values():
            side.set_color("lightgray")
            side.set_color("lightgray")
        inset_rect = Rectangle(
            (x1, y1), x2 - x1, y2 - y1, fc="none", ec="gray", ls="--"
        )
        ax.add_patch(inset_rect)

    mask = np.ones(len(pts_em), dtype=bool)
    mask[compliance_idx] = False
    fom_ax_em.scatter(*pts_em[mask].T, c="k", zorder=10)
    fom_ax_em.plot(*pts_em.T, ls="--", c="lightgray")
    fom_ax_mech.scatter(*pts_cpl[mask].T, c="k", zorder=10)
    fom_ax_mech.plot(*pts_cpl.T, ls="--", c="lightgray")
    fom_ax_em.set_xlabel(r"$\omega_{\mathrm{C}}$", labelpad=-6)
    fom_ax_em.set_xticks([0, 0.5, 1])
    fom_ax_em.set_xticklabels(["0", "", "1"])
    fom_ax_em.set_ylabel(r"$F_{\mathrm{EM}}$ (a.u.)")
    fom_ax_em.yaxis.set_label_position("right")
    fom_ax_em.yaxis.tick_right()
    fom_ax_mech.set_ylabel(r"$F_{\mathrm{C}}^{-1}$ (a.u.)")
    fom_ax_mech.yaxis.set_label_position("left")
    fom_ax_mech.yaxis.tick_left()
    if "fom_ylim" in args.keys():
        fom_ax_em.set_ylim(args["fom_ylim"])
    fom_ax_em.set_title(f"{next(abc)}", loc="left")
    fom_ax_em.tick_params(
        axis="both",
        which="both",
        length=6,
        color="black",
        direction="inout",
        width=1,
        top=False,
        bottom=True,
        left=False,
        right=True,
    )
    fom_ax_mech.tick_params(
        axis="both",
        which="both",
        length=6,
        color="black",
        direction="inout",
        width=1,
        top=False,
        bottom=False,
        left=True,
        right=False,
    )

    fig.savefig(args["plotfile"], bbox_inches="tight", dpi=600)


def main():
    paper_data = {
        "lens_nores_multistart_newloads": {
            "data": "./data/20200130_193502_hybrid_optimization.h5",
            "run": "lens_resonance_penalty_multistart_newloads",
            "datafile": "./data/paper_data_lens_nores_multistart_newloads.pkl",
            "plotfile": "./plots/fom_and_results_lens_nores_multistart_newloads.pdf",
            "compliance_samples": [0.00, 0.10, 0.50, 0.95],
            "inset_ax": [],
            "symmetry_cond": "force_symmetry",
        },
        "mode_converter_multistart_abs2_newgeom_lowbin": {
            "data": "./data/20200130_193502_hybrid_optimization.h5",
            "run": "mode_converter_multistart_abs2_newgeom_lowbin",
            "datafile": "./data/paper_data_mode_converter_multistart_abs2_newgeom_lowbin.pkl",
            "plotfile": "./plots/fom_and_results_mode_converter_multistart_abs2_newgeom_lowbin.pdf",
            "mode_data": "./data/mode_converter_multistart_abs2_newgeom_lowbin_mode_coeffs.json",
            "compliance_samples": [0.00, 0.10, 0.50, 0.90],
            "inset_ax": [],
            "symmetry_cond": "force_symmetry_axis",
        },
    }
    for _, data in paper_data.items():
        if not os.path.isfile(data["datafile"]):
            make_paper_data(data)
        fom_and_results(data, cmap="magma")


if __name__ == "__main__":
    main()
