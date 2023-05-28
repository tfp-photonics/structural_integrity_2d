#!/usr/bin/env python3
# vim: set fileencoding=utf-8

import numpy as np
from scipy.constants import c
from fdfdpy import Simulation

from topopt import geom_util


def default_args():
    return {
        "dx": 12,
        "dy": 8,
        "res": 20,
        "l0": 1e-6,
        "lcen": 1,
        "pol": "Ez",
        "dpml": 1,
        "n1": 1,
        "n2": 1.5,
        "sigma": 1,
        "proj_k": 30,
        "penalty": 3,
        "use_same_transforms": True,
        "young_min": 1e-6,
        "young_max": 1,
        "poisson": 0.3,
        "compliance_factor": 0.5,
        "binarization_factor": 2,
        "force_symmetry_axis": -1,
        "vol_frac": 0.5,
        "method": "ipopt",
        "max_eval": 1000,
        "rel_tol": 1e-6,
        "max_time": 0,
        "verbose": False,
        "bragg": False,
    }


def pol2cart(theta, rho):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y


def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho


def get_stiffness_matrix(young, poisson):
    e, nu = young, poisson
    k = np.array(
        [
            1 / 2 - nu / 6,
            1 / 8 + nu / 8,
            -1 / 4 - nu / 12,
            -1 / 8 + 3 * nu / 8,
            -1 / 4 + nu / 12,
            -1 / 8 - nu / 8,
            nu / 6,
            1 / 8 - 3 * nu / 8,
        ]
    )
    return (
        e
        / (1 - nu ** 2)
        * np.array(
            [
                [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
                [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
                [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
                [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
                [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
                [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
                [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
                [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]],
            ]
        )
    )


def setup_problem_lens(args):
    # simulation parameters
    dx = args.get("dx", 10)
    dy = args.get("dy", 10)
    lcen = args.get("lcen", 0.57)
    dpml = args.get("dpml", lcen)
    n2 = args.get("n2", 2)
    poisson = args.get("poisson", 0.3)
    res = args.get("res", 10)
    young_max = args.get("young_max", 1)

    # derived parameters
    sim_res = np.ceil(res / (lcen / n2))
    npml = [int(np.ceil(dpml * sim_res)), int(np.ceil(dpml * sim_res))]

    # set up simulation array
    sim_shape = np.array([round(dx * sim_res), round(dy * sim_res)], dtype=int)

    # base geometry
    rho = np.zeros(sim_shape)
    cx, cy = np.array(rho.shape) // 2
    rho[:, int(2.5 * npml[1]) : -int(2.5 * npml[1])] = 1

    # definition of design region
    roi = np.zeros(sim_shape)
    roi[2 * npml[0] : -2 * npml[0], int(2.5 * npml[1]) : -int(2.5 * npml[1])] = 1

    # initialize design region
    rho[roi == 1] = 0.5

    # set up sources
    src = np.zeros(sim_shape, dtype=np.complex128)
    src[:, npml[1]] = 1

    # define merit region
    merit_roi = np.zeros(sim_shape, dtype=np.float64)
    merit_roi[cx - 1 : cx + 1, -int(1.5 * npml[1])] = 1

    # set up structural problem
    nelx, nely = geom_util.make_submat(roi)(roi).shape

    dofs = np.arange(2 * (nelx + 1) * (nely + 1)).reshape(nelx + 1, nely + 1, 2)

    fixed = np.zeros_like(dofs, dtype=bool)

    fixed[0, :, :] = 1
    fixed[-1, :, :] = 1

    load = np.zeros_like(dofs, dtype=np.float64)

    # load[0, 1:-1, 0] = -1e-2
    # load[-1, 1:-1, 0] = 1e-2

    # load[(nelx + 1) // 2, 0, 1] = 1
    load[(nelx + 1) // 2, 1:-1, 1] = 1e-2

    forces = load.ravel()
    fixdofs = dofs[fixed].ravel()
    freedofs = dofs[~fixed].ravel()

    ke = get_stiffness_matrix(young_max, poisson)

    return {
        "nelx": nelx,
        "nely": nely,
        "forces": forces,
        "fixdofs": fixdofs,
        "freedofs": freedofs,
        "merit_roi": merit_roi,
        "design_roi": roi,
        "rho": rho,
        "source": src,
        "stiffness_matrix": ke,
    }


def setup_problem_mode_conv(args):
    # simulation parameters
    dx = args.get("dx", 10)
    dy = args.get("dy", 10)
    lcen = args.get("lcen", 0.57)
    dpml = args.get("dpml", lcen)
    n1 = args.get("n1", 1)
    n2 = args.get("n2", 2)
    poisson = args.get("poisson", 0.3)
    res = args.get("res", 10)
    young_max = args.get("young_max", 1)

    # derived parameters
    sim_res = np.ceil(res / (lcen / n2))
    dl = 1 / sim_res
    npml = [int(np.ceil(dpml * sim_res)), int(np.ceil(dpml * sim_res))]
    omega = 2 * np.pi * c / (lcen * 1e-6)

    # set up simulation array
    sim_shape = np.array([round(dx * sim_res), round(dy * sim_res)], dtype=int)

    # base geometry
    rho = np.zeros(sim_shape)
    cx, cy = np.array(rho.shape) // 2
    rho[:, int(2.5 * npml[1]) : -int(2.5 * npml[1])] = 1

    # merit region
    merit_roi = np.zeros_like(rho, dtype=np.complex128)
    wvg = np.pad(
        rho[-1, :].reshape(1, -1), ((0, rho.shape[0] - 1), (0, 0)), mode="edge"
    )
    wvg[wvg == 1] = n2 ** 2
    wvg[wvg == 0] = n1 ** 2
    sim = Simulation(omega, wvg, dl, 2 * np.array(npml), "Ez", 1e-6)
    sim.add_mode(0.98 * n2, "x", (npml[0], wvg.shape[1] // 2), wvg.shape[1] - 1)
    sim.setup_modes()
    merit_roi[-npml[0], :] = sim.src[npml[0], :]

    # definition of design region
    roi = np.zeros(sim_shape)
    roi[
        int(3.0 * npml[0]) : -int(3.0 * npml[0]),
        int(2.5 * npml[1]) : -int(2.5 * npml[1]),
    ] = 1

    # initialize design region
    rho[roi == 1] = 0.5

    # set up sources
    src = np.zeros(sim_shape, dtype=np.complex128)
    sim = Simulation(omega, wvg, dl, 2 * np.array(npml), "Ez", 1e-6)
    sim.add_mode(n2, "x", (npml[0], wvg.shape[1] // 2), wvg.shape[1] - 1)
    sim.setup_modes()
    src[npml[0], :] = sim.src[npml[0], :]

    # set up structural problem
    nelx, nely = geom_util.make_submat(roi)(roi).shape

    dofs = np.arange(2 * (nelx + 1) * (nely + 1)).reshape(nelx + 1, nely + 1, 2)

    fixed = np.zeros_like(dofs, dtype=bool)

    load = np.zeros_like(dofs, dtype=np.float64)

    load[0, nely // 2 - 30 : nely // 2 + 30, 0] = -2e-2
    load[-1, nely // 2 - 30 : nely // 2 + 30, 0] = 2e-2

    forces = load.ravel()
    fixdofs = dofs[fixed].ravel()
    freedofs = dofs[~fixed].ravel()

    ke = get_stiffness_matrix(young_max, poisson)

    return {
        "nelx": nelx,
        "nely": nely,
        "forces": forces,
        "fixdofs": fixdofs,
        "freedofs": freedofs,
        "merit_roi": merit_roi,
        "design_roi": roi,
        "rho": rho,
        "source": src,
        "stiffness_matrix": ke,
    }
