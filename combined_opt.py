#!/usr/bin/env python3
# vim: set fileencoding=utf-8

from time import time
from datetime import datetime
import nlopt
from autograd import value_and_grad
from autograd import numpy as np
from autograd.extend import primitive, defvjp
from fdfdpy import Simulation
from scipy.constants import epsilon_0, c
from skimage.measure import label

from topopt import geom_util
from autograd_lib.sparse_solvers import solve_coo
from autograd_lib.utilities import ndarray_lru_cache
from problems import default_args, setup_problem_lens, setup_problem_mode_conv


def remove_floating_elements(rho, eps, roi, force_symmetry_axis, thresh=True):
    if force_symmetry_axis >= 0:
        rho = geom_util.make_symmetric(rho, axis=force_symmetry_axis)
        eps = geom_util.make_symmetric(eps, axis=force_symmetry_axis)

    img = np.copy(rho)
    threshed = np.copy(eps)
    if thresh:
        threshed = (eps >= (np.min(eps) + (np.max(eps) - np.min(eps)) / 2)) * 1.0
    labels = label(threshed, background=np.min(threshed))
    img[(roi > 0) & (labels > 1)] = np.min(img[roi > 0])

    return img.ravel()


def make_maxwell_solver(omega, mode, dl, npml, L0):
    def solver(eps):
        return solve_fields(eps, omega, mode, dl, npml, L0)

    return solver


def make_structural_solver(ke, forces, freedofs, fixdofs):
    def solver(young):
        return displace(young, ke, forces, freedofs, fixdofs)

    return solver


def inverse_permutation(indices):
    inverse_perm = np.zeros(len(indices), dtype=np.int64)
    inverse_perm[indices] = np.arange(len(indices), dtype=np.int64)
    return inverse_perm


@ndarray_lru_cache(1)
def _get_dof_indices(freedofs, fixdofs, k_xlist, k_ylist):
    index_map = inverse_permutation(np.concatenate([freedofs, fixdofs]))
    keep = np.isin(k_xlist, freedofs) & np.isin(k_ylist, freedofs)
    i = index_map[k_ylist][keep]
    j = index_map[k_xlist][keep]
    return index_map, keep, np.stack([i, j])


def displace(x_phys, ke, forces, freedofs, fixdofs):
    k_entries, k_ylist, k_xlist = get_k(x_phys, ke)
    index_map, keep, indices = _get_dof_indices(freedofs, fixdofs, k_ylist, k_xlist)
    u_nonzero = solve_coo(k_entries[keep], indices, forces[freedofs])
    u_values = np.concatenate([u_nonzero, np.zeros(len(fixdofs))])

    return u_values[index_map]


def get_k(stiffness, ke):
    nelx, nely = stiffness.shape

    # get position of the nodes of each element in the stiffness matrix
    elx, ely = np.meshgrid(np.arange(nelx), np.arange(nely))
    elx, ely = elx.reshape(-1, 1), ely.reshape(-1, 1)

    n1 = (nely + 1) * (elx + 0) + (ely + 0)
    n2 = (nely + 1) * (elx + 1) + (ely + 0)
    n3 = (nely + 1) * (elx + 1) + (ely + 1)
    n4 = (nely + 1) * (elx + 0) + (ely + 1)
    edof = np.array(
        [2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1, 2 * n3, 2 * n3 + 1, 2 * n4, 2 * n4 + 1]
    ).T[0]

    x_list = np.repeat(edof, 8)  # flat list pointer of each node in an element
    y_list = np.tile(edof, 8).ravel()  # flat list pointer of each node in elem

    # make the stiffness matrix
    kd = stiffness.T.reshape(nelx * nely, 1, 1)
    value_list = (kd * np.tile(ke, kd.shape)).ravel()
    return value_list, y_list, x_list


def compliance(x_phys, u, ke):
    # index map
    nelx, nely = x_phys.shape
    elx, ely = np.meshgrid(np.arange(nelx), np.arange(nely))

    # nodes
    n1 = (nely + 1) * (elx + 0) + (ely + 0)
    n2 = (nely + 1) * (elx + 1) + (ely + 0)
    n3 = (nely + 1) * (elx + 1) + (ely + 1)
    n4 = (nely + 1) * (elx + 0) + (ely + 1)
    all_ixs = np.array(
        [2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1, 2 * n3, 2 * n3 + 1, 2 * n4, 2 * n4 + 1]
    )

    # select from u matrix
    u_selected = u[all_ixs]

    # compute x * U.T @ ke @ U in a vectorized way
    ke_u = np.einsum("ij,jkl->ikl", ke, u_selected)
    ce = np.einsum("ijk,ijk->jk", u_selected, ke_u)
    C = x_phys * ce.T
    return np.sum(C)


@primitive
def solve_fields(eps, omega, mode, dl, npml, L0):
    sim = Simulation(omega, eps, dl, npml, mode["pol"], L0)
    sim.src = mode["src"]
    Ez = sim.solve_fields()[2]
    return Ez


def solve_fields_vjp(ans, eps, omega, mode, dl, npml, L0):
    def vjp(dJdE):
        sim = Simulation(omega, eps, dl, npml, mode["pol"], L0)
        sim.src = dJdE / 1j
        sim.A = sim.A.T
        ez_aj = sim.solve_fields()[2]
        aj_grad = -omega * epsilon_0 * L0 * np.real(ans * ez_aj)
        return aj_grad

    return vjp


defvjp(solve_fields, solve_fields_vjp)


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


def combined_opt(args, mats):
    # simulation parameters
    L0 = args.get("l0", 1e-6)
    lcen = args.get("lcen", 0.57)
    dpml = args.get("dpml", lcen)
    n1 = args.get("n1", 1)
    n2 = args.get("n2", 2)
    pol = args.get("pol", "Ez")
    res = args.get("res", 10)
    use_resonance_penalty = args.get("use_resonance_penalty", False)
    a = args.get("compliance_factor", 0.5)
    force_symmetry_axis = args.get("force_symmetry_axis", -1)
    verbose = args.get("verbose", True)

    # derived parameters
    sim_res = np.ceil(res / (lcen / n2))
    dl = 1 / sim_res
    omega = 2 * np.pi * c / (lcen * L0)
    npml = [int(np.ceil(dpml * sim_res)), int(np.ceil(dpml * sim_res))]

    rho = mats["rho"]
    src = mats["source"]
    merit_roi = mats["merit_roi"]
    design_roi = mats["design_roi"]
    nelx = mats["nelx"]
    nely = mats["nely"]
    forces = mats["forces"]
    fixdofs = mats["fixdofs"]
    freedofs = mats["freedofs"]
    ke = mats["stiffness_matrix"]

    rho2eps, rho2young = make_rho_transforms(args, mats)

    rho2filt = geom_util.make_rho2filt_sp(args["sigma"], mats["design_roi"])
    filt2proj = geom_util.make_filt2proj(args["proj_k"], 0.5, mats["design_roi"])
    rho2proj = geom_util.compose(filt2proj, rho2filt)

    # set up solvers
    mode = {"pol": pol, "src": src}
    fdfd_solver = make_maxwell_solver(omega, mode, dl, npml, L0)
    fem_solver = make_structural_solver(ke, forces, freedofs, fixdofs)

    merit_vals = {
        "Field objective": [],
        "Compliance": [],
        "Binarization": [],
        "Loss": [],
    }

    def binarization(x, cutoff=2):
        _x = x[design_roi == 1]
        _b = -np.log10(np.sum(4 * _x * (1 - _x)) / _x.size)
        # the cutoff is necessary because after a point the log causes issues,
        # and binarization of 1e-2 is definitely good enough
        return np.minimum(_b, cutoff) / cutoff

    def resonance_penalty(field, merit):
        return design_roi[design_roi == 0].size / np.sum(np.abs(field * design_roi))

    def optical_objective(eps):
        field = fdfd_solver(eps)
        overlap = np.abs(np.sum(field * np.conj(merit_roi))) ** 2 / 4
        penalty = resonance_penalty(field, overlap) if use_resonance_penalty else 1
        return overlap * penalty

    def mechanical_objective(young):
        u = fem_solver(young)
        return compliance(young, u, ke)

    def objective(density):
        if force_symmetry_axis >= 0:
            density = geom_util.make_symmetric(density, axis=force_symmetry_axis)

        F = 0
        C = 0
        if a != 1:
            F = optical_objective(rho2eps(density))
        if a != 0:
            C = mechanical_objective(rho2young(density))

        merit_fc = (1 - a) * F - a * C

        B = binarization(rho2proj(density))
        merit = merit_fc + b * B

        # this is dumb but whatever
        if hasattr(F, "_value"):
            merit_vals["Field objective"].append(F._value)
        else:
            merit_vals["Field objective"].append(F)
        if hasattr(C, "_value"):
            merit_vals["Compliance"].append(C._value)
        else:
            merit_vals["Compliance"].append(C)
        if hasattr(merit, "_value"):
            merit_vals["Loss"].append(-merit._value)
        else:
            merit_vals["Loss"].append(-merit)
        if hasattr(B, "_value"):
            merit_vals["Binarization"].append(B._value)
        else:
            merit_vals["Binarization"].append(B)

        return -merit

    def callback(obj, x):
        if verbose:
            print(f'F: {merit_vals["Field objective"][-1]:.5e} | ', end="")
            print(f'C: {merit_vals["Compliance"][-1]:.5e} | ', end="")
            print(f'B: {merit_vals["Binarization"][-1]:.5e} | ', end="")
            print(f'obj: {merit_vals["Loss"][-1]:.5e}')
        if force_symmetry_axis >= 0:
            x = geom_util.make_symmetric(x, axis=force_symmetry_axis)

    def nlopt_obj(x, gd):
        obj, grad = value_and_grad(objective)(np.reshape(x, rho.shape))
        callback(obj, np.reshape(x, rho.shape))
        if gd.size > 0:
            gd[:] = grad.ravel()
        return obj

    x0 = rho.ravel()
    roi = design_roi.ravel()
    lb = np.zeros_like(roi)
    ub = np.copy(roi)
    ub[np.where(roi == 0)] = x0[np.where(roi == 0)]
    lb[np.where(roi == 0)] = x0[np.where(roi == 0)]

    opt = nlopt.opt(nlopt.LD_LBFGS, x0.size)
    opt.set_min_objective(nlopt_obj)
    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)
    opt.set_ftol_rel(args["rel_tol"])
    opt.set_maxeval(args["max_eval"])
    opt.set_maxtime(args["max_time"])

    # optimize first round without binarization penalty
    b = 0
    x0 = opt.optimize(x0)

    # enable binarization, optimize second round
    b = args.get("binarization_factor", 1)
    xopt = opt.optimize(x0)

    # if optimizing with compliance, remove floating elements and do final refinement
    if args["compliance_factor"] > 0:
        x0 = np.reshape(xopt, rho.shape)
        x0 = remove_floating_elements(
            x0, rho2eps(x0), design_roi, force_symmetry_axis, thresh=True
        )
        xopt = opt.optimize(x0)

    return {
        "timestamp": datetime.fromtimestamp(time()).strftime("%Y%m%d_%H%M%S"),
        "options": args,
        "nelx": nelx,
        "nely": nely,
        "forces": forces,
        "fixdofs": fixdofs,
        "freedofs": freedofs,
        "merit_roi": merit_roi,
        "design_roi": design_roi,
        "source": src,
        "rho_final": xopt,
        "stiffness_matrix": ke,
        "field_objective": np.array(merit_vals["Field objective"]),
        "compliance": np.array(merit_vals["Compliance"]),
        "loss": np.array(merit_vals["Loss"]),
        "binarization": np.array(merit_vals["Binarization"]),
    }


if __name__ == "__main__":
    args = default_args()
    args["verbose"] = True
    args["method"] = "bfgs"
    args["force_symmetry_axis"] = -1
    args["compliance_factor"] = 0.5
    args["sigma"] = 1.0
    args["binarization_factor"] = 3
    args["use_resonance_penalty"] = False
    args["n2"] = 1.5

    # specify which example to optimize
    mats = setup_problem_lens(args)
    # mats = setup_problem_mode_conv(args)

    results = combined_opt(args, mats)
