# vim: set fileencoding=utf-8

from functools import reduce
from autograd import numpy as np

from autograd_lib import filters


def make_rho2filt_sp(sigma, roi):
    def rho2filt(rho):
        filt = filters.gaussian_filter(rho, sigma, mode="reflect")
        filt = filt * roi + (1 - roi) * rho
        return filt

    return rho2filt


def make_filt2proj(a, b, roi):
    def filt2proj(filt):
        if a == 0:
            return filt
        num = np.tanh(a * b) + np.tanh(a * (filt - b))
        denom = np.tanh(a * b) + np.tanh(a * (1 - b))
        proj = num / denom
        # only apply projection to design area
        proj = proj * roi + (1 - roi) * filt
        return proj

    return filt2proj


def make_proj2eps(eps_min, eps_max, penalty=1):
    def proj2eps(proj):
        eps = eps_min + (eps_max - eps_min) * proj ** penalty
        return eps

    return proj2eps


def make_submat(roi):
    def submat(x):
        out = x[~np.all(roi == 0, axis=1)]
        out = out[:, ~np.all(roi == 0, axis=0)]
        return out

    return submat


def make_symmetric(x, axis=0):
    """Enforces symmetry along an axis in a 2D numpy array.
    """
    _x = x.T if axis == 1 else x
    _x = np.vsplit(_x, 2)[0]
    _x = np.concatenate([_x, np.flipud(_x)], axis=0)
    _x = _x.T if axis == 1 else _x
    return _x


def compose(*F):
    def _compose2(f, g):
        return lambda arg: f(g(arg))

    return reduce(_compose2, F)
