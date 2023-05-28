# vim: set fileencoding=utf-8

import numpy as np
from pyMKL import pardisoSolver
from autograd.extend import primitive, defvjp

from .utilities import ndarray_lru_cache

from .sparse_utils import make_sparse, is_symmetric, is_positive_definite


def _Solver(A, check_spd=False):
    sym = is_symmetric(A)
    pos = False
    if check_spd:
        pos = is_positive_definite(A)

    if A.dtype in (np.complex64, np.complex128):
        if pos:
            mtype = 4
        elif sym:
            mtype = -4
        else:
            mtype = 13
    else:
        if pos:
            mtype = 2
        elif sym:
            mtype = -2
        else:
            mtype = 11

    solver = pardisoSolver(A, mtype=mtype)

    def solve(b):
        x = solver.run_pardiso(phase=13, rhs=b)
        solver.clear()
        return x

    return solve


@ndarray_lru_cache(1)
def _get_solver(entries_a, indices_a, size):
    """Get a solver for applying the desired matrix factorization."""
    A = make_sparse(entries_a, indices_a, (size, size))
    return _Solver(A)


@primitive
def solve_coo(entries_a, indices_a, b):
    solver = _get_solver(entries_a, indices_a, b.size)
    return solver(b)


def vjp_solve_coo_entries(ans, entries_a, indices_a, b):
    def vjp(grad_ans):
        lambda_ = solve_coo(entries_a, indices_a[::-1], grad_ans)
        i, j = indices_a
        return -lambda_[i] * ans[j]

    return vjp


defvjp(solve_coo, vjp_solve_coo_entries)
