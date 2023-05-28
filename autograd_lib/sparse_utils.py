# vim: set fileencoding=utf-8

import autograd.numpy as anp
from scipy.sparse import coo_matrix
from sksparse.cholmod import cholesky
from sksparse.cholmod import CholmodNotPositiveDefiniteError


def make_sparse(a_entries, a_indices, shape):
    return coo_matrix((a_entries, a_indices), shape=shape).tocsc()


def is_symmetric(A):
    sym_err = A - A.H
    return anp.allclose(anp.abs(sym_err.data), 0)


def is_positive_definite(A):
    if is_symmetric(A):
        try:
            cholesky(A)
            return True
        except CholmodNotPositiveDefiniteError:
            return False
    return False
