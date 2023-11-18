# vim: set fileencoding=utf-8

from autograd.extend import primitive, defvjp, defjvp

from autograd_lib.sparse_utils import make_sparse, get_entries_indices, make_io_matrices


@primitive
def sp_mult(entries, indices, x):
    A = make_sparse(entries, indices, (x.size, x.size))
    return A @ x


def vjp_sp_mult_entries_reverse(ans, entries, indices, x):
    # x^T @ dA/de^T @ v => the outer product of x and v using the indices of A
    ia, ja = indices

    def vjp(v):
        return v[ia] * x[ja]

    return vjp


def vjp_sp_mult_x_reverse(b, entries, indices, x):
    # dx/de^T @ A^T @ v => multiplying A^T by v

    def vjp(v):
        return sp_mult(entries, indices[::-1], v)

    return vjp


defvjp(sp_mult, vjp_sp_mult_entries_reverse, None, vjp_sp_mult_x_reverse)


def jvp_sp_mult_entries_forward(g, b, entries, indices, x):
    # dA/de @ x @ g => use `g` as the entries into A and multiply by x
    return sp_mult(g, indices, x)


def jvp_sp_mult_x_forward(g, b, entries, indices, x):
    # A @ dx/de @ g -> simply multiply A @ g
    return sp_mult(entries, indices, g)


defjvp(sp_mult, jvp_sp_mult_entries_forward, None, jvp_sp_mult_x_forward)


@primitive
def spsp_mult(entries_a, indices_a, entries_x, indices_x, N):
    A = make_sparse(entries_a, indices_a, (N, N))
    X = make_sparse(entries_x, indices_x, (N, N))
    B = A @ X
    return get_entries_indices(B)


def vjp_spsp_mult_entries_a(b_out, entries_a, indices_a, entries_x, indices_x, N):
    # make the indices matrices for A
    _, indices_b = b_out
    Ia, Oa = make_io_matrices(indices_a, N)

    def vjp(v):
        # multiply the v_entries with X^T using the indices of B
        entries_v, _ = v
        entries_vxt, indices_vxt = spsp_mult(
            entries_v, indices_b, entries_x, indices_x[::-1], N
        )

        # turn this into a sparse matrix and convert to the basis of A's indices
        VXT = make_sparse(entries_vxt, indices_vxt, (N, N))
        M = Ia.T @ VXT @ Oa.T

        return M.diagonal()

    return vjp


def vjp_spsp_mult_entries_x(b_out, entries_a, indices_a, entries_x, indices_x, N):
    # get the transposes of the original problem
    entries_b, indices_b = b_out
    b_T_out = entries_b, indices_b[::-1]

    # call the vjp maker for AX=B using the substitution A=>X^T, X=>A^T, B=>B^T
    vjp_XT_AT = vjp_spsp_mult_entries_a(
        b_T_out, entries_x, indices_x[::-1], entries_a, indices_a[::-1], N
    )

    # return the function of the transpose vjp maker being called on the backprop vector
    return lambda v: vjp_XT_AT(v)


defvjp(
    spsp_mult,
    vjp_spsp_mult_entries_a,
    None,
    vjp_spsp_mult_entries_x,
    None,
    None,
)


def jvp_spsp_mult_entries_a(g, b_out, entries_a, indices_a, entries_x, indices_x, N):
    raise NotImplementedError


def jvp_spsp_mult_entries_x(g, b_out, entries_a, indices_a, entries_x, indices_x, N):
    raise NotImplementedError


defjvp(
    spsp_mult,
    jvp_spsp_mult_entries_a,
    None,
    jvp_spsp_mult_entries_x,
    None,
    None,
)
