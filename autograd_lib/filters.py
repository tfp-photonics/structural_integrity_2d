# vim: set fileencoding=utf-8

import warnings
import scipy.ndimage
from autograd.extend import primitive, defvjp


@primitive
def gaussian_filter(x, width, mode="reflect"):
    if mode == "constant":
        warnings.warn(
            UserWarning(
                "Filtering with constant boundaries is not unitary, "
                "i.e. it does not preserve total pixel intensity!"
            )
        )
    if mode in ["nearest", "mirror"]:
        raise ValueError(f"Gradient not defined for boundary mode: {mode}.")
    return scipy.ndimage.gaussian_filter(x, width, mode=mode)


def vjp_gaussian_filter(ans, x, width, mode):
    del ans, x  # unused

    def vjp(g):
        return gaussian_filter(g, width, mode=mode)

    return vjp


defvjp(gaussian_filter, vjp_gaussian_filter)
