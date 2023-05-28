# vim: set fileencoding=utf-8

import numpy as np
from functools import lru_cache, wraps


class _WrappedArray:
    def __init__(self, value):
        self.value = value
        self.__hash = hash(repr(self.value.ravel()))

    def __eq__(self, other):
        return np.array_equal(self.value, other.value)

    def __hash__(self):
        return self.__hash


def ndarray_lru_cache(*args, **kwargs):
    def decorator(func):
        @lru_cache(*args, **kwargs)
        def cached_func(*args, **kwargs):
            args = tuple(a.value if isinstance(a, _WrappedArray) else a for a in args)
            kwargs = {
                k: v.value if isinstance(v, _WrappedArray) else v
                for k, v in kwargs.items()
            }
            return func(*args, **kwargs)

        @wraps(func)
        def wrapper(*args, **kwargs):
            args = tuple(
                _WrappedArray(a) if isinstance(a, np.ndarray) else a for a in args
            )
            kwargs = {
                k: _WrappedArray(v) if isinstance(v, np.ndarray) else v
                for k, v in kwargs.items()
            }
            return cached_func(*args, **kwargs)

        wrapper.cache_info = cached_func.cache_info
        wrapper.cache_clear = cached_func.cache_clear

        return wrapper

    return decorator
