"""
Backend-aware SciPy-compatible function dispatch.

A handful of lumenairy modules use ``scipy.special`` /
``scipy.linalg`` operations on array data.  When the input array is
JAX, those calls need to land in ``jax.scipy.*`` instead so the
operation stays in the JAX trace and is differentiable / JIT-able.

Dispatch by input array type:

    NumPy / CuPy arrays -> ``scipy.*`` (host) or ``cupyx.scipy.*``
    JAX arrays          -> ``jax.scipy.*``

Author: Andrew Traverso
"""

from __future__ import annotations

import numpy as np

import scipy.special as _sp_special
import scipy.linalg as _sp_linalg

from ._array import (
    JAX_AVAILABLE,
    CUPY_AVAILABLE,
    is_jax_array,
    is_cupy_array,
)

if JAX_AVAILABLE:
    import jax.scipy.special as _jax_special
    import jax.scipy.linalg as _jax_linalg
else:
    _jax_special = None
    _jax_linalg = None


def _dispatch_special(name, x, *args, **kwargs):
    """Generic dispatch for ``scipy.special`` functions on input
    ``x``."""
    if is_jax_array(x):
        fn = getattr(_jax_special, name, None)
        if fn is None:
            raise NotImplementedError(
                f"jax.scipy.special.{name} is not available.")
        return fn(x, *args, **kwargs)
    if is_cupy_array(x):
        try:
            import cupyx.scipy.special as _cu_special
            fn = getattr(_cu_special, name, None)
            if fn is None:
                import cupy as cp
                x_host = cp.asnumpy(x)
                result_host = getattr(_sp_special, name)(x_host, *args, **kwargs)
                return cp.asarray(result_host)
            return fn(x, *args, **kwargs)
        except ImportError:
            import cupy as cp
            x_host = cp.asnumpy(x)
            result_host = getattr(_sp_special, name)(x_host, *args, **kwargs)
            return cp.asarray(result_host)
    return getattr(_sp_special, name)(x, *args, **kwargs)


def jv(v, x):
    """Bessel function of the first kind, order ``v``, at ``x``."""
    if is_jax_array(x):
        if hasattr(_jax_special, 'bessel_jv'):
            return _jax_special.bessel_jv(v, x)
        raise NotImplementedError(
            "jv is not available in jax.scipy.special for arbitrary "
            "orders.  Convert to NumPy first.")
    return _dispatch_special('jv', x, v)


def erf(x):
    """Error function."""
    return _dispatch_special('erf', x)


def gammaln(x):
    """Log-gamma."""
    return _dispatch_special('gammaln', x)


def expi(x):
    """Exponential integral Ei."""
    return _dispatch_special('expi', x)


def solve(A, b):
    """Solve ``A x = b`` on the appropriate backend."""
    if is_jax_array(A) or is_jax_array(b):
        return _jax_linalg.solve(A, b)
    if is_cupy_array(A) or is_cupy_array(b):
        import cupy as cp
        return cp.linalg.solve(A, b)
    return _sp_linalg.solve(A, b)


def lstsq(A, b, **kwargs):
    """Least-squares solve."""
    if is_jax_array(A) or is_jax_array(b):
        import jax.numpy as jnp
        return jnp.linalg.lstsq(A, b, **kwargs)
    if is_cupy_array(A) or is_cupy_array(b):
        import cupy as cp
        return cp.linalg.lstsq(A, b, **kwargs)
    return _sp_linalg.lstsq(A, b, **kwargs)


def eigh(A):
    """Hermitian eigendecomposition."""
    if is_jax_array(A):
        import jax.numpy as jnp
        return jnp.linalg.eigh(A)
    if is_cupy_array(A):
        import cupy as cp
        return cp.linalg.eigh(A)
    return _sp_linalg.eigh(A)


__all__ = ['jv', 'erf', 'gammaln', 'expi', 'solve', 'lstsq', 'eigh']
