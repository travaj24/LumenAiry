"""
Backend-aware FFT dispatch.

Provides a single public 2-D / 1-D FFT entry point that all
lumenairy modules can call regardless of the input array's backend.
The priority chain matches the long-standing behaviour of
:func:`lumenairy.propagation._fft2`:

    For CuPy arrays            -> ``cupy.fft`` (cuFFT)
    For JAX arrays             -> ``jax.numpy.fft`` (XLA)
    For NumPy arrays:
        1. **pyFFTW** (preferred) -- multi-threaded CPU FFT with
           per-shape plan caching.  Falls back automatically on
           allocation failure.
        2. **scipy.fft** -- multi-threaded ``workers=-1`` pocketfft.
        3. **numpy.fft** -- single-threaded fallback (always
           available).

JAX FFT dispatch is added on top of the existing chain so that
``fft2(jnp.ones(...))`` returns a JAX array suitable for
``jax.grad`` / ``jax.jit``.

This module is the public FFT entry point for new code.
:mod:`lumenairy.propagation` still owns the pyFFTW plan cache, the
scipy thread pool, and the bad-shape blacklist; this module
delegates into that infrastructure for the NumPy / CuPy paths so
the priority chain and caches are shared, not duplicated.

Author: Andrew Traverso
"""

from __future__ import annotations

import numpy as np

from .array import (
    JAX_AVAILABLE,
    is_jax_array,
    is_cupy_array,
)

if JAX_AVAILABLE:
    import jax.numpy as _jnp
else:
    _jnp = None


# ============================================================================
# 2-D FFT entry points
# ============================================================================

def fft2(x):
    """Backend-aware 2-D forward FFT on the last two axes.

    Dispatch order:

    * JAX array  -> ``jax.numpy.fft.fft2``
    * Otherwise (NumPy / CuPy)  -> :func:`lumenairy.propagation._fft2`
      which preserves the pyFFTW > scipy.fft > numpy.fft priority
      chain plus CuPy short-circuit.
    """
    if is_jax_array(x):
        return _jnp.fft.fft2(x)
    from ..propagators import propagation as _prop
    return _prop._fft2(x)


def ifft2(x):
    """Backend-aware 2-D inverse FFT.  Same priority chain as
    :func:`fft2`."""
    if is_jax_array(x):
        return _jnp.fft.ifft2(x)
    from ..propagators import propagation as _prop
    return _prop._ifft2(x)


# ============================================================================
# 1-D FFT entry points
# ============================================================================

def fft(x, axis=-1, n=None):
    """Backend-aware 1-D forward FFT."""
    if is_jax_array(x):
        return _jnp.fft.fft(x, n=n, axis=axis)
    if is_cupy_array(x):
        import cupy as cp
        return cp.fft.fft(x, n=n, axis=axis)
    from ..propagators import propagation as _prop
    if _prop.USE_SCIPY_FFT and _prop.SCIPY_FFT_AVAILABLE:
        import scipy.fft as _sp
        return _sp.fft(x, n=n, axis=axis, workers=_prop.SCIPY_FFT_WORKERS)
    return np.fft.fft(x, n=n, axis=axis)


def ifft(x, axis=-1, n=None):
    """Backend-aware 1-D inverse FFT."""
    if is_jax_array(x):
        return _jnp.fft.ifft(x, n=n, axis=axis)
    if is_cupy_array(x):
        import cupy as cp
        return cp.fft.ifft(x, n=n, axis=axis)
    from ..propagators import propagation as _prop
    if _prop.USE_SCIPY_FFT and _prop.SCIPY_FFT_AVAILABLE:
        import scipy.fft as _sp
        return _sp.ifft(x, n=n, axis=axis, workers=_prop.SCIPY_FFT_WORKERS)
    return np.fft.ifft(x, n=n, axis=axis)


# ============================================================================
# FFT shifts and frequency grids
# ============================================================================

def fftshift(x, axes=None):
    """Backend-dispatched ``fft.fftshift``."""
    if is_jax_array(x):
        return _jnp.fft.fftshift(x, axes=axes)
    if is_cupy_array(x):
        import cupy as cp
        return cp.fft.fftshift(x, axes=axes)
    return np.fft.fftshift(x, axes=axes)


def ifftshift(x, axes=None):
    """Backend-dispatched ``fft.ifftshift``."""
    if is_jax_array(x):
        return _jnp.fft.ifftshift(x, axes=axes)
    if is_cupy_array(x):
        import cupy as cp
        return cp.fft.ifftshift(x, axes=axes)
    return np.fft.ifftshift(x, axes=axes)


def fftfreq(n, d=1.0, xp=None):
    """Backend-aware ``fft.fftfreq``.

    Pass ``xp`` to materialise the grid in a specific backend
    (CuPy GPU, JAX traced).  When ``xp`` is None, NumPy is used.
    """
    if xp is None:
        return np.fft.fftfreq(n, d=d)
    return xp.fft.fftfreq(n, d=d)


# ============================================================================
# Diagnostic
# ============================================================================

def fft_backend_for(x) -> str:
    """Report which backend a call to :func:`fft2` would use for the
    given input.  Returns: ``'jax'`` / ``'cupy'`` / ``'pyfftw'`` /
    ``'scipy.fft'`` / ``'numpy.fft'``."""
    if is_jax_array(x):
        return 'jax'
    if is_cupy_array(x):
        return 'cupy'
    from ..propagators import propagation as _prop
    shape = tuple(x.shape) if hasattr(x, 'shape') else ()
    if (_prop.USE_PYFFTW and _prop.PYFFTW_AVAILABLE
            and len(shape) >= 2
            and shape[0] >= _prop.FFTW_MIN_SIZE
            and shape not in _prop._PYFFTW_BAD_SHAPES):
        return 'pyfftw'
    if _prop.USE_SCIPY_FFT and _prop.SCIPY_FFT_AVAILABLE:
        return 'scipy.fft'
    return 'numpy.fft'


__all__ = [
    'fft2', 'ifft2',
    'fft', 'ifft',
    'fftshift', 'ifftshift', 'fftfreq',
    'fft_backend_for',
]
