"""
lumenairy.backend -- numerical-backend dispatch and FFT/RNG/scipy
compatibility layers.

Provides a single set of public entry points for backend-aware
operations:

* :mod:`lumenairy.backend.array` -- ``array_namespace``,
  ``is_numpy_array`` / ``is_cupy_array`` / ``is_jax_array``,
  ``to_numpy`` / ``to_backend``.
* :mod:`lumenairy.backend.fft` -- ``fft2`` / ``ifft2`` / 1-D FFT /
  ``fftshift`` / ``fftfreq`` -- preserves the
  pyFFTW > scipy.fft > numpy.fft priority chain plus CuPy and JAX
  short-circuits.
* :mod:`lumenairy.backend.random` -- ``RandomState`` wrapper that
  dispatches to NumPy / CuPy stateful generators or JAX functional
  PRNG keys.
* :mod:`lumenairy.backend.scipy` -- backend-aware scipy.special /
  scipy.linalg dispatch.
* :mod:`lumenairy.backend.cpu` -- ``available_cpus`` (affinity-aware
  CPU count).

Author: Andrew Traverso
"""

from __future__ import annotations

from .array import (
    array_namespace,
    is_numpy_array,
    is_cupy_array,
    is_jax_array,
    backend_name,
    to_numpy,
    to_backend,
    CUPY_AVAILABLE,
    JAX_AVAILABLE,
)
from .fft import (
    fft2, ifft2, fft, ifft,
    fftshift, ifftshift, fftfreq,
    fft_backend_for,
)
from .random import RandomState
from . import scipy as scipy
from .cpu import available_cpus


__all__ = [
    'array_namespace',
    'is_numpy_array',
    'is_cupy_array',
    'is_jax_array',
    'backend_name',
    'to_numpy',
    'to_backend',
    'CUPY_AVAILABLE',
    'JAX_AVAILABLE',
    'fft2', 'ifft2', 'fft', 'ifft',
    'fftshift', 'ifftshift', 'fftfreq',
    'fft_backend_for',
    'RandomState',
    'scipy',
    'available_cpus',
]
