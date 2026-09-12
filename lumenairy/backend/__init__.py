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

(``available_cpus`` -- the affinity-aware process CPU count -- lives in
:mod:`lumenairy.memory`, not here: it is a runtime-resource query, not an
array backend.)

Author: Andrew Traverso
"""

from __future__ import annotations

import importlib as _importlib
from types import ModuleType
from typing import List

from .array import (
    CUPY_AVAILABLE,
    JAX_AVAILABLE,
    array_namespace,
    backend_name,
    is_cupy_array,
    is_jax_array,
    is_numpy_array,
    to_backend,
    to_numpy,
)
from .fft import (
    fft,
    fft2,
    fft_backend_for,
    fftfreq,
    fftshift,
    ifft,
    ifft2,
    ifftshift,
)
from .random import RandomState


def __getattr__(name: str) -> ModuleType:
    """Load :mod:`lumenairy.backend.scipy` on FIRST USE (PEP 562).

    ``backend/scipy.py`` imports ``scipy.linalg`` and ``scipy.special`` at
    module scope.  Together those cost ~570 ms of a ~745 ms
    ``import lumenairy`` on this box, and this package has 41 module-level
    importers (``propagators.rs``, ``elements.elements``, ``analysis.beam_stats``,
    ...) that want only ``array_namespace`` / ``is_*_array`` -- so every user
    paid the rigorous-solver's scipy bill at import.  ``la.backend.scipy.jv(x)``
    and ``from lumenairy.backend import scipy`` behave exactly as before; the
    submodule is simply imported the first time something asks for it, then
    cached in this module's globals so the second access is a dict hit.

    Raises ``AttributeError`` (never ``ImportError``) for an unknown name, so
    ``hasattr`` and ``getattr(..., default)`` keep working.

    Annotated ``-> ModuleType`` rather than ``-> Any``: ``'scipy'`` is the ONLY
    name this forward resolves, and it is always a module, so the narrower type
    is honest.  A future forward that returns a value would widen it.
    """
    if name == 'scipy':
        mod = _importlib.import_module('.scipy', __name__)
        globals()['scipy'] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    """``dir()`` still lists ``scipy`` even before anything has loaded it."""
    return sorted(set(globals()) | {'scipy'})


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
]
