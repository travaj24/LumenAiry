"""
Array-backend namespace dispatch.

Lumenairy supports four numerical backends interchangeably:

* **NumPy** (always available) -- the default CPU path.
* **CuPy** (optional) -- NVIDIA GPU acceleration; same API as NumPy.
* **JAX** (optional) -- functional/immutable array library with XLA
  JIT compilation, automatic differentiation, and CPU+GPU+TPU
  backends.
* **SciPy** -- not an array backend per se; used as a fallback FFT
  provider for the NumPy path (handled in :mod:`lumenairy._fft`).

The dispatch model follows the Python Array API standard: code is
written against ``xp.*`` where ``xp`` is the namespace returned by
:func:`array_namespace`, and the same code runs unmodified across
all three array backends.  Backend selection happens at the boundary
-- whichever array type the user passes in determines which
namespace the function uses internally.

The four backends are mutually exclusive within a single call: an
array is either NumPy, CuPy, or JAX, and the dispatch picks one
namespace.  Mixing array types in a single call is not supported.

See ``REFERENCES.txt`` Section I for the Array API standard.

Author: Andrew Traverso
"""

from __future__ import annotations

import importlib.util as _importlib_util
from typing import Any

import numpy as np

# ============================================================================
# Optional-backend availability checks (lazy)
# ============================================================================
#
# Heavy optional dependencies (CuPy, JAX) are NOT loaded at import time.
# Only their availability is detected via importlib.util.find_spec, which
# is essentially free and does not actually import the package.  The
# modules themselves are loaded lazily on first call to a function that
# needs them, via the _get_*() accessors below.
#
# Keeping JAX_AVAILABLE / CUPY_AVAILABLE as module-level constants
# preserves the public API used by ~60+ callers across the package.

CUPY_AVAILABLE = _importlib_util.find_spec('cupy') is not None
JAX_AVAILABLE = _importlib_util.find_spec('jax') is not None

# Cached lazy module references.  Populated on first access.
_cp = None
_jnp = None
_jax = None


def _get_cupy():
    """Return the cupy module, importing on first call.  None if CuPy
    is not installed."""
    global _cp
    if _cp is None and CUPY_AVAILABLE:
        import cupy as _cp_mod
        _cp = _cp_mod
    return _cp


def _get_jax():
    """Return the jax module, importing on first call.  None if JAX is
    not installed."""
    global _jax
    if _jax is None and JAX_AVAILABLE:
        import jax as _jax_mod
        _jax = _jax_mod
    return _jax


def _get_jnp():
    """Return the jax.numpy module, importing on first call.  None if
    JAX is not installed."""
    global _jnp
    if _jnp is None and JAX_AVAILABLE:
        import jax.numpy as _jnp_mod
        _jnp = _jnp_mod
    return _jnp


# ============================================================================
# Array type predicates
# ============================================================================

def is_numpy_array(x: Any) -> bool:
    """Return True only for NumPy ndarrays.  CuPy / JAX arrays return
    False even though they may share base classes via the Python
    Array API.

    Robust against NumPy 2.x: ``ndarray.device`` exists in NumPy 2.x
    as part of the array-API surface, so duck-typing on ``.device``
    no longer distinguishes NumPy from CuPy / JAX.
    """
    return isinstance(x, np.ndarray)


def is_cupy_array(x: Any) -> bool:
    """Return True if ``x`` is a CuPy ndarray.  False if CuPy is not
    installed."""
    if not CUPY_AVAILABLE:
        return False
    cp = _get_cupy()
    if cp is None:
        return False
    return isinstance(x, cp.ndarray)


def is_jax_array(x: Any) -> bool:
    """Return True if ``x`` is a JAX array (concrete or traced).

    Covers ``jax.Array`` (post-0.4) and JIT-traced ``Tracer`` objects.
    """
    if not JAX_AVAILABLE:
        return False
    jax_mod = _get_jax()
    if jax_mod is None:
        return False
    if isinstance(x, jax_mod.Array):
        return True
    try:
        return isinstance(x, jax_mod.core.Tracer)
    except Exception:
        return False


# ============================================================================
# Namespace dispatch
# ============================================================================

def array_namespace(*arrays: Any) -> Any:
    """Return the array namespace (``numpy``, ``cupy``, or
    ``jax.numpy``) appropriate for the given arrays.

    All arrays must belong to the same backend (or be Python scalars).
    Mixing arrays from different backends raises ``TypeError`` --
    explicit conversion is required at the call site.

    If no arrays are given, or all arguments are Python scalars,
    returns NumPy.
    """
    saw_jax = False
    saw_cupy = False
    saw_numpy = False

    for a in arrays:
        if a is None:
            continue
        if is_jax_array(a):
            saw_jax = True
        elif is_cupy_array(a):
            saw_cupy = True
        elif is_numpy_array(a):
            saw_numpy = True
        # Python scalars / lists / tuples don't pin a backend.

    n = sum([saw_jax, saw_cupy, saw_numpy])
    if n > 1:
        raise TypeError(
            "lumenairy: array_namespace was given arrays from multiple "
            "backends (NumPy / CuPy / JAX).  Explicitly convert all "
            "inputs to a single backend before calling.")

    if saw_jax:
        return _get_jnp()
    if saw_cupy:
        return _get_cupy()
    return np


def backend_name(xp: Any) -> str:
    """Short, human-readable name of an xp namespace returned by
    :func:`array_namespace`."""
    if xp is np:
        return 'numpy'
    if CUPY_AVAILABLE and xp is _get_cupy():
        return 'cupy'
    if JAX_AVAILABLE and xp is _get_jnp():
        return 'jax'
    return getattr(xp, '__name__', repr(xp))


# ============================================================================
# Conversion helpers
# ============================================================================

def to_numpy(x: Any) -> np.ndarray:
    """Materialise ``x`` as a NumPy ndarray on the host.

    Use this at I/O boundaries (HDF5 / Zarr writes, plotting,
    .npy save) where downstream code expects a host NumPy array.
    """
    if is_numpy_array(x):
        return x
    if is_cupy_array(x):
        return _get_cupy().asnumpy(x)
    if is_jax_array(x):
        return np.asarray(x)
    return np.asarray(x)


def to_backend(x: Any, xp: Any) -> Any:
    """Convert ``x`` to the namespace ``xp`` (numpy / cupy /
    jax.numpy).

    Cheap if ``x`` is already on the target backend.  For
    cross-backend conversion this materialises through the host.
    """
    if xp is np:
        return to_numpy(x)
    if CUPY_AVAILABLE and xp is _get_cupy():
        if is_cupy_array(x):
            return x
        return _get_cupy().asarray(to_numpy(x))
    if JAX_AVAILABLE and xp is _get_jnp():
        if is_jax_array(x):
            return x
        return _get_jnp().asarray(to_numpy(x))
    raise TypeError(
        f"to_backend: unrecognised target namespace {xp!r}.  Expected "
        f"numpy, cupy, or jax.numpy.")


__all__ = [
    'CUPY_AVAILABLE',
    'JAX_AVAILABLE',
    'is_numpy_array',
    'is_cupy_array',
    'is_jax_array',
    'array_namespace',
    'backend_name',
    'to_numpy',
    'to_backend',
]
