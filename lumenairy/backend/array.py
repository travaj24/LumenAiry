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
import os as _os
import threading as _threading
from typing import Any, Optional, cast

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
# LUMENAIRY_DISABLE_JAX=1 forces the JAX path off even when jax is
# INSTALLED: find_spec alone cannot detect an install whose native DLLs are
# blocked (seen 2026-09-01: a Windows Application Control policy blocked
# jaxlib's cpu_feature_guard on a worker box -- the lazy import then died
# mid-initialization and every later touch of the half-imported module
# raised "partially initialized module 'jax'").
JAX_AVAILABLE = (_importlib_util.find_spec('jax') is not None
                 and not _os.environ.get('LUMENAIRY_DISABLE_JAX'))

# Cached lazy module references.  Populated on first access.
_cp = None
_jnp = None
_jax = None


def _get_cupy() -> Optional[Any]:
    """Return the cupy module, importing on first call.  None if CuPy
    is not installed."""
    global _cp
    if _cp is None and CUPY_AVAILABLE:
        import cupy as _cp_mod
        _cp = _cp_mod
    return _cp


def _get_jax() -> Optional[Any]:
    """Return the jax module, importing on first call.  None if JAX is
    not installed."""
    global _jax
    if _jax is None and JAX_AVAILABLE:
        import jax as _jax_mod
        _jax = _jax_mod
    return _jax


def _get_jnp() -> Optional[Any]:
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
    except AttributeError:
        # Older JAX versions did not expose ``jax_mod.core.Tracer``.
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
    # The type
    # predicates above narrow at runtime but mypy can't follow that;
    # cast through ndarray on the explicit-narrow branches.
    if is_numpy_array(x):
        return cast(np.ndarray, x)
    if is_cupy_array(x):
        cp = _get_cupy()
        assert cp is not None  # is_cupy_array guarantees CUPY_AVAILABLE
        return cast(np.ndarray, cp.asnumpy(x))
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
        cp = _get_cupy()
        assert cp is not None  # CUPY_AVAILABLE branch guarantees module
        return cp.asarray(to_numpy(x))
    if JAX_AVAILABLE and xp is _get_jnp():
        if is_jax_array(x):
            return x
        jnp = _get_jnp()
        assert jnp is not None  # JAX_AVAILABLE branch guarantees module
        return jnp.asarray(to_numpy(x))
    raise TypeError(
        f"to_backend: unrecognised target namespace {xp!r}.  Expected "
        f"numpy, cupy, or jax.numpy.")


# ============================================================================
# The degenerate-cluster rule of the JAX twins (ONE library-wide switch)
# ============================================================================
#
# Every differentiable (JAX) solver that eigen-decomposes a modal operator
# routes the eig AND everything downstream of it through
# ``lumenairy.elements.rcwa._core._jax_eig_cluster_adjoint``, whose reverse
# pass is correct at a DEGENERATE eigenvalue cluster (a symmetric structure
# differentiated in a symmetry-breaking direction -- without it such a
# gradient is wrong by percent to hundreds of percent, by an amount that
# differs between BLAS builds).  A caller who knows every point it
# differentiates is far from any symmetry can switch the rule off to drop
# its compile time and memory (the CHANGELOG's cost table).  OFF AT
# A SYMMETRIC POINT THE GRADIENT IS WRONG.
#
# SEMANTICS (as implemented).
# * The effective setting is: the innermost ACTIVE ``jax_cluster_rule``
#   scope of the CURRENT THREAD, else the process value
#   (``set_jax_cluster_rule``; at import ``LUMENAIRY_JAX_CLUSTER_RULE``,
#   default on).  Scopes are removed by identity on exit, so two scopes left
#   out of order do not clobber each other, and a scope in one thread does
#   not affect another thread.
# * It is part of the JAX TRACE CACHE KEY (a ``jax`` config state with
#   ``include_in_trace_context``): a jitted function called under a new
#   setting is retraced with it -- whether it is the same ``jax.jit`` object
#   or a new wrapper of the same callable.  The setting that applies is the
#   one in force AT THE CALL.  If the running JAX lacks that (private)
#   config hook, the setting is still honoured by every new trace but a
#   cached compiled function keeps the setting of its first trace
#   (``jax_cluster_rule_trace_keyed()`` says which; ``jax.clear_caches()``
#   forces a retrace).

_ENV_TRUE = ('1', 'on', 'true', 'yes')
_ENV_FALSE = ('0', 'off', 'false', 'no')


def _parse_rule_env(raw: Optional[str]) -> bool:
    """``LUMENAIRY_JAX_CLUSTER_RULE``: unset / empty -> on; one of
    ``1 on true yes`` / ``0 off false no`` (any case); anything else is
    REFUSED (``ValueError``) rather than silently leaving the rule on."""
    if raw is None or raw.strip() == '':
        return True
    v = raw.strip().lower()
    if v in _ENV_TRUE:
        return True
    if v in _ENV_FALSE:
        return False
    raise ValueError(
        f"LUMENAIRY_JAX_CLUSTER_RULE={raw!r} is not a recognised value: use "
        f"one of {', '.join(_ENV_TRUE)} (rule on) or {', '.join(_ENV_FALSE)} "
        f"(rule off), or unset it.")


_JAX_CLUSTER_RULE = _parse_rule_env(_os.environ.get('LUMENAIRY_JAX_CLUSTER_RULE'))
_RULE_SCOPES = _threading.local()
_RULE_STATE: Any = None          # the jax config state, built on first use
_RULE_UNSET: Any = None          # its "no thread-local value" sentinel


def _rule_state() -> Any:
    """The ``jax`` config state that carries the effective setting into
    JAX's trace cache key, or ``None`` when JAX (or the private hook) is not
    available."""
    global _RULE_STATE, _RULE_UNSET
    if _RULE_STATE is None:
        try:
            from jax._src import config as _jc
            st = _jc.bool_state(
                name='lumenairy_jax_cluster_rule_trace_key',
                default=_JAX_CLUSTER_RULE,
                help='lumenairy: the degenerate-cluster gradient rule '
                     '(managed by lumenairy.backend.set_jax_cluster_rule).',
                include_in_jit_key=True, include_in_trace_context=True)
            _RULE_UNSET = st.get_local()
            st.set_global(_JAX_CLUSTER_RULE)
            _RULE_STATE = st
        except (ImportError, AttributeError, TypeError):  # pragma: no cover
            # JAX absent (ImportError), the private hook gone (AttributeError:
            # no bool_state / get_local / set_global) or its signature changed
            # (TypeError).  A duplicate-name registration -- which jax reports
            # as a bare Exception -- cannot happen: the state is built once per
            # process and cached in _RULE_STATE.
            _RULE_STATE = False
    return None if _RULE_STATE is False else _RULE_STATE


def _scope_stack() -> "list[jax_cluster_rule]":
    st = getattr(_RULE_SCOPES, 'stack', None)
    if st is None:
        st = _RULE_SCOPES.stack = []
    return st


def _sync_rule_state() -> None:
    state = _rule_state()
    if state is None:
        return
    stack = _scope_stack()
    state.set_local(stack[-1].enabled if stack else _RULE_UNSET)


def jax_cluster_rule_enabled() -> bool:
    """The effective setting of the degenerate-cluster rule for solves traced
    now in this thread (see :func:`set_jax_cluster_rule`)."""
    stack = getattr(_RULE_SCOPES, 'stack', None)
    if stack:
        return bool(stack[-1].enabled)
    return _JAX_CLUSTER_RULE


def jax_cluster_rule_trace_keyed() -> bool:
    """True when the setting is part of JAX's trace cache key on this JAX
    (a jitted function is retraced when it changes); False when the private
    JAX hook is unavailable and a cached compiled function keeps the setting
    of its first trace."""
    return _rule_state() is not None


def set_jax_cluster_rule(enabled: bool) -> bool:
    """Set the PROCESS value of the degenerate-cluster rule of every JAX
    twin (default ON); returns the previous process value.  A
    :class:`jax_cluster_rule` scope active in a thread overrides it there.

    OFF makes a gradient through a degenerate eigenvalue cluster WRONG (a
    symmetric structure -- a four-fold cell, a mirror-symmetric grating at
    exactly normal incidence, an isotropic layer, a uniform layer of a
    stack -- differentiated in a direction that breaks the symmetry) and
    leaves every other gradient unchanged; use it only away from any
    symmetry, to save the rule's compile time and memory (the CHANGELOG's
    cost table).  The setting in force WHEN A JITTED FUNCTION IS
    CALLED applies: it is part of JAX's trace cache key, so a change
    retraces (see ``jax_cluster_rule_trace_keyed``)."""
    global _JAX_CLUSTER_RULE
    previous = _JAX_CLUSTER_RULE
    _JAX_CLUSTER_RULE = bool(enabled)
    state = _rule_state()
    if state is not None:
        state.set_global(_JAX_CLUSTER_RULE)
    return previous


class jax_cluster_rule:
    """Context manager: ``with jax_cluster_rule(False): ...`` -- the rule's
    setting for this THREAD while the scope is active (see
    :func:`set_jax_cluster_rule` for what OFF means).  Scopes nest; one left
    out of order is removed by identity, so it cannot clobber another
    scope's setting."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

    def __enter__(self) -> "jax_cluster_rule":
        _scope_stack().append(self)
        _sync_rule_state()
        return self

    def __exit__(self, *exc: Any) -> None:
        stack = _scope_stack()
        for i in range(len(stack) - 1, -1, -1):
            if stack[i] is self:
                del stack[i]
                break
        _sync_rule_state()


__all__ = [
    'CUPY_AVAILABLE',
    'JAX_AVAILABLE',
    'jax_cluster_rule',
    'jax_cluster_rule_enabled',
    'jax_cluster_rule_trace_keyed',
    'set_jax_cluster_rule',
    'is_numpy_array',
    'is_cupy_array',
    'is_jax_array',
    'array_namespace',
    'backend_name',
    'to_numpy',
    'to_backend',
]
