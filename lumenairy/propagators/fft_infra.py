"""
FFT backend / cache / default-config infrastructure for the propagator family.
==============================================================================

v5.1.0 split (Agent C): the formerly-monolithic
``lumenairy/propagators/propagation.py`` is reorganised into six
submodules sharing this one infrastructure layer.  This module owns:

* Backend availability flags + lazy loaders (CuPy / pyFFTW / scipy.fft).
* FFT backend configuration globals + their setters / getters
  (``set_fft_threads`` / ``set_pyfftw_planner`` / etc.).
* The multi-slot pyFFTW plan cache + double-buffer ping-pong.
* The ASM transfer-function / frequency-grid / band-limit LRU caches.
* The unified ``_fft2`` / ``_ifft2`` / ``_fft2_nd`` / ``_ifft2_nd``
  dispatchers (CuPy / pyFFTW / scipy.fft / numpy).
* The four library-wide ``DEFAULT_*`` knobs and their setters / getters.
* JAX dtype resolvers (``_resolve_jax_complex_dtype`` /
  ``_resolve_jax_real_dtype``).
* The shared ``_validate_propagator_inputs`` guard.

Public API contract: every name previously importable from
``lumenairy.propagators.propagation`` is re-exported there unchanged
(see ``propagation.py``).  No public behaviour changed in the v5.1.0
split -- it is a pure file-level refactor.

Author:  Andrew Traverso
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Dict, Optional

import numpy as np

# ============================================================================
# Optional backend imports
# ============================================================================

# GPU acceleration via CuPy (lazy-loaded -- ~150 ms init cost on
# CUDA-equipped boxes, none of which is needed for the NumPy / pyFFTW
# default path).
import importlib.util as _importlib_util_for_cupy
CUPY_AVAILABLE = _importlib_util_for_cupy.find_spec('cupy') is not None
cp = None  # populated lazily on first use


def _ensure_cupy_loaded():
    global cp
    if cp is None and CUPY_AVAILABLE:
        import cupy as _c
        cp = _c
    return cp is not None


def _is_cupy_array(x):
    """
    Reliable CuPy array check.

    Historically this module used ``hasattr(x, 'device')`` as a duck-type
    test for a CuPy device array.  That broke in NumPy 2.x: ``ndarray``
    now exposes ``.device`` as part of the Python Array API standard, so
    every NumPy array falsely tests as a CuPy array and gets routed
    through the (unusable without CUDA) CuPy FFT path.  Use ``isinstance``
    against the real CuPy type instead.
    """
    if not CUPY_AVAILABLE:
        return False
    if cp is None and not _ensure_cupy_loaded():
        return False
    return isinstance(x, cp.ndarray)

# Multi-threaded FFT via pyFFTW (lazy-loaded -- pyFFTW pulls in a
# substantial native lib at import time, so we only load it when
# something actually wants the pyFFTW path).
import importlib.util as _importlib_util
PYFFTW_AVAILABLE = _importlib_util.find_spec('pyfftw') is not None
pyfftw = None  # populated lazily by _ensure_pyfftw_loaded()


def _ensure_pyfftw_loaded():
    global pyfftw
    if not PYFFTW_AVAILABLE:
        return False
    if pyfftw is None:
        import pyfftw as _p
        _p.interfaces.cache.enable()
        _p.interfaces.cache.set_keepalive_time(30.0)
        pyfftw = _p
    return True

# SciPy FFT (multi-threaded via workers parameter, always available with scipy)
try:
    import scipy.fft as _scipy_fft
    SCIPY_FFT_AVAILABLE = True
except ImportError:
    SCIPY_FFT_AVAILABLE = False

# Affinity-aware CPU count -- respects cgroups / taskset / Python 3.13+
# process_cpu_count so we don't oversubscribe a restricted machine.
from ..memory import available_cpus as _available_cpus


# ============================================================================
# FFT backend configuration
# ============================================================================

# Number of threads for pyFFTW.  Initialised once at import time from
# :func:`lumenairy._backends.available_cpus`; pass a positive
# int to override.  ``0`` falls back to pyFFTW's own default (all cores
# as seen by libfftw3 regardless of affinity).
FFTW_THREADS = _available_cpus()

# Master switch -- route CPU FFTs through pyFFTW when the library is
# installed.  On a 4096x4096 complex128 FFT, pyFFTW + 24 threads is
# measured ~10x faster than NumPy's single-threaded pocketfft and
# ~6x faster than SciPy's ``workers=-1`` pocketfft.  Plan caching is
# enabled above, so repeated calls at the same shape hit a cached plan
# and don't re-plan.
#
# Memory overhead: pyFFTW allocates an aligned workspace per (shape,
# dtype) pair.  On typical optics grids (<= 8192) this is a few hundred
# MB; set ``USE_PYFFTW = False`` if memory is tighter than throughput.
USE_PYFFTW = PYFFTW_AVAILABLE

# Master switch -- set True to route CPU FFTs through scipy.fft.
# Used when pyFFTW is unavailable.  scipy.fft supports multi-threading
# via the ``workers`` parameter and is always available with scipy.
USE_SCIPY_FFT = True

# Number of threads for scipy.fft (-1 = all available cores).  Memory
# usage is unchanged by threading -- pocketfft splits a single FFT
# across cores sharing the same input/output buffers, not by
# allocating per-thread copies.
SCIPY_FFT_WORKERS = -1

# Minimum grid dimension (along axis-0) before pyFFTW is invoked.
# Below this size the single-thread dispatch avoids the planning /
# buffer-alignment overhead that dominates for tiny FFTs.  256 is
# empirically where pyFFTW + plan cache starts to beat NumPy.
FFTW_MIN_SIZE = 256

# Shapes that have failed pyFFTW allocation once in this process.  On
# subsequent calls we skip straight to the scipy fallback for those
# shapes rather than eating the allocation failure repeatedly and
# thrashing the plan cache.  Gets reset when the user explicitly
# flushes via ``reset_fft_backend()``.
_PYFFTW_BAD_SHAPES: set[tuple] = set()

# Single toggle: when True, ``_fft2`` / ``_ifft2`` wrap the pyFFTW call
# in try/except and fall back to scipy.fft (or numpy.fft) on any
# exception.  Allocation failures on large grids (contiguous aligned
# buffer can't fit in free RAM) are the common case, but the
# wrapper also catches plan-cache eviction races and thread-pool
# exhaustion.  Disable via ``set_fft_fallback(False)`` if you want
# pyFFTW errors to propagate (e.g. to detect genuine bugs in the
# backend rather than silently degrading performance).
PYFFTW_FALLBACK_ON_ERROR = True

# Default complex dtype for functions that need to allocate a fresh
# complex array when the caller passes a real input.  Functions that
# operate on an input complex field (e.g. :func:`apply_real_lens`,
# :func:`angular_spectrum_propagate`) infer the target dtype from
# ``E_in.dtype`` and only fall back to this default when the input is
# not complex.  Flip to ``np.complex64`` for ~2x memory + throughput
# at the cost of ~80 dB (not ~140 dB) effective cumulative dynamic
# range; the in-library kernel-phase and phase-screen mitigations
# keep ASM and lens accuracy at single-precision-FFT noise-floor
# levels rather than degrading further with phase magnitude.
DEFAULT_COMPLEX_DTYPE = np.complex128


def set_default_complex_dtype(dtype: Any) -> None:
    """Set the default complex precision used when functions need to
    allocate a fresh complex array (e.g. when callers pass real-valued
    inputs).  Functions that operate on already-complex inputs preserve
    the caller's dtype, so this affects only the "no input dtype"
    paths and any code that deliberately reads ``DEFAULT_COMPLEX_DTYPE``.

    Pass ``np.complex64`` for ~1.6x FFT throughput and ~2x memory
    headroom.  The library carries explicit kernel-phase mod-2pi
    folding everywhere accuracy depends on the absolute kernel
    argument, so single-precision propagation stays at the float32
    noise floor (~80 dB cumulative dynamic range) rather than
    degrading further with phase magnitude.

    Pass ``np.complex128`` to revert (this is the library default).
    """
    global DEFAULT_COMPLEX_DTYPE
    dt = np.dtype(dtype)
    if dt not in (np.dtype(np.complex64), np.dtype(np.complex128)):
        raise ValueError(
            f"set_default_complex_dtype: dtype must be np.complex64 or "
            f"np.complex128, got {dt!r}.")
    DEFAULT_COMPLEX_DTYPE = dt
    # Switching precision invalidates the H cache (different storage
    # dtype keys all the cached entries); the freq-grid / band-limit
    # caches are dtype-independent so leave them alone.
    with _ASM_CACHE_LOCK:
        _H_CACHE.clear()


def get_default_complex_dtype() -> Any:
    """Return the currently-configured default complex dtype."""
    return DEFAULT_COMPLEX_DTYPE


# ---------------------------------------------------------------------------
# v4.16.2 (pre-v5.0 prep) -- 3 additional library-wide default-config knobs
# ---------------------------------------------------------------------------
#
# Parallel to ``set_default_complex_dtype`` above.  Each knob:
#   * stores its value in a module-level global,
#   * has a paired ``get_default_*`` accessor,
#   * is validated at set-time (raises ValueError on bad input),
#   * is consumed via the canonical resolver function (e.g.
#     ``_resolve_default_dy``) at function entry rather than via the
#     def-time default-value evaluation pattern (which would freeze
#     the global at import time and ignore later ``set_default_*``
#     calls).
#
# Consumers honour the knob via the resolver pattern::
#
#     def apply_xxx(field, *, dy=None, ...):
#         dy = _resolve_default_dy(dy)
#         ...
#
# v4.16.3 (audit P2-NEW-F1-3 + P2-NEW-F1-4) status of consumer rollout:
#
#   * ``DEFAULT_REAL_DTYPE``       -- consumed at one site
#     (``propagate_ensemble``'s no-input-dtype real-accumulator
#     fallback, ``ensemble.py:~347``).  v4.16.2 shipped this consumer
#     behind an unreachable ``except`` branch; v4.16.3 re-shapes the
#     consumer so the ``in_dtype is None`` path is the canonical
#     fallback.
#   * ``DEFAULT_WAVE_PROPAGATOR``  -- v5.1.0: consumer-wired across
#     ``apply_real_lens`` (``_lens_real.py``),
#     ``apply_real_lens_traced`` (``_lens_traced.py``), and the
#     ``method=`` argument of ``propagate_through_system``
#     (``system.py``).  v4.16.2 / v4.16.3 / v5.0.x shipped the SET /
#     GET API only and emitted a one-shot UserWarning that no
#     library consumer existed; v5.1.0 retires that warning and
#     wires the resolver at the entry points listed above.
#   * ``DEFAULT_DY``               -- v5.1.0: consumer-wired across
#     ``apply_real_lens`` and ``apply_real_lens_traced``.  Same
#     status history as ``DEFAULT_WAVE_PROPAGATOR``: API-only at
#     v4.16.2-v5.0.x, library-wide rollout at v5.1.0.
#
# Multiprocess / fork notes (v4.16.3, audit P3-NEW-F1-2 + P3-NEW-F1-3)
# --------------------------------------------------------------------
# The three ``DEFAULT_*`` module-level globals below are PLAIN PYTHON
# globals -- they live in the parent process's import of this module.
# Two correctness corollaries follow:
#
# 1. **Pickle / fork-safety**: under
#    ``multiprocessing.get_context('spawn')`` (the only context that
#    works portably on Windows + macOS Python 3.8+) child workers
#    RE-IMPORT this module from scratch.  Each child sees the
#    library-default ``DEFAULT_*`` values -- NOT whatever the parent
#    set them to via the ``set_default_*`` accessors.  A parent that
#    calls ``set_default_real_dtype(np.float32)`` then submits work to
#    a ``ProcessPoolExecutor`` (default = spawn on Windows / macOS)
#    silently loses the override in each worker.  Workaround until v5.0
#    rolls out a shared-state mechanism: re-call ``set_default_*`` at
#    the top of each worker callable, or pass the value explicitly per
#    call.
# 2. **One-shot latch fork-safety**: the
#    ``_DEFAULT_WAVE_PROPAGATOR_NO_CONSUMER_WARNED`` and
#    ``_DEFAULT_DY_NO_CONSUMER_WARNED`` latches (and the sibling
#    ``_MULTIWL_AVG_WARNED`` in ``optimize.core``,
#    ``_TRANSFER_JAX_WARNING_EMITTED`` in ``raytrace.jax_trace``) reset
#    to ``False`` per worker on spawn.  An optimisation loop running
#    ``differential_evolution(workers=8)`` will see each worker re-emit
#    the one-shot warning the first time it triggers the underlying
#    condition -- up to N copies of the warning across N workers
#    instead of one across the process tree.  Not a correctness bug
#    (each emission is still capped at one per worker process) but
#    documented here so the user isn't surprised.
#
# Neither limitation is fixed in v4.16.3.  The fixes require either a
# ``multiprocessing.Manager``-backed shared dict (heavy) or a stamped
# environment-variable handshake (fragile across the spawn boundary);
# both push out into the v5.0 default-config-knob consumer-wiring
# scope.

DEFAULT_REAL_DTYPE = np.float64

DEFAULT_WAVE_PROPAGATOR = 'asm'

DEFAULT_DY = None  # None means "match dx"

# v4.16.3 (audit P2-NEW-F1-4): module-level latches that gated the
# "API-only in v4.16.2/v4.16.3; consumer wiring follows in v5.0"
# one-shot UserWarning emitted by the wave_propagator / dy setters.
# v5.1.0 retired both warnings (the resolver is now wired across
# ``apply_real_lens`` / ``apply_real_lens_traced`` /
# ``propagate_through_system``), so the setters no longer reach the
# emission branch.  The latch globals are preserved (pinned permanently
# to ``True``) for back-compat with code that introspects the module's
# attribute surface; flipping them back to ``False`` no longer revives
# the warning.
_DEFAULT_WAVE_PROPAGATOR_NO_CONSUMER_WARNED = True
_DEFAULT_DY_NO_CONSUMER_WARNED = True


def set_default_real_dtype(dtype: Any) -> None:
    """Set the library-wide default real precision.

    Parallel to :func:`set_default_complex_dtype` for real-valued
    arrays (slope grids, intensity accumulators, OPD maps).  Pass
    ``np.float32`` for ~2x memory headroom (matches the float32 noise
    floor of the FFT kernels), or ``np.float64`` for double-precision
    accumulators (the library default).

    The change applies to subsequent allocations; existing arrays
    keep their dtype.

    Consumer scope (v4.16.3 + v5.1.0):

    * :func:`lumenairy.propagate_ensemble` -- the no-input-dtype
      real-accumulator fallback (the canonical ``in_dtype is None``
      path).  v4.16.3 wiring.

    Other user-facing entry points correctly DERIVE their real
    dtype from the input complex array's real counterpart (e.g.
    ``complex64 -> float32``) and therefore do NOT need a default
    knob.  The library follows the convention "match the input's
    precision" wherever a complex input array is present; the
    default-real-dtype knob fills the gap only at sites that
    allocate a fresh real array with no complex parent.  See the
    v5.1.0 release notes for the per-site rationale.
    """
    global DEFAULT_REAL_DTYPE
    dt = np.dtype(dtype)
    if dt not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError(
            f"set_default_real_dtype: dtype must be np.float32 or "
            f"np.float64, got {dt!r}.")
    DEFAULT_REAL_DTYPE = dt


def get_default_real_dtype() -> Any:
    """Return the currently-configured default real dtype."""
    return DEFAULT_REAL_DTYPE


def set_default_wave_propagator(name: str) -> None:
    """Set the library-wide default ``wave_propagator`` choice.

    Avoids having to pass ``wave_propagator='fresnel'`` (or any
    non-ASM choice) to every entry point.  Valid names: ``'asm'``,
    ``'sas'``, ``'fresnel'``, ``'rayleigh_sommerfeld'``, ``'rs'``.

    Consumers that honour this default (v5.1.0):

    * :func:`lumenairy.apply_real_lens` -- the analytic split-step
      real-lens propagator.  ``wave_propagator=None`` (the new
      default) resolves via :func:`get_default_wave_propagator`.
      Pass ``wave_propagator=<name>`` per-call to override.
    * :func:`lumenairy.apply_real_lens_traced` -- the per-pixel
      ray-traced real-lens propagator.  Same resolution rule.
    * :func:`lumenairy.propagate_through_system` -- the ``method=``
      argument (``method='asm'`` was the legacy hardcoded default;
      ``method=None`` now resolves via the library default).  Note:
      ``propagate_through_system`` natively supports the
      ``'asm' / 'sas' / 'fresnel'`` subset; if the library default
      resolves to ``'rs'`` / ``'rayleigh_sommerfeld'`` the call
      raises ``ValueError`` with a clear migration recipe.

    Pre-v5.1.0 the setter stored the value but no library consumer
    read it (API-only) and a one-shot ``UserWarning`` advertised the
    gap; v5.1.0 retired that warning.
    """
    global DEFAULT_WAVE_PROPAGATOR
    _VALID = ('asm', 'sas', 'fresnel', 'rayleigh_sommerfeld', 'rs')
    if not isinstance(name, str):
        raise ValueError(
            f"set_default_wave_propagator: name must be str, got "
            f"{type(name).__name__}: {name!r}.")
    if name not in _VALID:
        raise ValueError(
            f"set_default_wave_propagator: unknown propagator "
            f"{name!r}.  Valid choices: {_VALID}.")
    DEFAULT_WAVE_PROPAGATOR = name


def get_default_wave_propagator() -> str:
    """Return the currently-configured default wave_propagator name."""
    return DEFAULT_WAVE_PROPAGATOR


def set_default_dy(value: Any) -> None:
    """Set the library-wide default ``dy`` (anamorphic grid spacing).

    Pass ``None`` (the library default) to mean "match ``dx``" -- the
    classic isotropic-grid behavior.  Pass a positive float to set a
    library-wide anamorphic spacing without per-call kwarg.

    Consumers that honour this default (v5.1.0):

    * :func:`lumenairy.apply_real_lens` -- ``dy=None`` (the default)
      resolves via ``get_default_dy()``; if the resolved value is
      still ``None``, it falls back to ``dx`` (the classic isotropic
      grid).  Pass ``dy=<float>`` per-call to override.
    * :func:`lumenairy.apply_real_lens_traced` -- same resolution
      chain.  Note: the traced variant requires square pixels and
      raises ``ValueError`` if the resolved ``dy`` differs from
      ``dx`` more than 1e-15 relative; this guard is preserved from
      v4.x.

    Pre-v5.1.0 the setter stored the value but no library consumer
    read it (API-only) and a one-shot ``UserWarning`` advertised the
    gap; v5.1.0 retired that warning.
    """
    global DEFAULT_DY
    if value is None:
        DEFAULT_DY = None
    else:
        try:
            v = float(value)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"set_default_dy: value must be None or a positive float, "
                f"got {type(value).__name__}: {value!r} ({e!s}).") from None
        if not (v > 0.0 and np.isfinite(v)):
            raise ValueError(
                f"set_default_dy: value must be None or a positive finite "
                f"float, got {v!r}.")
        DEFAULT_DY = v


def get_default_dy() -> Any:
    """Return the currently-configured default ``dy`` (None means
    "match dx").
    """
    return DEFAULT_DY


def _resolve_jax_complex_dtype(dtype: Any = None) -> Any:
    """Resolve a JAX complex dtype that honours ``set_default_complex_dtype``.

    v4.13.0 (audit L2): JAX-side code historically hard-cast to
    ``jnp.complex64`` (or read ``jax.config.jax_enable_x64`` directly),
    which silently overrides the user's
    :func:`set_default_complex_dtype` setting and gives float32-precision
    answers with no warning.  This helper centralises the
    NumPy-default-dtype -> JAX-dtype mapping so every JAX entry point
    obeys the same configuration knob.

    Parameters
    ----------
    dtype : numpy / jax complex dtype, optional
        When given, takes precedence (per-call override).  Must resolve
        to ``np.complex64`` or ``np.complex128``.  Pass ``None`` (the
        default) to read the library-wide default via
        :func:`get_default_complex_dtype`.

    Returns
    -------
    jax_dtype : ``jnp.complex64`` or ``jnp.complex128``

    Notes
    -----
    Imports ``jax.numpy`` lazily so this module stays importable
    without JAX.  ``ImportError`` is raised when JAX is unavailable
    and the caller asks for a JAX dtype.

    When ``complex128`` is requested but JAX's ``jax_enable_x64`` is
    False, auto-enable it with a one-shot :class:`RuntimeWarning`
    (matches the long-standing behaviour in
    :func:`fit_canonical_polynomials_jax` -- without ``x64`` enabled
    JAX silently truncates every ``complex128`` allocation back to
    ``complex64``).
    """
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError(
            "_resolve_jax_complex_dtype: JAX is not installed."
        ) from exc
    target = np.dtype(dtype) if dtype is not None else np.dtype(
        DEFAULT_COMPLEX_DTYPE)
    if target == np.dtype(np.complex64):
        return jnp.complex64
    if target == np.dtype(np.complex128):
        if not jax.config.jax_enable_x64:
            import warnings as _warnings
            _warnings.warn(
                "_resolve_jax_complex_dtype: complex128 requested but "
                "jax_enable_x64 is False.  JAX silently truncates "
                "complex128 to complex64 unless x64 is enabled.  "
                "Auto-enabling via jax.config.update('jax_enable_x64', "
                "True) for the rest of this Python session so "
                "set_default_complex_dtype(np.complex128) actually "
                "yields double precision on the JAX path.",
                RuntimeWarning, stacklevel=2,
            )
            jax.config.update('jax_enable_x64', True)
        return jnp.complex128
    raise ValueError(
        f"_resolve_jax_complex_dtype: dtype must resolve to "
        f"np.complex64 or np.complex128, got {target!r}.")


def _resolve_jax_real_dtype(dtype: Any = None) -> Any:
    """Resolve a JAX real dtype paired with the default complex dtype.

    v4.13.0 (audit L2 companion): real-valued JAX kernels (phase
    arrays, masks, prefactors) also need a precision twin to match the
    complex dtype.  Returns ``jnp.float64`` when the default complex
    dtype is ``np.complex128`` and ``jnp.float32`` for
    ``np.complex64``.

    Parameters mirror :func:`_resolve_jax_complex_dtype`.  Auto-
    enables ``jax_enable_x64`` (with a one-shot warning) when the
    request needs float64.
    """
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError(
            "_resolve_jax_real_dtype: JAX is not installed."
        ) from exc
    target = np.dtype(dtype) if dtype is not None else np.dtype(
        DEFAULT_COMPLEX_DTYPE)
    if target == np.dtype(np.complex64):
        return jnp.float32
    if target == np.dtype(np.complex128):
        if not jax.config.jax_enable_x64:
            import warnings as _warnings
            _warnings.warn(
                "_resolve_jax_real_dtype: float64 paired with complex128 "
                "requested but jax_enable_x64 is False.  Auto-enabling "
                "for the rest of this Python session.",
                RuntimeWarning, stacklevel=2,
            )
            jax.config.update('jax_enable_x64', True)
        return jnp.float64
    raise ValueError(
        f"_resolve_jax_real_dtype: dtype must resolve to "
        f"np.complex64 or np.complex128, got {target!r}.")


# ----------------------------------------------------------------------------
# Multi-slot pyFFTW plan cache (3.2.14)
# ----------------------------------------------------------------------------
# Earlier the cache held *one* plan per direction (forward / inverse).
# That worked when a single call site dominated, but optimization
# loops, JonesField (Ex/Ey at one shape, then a 3D batch shape),
# Maslov (mixes the input grid and per-axis 1-D FFTs), and any code
# that propagates at multiple sizes thrashes the single slot --
# every call between two shapes has to reallocate the bound buffer
# and re-plan.  An LRU dict keyed by ``(direction, shape, dtype,
# threads)`` lets several recently-used plans stay resident, with
# bounded memory because old entries fall out the back of the LRU.
#
# Entry layout (4.12 double-buffer):
#   OrderedDict[key] = {
#       'plans':   [plan_a, plan_b],   # one in-place pyFFTW plan per slot
#       'bufs':    [buf_a,  buf_b],    # aligned workspaces, plan_i bound to bufs[i]
#       'lock':    threading.Lock(),   # serialises execution across slots
#       'idx':     int,                # next slot to use, toggled each call
#       'flag':    'FFTW_ESTIMATE' | 'FFTW_MEASURE' | ... (per-entry planner)
#       'calls':   int,                # call count for auto-promote tracking
#       'promoted':bool,               # True once promoted to MEASURE
#   }
# The two-buffer ping-pong lets ``_fft2`` / ``_ifft2`` return ``bufs[idx]``
# directly instead of paying for a buf.copy(); the next call at the same
# key writes into ``bufs[1 - idx]`` so the previous caller's reference
# stays valid until they release it (or call _fft2/_ifft2 again at the
# same key, in which case they MUST have consumed the prior result -- see
# the docstring on _fft2 / _ifft2).
_PYFFTW_PLAN_CACHE: 'OrderedDict[tuple, dict]' = OrderedDict()
_PYFFTW_PLAN_CACHE_SIZE = 8       # # of plans to keep resident
_PYFFTW_PLAN_LOCK = threading.Lock()

# pyFFTW planner flag.  FFTW_ESTIMATE (default) takes ~1 ms to plan
# and gives "good enough" performance.  FFTW_MEASURE takes
# ~0.1-1 s to plan but produces ~20% faster execution; worth opting
# into for production runs that re-use the same shapes thousands of
# times.  FFTW_PATIENT (~10-60 s plan) gives a few more % beyond
# MEASURE.  Switch at runtime via ``set_pyfftw_planner()``.
_PYFFTW_PLAN_FLAGS = ('FFTW_ESTIMATE',)

# 4.12 auto-promote: when an ESTIMATE-flagged plan key gets called this
# many times, the plan is evicted and rebuilt with FFTW_MEASURE so the
# steady-state hot path benefits from the better planner without making
# the user opt in explicitly.  Below the threshold the one-shot MEASURE
# planning cost (~0.1-1 s on a 4k grid) isn't recouped, so we stay with
# ESTIMATE.  Disable globally via ``set_fft_auto_promote(False)`` for
# bit-for-bit deterministic-output workflows.
_PYFFTW_AUTO_PROMOTE = True
_PYFFTW_AUTO_PROMOTE_THRESHOLD = 5


def set_pyfftw_planner(planner: str = 'FFTW_ESTIMATE') -> None:
    """Configure the pyFFTW planning effort.

    Parameters
    ----------
    planner : ``'FFTW_ESTIMATE'`` (default) / ``'FFTW_MEASURE'`` /
              ``'FFTW_PATIENT'`` / ``'FFTW_EXHAUSTIVE'``
        Higher-effort planners take longer to build plans but produce
        ~20-30% faster FFT execution.  Note that planning measures
        actually overwrite the supplied buffer, so plan rebuilds may
        clobber whatever was in the buffer.

    Notes
    -----
    Switching the planner clears the existing plan cache so that
    subsequent calls re-plan with the new flag.  For optimisation
    runs that hit a small number of shapes, calling this once at
    script start with ``'FFTW_MEASURE'`` is the recommended
    production setup.
    """
    global _PYFFTW_PLAN_FLAGS, _PYFFTW_AUTO_PROMOTE_LOGGED
    valid = {'FFTW_ESTIMATE', 'FFTW_MEASURE',
             'FFTW_PATIENT', 'FFTW_EXHAUSTIVE'}
    if planner not in valid:
        raise ValueError(
            f"set_pyfftw_planner: planner must be one of {sorted(valid)}, "
            f"got {planner!r}.")
    _PYFFTW_PLAN_FLAGS = (planner,)
    # Clear the plan cache so subsequent calls re-plan with the new flag.
    with _PYFFTW_PLAN_LOCK:
        _PYFFTW_PLAN_CACHE.clear()
    # Reset the auto-promote log gate: a user-driven planner change
    # is the start of a fresh planning regime, so any subsequent
    # promotion (which will only fire if the user re-selected
    # ESTIMATE) should be announced again.
    _PYFFTW_AUTO_PROMOTE_LOGGED = False


def reset_fft_backend() -> None:
    """Clear the pyFFTW plan cache and the bad-shape blacklist.

    Useful after a transient memory crunch has passed (e.g. one big
    allocation has since been freed) so subsequent large FFTs can
    retry the fast pyFFTW path instead of staying pinned to the
    scipy fallback.  Also drops every cached plan, freeing all
    aligned workspaces.
    """
    global _PYFFTW_BAD_SHAPES, _PYFFTW_AUTO_PROMOTE_LOGGED
    _PYFFTW_BAD_SHAPES = set()
    with _PYFFTW_PLAN_LOCK:
        _PYFFTW_PLAN_CACHE.clear()
    # Reset the auto-promote one-shot log gate so that the next
    # ESTIMATE->MEASURE promotion (if any) is announced again --
    # useful when a long-running notebook session calls
    # reset_fft_backend() between experiments.
    _PYFFTW_AUTO_PROMOTE_LOGGED = False
    # Also flush the H / kxky / bandlimit caches -- they are sized
    # for "what was just being computed" so dropping them on backend
    # reset matches the user's mental model.
    clear_asm_caches()
    if PYFFTW_AVAILABLE and _ensure_pyfftw_loaded():
        try:
            pyfftw.interfaces.cache.disable()
            pyfftw.interfaces.cache.enable()
            pyfftw.interfaces.cache.set_keepalive_time(30.0)
        except (RuntimeError, ValueError, AttributeError, MemoryError) as _exc:
            # pyfftw can fail to reset its cache on bad PyFFTW
            # internal state; surface the failure as a warning instead
            # of swallowing it silently (audit feedback 3.5.4).
            import warnings as _warnings
            _warnings.warn(
                f"pyfftw cache reset in reset_fft_backend failed: "
                f"{type(_exc).__name__}: {_exc}.  pyFFTW will "
                f"continue using the existing cache.",
                RuntimeWarning, stacklevel=2)


def set_fft_plan_cache_size(n: int) -> None:
    """Set the maximum number of pyFFTW plans kept resident.  Default
    is 8.  Pass ``1`` to mimic the legacy single-slot behaviour."""
    global _PYFFTW_PLAN_CACHE_SIZE
    _PYFFTW_PLAN_CACHE_SIZE = max(1, int(n))
    with _PYFFTW_PLAN_LOCK:
        while len(_PYFFTW_PLAN_CACHE) > _PYFFTW_PLAN_CACHE_SIZE:
            _PYFFTW_PLAN_CACHE.popitem(last=False)


def warmup_fft_plans(shapes: Any, dtype: Optional[Any] = None, threads: Optional[int] = None) -> int:
    """Pre-build pyFFTW plans for the given shapes so the first
    propagation at each shape pays the planning cost only once at
    warmup, not inside a hot loop.

    The first ``angular_spectrum_propagate`` / ``apply_real_lens`` /
    similar call at a new shape spends ~100-1000 ms inside FFTW's
    plan optimiser.  Subsequent calls at the same shape reuse the
    cached plan in ~10-100 ms.  For optimisation runs and parameter
    sweeps that hit the same handful of shapes thousands of times
    this is a no-op (the cache warms itself on the first iter); for
    interactive sessions where the user wants the first call to be
    fast, call this once at script start.

    Parameters
    ----------
    shapes : iterable of tuple
        2-D shapes ``(Ny, Nx)`` (or higher-D batched shapes) to
        plan for.  Both forward and inverse plans are pre-built.
    dtype : numpy dtype, optional
        Complex dtype for the planned buffers.  Defaults to
        ``np.complex128``.  Pass ``np.complex64`` for the
        single-precision path used by the JAX/CuPy paths.
    threads : int, optional
        Threads per plan.  Defaults to
        :func:`lumenairy._backends.available_cpus`.

    Returns
    -------
    n_plans : int
        Number of plans built (2 per requested shape; both
        directions).  ``0`` if pyFFTW is not installed.

    Examples
    --------
    Pre-warm a typical optimisation grid::

        import lumenairy as la
        la.warmup_fft_plans([(1024, 1024)])
        # Subsequent ASM calls at N=1024 hit the cached plan immediately.
    """
    if not PYFFTW_AVAILABLE or not _ensure_pyfftw_loaded():
        return 0
    if dtype is None:
        dtype = np.complex128
    if threads is None:
        threads = max(1, _available_cpus())
    n = 0
    for shape in shapes:
        for direction in ('fwd', 'inv'):
            _get_or_make_plan(direction, tuple(shape), dtype,
                                int(threads))
            n += 1
    return n


# Whether we have emitted the first-promote info log this process.
_PYFFTW_AUTO_PROMOTE_LOGGED = False


def _build_plan_entry(direction, shape_t, dt, threads, flag):
    """Build a fresh double-buffered plan entry.

    Allocates two aligned workspaces and one in-place pyFFTW plan per
    buffer (each plan is bound to its own buffer so the two can be
    used in alternation without any ``update_arrays`` overhead).
    Returns the cache-entry dict; the caller is responsible for
    storing it under the appropriate key.

    Note: on ``flag = 'FFTW_MEASURE'`` (or stronger), the pyFFTW
    constructor runs the planner against the supplied buffer, which
    overwrites its contents.  That's fine here because the buffers
    are freshly allocated by this function and aren't visible to any
    caller until we return.
    """
    if len(shape_t) <= 2:
        axes = (0, 1)
    else:
        axes = (len(shape_t) - 2, len(shape_t) - 1)
    direction_flag = 'FFTW_FORWARD' if direction == 'fwd' else 'FFTW_BACKWARD'

    bufs = [pyfftw.empty_aligned(shape_t, dtype=dt),
            pyfftw.empty_aligned(shape_t, dtype=dt)]
    plans = [
        pyfftw.FFTW(
            bufs[0], bufs[0],
            axes=axes,
            direction=direction_flag,
            flags=(flag,),
            threads=max(1, int(threads)),
        ),
        pyfftw.FFTW(
            bufs[1], bufs[1],
            axes=axes,
            direction=direction_flag,
            flags=(flag,),
            threads=max(1, int(threads)),
        ),
    ]
    return {
        'plans': plans,
        'bufs': bufs,
        'lock': threading.Lock(),
        'idx': 0,
        'flag': str(flag),
        'calls': 0,
        'promoted': False,
    }


def _promote_entry_to_measure(entry, direction, shape_t, dt, threads):
    """Rebuild the cache entry under FFTW_MEASURE in-place.

    Returns the new entry (a fresh dict produced by
    :func:`_build_plan_entry`).  Caller holds ``_PYFFTW_PLAN_LOCK`` and
    is responsible for updating the cache.  Emits a one-time info log
    so users see "lumenairy: promoting (..., MEASURE) ..." the first
    time any key promotes.
    """
    import logging
    _log = logging.getLogger('lumenairy')
    if not _PYFFTW_AUTO_PROMOTE_LOGGED:
        _log.info(
            "lumenairy: promoting (%s, %s, %s, %d-thread) FFT plan from "
            "ESTIMATE to MEASURE (will pay one-shot planning cost; "
            "disable with lumenairy.set_fft_auto_promote(False))",
            shape_t, direction, dt.str, int(threads))
        # Mark logged so subsequent promotions at other keys don't
        # spam (one info per key family is enough to make the
        # behaviour visible).  Promoted keys are tracked individually
        # via entry['promoted'] = True so we don't promote twice.
        globals()['_PYFFTW_AUTO_PROMOTE_LOGGED'] = True
    new_entry = _build_plan_entry(direction, shape_t, dt, threads,
                                    'FFTW_MEASURE')
    new_entry['promoted'] = True
    return new_entry


def _get_or_make_plan(direction, shape, dtype, threads):
    """Return a cached in-place pyFFTW plan slot for ``direction``
    ('fwd' or 'inv') at the requested ``shape`` / ``dtype`` /
    ``threads``.

    Multi-slot LRU: each (direction, shape, dtype, threads) combination
    has its own resident entry with **two** aligned workspaces (a
    double-buffer ping-pong).  Each call alternates between slots, so
    the buffer returned to the previous caller is not clobbered by
    the current call -- the wrapper can return ``buf`` directly
    instead of paying for a ``.copy()``.

    Returns
    -------
    plan : pyfftw.FFTW
        The plan bound to ``buf`` for this call.  In-place transform.
    buf : ndarray
        Aligned workspace for this call.  Caller must
        ``np.copyto(buf, data)`` before executing ``plan()``.  The
        returned ``buf`` IS the result after ``plan()``; callers may
        return it directly to their own callers without ``.copy()``
        **as long as they accept that this buffer will be overwritten
        on the *second* subsequent call at the same key** (the first
        such call writes into the alternate slot and leaves this one
        alone).
    lock : threading.Lock
        Per-key lock; serialises concurrent execution at this key.
        Two threads at the same key still don't get to interleave on
        the same slot.

    Notes
    -----
    Callers that hold the returned buffer reference across more than
    one subsequent FFT call at the same key (e.g. caching the raw
    pyFFTW output for long-term reuse) must explicitly ``buf.copy()``
    or otherwise materialise an independent array before the third
    call.  See :func:`_fft2` / :func:`_ifft2` docstrings for the
    contract.

    Auto-promote: when ``_PYFFTW_AUTO_PROMOTE`` is True (default), an
    entry whose plan was built with ``FFTW_ESTIMATE`` is rebuilt with
    ``FFTW_MEASURE`` after its 5th call.  See
    :func:`set_fft_auto_promote`.
    """
    shape_t = tuple(int(s) for s in shape)
    dt = np.dtype(dtype)
    key = (str(direction), shape_t, dt.str, int(threads))

    with _PYFFTW_PLAN_LOCK:
        if key in _PYFFTW_PLAN_CACHE:
            entry = _PYFFTW_PLAN_CACHE[key]
            _PYFFTW_PLAN_CACHE.move_to_end(key)
            # Sanity check: buffer must still have the requested shape
            # / dtype.  Should never fail, but if it does (rare,
            # defensive) we drop and rebuild.
            buf0 = entry['bufs'][0]
            if buf0.shape == shape_t and buf0.dtype == dt:
                # Bump call count and check for auto-promote.  The
                # explicit user planner override
                # (set_pyfftw_planner(...)) is honored ahead of
                # auto-promote: if the user has globally selected a
                # planner stronger than (or equal to) MEASURE, we
                # don't promote ESTIMATE entries built before that
                # user override -- but since set_pyfftw_planner
                # clears the cache, the next call after the override
                # will rebuild under the new flag and we won't enter
                # this branch with a stale ESTIMATE entry anyway.
                entry['calls'] += 1
                if (_PYFFTW_AUTO_PROMOTE
                        and not entry['promoted']
                        and entry['flag'] == 'FFTW_ESTIMATE'
                        and _PYFFTW_PLAN_FLAGS[0] == 'FFTW_ESTIMATE'
                        and entry['calls'] >= _PYFFTW_AUTO_PROMOTE_THRESHOLD):
                    # Promote: build MEASURE plans, swap into the
                    # cache, and use the new entry for this call.
                    # Promotion is one-shot; subsequent hits at this
                    # key use the MEASURE plan without re-planning.
                    _ensure_pyfftw_loaded()
                    new_entry = _promote_entry_to_measure(
                        entry, direction, shape_t, dt, threads)
                    new_entry['calls'] = entry['calls']
                    _PYFFTW_PLAN_CACHE[key] = new_entry
                    entry = new_entry
                # Advance ping-pong slot under the cache lock so two
                # threads contending at the same key can't both grab
                # the same slot.
                slot = entry['idx']
                entry['idx'] = 1 - slot
                return entry['plans'][slot], entry['bufs'][slot], entry['lock']
            # Buffer mutated under us (rare; defensive); fall through
            # and rebuild.
            del _PYFFTW_PLAN_CACHE[key]

    # Fresh allocation + plan.  FFTW_ESTIMATE keeps planning cost
    # negligible; speedup vs scipy.fft comes from buffer reuse and
    # the threaded kernel.  pyfftw.FFTW.__call__ on the same buf is
    # not thread-safe -- the per-plan lock guards that.
    _ensure_pyfftw_loaded()
    new_entry = _build_plan_entry(direction, shape_t, dt, threads,
                                    _PYFFTW_PLAN_FLAGS[0])
    new_entry['calls'] = 1
    with _PYFFTW_PLAN_LOCK:
        _PYFFTW_PLAN_CACHE[key] = new_entry
        while len(_PYFFTW_PLAN_CACHE) > _PYFFTW_PLAN_CACHE_SIZE:
            _PYFFTW_PLAN_CACHE.popitem(last=False)
        slot = new_entry['idx']
        new_entry['idx'] = 1 - slot
    return new_entry['plans'][slot], new_entry['bufs'][slot], new_entry['lock']


# ----------------------------------------------------------------------------
# ASM transfer-function and frequency-grid caches (3.2.14)
# ----------------------------------------------------------------------------
# ``angular_spectrum_propagate`` builds three scalar/vector grids on
# every call: kx_sq[Nx], ky_sq[Ny], the band-limit masks bl_x[Nx] /
# bl_y[Ny], and finally the (Ny x Nx) transfer function H = exp(1j *
# kz * z) (band-limited).  The caches below short-circuit those
# rebuilds when the geometry repeats:
#
# * ``_FREQ_GRID_CACHE``   -- (kx_sq, ky_sq) keyed by (Ny, Nx, dy, dx,
#                              cdtype) so any two ASM calls on the
#                              same grid share the spatial-frequency
#                              vectors regardless of z / wavelength.
# * ``_BANDLIMIT_CACHE``   -- (bl_x, bl_y) keyed by the additional
#                              (lambda, |z|) pair (the fx_max, fy_max
#                              cutoffs depend on those).
# * ``_H_CACHE``           -- the full transfer function H, keyed by
#                              (Ny, Nx, dy, dx, lambda, z, bandlimit,
#                              cdtype).  Hits avoid the entire
#                              chunked construction loop -- typically
#                              30-50% of ASM call time on 2k+ grids.
#
# All three are bounded LRU dicts so peak memory stays controlled
# even if user code propagates at many distinct (z, lambda) pairs.

_FREQ_GRID_CACHE: 'OrderedDict[tuple, tuple]' = OrderedDict()
_FREQ_GRID_CACHE_SIZE = 16

_BANDLIMIT_CACHE: 'OrderedDict[tuple, tuple]' = OrderedDict()
_BANDLIMIT_CACHE_SIZE = 16

_H_CACHE: 'OrderedDict[tuple, np.ndarray]' = OrderedDict()
_H_CACHE_SIZE = 8

# 3.2.14.1: Per-entry size cap.  At N=32768 each H is 16 GB
# complex128; without this cap, an 8-entry cache can hold up to
# 128 GB of transfer functions and starve apply_real_lens of the
# few GB it needs for its own sag intermediates.  Above this
# threshold the cache transparently skips storage (lookups still
# work for any small entries already cached).
_H_CACHE_MAX_BYTES_PER_ENTRY = 2 * 1024 * 1024 * 1024   # 2 GB
# Total budget across all entries -- the cache evicts the oldest
# entries until total bytes fits this bound, regardless of count.
# Set generous enough that typical N <= 4k workflows fill up the
# count limit before the bytes limit; at larger N the bytes limit
# kicks in first.
_H_CACHE_MAX_TOTAL_BYTES = 8 * 1024 * 1024 * 1024       # 8 GB

_ASM_CACHE_LOCK = threading.Lock()


def _clear_local_asm_caches() -> None:
    """Drop the local propagation.py LRU caches under
    :data:`_ASM_CACHE_LOCK`.

    Called by the central cache-clearer registry (registered as
    ``'asm_local'`` in v4.16.0).  Drains:

    - ``_FREQ_GRID_CACHE``  : kx_sq / ky_sq frequency-grid pairs.
    - ``_BANDLIMIT_CACHE``  : ASM band-limit filter pairs.
    - ``_H_CACHE``          : ASM transfer function H per shape/wavelength/z.
    - ``_PYFFTW_PLAN_CACHE``: jit-built pyFFTW plans (v4.12.2 add).
    - ``_PYFFTW_BAD_SHAPES``: pyFFTW "skip this shape" memo (v4.12.2 add).

    Public API users should call :func:`clear_asm_caches` instead so
    every sibling cache is drained in the same operation.
    """
    with _ASM_CACHE_LOCK:
        _FREQ_GRID_CACHE.clear()
        _BANDLIMIT_CACHE.clear()
        _H_CACHE.clear()
        _PYFFTW_PLAN_CACHE.clear()
        _PYFFTW_BAD_SHAPES.clear()


# v4.16.0 (ROADMAP #15): register the local-ASM clearer with the
# central registry at module-import time.  ``clear_asm_caches`` now
# walks the registry rather than enumerating clear calls by hand.
#
# v4.16.1 (audit P1-NEW-F1-2 / C.5): late-binding lambda matching
# the canonical pattern used by the other 8 cache-owning modules.
# The registered entry re-resolves ``_clear_local_asm_caches`` from
# the module's current namespace at call time -- this preserves the
# pre-v4.16 ``mock.patch.object`` semantic where tests that monkey-
# patch the clear-function still observe their counter increment
# when ``clear_asm_caches`` walks the registry.  Pre-v4.16.1 the
# registration captured the function object directly (early-binding),
# which silently bypassed ``mock.patch.object`` -- a tester-visible
# inconsistency vs the other 8 caches.  The cost is one attribute
# lookup per cache per drain (negligible vs the cache-clear work).
#
# v5.1.0 (Agent C, propagation.py split): the registry entry must
# remain registered under the legacy module path so external code that
# monkey-patches ``lumenairy.propagators.propagation._clear_local_asm_caches``
# still observes the patch.  ``propagation`` re-exports
# ``_clear_local_asm_caches`` from this module, and the registry hook
# below resolves the attribute on the ``propagation`` module
# namespace so the legacy patch path stays intact.
from .._cache_registry import (
    register_cache_clearer as _register_cache_clearer,
    clear_all_registered_caches as _clear_all_registered_caches,
)
import sys as _sys


def _resolve_asm_clearer():
    """Resolve the current ``_clear_local_asm_caches`` callable.

    Looks up the attribute on the ``propagation`` module first
    (preserves the pre-v5.1.0 monkey-patch point) and falls back to
    this module's local function if propagation has not been imported
    yet.
    """
    prop_mod = _sys.modules.get(__name__.rsplit('.', 1)[0] + '.propagation')
    if prop_mod is not None and hasattr(prop_mod, '_clear_local_asm_caches'):
        return prop_mod._clear_local_asm_caches
    return _clear_local_asm_caches


_register_cache_clearer(
    'asm_local',
    lambda: _resolve_asm_clearer()(),
)


def clear_asm_caches() -> None:
    """Drop every propagator-adjacent cache the library can grow.

    v4.16.0 retires the lazy-import fan-out that this function used
    pre-v4.15 in favour of the central cache-clearer registry (see
    :mod:`lumenairy._cache_registry`).  Each cache-owning module
    registers its own clear function at import time via
    :func:`lumenairy.register_cache_clearer`, and this function simply
    walks the registry.

    The external contract is preserved bit-for-bit: every cache that
    was drained pre-v4.16 is still drained, and the same narrowed-
    except classes (``ImportError``, ``RuntimeError``,
    ``AttributeError``) are swallowed so a partial install does not
    strand the rest of the chain.

    Caches drained on a typical install:

    **Local (propagation.py) caches** -- registered as ``'asm_local'``:

    - ``_FREQ_GRID_CACHE``  : kx_sq / ky_sq frequency-grid pairs.
    - ``_BANDLIMIT_CACHE``  : ASM band-limit filter pairs.
    - ``_H_CACHE``          : ASM transfer function H per shape/wavelength/z.
    - ``_PYFFTW_PLAN_CACHE``: jit-built pyFFTW plans (v4.12.2 add).
    - ``_PYFFTW_BAD_SHAPES``: pyFFTW "skip this shape" memo (v4.12.2 add).

    **Sibling-module caches** -- each registered at the owning
    module's import time.  Cross-references to the underlying
    clear-function names are kept for grep-discoverability:

    - LG / HG mode-stack cache (``'lg_mode_stack'`` --
      :func:`lumenairy.propagators.asymptotic.clear_lg_mode_stack_cache`).
    - LG polynomial coefficient cache (``'lg_polynomial_items'`` --
      :func:`lumenairy.propagators.asymptotic.clear_lg_polynomial_cache`).
    - optimize/core wrapper-merit meshgrid cache
      (``'wrapper_merit_meshgrid'`` --
      :func:`lumenairy.optimize.core._clear_wrapper_merit_cache`).
    - Zernike basis cache (``'zernike_basis'`` --
      :func:`lumenairy.analysis.core.clear_zernike_basis_cache`).
    - Through-focus JAX scan cache (``'through_focus_scan_jax'`` --
      :func:`lumenairy.analysis.through_focus.clear_through_focus_scan_jax_cache`).
    - Propagate-through-system JAX cache (``'propagate_system_jax'``
      -- :func:`lumenairy.propagators.system.clear_propagate_system_jax_cache`).
    - Phase-retrieval GS / ER / HIO kernel caches
      (``'phase_retrieval_kernels'`` --
      :func:`lumenairy.analysis.phase_retrieval.clear_phase_retrieval_caches`).
    - Ray-trace JAX kernel cache (``'trace_jax'`` --
      :func:`lumenairy.raytrace.jax_trace.clear_trace_jax_cache`).

    Use :func:`lumenairy.list_registered_cache_clearers` to introspect
    the live registry contents (useful for the meta-pin
    ``test_all_known_caches_are_registered``).

    Historical notes:
        The original 3.2.14 perf-pass only cleared the first three
        local caches.  v4.12.2 extended this to drop the pyFFTW plan
        cache + bad-shape memo.  v4.14.1 chained the LG/HG mode-stack
        and wrapper-merit caches.  v4.14.2 (AUDIT_V4_14_1_2026_05_17
        P1-NEW-3 / Agent C) chained the five additional sibling caches.
        v4.14.3 chained the LG polynomial coefficient cache (8th
        sibling).  v4.16.0 retires the lazy-import fan-out entirely
        in favour of the central registry.
    """
    _clear_all_registered_caches()


def set_asm_cache_size(
    h_cache: Optional[int] = None,
    freq_cache: Optional[int] = None,
    bandlimit_cache: Optional[int] = None,
    h_max_bytes_per_entry: Optional[int] = None,
    h_max_total_bytes: Optional[int] = None,
) -> None:
    """Tune the per-cache LRU bounds.  Pass ``None`` to leave a
    bound unchanged.

    Parameters
    ----------
    h_cache, freq_cache, bandlimit_cache : int, optional
        Maximum number of entries.
    h_max_bytes_per_entry : int, optional
        Per-entry size cap for the H cache.  An H above this
        threshold is not stored.  Default 2 GB; at N=32768 each H
        is 16 GB so the cache transparently skips storage.
    h_max_total_bytes : int, optional
        Total bytes budget across all H entries.  Oldest entries
        are evicted until total bytes fit.  Default 8 GB.
    """
    global _H_CACHE_SIZE, _FREQ_GRID_CACHE_SIZE, _BANDLIMIT_CACHE_SIZE
    global _H_CACHE_MAX_BYTES_PER_ENTRY, _H_CACHE_MAX_TOTAL_BYTES
    with _ASM_CACHE_LOCK:
        if h_cache is not None:
            _H_CACHE_SIZE = max(1, int(h_cache))
        if h_max_bytes_per_entry is not None:
            _H_CACHE_MAX_BYTES_PER_ENTRY = int(h_max_bytes_per_entry)
        if h_max_total_bytes is not None:
            _H_CACHE_MAX_TOTAL_BYTES = int(h_max_total_bytes)
        # Apply count + bytes bounds together.  _entry_bytes handles
        # both single-array and tuple-bundle entries (the latter is
        # used by SAS to cache delta_H + H1 + H2 jointly).
        total = sum(_entry_bytes(v) for v in _H_CACHE.values())
        while (len(_H_CACHE) > _H_CACHE_SIZE
               or total > _H_CACHE_MAX_TOTAL_BYTES):
            try:
                _, dropped = _H_CACHE.popitem(last=False)
                total -= _entry_bytes(dropped)
            except KeyError:
                break
        if freq_cache is not None:
            _FREQ_GRID_CACHE_SIZE = max(1, int(freq_cache))
            while len(_FREQ_GRID_CACHE) > _FREQ_GRID_CACHE_SIZE:
                _FREQ_GRID_CACHE.popitem(last=False)
        if bandlimit_cache is not None:
            _BANDLIMIT_CACHE_SIZE = max(1, int(bandlimit_cache))
            while len(_BANDLIMIT_CACHE) > _BANDLIMIT_CACHE_SIZE:
                _BANDLIMIT_CACHE.popitem(last=False)


def _get_or_make_freq_grids(Ny, Nx, dy, dx, xp_is_numpy):
    """Cached (kx_sq, ky_sq) 1-D float64 grids for the current shape /
    pixel pitch.  CuPy callers skip the cache (device arrays don't
    survive a host-side dict)."""
    if not xp_is_numpy:
        kx_sq = (2 * np.pi * (cp.arange(Nx) - Nx / 2) / (Nx * dx)) ** 2
        ky_sq = (2 * np.pi * (cp.arange(Ny) - Ny / 2) / (Ny * dy)) ** 2
        # Note: matches the integer arithmetic of the legacy code,
        # which used `(arange(N) - N/2) * (1/(N*dx))` then squared.
        return kx_sq, ky_sq
    key = (int(Ny), int(Nx), float(dy), float(dx))
    with _ASM_CACHE_LOCK:
        if key in _FREQ_GRID_CACHE:
            _FREQ_GRID_CACHE.move_to_end(key)
            return _FREQ_GRID_CACHE[key]
    dfx = 1.0 / (Nx * dx)
    dfy = 1.0 / (Ny * dy)
    fx = (np.arange(Nx) - Nx / 2) * dfx
    fy = (np.arange(Ny) - Ny / 2) * dfy
    kx_sq = (2 * np.pi * fx) ** 2
    ky_sq = (2 * np.pi * fy) ** 2
    with _ASM_CACHE_LOCK:
        _FREQ_GRID_CACHE[key] = (kx_sq, ky_sq)
        while len(_FREQ_GRID_CACHE) > _FREQ_GRID_CACHE_SIZE:
            _FREQ_GRID_CACHE.popitem(last=False)
    return kx_sq, ky_sq


def _get_or_make_bandlimit(Ny, Nx, dy, dx, wavelength, abs_z, xp_is_numpy):
    """Cached 1-D band-limit masks.  Both axes share a single key."""
    if abs_z == 0:
        return None, None
    if not xp_is_numpy:
        Lx = Nx * dx
        Ly = Ny * dy
        fx_max = Lx / (2 * wavelength * abs_z)
        fy_max = Ly / (2 * wavelength * abs_z)
        fx = (cp.arange(Nx) - Nx / 2) / (Nx * dx)
        fy = (cp.arange(Ny) - Ny / 2) / (Ny * dy)
        return cp.abs(fx) < fx_max, cp.abs(fy) < fy_max
    key = (int(Ny), int(Nx), float(dy), float(dx),
           float(wavelength), float(abs_z))
    with _ASM_CACHE_LOCK:
        if key in _BANDLIMIT_CACHE:
            _BANDLIMIT_CACHE.move_to_end(key)
            return _BANDLIMIT_CACHE[key]
    Lx = Nx * dx
    Ly = Ny * dy
    fx_max = Lx / (2 * wavelength * abs_z)
    fy_max = Ly / (2 * wavelength * abs_z)
    fx = (np.arange(Nx) - Nx / 2) / (Nx * dx)
    fy = (np.arange(Ny) - Ny / 2) / (Ny * dy)
    bl_x = np.abs(fx) < fx_max
    bl_y = np.abs(fy) < fy_max
    with _ASM_CACHE_LOCK:
        _BANDLIMIT_CACHE[key] = (bl_x, bl_y)
        while len(_BANDLIMIT_CACHE) > _BANDLIMIT_CACHE_SIZE:
            _BANDLIMIT_CACHE.popitem(last=False)
    return bl_x, bl_y


def _h_cache_lookup(key):
    with _ASM_CACHE_LOCK:
        if key in _H_CACHE:
            _H_CACHE.move_to_end(key)
            return _H_CACHE[key]
    return None


def _entry_bytes(v):
    """Bytes accounting that handles both single arrays (the ASM H
    case) and tuples of arrays (the SAS case, where the cache entry
    bundles ``(delta_H, H1, H2_or_None)`` under one key).  Anything
    without ``.nbytes`` and not a tuple/list is treated as zero
    bytes -- safe since the eviction loop only undercounts, which
    can lead to a slightly larger working set but never to incorrect
    behaviour."""
    if isinstance(v, (tuple, list)):
        return sum(int(getattr(a, 'nbytes', 0))
                   for a in v if a is not None)
    return int(getattr(v, 'nbytes', 0))


def _h_cache_store(key, H):
    """Store H in the cache, honoring both count and byte bounds.

    Per-entry cap (`_H_CACHE_MAX_BYTES_PER_ENTRY`): if H is bigger
    than this, skip storage entirely.  At N=32768 a complex128 H is
    16 GB; caching it would starve every other allocator on the
    machine.  Lookups for that key will miss, the kernel rebuilds
    H, and the caller still works correctly -- just without the
    speedup (which the cache could never have meaningfully delivered
    on a memory-tight grid anyway).

    Total cap (`_H_CACHE_MAX_TOTAL_BYTES`): after insertion, evict
    oldest entries until total bytes fits.  Keeps the cache useful
    on intermediate grids (e.g. N=4096 at ~256 MB per H, holds 32+
    entries within the budget) without ballooning at large N.

    ``H`` may be a single ndarray (the ASM / RS / tilted-ASM /
    ASM-MFT case) or a tuple of arrays (the SAS case, which bundles
    ``delta_H``, ``H1``, and ``H2`` -- all sharing one geometry --
    into a single cache entry).  Sizing uses :func:`_entry_bytes`
    so tuple bundles are budgeted by their summed array bytes.
    """
    h_bytes = _entry_bytes(H)
    if h_bytes > _H_CACHE_MAX_BYTES_PER_ENTRY:
        return  # too big to cache; lookups will miss + rebuild
    with _ASM_CACHE_LOCK:
        _H_CACHE[key] = H
        # Drop oldest entries until count and total-bytes fit.
        total = sum(_entry_bytes(v) for v in _H_CACHE.values())
        while (len(_H_CACHE) > _H_CACHE_SIZE
               or total > _H_CACHE_MAX_TOTAL_BYTES):
            try:
                _, dropped = _H_CACHE.popitem(last=False)
                total -= _entry_bytes(dropped)
            except KeyError:
                break


def set_fft_fallback(enabled: bool) -> None:
    """Enable / disable the automatic pyFFTW -> scipy.fft fallback on
    allocation or runtime errors."""
    global PYFFTW_FALLBACK_ON_ERROR
    PYFFTW_FALLBACK_ON_ERROR = bool(enabled)


def set_fft_threads(n: Optional[int]) -> None:
    """Override the thread count used by the pyFFTW / scipy.fft path.

    Pass a positive int to pin to that many threads (useful inside a
    process pool where you don't want each worker to spin up its own
    thread farm -- ``set_fft_threads(1)`` gives each worker a
    single-threaded FFT and avoids oversubscription).  Pass ``0`` or
    ``None`` to restore the affinity-aware default from
    :func:`lumenairy._backends.available_cpus`.
    """
    global FFTW_THREADS, SCIPY_FFT_WORKERS
    if n is None or n == 0:
        FFTW_THREADS = _available_cpus()
        SCIPY_FFT_WORKERS = -1
    else:
        FFTW_THREADS = max(1, int(n))
        SCIPY_FFT_WORKERS = max(1, int(n))


def get_fft_threads() -> int:
    """Return the current pyFFTW thread count.

    Mirrors :func:`set_fft_threads`.  Returns the effective integer
    actually being passed to pyFFTW (affinity-aware default or the
    user override).  Companion accessor introduced in 4.8.1 alongside
    the :func:`lumenairy.lumenairy_context` manager, which needs it
    to snapshot/restore state.
    """
    return int(FFTW_THREADS)


def get_pyfftw_planner() -> str:
    """Return the current pyFFTW planner flag.

    Mirrors :func:`set_pyfftw_planner`.  Returns one of
    ``'FFTW_ESTIMATE'`` (the import-time default), ``'FFTW_MEASURE'``,
    ``'FFTW_PATIENT'``, ``'FFTW_EXHAUSTIVE'``.  Companion accessor
    introduced in 4.8.1 alongside :func:`lumenairy.lumenairy_context`.
    """
    return _PYFFTW_PLAN_FLAGS[0]


def set_fft_auto_promote(enabled: bool) -> None:
    """Enable or disable automatic ESTIMATE -> MEASURE plan promotion (4.12).

    When enabled (the default), the pyFFTW plan cache tracks the
    call count for every ``(direction, shape, dtype, threads)`` key
    that was built under :data:`_PYFFTW_PLAN_FLAGS` =
    ``'FFTW_ESTIMATE'`` (i.e. the library default).  After the 5th
    call at the same key, the plan is rebuilt with ``FFTW_MEASURE``
    so that the steady-state hot path benefits from FFTW's more
    aggressive algorithm choice (~1.3-2x faster execution on common
    optics shapes, in exchange for a one-shot ~100-1000 ms planning
    cost at the promotion call).

    Promotion is one-shot per key and survives until the next
    :func:`reset_fft_backend` / :func:`set_pyfftw_planner` /
    :func:`set_fft_plan_cache_size` call that clears the cache.
    Eviction from the LRU restarts the call counter from zero, so a
    rarely-used key won't keep paying the MEASURE cost on rebuild.

    Pass ``False`` for bit-for-bit reproducibility across runs: the
    FFTW MEASURE planner can pick a different algorithm than
    ESTIMATE, and while the change is at the float-LSB level it can
    perturb the final ULP of cumulative-sum outputs.  Disable also
    if you have an explicit :func:`set_pyfftw_planner` strategy
    (e.g. ``'FFTW_PATIENT'``) and want to control the planner
    yourself.

    Companion to :func:`get_fft_auto_promote`.  No-op when pyFFTW
    is not installed.
    """
    global _PYFFTW_AUTO_PROMOTE
    _PYFFTW_AUTO_PROMOTE = bool(enabled)


def get_fft_auto_promote() -> bool:
    """Return whether auto-promote of ESTIMATE -> MEASURE is on (4.12).

    Mirrors :func:`set_fft_auto_promote`.  Defaults to ``True`` at
    import time.
    """
    return bool(_PYFFTW_AUTO_PROMOTE)


def get_asm_cache_size() -> Dict[str, int]:
    """Return the current ASM-cache bounds as a dict.

    Returns the four knobs that :func:`set_asm_cache_size` accepts:

    - ``h_cache`` : max H-cache entries (default 8)
    - ``h_max_bytes_per_entry`` : per-entry byte cap (default 2 GB)
    - ``h_max_total_bytes`` : total bytes across all H entries (default 8 GB)
    - ``freq_cache`` / ``bandlimit_cache`` : entry counts on the two
      ancillary caches

    Snapshot-friendly: ``set_asm_cache_size(**get_asm_cache_size())``
    is a round-trip.  Introduced in 4.8.1 to support
    :func:`lumenairy.lumenairy_context`.
    """
    return {
        'h_cache': int(_H_CACHE_SIZE),
        'h_max_bytes_per_entry': int(_H_CACHE_MAX_BYTES_PER_ENTRY),
        'h_max_total_bytes': int(_H_CACHE_MAX_TOTAL_BYTES),
        'freq_cache': int(_FREQ_GRID_CACHE_SIZE),
        'bandlimit_cache': int(_BANDLIMIT_CACHE_SIZE),
    }


# ============================================================================
# FFT helper functions
# ============================================================================

def _scipy_or_numpy_fft2(x):
    """Used by the fallback path (and by small-grid calls)."""
    if USE_SCIPY_FFT and SCIPY_FFT_AVAILABLE:
        return _scipy_fft.fft2(x, workers=SCIPY_FFT_WORKERS)
    return np.fft.fft2(x)


def _scipy_or_numpy_ifft2(x):
    if USE_SCIPY_FFT and SCIPY_FFT_AVAILABLE:
        return _scipy_fft.ifft2(x, workers=SCIPY_FFT_WORKERS)
    return np.fft.ifft2(x)


def _handle_pyfftw_failure(x, op_name, exc):
    """Record a bad shape and emit a one-time warning.

    Called from ``_fft2`` / ``_ifft2`` when a pyFFTW call raises (most
    commonly ``MemoryError`` from an inability to allocate the
    aligned contiguous plan buffer on large grids with tight RAM).
    """
    shape = tuple(x.shape)
    was_new = shape not in _PYFFTW_BAD_SHAPES
    _PYFFTW_BAD_SHAPES.add(shape)
    if was_new:
        import sys, warnings
        # Flush pyFFTW's plan cache so the failed buffers are freed
        # and subsequent SMALLER calls have room.  We keep caching
        # enabled so that unaffected shapes still get plan reuse.
        try:
            pyfftw.interfaces.cache.disable()
            pyfftw.interfaces.cache.enable()
            pyfftw.interfaces.cache.set_keepalive_time(30.0)
        except (RuntimeError, ValueError, AttributeError, MemoryError) as _cache_exc:
            warnings.warn(
                f"pyfftw cache reset after FFT failure threw "
                f"{type(_cache_exc).__name__}: {_cache_exc}.  Continuing "
                f"with the existing cache; if subsequent shapes also "
                f"fail consider calling reset_fft_backend() explicitly.",
                RuntimeWarning, stacklevel=2)
        warnings.warn(
            f'pyFFTW {op_name} failed on shape {shape}: '
            f'{type(exc).__name__}: {exc}.  Falling back to '
            f'scipy.fft for this shape.  (Likely cause: aligned '
            f'contiguous buffer allocation failed under memory '
            f'pressure.)  Call '
            f'lumenairy.propagation.reset_fft_backend() '
            f'after the large allocation is freed to re-enable '
            f'pyFFTW for future calls at this shape.',
            RuntimeWarning, stacklevel=3)


def _fft2(x):
    """
    2-D FFT dispatcher.

    Priority order:
        1. CuPy (if input is a CuPy array)
        2. pyFFTW via the multi-slot cached plan (if USE_PYFFTW, array
           large enough, shape not in the bad-shape blacklist).  Hits
           :func:`_get_or_make_plan` which reuses an existing plan
           when ``(shape, dtype, threads)`` matches exactly, otherwise
           reallocates.
        3. SciPy FFT (if USE_SCIPY_FFT)
        4. NumPy FFT (fallback)

    pyFFTW calls are wrapped in try/except so a failed aligned-buffer
    allocation (common on very large grids under memory pressure)
    falls back to scipy.fft rather than propagating the error.  See
    :data:`PYFFTW_FALLBACK_ON_ERROR`.

    Returned-buffer ownership (4.12 double-buffer)
    ----------------------------------------------
    On the pyFFTW path the returned array IS one of the two ping-pong
    workspace buffers held inside the plan cache, not a fresh
    allocation.  This skips the ~256 MB-1 GB per-call copy at 4k-8k
    grids in exchange for one constraint on callers:

    * **The returned buffer stays stable across exactly ONE
      subsequent FFT call at the same (direction, shape, dtype,
      threads) key**.  The next call writes into the alternate slot
      and leaves yours untouched.  The call *after* that recycles
      your slot.

    * Almost every in-library caller consumes the result immediately
      (multiply, IFFT, or fftshift) before issuing another FFT call
      at the same key, so the contract is satisfied for free.

    * Callers that need a longer-lived independent copy (e.g. caching
      the raw FFT output for reuse across many subsequent FFTs, as
      :func:`rayleigh_sommerfeld_propagate` does with its kernel
      transfer function) must explicitly ``.copy()`` the result
      themselves.  The scipy / numpy fallback path always returns a
      fresh array, so a defensive ``.copy()`` is a no-op on
      non-pyFFTW backends.
    """
    if _is_cupy_array(x):
        return cp.fft.fft2(x)
    shape = tuple(x.shape)
    if (USE_PYFFTW and PYFFTW_AVAILABLE
            and x.shape[0] >= FFTW_MIN_SIZE
            and shape not in _PYFFTW_BAD_SHAPES):
        threads = FFTW_THREADS if FFTW_THREADS > 0 else 1
        try:
            plan, buf, lock = _get_or_make_plan('fwd', shape, x.dtype, threads)
            # Hold the lock across copy-in and execute so a
            # concurrent ``_fft2`` / ``_ifft2`` caller at the same
            # key can't race on this slot's buffer.  The ping-pong
            # slot index has already been advanced under the cache
            # lock, so the next caller at this key receives the
            # ALTERNATE buffer; this slot stays valid until the call
            # after next.  See module docstring for the contract.
            with lock:
                np.copyto(buf, x, casting='no')
                plan()
                return buf
        except (RuntimeError, MemoryError, ValueError, TypeError,
                AttributeError) as e:
            # pyFFTW failure modes: RuntimeError (planner internal
            # error), MemoryError (aligned-buffer allocation failed
            # under pressure), ValueError (shape/stride/dtype
            # rejected), TypeError (passing a non-ndarray), and
            # AttributeError (pyfftw.FFTW callable lost across a
            # plan-cache reset under threads).  Anything else is a
            # real upstream bug we want to propagate.
            if not PYFFTW_FALLBACK_ON_ERROR:
                raise
            _handle_pyfftw_failure(x, 'fft2', e)
            # fall through to scipy/numpy
    return _scipy_or_numpy_fft2(x)


def _ifft2(x):
    """
    2-D inverse FFT dispatcher.

    Same priority as :func:`_fft2`.  Uses a separate cache slot for
    the inverse direction; ``pyfftw.FFTW`` with
    ``direction='FFTW_BACKWARD'`` normalises by ``N`` by default,
    matching ``numpy.fft.ifft2`` semantics.

    Returned-buffer ownership: same contract as :func:`_fft2` -- the
    pyFFTW path returns a reference to one of two ping-pong
    workspace buffers, stable across exactly one subsequent
    ``_ifft2`` call at the same key.  Callers caching the result for
    multiple subsequent FFT calls must ``.copy()``.
    """
    if _is_cupy_array(x):
        return cp.fft.ifft2(x)
    shape = tuple(x.shape)
    if (USE_PYFFTW and PYFFTW_AVAILABLE
            and x.shape[0] >= FFTW_MIN_SIZE
            and shape not in _PYFFTW_BAD_SHAPES):
        threads = FFTW_THREADS if FFTW_THREADS > 0 else 1
        try:
            plan, buf, lock = _get_or_make_plan('inv', shape, x.dtype, threads)
            with lock:
                np.copyto(buf, x, casting='no')
                plan()
                return buf
        except (RuntimeError, MemoryError, ValueError, TypeError,
                AttributeError) as e:
            # Same pyFFTW failure spectrum as ``_fft2``; see comment
            # there for the rationale.
            if not PYFFTW_FALLBACK_ON_ERROR:
                raise
            _handle_pyfftw_failure(x, 'ifft2', e)
    return _scipy_or_numpy_ifft2(x)


def _fft2_nd(x):
    """N-D forward 2-D FFT over the last two axes.  Uses pyFFTW for
    contiguous (B, Ny, Nx) shapes when large enough; falls back to
    scipy.fft / numpy for everything else.

    Returned-buffer ownership follows the same double-buffer
    contract as :func:`_fft2`: the pyFFTW path returns a reference
    to one of two ping-pong workspace buffers.  Callers caching the
    result across multiple subsequent FFT calls at this shape must
    ``.copy()`` defensively."""
    if _is_cupy_array(x):
        return cp.fft.fft2(x, axes=(-2, -1))
    shape = tuple(x.shape)
    if (USE_PYFFTW and PYFFTW_AVAILABLE and len(shape) >= 2
            and shape[-2] >= FFTW_MIN_SIZE
            and shape not in _PYFFTW_BAD_SHAPES):
        threads = FFTW_THREADS if FFTW_THREADS > 0 else 1
        try:
            plan, buf, lock = _get_or_make_plan(
                'fwd', shape, x.dtype, threads)
            with lock:
                np.copyto(buf, x, casting='no')
                plan()
                return buf
        except (RuntimeError, MemoryError, ValueError, TypeError,
                AttributeError) as e:
            if not PYFFTW_FALLBACK_ON_ERROR:
                raise
            _handle_pyfftw_failure(x, 'fft2_nd', e)
    if USE_SCIPY_FFT and SCIPY_FFT_AVAILABLE:
        return _scipy_fft.fft2(x, axes=(-2, -1), workers=SCIPY_FFT_WORKERS)
    return np.fft.fft2(x, axes=(-2, -1))


def _ifft2_nd(x):
    """N-D inverse 2-D FFT.  Same priority chain as :func:`_fft2_nd`
    and same double-buffer ownership contract: pyFFTW returns a
    reference into a ping-pong slot; callers caching across
    multiple subsequent calls must ``.copy()``."""
    if _is_cupy_array(x):
        return cp.fft.ifft2(x, axes=(-2, -1))
    shape = tuple(x.shape)
    if (USE_PYFFTW and PYFFTW_AVAILABLE and len(shape) >= 2
            and shape[-2] >= FFTW_MIN_SIZE
            and shape not in _PYFFTW_BAD_SHAPES):
        threads = FFTW_THREADS if FFTW_THREADS > 0 else 1
        try:
            plan, buf, lock = _get_or_make_plan(
                'inv', shape, x.dtype, threads)
            with lock:
                np.copyto(buf, x, casting='no')
                plan()
                return buf
        except (RuntimeError, MemoryError, ValueError, TypeError,
                AttributeError) as e:
            if not PYFFTW_FALLBACK_ON_ERROR:
                raise
            _handle_pyfftw_failure(x, 'ifft2_nd', e)
    if USE_SCIPY_FFT and SCIPY_FFT_AVAILABLE:
        return _scipy_fft.ifft2(x, axes=(-2, -1), workers=SCIPY_FFT_WORKERS)
    return np.fft.ifft2(x, axes=(-2, -1))


# ============================================================================
# Input validation
# ============================================================================

def _validate_propagator_inputs(E_in, z, wavelength, dx, dy=None, *,
                                fn_name='propagator'):
    """Validate the standard ``(E, z, wavelength, dx, dy)`` signature.

    Raises a ``ValueError`` with the offending value and the offending
    parameter quoted, rather than letting downstream FFTs / divisions
    fail with cryptic ``ZeroDivisionError`` / ``ValueError`` / silent
    "all zeros" outputs.

    Checks performed:

    * ``E_in`` is a 2-D array of nonzero size (``Ny >= 4`` and
      ``Nx >= 4`` -- below that the band-limit and Nyquist
      diagnostics are meaningless).
    * ``z`` is a real, finite scalar (any sign; ``z = 0`` is allowed
      and returns the input unchanged via the propagator path).
    * ``wavelength`` is a real, finite, **positive** scalar with a
      magnitude that *looks* like metres (i.e. < 1 mm).  Values
      above 1 mm raise -- almost always the caller forgot the
      micron-to-metre conversion.
    * ``dx`` (and ``dy`` if given) is a real, finite, **positive**
      scalar with a magnitude that *looks* like a metre-scale
      pixel pitch.  ``dx > 1 mm`` raises for the same reason.

    Parameters
    ----------
    E_in : array_like
    z, wavelength, dx, dy : float
    fn_name : str
        Name of the calling propagator, used in the error message.

    Raises
    ------
    ValueError
        If any of the above invariants fails.

    Notes
    -----
    This helper is intentionally *fast* -- it does
    ``np.asarray(E_in).shape`` and a handful of scalar comparisons,
    so the overhead is negligible compared to even a 256x256 FFT.
    """
    # E_in shape
    try:
        Ny, Nx = E_in.shape
    except AttributeError:
        raise ValueError(
            f'{fn_name}: E_in must be a 2-D array, got {type(E_in).__name__}.'
        )
    except ValueError:
        raise ValueError(
            f'{fn_name}: E_in must be a 2-D array of shape (Ny, Nx); '
            f'got shape {tuple(E_in.shape)!r}.'
        )
    if Ny < 4 or Nx < 4:
        raise ValueError(
            f'{fn_name}: E_in is too small ({Ny}x{Nx}); each axis must '
            f'be >= 4 samples for the propagator to be meaningful.'
        )
    # z
    z_f = float(z)
    if not np.isfinite(z_f):
        raise ValueError(
            f'{fn_name}: z must be a finite real scalar; got {z!r}.'
        )
    # wavelength
    lam_f = float(wavelength)
    if not np.isfinite(lam_f):
        raise ValueError(
            f'{fn_name}: wavelength must be a finite real scalar; '
            f'got {wavelength!r}.'
        )
    if lam_f <= 0.0:
        raise ValueError(
            f'{fn_name}: wavelength must be positive [m]; got '
            f'{lam_f!r}. (Did you pass a negative value, or units of '
            f'microns / nm? LumenAiry uses metres: 1310 nm -> 1.31e-6.)'
        )
    if lam_f > 1e-3:
        raise ValueError(
            f'{fn_name}: wavelength = {lam_f!r} m looks suspicious '
            f'(> 1 mm).  LumenAiry uses SI metres, not microns or '
            f'nanometres.  1310 nm -> 1.31e-6; 1.55 um -> 1.55e-6.'
        )
    # dx
    dx_f = float(dx)
    if not np.isfinite(dx_f):
        raise ValueError(
            f'{fn_name}: dx must be a finite real scalar; got {dx!r}.'
        )
    if dx_f <= 0.0:
        raise ValueError(
            f'{fn_name}: dx must be positive [m]; got {dx_f!r}. '
            f'(LumenAiry uses metres; for a 2 um pixel pitch pass '
            f'dx = 2e-6, not 2.)'
        )
    # 4.9 fix (audit #4.3): the > 1 mm guard was over-strict.  Large
    # telescope pupils (Hale-class ground apertures, JWST-scale segment
    # arrays) are legitimately sampled at the mm scale.  Loosened to
    # > 100 mm (above which a unit-error is far more likely than a
    # genuine telescope-class problem); the legacy < 1 mm sanity floor
    # downgrades to a one-time RuntimeWarning so users who really do
    # have mm-scale pixel pitch see the warning once and proceed.
    if dx_f > 1e-1:
        raise ValueError(
            f'{fn_name}: dx = {dx_f!r} m looks suspicious (> 100 mm). '
            f'LumenAiry uses SI metres for the grid pitch; for 2 um '
            f'pass dx = 2e-6, not dx = 2.'
        )
    elif dx_f > 1e-3:
        import warnings
        warnings.warn(
            f"{fn_name}: dx = {dx_f!r} m is unusually large (> 1 mm) "
            f"but allowed.  This is valid for large-aperture telescope "
            f"pupils sampled at the mm scale; if you meant microns, "
            f"pass dx = 2e-6 (not dx = 2e-3).",
            RuntimeWarning, stacklevel=3,
        )
    # dy (optional)
    if dy is not None:
        dy_f = float(dy)
        if not np.isfinite(dy_f):
            raise ValueError(
                f'{fn_name}: dy must be a finite real scalar; got {dy!r}.'
            )
        if dy_f <= 0.0:
            raise ValueError(
                f'{fn_name}: dy must be positive [m]; got {dy_f!r}.'
            )
        if dy_f > 1e-1:
            raise ValueError(
                f'{fn_name}: dy = {dy_f!r} m looks suspicious (> 100 mm); '
                f'expected SI metres.'
            )


__all__ = [
    # Backend flags + loaders
    # v5.1.0 (Wave-4 integration / V9 walker symmetry): module handles
    # ``cp`` / ``pyfftw`` and internal config flags (FFTW_MIN_SIZE /
    # FFTW_THREADS / PYFFTW_FALLBACK_ON_ERROR / SCIPY_FFT_AVAILABLE /
    # SCIPY_FFT_WORKERS / USE_PYFFTW / USE_SCIPY_FFT) are intentionally
    # NOT in __all__ -- they're module-attribute-accessible for power
    # users (``lumenairy.propagators.propagation.USE_PYFFTW = True``)
    # but not part of the public top-level API.  CUPY_AVAILABLE /
    # PYFFTW_AVAILABLE stay public because they're documented as
    # capability-probe flags in the README.
    'CUPY_AVAILABLE', 'PYFFTW_AVAILABLE',
    '_ensure_cupy_loaded', '_ensure_pyfftw_loaded', '_is_cupy_array',
    # FFT backend config (setters only -- the underlying globals stay
    # module-attribute-accessible but out of __all__)
    'set_fft_fallback', 'set_fft_threads', 'get_fft_threads',
    'set_pyfftw_planner', 'get_pyfftw_planner',
    'set_fft_plan_cache_size', 'warmup_fft_plans',
    'set_fft_auto_promote', 'get_fft_auto_promote',
    'reset_fft_backend',
    # Default-config knobs
    'DEFAULT_COMPLEX_DTYPE', 'DEFAULT_REAL_DTYPE',
    'DEFAULT_WAVE_PROPAGATOR', 'DEFAULT_DY',
    'set_default_complex_dtype', 'get_default_complex_dtype',
    'set_default_real_dtype', 'get_default_real_dtype',
    'set_default_wave_propagator', 'get_default_wave_propagator',
    'set_default_dy', 'get_default_dy',
    # JAX dtype resolvers
    '_resolve_jax_complex_dtype', '_resolve_jax_real_dtype',
    # ASM caches + helpers
    '_FREQ_GRID_CACHE', '_BANDLIMIT_CACHE', '_H_CACHE',
    '_ASM_CACHE_LOCK',
    'clear_asm_caches', '_clear_local_asm_caches',
    'set_asm_cache_size', 'get_asm_cache_size',
    '_get_or_make_freq_grids', '_get_or_make_bandlimit',
    '_h_cache_lookup', '_h_cache_store', '_entry_bytes',
    # pyFFTW plan-cache internals
    '_PYFFTW_PLAN_CACHE', '_PYFFTW_PLAN_LOCK',
    '_PYFFTW_BAD_SHAPES',
    '_get_or_make_plan', '_build_plan_entry', '_promote_entry_to_measure',
    # FFT dispatchers
    '_fft2', '_ifft2', '_fft2_nd', '_ifft2_nd',
    '_scipy_or_numpy_fft2', '_scipy_or_numpy_ifft2',
    '_handle_pyfftw_failure',
    # Validation
    '_validate_propagator_inputs',
    # Back-compat latches
    '_DEFAULT_WAVE_PROPAGATOR_NO_CONSUMER_WARNED',
    '_DEFAULT_DY_NO_CONSUMER_WARNED',
]
