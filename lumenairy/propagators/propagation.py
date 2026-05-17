"""
Core Optical Propagation Module
================================

Provides exact and approximate free-space propagation of coherent optical
fields on discrete 2-D grids.

Methods implemented:
    - Angular Spectrum Method (ASM) -- exact, band-limited
    - Tilted / off-axis ASM        -- ASM with carrier frequency removal
    - Single-FFT Fresnel           -- paraxial, changes output grid spacing
    - Fraunhofer (far-field)       -- single FFT, large-z limit
    - Rayleigh-Sommerfeld          -- convolution with Green's function

Convention
----------
Time dependence:  exp(-i*omega*t)  throughout.
Units:            SI meters for all spatial quantities.

Return-type contract
--------------------
Propagators that **preserve the grid spacing** (ASM, tilted ASM,
Rayleigh-Sommerfeld) return the field as a bare ``ndarray``::

    E_out = angular_spectrum_propagate(E, z, lam, dx)
    E_out = rayleigh_sommerfeld_propagate(E, z, lam, dx)

Propagators that **change the grid spacing** (Fresnel, Fraunhofer)
return a 3-tuple ``(E_out, dx_out, dy_out)`` so callers can resample
or update their pixel pitch::

    E_out, dx_out, dy_out = fresnel_propagate(E, z, lam, dx)
    E_out, dx_out, dy_out = fraunhofer_propagate(E, z, lam, dx)

This split is intentional and stable -- code that treats the bare
return as iterable will fail loudly rather than silently miscompute.

Backends
--------
    NumPy   -- CPU, always available (default)
    CuPy    -- GPU, auto-detected at import time
    pyFFTW  -- multi-threaded CPU FFT, opt-in via USE_PYFFTW flag

Author:  Andrew Traverso
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple, Union

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
        except Exception as _exc:
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


def clear_asm_caches() -> None:
    """Drop the propagator-adjacent caches (v4.12.2: extended).

    Clears:

    - ``_FREQ_GRID_CACHE``  : kx_sq / ky_sq frequency-grid pairs.
    - ``_BANDLIMIT_CACHE``  : ASM band-limit filter pairs.
    - ``_H_CACHE``          : ASM transfer function H per shape/wavelength/z.
    - ``_PYFFTW_PLAN_CACHE``: jit-built pyFFTW plans (v4.12.2 add).
    - ``_PYFFTW_BAD_SHAPES``: pyFFTW "skip this shape" memo (v4.12.2 add).

    The original 3.2.14 perf-pass only cleared the first three.  v4.12.2
    extends the function to ALSO drop the pyFFTW plan cache + bad-shape
    memo so a single call leaves the propagation-layer state completely
    pristine.  Companion ``clear_*`` helpers exist in sibling modules
    for the Zernike basis cache, LG-polynomial cache, JAX trace cache,
    JAX propagate-through-system cache, and phase-retrieval kernel
    caches; :func:`lumenairy.lumenairy_context` with
    ``clear_caches_on_exit=True`` invokes all of them.
    """
    with _ASM_CACHE_LOCK:
        _FREQ_GRID_CACHE.clear()
        _BANDLIMIT_CACHE.clear()
        _H_CACHE.clear()
        _PYFFTW_PLAN_CACHE.clear()
        _PYFFTW_BAD_SHAPES.clear()


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
        except Exception as _cache_exc:
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
        except Exception as e:
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
        except Exception as e:
            if not PYFFTW_FALLBACK_ON_ERROR:
                raise
            _handle_pyfftw_failure(x, 'ifft2', e)
    return _scipy_or_numpy_ifft2(x)


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


# ============================================================================
# Angular Spectrum Method (ASM) -- exact, band-limited
# ============================================================================

def angular_spectrum_propagate(
    E_in: np.ndarray,
    z: float,
    wavelength: float,
    dx: float,
    dy: Optional[float] = None,
    bandlimit: bool = True,
    return_transfer_function: bool = False,
    use_gpu: bool = False,
    verbose: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Propagate an optical field using the Angular Spectrum Method (ASM).

    This function propagates a 2-D complex electric field through free space
    using the exact transfer function (no paraxial approximation).

    Parameters
    ----------
    E_in : ndarray (complex)
        Input electric field, shape (Ny, Nx).  Can be a NumPy or CuPy array.

    z : float
        Propagation distance in meters.
        Positive z = forward propagation (away from source).
        Negative z = backward propagation (toward source).

    wavelength : float
        Optical wavelength in meters (e.g. 1.31e-6 for 1310 nm).

    dx : float
        Grid spacing in x-direction in meters (e.g. 1e-6 for 1 um).

    dy : float, optional
        Grid spacing in y-direction in meters.  If None, assumes dy = dx.

    bandlimit : bool, default True
        If True, applies band-limiting to suppress Fresnel aliasing.
        The band-limit cutoff per axis is:  f_max = L / (2 * lambda * |z|).
        Recommended for large propagation distances.

    return_transfer_function : bool, default False
        If True, also returns the transfer function H.

    use_gpu : bool, default False
        If True and CuPy is available, performs computation on GPU.
        If *E_in* is already a CuPy array, GPU is used automatically.

    verbose : bool, default False
        If True, prints diagnostic information.

    Returns
    -------
    E_out : ndarray (complex)
        Propagated electric field, same shape and array type as *E_in*.

    H : ndarray (complex), optional
        Transfer function (only returned when *return_transfer_function=True*).

    Notes
    -----
    Sampling requirements for accurate results:

    1. ``dx < lambda / 2`` -- Nyquist for propagating waves.
    2. ``L > 2 * lambda * z / d_min`` -- avoids Fresnel aliasing, where
       L = N * dx is the grid extent and d_min is the smallest feature size
       to be resolved.

    Memory: approximately 3x the size of the input array (E_in, E_fft, H).

    Examples
    --------
    >>> import numpy as np
    >>> from lumenairy.propagation import angular_spectrum_propagate
    >>>
    >>> N = 512
    >>> dx = 1e-6                    # 1 um grid spacing
    >>> wavelength = 1.31e-6         # 1310 nm
    >>> x = (np.arange(N) - N/2) * dx
    >>> X, Y = np.meshgrid(x, x)
    >>> sigma = 10e-6                # 10 um beam waist
    >>> E_in = np.exp(-(X**2 + Y**2) / (2 * sigma**2)).astype(complex)
    >>>
    >>> E_out = angular_spectrum_propagate(E_in, z=1e-3,
    ...                                    wavelength=wavelength, dx=dx)
    >>> print(f"Input power:  {np.sum(np.abs(E_in)**2):.4f}")
    >>> print(f"Output power: {np.sum(np.abs(E_out)**2):.4f}")

    References
    ----------
    [1] Goodman, J.W. "Introduction to Fourier Optics" (3rd ed.), Ch. 3-4.
    [2] Matsushima, K. and Shimobaba, T. (2009). "Band-limited angular
        spectrum method for numerical simulation of free-space propagation
        in far and near fields." Opt. Express 17(22): 19662-19673.
    """
    _validate_propagator_inputs(E_in, z, wavelength, dx, dy,
                                fn_name='angular_spectrum_propagate')

    # -- array library selection (NumPy / CuPy / JAX) ----------------------
    # JAX arrays bypass the chunked H construction and the host cache;
    # they take a one-shot all-NxN H and stay in the input backend.
    from ..backend import is_jax_array, JAX_AVAILABLE
    is_jax = is_jax_array(E_in)
    if is_jax:
        import jax.numpy as _jnp
        xp = _jnp
    elif CUPY_AVAILABLE and (use_gpu or _is_cupy_array(E_in)):
        xp = cp
        if not _is_cupy_array(E_in):
            E_in = cp.asarray(E_in)
    else:
        xp = np
        if _is_cupy_array(E_in):
            E_in = E_in.get()  # CuPy -> NumPy when GPU not requested

    Ny, Nx = E_in.shape

    if dy is None:
        dy = dx

    # -- wave parameters -----------------------------------------------------
    k = 2 * np.pi / wavelength

    # Target complex dtype for the transfer function and the output.
    # Inferred from E_in so the caller controls precision by the dtype of
    # the field they pass in.  Non-complex input (e.g. float arrays used
    # in examples) falls back to DEFAULT_COMPLEX_DTYPE.
    if xp.iscomplexobj(E_in):
        target_cdtype = E_in.dtype
    else:
        target_cdtype = np.dtype(DEFAULT_COMPLEX_DTYPE)
    target_fdtype = np.float32 if target_cdtype == np.complex64 else np.float64

    # ── 3.2.14 H cache ───────────────────────────────────────────────
    # Geometry signature.  Hits return the previously-built H without
    # re-running the chunked kernel construction (~30-50% of total
    # ASM time on 2k+ grids).  CuPy device arrays and JAX traced
    # arrays are kept out of the cache (host-side dict can't safely
    # retain device pointers / traced objects).
    h_key = None
    H = None
    if xp is np:
        # 4.10: add 'ASM' tag string to the cache key so plain-ASM
        # entries are guaranteed disjoint from ASM_TILTED / ASM_MFT /
        # RS / SAS even if those keys ever evolve to the same tuple
        # length.  Defensive future-proofing.
        h_key = (int(Ny), int(Nx), float(dy), float(dx),
                 float(wavelength), float(z), bool(bandlimit),
                 np.dtype(target_cdtype).str, 'ASM')
        H = _h_cache_lookup(h_key)

    if H is None and is_jax:
        # JAX path: build H in one shot (no chunking, no in-place
        # writes; jax.numpy is functional / immutable).
        fx = (xp.arange(Nx, dtype=xp.float64) - Nx / 2) / (Nx * dx)
        fy = (xp.arange(Ny, dtype=xp.float64) - Ny / 2) / (Ny * dy)
        kx_sq = (2 * float(np.pi) * fx) ** 2
        ky_sq = (2 * float(np.pi) * fy) ** 2
        kz_sq = k ** 2 - kx_sq[None, :] - ky_sq[:, None]
        prop = kz_sq > 0
        kz = xp.where(prop, xp.sqrt(xp.where(prop, kz_sq, 0.0)), 0.0)
        H = xp.where(prop, xp.exp(1j * kz * z), 0.0).astype(target_cdtype)
        if bandlimit and z != 0:
            Lx = Nx * dx
            Ly = Ny * dy
            fx_max = Lx / (2 * wavelength * abs(z))
            fy_max = Ly / (2 * wavelength * abs(z))
            bl_x = xp.abs(fx) < fx_max
            bl_y = xp.abs(fy) < fy_max
            mask = bl_x[None, :] & bl_y[:, None]
            H = H * mask.astype(target_cdtype)

    if H is None:
        # Spatial-frequency squared vectors (cached on numpy path).
        kx_sq, ky_sq = _get_or_make_freq_grids(Ny, Nx, dy, dx, xp is np)
        if bandlimit and z != 0:
            bl_x, bl_y = _get_or_make_bandlimit(
                Ny, Nx, dy, dx, wavelength, abs(z), xp is np)
        else:
            bl_x = bl_y = None

        # Chunked H construction, sized to fit a small slice of RAM.
        from ..memory import get_ram_budget
        ram = get_ram_budget()
        row_cost = 3 * Nx * 16   # bytes per row of workspace (complex128)
        if row_cost > 0:
            max_chunk = max(1, int(ram * 0.1 / row_cost))
        else:
            max_chunk = Ny
        chunk = min(Ny, max_chunk)

        H = xp.empty((Ny, Nx), dtype=target_cdtype)
        kept_count = 0
        for j0 in range(0, Ny, chunk):
            j1 = min(Ny, j0 + chunk)
            # kz_sq is float64 regardless of target dtype to keep the
            # huge kernel argument (kz * z up to ~1e6 rad) accurate.
            kz_sq_c = k**2 - kx_sq[None, :] - ky_sq[j0:j1, None]
            prop = kz_sq_c > 0
            kz_c = xp.where(prop, xp.sqrt(xp.maximum(kz_sq_c, 0)), 0)
            if target_cdtype == np.complex128:
                H_c = xp.where(prop, xp.exp(1j * kz_c * z), 0)
            else:
                # complex64 path: fold phase mod 2*pi in float64
                # BEFORE casting to float32 so the float32 precision
                # floor doesn't inject speckle-like noise.
                phase = xp.mod(kz_c * z, 2.0 * np.pi)
                c = xp.cos(phase).astype(target_fdtype)
                s = xp.sin(phase).astype(target_fdtype)
                H_c = xp.empty((j1 - j0, Nx), dtype=target_cdtype)
                H_c.real[:] = xp.where(prop, c, target_fdtype(0))
                H_c.imag[:] = xp.where(prop, s, target_fdtype(0))
            if bl_x is not None:
                bl_mask = bl_x[None, :] & bl_y[j0:j1, None]
                H_c *= bl_mask
                if verbose:
                    kept_count += int(xp.sum(bl_mask))
            H[j0:j1, :] = H_c

        if verbose and bl_x is not None:
            kept_frac = kept_count / (Nx * Ny)
            print(f"  Band-limiting: keeping {kept_frac*100:.1f}% of spectrum")
        if verbose:
            print(f"  ASM propagation: z = {z*1e3:.3f} mm  "
                  f"(H cache miss, built in {chunk}-row chunks)")
            print(f"  Grid: {Ny}x{Nx}, dx={dx*1e6:.3f} um, dy={dy*1e6:.3f} um")
            print(f"  Wavelength: {wavelength*1e9:.1f} nm")
        # Store under the numpy key only.  The cached H is read-only
        # in normal use; we don't deep-copy on lookup, so callers must
        # not mutate it in place.
        if h_key is not None:
            _h_cache_store(h_key, H)
    elif verbose:
        print(f"  ASM propagation: z = {z*1e3:.3f} mm  (H cache HIT)")

    # -- propagate: E_out = IFFT{ FFT{E_in} * H } ---------------------------
    if is_jax:
        E_fft = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(E_in)))
        E_out = xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(E_fft * H)))
    elif xp is np:
        E_fft = np.fft.fftshift(_fft2(np.fft.ifftshift(E_in)))
        E_out = np.fft.fftshift(_ifft2(np.fft.ifftshift(E_fft * H)))
    else:
        E_fft = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(E_in)))
        E_out = xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(E_fft * H)))

    if return_transfer_function:
        # 4.10: return a copy of the cached H so a caller that does
        # ``E_out, H = angular_spectrum_propagate(..., return_transfer_function=True)``
        # then ``H *= mask`` cannot mutate the cached entry and corrupt
        # every subsequent ASM call at the same key.
        H_returned = H.copy() if hasattr(H, 'copy') else xp.asarray(H)
        return E_out, H_returned
    else:
        return E_out


def apply_fresnel_curvature(
    E: np.ndarray,
    dx: float,
    wavelength: float,
    R: float,
    sign: int = +1,
    dy: Optional[float] = None,
) -> np.ndarray:
    """Apply (or remove) a Fresnel quadratic phase ``exp(i*sign*k*r^2/(2R))``.

    Used to convert between phase conventions when comparing fields
    produced by different libraries.

    Background
    ----------
    Lumenairy's propagators (and the standard Fresnel/ASM family --
    LightPipes, prysm, diffractio, POPPy, Zemax POP) keep the **full
    physical phase** at the output plane.  Some ray-trace-rooted
    aberration-analysis tools (notably OPDPy and Zemax wavefront
    operands like ``OPDX``) instead store the **chief-relative OPD**,
    which implicitly subtracts the natural Gaussian-beam wavefront
    curvature at the image plane.

    The two conventions differ by exactly a Fresnel quadratic phase
    ``exp(i*k*r^2/(2*R))`` with ``R = v - f`` for a thin-lens
    imager (image distance minus focal length).

    Use this function to round-trip between conventions:

    .. code-block:: python

        # Convert OPDPy / Zemax-OPD output to Lumenairy / LightPipes:
        E_absolute = apply_fresnel_curvature(
            E_chief_relative, dx, wavelength, R=v - f, sign=+1)

        # Convert Lumenairy / LightPipes output to chief-relative:
        E_chief_relative = apply_fresnel_curvature(
            E_absolute, dx, wavelength, R=v - f, sign=-1)

    For multi-element systems, ``R`` is the wavefront radius of
    curvature at the image plane predicted by Gaussian-beam ABCD
    propagation -- see Saleh & Teich, *Fundamentals of Photonics*,
    Section 3.1.

    Parameters
    ----------
    E : ndarray, complex 2D
        Input field.  Grid is assumed to be centred on the chief image
        point (the centre pixel is at coordinate ``(0, 0)``, with the
        same half-pixel offset convention as
        :func:`angular_spectrum_propagate`).
    dx : float
        Pixel pitch in the x-direction (metres).
    wavelength : float
        Wavelength (metres).
    R : float
        Wavefront radius of curvature (metres).  For a thin-lens
        imager, ``R = image_distance - focal_length``.
    sign : int, default ``+1``
        ``+1`` adds the curvature (chief-relative -> absolute).
        ``-1`` removes the curvature (absolute -> chief-relative).
    dy : float, optional
        Pixel pitch in y.  Defaults to ``dx``.

    Returns
    -------
    E_out : ndarray, complex
        Same shape and dtype as ``E``, with the Fresnel curvature
        multiplied (or divided) in.

    See also
    --------
    Wiki: "Phase conventions and inter-library comparison"
    """
    if dy is None:
        dy = dx
    # R = 0 / inf / NaN is treated as a no-op so multi-element
    # prescriptions where v-f is ill-defined (e.g. an afocal section)
    # can pass through without curvature.  This is documented behaviour
    # locked in by the test suite.
    if R == 0 or not np.isfinite(R):
        return E.copy()
    if sign not in (+1, -1):
        raise ValueError(f"sign must be +1 or -1, got {sign}")
    Ny, Nx = E.shape
    # 4.10: drop the spurious +0.5 half-pixel offset.  Every other
    # propagator in this file builds coordinates as (arange(N) - N/2)*dx
    # (no +0.5).  The mismatch produced a half-pixel walk-off in the
    # curvature centre relative to the propagated field grid, visible
    # as a small coma-like residual in OPDPy cross-checks.
    ax_x = (np.arange(Nx) - Nx / 2) * dx
    ax_y = (np.arange(Ny) - Ny / 2) * dy
    Y, X = np.meshgrid(ax_y, ax_x, indexing='ij')
    r2 = X * X + Y * Y
    k = 2.0 * np.pi / wavelength
    return E * np.exp(sign * 1j * k * r2 / (2.0 * R))


def angular_spectrum_propagate_batch(
    E_stack: np.ndarray,
    z: float,
    wavelength: float,
    dx: float,
    dy: Optional[float] = None,
    bandlimit: bool = True,
    use_gpu: bool = False,
) -> np.ndarray:
    """ASM propagation of a stack of fields ``(B, Ny, Nx)`` in one
    fused FFT pair (3.2.14).

    All ``B`` fields share the same grid + wavelength + propagation
    distance, so the transfer function ``H`` is built once (reusing
    the H cache) and broadcast across the batch.  Two batched FFTs
    (forward + inverse, axes ``(-2, -1)``) replace ``2*B`` separate
    2-D FFTs, which on JonesField (Ex, Ey) is ~30-60% wall-clock
    faster than calling :func:`angular_spectrum_propagate` per
    component.

    Parameters
    ----------
    E_stack : ndarray, complex, shape (B, Ny, Nx)
        Input field stack.  ``B`` must be at least 1.
    z, wavelength, dx, dy, bandlimit, use_gpu
        Same semantics as :func:`angular_spectrum_propagate`.

    Returns
    -------
    E_out : ndarray, complex, shape (B, Ny, Nx)
        Propagated stack, same dtype + array library as input.
    """
    if E_stack.ndim != 3:
        raise ValueError(
            f"angular_spectrum_propagate_batch: input must be 3-D "
            f"(B, Ny, Nx), got shape {E_stack.shape}.")
    # Validate using a representative 2-D slice; the batched call has
    # the same (z, wavelength, dx, dy) constraints as the scalar
    # propagator, so reuse the helper.
    _validate_propagator_inputs(E_stack[0], z, wavelength, dx, dy,
                                fn_name='angular_spectrum_propagate_batch')

    if CUPY_AVAILABLE and (use_gpu or _is_cupy_array(E_stack)):
        xp = cp
        if not _is_cupy_array(E_stack):
            E_stack = cp.asarray(E_stack)
    else:
        xp = np
        if _is_cupy_array(E_stack):
            E_stack = E_stack.get()

    B, Ny, Nx = E_stack.shape
    if dy is None:
        dy = dx

    if xp.iscomplexobj(E_stack):
        target_cdtype = E_stack.dtype
    else:
        target_cdtype = np.dtype(DEFAULT_COMPLEX_DTYPE)
        E_stack = E_stack.astype(target_cdtype)

    # Reuse the H cache from the scalar propagator: build H by
    # delegating to the scalar function on a tiny ``Ny x Nx`` field
    # of the right dtype with ``return_transfer_function=True``.  H
    # is read-only after construction so it is safe to reuse across
    # the batch.
    _proxy = xp.empty((Ny, Nx), dtype=target_cdtype)
    _, H = angular_spectrum_propagate(
        _proxy, z, wavelength, dx, dy=dy, bandlimit=bandlimit,
        return_transfer_function=True, use_gpu=(xp is not np),
    )

    # Single batched FFT pair across the last two axes.  pyFFTW's
    # multi-slot plan cache (also new in 3.2.14) keys on the full
    # shape including the batch dimension, so a 3-D plan is built on
    # the first call and reused thereafter.  The numpy / scipy
    # fallback paths handle 3-D input natively via ``fft2`` over the
    # last two axes.
    if xp is np:
        # Use scipy.fft for ND batched (workers parameter), pyFFTW
        # plan cache picks up the (B, Ny, Nx) shape automatically via
        # ``_fft2`` if the array is large enough.
        E_fft = xp.fft.fftshift(
            _fft2_nd(xp.fft.ifftshift(E_stack, axes=(-2, -1))),
            axes=(-2, -1))
        E_out = xp.fft.fftshift(
            _ifft2_nd(xp.fft.ifftshift(E_fft * H[None, :, :],
                                        axes=(-2, -1))),
            axes=(-2, -1))
    else:
        E_fft = xp.fft.fftshift(
            xp.fft.fft2(xp.fft.ifftshift(E_stack, axes=(-2, -1)),
                        axes=(-2, -1)),
            axes=(-2, -1))
        E_out = xp.fft.fftshift(
            xp.fft.ifft2(xp.fft.ifftshift(E_fft * H[None, :, :],
                                            axes=(-2, -1)),
                          axes=(-2, -1)),
            axes=(-2, -1))
    return E_out


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
        except Exception as e:
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
        except Exception as e:
            if not PYFFTW_FALLBACK_ON_ERROR:
                raise
            _handle_pyfftw_failure(x, 'ifft2_nd', e)
    if USE_SCIPY_FFT and SCIPY_FFT_AVAILABLE:
        return _scipy_fft.ifft2(x, axes=(-2, -1), workers=SCIPY_FFT_WORKERS)
    return np.fft.ifft2(x, axes=(-2, -1))


# ============================================================================
# Tilted / off-axis ASM propagation
# ============================================================================

def angular_spectrum_propagate_tilted(
    E_in: np.ndarray,
    z: float,
    wavelength: float,
    dx: float,
    dy: Optional[float] = None,
    tilt_x: float = 0.0,
    tilt_y: float = 0.0,
    bandlimit: bool = True,
    *,
    tilt_x_deg: Optional[float] = None,
    tilt_y_deg: Optional[float] = None,
) -> np.ndarray:
    """
    ASM propagation with a carrier tilt (off-axis propagation).

    Propagates the field while accounting for a mean propagation direction
    that is tilted relative to the optical axis.  This is useful for:

    - Beams arriving at an angle
    - Propagation after a prism or wedge
    - Off-axis portions of a wide-field system

    The tilt is handled by shifting the frequency-domain transfer function,
    which is equivalent to propagating the field in a tilted reference frame.

    Parameters
    ----------
    E_in : ndarray (complex, Ny x Nx)
        Input electric field.

    z : float
        Propagation distance [m] along the tilted axis.

    wavelength : float
        Optical wavelength [m].

    dx : float
        Grid spacing in x [m].

    dy : float, optional
        Grid spacing in y [m].  Defaults to dx.

    tilt_x, tilt_y : float, default 0.0
        Tilt angles [radians] of the propagation direction relative to the
        z-axis.  The beam propagates at angle (tilt_x, tilt_y) from the
        optical axis.

    bandlimit : bool, default True
        Apply band-limiting to avoid aliasing.

    Returns
    -------
    E_out : ndarray (complex, Ny x Nx)
        Propagated electric field.

    Notes
    -----
    The method removes the carrier frequency (tilt) before propagation,
    then restores it afterwards.  This keeps the field well-centred on
    the grid even for large tilt angles, avoiding grid walk-off.

    The carrier spatial frequencies are::

        fx0 = sin(tilt_x) / wavelength
        fy0 = sin(tilt_y) / wavelength

    The field is demodulated as::

        E_demod = E_in * exp(-i * 2*pi * (fx0*X + fy0*Y))

    propagated with a shifted transfer function, then remodulated::

        E_out = E_prop * exp(+i * 2*pi * (fx0*X + fy0*Y))

    For ``tilt_x = tilt_y = 0`` this reduces to standard ASM propagation.

    4.7+: convenience kwargs ``tilt_x_deg`` / ``tilt_y_deg`` accept the
    angle in degrees and take precedence over the radian forms when
    supplied.  These are part of the broader push toward ``_deg`` as
    the canonical user-facing angle unit (see the polish-pass note in
    :ref:`Release Notes`).
    """
    _validate_propagator_inputs(E_in, z, wavelength, dx, dy,
                                fn_name='angular_spectrum_propagate_tilted')
    if tilt_x_deg is not None:
        tilt_x = float(np.radians(tilt_x_deg))
    if tilt_y_deg is not None:
        tilt_y = float(np.radians(tilt_y_deg))
    if dy is None:
        dy = dx

    Ny, Nx = E_in.shape

    # -- carrier spatial frequencies from tilt angles ------------------------
    fx0 = np.sin(tilt_x) / wavelength
    fy0 = np.sin(tilt_y) / wavelength

    # Shortcut: no tilt -> fall back to standard ASM
    if abs(fx0) < 1e-15 and abs(fy0) < 1e-15:
        return angular_spectrum_propagate(E_in, z, wavelength, dx, dy,
                                          bandlimit=bandlimit)

    # Target complex dtype (matches angular_spectrum_propagate / RS).
    if np.iscomplexobj(E_in):
        target_cdtype = E_in.dtype
    else:
        target_cdtype = np.dtype(DEFAULT_COMPLEX_DTYPE)

    # -- spatial coordinate grids (carrier; per-call) ------------------------
    x = (np.arange(Nx) - Nx / 2) * dx
    y = (np.arange(Ny) - Ny / 2) * dy
    X, Y = np.meshgrid(x, y)

    # -- demodulate: remove carrier tilt -------------------------------------
    carrier = np.exp(-1j * 2 * np.pi * (fx0 * X + fy0 * Y))
    E_demod = E_in * carrier

    # ── H cache (NumPy backend) ───────────────────────────────────────
    # The shifted transfer function depends on (Ny, Nx, dy, dx,
    # wavelength, z, fx0, fy0, bandlimit, dtype).  fx0/fy0 are the
    # tilt-derived carrier frequencies and they fully encode the
    # propagation-direction shift, so the cache key handles arbitrary
    # tilt angles without needing tilt_x / tilt_y in the key directly.
    # 'ASM_TILTED' tag keeps these entries disjoint from plain-ASM ones.
    h_key = (int(Ny), int(Nx), float(dy), float(dx),
             float(wavelength), float(z),
             float(fx0), float(fy0),
             bool(bandlimit),
             np.dtype(target_cdtype).str, 'ASM_TILTED')
    H = _h_cache_lookup(h_key)

    if H is None:
        # -- shifted transfer function -------------------------------------
        # kz is evaluated at (fx + fx0, fy + fy0) so the baseband field
        # propagates with the correct kz for each plane-wave component.
        k = 2 * np.pi / wavelength
        dfx = 1.0 / (Nx * dx)
        dfy = 1.0 / (Ny * dy)
        fx = (np.arange(Nx) - Nx / 2) * dfx
        fy = (np.arange(Ny) - Ny / 2) * dfy
        FX, FY = np.meshgrid(fx, fy)

        FX_shifted = FX + fx0
        FY_shifted = FY + fy0
        kx = 2 * np.pi * FX_shifted
        ky = 2 * np.pi * FY_shifted

        kz_sq = k**2 - kx**2 - ky**2
        kz = np.where(kz_sq > 0, np.sqrt(np.maximum(kz_sq, 0)), 0)
        H = np.exp(1j * kz * z)
        H = np.where(kz_sq > 0, H, 0)

        # -- band-limiting on the ORIGINAL-FRAME spectrum ----------------
        # Matsushima bounds the FREQUENCY OF THE CHIRP in the angular-
        # spectrum kernel, which depends on the original (non-shifted)
        # frequency (FX + fx0, FY + fy0): that's where the chirp's
        # phase-derivative is taken.  The H built above is also
        # evaluated at the shifted arguments, so the mask must use
        # FX_shifted = FX + fx0 (and FY + fy0) -- otherwise it clips the
        # *baseband* (around FX=0) and lets through the actual aliasing-
        # prone high-(FX+fx0) bands.  Pre-4.10 used `|FX| < fx_max`,
        # which for any non-trivial tilt killed the baseband DC and
        # zeroed the propagated field.
        if bandlimit and z != 0:
            Lx = Nx * dx
            Ly = Ny * dy
            fx_max = Lx / (2 * wavelength * abs(z))
            fy_max = Ly / (2 * wavelength * abs(z))
            H = np.where((np.abs(FX_shifted) < fx_max) &
                          (np.abs(FY_shifted) < fy_max), H, 0)

        if H.dtype != target_cdtype:
            H = H.astype(target_cdtype)

        _h_cache_store(h_key, H)

    # -- propagate baseband with shifted transfer function -------------------
    E_fft = np.fft.fftshift(_fft2(np.fft.ifftshift(E_demod)))
    E_prop = np.fft.fftshift(_ifft2(np.fft.ifftshift(E_fft * H)))

    # -- remodulate: restore carrier tilt ------------------------------------
    E_out = E_prop * np.conj(carrier)

    return E_out


# ============================================================================
# ASM with arbitrary user-specified output grid (MFT inverse-FT)
# ============================================================================

def angular_spectrum_propagate_mft(
    E_in: np.ndarray,
    z: float,
    wavelength: float,
    dx_in: float,
    dx_out: float,
    N_out: int,
    *,
    dy_in: Optional[float] = None,
    dy_out: Optional[float] = None,
    centre_out: Tuple[float, float] = (0.0, 0.0),
    bandlimit: bool = True,
    use_gpu: bool = False,
) -> np.ndarray:
    """Exact Angular Spectrum Method propagation onto an arbitrary
    user-specified output grid.

    Identical math to :func:`angular_spectrum_propagate` for the
    forward (FFT + transfer-function) step, but the inverse Fourier
    step samples the output field on a user-specified grid via a
    Bluestein chirp-Z transform rather than reusing the input's FFT
    grid.  This enables **focal-plane zoom for high-NA / non-paraxial
    systems** without zero-padding the input.

    Use cases where this beats :func:`fresnel_propagate_mft`:
        - High NA (NA > ~0.1) where the paraxial Fresnel assumption breaks.
        - Fields with significant evanescent content (sub-wavelength
          structures, near-field).
        - Anywhere :func:`angular_spectrum_propagate` is preferred over
          :func:`fresnel_propagate`, but you also want non-natural output
          grid sampling.

    Parameters
    ----------
    E_in : ndarray (complex, Ny x Nx)
    z : float
        Propagation distance [m].
    wavelength : float
    dx_in : float
        Input grid spacing in x [m].
    dx_out : float
        Output grid spacing in x [m] (independent of ``dx_in`` and ``z``).
    N_out : int
        Output grid size (square).
    dy_in, dy_out : float, optional
    centre_out : (float, float), optional
        Physical ``(x, y)`` centre of the output grid [m].  Defaults to
        ``(0, 0)`` (on-axis).
    bandlimit : bool, default True
        Apply Matsushima-Shimobaba band-limiting to the ASM transfer
        function on the input frequency grid.  Same default and effect
        as :func:`angular_spectrum_propagate`.
    use_gpu : bool, default False

    Returns
    -------
    E_out : ndarray (complex, N_out x N_out)
        Output field on the user's centred grid.  Carries the absolute
        physical phase including the natural Fresnel curvature -- this
        is the same convention as :func:`angular_spectrum_propagate`,
        :func:`fresnel_propagate`, and every other Lumenairy propagator.

    Notes
    -----
    Algorithm: regular FFT of the input, transfer function multiplication,
    then Bluestein chirp-Z inverse FT onto the chosen output grid.
    Cost: 2 * O(N^2 log N)  + O((N + M) log (N + M))  per axis.

    For the same-grid case (``dx_out == dx_in`` and ``N_out == Nx_in``,
    ``centre_out == (0, 0)``) the result agrees with
    :func:`angular_spectrum_propagate` to roughly float64 round-off
    (~1e-12 relative error).

    See also
    --------
    angular_spectrum_propagate : exact ASM with the natural FFT grid.
    fresnel_propagate_mft : paraxial Fresnel with arbitrary output grid.
    fraunhofer_propagate_mft : far-field with arbitrary output grid.
    """
    _validate_propagator_inputs(E_in, z, wavelength, dx_in, dy_in,
                                fn_name='angular_spectrum_propagate_mft')
    from ..backend import is_jax_array
    from ._bluestein import _bluestein_centred_2d

    is_jax = is_jax_array(E_in)
    if is_jax:
        import jax.numpy as _jnp
        xp = _jnp
        fft2 = _jnp.fft.fft2
        ifft2 = _jnp.fft.ifft2
    elif CUPY_AVAILABLE and (use_gpu or _is_cupy_array(E_in)):
        if not _ensure_cupy_loaded():
            raise RuntimeError("CuPy requested but failed to load.")
        xp = cp
        fft2 = cp.fft.fft2
        ifft2 = cp.fft.ifft2
        if not _is_cupy_array(E_in):
            E_in = cp.asarray(E_in)
    else:
        xp = np
        fft2 = _fft2
        ifft2 = _ifft2
        if _is_cupy_array(E_in):
            E_in = E_in.get()

    if dy_in is None:
        dy_in = dx_in
    if dy_out is None:
        dy_out = dx_out

    Ny_in, Nx_in = E_in.shape
    Ny_out = Nx_out = int(N_out)

    if xp.iscomplexobj(E_in):
        target_cdtype = E_in.dtype
    else:
        target_cdtype = np.dtype(DEFAULT_COMPLEX_DTYPE)

    k = 2.0 * np.pi / wavelength
    xc, yc = float(centre_out[0]), float(centre_out[1])

    # ----- 1) Build ASM transfer function H(fx, fy) on the input freq grid --
    # Same construction as angular_spectrum_propagate (centred convention,
    # exact `kz = sqrt(k^2 - kx^2 - ky^2)`, optional Matsushima band-limit).
    #
    # H depends only on the input geometry (Ny_in, Nx_in, dy_in, dx_in,
    # wavelength, z, bandlimit, dtype) -- the user-specified output grid
    # (dx_out, N_out, centre_out) only enters in the Bluestein step
    # below.  Cache H on the input-geometry signature so repeat calls
    # at the same input plane onto different output grids share one
    # H build.  ``'ASM_MFT'`` tag keeps these entries disjoint from
    # plain ASM.  4.12.0: both backends now use ``fx < fx_max`` (open
    # interval, matching the Matsushima-Shimobaba paper and plain ASM);
    # pre-4.12 the NumPy branch used `<=` (one-bin off from JAX).
    if is_jax:
        # JAX path: build under the tracer (no host-side cache).
        fx = (xp.arange(Nx_in, dtype=xp.float64) - Nx_in / 2.0) / (Nx_in * dx_in)
        fy = (xp.arange(Ny_in, dtype=xp.float64) - Ny_in / 2.0) / (Ny_in * dy_in)
        kx_sq = (2.0 * float(np.pi) * fx) ** 2
        ky_sq = (2.0 * float(np.pi) * fy) ** 2
        kz_sq = k * k - kx_sq[None, :] - ky_sq[:, None]
        prop_mask = kz_sq > 0
        kz = xp.where(prop_mask, xp.sqrt(xp.where(prop_mask, kz_sq, 0.0)),
                      0.0)
        H = xp.where(prop_mask, xp.exp(1j * kz * z), 0.0).astype(target_cdtype)
        if bandlimit and z != 0:
            Lx_phys = Nx_in * dx_in
            Ly_phys = Ny_in * dy_in
            fx_max = Lx_phys / (2.0 * wavelength * abs(z))
            fy_max = Ly_phys / (2.0 * wavelength * abs(z))
            # 4.10: use strict less-than (matches plain ASM at line
            # 1200 and the Matsushima-Shimobaba paper, which uses an
            # open-interval cutoff).  Pre-4.10 ASM-MFT used <= here,
            # one-bin off from the ASM reference.
            bl_mask = ((xp.abs(fx)[None, :] < fx_max)
                       & (xp.abs(fy)[:, None] < fy_max))
            H = xp.where(bl_mask, H, 0.0).astype(target_cdtype)
    else:
        # NumPy / CuPy paths share a NumPy-host H cache; CuPy uploads
        # via xp.asarray on demand.
        h_key = (int(Ny_in), int(Nx_in), float(dy_in), float(dx_in),
                 float(wavelength), float(z),
                 bool(bandlimit),
                 np.dtype(target_cdtype).str, 'ASM_MFT')
        H_np = _h_cache_lookup(h_key)
        if H_np is None:
            fx = (np.arange(Nx_in, dtype=np.float64) - Nx_in / 2.0) / (Nx_in * dx_in)
            fy = (np.arange(Ny_in, dtype=np.float64) - Ny_in / 2.0) / (Ny_in * dy_in)
            kx_sq = (2.0 * np.pi * fx) ** 2
            ky_sq = (2.0 * np.pi * fy) ** 2
            kz_sq = k * k - kx_sq[None, :] - ky_sq[:, None]
            prop_mask = kz_sq > 0
            kz = np.where(prop_mask,
                          np.sqrt(np.where(prop_mask, kz_sq, 0.0)), 0.0)
            H_np = np.where(prop_mask, np.exp(1j * kz * z), 0.0).astype(
                target_cdtype)
            if bandlimit and z != 0:
                Lx_phys = Nx_in * dx_in
                Ly_phys = Ny_in * dy_in
                fx_max = Lx_phys / (2.0 * wavelength * abs(z))
                fy_max = Ly_phys / (2.0 * wavelength * abs(z))
                # 4.12.0 (audit round-4 B1-4): use strict `<` to match
                # the JAX branch above (and plain ASM at line ~1200).
                # Pre-4.12 NumPy used `<=` -- one-bin disagreement
                # between backends at the band-limit boundary.
                bl_mask = ((np.abs(fx)[None, :] < fx_max)
                           & (np.abs(fy)[:, None] < fy_max))
                H_np = np.where(bl_mask, H_np, 0.0).astype(target_cdtype)
            _h_cache_store(h_key, H_np)
        H = H_np if xp is np else xp.asarray(H_np)

    # ----- 2) FFT input to angular spectrum (centred convention) ------------
    # Match the existing angular_spectrum_propagate's fftshift/ifftshift
    # idiom so the centred-frequency grid lines up.
    if is_jax:
        E_fft = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(E_in)))
    else:
        E_fft = xp.fft.fftshift(fft2(xp.fft.ifftshift(E_in)))
    A_propagated = E_fft * H

    # ----- 3) Inverse FT onto user-specified output grid via Bluestein ------
    # The inverse FT of the propagated angular spectrum is
    #   E_out(x_out, y_out) = integral A(fx, fy) * exp(+2*pi*j*(fx*x_out + fy*y_out)) dfx dfy
    # Discretised on the centred input frequency grid:
    #   fx[nx] = (nx - Nx_in/2) / (Nx_in * dx_in)   (with dfx = 1/(Nx_in * dx_in))
    #   x_out[kx] = (kx - Nx_out/2) * dx_out + xc
    # The product fx[nx] * x_out[kx] expands to a centred Bluestein form
    # with alpha = dfx * dx_out = dx_out / (Nx_in * dx_in) and sign = +1.
    # The 1/(Nx_in*Ny_in) prefactor matches numpy/scipy's IFFT normalisation
    # so that round-tripping through ASM-MFT recovers the input on the
    # natural grid.
    alpha_x = dx_out / (Nx_in * dx_in)
    alpha_y = dy_out / (Ny_in * dy_in)
    kc_x = Nx_out / 2.0 - xc / dx_out
    kc_y = Ny_out / 2.0 - yc / dy_out

    F = _bluestein_centred_2d(
        A_propagated, alpha_x, alpha_y, Ny_out, Nx_out,
        n_centre_in_x=Nx_in / 2.0,
        n_centre_in_y=Ny_in / 2.0,
        k_centre_out_x=kc_x,
        k_centre_out_y=kc_y,
        sign=+1, xp=xp, fft2=fft2, ifft2=ifft2,
        target_cdtype=target_cdtype,
    )

    norm = target_cdtype.type(1.0 / (Nx_in * Ny_in))
    E_out = F * norm
    if E_out.dtype != target_cdtype:
        E_out = E_out.astype(target_cdtype)
    return E_out


# ============================================================================
# Single-FFT Fresnel propagation
# ============================================================================

def fresnel_propagate(
    E_in: np.ndarray,
    z: float,
    wavelength: float,
    dx: float,
    dy: Optional[float] = None,
) -> Tuple[np.ndarray, float, float]:
    """
    Propagate a field using the single-FFT Fresnel method.

    This is the Fresnel (paraxial) approximation to diffraction.  It uses a
    single FFT and is faster than ASM for long propagation distances, but
    **changes the grid spacing** in the output plane.

    Parameters
    ----------
    E_in : ndarray (complex, Ny x Nx)
        Input field.

    z : float
        Propagation distance [m].

    wavelength : float
        Wavelength [m].

    dx : float
        Input grid spacing in x [m].

    dy : float, optional
        Input grid spacing in y [m].  Defaults to dx.

    Returns
    -------
    E_out : ndarray (complex, Ny x Nx)
        Output field.

    dx_out : float
        Output grid spacing in x [m].

    dy_out : float
        Output grid spacing in y [m].

    Notes
    -----
    The output grid spacing is::

        dx_out = wavelength * |z| / (Nx * dx)

    This method is valid when the Fresnel number is moderate::

        N_F = a^2 / (lambda * z) ~ 1

    where *a* is the beam / aperture radius.

    For very short distances (large Fresnel number), use ASM instead.
    For very long distances (small Fresnel number), this becomes equivalent
    to the Fraunhofer approximation.
    """
    _validate_propagator_inputs(E_in, z, wavelength, dx, dy,
                                fn_name='fresnel_propagate')
    from ..backend import array_namespace, is_jax_array
    xp = array_namespace(E_in)
    is_jax = is_jax_array(E_in)

    if dy is None:
        dy = dx

    # 4.9 fix (audit #3.3): Fresnel is a forward-propagating paraxial
    # kernel; z <= 0 is unphysical here (the FFT direction isn't
    # flipped for back-prop, and the abs(z) in dx_out below mixes
    # signs with the raw-z phase prefactor giving mathematically
    # nonsense output rather than a back-propagated field).  Refuse
    # with a clear error pointing at ASM or RS, both of which do
    # handle back-propagation correctly for the propagating spectrum.
    if z <= 0:
        raise ValueError(
            f"fresnel_propagate: z must be > 0 (got z={z}).  Fresnel "
            f"is a forward-propagating paraxial kernel and does not "
            f"support back-propagation.  Use angular_spectrum_propagate "
            f"or rayleigh_sommerfeld_propagate (both handle negative z) "
            f"if you need to back-propagate a field.")

    Ny, Nx = E_in.shape
    k = 2 * np.pi / wavelength

    # 4.10: honour caller dtype so a complex64 E_in stays complex64
    # through the Fresnel pipeline (pre-4.10 it was silently promoted
    # to complex128 via the python-float `2 * np.pi` and `1j` constants).
    if xp.iscomplexobj(E_in):
        target_cdtype = E_in.dtype
    else:
        target_cdtype = np.dtype(DEFAULT_COMPLEX_DTYPE)
    fdtype = np.float64 if np.dtype(target_cdtype) == np.complex128 else np.float32

    # -- input / output coordinates -----------------------------------------
    x1 = (xp.arange(Nx, dtype=fdtype) - Nx / 2) * dx
    y1 = (xp.arange(Ny, dtype=fdtype) - Ny / 2) * dy
    X1, Y1 = xp.meshgrid(x1, y1, indexing='xy')
    dx_out = wavelength * z / (Nx * dx)
    dy_out = wavelength * z / (Ny * dy)
    x2 = (xp.arange(Nx, dtype=fdtype) - Nx / 2) * dx_out
    y2 = (xp.arange(Ny, dtype=fdtype) - Ny / 2) * dy_out
    X2, Y2 = xp.meshgrid(x2, y2, indexing='xy')

    # -- quadratic phase in input plane --------------------------------------
    phase_in = xp.exp(1j * k / (2 * z) * (X1**2 + Y1**2)).astype(target_cdtype)
    E_mod = E_in.astype(target_cdtype) * phase_in

    # -- FFT -- use _fft for NumPy/CuPy fast path; jnp.fft for JAX -----------
    if is_jax:
        E_fft = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(E_mod)))
    else:
        E_fft = np.fft.fftshift(_fft2(np.fft.ifftshift(E_mod)))

    # -- quadratic phase in output plane + prefactor -------------------------
    prefactor = (xp.exp(1j * k * z) / (1j * wavelength * z)
                 * xp.exp(1j * k / (2 * z) * (X2**2 + Y2**2))
                 * dx * dy)
    prefactor = prefactor.astype(target_cdtype)

    E_out = prefactor * E_fft

    return E_out, dx_out, dy_out


# =============================================================================
# FRAUNHOFER (FAR-FIELD) PROPAGATION
# =============================================================================

def resample_field(
    E_in: np.ndarray,
    dx_in: float,
    dx_out: float,
    N_out: Optional[int] = None,
    order: int = 3,
) -> Tuple[np.ndarray, float]:
    """
    Resample a complex optical field from one grid spacing to another.

    This is the bridge function for switching between propagation methods
    that use different grid spacings (e.g. Fresnel output -> ASM input,
    or vice versa).  Both amplitude and phase are interpolated using
    scipy's map_coordinates.

    Parameters
    ----------
    E_in : ndarray (complex, Ny x Nx)
        Input field on a grid with spacing ``dx_in``.
    dx_in : float
        Input grid spacing [m].
    dx_out : float
        Desired output grid spacing [m].
    N_out : int or None
        Output grid size.  If ``None``, chosen so the output covers the
        same physical extent as the input: ``N_out = round(N_in * dx_in / dx_out)``.
    order : int, default 3
        Interpolation order (1=linear, 3=cubic, 5=quintic).

    Returns
    -------
    E_out : ndarray (complex, N_out x N_out)
        Resampled field on the new grid.
    dx_out : float
        The output grid spacing (same as the input parameter, returned
        for convenience so callers can chain: ``E, dx = resample_field(...)``).

    Notes
    -----
    - Interpolation introduces a small error proportional to (dx_out/feature_size)^order.
      For order=3 (cubic), this is < 0.1% when features are sampled at >= 4 pixels.
    - For downsampling (dx_out > dx_in), consider anti-alias filtering first.
    - The field is assumed to be on a centered grid: x = (arange(N) - N/2) * dx.
    """
    from scipy.ndimage import map_coordinates

    Ny_in, Nx_in = E_in.shape
    if N_out is None:
        Nx_out = int(round(Nx_in * dx_in / dx_out))
        Ny_out = int(round(Ny_in * dx_in / dx_out))
    else:
        Nx_out = Ny_out = int(N_out)

    # Output coordinates in input-pixel units.
    # Input grid:  x_in[i]  = (i - Nx_in/2)  * dx_in
    # Output grid: x_out[j] = (j - Nx_out/2) * dx_out
    # Map: i = x_out / dx_in + Nx_in/2 = (j - Nx_out/2) * dx_out/dx_in + Nx_in/2
    scale = dx_out / dx_in
    jx = np.arange(Nx_out)
    jy = np.arange(Ny_out)
    ix = (jx - Nx_out / 2) * scale + Nx_in / 2
    iy = (jy - Ny_out / 2) * scale + Ny_in / 2
    IX, IY = np.meshgrid(ix, iy)
    coords = np.array([IY.ravel(), IX.ravel()])

    # Interpolate real and imaginary parts separately
    real_out = map_coordinates(E_in.real, coords, order=order, mode='constant', cval=0.0)
    imag_out = map_coordinates(E_in.imag, coords, order=order, mode='constant', cval=0.0)
    E_out = (real_out + 1j * imag_out).reshape(Ny_out, Nx_out)

    return E_out, dx_out


# ============================================================================
# Fresnel propagation onto an arbitrary user-specified output grid (MFT)
# ============================================================================

def fresnel_propagate_mft(
    E_in: np.ndarray,
    z: float,
    wavelength: float,
    dx_in: float,
    dx_out: float,
    N_out: int,
    *,
    dy_in: Optional[float] = None,
    dy_out: Optional[float] = None,
    centre_out: Tuple[float, float] = (0.0, 0.0),
    use_gpu: bool = False,
) -> np.ndarray:
    """Fresnel propagation onto an arbitrary user-specified output grid.

    Unlike :func:`fresnel_propagate`, which forces ``dx_out = lambda*z/(N*dx_in)``
    and ``N_out = N_in``, this routine evaluates the Fresnel diffraction
    integral at exactly the output grid you ask for.  It is the standard
    tool for **focal-plane zoom** -- sampling a tightly-focused region of
    the output plane at sub-FFT-pitch resolution without padding the input
    grid by the corresponding factor.

    The math is the same as :func:`fresnel_propagate` (single-FFT paraxial
    Fresnel: input quadratic phase + Fourier transform + output quadratic
    phase + ``exp(i*k*z) / (i*lambda*z)`` carrier prefactor).  Only the
    Fourier-transform step is replaced with a Bluestein chirp-Z transform
    that samples directly onto the chosen output grid.

    Parameters
    ----------
    E_in : ndarray (complex, Ny x Nx)
        Input field on a centred grid (pixel ``[Ny//2, Nx//2]`` at
        coordinate ``(0, 0)``, matching the convention of
        :func:`fresnel_propagate`).
    z : float
        Propagation distance [m].
    wavelength : float
        Wavelength [m].
    dx_in : float
        Input grid spacing in x [m].
    dx_out : float
        Desired output grid spacing in x [m].  This is independent of
        ``dx_in`` and ``z`` -- pick whatever sampling you want at the
        output plane.
    N_out : int
        Output grid size (square output: ``N_out`` by ``N_out``).
    dy_in, dy_out : float, optional
        Input / output grid spacings in y.  Default to ``dx_in`` / ``dx_out``.
    centre_out : (float, float), optional
        Physical ``(x, y)`` centre of the output grid [m].  Defaults to
        ``(0, 0)`` (on-axis).  Use a non-zero value to zoom into an
        off-axis region of the output plane (e.g. the chief image of a
        field point off the optical axis).
    use_gpu : bool, default False
        Route through CuPy if available.  Auto-detected from ``E_in``
        (CuPy / JAX arrays use their native backend regardless of this
        flag).

    Returns
    -------
    E_out : ndarray (complex, N_out x N_out)
        Output field on the user's centred grid.

    Notes
    -----
    Algorithm: O((N + M) log (N + M)) per axis via Bluestein's chirp-Z
    transform with two zero-padded 2-D FFTs.  Substantially faster than
    a direct matrix-Fourier transform (O(N^2 M^2)) for typical
    focal-zoom workflows.

    Sampling: the same Fresnel-number heuristic as :func:`fresnel_propagate`
    applies to ``dx_in`` and ``z`` (no new validity restriction comes
    from the user-specified output grid).  In particular the input must
    Nyquist-resolve the input quadratic phase
    ``exp(i*k/(2z) * (X_in^2 + Y_in^2))`` -- if ``dx_in`` is too coarse
    for the chosen ``z``, the output is aliased regardless of ``dx_out``.

    See also
    --------
    fresnel_propagate : single-FFT Fresnel with the natural FFT output grid.
    fraunhofer_propagate_mft : far-field counterpart.
    angular_spectrum_propagate_mft : exact ASM with arbitrary output grid.
    """
    _validate_propagator_inputs(E_in, z, wavelength, dx_in, dy_in,
                                fn_name='fresnel_propagate_mft')
    # 4.9 fix (audit #3.3): forward-only -- see ``fresnel_propagate``.
    if z <= 0:
        raise ValueError(
            f"fresnel_propagate_mft: z must be > 0 (got z={z}).  "
            f"Fresnel-MFT is the focal-plane-zoomed variant of "
            f"fresnel_propagate and inherits its forward-only "
            f"restriction.  Use angular_spectrum_propagate_mft for "
            f"back-propagation with arbitrary output grid.")
    # ----- backend dispatch -------------------------------------------------
    from ..backend import is_jax_array
    from ._bluestein import _bluestein_centred_2d

    is_jax = is_jax_array(E_in)
    if is_jax:
        import jax.numpy as _jnp
        xp = _jnp
        fft2 = _jnp.fft.fft2
        ifft2 = _jnp.fft.ifft2
    elif CUPY_AVAILABLE and (use_gpu or _is_cupy_array(E_in)):
        if not _ensure_cupy_loaded():
            raise RuntimeError("CuPy requested but failed to load.")
        xp = cp
        fft2 = cp.fft.fft2
        ifft2 = cp.fft.ifft2
        if not _is_cupy_array(E_in):
            E_in = cp.asarray(E_in)
    else:
        xp = np
        fft2 = _fft2
        ifft2 = _ifft2
        if _is_cupy_array(E_in):
            E_in = E_in.get()

    if dy_in is None:
        dy_in = dx_in
    if dy_out is None:
        dy_out = dx_out

    Ny_in, Nx_in = E_in.shape
    Ny_out = Nx_out = int(N_out)

    if xp.iscomplexobj(E_in):
        target_cdtype = E_in.dtype
    else:
        target_cdtype = np.dtype(DEFAULT_COMPLEX_DTYPE)

    k = 2.0 * np.pi / wavelength
    xc, yc = float(centre_out[0]), float(centre_out[1])

    # ----- coordinate grids (numpy for chirp construction) ------------------
    n_x = np.arange(Nx_in, dtype=np.float64)
    n_y = np.arange(Ny_in, dtype=np.float64)
    k_x = np.arange(Nx_out, dtype=np.float64)
    k_y = np.arange(Ny_out, dtype=np.float64)
    x_in = (n_x - Nx_in / 2.0) * dx_in
    y_in = (n_y - Ny_in / 2.0) * dy_in
    x_out = (k_x - Nx_out / 2.0) * dx_out + xc
    y_out = (k_y - Ny_out / 2.0) * dy_out + yc
    X_in_np, Y_in_np = np.meshgrid(x_in, y_in, indexing='xy')
    X_out_np, Y_out_np = np.meshgrid(x_out, y_out, indexing='xy')

    def _to_xp(arr_np_complex):
        a = arr_np_complex.astype(target_cdtype, copy=False)
        if xp is np:
            return a
        return xp.asarray(a)

    # ----- 1) input-plane quadratic phase -----------------------------------
    quad_in = _to_xp(np.exp(1j * k / (2.0 * z) * (X_in_np**2 + Y_in_np**2)))
    E_mod = E_in * quad_in

    # ----- 2) Fresnel Fourier integral via Bluestein on centred grid --------
    # Sum_{ny, nx} E_mod[ny, nx] * exp(-i*k/z * (x_in[nx]*x_out[kx] + y_in[ny]*y_out[ky]))
    # = Sum_{ny, nx} E_mod[ny, nx]
    #              * exp(-2*pi*j*alpha_x*(nx - Nx_in/2)*(kx - kc_x))
    #              * exp(-2*pi*j*alpha_y*(ny - Ny_in/2)*(ky - kc_y))
    # with
    #     alpha = dx_in*dx_out/(lambda*z)   ("natural" Bluestein coefficient)
    #     kc_x  = Nx_out/2 - xc/dx_out      (output centre shifted by centre_out)
    alpha_x = dx_in * dx_out / (wavelength * z)
    alpha_y = dy_in * dy_out / (wavelength * z)
    kc_x = Nx_out / 2.0 - xc / dx_out
    kc_y = Ny_out / 2.0 - yc / dy_out

    F = _bluestein_centred_2d(
        E_mod, alpha_x, alpha_y, Ny_out, Nx_out,
        n_centre_in_x=Nx_in / 2.0,
        n_centre_in_y=Ny_in / 2.0,
        k_centre_out_x=kc_x,
        k_centre_out_y=kc_y,
        sign=-1, xp=xp, fft2=fft2, ifft2=ifft2,
        target_cdtype=target_cdtype,
    )

    # ----- 3) output-plane quadratic phase + carrier prefactor + area -------
    quad_out = _to_xp(np.exp(1j * k / (2.0 * z) * (X_out_np**2 + Y_out_np**2)))
    prefactor = (np.exp(1j * k * z) / (1j * wavelength * z)) * dx_in * dy_in
    prefactor_c = target_cdtype.type(prefactor)

    E_out = prefactor_c * quad_out * F
    if E_out.dtype != target_cdtype:
        E_out = E_out.astype(target_cdtype)
    return E_out


def fraunhofer_propagate(
    E_in: np.ndarray,
    z: float,
    wavelength: float,
    dx: float,
    dy: Optional[float] = None,
) -> Tuple[np.ndarray, float, float]:
    """
    Propagate a field to the Fraunhofer (far-field) diffraction pattern.

    This is the far-field limit of the Fresnel propagator, valid when the
    Fresnel number is small:

        N_F = a^2 / (lambda * z) << 1

    where ``a`` is the characteristic aperture/beam radius. In practice this
    means ``z`` must be large compared to ``a^2 / lambda``. For smaller
    distances, use :func:`fresnel_propagate` or :func:`angular_spectrum_propagate`.

    Mathematically, the Fraunhofer integral reduces to a single Fourier
    transform of the input field with a quadratic phase and scaling prefactor::

        E(x2, y2) = [exp(i*k*z) / (i*lambda*z)]
                    * exp(i*k/(2z) * (x2^2 + y2^2))
                    * FFT{E(x1, y1)} * dx*dy

    Parameters
    ----------
    E_in : ndarray (complex, N×N)
        Input field.
    z : float
        Propagation distance [m].
    wavelength : float
        Free-space wavelength [m].
    dx : float
        Input grid spacing in x [m].
    dy : float, optional
        Input grid spacing in y [m]. Defaults to dx.

    Returns
    -------
    E_out : ndarray (complex, N×N)
        Field in the far-field plane.
    dx_out : float
        Output grid spacing in x [m] = wavelength * |z| / (N * dx).
    dy_out : float
        Output grid spacing in y [m] = wavelength * |z| / (N * dy).

    Notes
    -----
    The output grid spacing is the same as :func:`fresnel_propagate`:

        dx_out = wavelength * |z| / (N * dx)

    The difference from Fresnel is that Fraunhofer drops the input-plane
    quadratic phase (assumed to be negligible at large z), so there is only
    one FFT and one scalar multiplication. It is slightly faster and more
    numerically stable than Fresnel at large distances.

    For focal-plane computation of a converging beam (e.g. after a lens),
    Fraunhofer is the standard approach: place the input field at the lens,
    set z = focal length, and the output is the focal-plane field.
    """
    _validate_propagator_inputs(E_in, z, wavelength, dx, dy,
                                fn_name='fraunhofer_propagate')
    from ..backend import array_namespace, is_jax_array
    xp = array_namespace(E_in)
    is_jax = is_jax_array(E_in)

    if dy is None:
        dy = dx

    # 4.9 fix (audit #3.3): Fraunhofer is the far-field limit of
    # Fresnel and inherits the same forward-only restriction.  See
    # the matching check in ``fresnel_propagate``.
    if z <= 0:
        raise ValueError(
            f"fraunhofer_propagate: z must be > 0 (got z={z}).  "
            f"Fraunhofer is the far-field limit of Fresnel and is "
            f"forward-only.  Use angular_spectrum_propagate or "
            f"rayleigh_sommerfeld_propagate for back-propagation.")

    Ny, Nx = E_in.shape
    k = 2 * np.pi / wavelength

    # 4.10: honour caller dtype (see fresnel_propagate for rationale).
    if xp.iscomplexobj(E_in):
        target_cdtype = E_in.dtype
    else:
        target_cdtype = np.dtype(DEFAULT_COMPLEX_DTYPE)
    fdtype = np.float64 if np.dtype(target_cdtype) == np.complex128 else np.float32

    # Output grid spacing
    dx_out = wavelength * z / (Nx * dx)
    dy_out = wavelength * z / (Ny * dy)

    # Output coordinates
    x2 = (xp.arange(Nx, dtype=fdtype) - Nx / 2) * dx_out
    y2 = (xp.arange(Ny, dtype=fdtype) - Ny / 2) * dy_out
    X2, Y2 = xp.meshgrid(x2, y2, indexing='xy')

    # Single FFT of the input field
    E_cast = E_in.astype(target_cdtype) if E_in.dtype != target_cdtype else E_in
    if is_jax:
        E_fft = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(E_cast)))
    else:
        E_fft = np.fft.fftshift(_fft2(np.fft.ifftshift(E_cast)))

    # Output quadratic phase + prefactor
    prefactor = (xp.exp(1j * k * z) / (1j * wavelength * z)
                 * xp.exp(1j * k / (2 * z) * (X2**2 + Y2**2))
                 * dx * dy)
    prefactor = prefactor.astype(target_cdtype)

    E_out = prefactor * E_fft

    return E_out, dx_out, dy_out


# ============================================================================
# Fraunhofer propagation onto an arbitrary user-specified output grid (MFT)
# ============================================================================

def fraunhofer_propagate_mft(
    E_in: np.ndarray,
    z: float,
    wavelength: float,
    dx_in: float,
    dx_out: float,
    N_out: int,
    *,
    dy_in: Optional[float] = None,
    dy_out: Optional[float] = None,
    centre_out: Tuple[float, float] = (0.0, 0.0),
    use_gpu: bool = False,
) -> np.ndarray:
    """Fraunhofer (far-field) propagation onto an arbitrary user-specified
    output grid.

    Identical math to :func:`fraunhofer_propagate` (single Fourier transform
    with output quadratic phase + carrier prefactor) except the FT step
    uses a Bluestein chirp-Z transform that samples directly onto the
    chosen output grid.  This is the standard tool for **coronagraph and
    high-contrast imaging codes** -- you can sample the far-field at
    sub-lambda/D resolution around an off-axis stellar PSF without zero-
    padding the input pupil to enormous sizes.

    Differs from :func:`fresnel_propagate_mft` by skipping the input-plane
    quadratic phase ``exp(i*k/(2z)*(X_in^2 + Y_in^2))`` -- it is assumed
    negligible at large z (small Fresnel number).

    Parameters
    ----------
    E_in : ndarray (complex, Ny x Nx)
    z : float
        Propagation distance [m].
    wavelength : float
    dx_in : float
        Input grid spacing [m].
    dx_out : float
        Output grid spacing [m] (independent of ``dx_in`` and ``z``).
    N_out : int
        Output grid size (square).
    dy_in, dy_out : float, optional
    centre_out : (float, float), optional
        Physical ``(x, y)`` centre of the output grid [m].  Defaults to
        ``(0, 0)``.  Use a non-zero value to evaluate the far-field at
        an off-axis point (e.g. an exoplanet location relative to a
        stellar chief image).
    use_gpu : bool, default False

    Returns
    -------
    E_out : ndarray (complex, N_out x N_out)

    See also
    --------
    fraunhofer_propagate : single-FFT Fraunhofer with the natural FFT grid.
    fresnel_propagate_mft : near-field counterpart, includes input-plane
        quadratic phase.
    """
    _validate_propagator_inputs(E_in, z, wavelength, dx_in, dy_in,
                                fn_name='fraunhofer_propagate_mft')
    # 4.9 fix (audit #3.3): forward-only -- see ``fraunhofer_propagate``.
    if z <= 0:
        raise ValueError(
            f"fraunhofer_propagate_mft: z must be > 0 (got z={z}).  "
            f"Fraunhofer-MFT is forward-only (it's the far-field "
            f"limit of fresnel_propagate_mft).  Use "
            f"angular_spectrum_propagate_mft for back-propagation.")
    from ..backend import is_jax_array
    from ._bluestein import _bluestein_centred_2d

    is_jax = is_jax_array(E_in)
    if is_jax:
        import jax.numpy as _jnp
        xp = _jnp
        fft2 = _jnp.fft.fft2
        ifft2 = _jnp.fft.ifft2
    elif CUPY_AVAILABLE and (use_gpu or _is_cupy_array(E_in)):
        if not _ensure_cupy_loaded():
            raise RuntimeError("CuPy requested but failed to load.")
        xp = cp
        fft2 = cp.fft.fft2
        ifft2 = cp.fft.ifft2
        if not _is_cupy_array(E_in):
            E_in = cp.asarray(E_in)
    else:
        xp = np
        fft2 = _fft2
        ifft2 = _ifft2
        if _is_cupy_array(E_in):
            E_in = E_in.get()

    if dy_in is None:
        dy_in = dx_in
    if dy_out is None:
        dy_out = dx_out

    Ny_in, Nx_in = E_in.shape
    Ny_out = Nx_out = int(N_out)

    if xp.iscomplexobj(E_in):
        target_cdtype = E_in.dtype
    else:
        target_cdtype = np.dtype(DEFAULT_COMPLEX_DTYPE)

    k = 2.0 * np.pi / wavelength
    xc, yc = float(centre_out[0]), float(centre_out[1])

    n_x = np.arange(Nx_in, dtype=np.float64)
    n_y = np.arange(Ny_in, dtype=np.float64)
    k_x = np.arange(Nx_out, dtype=np.float64)
    k_y = np.arange(Ny_out, dtype=np.float64)
    x_out = (k_x - Nx_out / 2.0) * dx_out + xc
    y_out = (k_y - Ny_out / 2.0) * dy_out + yc
    X_out_np, Y_out_np = np.meshgrid(x_out, y_out, indexing='xy')

    def _to_xp(arr_np_complex):
        a = arr_np_complex.astype(target_cdtype, copy=False)
        if xp is np:
            return a
        return xp.asarray(a)

    # No input quadratic phase (Fraunhofer assumption).
    # Bluestein FT directly on E_in.
    alpha_x = dx_in * dx_out / (wavelength * z)
    alpha_y = dy_in * dy_out / (wavelength * z)
    kc_x = Nx_out / 2.0 - xc / dx_out
    kc_y = Ny_out / 2.0 - yc / dy_out

    F = _bluestein_centred_2d(
        E_in, alpha_x, alpha_y, Ny_out, Nx_out,
        n_centre_in_x=Nx_in / 2.0,
        n_centre_in_y=Ny_in / 2.0,
        k_centre_out_x=kc_x,
        k_centre_out_y=kc_y,
        sign=-1, xp=xp, fft2=fft2, ifft2=ifft2,
        target_cdtype=target_cdtype,
    )

    quad_out = _to_xp(np.exp(1j * k / (2.0 * z) * (X_out_np**2 + Y_out_np**2)))
    prefactor = (np.exp(1j * k * z) / (1j * wavelength * z)) * dx_in * dy_in
    prefactor_c = target_cdtype.type(prefactor)

    E_out = prefactor_c * quad_out * F
    if E_out.dtype != target_cdtype:
        E_out = E_out.astype(target_cdtype)
    return E_out


# ============================================================================
# Rayleigh-Sommerfeld propagation
# ============================================================================

def rayleigh_sommerfeld_propagate(
    E_in: np.ndarray,
    z: float,
    wavelength: float,
    dx: float,
    dy: Optional[float] = None,
    bandlimit: bool = False,
    use_gpu: bool = False,
    verbose: bool = False,
) -> np.ndarray:
    """
    Propagate an optical field using the Rayleigh-Sommerfeld convolution.

    This method computes the first Rayleigh-Sommerfeld solution by
    convolving the input field with the free-space impulse response
    (Green's function).  Unlike the ASM transfer-function approach,
    the RS convolution constructs the propagation kernel in the
    *spatial* domain and performs the convolution via FFT, which
    naturally captures near-field diffraction effects without the
    band-limiting approximation used in ASM.

    The impulse response is (Goodman *Introduction to Fourier
    Optics*, 3rd ed., eq. 3-43):

        h(x, y, z) = (1 / 2pi) * (z / r^2) * (1/r - ik) * exp(ikr)

    where ``r = sqrt(x^2 + y^2 + z^2)`` and ``k = 2*pi / lambda``.
    Pre-4.10 the kernel used the negated ``(ik - 1/r)`` form, so
    superposing RS with ASM / Fresnel results was 180-degrees out of
    phase.  The docstring formula was updated in 4.11.1 to match the
    corrected code.

    The convolution is computed as::

        E_out = IFFT{ FFT{E_in} * FFT{h} }

    using zero-padded arrays (2N x 2N) to avoid circular convolution
    artifacts.

    Parameters
    ----------
    E_in : ndarray (complex, Ny x Nx)
        Input electric field.
    z : float
        Propagation distance [m].  Positive = forward.
    wavelength : float
        Free-space wavelength [m].
    dx : float
        Grid spacing in x [m].
    dy : float, optional
        Grid spacing in y [m].  Defaults to dx.
    bandlimit : bool, default False
        Apply a Matsushima-style frequency cutoff to the FFT'd kernel
        ``H = FFT(h)``.  Default ``False`` preserves the historical
        "exact Green's function" character of RS that justifies its
        use over ASM in the near field.  Set ``True`` to suppress
        aliasing artifacts on coarse grids at long propagation
        distances (where the kernel chirp under-samples on the
        discrete grid, the same regime where ASM's ``bandlimit=True``
        default is needed).  Cutoff is computed on the padded
        (2N x 2N) grid so the resulting bandwidth budget matches the
        FFT length actually used by the convolution.
    use_gpu : bool, default False
        Use CuPy GPU acceleration if available.
    verbose : bool, default False
        Print diagnostic info.

    Returns
    -------
    E_out : ndarray (complex, Ny x Nx)
        Propagated field (same shape as input).

    Notes
    -----
    **When to use RS instead of ASM:**

    - Near-field propagation (z ~ a few wavelengths) where ASM's
      band-limiting can suppress valid high-frequency content.
    - Validation / cross-check against ASM results.
    - Situations where the exact Green's function is preferred over
      the plane-wave decomposition.

    **Computational cost:** ~4x ASM due to zero-padding (2N FFTs
    instead of N FFTs) and the spatial-domain kernel construction.

    **Memory:** ~6x input array size (padded E, padded h, FFTs).

    At large distances (z >> a^2 / lambda), RS and ASM give identical
    results.  For intermediate distances they agree to machine precision
    when ASM uses no band-limiting (``bandlimit=False``).

    **H caching:** the FFT'd kernel ``H`` is cached on the NumPy backend
    keyed on ``(2*Ny, 2*Nx, dy, dx, wavelength, z, bandlimit, dtype)``.
    Repeat calls at the same geometry skip the kernel build and FFT
    (~30-40% of total RS time on 2k+ grids).  The cache is shared
    with :func:`angular_spectrum_propagate` and obeys the same byte
    budgets configured via :func:`set_asm_cache_size`.  CuPy and JAX
    arrays are kept out of the cache (host-side dict can't safely
    retain device pointers / traced objects); rebuild every call.

    References
    ----------
    [1] Goodman, J.W. "Introduction to Fourier Optics" (3rd ed.),
        Section 3.5: Rayleigh-Sommerfeld Diffraction Theory.
    [2] Shen, F. and Wang, A. (2006). "Fast-Fourier-transform based
        numerical integration method for the Rayleigh-Sommerfeld
        diffraction formula." Appl. Opt. 45(6): 1102-1110.
    [3] Matsushima, K. and Shimobaba, T. (2009). "Band-limited angular
        spectrum method for numerical simulation of free-space
        propagation in far and near fields." Opt. Express 17(22):
        19662-19673.

    Examples
    --------
    >>> import numpy as np
    >>> from lumenairy.propagation import rayleigh_sommerfeld_propagate
    >>>
    >>> N = 512; dx = 1e-6; wv = 0.633e-6
    >>> x = (np.arange(N) - N/2) * dx
    >>> X, Y = np.meshgrid(x, x)
    >>> E_in = (np.sqrt(X**2 + Y**2) < 50e-6).astype(complex)  # circular aperture
    >>>
    >>> E_out = rayleigh_sommerfeld_propagate(E_in, z=1e-3, wavelength=wv, dx=dx)
    """
    # 4.12.0 (audit round-4 B1-3): RS is forward-only.  Pre-4.12 the
    # function accepted z <= 0 silently and computed a 180-degrees-
    # wrong-phase kernel for the back-propagation case.  Match the
    # existing Fresnel / Fraunhofer / SAS guards: hard error with
    # guidance to use ASM / ASM-MFT for back-propagation.
    if z <= 0:
        raise ValueError(
            f"rayleigh_sommerfeld_propagate: z must be > 0 (got "
            f"{z!r}).  RS is forward-only; use "
            f"angular_spectrum_propagate or "
            f"angular_spectrum_propagate_mft for back-propagation "
            f"(those handle the z < 0 case correctly).")
    _validate_propagator_inputs(E_in, z, wavelength, dx, dy,
                                fn_name='rayleigh_sommerfeld_propagate')

    # -- array library selection -----------------------------------------------
    from ..backend import is_jax_array, JAX_AVAILABLE
    is_jax = is_jax_array(E_in)
    if is_jax:
        import jax.numpy as _jnp
        xp = _jnp
    elif CUPY_AVAILABLE and (use_gpu or _is_cupy_array(E_in)):
        xp = cp
        if not _is_cupy_array(E_in):
            E_in = cp.asarray(E_in)
    else:
        xp = np
        if _is_cupy_array(E_in):
            E_in = E_in.get()

    Ny, Nx = E_in.shape
    if dy is None:
        dy = dx

    # Target complex dtype for h, H, and the padded buffer.  Inferred
    # from E_in so the caller controls precision via input dtype.
    # Non-complex input falls back to DEFAULT_COMPLEX_DTYPE to match
    # the standardisation used by angular_spectrum_propagate and the
    # MFT propagator family.
    if xp.iscomplexobj(E_in):
        target_cdtype = E_in.dtype
    else:
        target_cdtype = np.dtype(DEFAULT_COMPLEX_DTYPE)

    k = 2 * np.pi / wavelength

    # -- zero-pad to avoid circular convolution --------------------------------
    Ny2 = 2 * Ny
    Nx2 = 2 * Nx

    # ── H cache (NumPy backend only) ─────────────────────────────────
    # Geometry signature.  Hits return the previously-built H without
    # re-running the kernel construction or its FFT (~30-40% of total
    # RS time on 2k+ grids).  The 'RS' tag keeps RS keys disjoint from
    # ASM keys even when the unpadded grid sizes happen to coincide.
    h_key = None
    H = None
    if xp is np:
        h_key = (int(Ny2), int(Nx2), float(dy), float(dx),
                 float(wavelength), float(z), bool(bandlimit),
                 np.dtype(target_cdtype).str, 'RS')
        H = _h_cache_lookup(h_key)

    cache_hit = H is not None

    if H is None:
        # -- build the RS impulse response h(x, y, z) on the padded grid -------
        # h = -(1/2π) ∂/∂z[exp(ikr)/r]
        #   = (z / (2π r²)) · (1/r − ik) · exp(ikr)         (Goodman 3-43)
        # Pre-4.10 implementation flipped this to (ik − 1/r), producing
        # −h_correct.  Output amplitudes look fine for |E|² consumers but
        # any coherent sum of RS with ASM/Fresnel was 180° out of phase.
        x = (xp.arange(Nx2) - Nx2 / 2) * dx
        y = (xp.arange(Ny2) - Ny2 / 2) * dy
        X, Y = xp.meshgrid(x, y, indexing='xy')
        r = xp.sqrt(X ** 2 + Y ** 2 + z ** 2)
        h = (z / (2 * np.pi * r ** 2)) * xp.exp(1j * k * r) * (1.0 / r - 1j * k)
        h = h * (dx * dy)
        if h.dtype != target_cdtype:
            h = h.astype(target_cdtype)

        if verbose:
            print(f"  RS propagation: z = {z*1e3:.3f} mm  (H cache miss)")
            print(f"  Grid: {Ny}x{Nx} -> padded {Ny2}x{Nx2}")
            print(f"  Wavelength: {wavelength*1e9:.1f} nm")
            try:
                print(f"  Kernel max |h|: {float(xp.max(xp.abs(h))):.4e}")
            except Exception as _exc:
                # xp.max + float() can fail under JAX tracing where h is
                # an abstract array.  Cosmetic print failure -- demote to
                # a brief diagnostic so debug runs surface the cause.
                print(f"  Kernel max |h|: <unavailable: "
                      f"{type(_exc).__name__}>")

        # -- FFT the kernel ----------------------------------------------------
        # The result is cached via _h_cache_store below and reused
        # across many subsequent _fft2/_ifft2 calls; under the 4.12
        # double-buffer contract on _fft2, we must take an explicit
        # copy here so the cached H survives the third subsequent
        # call at this shape (which would recycle the slot).  The
        # bandlimit branch below already produces a fresh array via
        # H * mask_c, but the non-bandlimit path would otherwise
        # alias the plan workspace; copy unconditionally for clarity.
        if is_jax:
            H = xp.fft.fft2(xp.fft.ifftshift(h))
        elif xp is np:
            H = _fft2(np.fft.ifftshift(h)).copy()
        else:
            H = xp.fft.fft2(xp.fft.ifftshift(h))

        # -- Matsushima-style bandlimit on the padded H ------------------------
        # Cutoff matches the ASM derivation but uses the padded extent
        # (Lx2 = 2*Nx*dx) since the FFT length is what determines the
        # discrete frequency support.  The mask is built in centred
        # order then ifftshifted to align with H's DC-at-corner layout.
        if bandlimit and z != 0:
            fx = (np.arange(Nx2) - Nx2 / 2) / (Nx2 * dx)
            fy = (np.arange(Ny2) - Ny2 / 2) / (Ny2 * dy)
            Lx2 = Nx2 * dx
            Ly2 = Ny2 * dy
            fx_max = Lx2 / (2 * wavelength * abs(z))
            fy_max = Ly2 / (2 * wavelength * abs(z))
            bl_x = np.abs(fx) < fx_max
            bl_y = np.abs(fy) < fy_max
            mask = (bl_y[:, None] & bl_x[None, :])
            mask = np.fft.ifftshift(mask)
            mask_c = mask.astype(target_cdtype)
            if xp is np:
                H = H * mask_c
            else:
                H = H * xp.asarray(mask_c)
            if verbose:
                kept_frac = float(np.mean(mask))
                print(f"  Bandlimit: keeping {kept_frac*100:.1f}% of "
                      f"padded spectrum")

        # Store under the NumPy key only.  See _h_cache_store for the
        # byte-budget eviction policy.  Cached H is used read-only.
        if h_key is not None:
            _h_cache_store(h_key, H)
    elif verbose:
        print(f"  RS propagation: z = {z*1e3:.3f} mm  (H cache HIT)")

    # -- build the padded input field -----------------------------------------
    if is_jax:
        # JAX is functional / immutable -- can't write into a pre-allocated
        # array.  Build the padded array via jnp.zeros + at[].set.
        E_padded = xp.zeros((Ny2, Nx2), dtype=target_cdtype)
        y0 = Ny // 2
        x0 = Nx // 2
        E_padded = E_padded.at[y0:y0 + Ny, x0:x0 + Nx].set(E_in)
    else:
        E_padded = xp.zeros((Ny2, Nx2), dtype=target_cdtype)
        y0 = Ny // 2
        x0 = Nx // 2
        E_padded[y0:y0 + Ny, x0:x0 + Nx] = E_in

    # -- convolve via FFT ------------------------------------------------------
    if is_jax:
        E_fft = xp.fft.fft2(E_padded)
        E_conv = xp.fft.ifft2(E_fft * H)
    elif xp is np:
        E_fft = _fft2(E_padded)
        E_conv = _ifft2(E_fft * H)
    else:
        E_fft = xp.fft.fft2(E_padded)
        E_conv = xp.fft.ifft2(E_fft * H)

    # -- extract the valid region (same location as input was placed) ----------
    E_out = E_conv[y0:y0 + Ny, x0:x0 + Nx]

    return E_out


# =============================================================================
# SCALABLE ANGULAR SPECTRUM (SAS) — Heintzmann / Loetgering / Wechsler, 2023
# =============================================================================


def scalable_angular_spectrum_propagate(
    E_in: np.ndarray,
    z: float,
    wavelength: float,
    dx: float,
    pad: int = 2,
    skip_final_phase: bool = False,
    use_gpu: bool = False,
    verbose: bool = False,
) -> Tuple[np.ndarray, float, float]:
    """
    Scalable-angular-spectrum propagator with variable output pitch.

    Implements the Heintzmann-Loetgering-Wechsler 2023 three-FFT kernel:
    apply an ASM-minus-Fresnel precompensation phase in the spatial-frequency
    domain, then a Fresnel chirp + single FFT (+ optional final quadratic
    phase).  The output grid has pixel pitch
    ``dx_out = lambda * z / (pad * N * dx)`` which can be much larger than
    the input pitch, letting one propagate over distances where a standard
    angular-spectrum call would need impractically many samples to span the
    geometric cone of the beam.

    The kernel is exact up to the ASM-vs-Fresnel band-limit cutoff ``W``
    baked into the precompensation; beyond that cutoff the method gracefully
    reduces to a zeroed transfer function (high-NA components are dropped).
    A closed-form ``z_limit`` from the paper bounds the propagation distance
    for which the method remains valid at the input sampling; we warn (not
    raise) when ``z > z_limit`` so the caller can still experiment.

    Parameters
    ----------
    E_in : ndarray (complex, N x N)
        Input field.  Must be square (Ny == Nx = N).  NumPy or CuPy.

    z : float
        Propagation distance [m].  Positive = forward.

    wavelength : float
        Wavelength [m].

    dx : float
        Input grid pitch [m].  Input extent is ``L = N * dx``.

    pad : int, default 2
        Zero-padding factor applied before the SAS kernel.  The reference
        implementation uses 2; larger values reduce aliasing further at the
        cost of more compute.  Output is cropped back to ``N x N`` after
        the kernel runs.

    skip_final_phase : bool, default False
        If True, skip the final post-FFT quadratic phase.  The resulting
        complex field has the correct *intensity* but not the correct phase
        at the output plane.  Equivalent to the paper's
        ``skip_final_phase=True`` mode; cheaper by one N^2 multiply.

    use_gpu : bool, default False
        Run on CuPy when available.  Like the other propagators, if
        ``E_in`` is already a CuPy array this is honoured automatically.

    verbose : bool, default False
        Print grid, pitch, and band-limit diagnostics.

    Returns
    -------
    E_out : ndarray (complex, N x N)
        Propagated field on a grid of pitch ``dx_out``.

    dx_out : float
        Output grid pitch = ``wavelength * z / (pad * N * dx)``.

    dy_out : float
        Output grid pitch in y (equal to ``dx_out`` for square input).

    Notes
    -----
    Choice of propagator by regime (for free-space diffraction of an N x N
    field, extent L, wavelength lam, distance z):

    * ``z << L^2 / (N * lam)``  — use :func:`angular_spectrum_propagate`
      (Fresnel number large, output pitch = input pitch).
    * ``z ~ L^2 / (N * lam)``   — either ASM or SAS work; SAS gives a
      better-scaled output grid if the beam has diverged past the input
      window.
    * ``z >> L^2 / (N * lam)``  — use SAS.  Plain ASM needs a much larger
      N to avoid aliasing; pure Fresnel loses the phase accuracy that SAS
      recovers through its precompensation term.
    * ``z -> infinity``         — :func:`fraunhofer_propagate`.

    Sampling assumption: the input field is centred in its array and the
    returned array is centred (fftshift applied to the SAS output).  This
    differs from the reference notebook which returns FFT-natural order.

    References
    ----------
    [1] Heintzmann, R.; Loetgering, L.; Wechsler, F. (2023).  "Scalable
        angular spectrum propagation".  *Optica* 10(11): 1407-1416.
        doi:10.1364/OPTICA.497809
    [2] Reference PyTorch implementation:
        https://github.com/bionanoimaging/Scalable-Angular-Spectrum-Method-SAS
    """
    _validate_propagator_inputs(E_in, z, wavelength, dx, None,
                                fn_name='scalable_angular_spectrum_propagate')
    # 4.9 fix (audit #3.3): SAS's exp(j·k·z·(h_AS - h_Fr)) precompensation
    # factor can give exp(+real) blow-up for z < 0 with evanescent
    # components, so back-propagation is not supported.
    if z <= 0:
        raise ValueError(
            f"scalable_angular_spectrum_propagate: z must be > 0 "
            f"(got z={z}).  SAS is a forward-only propagator (the "
            f"precompensation phase isn't sign-symmetric).  Use "
            f"angular_spectrum_propagate or rayleigh_sommerfeld_propagate "
            f"for back-propagation.")
    # -- array library selection (NumPy vs. CuPy) ---------------------------
    if CUPY_AVAILABLE and (use_gpu or _is_cupy_array(E_in)):
        xp = cp
        if not _is_cupy_array(E_in):
            E_in = cp.asarray(E_in)
    else:
        xp = np
        if _is_cupy_array(E_in):
            E_in = E_in.get()

    # -- validate input ------------------------------------------------------
    if E_in.ndim != 2:
        raise ValueError(
            f"scalable_angular_spectrum_propagate: expected 2-D field, "
            f"got shape {E_in.shape}.")
    Ny, Nx = E_in.shape
    if Ny != Nx:
        raise ValueError(
            f"scalable_angular_spectrum_propagate: input must be square "
            f"(got {Ny}x{Nx}).  The SAS kernel is derived for a single N.")
    N = Nx
    L = N * dx
    pad = int(pad)
    if pad < 1:
        raise ValueError(f"pad must be >= 1, got {pad}")

    # -- closed-form z-limit from Heintzmann et al. (2023) -------------------
    # Beyond z_limit the band-limit filter W kills the ASM-like components
    # that the precompensation phase is meant to correct, and SAS reduces to
    # plain Fresnel with the usual far-field error.
    lam = wavelength
    s = L ** 2 / (8 * L ** 2 + N ** 2 * lam ** 2)
    denom = lam * (-1.0 + 2.0 * np.sqrt(2.0) * np.sqrt(s))
    if abs(denom) < 1e-30:
        z_limit = float("inf")
    else:
        z_limit = float(
            -4.0 * L * np.sqrt(8.0 * L ** 2 / N ** 2 + lam ** 2)
            * np.sqrt(s) / denom)
    if z > z_limit > 0 and verbose:
        print(f"  SAS: z = {z*1e3:.2f} mm exceeds z_limit = "
              f"{z_limit*1e3:.2f} mm; accuracy may degrade.")

    # -- padded grid ---------------------------------------------------------
    L_new = pad * L
    N_new = pad * N

    # -- choose precision from input dtype (matches angular_spectrum_propagate)
    if xp.iscomplexobj(E_in):
        target_cdtype = E_in.dtype
    else:
        target_cdtype = np.dtype(DEFAULT_COMPLEX_DTYPE)
    target_fdtype = (np.float32
                     if target_cdtype == np.complex64 else np.float64)

    # -- zero-pad the input, centred ----------------------------------------
    # 4.12.0 (audit round-4 B1-5): `as1 = (N + 1) // 2` was only
    # correct for pad=2 (then `N_new = 2*N` and `(N_new - N)//2 = N/2`
    # ≈ `(N+1)//2`).  For pad=4 with N=512, `(N+1)//2 = 256` but
    # `N_new/2 = 1024` -- input ends up off-centre by ~N/4 pixels.
    # The correct centring is `(N_new - N) // 2`.
    psi_p = xp.zeros((N_new, N_new), dtype=target_cdtype)
    as1 = (N_new - N) // 2
    psi_p[as1:as1 + N, as1:as1 + N] = E_in

    # ── Kernel cache (NumPy backend) ────────────────────────────────
    # SAS builds three padded-grid kernels per call: the ASM-minus-Fresnel
    # precompensation ``delta_H``, the Fresnel input chirp ``H1``, and
    # (when skip_final_phase is False) the output-plane quadratic phase
    # ``H2``.  All three depend only on (N_new, dx, lam, z,
    # skip_final_phase, dtype), so we bundle them under a single cache
    # entry keyed on that signature.  Cached as a tuple
    # ``(delta_H, H1, H2_or_None)``; _entry_bytes accounts for tuple
    # bundles in the H-cache byte budget.
    h_key = None
    cached = None
    if xp is np:
        h_key = (int(N_new), float(dx), float(lam), float(z),
                 bool(skip_final_phase),
                 np.dtype(target_cdtype).str, 'SAS')
        cached = _h_cache_lookup(h_key)

    if cached is None:
        # -- spatial-frequency axes (natural FFT order) -------------------
        #   fftfreq(N_new, d=L_new/N_new) = fftfreq(N_new, d=dx)
        f_x = xp.fft.fftfreq(N_new, d=dx).astype(target_fdtype)
        f_y = f_x  # square grid

        # -- band-limit W: ASM-vs-Fresnel validity region -----------------
        # Paper eq. (12): the precompensation is valid wherever both
        # inequalities hold, else drop the mode.
        two_z = 2.0 * z if z != 0 else 1e-30
        cx = lam * f_x[None, :]
        cy = lam * f_y[:, None]
        tx = L_new / two_z + xp.abs(cx)
        ty = L_new / two_z + xp.abs(cy)
        W = ((cx ** 2 * (1.0 + tx ** 2) / tx ** 2 + cy ** 2 <= 1.0)
             & (cy ** 2 * (1.0 + ty ** 2) / ty ** 2 + cx ** 2 <= 1.0))

        # -- ASM-minus-Fresnel precompensation phase ----------------------
        #   H_AS  = sqrt(1 - (lam*fx)^2 - (lam*fy)^2)
        #   H_Fr  = 1 - ((lam*fx)^2 + (lam*fy)^2) / 2
        #   delta_H = W * exp( i * k * z * (H_AS - H_Fr) )
        k = 2 * np.pi / lam
        h_AS = xp.sqrt((1.0 + 0j) - cx ** 2 - cy ** 2)
        h_Fr = 1.0 - 0.5 * (cx ** 2 + cy ** 2)
        delta_H = W * xp.exp(1j * k * z * (h_AS - h_Fr))
        delta_H = delta_H.astype(target_cdtype, copy=False)

        # -- Fresnel chirp on natural-order grid --------------------------
        coord_centred = xp.linspace(
            -L_new / 2, L_new / 2, N_new, endpoint=False,
            dtype=target_fdtype)
        coord_nat = xp.fft.ifftshift(coord_centred)
        x = coord_nat[None, :]
        y = coord_nat[:, None]
        H1 = xp.exp(1j * k / (2.0 * z) * (x ** 2 + y ** 2))
        if H1.dtype != target_cdtype:
            H1 = H1.astype(target_cdtype, copy=False)

        # -- output-plane quadratic phase (optional) ----------------------
        if skip_final_phase:
            H2 = None
        else:
            dq = lam * z / L_new  # output pitch on padded grid
            Q = dq * N_new        # full extent of padded output grid
            q_centred = xp.linspace(
                -Q / 2, Q / 2, N_new, endpoint=False, dtype=target_fdtype)
            q_nat = xp.fft.ifftshift(q_centred)
            qx = q_nat[None, :]
            qy = q_nat[:, None]
            H2 = xp.exp(1j * k * z) * xp.exp(
                1j * k / (2.0 * z) * (qx ** 2 + qy ** 2))
            if H2.dtype != target_cdtype:
                H2 = H2.astype(target_cdtype, copy=False)

        if h_key is not None:
            _h_cache_store(h_key, (delta_H, H1, H2))
    else:
        delta_H, H1, H2 = cached

    # -- apply precompensation in frequency space ---------------------------
    # The reference uses ifftshift(psi_p) then fft2, i.e. treat the centred
    # array as if its zero-pixel is at the centre.  Match exactly.
    psi_precomp = xp.fft.ifft2(
        xp.fft.fft2(xp.fft.ifftshift(psi_p)) * delta_H)

    # -- Fresnel chirp + single FFT -----------------------------------------
    # Fresnel-style amplitude prefactor so the output is the physical
    # diffracted field (not a raw DFT sample).  This matches the
    # normalization used by fresnel_propagate in this library and is
    # absent from the reference PyTorch notebook.
    amp_pref = (dx * dx) / (1j * lam * z)
    if skip_final_phase:
        psi_p_final = amp_pref * xp.fft.fftshift(
            xp.fft.fft2(H1 * psi_precomp))
    else:
        psi_p_final = amp_pref * xp.fft.fftshift(
            H2 * xp.fft.fft2(H1 * psi_precomp))

    # -- crop back to original N x N ----------------------------------------
    E_out = psi_p_final[as1:as1 + N, as1:as1 + N]

    # -- output grid pitch ---------------------------------------------------
    dx_out = lam * z / (pad * N * dx)

    if verbose:
        print(f"  SAS propagation: z = {z*1e3:.3f} mm")
        print(f"  Input  grid: {N}x{N}  pitch {dx*1e6:.3f} um  "
              f"extent {L*1e3:.3f} mm")
        print(f"  Output grid: {N}x{N}  pitch {dx_out*1e6:.3f} um  "
              f"extent {N*dx_out*1e3:.3f} mm  "
              f"(zoom {dx_out/dx:.2f}x)")
        # delta_H = W * exp(...) is zero exactly where W is, so the
        # nonzero count of delta_H reproduces W's count without needing
        # W itself (which is now scoped inside the cache-miss branch).
        kept = float(xp.sum(delta_H != 0)) / (N_new * N_new)
        print(f"  Band-limit kept: {kept*100:.1f}% of SAS spectrum")
        if z_limit > 0:
            print(f"  z_limit from paper: {z_limit*1e3:.2f} mm")

    return E_out, dx_out, dx_out
