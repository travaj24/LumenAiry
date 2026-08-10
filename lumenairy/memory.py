"""
Runtime resource helpers (memory + CPU).
=========================================

Two responsibilities, both system-resource queries:

1. **Memory-aware batching.**  Estimate the memory cost of an
   operation and decide whether to run it straight-through or
   split it into batches that fit available RAM::

       cost = estimate_op_memory(shape, dtype, n_work_arrays)
       batch = pick_batch_size(n_items, cost_per_item, safety=0.5)

2. **Affinity-aware CPU count.**  :func:`available_cpus` returns
   "what this process can actually use" rather than the raw
   ``os.cpu_count()`` -- respects ``taskset`` / cgroup limits /
   ``os.process_cpu_count()`` (Python 3.13+) / Windows process
   affinity.  Worker pools, the pyFFTW thread setting, and any
   future thread / process dispatch should call this rather than
   ``os.cpu_count()`` directly.

The helpers degrade gracefully: if :mod:`psutil` is not installed
or the OS does not expose memory info, they fall back to
conservative defaults and a warning is emitted.

Author: Andrew Traverso
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Dict, Literal, Optional, Tuple, Union, overload

import numpy as np

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Fallback defaults used when psutil is not available
# ---------------------------------------------------------------------------
_DEFAULT_AVAILABLE_BYTES = 4 * 1024**3   # 4 GB
_DEFAULT_TOTAL_BYTES     = 8 * 1024**3   # 8 GB

# ---------------------------------------------------------------------------
# RAM budget configuration
#
# The library auto-detects available system RAM at runtime and uses it
# to decide between fast-but-memory-hungry code paths and lean-but-slower
# alternatives (e.g. chunked transfer-function construction in ASM).
#
# Users can override the auto-detected budget with set_max_ram() for
# situations where psutil is unavailable, the available-memory query is
# inaccurate (e.g. shared HPC nodes with cgroups), or they want to
# reserve headroom for other processes.
# ---------------------------------------------------------------------------
_MAX_RAM_OVERRIDE = None   # None = auto-detect via psutil


def get_ram_budget() -> int:
    """
    Return the effective RAM budget in bytes.

    If :func:`set_max_ram` was called with a value, returns that value.
    Otherwise returns the currently available physical memory via psutil
    (or the 4 GB fallback when psutil is not installed).
    """
    if _MAX_RAM_OVERRIDE is not None:
        return _MAX_RAM_OVERRIDE
    return available_memory_bytes()


def set_max_ram(value: Optional[Union[int, float]]) -> None:
    """
    Set a manual RAM budget override for the library.

    All memory-aware code paths in the library (e.g. ASM propagation,
    batch-size selection) consult :func:`get_ram_budget` before deciding
    whether to spill temporaries to disk or use lean in-memory
    algorithms.  This function lets you pin that budget to a fixed
    value instead of relying on the auto-detected available memory.

    Parameters
    ----------
    value : float, int, or None
        - If ``< 1024``: treated as **gigabytes** (e.g. ``16`` = 16 GB).
        - If ``>= 1024``: treated as **bytes**.
        - If ``None``: revert to auto-detection (the default).

    Examples
    --------
    >>> from lumenairy import set_max_ram
    >>> set_max_ram(16)         # 16 GB budget
    >>> set_max_ram(64 * 1e9)   # 64 GB budget (as bytes)
    >>> set_max_ram(None)        # auto-detect
    """
    global _MAX_RAM_OVERRIDE
    if value is None:
        _MAX_RAM_OVERRIDE = None
        return
    # v4.14 (audit P3 #18): reject negative budgets explicitly.  Pre-
    # v4.14 a negative value was silently accepted (treated as
    # negative bytes); ``pick_batch_size`` then clamped via
    # ``min_batch=1`` so the bug only surfaced as quiet single-batch
    # processing on huge workloads.  Zero is also nonsensical (no
    # work could ever fit) so reject it too.
    if value <= 0:
        raise ValueError(
            f"set_max_ram: value must be positive (got {value!r}). "
            f"Use a positive number of GB (< 1024) or a positive "
            f"byte count (>= 1024), or pass None to revert to "
            f"auto-detection.")
    if value < 1024:
        _MAX_RAM_OVERRIDE = int(value * 1024**3)
    else:
        _MAX_RAM_OVERRIDE = int(value)


def get_max_ram() -> Optional[int]:
    """Return the current manual RAM-budget override in bytes, or
    ``None`` if no override is in effect (psutil auto-detection
    applies).

    Mirrors :func:`set_max_ram` exactly: round-trips through
    ``set_max_ram(get_max_ram())`` are a no-op.  Use
    :func:`get_ram_budget` instead when you want the *effective*
    budget (override or auto-detected value).  Introduced in 4.8.1
    to support :func:`lumenairy.lumenairy_context`.
    """
    return _MAX_RAM_OVERRIDE


# ---------------------------------------------------------------------------
# Memory queries
# ---------------------------------------------------------------------------
def available_memory_bytes() -> int:
    """
    Return the currently available physical memory in bytes.

    Uses :mod:`psutil.virtual_memory().available` when available.
    Falls back to a conservative 4 GB default with a warning.
    """
    if _PSUTIL_AVAILABLE:
        return int(psutil.virtual_memory().available)
    warnings.warn(
        "psutil not installed — assuming 4 GB available memory. "
        "Install psutil for accurate memory-aware batching.",
        RuntimeWarning,
    )
    return _DEFAULT_AVAILABLE_BYTES


def total_memory_bytes() -> int:
    """Return the total physical memory in bytes."""
    if _PSUTIL_AVAILABLE:
        return int(psutil.virtual_memory().total)
    return _DEFAULT_TOTAL_BYTES


def memory_info() -> Dict[str, Any]:
    """
    Return a dictionary with current memory statistics.

    Returns
    -------
    info : dict
        Keys: 'available', 'total', 'used', 'percent_used', 'available_gb',
        'total_gb', 'has_psutil'.
    """
    if _PSUTIL_AVAILABLE:
        vm = psutil.virtual_memory()
        return {
            'available':   int(vm.available),
            'total':       int(vm.total),
            'used':        int(vm.used),
            'percent_used': float(vm.percent),
            'available_gb': vm.available / (1024**3),
            'total_gb':    vm.total / (1024**3),
            'has_psutil':  True,
        }
    return {
        'available':    _DEFAULT_AVAILABLE_BYTES,
        'total':        _DEFAULT_TOTAL_BYTES,
        'used':         _DEFAULT_TOTAL_BYTES - _DEFAULT_AVAILABLE_BYTES,
        'percent_used': 50.0,
        'available_gb': _DEFAULT_AVAILABLE_BYTES / (1024**3),
        'total_gb':     _DEFAULT_TOTAL_BYTES / (1024**3),
        'has_psutil':   False,
    }


# ---------------------------------------------------------------------------
# Array memory estimation
# ---------------------------------------------------------------------------
def bytes_per_element(dtype: Any) -> int:
    """
    Return the number of bytes per element for a numpy dtype.

    Accepts either a dtype object, a dtype string ('complex128'),
    or a Python type (complex, float, int).
    """
    return int(np.dtype(dtype).itemsize)


def array_bytes(shape: Union[int, Tuple[int, ...]],
                dtype: Any = 'complex128') -> int:
    """
    Estimate the memory footprint of a numpy array of the given shape.

    Parameters
    ----------
    shape : int or tuple of ints
        Array shape.
    dtype : numpy dtype or str
        Element type.  Default: ``'complex128'`` (16 bytes/element).

    Returns
    -------
    nbytes : int
    """
    if isinstance(shape, (int, np.integer)):
        n = int(shape)
    else:
        n = int(np.prod(shape))
    return n * bytes_per_element(dtype)


def estimate_op_memory(shape: Union[int, Tuple[int, ...]],
                       dtype: Any = 'complex128',
                       n_work_arrays: int = 3,
                       extra_bytes: int = 0) -> int:
    """
    Estimate the peak memory cost of a typical operation on an array.

    Most library operations allocate:
    - the input array (already in memory, not counted here)
    - a few temporary working arrays (FFT buffer, phase mask, output, ...)
    - some small constant overhead

    Parameters
    ----------
    shape : int or tuple of ints
        Primary array shape.
    dtype : numpy dtype or str
        Primary element type.  Default ``'complex128'``.
    n_work_arrays : int
        Number of temporary arrays of the same shape and dtype.
        Typical values:
        - ASM propagation: 3 (fft input, transfer function, fft output)
        - Element-wise mask (lens phase, aperture): 2 (mask + output)
        - Per-source synthesis: 1 (accumulator)
    extra_bytes : int
        Additional constant overhead (e.g. small auxiliary arrays).

    Returns
    -------
    nbytes : int
        Peak additional memory required.
    """
    per_array = array_bytes(shape, dtype)
    return n_work_arrays * per_array + int(extra_bytes)


# ---------------------------------------------------------------------------
# Batch-size selection
# ---------------------------------------------------------------------------
def pick_batch_size(n_items: int, cost_per_item: int,
                    available: Optional[int] = None,
                    safety: float = 0.5,
                    min_batch: int = 1,
                    max_batch: Optional[int] = None) -> int:
    """
    Choose the largest batch size that comfortably fits in available RAM.

    Given a workload of ``n_items`` whose per-item memory cost is
    ``cost_per_item`` bytes, return the largest batch size ``k`` such
    that ``k * cost_per_item`` stays under ``safety * available``.

    Parameters
    ----------
    n_items : int
        Total number of items to process.
    cost_per_item : int
        Memory cost of ONE item in bytes.  ``0`` means "free" (the whole
        workload fits in one batch); a NEGATIVE cost is rejected (audit
        A-9..A-14 -- pre-v5.29.1 it was silently treated as free, so a
        sign-flipped or subtracted-in-the-wrong-order caller got the
        maximum batch size and OOMed instead of being told).
    available : int or None
        Available memory in bytes.  If ``None``, uses
        :func:`get_ram_budget` (the :func:`set_max_ram` override when
        set, else the auto-detected available memory) -- audit P2-21:
        pre-v5.17.2 this bypassed the override via
        :func:`available_memory_bytes`.
    safety : float
        Fraction of available memory to use.  Default 0.5 leaves
        half of the available RAM for other processes and OS overhead.
    min_batch : int
        Minimum batch size; never return less than this.
    max_batch : int or None
        Optional upper bound on the batch size.

    Returns
    -------
    batch : int
        Recommended batch size, clamped to ``[min_batch, n_items]`` (and
        to ``max_batch`` if given).

    Examples
    --------
    >>> # Processing 144 sources, each needs (8192, 8192) complex128 = 1 GB
    >>> cost = array_bytes((8192, 8192), 'complex128')
    >>> batch = pick_batch_size(144, cost)
    >>> print(f'Batch size: {batch}')  # e.g. 12 on a 24 GB machine
    """
    if available is None:
        available = get_ram_budget()

    if cost_per_item < 0:
        raise ValueError(
            f"pick_batch_size: cost_per_item must be >= 0 bytes (got "
            f"{cost_per_item!r}).  A negative memory cost is meaningless; "
            f"pass 0 if the per-item cost really is negligible.")
    if cost_per_item == 0:
        return max(min_batch, n_items)

    budget = int(available * safety)
    k = max(1, budget // cost_per_item)
    k = min(k, n_items)
    k = max(k, min_batch)
    if max_batch is not None:
        k = min(k, max_batch)
    return int(k)


def should_split(total_cost: int,
                 available: Optional[int] = None,
                 safety: float = 0.5) -> bool:
    """
    Decide whether an operation needs to be split into batches.

    Parameters
    ----------
    total_cost : int
        Estimated memory cost of running the whole operation at once,
        in bytes.  Must be >= 0: a negative cost is rejected rather than
        silently answering "no split needed" (audit A-9..A-14).
    available : int or None
        Available memory in bytes.  Defaults to :func:`get_ram_budget`
        (honours a :func:`set_max_ram` override -- audit P2-21).
    safety : float
        Fraction of available memory considered "safe" to use.

    Returns
    -------
    split : bool
        True if ``total_cost`` exceeds ``safety * available``.

    Raises
    ------
    ValueError
        If ``total_cost`` is negative.
    """
    if available is None:
        available = get_ram_budget()
    if total_cost < 0:
        raise ValueError(
            f"should_split: total_cost must be >= 0 bytes (got "
            f"{total_cost!r}).  A negative memory cost is meaningless.")
    return total_cost > int(available * safety)


# ---------------------------------------------------------------------------
# Pretty-printing helpers
# ---------------------------------------------------------------------------
def format_bytes(nbytes: Union[int, float]) -> str:
    """
    Format a byte count as a human-readable string.

    >>> format_bytes(1536)
    '1.5 KB'
    >>> format_bytes(1073741824)
    '1.00 GB'
    """
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    x = float(nbytes)
    for u in units:
        if abs(x) < 1024.0:
            return f'{x:.2f} {u}' if u != 'B' else f'{int(x)} B'
        x /= 1024.0
    return f'{x:.2f} PB'


def print_memory_report(planned_cost_bytes: Optional[int] = None,
                        prefix: str = '') -> None:
    """
    Print a one-line human-readable memory status report.

    Parameters
    ----------
    planned_cost_bytes : int or None
        If given, include the planned operation cost and whether it fits.
    prefix : str
        Optional prefix string (e.g. for indentation).
    """
    info = memory_info()
    avail = format_bytes(info['available'])
    total = format_bytes(info['total'])
    msg = (f"{prefix}Memory: {avail} free / {total} total "
           f"({info['percent_used']:.0f}% used)")
    if planned_cost_bytes is not None:
        cost = format_bytes(planned_cost_bytes)
        fits = not should_split(planned_cost_bytes, info['available'])
        status = 'OK' if fits else 'SPLIT needed'
        msg += f" — planned op: {cost} [{status}]"
    print(msg, flush=True)


# ============================================================================
# CPU-count helper (affinity / cgroup aware)
# ============================================================================

def available_cpus() -> int:
    """Return the number of CPUs this process can actually use.

    Preference order:

    1. ``os.process_cpu_count()`` (Python 3.13+): the canonical
       "CPUs available to this process" number, respects CPU-quota
       cgroups and affinity masks.
    2. ``len(os.sched_getaffinity(0))`` (Linux / BSD): honours
       ``taskset`` restrictions.
    3. ``len(psutil.Process().cpu_affinity())`` (optional cross-
       platform path, captures Windows process affinity).
    4. ``os.cpu_count()`` fallback -- the raw logical-CPU count,
       used only when nothing above is available.

    Always returns at least 1.
    """
    if hasattr(os, 'process_cpu_count'):
        try:
            n = os.process_cpu_count()
            if n:
                return int(n)
        except (OSError, AttributeError, NotImplementedError):
            # Python < 3.13 may have the attribute as a stub that
            # raises NotImplementedError on some platforms.
            pass

    if hasattr(os, 'sched_getaffinity'):
        try:
            n = len(os.sched_getaffinity(0))
            if n > 0:
                return int(n)
        except (OSError, AttributeError, NotImplementedError):
            # sched_getaffinity is POSIX-only; missing on Windows /
            # macOS where the hasattr check still passes via shim.
            pass

    try:
        import psutil
        n = len(psutil.Process().cpu_affinity())
        if n > 0:
            return int(n)
    except (ImportError, AttributeError, OSError,
            NotImplementedError) as _exc:
        # psutil missing or cpu_affinity unsupported (macOS).
        pass

    return max(1, int(os.cpu_count() or 1))


# ============================================================================
# System-level memory estimation + autodetect guardrail (v5.16.1)
# ----------------------------------------------------------------------------
# ``estimate_op_memory`` above models ONE elementwise/FFT operation as
# ``n_work_arrays`` same-dtype temporaries.  That is accurate for a bare ASM
# propagation but badly under-counts a full free-space + ray-traced-lens
# simulation, whose PEAK step is the lens amplitude pass: a stack of
# FLOAT64-FIXED full-grid arrays (the coordinate lineage seeded by
# ``np.arange(N)*dx`` with no ``dtype=`` -> float64, then sag/opd, the
# ``np.indices((N,N))`` + ``(2,N,N)`` map_coordinates upsample stack, and
# delta_phase) that does NOT shrink with complex64.  The helpers below model
# that peak so callers can (a) get an itemised estimate and (b) fail FAST with
# an actionable message + concrete claw-backs instead of OOMing mid-run.
#
# CALIBRATION (v5.17.1, POST-lifetime-fix; measured system-wide peak-used
# deltas on a quiet 137 GB / 24-core box, apply_real_lens_traced on a real
# 2-surface relay lens, parallel_amp=False, plan cache=1, auto-promote off).
# Whole-grid anchors at N=16384 (fit the 3-parameter model EXACTLY):
#     c64  sub=8  -> 29.69 GB | c64 sub=16 -> 21.87 GB | c128 sub=8 -> 31.39 GB
# Chunked anchor: c128 sub=16 -> 26.30 GB (the c64-chunked reading is
# unreliable in a shared process -- persistent FFT plan buffers cross-
# contaminate the delta -- so the chunked constant is calibrated from the
# LARGER c128 point: conservative, i.e. the fail-safe direction).
# Pre-v5.17.0 anchors (44.5/37.2/57.3 GB) reflected the since-fixed v4.10
# tilt-check leak + upsample double-build and are obsolete.
# ============================================================================

# Effective live full-grid array counts (per N^2). DTYPE-INDEPENDENT (float64).
_LENS_F64_ARRAYS = 6.2            # coord lineage + sag/opd transients + opl_map/upsample (post-fix)
_LENS_SLANT_F64_ARRAYS = 6.0     # extra angle stack (dsag_dx/dy, grad_sq, cos_ti/tt, opd) when slant/fresnel
_LENS_COMPLEX_ARRAYS = 0.8       # resident complex set after the v5.17.0 eager frees
_NEWTON_BYTES_PER_COARSE_PT = 2490.0   # coarse-grid Newton solve + poly fit + map_coordinates, per (N/sub)^2 pt
# Bare ASM step (audit A-6, RE-DERIVED 2026-07-25 from fresh-interpreter
# tracemalloc profiles of ``angular_spectrum_propagate`` at N=64..2048 in
# complex64 and complex128; see :func:`estimate_asm_memory`).  The measured
# allocation profile separates into three terms that each fit to <1%:
#
#   * 2 complex full-grid arrays -- the cached transfer function H and the
#     returned output field.  (The steady-state per-call peak, with H and the
#     pyFFTW plans already resident, measures 1.00x N^2*itemsize exactly: the
#     output field alone.)
#   * ~0.63 FLOAT64 full-grid arrays -- the fx/fy/kz frequency-grid
#     transients, which do NOT shrink with complex64 (visible only in the
#     complex64 readings, where the shape term is 6.63x itemsize vs 6.00x for
#     complex128).  Rounded UP to 0.7 so the complex128 case is bounded too.
#   * 2 aligned pyFFTW workspace buffers per resident plan key
#     (``plan_cache_keys``), added separately below.
#
# The pre-A-6 constant was 4.0 with no float64 or fixed term, giving
# est/measured 0.53 (N=512) / 0.96 (1024) / 1.22 (2048) -- neither a bound
# nor a steady-state figure.
_ASM_COMPLEX_ARRAYS = 2.0
_ASM_F64_GRID_ARRAYS = 0.7       # dtype-independent frequency-grid transients
# One-time, N-INDEPENDENT cost of the first ASM call in a fresh process: the
# lazy import of the FFT backend (pyFFTW / scipy.fft) and its plan
# infrastructure.  This term is what made the pre-A-6 estimate a 0.53x
# UNDER-estimate at N=512 -- at small N it dominates.
#
# RE-MEASURED 2026-08-01 (release verification for v5.32.0), same method as
# the A-6 derivation: fresh interpreter + tracemalloc, N=64..2048 x
# {complex64, complex128}, fitting ``cold = slope * N^2 + fixed``.  The
# backend-import cost has GROWN with the dependency stack (numpy 2.4.4 /
# scipy 1.17.1 / scipy-openblas 0.3.31 on the Windows calibration box) from
# the 38.17-38.50 MB measured at derivation time to
#
#     pair    256 ->  512 :  fixed  52.53 MiB (c128)   52.63 MiB (c64)
#     pair    512 -> 1024 :  fixed  52.96 MiB (c128)   52.64 MiB (c64)
#     pair   1024 -> 2048 :  fixed  52.97 MiB (c128)   49.91 MiB (c64)
#
# (the N=64/128 pairs read ~40 MiB because the backend import has not yet
# paid its large-transform workspace there -- the N >= 256 asymptote is the
# one an estimate must bound).  The 40 MiB constant therefore stopped being
# a BOUND: est/measured fell to 0.79 (N=256), 0.85 (512), 0.95 (1024) --
# the A-6 contract is ``>= 1.0``.  Raised to 56 MiB, which restores the
# documented tightness band: est/measured 1.06-1.08 over the six N >= 256
# points, both dtypes (the shape term is untouched -- the measured slope is
# 96.0 B/px at c128 / 48.0 at c64, still under the 101.6 / 52.8 the formula
# uses).  Fail-safe direction: on CI Linux the cold peak is much smaller
# still, so the bound only widens there.
_ASM_FIRST_CALL_FIXED_BYTES = 56 * 1024 * 1024
# Row-band (sag_chunk_rows) mode: the full-grid float64 lens stack never
# materialises; the peak is the resident complex fields + FFT plan buffers +
# band transients.  Calibrated from the c128 chunked anchor (26.3 GB at
# N=16384/sub=16) -> ~5.3 complex-array equivalents; over-predicts c64
# (fail-safe).
_LENS_CHUNKED_COMPLEX_ARRAYS = 5.3


def _as_complex_itemsize(dtype: Any) -> int:
    """Bytes/element for the field complex dtype (defaults to complex128)."""
    return int(np.dtype(dtype).itemsize)


@overload
def estimate_lens_memory(
    n_grid: int, complex_dtype: Any = ..., *, lens_model: str = ...,
    ray_subsample: int = ..., parallel_amp: bool = ...,
    slant_correction: bool = ..., sag_dtype: Any = ...,
    sag_chunk_rows: Optional[int] = ...,
    itemized: Literal[False] = ...) -> int: ...


@overload
def estimate_lens_memory(
    n_grid: int, complex_dtype: Any = ..., *, lens_model: str = ...,
    ray_subsample: int = ..., parallel_amp: bool = ...,
    slant_correction: bool = ..., sag_dtype: Any = ...,
    sag_chunk_rows: Optional[int] = ...,
    itemized: Literal[True]) -> Dict[str, Any]: ...


def estimate_lens_memory(n_grid: int,
                         complex_dtype: Any = 'complex128',
                         *,
                         lens_model: str = 'traced',
                         ray_subsample: int = 8,
                         parallel_amp: bool = True,
                         slant_correction: bool = False,
                         sag_dtype: Any = None,
                         sag_chunk_rows: Optional[int] = None,
                         itemized: bool = False
                         ) -> Union[int, Dict[str, Any]]:
    """Estimate the peak RAM (bytes) of ONE ``apply_real_lens_traced`` (or
    ``apply_real_lens``) call -- the memory-determining step of a free-space
    + real-lens simulation.

    The model separates the **float64-fixed geometric core** (which does NOT
    shrink with complex64) from the dtype-scaling complex fields, the
    ``complex128``-first ``phase_exp`` transient, and the ``(N/ray_subsample)^2``
    Newton coarse solve.  See the module-level CALIBRATION note.

    Parameters
    ----------
    n_grid : int
        Square grid size ``N`` (peak ~ N**2).
    complex_dtype : dtype-like, default ``'complex128'``
        Field dtype.  ``'complex64'`` halves ONLY the complex terms.
    lens_model : ``'traced'`` | ``'real'``, default ``'traced'``
        ``'real'`` (bare ``apply_real_lens``) omits the traced final-assembly
        (Newton + ``map_coordinates`` + delta_phase) float64 arrays.
    ray_subsample : int, default 8
        Ray-trace OPL subsample.  Larger -> smaller Newton coarse solve.
    parallel_amp : bool, default True
        When True the amp + amp(pw) legs run concurrently -> ~2x the lens
        working set (the single largest claw-back when turned off).
    slant_correction : bool, default False
        Adds the ~6 float64 angle-gradient arrays.
    sag_dtype : dtype-like or None
        Geometry dtype.  ``None`` -> float64 (the default + only validated
        precision).  ``np.float32`` halves the geometric core AND removes the
        complex128-first ``phase_exp`` transient (opt-in; accuracy-risky).
    sag_chunk_rows : int or None
        Row-band lens mode (v5.16.2 opt-in, BYTE-IDENTICAL).  When set, the
        full-grid float64 stack never materialises -- the peak collapses to
        the resident complex fields + band transients (measured 18.4 GB vs
        43.6 GB whole-grid at N=16384/sub=16/c64).  The single largest
        fidelity-preserving claw-back.
    itemized : bool, default False
        When True, return ``{'total', 'items', ...}`` instead of a bare int.

    Returns
    -------
    int or dict
        Peak additional bytes (or an itemised dict).
    """
    N = int(n_grid)
    npix = N * N
    cb = _as_complex_itemsize(complex_dtype)
    sb = 4 if (sag_dtype is not None and np.dtype(sag_dtype) == np.float32) else 8

    # v5.17.0: mirror the runtime AUTO default (None -> banded when
    # N >= 4096; 0 -> whole-grid) so estimates match what a default call
    # actually does.  Lazy import keeps memory.py dependency-light.
    from .elements._lens_real import _resolve_sag_chunk_rows
    sag_chunk_rows = _resolve_sag_chunk_rows(sag_chunk_rows, N)

    if sag_chunk_rows is not None and int(sag_chunk_rows) > 0:
        # Row-band mode: calibrated complex-equivalents envelope + the
        # coarse Newton solve + one band's worth of float64 transients.
        sub_c = max(1, int(ray_subsample))
        newton_c = (_NEWTON_BYTES_PER_COARSE_PT * (N / sub_c) ** 2
                    if lens_model == 'traced' else 0.0)
        band = 9 * sb * N * max(1, int(sag_chunk_rows))
        resident_c = _LENS_CHUNKED_COMPLEX_ARRAYS * cb * npix
        # slant_correction disables the narrow per-surface chunking
        # (``_lens_real._narrow_chunk`` requires ``not slant_correction``),
        # so the full-grid angle stack materialises even in row-band mode.
        slant_c = (_LENS_SLANT_F64_ARRAYS * sb * npix
                   if slant_correction else 0.0)
        # v5.17.2 (audit P2-22): the runtime row-band path still runs the
        # amp + amp(pw) legs concurrently when parallel_amp=True, so the
        # LEG-LOCAL working set (resident complex fields + band transients
        # + any slant fall-through stack) doubles -- the same rule the
        # whole-grid branch applies below; the post-legs Newton solve
        # stays single.  Conservative for the shared FFT plan-buffer share
        # of the calibrated envelope (the fail-safe direction).
        if parallel_amp:
            resident_c *= 2.0
            band *= 2
            slant_c *= 2.0
        total_c = int(resident_c + newton_c + band + slant_c)
        items_c = {
            'chunked_resident_complex': int(resident_c),
            'newton_coarse_solve': int(newton_c),
            'band_transients': int(band),
        }
        if slant_correction:
            items_c['slant_fullgrid_stack'] = int(slant_c)
        if not itemized:
            return total_c
        return {'total': total_c, 'items': items_c, 'n_grid': N,
                'complex_dtype': str(np.dtype(complex_dtype)),
                'sag_dtype': 'float32' if sb == 4 else 'float64'}

    f64_core = _LENS_F64_ARRAYS * sb * npix
    if slant_correction:
        f64_core += _LENS_SLANT_F64_ARRAYS * sb * npix
    complex_part = _LENS_COMPLEX_ARRAYS * cb * npix
    # phase_exp = np.exp(1j*delta_phase) is built complex128-FIRST whenever
    # delta_phase is float64 (the default sag), so a c128-sized transient
    # rides even in a complex64 run.  float32 sag removes it.
    phase_exp_trap = 16 * npix if sb == 8 else 0

    if lens_model == 'traced':
        sub = max(1, int(ray_subsample))
        newton = _NEWTON_BYTES_PER_COARSE_PT * (N / sub) ** 2
    else:  # bare apply_real_lens has no traced final assembly
        newton = 0.0
        # 'real' holds fewer of the assembly float64 arrays
        f64_core *= (5.0 / _LENS_F64_ARRAYS)

    if parallel_amp:
        f64_core *= 2.0
        complex_part *= 2.0

    items = {
        'float64_geometric_core': int(f64_core),
        'complex_fields': int(complex_part),
        'phase_exp_c128_transient': int(phase_exp_trap),
        'newton_coarse_solve': int(newton),
    }
    total = sum(items.values())
    if not itemized:
        return total
    return {'total': total, 'items': items, 'n_grid': N,
            'complex_dtype': str(np.dtype(complex_dtype)),
            'sag_dtype': 'float32' if sb == 4 else 'float64'}


def estimate_asm_memory(n_grid: int,
                        complex_dtype: Any = 'complex128',
                        *,
                        plan_cache_keys: int = 2) -> int:
    """Estimate the FIRST-CALL peak RAM (bytes) of a band-limited ASM step.

    WHICH QUANTITY THIS ESTIMATES (audit A-6): the peak additional memory
    of the FIRST :func:`angular_spectrum_propagate` call at a given grid in
    a fresh process -- the worst case, and the quantity a pre-flight
    guardrail needs.  That peak comprises

    * the cached transfer function H + the returned output field
      (``_ASM_COMPLEX_ARRAYS`` full-grid complex arrays),
    * the float64 frequency-grid transients, which do not shrink with
      ``complex64`` (``_ASM_F64_GRID_ARRAYS``),
    * two aligned pyFFTW workspace buffers per resident plan key
      (``plan_cache_keys`` distinct fwd/inv keys), and
    * a fixed, N-independent one-time cost for the lazy FFT-backend import
      (``_ASM_FIRST_CALL_FIXED_BYTES``).

    It is deliberately NOT the steady-state figure.  Once H and the pyFFTW
    plans are resident, the measured per-call peak is just the output field
    -- ``1.00 x N^2 x itemsize`` (measured 1.000-1.008 over
    N = 256..2048) -- so this estimate runs ~6.4x that at the shapes where a
    plan key still holds TWO workspaces, ~4.4x once the v5.33.2 per-key byte
    cap drops it to one (N >= 11181 at complex128, ``plan_cache_keys=2``),
    and more at small N where the fixed import term dominates (20x at N=512
    complex128).  Use ``N * N * np.dtype(complex_dtype).itemsize`` if a
    steady-state per-call transient is what you want.

    Accuracy (RE-MEASURED 2026-08-01, fresh-interpreter ``tracemalloc``,
    pyFFTW present with the double-buffer ping-pong enabled): est/measured
    first-call peak = **1.06-1.09** over the eight points
    N = 256 / 512 / 1024 / 2048 x {complex64, complex128} -- conservative
    (a bound) at every one, within 9%.  (At derivation time,
    2026-07-25, the same band read 1.02-1.09; the dependency stack has
    since grown the one-time FFT-backend import from ~38 MB to ~53 MiB and
    ``_ASM_FIRST_CALL_FIXED_BYTES`` was re-calibrated 40 -> 56 MiB to keep
    the ``>= 1.0`` bound -- see the constant's comment for the fit table.
    Below N = 256 the ratio is looser, 1.37, because the backend import has
    not yet paid its large-transform workspace there; the A-6 measured pins
    sample N = 512 / 1024.)
    The pre-A-6 formula read 0.53 / 0.96 / 1.22 at N = 512 / 1024 / 2048
    complex128: an under-estimate where it mattered most.  On a box with no
    pyFFTW the plan-buffer and import terms over-predict, which is the
    fail-safe direction.

    Raises
    ------
    ValueError
        If ``n_grid <= 0`` or ``plan_cache_keys < 0`` (audit A-9..A-14:
        junk sizing inputs raise rather than silently clamping).
    """
    N = int(n_grid)
    if N <= 0:
        raise ValueError(
            f"estimate_asm_memory: n_grid must be positive (got {n_grid!r}).")
    keys = int(plan_cache_keys)
    if keys < 0:
        raise ValueError(
            f"estimate_asm_memory: plan_cache_keys must be >= 0 "
            f"(got {plan_cache_keys!r}).")
    npix = N * N
    cb = _as_complex_itemsize(complex_dtype)
    work = _ASM_COMPLEX_ARRAYS * cb * npix
    grids = _ASM_F64_GRID_ARRAYS * 8 * npix
    # v5.33.2: a plan key holds TWO aligned workspaces only while the
    # ping-pong is on AND both fit the per-key byte cap
    # (``fft_infra._plan_entry_n_bufs``); above the cap it holds one and the
    # dispatcher copies.  Reading the same predicate keeps this estimate from
    # over-predicting by a full grid per key at exactly the sizes where a
    # grid is gigabytes.  Unchanged at every N the A-6 pins sample (two
    # complex128 buffers at N = 1024 are 33.5 MB, three orders under the cap).
    #
    # v5.33.3 (VERIFY_PERF_BRANCH_2026_08_10 D1): the dtype handed to the
    # predicate is the CALLER's, not a re-spelled one.  The first cut built
    # ``np.dtype(f'c{2 * cb}')`` -- but ``cb`` is ALREADY the complex
    # itemsize, so that asked for a dtype of twice the element size and the
    # workspace was priced at 2x.  It failed differently on each platform,
    # which is why no pin saw it: ``'c32'`` is ``complex256`` on Linux (a
    # silent 43 % UNDER-estimate at N=8192/complex128, because a 2.147 GB
    # phantom workspace fails the 2 GB cap the true 1.074 GB one passes),
    # and a ``TypeError`` on MSVC that the except below swallowed into
    # ``n_bufs = 2`` -- a no-op for complex128 and, via the perfectly valid
    # ``'c16'``, a 9.66 GB UNDER-estimate for complex64 at N=12288 on
    # Windows too.  ``_plan_entry_n_bufs`` reads only ``.itemsize``, so
    # passing ``complex_dtype`` straight through is both correct and
    # incapable of raising for any dtype ``_as_complex_itemsize`` accepted.
    try:
        from .propagators.fft_infra import _plan_entry_n_bufs
        n_bufs = int(_plan_entry_n_bufs((N, N), np.dtype(complex_dtype)))
    except (ImportError, TypeError, ValueError):
        n_bufs = 2
    plan_bufs = keys * n_bufs * cb * npix
    return int(work + grids + plan_bufs + _ASM_FIRST_CALL_FIXED_BYTES)


@overload
def estimate_sim_memory(
    n_grid: int, complex_dtype: Any = ..., *, lens_model: str = ...,
    ray_subsample: int = ..., parallel_amp: bool = ...,
    slant_correction: bool = ..., sag_dtype: Any = ...,
    sag_chunk_rows: Optional[int] = ...,
    resume_field_bytes: int = ..., plan_cache_keys: int = ...,
    safety_factor: float = ..., itemized: Literal[False] = ...) -> int: ...


@overload
def estimate_sim_memory(
    n_grid: int, complex_dtype: Any = ..., *, lens_model: str = ...,
    ray_subsample: int = ..., parallel_amp: bool = ...,
    slant_correction: bool = ..., sag_dtype: Any = ...,
    sag_chunk_rows: Optional[int] = ...,
    resume_field_bytes: int = ..., plan_cache_keys: int = ...,
    safety_factor: float = ..., itemized: Literal[True]) -> Dict[str, Any]: ...


def estimate_sim_memory(n_grid: int,
                        complex_dtype: Any = 'complex128',
                        *,
                        lens_model: str = 'traced',
                        ray_subsample: int = 8,
                        parallel_amp: bool = True,
                        slant_correction: bool = False,
                        sag_dtype: Any = None,
                        sag_chunk_rows: Optional[int] = None,
                        resume_field_bytes: int = 0,
                        plan_cache_keys: int = 2,
                        safety_factor: float = 1.15,
                        itemized: bool = False
                        ) -> Union[int, Dict[str, Any]]:
    """Estimate the peak RAM (bytes) of a full free-space + real-lens
    simulation: ``safety_factor * max(lens_step, asm_step) + resume_field``.

    The consumer resets the FFT plan cache before each lens, so the lens
    step and the ASM step do NOT overlap -- the peak is the larger of the
    two, plus any resumed-checkpoint plane held resident.

    Returns a bare int (peak bytes) or, with ``itemized=True``, a dict with
    the per-step breakdown and the driving step.
    """
    lens = estimate_lens_memory(
        n_grid, complex_dtype, lens_model=lens_model, ray_subsample=ray_subsample,
        parallel_amp=parallel_amp, slant_correction=slant_correction,
        sag_dtype=sag_dtype, sag_chunk_rows=sag_chunk_rows, itemized=True)
    asm = estimate_asm_memory(n_grid, complex_dtype, plan_cache_keys=plan_cache_keys)
    step = max(lens['total'], asm)
    driver = 'lens' if lens['total'] >= asm else 'asm'
    raw = step + int(resume_field_bytes)
    peak = int(raw * float(safety_factor))
    if not itemized:
        return peak
    return {
        'peak_bytes': peak,
        'raw_bytes': int(raw),
        'safety_factor': float(safety_factor),
        'driving_step': driver,
        'lens_step_bytes': lens['total'],
        'asm_step_bytes': asm,
        'resume_field_bytes': int(resume_field_bytes),
        'lens_items': lens['items'],
        'n_grid': int(n_grid),
        'complex_dtype': str(np.dtype(complex_dtype)),
    }


def check_sim_memory(n_grid: int,
                     complex_dtype: Any = 'complex128',
                     *,
                     lens_model: str = 'traced',
                     ray_subsample: int = 8,
                     parallel_amp: bool = True,
                     slant_correction: bool = False,
                     sag_dtype: Any = None,
                     sag_chunk_rows: Optional[int] = None,
                     resume_field_bytes: int = 0,
                     plan_cache_keys: int = 2,
                     safety_factor: float = 1.15,
                     available: Optional[int] = None,
                     mode: str = 'warn',
                     verbose: bool = True) -> Dict[str, Any]:
    """Autodetect guardrail: estimate the true peak, compare to available RAM,
    and (depending on ``mode``) warn / raise / just report -- ALWAYS returning
    a structured verdict that, when it does not fit, lists concrete claw-backs
    that DO fit (sag_chunk_rows [byte-identical] -> parallel_amp off ->
    complex64 -> coarser ray_subsample -> smaller N), each with its
    estimated peak.

    Parameters
    ----------
    mode : ``'warn'`` (default) | ``'raise'`` | ``'silent'``
        ``'warn'`` emits a ``RuntimeWarning`` if it will not fit; ``'raise'``
        raises ``MemoryError``; ``'silent'`` only returns the verdict.
    available : int or None
        Available bytes; defaults to :func:`get_ram_budget`.

    Returns
    -------
    dict
        ``{'fits', 'peak_bytes', 'available_bytes', 'driving_step',
        'breakdown', 'message', 'recommendations'}``.
    """
    if available is None:
        available = get_ram_budget()
    est = estimate_sim_memory(
        n_grid, complex_dtype, lens_model=lens_model, ray_subsample=ray_subsample,
        parallel_amp=parallel_amp, slant_correction=slant_correction,
        sag_dtype=sag_dtype, sag_chunk_rows=sag_chunk_rows,
        resume_field_bytes=resume_field_bytes,
        plan_cache_keys=plan_cache_keys, safety_factor=safety_factor, itemized=True)
    peak = est['peak_bytes']
    fits = peak <= int(available)

    # Build the claw-back ladder: byte-identical knobs first (row-band lens
    # mode preserves fidelity EXACTLY, so it always leads), then accuracy/
    # fidelity trades, then grid reduction.  Each entry re-estimates and is
    # reported only if it actually fits.
    recs = []
    def _try(label: str, **over: Any) -> None:
        kw = dict(lens_model=lens_model, ray_subsample=ray_subsample,
                  parallel_amp=parallel_amp, slant_correction=slant_correction,
                  sag_dtype=sag_dtype, sag_chunk_rows=sag_chunk_rows,
                  resume_field_bytes=resume_field_bytes,
                  plan_cache_keys=plan_cache_keys, safety_factor=safety_factor)
        ngrid = over.pop('n_grid', n_grid)
        cdt = over.pop('complex_dtype', complex_dtype)
        kw.update(over)
        p = estimate_sim_memory(ngrid, cdt, **kw)
        if p <= int(available):
            recs.append({'change': label, 'peak_bytes': p,
                         'peak_gb': p / 1e9, 'fits': True})
    if not fits:
        _n = int(n_grid)
        # v5.17.0: chunking is the AUTO default (None); this rung only
        # applies when the caller explicitly forced the whole-grid path.
        if (lens_model == 'traced' and sag_chunk_rows is not None
                and int(sag_chunk_rows) == 0):
            _try(f'set sag_chunk_rows={max(256, _n // 16)} (row-band lens, '
                 f'byte-identical)', sag_chunk_rows=max(256, _n // 16))
            _try(f'sag_chunk_rows={max(256, _n // 16)} + parallel_amp=False '
                 f'(both byte-identical)',
                 sag_chunk_rows=max(256, _n // 16), parallel_amp=False)
        if parallel_amp:
            _try('set parallel_amp=False (byte-identical)', parallel_amp=False)
        if np.dtype(complex_dtype) == np.complex128:
            # audit P3-45: the re-estimate assumes parallel_amp=False too,
            # so the label must say so (matches the neighbouring rungs).
            _try('use complex64 fields (+parallel_amp=False)',
                 complex_dtype='complex64', parallel_amp=False)
        if lens_model == 'traced' and int(ray_subsample) < 16:
            _try('ray_subsample=16 (+parallel_amp=False)',
                 ray_subsample=16, parallel_amp=False)
        # grid reductions
        for ng in (24576, 16384, 12288, 8192):
            if ng < int(n_grid):
                _try(f'reduce N_GRID to {ng} (+parallel_amp=False)',
                     n_grid=ng, parallel_amp=False)
                break

    def gb(b: float) -> float:
        return b / 1e9
    items = est['lens_items']
    # Itemisation differs between the whole-grid and row-band lens models.
    items_str = " + ".join(
        f"{k.replace('_', '-')} {gb(v):.0f} GB" for k, v in items.items())
    msg = (f"N={n_grid} {np.dtype(complex_dtype)} {lens_model}-lens "
           f"(sub={ray_subsample}, parallel_amp={parallel_amp}"
           f"{f', chunk_rows={sag_chunk_rows}' if sag_chunk_rows else ''}): "
           f"peak ~{gb(peak):.0f} GB (driver: {est['driving_step']} step; "
           f"{items_str}), "
           f"x{safety_factor} safety; have {gb(available):.0f} GB.")
    if fits:
        msg = "OK: " + msg
    else:
        msg = "INSUFFICIENT RAM: " + msg
        if recs:
            msg += " Claw-backs that fit: " + "; ".join(
                f"{r['change']} -> ~{r['peak_gb']:.0f} GB" for r in recs) + "."
        else:
            msg += (" No single claw-back fits; combine complex64 + sub=16 + "
                    "lower N, or free RAM.")

    verdict = {
        'fits': bool(fits), 'peak_bytes': int(peak),
        'available_bytes': int(available), 'driving_step': est['driving_step'],
        'breakdown': est, 'message': msg, 'recommendations': recs,
    }
    if verbose:
        print(("  " if fits else "  ** ") + msg, flush=True)
    if not fits:
        if mode == 'raise':
            raise MemoryError(msg)
        if mode == 'warn':
            warnings.warn(msg, RuntimeWarning)
    return verdict


# ----------------------------------------------------------------------------
# Low-memory preset
# ----------------------------------------------------------------------------
# v5.17.2 (audit P2-23 / P3-46): the values captured at the FIRST
# set_low_memory(True) since the last disable, so set_low_memory(False)
# restores exactly what the user had (including the aggressive complex64
# default-dtype flip and any pre-existing non-default knob settings)
# instead of clobbering to shipped defaults.
_LOW_MEMORY_PRIOR: Optional[Dict[str, Any]] = None

# Shipped library defaults for the byte-safe knobs, used only when
# set_low_memory(False) is called with no enable on record (or when a
# getter failed at capture time).
_LOW_MEMORY_SHIPPED_DEFAULTS: Dict[str, Any] = {
    'plan_cache_size': 8,
    'lens_parallel_amp': True,
    'fft_double_buffer': True,
    # v5.30.1 (audit W9): tracks the fft_infra default, which flipped
    # True -> False when ESTIMATE->MEASURE auto-promote became opt-in.
    # Must stay in sync -- this table is what set_low_memory(False)
    # restores when there is no enable on record, so a stale True here
    # would silently switch a caller INTO the non-reproducible planner.
    # Companion-locked to fft_infra._PYFFTW_AUTO_PROMOTE_SHIPPED by
    # tests/unit/test_niche_audit_w9_traced_determinism.py.
    'fft_auto_promote': False,
}


def set_low_memory(enabled: bool = True, *, aggressive: bool = False) -> Dict[str, Any]:
    """Flip the BYTE-SAFE memory-lean knobs together (and, with
    ``aggressive=True``, the numerics-changing ones).  Returns the prior
    values so the change round-trips via ``set_low_memory(False)``, which
    restores exactly the values captured at the first ``set_low_memory(True)``
    -- including a user's pre-existing non-default settings and the
    aggressive default-dtype flip.  ``set_low_memory(False)`` without a
    matching enable restores the shipped library defaults.

    Safe set (all byte-identical to a default run -- memory/speed trade only):
      * ``set_fft_plan_cache_size(2)``     -- the fwd+inv floor for one grid
      * ``set_lens_parallel_amp(False)``   -- sequential amp (largest claw-back)
      * ``set_fft_double_buffer(False)``   -- one aligned buffer per plan key
      * ``set_fft_auto_promote(False)``    -- no transient MEASURE re-plan
        (v5.30.1: already the shipped default, so this is a no-op unless
        the caller opted in; before v5.30.1 the "byte-identical to a
        default run" claim above was false for THIS knob, since a default
        run promoted to MEASURE after the 5th call at a key)

    Aggressive set (CHANGE NUMERICS -- gated + logged):
      * ``set_default_complex_dtype(np.complex64)``  (~80 dB dynamic range)

    (The double-buffer-off and float32/chunked-sag knobs land in a follow-up.)
    """
    global _LOW_MEMORY_PRIOR
    # Lazy imports keep memory.py dependency-light (no module-level edge into
    # propagators/ or elements/, avoiding an import cycle).
    from .elements._lens_traced import get_lens_parallel_amp, set_lens_parallel_amp
    from .propagators.fft_infra import (
        get_fft_auto_promote,
        get_fft_double_buffer,
        get_fft_plan_cache_size,
        set_fft_auto_promote,
        set_fft_double_buffer,
        set_fft_plan_cache_size,
    )

    prior: Dict[str, Any] = {}

    def _capture(key: str, getter: Any) -> None:
        try:
            prior[key] = getter()
        except (ImportError, RuntimeError, AttributeError):
            # Best-effort: a getter failing on a partial install must not
            # strand the rest of the capture (the registry-walk tuple).
            prior[key] = None

    if enabled:
        _capture('plan_cache_size', get_fft_plan_cache_size)
        _capture('lens_parallel_amp', get_lens_parallel_amp)
        _capture('fft_double_buffer', get_fft_double_buffer)
        # audit P3-46: read the LIVE value like the other knobs (pre-fix
        # this hardcoded True, losing e.g. a byte-reproducibility pin set
        # via set_fft_auto_promote(False)).
        _capture('fft_auto_promote', get_fft_auto_promote)
        set_fft_plan_cache_size(2)
        set_lens_parallel_amp(False)
        set_fft_double_buffer(False)
        set_fft_auto_promote(False)
        if aggressive:
            try:
                from .propagators.fft_infra import (
                    get_default_complex_dtype,
                    set_default_complex_dtype,
                )
                prior['complex_dtype'] = get_default_complex_dtype()
                set_default_complex_dtype(np.complex64)
                warnings.warn(
                    "set_low_memory(aggressive=True) set the default field "
                    "dtype to complex64 (~80 dB dynamic range). Validate "
                    "deep-null / stray-light-sensitive results.",
                    RuntimeWarning)
            except Exception:
                pass
        # Stash the FIRST-enable snapshot: repeated set_low_memory(True)
        # calls must not overwrite the true prior with low-memory values.
        # A later aggressive enable still records the pre-flip dtype
        # (setdefault: an already-stashed dtype wins).
        if _LOW_MEMORY_PRIOR is None:
            _LOW_MEMORY_PRIOR = dict(prior)
        elif 'complex_dtype' in prior:
            _LOW_MEMORY_PRIOR.setdefault('complex_dtype',
                                         prior['complex_dtype'])
    else:
        # audit P2-23 / P3-46: restore EXACTLY the values captured at
        # enable time (falling back to the shipped default for any knob
        # whose getter failed at capture, or for a disable with no enable
        # on record).
        stash = _LOW_MEMORY_PRIOR
        _LOW_MEMORY_PRIOR = None
        if stash is None:
            stash = dict(_LOW_MEMORY_SHIPPED_DEFAULTS)

        def _restore(key: str, setter: Any) -> None:
            val = stash.get(key)
            if val is None:
                val = _LOW_MEMORY_SHIPPED_DEFAULTS[key]
            setter(val)

        _restore('plan_cache_size', set_fft_plan_cache_size)
        _restore('lens_parallel_amp', set_lens_parallel_amp)
        _restore('fft_double_buffer', set_fft_double_buffer)
        _restore('fft_auto_promote', set_fft_auto_promote)
        # audit P2-23: revert the aggressive default-dtype flip (only when
        # an aggressive enable actually captured one).
        if stash.get('complex_dtype') is not None:
            try:
                from .propagators.fft_infra import set_default_complex_dtype
                set_default_complex_dtype(stash['complex_dtype'])
            except Exception:
                pass
    return prior


__all__ = [
    'get_ram_budget', 'set_max_ram', 'get_max_ram',
    'available_memory_bytes', 'total_memory_bytes', 'memory_info',
    'bytes_per_element', 'array_bytes',
    'estimate_op_memory', 'pick_batch_size',
    'should_split',
    'format_bytes', 'print_memory_report',
    'available_cpus',
    # v5.16.1 system-level estimation + guardrail
    'estimate_lens_memory', 'estimate_asm_memory', 'estimate_sim_memory',
    'check_sim_memory', 'set_low_memory',
]
