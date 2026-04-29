"""
Memory-Aware Batching Helpers
==============================

Utilities for estimating the memory cost of an operation and deciding
whether to run it straight-through or split it into batches that fit in
the machine's available RAM.

The core idea is::

    cost = estimate_op_memory(shape, dtype, n_work_arrays)
    batch = pick_batch_size(n_items, cost_per_item, safety=0.5)

Most functions in the library operate on a single (Ny, Nx) complex field
and allocate a handful of working arrays internally (FFT buffers, phase
masks, etc.).  For operations that loop over K sources / slices / lenslets,
these helpers pick the largest K' that will comfortably fit in RAM and
return that as the recommended batch size.

The helpers degrade gracefully: if :mod:`psutil` is not installed or the
OS does not expose memory info, they fall back to conservative defaults
and a warning is emitted.

Author: Andrew Traverso
"""

import warnings
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


def get_ram_budget():
    """
    Return the effective RAM budget in bytes.

    If :func:`set_max_ram` was called with a value, returns that value.
    Otherwise returns the currently available physical memory via psutil
    (or the 4 GB fallback when psutil is not installed).
    """
    if _MAX_RAM_OVERRIDE is not None:
        return _MAX_RAM_OVERRIDE
    return available_memory_bytes()


def set_max_ram(value):
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
    elif value < 1024:
        _MAX_RAM_OVERRIDE = int(value * 1024**3)
    else:
        _MAX_RAM_OVERRIDE = int(value)


# ---------------------------------------------------------------------------
# Memory queries
# ---------------------------------------------------------------------------
def available_memory_bytes():
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


def total_memory_bytes():
    """Return the total physical memory in bytes."""
    if _PSUTIL_AVAILABLE:
        return int(psutil.virtual_memory().total)
    return _DEFAULT_TOTAL_BYTES


def memory_info():
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
def bytes_per_element(dtype):
    """
    Return the number of bytes per element for a numpy dtype.

    Accepts either a dtype object, a dtype string ('complex128'),
    or a Python type (complex, float, int).
    """
    return np.dtype(dtype).itemsize


def array_bytes(shape, dtype='complex128'):
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


def estimate_op_memory(shape, dtype='complex128',
                       n_work_arrays=3, extra_bytes=0):
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
def pick_batch_size(n_items, cost_per_item, available=None,
                    safety=0.5, min_batch=1, max_batch=None):
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
        Memory cost of ONE item in bytes.
    available : int or None
        Available memory in bytes.  If ``None``, uses
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
        available = available_memory_bytes()

    if cost_per_item <= 0:
        return max(min_batch, n_items)

    budget = int(available * safety)
    k = max(1, budget // cost_per_item)
    k = min(k, n_items)
    k = max(k, min_batch)
    if max_batch is not None:
        k = min(k, max_batch)
    return int(k)


def should_split(total_cost, available=None, safety=0.5):
    """
    Decide whether an operation needs to be split into batches.

    Parameters
    ----------
    total_cost : int
        Estimated memory cost of running the whole operation at once,
        in bytes.
    available : int or None
        Available memory in bytes.  Defaults to current system state.
    safety : float
        Fraction of available memory considered "safe" to use.

    Returns
    -------
    split : bool
        True if ``total_cost`` exceeds ``safety * available``.
    """
    if available is None:
        available = available_memory_bytes()
    return total_cost > int(available * safety)


# ---------------------------------------------------------------------------
# Pretty-printing helpers
# ---------------------------------------------------------------------------
def format_bytes(nbytes):
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


def print_memory_report(planned_cost_bytes=None, prefix=''):
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
