"""
lumenairy.elements._lens_traced -- per-pixel ray-traced
``apply_real_lens_traced`` plus its support helpers.

Hybrid wave + ray-trace propagator: amplitude comes from a single
analytic ``apply_real_lens`` call (split-step ASM through glass),
phase comes from a per-pixel geometric ray-trace OPL evaluated on a
coarse entrance grid and Newton-inverted onto the wave grid via a
Chebyshev tensor-product fit.

Extracted from ``lenses.py`` in v3.5.5 to reduce that module's bloat.
``apply_real_lens_traced`` and ``close_worker_pool`` are re-exported
from :mod:`lumenairy.elements.lenses` for backwards-compatible
imports.

Author: Andrew Traverso
"""

from __future__ import annotations

import importlib.util as _importlib_util
from typing import Any, Dict, Optional

import numpy as np

# Optional CuPy backend (lazy).
CUPY_AVAILABLE = _importlib_util.find_spec('cupy') is not None
cp = None


def _ensure_cupy_loaded():
    global cp
    if cp is None and CUPY_AVAILABLE:
        import cupy as _c
        cp = _c
    return cp is not None


def _is_cupy_array(x):
    if not CUPY_AVAILABLE:
        return False
    if cp is None and not _ensure_cupy_loaded():
        return False
    return isinstance(x, cp.ndarray)


# Optional numexpr fused-expression backend (lazy).
NUMEXPR_AVAILABLE = _importlib_util.find_spec('numexpr') is not None
_ne = None


def _ensure_numexpr_loaded():
    global _ne
    if _ne is None and NUMEXPR_AVAILABLE:
        import numexpr as _n
        _ne = _n
    return _ne is not None


_NUMEXPR_MIN_SIZE = 1 << 20

# Optional Numba JIT, LAZILY imported on first kernel use (audit P2-D: the eager
# ``import numba`` cost ~1.8 s of ``import lumenairy`` cold start).  The kernel
# (``_cheb2d_val_grad_numba``) has a pure-NumPy fallback, so numba is pulled in
# only when a caller actually hits the fast path AND numba is installed.
import importlib.util as _ilu

_NUMBA_AVAILABLE = _ilu.find_spec("numba") is not None
_numba = None                         # populated by _load_numba() on first use
_njit = None
_prange = None
_NUMBA_KERNELS: dict = {}             # kernel-name -> compiled fn (or None)


def _load_numba():
    """Import numba + njit/prange on first use; cache the handles.  Returns True
    iff numba is importable (False -> caller takes the pure-NumPy fallback)."""
    global _numba, _njit, _prange
    if _numba is not None:
        return True
    if not _NUMBA_AVAILABLE:
        return False
    import numba as _nb
    from numba import njit as _nj
    from numba import prange as _pr
    _numba, _njit, _prange = _nb, _nj, _pr
    return True


# Newton iter cap default.  Set to 12 (the historical value).
# 3.5.5 dropped this to 8 based on an audit recommendation, but the
# active-mask early-exit already short-circuits converged pixels -- the
# cap only matters for outlier pixels that genuinely need 9-12 iters.
# Truncating those at 8 silently lost accuracy on cemented multi-element
# / strongly-aberrated systems.  3.5.6 reverts to the safe 12.  Override
# via apply_real_lens_traced(newton_max_iters=N) when profiling shows
# Newton dominates.
_NEWTON_MAX_ITERS = 12


# Module-level default for ``apply_real_lens_traced(parallel_amp=...)``.  The
# kwarg default is ``None`` -> resolves to this global, so a process-wide
# ``set_lens_parallel_amp(False)`` (or ``lumenairy.set_low_memory(True)``)
# flips the amp+amp(pw) concurrency off for callers that don't pass the kwarg.
# Shipped default True is byte-identical to the historical behaviour; turning
# it off is the single largest lens-step memory claw-back (~2x working set)
# and is numerically identical (same math, serialised).
_LENS_PARALLEL_AMP_DEFAULT = True


def set_lens_parallel_amp(enabled: bool) -> None:
    """Set the process-wide default for ``apply_real_lens_traced``'s
    concurrent amp + amp(pw) execution.  ``False`` halves the lens-step
    peak working set (byte-identical output, ~20% slower lens step)."""
    global _LENS_PARALLEL_AMP_DEFAULT
    _LENS_PARALLEL_AMP_DEFAULT = bool(enabled)


def get_lens_parallel_amp() -> bool:
    """Return the process-wide default for the lens amp/amp(pw) concurrency."""
    return bool(_LENS_PARALLEL_AMP_DEFAULT)


# Helpers shared with lenses.py (single-element sag, aperture warning).
from .lenses import (
    _warn_if_aperture_exceeds_grid,
    surface_sag_general,
)

_surface_sag_general = surface_sag_general

# Sibling-module imports.
# v5.3.2 (ROADMAP logging adoption sweep -- per-iteration telemetry):
# Module-level logger for apply_real_lens_traced entry / per-Newton-
# iteration progress.  Default-quiet via the lumenairy root logger's
# NullHandler -- users opt in by attaching a handler to the
# ``lumenairy`` logger.
from .._logging import get_logger
from ..glass import get_glass_index
from ..progress import ProgressScaler, call_progress

logger = get_logger(__name__)

# apply_real_lens (analytic split-step) is the workhorse for the
# amplitude leg of apply_real_lens_traced.  Lives in _lens_real.py
# since v3.5.5; re-imported here for the in-function callbacks.
from ._lens_real import apply_real_lens


def _newton_invert_chunk(args):
    """Module-level worker for ``apply_real_lens_traced`` Newton inversion.

    Rebuilds the three ``RectBivariateSpline`` objects from their knot
    data in-process (so we avoid pickling the SciPy spline objects,
    which is expensive) and runs the Newton loop on ``(x_chunk,
    y_chunk)`` for up to ``_NEWTON_MAX_ITERS`` iterations.  Returns
    the OPL at the converged entrance positions with NaN for any
    points that landed outside the fit domain.

    Lives at module scope so ``ProcessPoolExecutor`` can pickle it on
    Windows (spawn) workers.  The caller is ``_invert_newton`` inside
    :func:`apply_real_lens_traced`.
    """
    (knot_data, x_chunk, y_chunk) = args
    from scipy.interpolate import RectBivariateSpline
    xs_in = knot_data['xs_in']
    x_out_grid = knot_data['x_out_grid']
    y_out_grid = knot_data['y_out_grid']
    opl_grid = knot_data['opl_grid']
    launch_radius = knot_data['launch_radius']
    dx = knot_data['dx']
    bound = knot_data['bound']
    # Paraxial-magnification initial-guess factors.  See the docstring
    # in ``apply_real_lens_traced`` where these are computed from the
    # central finite-difference slope of the forward map.  Older knot
    # data written by pre-3.1.3 callers won't have these keys -- fall
    # back to the historical 1.10 multiplier so the worker stays
    # backwards compatible.
    inv_M_x = float(knot_data.get('inv_M_x', 1.10))
    inv_M_y = float(knot_data.get('inv_M_y', 1.10))

    Sx = RectBivariateSpline(xs_in, xs_in, x_out_grid, kx=3, ky=3)
    Sy = RectBivariateSpline(xs_in, xs_in, y_out_grid, kx=3, ky=3)
    So = RectBivariateSpline(xs_in, xs_in, opl_grid, kx=3, ky=3)

    xe = x_chunk.copy() * inv_M_x
    ye = y_chunk.copy() * inv_M_y
    tol = 0.01 * dx
    active = np.ones(xe.size, dtype=bool)
    for _it in range(_NEWTON_MAX_ITERS):
        if not active.any():
            break
        xa = xe[active]
        ya = ye[active]
        xw = x_chunk[active]
        yw = y_chunk[active]
        rx = Sx.ev(xa, ya) - xw
        ry = Sy.ev(xa, ya) - yw
        jxx = Sx.ev(xa, ya, dx=1)
        jxy = Sx.ev(xa, ya, dy=1)
        jyx = Sy.ev(xa, ya, dx=1)
        jyy = Sy.ev(xa, ya, dy=1)
        det = jxx * jyy - jxy * jyx
        safe = np.abs(det) > 1e-12
        inv_det = np.where(safe, 1.0 / det, 0.0)
        dxe = (jyy * rx - jxy * ry) * inv_det
        dye = (-jyx * rx + jxx * ry) * inv_det
        xa_new = np.clip(xa - dxe, -bound, bound)
        ya_new = np.clip(ya - dye, -bound, bound)
        xe[active] = xa_new
        ye[active] = ya_new
        res = np.sqrt(rx * rx + ry * ry)
        converged = res < tol
        idx_active = np.where(active)[0]
        active[idx_active[converged]] = False

    opl_flat = So.ev(xe, ye)
    out_of_domain = (xe * xe + ye * ye > (launch_radius * 0.99) ** 2)
    return np.where(out_of_domain, np.nan, opl_flat)


# --------------------------------------------------------------------------
# Persistent ProcessPool for apply_real_lens_traced Newton inversion.
#
# Pre-3.5.5: every apply_real_lens_traced call created+torn-down its own
# pool, paying the Windows-spawn startup cost (~5 s for n_workers=8) once
# per call.  For optimisation runs and tolerancing studies that call
# apply_real_lens_traced 100+ times the cumulative cost was minutes.
#
# 3.5.5+: a module-level pool is lazily created on first parallel-Newton
# call and reused across subsequent calls with the same worker count.
# An atexit handler shuts it down cleanly.  Call ``close_worker_pool()``
# explicitly to free the workers early (e.g. after a final optimisation
# step, before a long-running serial post-process).
# --------------------------------------------------------------------------

_PERSISTENT_POOL = None
_PERSISTENT_POOL_NWORKERS = None
_PERSISTENT_POOL_LOCK = None  # threading.Lock built lazily


def _get_persistent_worker_pool(n_workers):
    """Return a (possibly newly-created) shared ProcessPoolExecutor.

    Reuses the same pool across calls when the requested ``n_workers``
    matches the cached pool's size.  Tears down and rebuilds when
    ``n_workers`` changes.  Pool is registered with ``atexit`` for
    clean shutdown on interpreter exit.
    """
    global _PERSISTENT_POOL, _PERSISTENT_POOL_NWORKERS, _PERSISTENT_POOL_LOCK
    import threading
    if _PERSISTENT_POOL_LOCK is None:
        _PERSISTENT_POOL_LOCK = threading.Lock()
    with _PERSISTENT_POOL_LOCK:
        if _PERSISTENT_POOL is not None:
            if _PERSISTENT_POOL_NWORKERS == n_workers:
                return _PERSISTENT_POOL
            # n_workers changed: tear down the existing pool.
            try:
                _PERSISTENT_POOL.shutdown(wait=False)
            except (RuntimeError, OSError, BrokenPipeError):
                # Pool already torn down by atexit / signal handler,
                # or worker pipe broke under shutdown -- safe to
                # discard the reference.
                pass
            _PERSISTENT_POOL = None
        # v4.16.1 (audit M-2): force the ``spawn`` start method.  The
        # default on Linux is ``fork``, which inherits the parent's
        # FFT plan caches and threading state -- both of which are
        # unsafe to share between forked processes (pyFFTW's plan
        # cache holds module-private locks that the forked child
        # cannot release; numpy/MKL spin up a duplicate thread pool
        # that races with the parent).  ``spawn`` is portable across
        # Linux + macOS + Windows and matches the v4.16.0 CHANGELOG
        # claim that the library uses spawn (which was previously
        # only true of the multi-process storage tests, not the
        # library worker pool itself).
        import multiprocessing as _mp
        from concurrent.futures import ProcessPoolExecutor
        _spawn_ctx = _mp.get_context('spawn')
        _PERSISTENT_POOL = ProcessPoolExecutor(
            max_workers=int(n_workers),
            mp_context=_spawn_ctx,
        )
        _PERSISTENT_POOL_NWORKERS = int(n_workers)
        # Register atexit handler exactly once.
        import atexit
        atexit.register(close_worker_pool)
    return _PERSISTENT_POOL


def close_worker_pool() -> None:
    """Shut down the module-level worker pool used by
    :func:`apply_real_lens_traced`.

    Safe to call multiple times.  Called automatically at interpreter
    exit; only call explicitly when you want to free the workers
    before a long-running serial step (e.g. plotting, I/O).
    """
    global _PERSISTENT_POOL, _PERSISTENT_POOL_NWORKERS
    if _PERSISTENT_POOL is not None:
        try:
            _PERSISTENT_POOL.shutdown(wait=True)
        except (RuntimeError, OSError, BrokenPipeError):
            # Same shutdown-race tolerance as ``_get_pool``.
            pass
        _PERSISTENT_POOL = None
        _PERSISTENT_POOL_NWORKERS = None


# Sibling-module imports (created separately in this package) ----------------

# Typing: the Maslov section (merged in 3.2.2 from the former
# lens_maslov.py) uses Any / Dict / Optional / Tuple in function
# annotations.
# The Maslov section uses ``time`` for internal progress timing.


# ---------------------------------------------------------------------------
# Optional Numba JIT for the polynomial-evaluator inner loop.
#
# The hot path of _Cheb2DEvaluator.ev_value_and_grad is a doubly-nested
# reduction over (basis_term, output_sample).  Plain NumPy executes it
# as a chain of broadcast multiplies and sum-reductions with a handful
# of allocated temporaries; a @njit kernel collapses that to a single
# tight loop with zero temporaries and thread-parallel output rows.
#
# Guarded import -- fallback to pure-xp path (which is fine on NumPy and
# REQUIRED on CuPy) when numba isn't installed.  The kernel is compiled LAZILY
# on first call via _get_cheb2d_val_grad_numba() so ``import lumenairy`` never
# pays the numba import / compile cost (audit P2-D).
# ---------------------------------------------------------------------------


def _get_cheb2d_val_grad_numba():
    """Compile (once, on first call) and return the Chebyshev value+gradient
    numba kernel, or ``None`` if numba is unavailable."""
    if "cheb2d" in _NUMBA_KERNELS:
        return _NUMBA_KERNELS["cheb2d"]
    if not _load_numba():
        _NUMBA_KERNELS["cheb2d"] = None
        return None

    @_njit(cache=True, parallel=True, fastmath=True)
    def _cheb2d_val_grad_numba(coeffs, K1, K2, u_flat, v_flat, max_order):
        """Combined Chebyshev value + gradient via in-place recurrence.

        Computes f(u, v), df/du, df/dv at every (u_flat[i], v_flat[i])
        sample in parallel.  Chebyshev T and U (second kind) values are
        generated by 3-term recurrence on the stack per sample -- no
        Vandermonde matrices are materialised.  This implements the
        Clenshaw-style "#3" optimisation: O(order) stack work per
        sample instead of an O(N x order) materialised Vandermonde.

        Parameters
        ----------
        coeffs : (M,) float64   -- polynomial coefficients in total-degree order
        K1, K2 : (M,) int64     -- multi-indices (kx, ky) for each term
        u_flat, v_flat : (N,) float64 -- normalised sample coords in [-1, 1]
        max_order : int         -- maximum individual Chebyshev order

        Returns
        -------
        f, fx_u, fy_v : three (N,) float64 arrays: value and du/dv-partials
        in normalised coordinates.  Caller applies chain rule for
        physical derivatives.
        """
        N = u_flat.shape[0]
        M = coeffs.shape[0]
        f = np.zeros(N)
        fx = np.zeros(N)
        fy = np.zeros(N)

        for i in _prange(N):
            u = u_flat[i]
            v = v_flat[i]

            # T_n(u), T_n(v): first kind, by 3-term recurrence
            # T_0 = 1, T_1 = u, T_{n+1} = 2u T_n - T_{n-1}
            Tu = np.empty(max_order + 1)
            Tv = np.empty(max_order + 1)
            Tu[0] = 1.0
            Tv[0] = 1.0
            if max_order >= 1:
                Tu[1] = u
                Tv[1] = v
            for n in range(2, max_order + 1):
                Tu[n] = 2.0 * u * Tu[n - 1] - Tu[n - 2]
                Tv[n] = 2.0 * v * Tv[n - 1] - Tv[n - 2]

            # T'_n(u) = n * U_{n-1}(u); U_0 = 1, U_1 = 2u, U_{n+1} = 2u U_n - U_{n-1}
            # We store dTu[n] = T'_n(u) directly for n = 0..max_order
            dTu = np.zeros(max_order + 1)
            dTv = np.zeros(max_order + 1)
            if max_order >= 1:
                dTu[1] = 1.0          # T'_1 = 1 * U_0 = 1
                dTv[1] = 1.0
                if max_order >= 2:
                    U_prev_u = 1.0    # U_0
                    U_u = 2.0 * u     # U_1
                    U_prev_v = 1.0
                    U_v = 2.0 * v
                    dTu[2] = 2.0 * U_u    # T'_2 = 2 * U_1
                    dTv[2] = 2.0 * U_v
                    for n in range(3, max_order + 1):
                        U_next_u = 2.0 * u * U_u - U_prev_u
                        U_next_v = 2.0 * v * U_v - U_prev_v
                        U_prev_u = U_u
                        U_u = U_next_u
                        U_prev_v = U_v
                        U_v = U_next_v
                        dTu[n] = n * U_u
                        dTv[n] = n * U_v

            # Accumulate coefficient-weighted sum over multi-indices
            acc_f = 0.0
            acc_fx = 0.0
            acc_fy = 0.0
            for m in range(M):
                kx = K1[m]
                ky = K2[m]
                c = coeffs[m]
                tu = Tu[kx]
                tv = Tv[ky]
                acc_f  += c * tu * tv
                acc_fx += c * dTu[kx] * tv
                acc_fy += c * tu * dTv[ky]
            f[i] = acc_f
            fx[i] = acc_fx
            fy[i] = acc_fy
        return f, fx, fy

    _NUMBA_KERNELS["cheb2d"] = _cheb2d_val_grad_numba
    return _cheb2d_val_grad_numba


def _get_array_module(arr):
    """Return the array namespace (numpy or cupy) for ``arr``.

    Enables array-API polymorphism: code that uses only namespace-
    agnostic operations (xp.asarray, xp.sum, xp.meshgrid, ...) runs
    unchanged on NumPy or CuPy arrays.  Gracefully degrades to NumPy
    when CuPy isn't installed.
    """
    try:
        import cupy as _cp
        if isinstance(arr, _cp.ndarray):
            return _cp
    except ImportError:
        pass
    return np


class _Cheb2DEvaluator:
    """2-D Chebyshev tensor-product polynomial fit with an API compatible
    with a SciPy ``RectBivariateSpline`` for the subset used by
    :func:`apply_real_lens_traced` -- specifically the ``ev(x, y)``,
    ``ev(x, y, dx=1)``, and ``ev(x, y, dy=1)`` methods.

    This is the polynomial equivalent of the default spline interpolation
    used by ``apply_real_lens_traced``'s Newton inversion, enabled when
    ``newton_fit='polynomial'``.  For smooth refractive lenses where the
    entrance->exit coordinate map and the OPL are essentially polynomials
    of total degree up to 6 (all Seidel + higher-order aberrations of
    reasonable orders), this is both faster (closed-form analytic
    derivatives, no Fortran spline calls) and more accurate (no cubic
    truncation error on the coarse grid).

    Architecture
    ------------
    * **Array-API polymorphic**: the ``xp`` constructor kwarg selects
      the array backend (default :mod:`numpy`).  Pass ``xp=cupy`` to
      run the fit and evaluation on the GPU with zero code changes
      here -- every internal operation uses ``self.xp``'s namespace.
    * **Combined value + gradient** (``ev_value_and_grad``): returns
      ``(f, df/dx, df/dy)`` in one shared-basis pass, avoiding the 3x
      redundant Vandermonde builds that the separate ``.ev(dx=1)`` and
      ``.ev(dy=1)`` calls would do.
    * **Optional Numba JIT fastpath**: on the NumPy backend, if
      :mod:`numba` is importable, the combined evaluation drops into a
      ``@njit(parallel=True, fastmath=True)`` kernel that runs the
      Chebyshev recurrence inline per sample (no Vandermonde
      materialised).  Typical 3-10x speedup over the pure-NumPy path.
      Silently skipped on the CuPy backend -- the pure-xp fallback
      runs on GPU instead.

    GPU note
    --------
    To use this class on GPU with CuPy::

        import cupy as cp
        ev = _Cheb2DEvaluator(xs_in_cp, ys_in_cp, values_cp,
                              order=6, xp=cp)
        # Later:
        f, fx, fy = ev.ev_value_and_grad(xa_cp, ya_cp)

    All arrays (inputs, outputs, internal state) stay on the GPU;
    there is no implicit host-device copy.  The Newton loop in
    :func:`apply_real_lens_traced` is unchanged as long as ``xa, ya``
    are CuPy arrays.  A future ``use_gpu=True`` kwarg could dispatch
    this automatically.
    """

    __slots__ = ('order', 'coeffs', 'xmin', 'xmax', 'ymin', 'ymax',
                 '_mi', '_K1', '_K2', 'xp')

    def __init__(self, xs_in, ys_in, values, order=6, xp=None):
        if xp is None:
            xp = _get_array_module(values)
        self.xp = xp
        # The fit itself (a tiny lstsq -- typically a few hundred rows
        # by 28-70 terms) is always performed on CPU via NumPy, even
        # when xp=cupy.  Three reasons:
        #   1. NumPy lstsq is reliable and dependency-free; cupy.linalg.
        #      lstsq needs cuSOLVER which isn't guaranteed to be
        #      present on every cupy install.
        #   2. The fit is O(1) per apply_real_lens_traced call (one-
        #      time cost) and is negligible vs per-pixel Newton work.
        #      Routing it via the CPU has no measurable impact.
        #   3. The payoff from xp=cupy is in the Newton hot loop (N^2
        #      evaluator calls), where it does matter -- only the
        #      fitted coefficients need to live on the device.
        xs_np = np.asarray(xp.asnumpy(xs_in) if xp is not np else xs_in,
                            dtype=np.float64)
        ys_np = np.asarray(xp.asnumpy(ys_in) if xp is not np else ys_in,
                            dtype=np.float64)
        vals_np = np.asarray(xp.asnumpy(values) if xp is not np else values,
                              dtype=np.float64)
        self.order = int(order)
        # Scalars extracted as Python floats so chain-rule multiplies
        # stay backend-agnostic and don't pull host-device copies later.
        self.xmin = float(xs_np.min())
        self.xmax = float(xs_np.max())
        self.ymin = float(ys_np.min())
        self.ymax = float(ys_np.max())
        # Build total-degree multi-indices (kx, ky) with kx + ky <= order
        self._mi = [(kx, ky)
                     for kx in range(order + 1)
                     for ky in range(order + 1 - kx)]
        n_terms = len(self._mi)
        # Fit on CPU using NumPy
        X_np, Y_np = np.meshgrid(xs_np, ys_np, indexing='ij')
        u_np = (2.0 * X_np - (self.xmin + self.xmax)) / (self.xmax - self.xmin)
        v_np = (2.0 * Y_np - (self.ymin + self.ymax)) / (self.ymax - self.ymin)
        K1_np = np.asarray([m[0] for m in self._mi], dtype=np.int64)
        K2_np = np.asarray([m[1] for m in self._mi], dtype=np.int64)
        Tu_np = _cheb_vand_2d(u_np, order, np)
        Tv_np = _cheb_vand_2d(v_np, order, np)
        A_full = (Tu_np[K1_np] * Tv_np[K2_np]).reshape(n_terms, -1).T
        vals_flat = vals_np.ravel()
        finite = np.isfinite(vals_flat)
        if finite.all():
            A = A_full
            rhs = vals_flat
        else:
            A = A_full[finite, :]
            rhs = vals_flat[finite]
        c_np, *_ = np.linalg.lstsq(A, rhs, rcond=None)
        # Push coefficients + index arrays onto the target backend
        self.coeffs = xp.asarray(c_np, dtype=xp.float64)
        self._K1 = xp.asarray(K1_np, dtype=xp.int64)
        self._K2 = xp.asarray(K2_np, dtype=xp.int64)

    def _to_u(self, x):
        return (2.0 * x - (self.xmin + self.xmax)) / \
                 (self.xmax - self.xmin)

    def _to_v(self, y):
        return (2.0 * y - (self.ymin + self.ymax)) / \
                 (self.ymax - self.ymin)

    # ----------------------------------------------------------------
    # Backward-compat single-quantity API (RectBivariateSpline.ev()).
    # Supports dx=0/1 and dy=0/1 (up to first derivatives).
    # ----------------------------------------------------------------
    def ev(self, x, y, dx=0, dy=0):
        """Evaluate polynomial (or partial derivative) at (x, y).

        Compatible subset of SciPy RectBivariateSpline.ev: supports
        dx=0/1 and dy=0/1 (up to first derivatives).  When multiple
        derivatives are needed at the same (x, y), prefer
        :meth:`ev_value_and_grad` -- one call returns all three.
        """
        if (dx, dy) in ((0, 0), (1, 0), (0, 1)):
            f, fx, fy = self.ev_value_and_grad(x, y)
            if dx == 0 and dy == 0:
                return f
            if dx == 1 and dy == 0:
                return fx
            return fy
        raise NotImplementedError(
            f"_Cheb2DEvaluator.ev with dx={dx}, dy={dy} not supported; "
            f"only 0th and 1st derivatives in a single axis.")

    # ----------------------------------------------------------------
    # Combined value + gradient (#6) -- primary entry point for the
    # Newton loop in apply_real_lens_traced.  Shares Chebyshev basis
    # work across all three quantities.  Uses the Numba fastpath (#1)
    # when available on the NumPy backend; otherwise a pure-xp
    # implementation that runs on NumPy or CuPy alike.
    # ----------------------------------------------------------------
    def ev_value_and_grad(self, x, y):
        """Evaluate the polynomial and both partial derivatives in one
        pass.

        Returns
        -------
        f, df/dx, df/dy : arrays with the broadcast shape of (x, y)
            Value and physical-space partial derivatives (chain rule
            applied to undo the ``[-1, 1]`` normalisation).
        """
        xp = self.xp
        x = xp.asarray(x, dtype=xp.float64)
        y = xp.asarray(y, dtype=xp.float64)
        u = self._to_u(x)
        v = self._to_v(y)
        sx = 2.0 / (self.xmax - self.xmin)
        sy = 2.0 / (self.ymax - self.ymin)

        # Numba fastpath on the NumPy backend (kernel compiled lazily on first
        # use; None when numba is unavailable -> fall through to the pure-xp path)
        _cheb_kernel = (_get_cheb2d_val_grad_numba()
                        if xp is np and _NUMBA_AVAILABLE else None)
        if _cheb_kernel is not None:
            u_flat = np.ascontiguousarray(u.ravel(), dtype=np.float64)
            v_flat = np.ascontiguousarray(v.ravel(), dtype=np.float64)
            coeffs = np.ascontiguousarray(self.coeffs, dtype=np.float64)
            K1 = np.ascontiguousarray(self._K1, dtype=np.int64)
            K2 = np.ascontiguousarray(self._K2, dtype=np.int64)
            f_flat, fx_u_flat, fy_v_flat = _cheb_kernel(
                coeffs, K1, K2, u_flat, v_flat, self.order)
            shape = u.shape
            return (f_flat.reshape(shape),
                    fx_u_flat.reshape(shape) * sx,
                    fy_v_flat.reshape(shape) * sy)

        # Pure-xp fallback (always-on; REQUIRED for CuPy backend).
        # Build T and T' Vandermondes once, gather by multi-index, and
        # contract against the coefficient vector with one sum each.
        Tu = _cheb_vand_2d(u, self.order, xp)
        Tv = _cheb_vand_2d(v, self.order, xp)
        dTu = _cheb_deriv_vand_2d(u, self.order, xp)
        dTv = _cheb_deriv_vand_2d(v, self.order, xp)
        # Gather per-basis-term arrays: shape (M, ...u.shape)
        Tu_K = Tu[self._K1]
        Tv_K = Tv[self._K2]
        dTu_K = dTu[self._K1]
        dTv_K = dTv[self._K2]
        # Broadcast coefficients and sum over the basis-term axis.
        c_shape = (len(self._mi),) + (1,) * u.ndim
        c_b = self.coeffs.reshape(c_shape)
        f    = xp.sum(c_b * Tu_K  * Tv_K , axis=0)
        fx_u = xp.sum(c_b * dTu_K * Tv_K , axis=0)
        fy_v = xp.sum(c_b * Tu_K  * dTv_K, axis=0)
        return f, fx_u * sx, fy_v * sy


def _cheb_vand_2d(u, max_k, xp=None):
    """Chebyshev T_k(u) for k=0..max_k as (max_k+1,) + u.shape array.

    Backend-agnostic: pass ``xp=numpy`` (default) or ``xp=cupy`` to run
    on host or device respectively.
    """
    if xp is None:
        xp = _get_array_module(u)
    T = xp.empty((max_k + 1,) + u.shape, dtype=xp.float64)
    T[0] = 1.0
    if max_k >= 1:
        T[1] = u
    for n in range(2, max_k + 1):
        T[n] = 2.0 * u * T[n - 1] - T[n - 2]
    return T


def _cheb_deriv_vand_2d(u, max_k, xp=None):
    """T'_k(u) via T'_n = n U_{n-1}; shape (max_k+1,) + u.shape.

    Backend-agnostic: pass ``xp=numpy`` (default) or ``xp=cupy``.
    """
    if xp is None:
        xp = _get_array_module(u)
    Tp = xp.zeros((max_k + 1,) + u.shape, dtype=xp.float64)
    if max_k < 1:
        return Tp
    U = xp.empty((max_k + 1,) + u.shape, dtype=xp.float64)
    U[0] = 1.0
    if max_k >= 1:
        U[1] = 2.0 * u
    for n in range(2, max_k + 1):
        U[n] = 2.0 * u * U[n - 1] - U[n - 2]
    for n in range(1, max_k + 1):
        Tp[n] = float(n) * U[n - 1]
    return Tp


def _geometric_lens_phase(lens_prescription, wavelength, dx, N):
    """Compute the analytic per-surface sag-phase-screen sum for a lens.

    Returns the *geometric* component of the phase a plane wave would
    acquire after passing through the lens -- equivalent to
    ``np.angle(apply_real_lens(ones, ...))`` except that the ASM
    diffractive correction between surfaces is omitted.

    For smooth refractive lens prescriptions the omitted correction
    scales as ``t * k_perp^2 / (2k)`` where t is glass thickness and
    k_perp is the characteristic spatial-frequency of the sag.  On
    typical F/10+ refractive lenses this is under 10 nm OPL; for
    faster lenses (F/3 or below) validate before trusting.

    Parameters
    ----------
    lens_prescription : dict
        Same format as :func:`apply_real_lens`.
    wavelength : float
        Free-space wavelength [m].
    dx : float
        Grid spacing [m].
    N : int
        Grid size (N x N square).

    Returns
    -------
    phase : ndarray (N, N) float64
        Analytic geometric phase in radians, wrapped to the [-pi, pi]
        range so it can be used interchangeably with
        ``np.angle(E_analytic_pw)``.
    """
    from .. import raytrace as _rt
    surfaces = _rt.surfaces_from_prescription(lens_prescription)
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x, indexing='xy')
    k0 = 2.0 * np.pi / wavelength

    # v5.1.0 (default-knob resolver rollout): real-dtype OPL allocator
    # honours ``set_default_real_dtype(...)`` -- this is one of the
    # documented consumer wirings.  Falls back to np.float64 if the
    # propagators module is mid-load (defensive).
    try:
        from ..propagators.propagation import get_default_real_dtype
        _real_dtype = get_default_real_dtype()
    except ImportError:
        _real_dtype = np.float64
    phase = np.zeros((N, N), dtype=_real_dtype)

    # Accumulate per-surface sag phase: phi += -k0 * (n_after - n_before) * sag(x, y)
    # This matches the thin-element OPD used inside apply_real_lens's
    # phase-screen model (the default paraxial formula) -- so dropping
    # the ASM step is the only physics difference.
    for surf in surfaces:
        n1 = get_glass_index(surf.glass_before, wavelength)
        n2 = get_glass_index(surf.glass_after, wavelength)
        if abs(n2 - n1) < 1e-15:
            continue   # no refraction
        sag = _rt._surface_sag_xy(X, Y, surf)
        phase = phase + (-k0 * (n2 - n1) * sag)

    # Also add the bulk glass piston (constant k*n*t_i in each glass)
    # since the full apply_real_lens includes this via the ASM in-glass
    # propagation.  The piston is a rigid offset but keeping it
    # preserves absolute-phase consistency when this function is used
    # for the phase_analytic_lens reference.
    for surf in surfaces[:-1]:
        n_mid = get_glass_index(surf.glass_after, wavelength)
        phase = phase + k0 * n_mid * float(surf.thickness)

    # Wrap to match np.angle convention
    return np.angle(np.exp(1j * phase))


def _sample_local_tilts(E_in, wavelength, dx, entrance_x, entrance_y,
                         max_sin=0.5, smooth_sigma_px=4.0,
                         multimode_diagnostic=None):
    """Extract ``(L, M)`` direction cosines for each entrance ray from
    the local phase gradient of ``E_in``.

    For a field ``E_in = A(x,y) * exp(i*phi(x,y))``, the local wavevector
    at each pixel is ``k_local = grad(phi)``.  A ray launched from
    that pixel should carry direction cosines ``L = k_x / k0``,
    ``M = k_y / k0``, where ``k0 = 2*pi/wavelength``.

    We compute ``grad(phi)`` via the conjugate-product trick
    ``angle(E_shifted * conj(E))`` so the wrap-to-(-pi, pi] happens
    once per pair without a separate unwrap pass.  Low-amplitude
    pixels (below 0.1 % of peak) and NaN/inf phase are returned as
    zero tilt.  The final cosines are clipped to ``|L|, |M| <=
    max_sin`` for numerical safety.

    Why this function has to be careful
    -----------------------------------
    A single-mode field has ONE well-defined phase gradient at every
    pixel (plane wave, smooth Gaussian, MLA-tilted beamlet).  A
    multi-mode field -- a superposition of several plane-wave
    components like a post-DOE diffraction pattern -- has NO
    well-defined local direction: neighbouring pixels can report wildly
    different ``np.angle(E_shift * conj(E))`` values because the sum of
    components interferes coherently.  Feeding those aliased per-pixel
    directions straight into the entrance->exit spline in
    :func:`apply_real_lens_traced` produces a chaotic map that Newton
    cannot invert, resulting in an all-NaN OPL and a zero output field.

    Fix: **amplitude-weighted Gaussian smoothing of the tilt field**
    before it's returned.  The smoothing is a low-pass on the local
    wavevector, with the physical interpretation that a ray launched
    from an entrance pixel carries the *mean* direction of the wave
    components within a few-wavelength neighbourhood, rather than the
    single-pixel aliased fringe phase.

    *   Single-mode fields: the true tilt is a slowly-varying function
        of position, so a Gaussian of sigma a few pixels leaves it
        essentially unchanged.  MLA-modulated fields keep their
        per-beamlet tilts.
    *   Multi-mode fields: the tilt oscillates pixel-to-pixel with
        mean zero (for a balanced set of orders).  Gaussian smoothing
        pulls the tilt toward that zero mean, naturally degenerating
        to a classical collimated launch for post-DOE inputs.
    *   Amplitude weighting ensures low-amplitude pixels (between
        DOE orders, outside MLA beamlets, etc.) don't drag the
        smoothed tilt toward the noisy phase readings those pixels
        contribute.

    No threshold to tune, no global "reject" decision -- the smoothing
    is the universal fix.

    Parameters
    ----------
    smooth_sigma_px : float, default 4.0
        Gaussian smoothing radius (pixels) applied to the tilt field.
        Set to 0 to disable smoothing (pre-smoothing behaviour, NOT
        recommended for multi-mode inputs).  A few pixels is enough
        to suppress single-pixel aliasing while preserving tilts that
        vary on the scale of typical beam features (MLA beamlet
        diameters, Gaussian waists, etc.).
    multimode_diagnostic : dict, optional
        If provided, gets populated with tilt-field statistics before
        and after smoothing (``raw_rms_L``, ``smoothed_rms_L``,
        ``raw_rms_M``, ``smoothed_rms_M``, ``smoothing_ratio``).
        Useful for callers that want to log or verify the smoothing
        is doing what's expected.
    """
    k0 = 2.0 * np.pi / wavelength
    N_y, N_x = E_in.shape

    # Phase gradient: d(phi)/dx ~ angle(E[:, 1:] * conj(E[:, :-1])) / dx
    # Use np.roll so shapes match; the rolled-into-the-boundary pixels
    # get low weights after the amplitude mask.
    E_shift_x = np.roll(E_in, -1, axis=1)
    E_shift_y = np.roll(E_in, -1, axis=0)
    grad_phi_x = np.angle(E_shift_x * np.conj(E_in)) / dx
    grad_phi_y = np.angle(E_shift_y * np.conj(E_in)) / dx

    L_grid = grad_phi_x / k0
    M_grid = grad_phi_y / k0

    # Zero-out noise-floor pixels and boundary wrap
    amp = np.abs(E_in)
    amp_thresh = 1e-3 * float(amp.max()) if amp.size else 0.0
    mask = (amp > amp_thresh) & np.isfinite(L_grid) & np.isfinite(M_grid)
    L_grid = np.where(mask, L_grid, 0.0)
    M_grid = np.where(mask, M_grid, 0.0)

    # Statistics before smoothing -- for diagnostics and as the "raw"
    # baseline the smoothing is operating on.
    raw_rms_L = float(np.sqrt(np.mean(L_grid[mask] ** 2))) if mask.any() else 0.0
    raw_rms_M = float(np.sqrt(np.mean(M_grid[mask] ** 2))) if mask.any() else 0.0

    # ---- Amplitude-weighted Gaussian smoothing ---------------------
    # Low-pass the tilt field with an intensity-weighted kernel:
    #
    #     L_smooth = blur(|E|^2 * L) / blur(|E|^2)
    #     M_smooth = blur(|E|^2 * M) / blur(|E|^2)
    #
    # This averages neighbouring pixels' tilts using their amplitude
    # squared (intensity) as weights.  On a smooth single-mode field
    # this leaves L and M essentially unchanged (neighbours already
    # agree).  On a multi-mode superposition with pixel-to-pixel
    # aliased phase gradients, the oscillations average out and
    # amplitude-weighting discounts the low-amplitude interference
    # nulls where the phase is noisiest.  Low-amplitude regions
    # (between beamlets, outside the main field) where the raw
    # gradient is unreliable naturally decay to zero because both
    # numerator and denominator weight them out.
    if smooth_sigma_px > 0:
        from scipy.ndimage import gaussian_filter
        I = (amp * amp).astype(np.float64)
        sigma = float(smooth_sigma_px)
        num_L = gaussian_filter(I * L_grid, sigma=sigma, mode='nearest')
        num_M = gaussian_filter(I * M_grid, sigma=sigma, mode='nearest')
        den = gaussian_filter(I, sigma=sigma, mode='nearest')
        # Guard against division by zero far from the field support
        safe = den > (den.max() * 1e-6)
        L_grid = np.where(safe, num_L / np.where(safe, den, 1.0), 0.0)
        M_grid = np.where(safe, num_M / np.where(safe, den, 1.0), 0.0)

    smoothed_rms_L = float(np.sqrt(np.mean(L_grid[mask] ** 2))) if mask.any() else 0.0
    smoothed_rms_M = float(np.sqrt(np.mean(M_grid[mask] ** 2))) if mask.any() else 0.0
    if multimode_diagnostic is not None:
        multimode_diagnostic['raw_rms_L'] = raw_rms_L
        multimode_diagnostic['raw_rms_M'] = raw_rms_M
        multimode_diagnostic['smoothed_rms_L'] = smoothed_rms_L
        multimode_diagnostic['smoothed_rms_M'] = smoothed_rms_M
        # Ratio < 1 means smoothing reduced the tilt magnitude (i.e.
        # noise was averaged out); ratio ~= 1 means smoothing was a
        # no-op (field was already smooth).
        raw_mag = np.hypot(raw_rms_L, raw_rms_M)
        smoothed_mag = np.hypot(smoothed_rms_L, smoothed_rms_M)
        multimode_diagnostic['smoothing_ratio'] = (
            smoothed_mag / raw_mag if raw_mag > 0 else 1.0)

    # Clip to physical range -- rays with |sin(theta)| > max_sin are
    # unphysical for most lens designs and will overwhelm the Newton
    # fit domain.  After smoothing this clip typically never triggers,
    # but we keep it as a defence against pathological inputs.
    np.clip(L_grid, -max_sin, max_sin, out=L_grid)
    np.clip(M_grid, -max_sin, max_sin, out=M_grid)

    # Interpolate to launch positions (physical -> pixel index,
    # bilinear sample).  Launch positions outside the E_in grid
    # (|x| > N*dx/2) fall back to zero tilt (edge -- no information).
    from scipy.ndimage import map_coordinates
    pix_x = entrance_x / dx + N_x / 2.0
    pix_y = entrance_y / dx + N_y / 2.0
    coords = np.vstack([pix_y.ravel(), pix_x.ravel()])
    L = map_coordinates(L_grid, coords, order=1,
                        mode='constant', cval=0.0).reshape(entrance_x.shape)
    M = map_coordinates(M_grid, coords, order=1,
                        mode='constant', cval=0.0).reshape(entrance_x.shape)
    return L, M


def _reverse_prescription(prescription):
    """Build a prescription describing the same lens traversed in the
    backward direction.

    Used by the experimental backward-trace OPL inversion in
    :func:`apply_real_lens_traced`.  Reversing amounts to:

    *   Swap surface order.
    *   Negate every radius of curvature (curvature direction flips
        when viewed from the opposite side).  Conic constants and
        even-power aspheric coefficients are invariant under this
        reflection.
    *   Swap ``glass_before`` and ``glass_after`` on each surface.
    *   Reverse the thickness list (the gap AFTER surface i in the
        forward prescription is the gap BEFORE surface (N-1-i) in
        the reversed one, which is the same list read right-to-left).
    """
    surfaces = prescription['surfaces']
    thicknesses = prescription.get('thicknesses', [])
    rev_surfaces = []
    for s in reversed(surfaces):
        rs = dict(s)
        rs['radius'] = -rs['radius']
        if rs.get('radius_y') is not None:
            rs['radius_y'] = -rs['radius_y']
        rs['glass_before'], rs['glass_after'] = (
            rs['glass_after'], rs['glass_before'])
        rev_surfaces.append(rs)
    rev = {
        'surfaces': rev_surfaces,
        'thicknesses': list(reversed(thicknesses)),
    }
    if 'aperture_diameter' in prescription:
        rev['aperture_diameter'] = prescription['aperture_diameter']
    return rev


def _opl_by_backward_trace(E_analytic, lens_prescription, wavelength, dx,
                           N_grid, ray_subsample,
                           tilt_smooth_sigma_px=4.0):
    """Alternative to the Newton-based forward-map inversion in
    :func:`apply_real_lens_traced`.

    **Validation** (2026-04-18):

    *   Single-ray forward-vs-backward OPL on a plano-convex singlet:
        **< 1 pm** (machine-precision agreement) when the exit-vertex
        correction is applied to both ends.
    *   End-to-end ``apply_real_lens_traced`` OPD RMS vs the Newton
        path: **~35-40 nm** on singlets at N=512.  The residual is
        not a bug in the reversal; it comes from using the
        finite-difference phase gradient of ``E_analytic`` as the
        backward-launch direction estimate (Newton uses the
        forward-trace's exact entrance-plane direction).  For
        design-verification work at lambda/10 tolerance this is deep
        in the margin; for sub-nm precision use Newton.

    Measured speed at N=512: ~1.7x faster than Newton on a singlet.
    Scales better to large N because the work is ``O(N^2)`` rather
    than ``O(N^2 * newton_iters)``.

    Algorithm in brief:

    Instead of ray-tracing the entrance grid forward and then
    Newton-inverting the spline of that map to find each exit pixel's
    entrance ray, we trace rays BACKWARD from a coarse subsample of
    the exit grid through the lens to the entrance, accumulating
    OPL along the way.  Fermat's principle makes OPL path-reversible,
    so the backward-trace OPL is numerically the same as the
    forward-trace OPL up to a sign convention.

    The exit-plane ray directions are derived from the local phase
    gradient of ``E_analytic`` (same mechanism as the input-aware
    forward launch, just applied at the exit).  The
    amplitude-weighted Gaussian smoothing keeps this robust on
    multi-mode inputs.

    Advantages over the Newton path (when it works):
        *   No spline fit, no Newton iteration.  The entire
            computation is a single forward pass of ``trace()``
            through a reversed prescription plus interpolation
            of the OPL map to the wave grid.
        *   Embarrassingly parallel in the trace itself (no
            dependencies between rays).

    Disadvantages / caveats:
        *   Accuracy depends on how well the exit-plane direction
            is extracted from ``E_analytic``.  Near a focus the
            true direction varies rapidly and the smoothed
            gradient is less representative; Newton handles this
            via the spline without needing a direction estimate.
        *   Only tested on singlet and doublet geometries so far;
            compound systems with intermediate foci may behave
            unexpectedly.  **Labelled experimental.**
    """
    from ..raytrace import _make_bundle, surfaces_from_prescription, trace

    N = int(N_grid)
    sub = max(1, int(ray_subsample))
    # Coarse exit-plane sampling (same stride pattern as the Newton
    # path's ``X[::sub, ::sub]`` slice so the final interpolation
    # grids line up identically).
    idx_c = np.arange(0, N, sub)
    N_c = idx_c.size
    x_c = (idx_c - N / 2.0) * dx
    Xc, Yc = np.meshgrid(x_c, x_c)

    # Extract exit-plane direction cosines from the phase gradient
    # of E_analytic, smoothed per the 3.1.3 multi-mode fix.
    L_out, M_out = _sample_local_tilts(
        E_analytic, wavelength, dx, Xc, Yc,
        smooth_sigma_px=tilt_smooth_sigma_px)

    # Build the reversed prescription + its surface list.  Note:
    # surfaces_from_prescription uses the per-element semi-diameter
    # plus the prescription-level aperture_diameter for vignetting;
    # both carry through to the reverse automatically.
    rev_rx = _reverse_prescription(lens_prescription)
    rev_surfaces = surfaces_from_prescription(rev_rx)

    # Rays start at the exit vertex plane (z=0) with direction
    # cosines (-L_out, -M_out, +sqrt(1-L^2-M^2)).  The sign flip on
    # (L, M) accounts for tracing in the reversed-axis frame:
    # "forward" here == backward in the original frame.  _make_bundle
    # computes N = +sqrt(1-L^2-M^2) which is the correct "forward"
    # direction in the reversed frame.
    rays = _make_bundle(
        x=Xc.ravel(), y=Yc.ravel(),
        L=-L_out.ravel(), M=-M_out.ravel(),
        wavelength=wavelength,
    )
    result = trace(rays, rev_surfaces, wavelength)
    final = result.image_rays

    # ---- Exit-vertex correction on the backward trace ----
    # trace() leaves rays at z = sag(last_surface_in_reversed_frame)
    # = sag of original S1 (the original entrance-side vertex) in
    # the reversed frame.  Without propagating each ray to z=0 of
    # this reversed-frame last surface (the original entrance
    # vertex plane), we under-count the OPL by the
    # vertex-to-sag leg in the final medium -- exactly the same
    # correction the forward path applies in apply_real_lens_traced
    # at lenses.py:1548-1556.  For on-axis rays this is zero; for
    # marginal rays on a strong-curvature lens it's tens of nm to
    # hundreds of nm.  Missing this is what made the first draft
    # of this function disagree with Newton by ~343 nm RMS.
    rev_surfaces_list = rev_surfaces
    n_exit_backward = get_glass_index(
        rev_surfaces_list[-1].glass_after, wavelength)
    with np.errstate(divide='ignore', invalid='ignore'):
        t_to_vertex = np.where(
            final.alive & (np.abs(final.N) > 1e-30),
            -final.z / final.N, 0.0)
    final.opd = final.opd + n_exit_backward * t_to_vertex
    # (We don't actually need to update x/y/z since we only
    # consume final.opd downstream, but keep it consistent.)
    final.x = final.x + final.L * t_to_vertex
    final.y = final.y + final.M * t_to_vertex
    final.z = np.zeros_like(final.z)

    # OPL: set NaN for dead rays (TIR / vignetted during the
    # reverse trace) so downstream NaN-propagation matches the
    # Newton path's treatment of out-of-domain points.
    opl_flat = np.where(final.alive, final.opd, np.nan)
    opl_coarse = opl_flat.reshape(Xc.shape)

    # Reference to on-axis so the returned OPL has the same origin
    # as the Newton path.  (Forward Newton does this at the spline
    # fit step via ``opl_grid = opl_grid - opl_grid[i_axis, i_axis]``.)
    i_c = N_c // 2
    ref = opl_coarse[i_c, i_c]
    if np.isfinite(ref):
        opl_coarse = opl_coarse - ref

    # Interpolate coarse OPL to the full wave grid, with the same
    # mode='nearest' + NaN-majority masking the Newton path uses.
    from scipy.ndimage import map_coordinates
    ii, jj = np.indices((N, N), dtype=np.float64)
    coords = np.array([ii * N_c / N, jj * N_c / N])
    opl_map = map_coordinates(
        np.where(np.isnan(opl_coarse), 0.0, opl_coarse),
        coords, order=1, mode='nearest')
    nan_coarse = np.isnan(opl_coarse).astype(np.float64)
    nan_full = map_coordinates(
        nan_coarse, coords, order=1, mode='nearest')
    opl_map = np.where(nan_full > 0.5, np.nan, opl_map)
    return opl_map


def apply_real_lens_traced(
    E_in: np.ndarray,
    *,
    prescription: Dict[str, Any],
    wavelength: float,
    dx: float,
    dy: Optional[float] = None,
    bandlimit: bool = True,
    ray_subsample: int = 8,
    n_workers: Optional[int] = None,
    progress: Optional[Any] = None,
    min_coarse_samples_per_aperture: int = 32,
    on_undersample: str = 'error',
    preserve_input_phase: bool = True,
    tilt_aware_rays: bool = False,
    parallel_amp: Optional[bool] = None,
    parallel_amp_min_free_gb: float = 48.0,
    newton_amp_mask_rel: float = 1e-4,
    newton_mask_dilate_coarse_px: int = 2,
    newton_max_iters: Optional[int] = None,
    inversion_method: str = 'newton',
    fast_analytic_phase: bool = False,
    newton_fit: str = 'polynomial',
    newton_poly_order: int = 6,
    use_gpu: bool = False,
    amp_use_gpu: bool = False,
    wave_propagator: Optional[str] = None,
    sag_dtype: Optional[Any] = None,
    sag_chunk_rows: Optional[int] = None,
) -> np.ndarray:
    """Wave + per-pixel ray-traced phase variant of :func:`apply_real_lens`.

    See Also
    --------
    apply_real_lens :
        Faster (3-10x) analytic split-step model.  Use as the default
        when sub-nm OPD on multi-surface curved-interface systems
        isn't required and a coarser grid is preferable.
    apply_real_lens_maslov :
        Phase-space propagator via Chebyshev polynomial fit of the
        canonical map.  Caustic-safe and differentiable; preferable
        for JAX-autodiff optimisation loops and for output planes at
        or near a caustic.

    Quick decision guide
    --------------------
    * Default / fast wave model -> ``apply_real_lens``.
    * Sub-nm OPD on cemented doublets / multi-surface curved interfaces
      -> ``apply_real_lens_traced`` (this function).
    * Inside a JAX-autodiff design optimisation, or near a caustic
      -> ``apply_real_lens_maslov`` / ``apply_real_lens_maslov_jax``.

    Description
    -----------
    For each pixel of the simulation grid, a geometric ray is launched
    from the entrance plane straight through the prescription using the
    sequential ray tracer in :mod:`lumenairy.raytrace`.  The
    accumulated optical path length (OPL) per pixel is used as the
    exit-plane phase, while the wave's *amplitude* envelope (vignetting,
    diffraction, edge effects) comes from a single ASM propagation of
    the entrance aperture to the exit-vertex plane.

    This eliminates the uniform-glass-slab approximation that limits
    the closed-form thin-element model on cemented doublets and other
    multi-surface curved-interface systems: each pixel sees the
    geometrically-correct glass path for its (x,y) position.  In
    practice the OPD agrees with the geometric ray trace to the
    sampling limit of the grid, at the cost of one ray trace per
    pixel (~3-10x slowdown relative to the analytic phase-screen
    model).

    Critical sampling rule
    ----------------------
    Extracting OPD from a converging wavefront requires

        dx <= lambda * f / aperture

    where ``f`` is the back focal length and ``aperture`` is the pupil
    diameter.  Coarser sampling makes ``np.unwrap`` lose cycles at the
    pupil edge, giving catastrophically wrong OPD values there.  Run
    :func:`lumenairy.analysis.check_opd_sampling` before a
    large ``apply_real_lens_traced`` call to verify.  If a coarser
    grid is required, use :func:`apply_real_lens` (with
    ``seidel_correction=True`` for doublets) instead.

    Limitations
    -----------
    * Assumes the input field is approximately a collimated plane wave
      (each pixel ray launched parallel to z).  For converging or
      tilted input, fall back to :func:`apply_real_lens`.
    * Replaces the wave's exit phase with the geometric OPL; this
      gives correct OPD by construction but bypasses any wave-physics
      phase content that the ASM would have introduced (negligible for
      typical lens systems but worth noting).
    * Fresnel transmission and absorption are NOT applied here -- if
      you need them, run both this function and
      :func:`apply_real_lens` and combine.

    Parameters
    ----------
    E_in : ndarray, complex, shape (N, N)
    lens_prescription : dict
        Same format as :func:`apply_real_lens`.
    wavelength : float
    dx : float
        Grid spacing [m] (square pixels assumed for the traced model).
    dy : float, optional
        Grid spacing in y [m].  Defaults to ``dx``.  Accepted for API
        symmetry with the rest of the lens family; the traced ray-
        subsample / interpolation paths currently require ``dy == dx``
        and will raise otherwise.  Use :func:`apply_real_lens` for
        anamorphic grids.
    bandlimit : bool, default True
        Passed to the (single) ASM propagation used for amplitude
        evolution.
    ray_subsample : int, default 1
        Compute the ray-trace OPL on every ``ray_subsample``-th pixel
        and bilinearly interpolate to the full grid.  OPL is a very
        smooth function of pupil position, so ``ray_subsample=4``
        typically loses < 1 nm of fidelity while cutting cost ~16x.
        Recommended for production use on large grids.
    min_coarse_samples_per_aperture : int, default 32
        Guardrail against undersampled Newton inversion.  After
        ``ray_subsample`` is applied, the coarse output grid must have
        at least this many samples spanning the lens aperture (or
        ``launch_radius`` if no explicit aperture is set), otherwise
        the cubic-spline interpolation of the wavefront will alias and
        the result will be wrong.

        Empirical scaling on a singlet at lambda = 1.31 um:

        ====================  ==================
        coarse-samples / ap   typical RMS phase
        ====================  ==================
        64                    ~20 nm
        32 (default safe)     ~85 nm
        16                    ~350 nm  (unusable)
        ====================  ==================

        Pass ``0`` to disable the check entirely.
    on_undersample : ``'error'`` (default) / ``'warn'`` / ``'silent'``
        What to do when the coarse-sample count falls below
        ``min_coarse_samples_per_aperture``.  ``'error'`` raises
        ``ValueError`` with the safe ``ray_subsample`` value computed
        for the current grid; ``'warn'`` logs via the ``warnings``
        module and continues; ``'silent'`` is the explicit "I know
        what I'm doing" escape hatch.
    n_workers : int, optional
        Number of worker *processes* for the Newton-inversion step.
        Defaults to
        :func:`lumenairy._backends.available_cpus` -- the
        affinity-aware count of CPUs this process can actually use
        (respects cgroup limits, ``taskset`` masks, Python 3.13+
        ``process_cpu_count``, Windows process affinity).  Pass 1 to
        force the in-process serial path (useful for reproducible
        timings or when called from a parent pool that already
        saturates the machine).
    tilt_aware_rays : bool, default False
        If True, each ray's initial direction ``(L, M)`` is derived
        from the local phase gradient of ``E_in`` at the entrance
        position (the "Tier 1 input-aware ray launch" added in 3.1.2).
        If False (the default), collimated rays are launched
        (L = M = 0 everywhere) and the plane-wave lens-OPL reference
        is used.

        **Why the default flipped from True to False in 3.1.3:**  When
        ``preserve_input_phase=True`` (also the default), the exit
        field is assembled as

            E_out = E_analytic * exp(i * delta_phase)
            delta_phase = k0 * opl_traced - phase_analytic_lens

        where ``phase_analytic_lens`` is the phase produced by running
        :func:`apply_real_lens` on a unit PLANE WAVE -- i.e. a
        plane-wave reference.  For ``delta_phase`` to be a
        mathematically clean "ray-traced minus analytic" correction,
        ``opl_traced`` must use the same reference: a plane-wave
        entrance launch.  With ``tilt_aware_rays=True``, ``opl_traced``
        instead mixes the lens-model correction with per-pixel
        tilt-induced phase shifts that the plane-wave ``phase_analytic_lens``
        does not contain.  The resulting ``delta_phase`` is only
        approximately right for small/uniform input tilts, and breaks
        materially on multi-mode inputs (post-DOE fields, strongly
        off-axis compound beams) where the per-pixel tilts vary
        significantly across the pupil.

        The 3.1.4 default ``tilt_aware_rays=False`` restores the
        reference-consistent plane-wave launch that pre-3.1.2 releases
        used, so ``delta_phase`` remains well-defined for any input the
        wave model can represent.  If you have a specifically small,
        uniform input tilt and want the per-ray OPL variation (e.g.
        rigorous off-axis lens characterisation with a single tilted
        input), pass ``tilt_aware_rays=True`` explicitly and validate
        against the default on your specific case.

        When this flag is True, tilts are clipped to
        ``|sin(theta)| <= 0.5`` (~30 deg) for numerical safety and
        amplitude-weighted-Gaussian-smoothed (``smooth_sigma_px=4``
        by default inside :func:`_sample_local_tilts`) to tame
        multi-mode aliasing; neither applies when the flag is False
        (the default).

    preserve_input_phase : bool, default True
        If True, the input field's phase structure (source tilts,
        MLA / DOE phase modulation, off-axis wavefronts, etc.) is
        preserved through the lens and combined with the ray-traced
        OPL correction.  This is the physically-correct behaviour
        and matches what :func:`apply_real_lens` does (with the added
        benefit of corrected geometric OPL).

        If False (legacy behaviour prior to v3.1.2), the output is
        ``|E_analytic| * exp(i*k0*OPL_traced)`` -- the input-field
        phase is discarded entirely and only the lens's ray-traced
        OPL is retained.  Use this mode when you specifically want
        the lens-only OPD response on a synthetic plane wave;
        otherwise keep the default.

        Cost: ``preserve_input_phase=True`` runs the analytic
        apply_real_lens *twice* (once for the input field, once for
        a unit plane-wave reference so we can subtract the analytic
        lens phase before adding the traced one).  This roughly
        doubles the ~40 % amplitude-leg budget.  At large N the
        total overhead is ~20 %.

        Implementation note: the work is dispatched via
        ``concurrent.futures.ProcessPoolExecutor`` rather than threads
        because SciPy's ``RectBivariateSpline.ev`` does not release
        the GIL in current versions, so threading delivers no
        speedup.  Each worker rebuilds the splines locally from
        their knot data (cheap), avoiding the pickle cost of the
        spline objects themselves.  Sequential fallback is used when
        the coarse grid is below ~200 k pixels (pool startup cost
        dominates) or when pool spawn fails.  Measured speedup on
        large grids: ~8x on 16 cores.

    Returns
    -------
    E_out : ndarray, complex, shape (N, N)
        Field at the exit-vertex plane of the last surface.
    """
    # v4.15.3 (P0-NEW-F2-1): defensive guard via the shared
    # ``_check_2d_scalar_field`` helper -- siblings missed by the
    # v4.15.2 closure now share the same first-line guard.
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E_in, 'apply_real_lens_traced')

    # v5.1.0 (default-knob resolver rollout): resolve ``wave_propagator``
    # / ``dy`` from the library-wide defaults when callers leave them
    # at the ``None`` sentinel.  Explicit values bypass the resolver.
    if wave_propagator is None:
        from ..propagators.propagation import get_default_wave_propagator
        wave_propagator = get_default_wave_propagator()
    if dy is None:
        from ..propagators.propagation import get_default_dy
        dy = get_default_dy()
        if dy is None:
            dy = dx

    # 4.12.0 (B2-5): explicit mirror-in-surfaces guard.  The shared
    # ``_check_no_silent_fold_drop`` only looks at the prescription's
    # ``elements`` list (the full element sequence, populated by
    # ``load_zemax_zmx``); a hand-built prescription that puts a
    # mirror directly into ``surfaces`` (via ``is_mirror=True`` or
    # ``glass_after='MIRROR'``) would slip past the shared check, and
    # the ray-traced OPL leg would silently treat the mirror as a
    # refractor with the wrong sign.  Fail loudly with a
    # mirror-specific message before the trace begins.
    _surfaces_list = prescription.get('surfaces') or []
    _mirror_surf_idx = []
    for _i, _s in enumerate(_surfaces_list):
        if not isinstance(_s, dict):
            continue
        _gl_after = _s.get('glass_after')
        _is_mirror = bool(_s.get('is_mirror', False)) or (
            isinstance(_gl_after, str)
            and _gl_after.upper() == 'MIRROR'
        )
        if _is_mirror:
            _mirror_surf_idx.append(_i)
    if _mirror_surf_idx:
        raise ValueError(
            f"apply_real_lens_traced: prescription has "
            f"{len(_mirror_surf_idx)} mirror surface(s) at "
            f"indices {_mirror_surf_idx} -- apply_real_lens_traced "
            f"only walks refracting surfaces.  Running this "
            f"prescription as-is would silently treat the mirror as "
            f"a refractor (wrong sign / wrong focusing phase) and "
            f"propagate along the unfolded-equivalent axis.  Use "
            f"the per-segment trace + apply_mirror pattern for "
            f"folded designs: call "
            f"lumenairy.io.split_prescription_at_mirrors(rx) to "
            f"split the prescription at each fold, then alternate "
            f"apply_real_lens_traced (each segment) with "
            f"apply_mirror (each fold).  See Guide-Folded-Designs "
            f"section 'Wave-optics through a fold'.")

    # Folded-design silent-drop guard: same as apply_real_lens.
    from ._lens_real import _check_no_silent_fold_drop
    _check_no_silent_fold_drop(
        prescription, fn_name='apply_real_lens_traced')

    # Internal references keep the legacy local name to avoid a
    # sprawling rename across this 1500-line function body.
    lens_prescription = prescription

    # Local import to avoid a circular dep at module load time
    from ..raytrace import (
        _make_bundle,
        surfaces_from_prescription,
        trace,
    )

    call_progress(progress, 'real_lens_traced', 0.0, 'initialising')

    # v5.3.2 (ROADMAP logging adoption sweep -- per-iteration telemetry):
    # Entry log -- grid size + surface count + Newton iter cap so users
    # who attach a handler can see the call shape at a glance.  The
    # actual Newton-cap value is resolved further down (caller override
    # > module default); we reproduce that resolution here so the entry
    # log reports what the run will actually use.
    _ny_entry, _nx_entry = np.shape(E_in)
    _n_surfaces_entry = len(prescription.get('surfaces') or [])
    _newton_cap_entry = (int(newton_max_iters)
                         if newton_max_iters is not None
                         else _NEWTON_MAX_ITERS)
    logger.info(
        "apply_real_lens_traced: entry N=%dx%d n_surfaces=%d "
        "newton_max_iters=%d ray_subsample=%d",
        int(_ny_entry), int(_nx_entry), int(_n_surfaces_entry),
        int(_newton_cap_entry), int(ray_subsample))

    # Pre-flight grid vs prescription-aperture check.
    try:
        _warn_if_aperture_exceeds_grid(
            lens_prescription, int(np.shape(E_in)[0]), dx,
            source='apply_real_lens_traced')
    except (KeyError, ValueError, TypeError, AttributeError):
        # Aperture-check failure is informational only.
        pass

    Ny, Nx = E_in.shape
    if Ny != Nx:
        raise ValueError("apply_real_lens_traced requires a square grid")
    N = Nx

    if dy is None:
        dy = dx
    # The traced variant's ray-subsample + interpolation paths assume
    # a square, isotropic grid.  Anamorphic (dy != dx) propagation is
    # supported by :func:`apply_real_lens` and
    # :func:`apply_real_lens_maslov`; for the traced model, pass
    # equal dx + dy or fall back to the analytic model.
    if abs(float(dy) - float(dx)) > 1e-15 * max(abs(float(dx)), 1.0):
        raise ValueError(
            "apply_real_lens_traced currently requires square pixels "
            f"(dx == dy); got dx={dx!r}, dy={dy!r}.  Use apply_real_lens "
            "for anamorphic grids.")

    aperture = lens_prescription.get('aperture_diameter')
    thicknesses = lens_prescription['thicknesses']
    float(sum(thicknesses))

    # 4.11.2: warn if the prescription specifies a stop_index other than
    # the entrance (or carries a decentered stop).  ``apply_real_lens``
    # honours ``stop_index`` (the aperture is applied at the indicated
    # surface, possibly off-axis); but ``apply_real_lens_traced``'s
    # ray-tracing path launches rays from the entrance plane and the
    # final exit-aperture mask uses the entrance-aperture diameter.
    # Porting the per-surface stop-application logic into the ray-trace
    # leg is feature-scope; warn so the silent move-to-entrance is
    # visible to callers who have written a stop_index into their
    # prescription.
    _stop_index = lens_prescription.get('stop_index')
    if _stop_index is not None and int(_stop_index) != 0:
        import warnings
        warnings.warn(
            f"apply_real_lens_traced: prescription specifies "
            f"stop_index={_stop_index}, but the ray-traced phase leg "
            "launches rays from the entrance pupil only; the aperture "
            "stop is effectively applied at the entrance (index 0).  "
            "For physically-correct stop behaviour on a non-entrance "
            "stop, use apply_real_lens.",
            RuntimeWarning, stacklevel=2,
        )
    else:
        # Decentered entrance stop: warn similarly -- the inner amp
        # path applies the stop centred at the surface's ``decenter``,
        # but the ray-trace leg's launch geometry is centred on the
        # optical axis.
        _surfs = lens_prescription.get('surfaces') or []
        if _surfs:
            _stop_surf_idx = int(_stop_index) if _stop_index is not None else 0
            if 0 <= _stop_surf_idx < len(_surfs):
                _dec = _surfs[_stop_surf_idx].get('decenter') or (0.0, 0.0)
                if _dec[0] != 0.0 or _dec[1] != 0.0:
                    import warnings
                    warnings.warn(
                        f"apply_real_lens_traced: stop surface "
                        f"{_stop_surf_idx} has decenter={_dec}; the "
                        "ray-traced phase leg uses an on-axis launch "
                        "geometry and will not see the off-axis stop "
                        "correctly.  Use apply_real_lens for "
                        "decentered-stop systems.",
                        RuntimeWarning, stacklevel=2,
                    )

    x = (np.arange(N) - N / 2) * dx
    # Opt-in row-band (chunked) FINAL ASSEMBLY: when ``sag_chunk_rows`` is set
    # (and the standard sub>1 Newton path is active), the OPL upsample +
    # delta-phase + exit-field assembly run in row bands, so the full-grid
    # float64 stack (ii/jj indices, the (2,N,N) map_coordinates input,
    # opl_map, nan_full, delta_phase, the complex128-first phase_exp) never
    # materialises -- only (chunk_rows x N) bands.  Values are byte-identical
    # to the whole-grid path (map_coordinates order-1 is pointwise in the
    # output; the phase/mask algebra is elementwise) -- pinned by
    # test_chunked_assembly_byte_identical.  The full X/Y meshgrids are not
    # built on this path: the Newton coarse grid comes from the 1-D x
    # subsample and the exit-aperture mask is banded.
    _chunk_assembly = (
        sag_chunk_rows is not None and int(sag_chunk_rows) > 0
        and max(1, int(ray_subsample)) > 1
        and inversion_method == 'newton'
    )
    if _chunk_assembly:
        X = Y = None
    else:
        X, Y = np.meshgrid(x, x)

    # ----- Step 1: amplitude envelope from the ANALYTIC lens model -----
    #
    # WHY WE CALL apply_real_lens HERE (the "double call"):
    #
    # apply_real_lens_traced is a HYBRID method.  It combines:
    #   (a) AMPLITUDE from wave optics — diffraction, vignetting, and
    #       the physically correct in-glass beam evolution (Fresnel
    #       effects at curved surfaces, edge ripples, aperture clipping)
    #   (b) PHASE from geometric ray tracing — the exact OPL through
    #       every curved glass/air interface, per pixel, via vector
    #       Snell's law at each surface
    #
    # The thin-element model's accuracy limitation is in its PHASE
    # (it approximates curved surfaces as phase screens at a single
    # z-plane), NOT in its amplitude (ASM through a uniform glass slab
    # handles diffraction correctly).  So we:
    #   1. Run apply_real_lens to get the full exit-plane field
    #   2. Keep only |E| (amplitude) — the wave-optics part
    #   3. Replace the phase with the geometrically exact ray-traced
    #      OPL map computed in Step 2 below
    #
    # This gives sub-nanometre OPD agreement with the geometric ray
    # trace (the "truth") while retaining physically correct
    # diffraction effects that pure ray tracing cannot capture.
    #
    # An earlier version used a simple air-ASM for the amplitude,
    # which produced a ~3.5 mm focus offset because air propagation
    # ≠ glass propagation (different wavenumber k = n·k0).  Using
    # apply_real_lens for the amplitude solves this because it
    # propagates through the correct glass/air refractive index
    # sequence.
    # Allocate 40% of the budget to the amplitude (which runs a full
    # apply_real_lens with its own per-surface cost), 50% to the ray
    # trace + Newton inversion, and 10% to the final field assembly.
    # ---------- Parallelism decision for amp and amp(pw) --------------
    # The two apply_real_lens calls (``amp`` on the real input, and
    # ``amp(pw)`` on a unit plane wave to recover the analytic lens
    # OPD) are data-independent and can run concurrently.  We dispatch
    # them on a ThreadPoolExecutor so the non-FFT work (sag, phase
    # screens, numexpr-fused multiplies, glass-interval setup)
    # overlaps.  The pyFFTW plan cache in ``propagation._fft2`` /
    # ``_ifft2`` holds a per-plan lock so the actual FFT execution
    # serialises safely on the shared aligned buffer; overlap is
    # therefore bounded by the FFT share of each call (~45-50 %) but
    # still gives ~40 % wall-time savings on the combined amp step.
    #
    # Memory cost of parallelism: two E fields and two sets of lens
    # intermediates alive simultaneously (~2x the peak of a single
    # call).  The ``parallel_amp_min_free_gb`` guard drops back to
    # sequential execution when available RAM is too tight for this
    # doubled working set -- tuned for the N=32768 complex128 case,
    # where the single-call transient peak is ~25 GB and doubling
    # brings it to ~50 GB.
    # parallel_amp=None (the default) resolves to the module global, letting
    # ``set_lens_parallel_amp(False)`` / ``set_low_memory(True)`` flip the
    # default for callers that don't pass the kwarg.  Explicit True/False win.
    if parallel_amp is None:
        parallel_amp = _LENS_PARALLEL_AMP_DEFAULT
    _use_parallel_amp = (preserve_input_phase and parallel_amp)
    if _use_parallel_amp:
        try:
            import psutil as _psutil
            _free_gb = _psutil.virtual_memory().available / 1e9
            if _free_gb < parallel_amp_min_free_gb:
                _use_parallel_amp = False
        except (ImportError, AttributeError, OSError):
            # psutil missing or virtual_memory query failed --
            # leave parallel_amp enabled but the user can still
            # force off via the kwarg.
            pass

    amp_cb = ProgressScaler(progress, 'real_lens_traced',
                            lo=0.0, hi=0.50 if _use_parallel_amp else 0.40)

    if _use_parallel_amp:
        # Parallel path: run amp and amp(pw) concurrently.  Only the
        # amp call reports progress (0-50%); amp(pw) runs silently to
        # avoid interleaved status lines from two threads.  The ones-
        # like plane wave is materialised outside the thread so the
        # 17 GB allocation happens once, synchronously, with clear
        # OOM semantics.
        from concurrent.futures import ThreadPoolExecutor

        def _amp_call():
            return apply_real_lens(
                E_in, prescription=lens_prescription, wavelength=wavelength, dx=dx,
                bandlimit=bandlimit, use_gpu=amp_use_gpu,
                wave_propagator=wave_propagator,
                sag_dtype=sag_dtype, sag_chunk_rows=sag_chunk_rows,
                progress=lambda stage, frac, msg='':
                    amp_cb(frac, f'amp: {msg}'))

        if fast_analytic_phase and preserve_input_phase:
            # Skip the full amp(pw) ASM pass; compute the geometric
            # lens phase analytically from per-surface sag.
            E_analytic = _amp_call()
            # np.abs and np.angle work on cupy arrays via __array_function__
            # in recent numpy; but to be explicit, use xp.abs/xp.angle via
            # the module selector below.
            _xp = cp if _is_cupy_array(E_analytic) else np
            amp = _xp.abs(E_analytic)
            phase_analytic_lens = _geometric_lens_phase(
                lens_prescription, wavelength, dx, E_in.shape[0])
            if _xp is cp:
                phase_analytic_lens = cp.asarray(phase_analytic_lens)
        else:
            ones_input = np.ones_like(E_in)

            def _amp_pw_call():
                return apply_real_lens(
                    ones_input, prescription=lens_prescription, wavelength=wavelength, dx=dx,
                    bandlimit=bandlimit, use_gpu=amp_use_gpu,
                    wave_propagator=wave_propagator,
                    sag_dtype=sag_dtype, sag_chunk_rows=sag_chunk_rows,
                    progress=None)

            with ThreadPoolExecutor(max_workers=2,
                                    thread_name_prefix='rlt_amp') as _tp:
                fut_amp = _tp.submit(_amp_call)
                fut_pw = _tp.submit(_amp_pw_call)
                E_analytic = fut_amp.result()
                E_analytic_pw = fut_pw.result()
            del ones_input
            _xp = cp if _is_cupy_array(E_analytic) else np
            amp = _xp.abs(E_analytic)
            phase_analytic_lens = _xp.angle(E_analytic_pw)
            del E_analytic_pw  # free ~17 GB at N=32768 before Newton starts
    else:
        # Sequential fallback (preserve_input_phase=False or RAM tight).
        E_analytic = apply_real_lens(
            E_in, prescription=lens_prescription, wavelength=wavelength, dx=dx, bandlimit=bandlimit,
            use_gpu=amp_use_gpu, wave_propagator=wave_propagator,
            sag_dtype=sag_dtype, sag_chunk_rows=sag_chunk_rows,
            progress=lambda stage, frac, msg='': amp_cb(frac, f'amp: {msg}'))
        _xp = cp if _is_cupy_array(E_analytic) else np
        amp = _xp.abs(E_analytic)
        # When preserving input phase (the physically-correct default),
        # we also need to know the *analytic model's lens-only phase* so
        # we can subtract it out before adding the ray-traced OPL back in.
        # We extract it by running apply_real_lens on a unit plane wave --
        # the result's phase is exactly the analytic lens's OPL
        # (plus small wave-propagation-through-glass effects) applied to
        # a flat input.
        if preserve_input_phase:
            if fast_analytic_phase:
                # Analytic geometric phase: per-surface sag phase
                # screens summed locally, no ASM through glass.  On
                # Design 51 lenses this introduces at most ~10 nm OPL
                # error (L4, F/6.8 doublet) and essentially none on
                # slower singlets -- below the numerical noise floor
                # of the rest of the pipeline.
                phase_analytic_lens = _geometric_lens_phase(
                    lens_prescription, wavelength, dx, E_in.shape[0])
                if _xp is cp:
                    phase_analytic_lens = cp.asarray(phase_analytic_lens)
            else:
                analytic_pw_cb = ProgressScaler(progress, 'real_lens_traced',
                                                 lo=0.40, hi=0.50)
                E_analytic_pw = apply_real_lens(
                    np.ones_like(E_in), prescription=lens_prescription, wavelength=wavelength, dx=dx,
                    bandlimit=bandlimit, use_gpu=amp_use_gpu,
                    wave_propagator=wave_propagator,
                    sag_dtype=sag_dtype, sag_chunk_rows=sag_chunk_rows,
                    progress=lambda stage, frac, msg='':
                        analytic_pw_cb(frac, f'amp(pw): {msg}'))
                phase_analytic_lens = _xp.angle(E_analytic_pw)
                del E_analytic_pw
        else:
            phase_analytic_lens = None
    # When amp_use_gpu=True the amp pipeline returns CuPy arrays.  The
    # rest of apply_real_lens_traced (ray trace, Newton, final E_out
    # assembly) is CPU-only, so pull the amp outputs back to the host
    # here rather than xp-ifying the final-assembly section.
    if _is_cupy_array(E_analytic):
        E_analytic = cp.asnumpy(E_analytic)
    if _is_cupy_array(amp):
        amp = cp.asnumpy(amp)
    if phase_analytic_lens is not None and _is_cupy_array(phase_analytic_lens):
        phase_analytic_lens = cp.asnumpy(phase_analytic_lens)
    call_progress(progress, 'real_lens_traced', 0.40,
                  'ray-tracing exit pupil')

    # ----- Step 2: ray-traced OPL per (subsampled) pixel ---------------
    # Launch a dense grid of rays from the entrance pupil; each ray
    # bends through the lens and lands at a *different* (x_out, y_out)
    # at the exit plane.  We need OPL associated with the exit
    # position (matching the wave's exit-plane grid), not the entrance
    # position, so we scatter-interpolate ``opl(x_out, y_out)`` onto
    # the wave grid.
    #
    # IMPORTANT: build surfaces WITHOUT the prescription aperture so
    # rays launched slightly beyond the entrance pupil are not
    # vignetted -- they may end up landing *inside* the wave grid
    # after refraction-induced inward shift.  But we DO restrict the
    # entrance launch positions to a modest over-margin around the
    # actual aperture so ultra-marginal rays (at huge angles of
    # incidence on the first surface) don't contaminate the OPL
    # function with non-paraxial branches.  The wave amplitude mask is
    # applied separately and zeros any spurious phase outside the
    # physical aperture anyway.
    pres_no_ap = dict(lens_prescription)
    pres_no_ap.pop('aperture_diameter', None)
    surfaces = surfaces_from_prescription(pres_no_ap)

    sub = max(1, int(ray_subsample))
    # Pick the launch radius: aperture (if specified) plus a 50 %
    # over-margin so that the entrance-grid sampling covers all wave-
    # grid exit positions even for fast lenses (rays bend inward so
    # exit positions are closer to axis than entrance).
    if aperture is not None:
        launch_radius = 0.5 * aperture * 1.50
    else:
        launch_radius = 0.5 * N * dx

    # ----- Subsampling guardrail --------------------------------------
    # The Newton-inversion step builds a cubic-spline interpolant of the
    # entrance->exit map on a coarse grid and uses bilinear interp to
    # back-fill the full grid.  If the coarse grid is too sparse
    # relative to the lens aperture the interpolant aliases and the
    # whole exit-pupil OPD is wrong (RMS phase err blows up roughly
    # as (samples_per_aperture)^-2 from the benchmark sweep -- 32
    # samples gives ~85 nm at lambda = 1.31 um, 16 samples gives ~350
    # nm and is unusable).
    if min_coarse_samples_per_aperture and aperture is not None:
        ap_diameter = float(aperture)
        coarse_dx = dx * sub
        n_coarse_across = ap_diameter / coarse_dx if coarse_dx > 0 else 0
        if n_coarse_across < min_coarse_samples_per_aperture:
            # Compute the largest sub that *would* be safe so the
            # error message gives the user an actionable number.
            safe_sub = max(1, int(np.floor(
                ap_diameter / (dx * min_coarse_samples_per_aperture))))
            msg = (
                f'apply_real_lens_traced: ray_subsample={ray_subsample} '
                f'gives only {n_coarse_across:.1f} coarse samples across '
                f'the {ap_diameter*1e3:.2f}-mm aperture (threshold '
                f'{min_coarse_samples_per_aperture}).  At this density '
                f'the spline interpolation of the wavefront will alias '
                f'and the OPD will be wrong by ~lambda/4 or more.  '
                f'Drop to ray_subsample <= {safe_sub} (or pass '
                f'min_coarse_samples_per_aperture=0 to override).'
            )
            if on_undersample == 'error':
                raise ValueError(msg)
            elif on_undersample == 'warn':
                import warnings
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
            elif on_undersample != 'silent':
                raise ValueError(
                    f"on_undersample must be 'error', 'warn', or "
                    f"'silent' (got {on_undersample!r})")

    # Number of samples across the launch grid (subsampled).  Keep it
    # at least proportional to the grid resolution so the OPL
    # function is well sampled.
    n_launch = max(8, int(2 * launch_radius / (dx * sub)))
    # Ensure odd so there's a sample on the optical axis (entrance
    # centre) -- makes on-axis piston subtraction exact.
    if n_launch % 2 == 0:
        n_launch += 1
    xs_in = np.linspace(-launch_radius, launch_radius, n_launch)
    # Use indexing='ij' so that after reshaping trace results to
    # (n_launch, n_launch), array[i, j] corresponds to entrance
    # (X = xs_in[i], Y = xs_in[j]) -- matching scipy's
    # RectBivariateSpline(x_knots, y_knots, values) convention where
    # values[i, j] is the value at (x_knots[i], y_knots[j]).  With the
    # default 'xy' indexing the reshape transposes x/y, which makes
    # the spline's Jacobian wrong and Newton converges to bogus
    # points for 2D wave pixels off the symmetry axes.
    Xs_in, Ys_in = np.meshgrid(xs_in, xs_in, indexing='ij')
    h_x = Xs_in.ravel()
    h_y = Ys_in.ravel()
    # Tier 1 input-aware ray launch: derive each ray's direction from
    # the local phase gradient of E_in at its entrance position.  For
    # plane-wave inputs this reduces to L = M = 0 (identical to the
    # classical collimated launch); for structured inputs (MLA
    # modulation, off-axis sources, pre-aberrated wavefronts) the
    # rays correctly start at the angle implied by E_in, giving the
    # lens its actual per-ray OPL instead of a plane-wave-reference
    # OPL map.  See :func:`_sample_local_tilts` for the extraction.
    if tilt_aware_rays:
        L_in, M_in = _sample_local_tilts(E_in, wavelength, dx, Xs_in, Ys_in)
        L_in = L_in.ravel()
        M_in = M_in.ravel()
    else:
        # 4.10: emit a one-time warning when the input field has a
        # measurable transverse tilt and tilt_aware_rays=False.  The
        # plane-wave reference OPD becomes inaccurate when the input
        # tilt is comparable to lambda / aperture.  Estimate the
        # transverse tilt as the RMS of grad(phase) / k0 over the
        # support of |E_in|; cap the check via a try-except so degenerate
        # input fields don't crash apply_real_lens_traced.
        try:
            E_arr = np.asarray(E_in)
            mag = np.abs(E_arr)
            mask = mag > 0.05 * mag.max()
            del mag
            if mask.any():
                phase = np.angle(E_arr)
                dpy, dpx = np.gradient(phase, dx, dx)
                del phase
                k0 = 2.0 * np.pi / wavelength
                tilt_rms = float(np.sqrt(
                    (np.mean((dpx[mask] / k0) ** 2)
                     + np.mean((dpy[mask] / k0) ** 2))))
                # v5.16.2 (memory root-cause): free the full-grid
                # diagnostics IMMEDIATELY.  Pre-fix, mag/mask/phase/dpx/
                # dpy stayed referenced by this frame for the REST of the
                # lens call -- ~4 full-grid float32 + a bool (~18 GB at
                # N=32768) held through the ray trace, Newton, and
                # assembly.  tracemalloc-attributed as the largest single
                # component of the 3.2.14.1 -> 5.16.x traced-lens memory
                # growth.  Values/output unchanged (pure lifetime fix).
                del dpx, dpy
                if tilt_rms > 1e-4:
                    import warnings
                    warnings.warn(
                        "apply_real_lens_traced: tilt_aware_rays=False "
                        f"with a non-trivial input-field tilt (RMS = "
                        f"{tilt_rms:.2e} rad).  The plane-wave "
                        "reference OPD is off by an amount proportional "
                        "to (tilt * aperture); set tilt_aware_rays=True "
                        "for tilt-sensitive analyses.",
                        RuntimeWarning, stacklevel=3,
                    )
            del E_arr, mask
        except (ValueError, RuntimeError, ZeroDivisionError, IndexError,
                AttributeError, TypeError):
            # tilt-RMS estimation is best-effort; suppressing the
            # warning when it can't be computed is preferable to
            # blowing up the traced-lens path.
            pass
        L_in = np.zeros_like(h_x)
        M_in = np.zeros_like(h_x)
    rays = _make_bundle(x=h_x, y=h_y, L=L_in, M=M_in,
                        wavelength=wavelength)
    # output_filter='last': only keep the image-plane bundle.  We do
    # not consume any intermediate per-surface state here, so saving
    # ray_history for all surfaces would allocate ~1 GB per surface
    # at N=32768 and ~250 MB per surface at N=4096 (for an
    # apply_real_lens_traced call at ray_subsample=8) for no benefit.
    result = trace(rays, surfaces, wavelength, output_filter='last')
    final = result.image_rays
    if not final.alive.any():
        raise RuntimeError(
            'apply_real_lens_traced: no rays survived the prescription; '
            'check aperture and clear-aperture settings.')

    # ---- EXIT-VERTEX CORRECTION ----------------------------------------
    # trace() leaves rays at the SAG of the last surface, i.e. at
    # z = sag(h) ≠ 0 for curved exit surfaces.  But the wave model's
    # exit field is defined at the flat exit VERTEX plane (z = 0).
    # Without this correction, the OPL comparison between on-axis
    # (z = 0) and off-axis (z = sag < 0 for concave) rays is made
    # at DIFFERENT z-planes, which introduces a systematic defocus
    # error equal to n_exit * sag(h) — enough to shift the implied
    # focal length by tens of percent for cemented doublets with
    # curved rear surfaces.
    #
    # Fix: propagate each ray from its current sag position to z = 0
    # in the exit medium, accumulating the remaining OPL and updating
    # the exit position to the vertex plane.
    #
    # IMPORTANT: use SIGNED t, not abs(t).  For concave rear surfaces
    # (sag < 0, z < 0) the ray must go forward (t > 0) → add OPL.
    # For convex rear surfaces (sag > 0, z > 0) the ray is AHEAD of
    # the vertex and must go backward (t < 0) → subtract OPL.
    # Using abs() forces the wrong sign for convex exits (e.g.
    # negative meniscus lenses), producing ~45x worse OPD.
    n_exit = get_glass_index(surfaces[-1].glass_after, wavelength)
    with np.errstate(divide='ignore', invalid='ignore'):
        t_to_vertex = np.where(
            final.alive & (np.abs(final.N) > 1e-30),
            -final.z / final.N, 0.0)
    final.opd = final.opd + n_exit * t_to_vertex
    final.x = final.x + final.L * t_to_vertex
    final.y = final.y + final.M * t_to_vertex
    final.z = np.zeros_like(final.z)

    # Reshape final.x, final.y, final.opd onto the regular ENTRANCE
    # grid.  Dead rays would break RectBivariateSpline (which requires
    # strictly regular data); vignetting is rare for normal lenses but
    # we guard against it by filling dead entries with NaN and
    # extrapolating with the spline's natural extrapolation (OK inside
    # the entrance disc of interest).
    x_out_grid = final.x.reshape(n_launch, n_launch)
    y_out_grid = final.y.reshape(n_launch, n_launch)
    opl_grid = final.opd.reshape(n_launch, n_launch)
    if not final.alive.all():
        alive_grid = final.alive.reshape(n_launch, n_launch)
        # Fill NaN into dead entries to make spline fitting fail
        # cleanly (rare path -- vignetted prescriptions)
        x_out_grid = np.where(alive_grid, x_out_grid, np.nan)
        y_out_grid = np.where(alive_grid, y_out_grid, np.nan)
        opl_grid = np.where(alive_grid, opl_grid, np.nan)

    # Reference OPL to on-axis (center of the entrance grid is an
    # exact sample because n_launch is odd)
    i_axis = n_launch // 2
    opl_grid = opl_grid - opl_grid[i_axis, i_axis]

    # ----- OPTION B: RectBivariateSpline + Newton-inversion of the
    # entrance->exit mapping ------------------------------------------
    #
    # Because the rays were launched on a regular (xs_in, xs_in) grid,
    # final.x, final.y, final.opl are regular-grid functions of the
    # entrance position.  We build three 2-D splines:
    #
    #     Sx(xe, ye) = x_out at entrance (xe, ye)
    #     Sy(xe, ye) = y_out at entrance (xe, ye)
    #     So(xe, ye) = OPL   at entrance (xe, ye)
    #
    # For each wave-grid exit pixel (Xw, Yw) we find the entrance
    # (xe, ye) that lands there via Newton iteration on the residual
    # r = (Sx(xe,ye) - Xw, Sy(xe,ye) - Yw) = 0.  Then OPL at that
    # wave pixel = So(xe, ye).
    #
    # Advantages over the previous scatter-to-grid (griddata) path:
    #   * C^2 smooth interpolation (no Delaunay-edge spikes).
    #   * RectBivariateSpline.ev() is implemented in Fortran and DOES
    #     release the GIL, so we CAN multi-thread the Newton loop.
    #   * Works correctly even for fast lenses with caustic-like
    #     behaviour near the exit-pupil edge (the mapping is still
    #     single-valued on the entrance grid; inversion is stable).
    # ---- Validate use_gpu combination ---------------------------------
    _newton_xp = np  # default Newton array backend
    if use_gpu:
        if newton_fit != 'polynomial':
            raise ValueError(
                f"use_gpu=True requires newton_fit='polynomial'; "
                f"got newton_fit={newton_fit!r}.  The spline path uses "
                f"SciPy RectBivariateSpline which has no GPU backend.")
        if not CUPY_AVAILABLE:
            raise ImportError(
                "use_gpu=True requires the 'cupy' package.  Install with "
                "'pip install cupy-cuda12x' (NVIDIA, matching your CUDA "
                "version) or 'pip install cupy-rocm-6-1' (AMD ROCm); or set "
                "use_gpu=False to stay on the CPU path.")
        _newton_xp = cp

    if newton_fit == 'polynomial':
        # 2-D Chebyshev tensor-product fit -- closed-form evaluation and
        # analytic derivatives, better accuracy than bicubic spline on
        # smooth refractive-lens data.  Same .ev(...) API so the
        # Newton loop below is untouched.
        #
        # When use_gpu=True, build the evaluator on GPU (all arrays
        # pushed to device via cp.asarray).  The Newton loop below
        # auto-detects the evaluator backend and runs on the matching
        # device.
        _xp = _newton_xp
        _xs_xp = _xp.asarray(xs_in)
        _xout_xp = _xp.asarray(x_out_grid)
        _yout_xp = _xp.asarray(y_out_grid)
        _opl_xp = _xp.asarray(opl_grid)
        Sx = _Cheb2DEvaluator(_xs_xp, _xs_xp, _xout_xp,
                               order=newton_poly_order, xp=_xp)
        Sy = _Cheb2DEvaluator(_xs_xp, _xs_xp, _yout_xp,
                               order=newton_poly_order, xp=_xp)
        So = _Cheb2DEvaluator(_xs_xp, _xs_xp, _opl_xp,
                               order=newton_poly_order, xp=_xp)
    elif newton_fit == 'spline':
        try:
            from scipy.interpolate import RectBivariateSpline
        except ImportError:
            raise ImportError(
                'apply_real_lens_traced requires SciPy for spline '
                'interpolation.')
        Sx = RectBivariateSpline(xs_in, xs_in, x_out_grid, kx=3, ky=3)
        Sy = RectBivariateSpline(xs_in, xs_in, y_out_grid, kx=3, ky=3)
        So = RectBivariateSpline(xs_in, xs_in, opl_grid, kx=3, ky=3)
    else:
        raise ValueError(
            f"newton_fit must be 'spline' or 'polynomial', "
            f"got {newton_fit!r}")

    # ---- Paraxial magnification from the already-computed forward
    # trace.  Used as the Newton initial guess: (xe, ye) ~ (Xw, Yw) / M.
    #
    # We read the central finite-difference slope of the forward map:
    #     M_x = [x_out(i_c+1, i_c) - x_out(i_c-1, i_c)] / (2 * d_xs_in)
    #     M_y = [y_out(i_c, i_c+1) - y_out(i_c, i_c-1)] / (2 * d_xs_in)
    # where (i_c, i_c) is the on-axis entrance grid point (exact sample
    # because n_launch is odd).  4.11.2: the indices match the meshgrid
    # at the launch step
    #     ``Xs_in, Ys_in = np.meshgrid(xs_in, xs_in, indexing='ij')``
    # which puts x along axis 0 and y along axis 1, so ∂x_out/∂x_in
    # varies the FIRST index, not the second.  Pre-4.11.2 the indices
    # were swapped, computing ∂x_out/∂y_in (~zero by rotational
    # symmetry) instead of ∂x_out/∂x_in.  Newton still converged
    # because the polynomial Jacobian is right, but every pixel started
    # at the clipped-to-boundary initial guess (0.91-fallback) instead
    # of the actual paraxial slope.
    #
    # This stencil is strictly better than the previous hard-coded 1.10
    # multiplier: the old heuristic assumed M ~ 0.91 (converging system
    # "shrinks 10%") which is approximately right for singlets at their
    # exit vertex (M ~ 1) but wildly off for compound systems with real
    # imaging magnification (TX Design 36 full-system inversion would
    # have M = 0.25; using 1.10 as the guess puts Newton 4x from the
    # answer and costs several extra iterations per pixel).  Zero
    # additional compute -- the grid values are already in memory from
    # the forward trace above.
    i_c = n_launch // 2
    d_xs = float(xs_in[1] - xs_in[0])
    try:
        M_x = (float(x_out_grid[i_c + 1, i_c])
               - float(x_out_grid[i_c - 1, i_c])) / (2.0 * d_xs)
        M_y = (float(y_out_grid[i_c, i_c + 1])
               - float(y_out_grid[i_c, i_c - 1])) / (2.0 * d_xs)
    except (IndexError, ValueError):
        M_x = M_y = 0.91  # fallback to pre-3.1.3 heuristic (1/1.10)
    # Guard against NaNs from dead rays at the center (unlikely -- the
    # axial ray always survives in a well-posed prescription) and
    # against extreme values that would blow up the initial guess.
    if not (np.isfinite(M_x) and np.isfinite(M_y)):
        M_x = M_y = 0.91
    M_x = float(np.clip(abs(M_x), 1e-3, 1e3))
    M_y = float(np.clip(abs(M_y), 1e-3, 1e3))

    # Store spline knot data for potential process-pool pickling.
    # Include the inverse magnification so the process-pool path (which
    # rebuilds splines inside each worker) can seed Newton identically.
    _spline_data = {
        'xs_in': xs_in,
        'x_out_grid': x_out_grid,
        'y_out_grid': y_out_grid,
        'opl_grid': opl_grid,
        'launch_radius': launch_radius,
        'dx': dx,
        'bound': launch_radius * 0.999,
        'inv_M_x': 1.0 / M_x,
        'inv_M_y': 1.0 / M_y,
    }

    # Bound for the clipped Newton update (stay inside fitted domain)
    bound = launch_radius * 0.999

    # Newton iter cap: caller override > module default.  See the note
    # at _NEWTON_MAX_ITERS for the 8-vs-12 trade-off.
    MAX_NEWTON_ITERS = (int(newton_max_iters) if newton_max_iters is not None
                        else _NEWTON_MAX_ITERS)

    def _invert_newton(Xw, Yw, sub_progress=None):
        """Run Newton iteration to find (xe, ye) such that (Sx, Sy)
        evaluated at (xe, ye) equals (Xw, Yw).  Returns OPL at the
        converged entrance positions plus a validity mask.

        Fully vectorised over the input arrays -- ``Xw`` and ``Yw``
        may be any shape; result has the same shape.

        ``sub_progress`` is an optional ``ProgressScaler`` (or any
        callable ``f(frac, msg)``) driven once per Newton iteration.
        """
        # Detect Newton-loop array backend from the evaluator.  The
        # evaluator's xp is either numpy (CPU) or cupy (GPU when
        # use_gpu=True was set earlier).  Using xp uniformly inside
        # the Newton loop keeps this code device-agnostic -- the only
        # other GPU plumbing needed is pushing xe/ye/active/idx_active
        # to xp and pulling opl_flat back to numpy at the end.
        xp = getattr(Sx, 'xp', np)
        # Push wave-grid coordinates to the Newton backend.  On the
        # CPU path this is a zero-cost view; on GPU it's a H->D copy
        # of order (N_wave^2) floats, incurred once per Newton call.
        x_w_flat = xp.asarray(Xw.ravel())
        y_w_flat = xp.asarray(Yw.ravel())
        n_total = int(x_w_flat.size)
        # Initial guess: entrance ~ exit / M, where M is the paraxial
        # magnification measured from the central finite-difference slope
        # of the forward map (see `inv_M_x` / `inv_M_y` computed above from
        # the already-traced ray grid -- no extra compute).  This is a
        # strictly better guess than the pre-3.1.3 hard-coded 1.10
        # multiplier: for singlets with M ~ 1 the two are nearly identical,
        # but for compound systems or unusual magnifications the measured
        # value avoids putting Newton several iterations away from
        # convergence.
        xe = x_w_flat * _spline_data['inv_M_x']
        ye = y_w_flat * _spline_data['inv_M_y']
        tol = 0.01 * dx
        active = xp.ones(xe.size, dtype=bool)  # pixels still iterating
        if sub_progress is not None:
            sub_progress(0.0, f'newton 0/{MAX_NEWTON_ITERS}: {n_total} pixels')
        # When the fit objects support combined value+gradient
        # (polynomial path via _Cheb2DEvaluator), use it to halve the
        # number of Newton-hot-path evaluator calls per iteration from
        # 6 down to 2, and share Chebyshev basis work across f/fx/fy.
        _has_combined = (hasattr(Sx, 'ev_value_and_grad')
                          and hasattr(Sy, 'ev_value_and_grad'))
        for _it in range(MAX_NEWTON_ITERS):
            if not bool(active.any()):
                if sub_progress is not None:
                    sub_progress(1.0,
                                 f'newton converged after {_it} iters')
                # v5.3.2 (ROADMAP logging adoption sweep -- per-iteration
                # telemetry): emit a "converged" marker so an attached
                # handler sees the early-exit path.
                logger.info(
                    "apply_real_lens_traced: newton iter %d/%d converged "
                    "(all %d pixels)",
                    int(_it), int(MAX_NEWTON_ITERS), int(n_total))
                break
            # Only evaluate splines at active (unconverged) pixels
            xa = xe[active]
            ya = ye[active]
            xw = x_w_flat[active]
            yw = y_w_flat[active]
            if _has_combined:
                fx_val, jxx, jxy = Sx.ev_value_and_grad(xa, ya)
                fy_val, jyx, jyy = Sy.ev_value_and_grad(xa, ya)
                rx = fx_val - xw
                ry = fy_val - yw
            else:
                rx = Sx.ev(xa, ya) - xw
                ry = Sy.ev(xa, ya) - yw
                jxx = Sx.ev(xa, ya, dx=1)
                jxy = Sx.ev(xa, ya, dy=1)
                jyx = Sy.ev(xa, ya, dx=1)
                jyy = Sy.ev(xa, ya, dy=1)
            det = jxx * jyy - jxy * jyx
            safe = xp.abs(det) > 1e-12
            inv_det = xp.where(safe, 1.0 / det, 0.0)
            dxe = (jyy * rx - jxy * ry) * inv_det
            dye = (-jyx * rx + jxx * ry) * inv_det
            xa_new = xp.clip(xa - dxe, -bound, bound)
            ya_new = xp.clip(ya - dye, -bound, bound)
            xe[active] = xa_new
            ye[active] = ya_new
            # Mark converged pixels as inactive
            res = xp.sqrt(rx * rx + ry * ry)
            converged = res < tol
            idx_active = xp.where(active)[0]
            active[idx_active[converged]] = False
            if sub_progress is not None:
                remaining = int(active.sum())
                pct_done = 1.0 - remaining / max(n_total, 1)
                # Emit max(iteration-based, convergence-based) fraction,
                # bounded to <1 so the final "assembling" tick owns 1.0.
                frac = min(max((_it + 1) / MAX_NEWTON_ITERS, pct_done),
                           0.99)
                sub_progress(
                    frac,
                    f'newton {_it + 1}/{MAX_NEWTON_ITERS}: '
                    f'{remaining}/{n_total} pixels unconverged')
            # v5.3.2 (ROADMAP logging adoption sweep -- per-iteration
            # telemetry): per-Newton-iteration log, independent of the
            # sub_progress callback (sub_progress is None on the
            # serial / single-call code paths).  Reports current OPD
            # residual norm + remaining-active-pixel count so an
            # attached handler can track convergence.
            _remaining_log = int(active.sum())
            # v5.4 (audit P3): deduplicate -- reuse res from convergence check above
            try:
                _res_norm = float(res.max()) if res.size > 0 else 0.0
            except (ValueError, TypeError):
                _res_norm = float('nan')
            logger.info(
                "apply_real_lens_traced: newton iter %d/%d "
                "residual_max=%.3e m remaining=%d/%d",
                int(_it + 1), int(MAX_NEWTON_ITERS),
                _res_norm, _remaining_log, int(n_total))
        # Surface unconverged pixels.  Healthy prescriptions can have a
        # handful of out-of-domain edge pixels left active at the
        # iteration cap -- those are benign and don't warrant a warning.
        # Threshold: >1% of total pixels unconverged means a real
        # convergence problem.  Honour the same ``on_undersample`` knob
        # the rest of the function uses ('silent' suppresses, 'warn'
        # / 'error' default emits the warning).  Pre-3.5.6 unconverged
        # pixels were silently kept at their last Newton value.
        n_unconverged = int(active.sum()) if hasattr(
            active, 'sum') else 0
        n_total = int(active.size) if hasattr(active, 'size') else 1
        if (n_unconverged > 0 and n_unconverged > 0.01 * n_total
                and on_undersample != 'silent'):
            import warnings as _warnings
            _warnings.warn(
                f"apply_real_lens_traced Newton inversion: "
                f"{n_unconverged}/{n_total} pixels "
                f"({100.0*n_unconverged/n_total:.1f}%) did not converge "
                f"to tol={tol:.3e} m within {MAX_NEWTON_ITERS} "
                f"iterations.  Affected pixels keep their last Newton "
                f"value, which may carry residual error.  Increase "
                f"newton_max_iters if this matters for your tolerance "
                f"budget.",
                RuntimeWarning, stacklevel=3)
        opl_flat = So.ev(xe, ye)
        out_of_domain = (xe * xe + ye * ye > (launch_radius * 0.99) ** 2)
        opl_flat = xp.where(out_of_domain, xp.nan, opl_flat)
        # If we ran on GPU, pull the result back to the host so the
        # rest of apply_real_lens_traced -- which is CPU-only
        # (amplitude from apply_real_lens, final field assembly) --
        # sees a NumPy array.
        if xp is not np:
            opl_flat = cp.asnumpy(opl_flat)
        return opl_flat.reshape(Xw.shape)

    # ----- Coarse-grid Newton + interpolation --------------------------
    # The OPL map is extremely smooth (well-approximated by a
    # low-order polynomial), so evaluating the expensive Newton
    # inversion at every wave-grid pixel is wasteful.  Instead we
    # evaluate on a COARSER output grid and bilinearly interpolate to
    # the full wave grid.  ``ray_subsample`` controls the output
    # sub-sampling factor:
    #
    #   ray_subsample=1  -> Newton at every pixel (exact, slow)
    #   ray_subsample=4  -> Newton at every 4th pixel, interp rest
    #   ray_subsample=8  -> Newton at every 8th pixel (fastest)
    #
    # Parallelism: Newton is embarrassingly parallel (per-pixel
    # independent, immutable splines).  We dispatch to a process pool
    # when the grid is large enough that pool startup + knot-pickling
    # is worth it.  Threads don't help here: scipy's
    # ``RectBivariateSpline.ev`` does not release the GIL in current
    # versions, so threading delivers no speedup.

    from concurrent.futures import as_completed

    from ..memory import available_cpus

    # Affinity-aware: respect cgroup limits, taskset masks, Python 3.13+
    # process_cpu_count so we don't oversubscribe a restricted machine.
    # If the user pinned half the cores via taskset (or the container
    # has a CPU quota) we'll see the restricted count here, whereas
    # os.cpu_count() would still return the raw logical total.
    n_cpu = n_workers if n_workers is not None else available_cpus()
    n_cpu = max(1, int(n_cpu))

    # Heuristic: only spin up the pool when the chunk count can actually
    # fill it AND the work per chunk amortises the startup cost.  On
    # Windows spawn mode, pool startup is ~200-400 ms per worker.
    _POOL_MIN_PIXELS = 200_000

    def _invert_newton_parallel(Xw, Yw, sub_progress=None):
        """Dispatch ``_invert_newton`` work across a process pool when
        useful; fall back to the in-process serial path otherwise.

        Preserves the serial path's numerical behaviour exactly (same
        Newton iteration count, same convergence tolerance, same
        out-of-domain NaN policy -- see :func:`_newton_invert_chunk`).
        """
        # GPU path must stay in-process: the worker function
        # ``_newton_invert_chunk`` rebuilds SciPy splines per worker
        # (CPU-only), and shipping CuPy device arrays through a
        # ProcessPoolExecutor would host-copy them anyway.  Go direct.
        if use_gpu:
            return _invert_newton(Xw, Yw, sub_progress=sub_progress)
        # Spline-path worker pool is also incompatible with
        # newton_fit='polynomial' because the worker builds
        # RectBivariateSpline rather than _Cheb2DEvaluator.  Force
        # serial for polynomial until a worker-side polynomial path
        # is added (cheap on Newton-time at subsample=8 anyway).
        if newton_fit == 'polynomial':
            return _invert_newton(Xw, Yw, sub_progress=sub_progress)
        x_w_flat = Xw.ravel()
        y_w_flat = Yw.ravel()
        n_total = x_w_flat.size
        if n_cpu <= 1 or n_total < _POOL_MIN_PIXELS:
            return _invert_newton(Xw, Yw, sub_progress=sub_progress)

        if sub_progress is not None:
            sub_progress(0.0,
                         f'newton pool: {n_total} pts across '
                         f'{n_cpu} workers')
        # Split indices into roughly-equal chunks.  ``np.array_split``
        # handles the n_total % n_cpu != 0 case cleanly.
        chunk_idx = np.array_split(np.arange(n_total), n_cpu)
        args_list = [
            (_spline_data, x_w_flat[c].copy(), y_w_flat[c].copy())
            for c in chunk_idx]
        results = [None] * len(args_list)

        # Use the module-level persistent ProcessPool to amortise
        # Windows-spawn startup cost across repeated apply_real_lens_traced
        # calls (the dominant overhead for optimisation / tolerancing
        # workflows).  See _get_persistent_worker_pool docstring for
        # details.
        try:
            ex = _get_persistent_worker_pool(n_cpu)
            future_to_idx = {
                ex.submit(_newton_invert_chunk, a): i
                for i, a in enumerate(args_list)}
            done = 0
            for fut in as_completed(future_to_idx):
                i = future_to_idx[fut]
                results[i] = fut.result()
                done += 1
                if sub_progress is not None:
                    frac = min(done / max(len(args_list), 1), 0.99)
                    sub_progress(
                        frac,
                        f'newton chunk {done}/{len(args_list)} done')
        except (RuntimeError, OSError, BrokenPipeError, EOFError,
                ImportError, ValueError, MemoryError):
            # Any pool failure (Windows antivirus blocking spawn,
            # broken worker pipe, ImportError under pickled
            # closures, MemoryError on a tight box) falls through
            # to the serial path so the caller isn't left without
            # a result.
            return _invert_newton(Xw, Yw, sub_progress=sub_progress)

        opl_flat = np.concatenate(results)
        return opl_flat.reshape(Xw.shape)

    call_progress(progress, 'real_lens_traced', 0.55,
                  'inverting entrance->exit map')
    # Give the Newton loop its own slice of the parent budget
    # (0.55 -> 0.88) so the bar advances through the iterations
    # instead of sitting still between the 0.55 and 0.90 ticks.
    newton_cb = ProgressScaler(progress, 'real_lens_traced',
                               lo=0.55, hi=0.88)

    # ---------- Amplitude-mask the Newton work --------------------
    # Pixels where ``amp`` is well below peak produce a final field
    # of ``|E_analytic| * exp(...)`` that is already ~zero no matter
    # what OPL we compute for them, so running Newton there is
    # wasted effort.  We build a boolean mask on the coarse output
    # grid, dilate by ``newton_mask_dilate_coarse_px`` so bilinear
    # interpolation at the full grid always has real data in its
    # support near mask boundaries, and run Newton only on the
    # masked coarse pixels.  Skipped pixels get ``NaN`` which the
    # existing NaN-propagation logic below treats exactly like the
    # ray-domain-failure NaNs from Newton itself.
    #
    # Controls:
    #   newton_amp_mask_rel=0  disables masking (runs Newton on the
    #                          entire coarse grid, bit-identical to
    #                          pre-mask behaviour).
    #   newton_amp_mask_rel>0  threshold = that fraction of amp.max().
    #   newton_mask_dilate_coarse_px  0 for no dilation, else that
    #                          many iterations of binary_dilation.
    #
    # The mask is SKIPPED if it would capture essentially everything
    # (>95 %) -- in that case the filter overhead isn't worth it --
    # or essentially nothing (<1 %) -- which signals a pathological
    # amp field and we fall back to full-grid Newton rather than
    # returning garbage.
    def _build_newton_mask(amp_grid):
        if newton_amp_mask_rel <= 0.0:
            return None
        amp_max = float(amp_grid.max())
        if amp_max <= 0.0:
            return None
        thresh = amp_max * float(newton_amp_mask_rel)
        m = amp_grid > thresh
        if newton_mask_dilate_coarse_px > 0:
            from scipy.ndimage import binary_dilation
            m = binary_dilation(
                m, iterations=int(newton_mask_dilate_coarse_px))
        frac = float(m.mean())
        if frac > 0.95 or frac < 0.01:
            return None
        return m

    # Dispatch the OPL inversion to Newton (default) or the experimental
    # backward-trace alternative.  Both produce a wave-grid OPL map
    # with the same axis convention (on-axis referenced to zero, NaN
    # for out-of-domain / dead-ray pixels).
    if inversion_method == 'backward_trace':
        # Experimental path.  Bypasses the forward ray trace + Newton
        # spline inversion entirely; see _opl_by_backward_trace for
        # the algorithm and caveats.  Kept as an opt-in because the
        # accuracy on focused-beam exit planes has not been as
        # thoroughly validated as the Newton path.
        opl_map = _opl_by_backward_trace(
            E_analytic, lens_prescription, wavelength, dx,
            N_grid=N, ray_subsample=sub)
    elif sub > 1:
        # Evaluate Newton on sub-sampled output grid.  On the chunked-
        # assembly path the full X/Y meshgrids were never built; the coarse
        # grid from the 1-D subsampled vector is element-identical to
        # ``X[::sub, ::sub]`` (meshgrid(x,x) is x[j]/x[i] replicated).
        if X is None:
            _x_c = x[::sub]
            Xs, Ys = np.meshgrid(_x_c, _x_c)
        else:
            Xs = X[::sub, ::sub]
            Ys = Y[::sub, ::sub]
        amp_coarse = amp[::sub, ::sub]
        mask_coarse = _build_newton_mask(amp_coarse)
        if mask_coarse is None:
            opl_coarse = _invert_newton_parallel(
                Xs, Ys, sub_progress=newton_cb)
        else:
            Xs_masked = Xs[mask_coarse]
            Ys_masked = Ys[mask_coarse]
            opl_1d = _invert_newton_parallel(
                Xs_masked, Ys_masked, sub_progress=newton_cb)
            opl_coarse = np.full(Xs.shape, np.nan, dtype=opl_1d.dtype)
            opl_coarse[mask_coarse] = opl_1d
        # Bilinearly interpolate to full grid
        from scipy.ndimage import map_coordinates
        Ns = opl_coarse.shape[0]
        if _chunk_assembly:
            # Row-band path: defer the upsample into the Step-3 band loop
            # (map_coordinates order-1 is pointwise in the output, so the
            # banded interpolation is element-identical).  Only the SMALL
            # coarse arrays are kept; the full-grid ii/jj index pair, the
            # (2, N, N) coords stack, opl_map and nan_full never allocate.
            _opl_coarse_clean = np.where(
                np.isnan(opl_coarse), 0.0, opl_coarse)
            _nan_coarse = np.isnan(opl_coarse).astype(np.float64)
            opl_map = None
        else:
            # v5.16.2 (memory root-cause): build the (2, N, N) coordinate
            # stack ONCE and free ii/jj before interpolating.  Pre-fix the
            # stack was constructed twice (once per map_coordinates call)
            # with ii/jj held throughout -- ~4 extra full-grid float64
            # (~34 GB at N=32768) at the upsample peak.  Same coords,
            # same map_coordinates inputs -> byte-identical outputs.
            ii, jj = np.indices((N, N), dtype=np.float64)
            _coords = np.array([ii * Ns / N, jj * Ns / N])
            del ii, jj
            opl_map = map_coordinates(
                np.where(np.isnan(opl_coarse), 0.0, opl_coarse),
                _coords, order=1, mode='nearest')
            # Propagate NaN mask
            nan_coarse = np.isnan(opl_coarse).astype(np.float64)
            nan_full = map_coordinates(
                nan_coarse, _coords, order=1, mode='nearest')
            del _coords
            opl_map = np.where(nan_full > 0.5, np.nan, opl_map)
            del nan_full
    else:
        mask_full = _build_newton_mask(amp)
        if mask_full is None:
            opl_map = _invert_newton_parallel(
                X, Y, sub_progress=newton_cb)
        else:
            X_masked = X[mask_full]
            Y_masked = Y[mask_full]
            opl_1d = _invert_newton_parallel(
                X_masked, Y_masked, sub_progress=newton_cb)
            opl_map = np.full(X.shape, np.nan, dtype=opl_1d.dtype)
            opl_map[mask_full] = opl_1d
    call_progress(progress, 'real_lens_traced', 0.90,
                  'assembling exit field')

    # ----- Step 3: combine amplitude with geom phase -------------------
    # When preserve_input_phase=True (default, physically correct):
    #   We KEEP the full complex E_analytic (which already contains the
    #   input field's phase correctly propagated through the glass
    #   split-step) and APPLY A CORRECTION that replaces the analytic
    #   model's lens-only phase with the ray-traced OPL.
    #
    #   delta_phase = k0 * opl_traced - phase_analytic_lens
    #   E_out = E_analytic * exp(i * delta_phase)
    #
    # This preserves any input-field phase structure (source tilts, MLA
    # patterns, off-axis aberrations) that apply_real_lens correctly
    # carried through.  Before this fix, the input phase was silently
    # discarded -- tilted inputs focused on-axis, MLA-modulated inputs
    # came out as a featureless envelope, etc.
    #
    # When preserve_input_phase=False (legacy behaviour):
    #   E_out = |E_analytic| * exp(i * k0 * opl_traced).  Useful for
    #   measuring the lens-only OPD on a plane-wave input, where the
    #   input-phase question is moot.
    k0 = 2.0 * np.pi / wavelength
    # Preserve the caller's complex dtype: apply_real_lens (called
    # above to build E_analytic / amp) already returns a field in
    # E_in.dtype, but the ``* np.exp(1j * ...)`` multiply here would
    # silently upcast to complex128 unless we cast the exp() result.
    target_cdtype = E_in.dtype if np.iscomplexobj(E_in) else np.complex128
    if _chunk_assembly and opl_map is None:
        # Row-band assembly: upsample + delta-phase + combine + masks per
        # (chunk_rows x N) band, writing into E_analytic in place (it is not
        # read again after its own band is consumed).  Element-identical to
        # the whole-grid branch below: map_coordinates(order=1) interpolates
        # each output point independently from the WHOLE coarse grid, and
        # every other op is pointwise; the band aperture term
        # ``x[j]^2 + x[i]^2`` reproduces ``(X**2 + Y**2)[r0:r1]`` exactly.
        from scipy.ndimage import map_coordinates
        cr = int(sag_chunk_rows)
        r_ap_sq = (aperture / 2) ** 2 if aperture is not None else None
        E_out = E_analytic
        for r0 in range(0, N, cr):
            r1 = min(N, r0 + cr)
            ii_b, jj_b = np.indices((r1 - r0, N), dtype=np.float64)
            if r0:
                ii_b += r0
            coords_b = np.array([ii_b * Ns / N, jj_b * Ns / N])
            opl_b = map_coordinates(_opl_coarse_clean, coords_b,
                                    order=1, mode='nearest')
            nan_b = map_coordinates(_nan_coarse, coords_b,
                                    order=1, mode='nearest')
            del ii_b, jj_b, coords_b
            opl_b = np.where(nan_b > 0.5, np.nan, opl_b)
            valid_b = np.isfinite(opl_b)
            if preserve_input_phase:
                dp_b = np.where(
                    valid_b, k0 * opl_b - phase_analytic_lens[r0:r1], 0.0)
            else:
                dp_b = np.where(valid_b, k0 * opl_b, 0.0)
            pe_b = np.exp(1j * dp_b)
            if pe_b.dtype != target_cdtype:
                pe_b = pe_b.astype(target_cdtype)
            if preserve_input_phase:
                band = E_analytic[r0:r1] * pe_b
            else:
                band = amp[r0:r1] * pe_b
            band = np.where(valid_b, band, target_cdtype.type(0))
            if r_ap_sq is not None:
                h_b = x[None, :] ** 2 + x[r0:r1, None] ** 2
                band = np.where(h_b <= r_ap_sq, band,
                                target_cdtype.type(0))
            E_out[r0:r1] = band
        if E_out.dtype != target_cdtype:
            E_out = E_out.astype(target_cdtype)
        call_progress(progress, 'real_lens_traced', 1.0, 'done')
        return E_out
    valid = np.isfinite(opl_map)
    # v5.16.2: free each full-grid intermediate as soon as its consumer is
    # built (delta_phase/phase after phase_exp; phase_exp after E_out;
    # opl_map after the phase build).  Pure lifetime fixes -- values and
    # outputs unchanged.
    if preserve_input_phase:
        delta_phase = np.where(valid, k0 * opl_map - phase_analytic_lens, 0.0)
        del opl_map
        phase_exp = np.exp(1j * delta_phase)
        del delta_phase
        if phase_exp.dtype != target_cdtype:
            phase_exp = phase_exp.astype(target_cdtype)
        E_out = E_analytic * phase_exp
        del phase_exp
    else:
        phase = np.where(valid, k0 * opl_map, 0.0)
        del opl_map
        phase_exp = np.exp(1j * phase)
        del phase
        if phase_exp.dtype != target_cdtype:
            phase_exp = phase_exp.astype(target_cdtype)
        E_out = amp * phase_exp
        del phase_exp
    # Zero outside the exit-pupil (ray-coverage) region
    E_out = np.where(valid, E_out, target_cdtype.type(0))
    # And outside the entrance aperture (defensive: in practice the
    # ray-coverage region is a subset of the entrance aperture, so
    # this is a no-op except in pathological configurations)
    if aperture is not None:
        E_out = np.where(X ** 2 + Y ** 2 <= (aperture / 2) ** 2,
                         E_out, target_cdtype.type(0))
    if E_out.dtype != target_cdtype:
        E_out = E_out.astype(target_cdtype)
    call_progress(progress, 'real_lens_traced', 1.0, 'done')
    return E_out


__all__ = [
    'apply_real_lens_traced',
    'close_worker_pool',
    'set_lens_parallel_amp',
    'get_lens_parallel_amp',
]
