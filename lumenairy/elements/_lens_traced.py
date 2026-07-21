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


# N12 (P11): relative floor on |det J| for the opt-in ``amplitude_model=
# 'ray_density'`` mode.  The ray-density exit amplitude is
# ``|E_in| / sqrt(|det J|)``, which diverges as ``det J -> 0`` at a fold
# caustic.  |det J| is floored at this fraction of the median |det J| over the
# ray-covered region so the amplitude stays finite (never inf/nan); a fold is
# also flagged when |det J| drops below the floor OR det J changes sign between
# adjacent ray cells, which triggers a one-time caustic warning steering the
# caller to GBD/FGA.  1e-3 is well below the ~O(1) det-J variation a smooth
# (non-folding) coma redistribution produces, so it never clips a legitimate
# aberrated spot; it only engages at a genuine fold.
_RAY_DENSITY_CAUSTIC_FLOOR_REL = 1e-3

# N12 (P11): a |det J| dynamic-range (max/min over the ray-covered region) above
# this flags a near-caustic even when the coarse grid does not resolve the exact
# fold curve (a sign change) or drive a sample below the absolute floor -- e.g.
# tracing to a plane at/near a focus, where all rays crowd into a tiny region so
# |det J| spans orders of magnitude and the single-branch ray-density amplitude
# under-resolves the singular spot (energy is NOT conserved there).  A smooth,
# non-folding aberrated map varies |det J| by <~a few x, well below this; a
# genuine caustic spans >>30x.  Conservative (a false positive only steers the
# caller to GBD/FGA, never returns a wrong number).
_RAY_DENSITY_CAUSTIC_MAXMIN = 30.0


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


def _prescription_has_field_frame(prescription) -> bool:
    """True when any surface carries a P3-style FIELD-FRAME decenter / tilt /
    freeform ``sag_callable`` (the ``apply_real_lens`` displaced-pointwise
    convention).  For such elements the ray trace (``trace`` ->
    ``_intersect_surface`` / ``_refract`` via the shared field-frame
    ``_surface_sag_xy``) carries the transverse ray WALK-OFF -- the true induced
    coma -- into the geometry and OPL, so the traced centroid / sign-mirror /
    tilt are oracle-matched.

    Detection helper only (used by the tests and available for dispatcher
    routing).  It does NOT gate any amplitude change: the P9 field-frame
    amplitude override was REMOVED (2026-07-20) after the adversarial verifier
    proved it was a model-mixing artefact.  The traced hybrid's grid-indexed
    amplitude leg cannot carry the decentered walk-off (an asymmetric ray-
    density redistribution), so its decentered-spot EE is amplitude-limited -- a
    genuine model limit of the P3 single-plane class.  Route decentered-coma EE
    to ``apply_real_lens_gbd`` (N10b), whose beamlets carry the walk-off
    amplitude and BROADEN matching ZOS (1.035 @1.31um) + the geom-spot oracle.
    See docs/audit_real_lens_displaced_2026_07_19.md (P9 / N10a)."""
    for s in (prescription.get('surfaces') or []):
        if not isinstance(s, dict):
            continue
        if s.get('sag_callable') is not None:
            return True
        dec = s.get('decenter') or (0.0, 0.0)
        if float(dec[0]) != 0.0 or float(dec[1]) != 0.0:
            return True
        tl = s.get('tilt') or (0.0, 0.0)
        if float(tl[0]) != 0.0 or float(tl[1]) != 0.0:
            return True
    return False


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


def _solve_lstsq_thread_safe(A, b):
    """Least-squares solve ``A @ x ~= b`` (overdetermined, single or multi RHS)
    via the NORMAL EQUATIONS -- ``G x = A^T b`` with ``G = A^T A`` -- Cholesky
    then LU, falling back to ``np.linalg.lstsq`` only if ``G`` is not positive-
    definite (a genuinely rank-deficient design matrix).

    B7 (jax x OpenBLAS mitigation): the traced Chebyshev/coordinate fits used
    ``np.linalg.lstsq`` (LAPACK ``gelsd``, a divide-and-conquer SVD).  In one
    process alongside JAX, ``gelsd``'s multi-threaded OpenBLAS OpenMP pool nests
    inside JAX's OpenMP runtime and DEADLOCKS on the first large fit -- the CI
    worked around it with ``OMP_NUM_THREADS=1`` pins that cannot be relied on
    outside CI.  The normal equations reduce the factorisation to the tiny
    ``M x M`` Gram matrix (``M`` = number of fit terms, ~28-70), which stays
    below OpenBLAS's threading threshold and never takes the ``gelsd`` SVD path,
    so the deadlock cannot recur.  ``A`` here is a well-conditioned normalised
    tensor-Chebyshev / monomial Vandermonde (~1.5x oversampled), so squaring the
    condition number in ``G`` is safe and the solution matches ``lstsq`` to
    ~1e-12 relative (the M-P5 precedent, ``lenses_maslov._solve_fit``, measured
    2.6e-15).  The ``lstsq`` fallback runs only for the rare rank-deficient case
    that Cholesky AND LU both reject -- a path the well-conditioned traced fits
    never reach in practice.

    Returns ``x`` with the same trailing shape as ``b`` (1-D for a single RHS).
    """
    A = np.ascontiguousarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    G = A.T @ A
    rhs = A.T @ b
    try:
        from scipy.linalg import cho_factor, cho_solve
        return cho_solve(cho_factor(G, check_finite=False), rhs,
                         check_finite=False)
    except (ImportError, ValueError, np.linalg.LinAlgError):
        # scipy absent, or G not positive-definite (rank-deficient fit).
        try:
            return np.linalg.solve(G, rhs)
        except np.linalg.LinAlgError:
            x, *_ = np.linalg.lstsq(A, b, rcond=None)
            return x


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
        # B7: normal-equations solve (thread-safe; never takes gelsd's SVD path
        # that deadlocks against JAX's OpenMP runtime in a shared process).
        c_np = _solve_lstsq_thread_safe(A, rhs)
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


# F1 (audit): residual transverse-angular-spread (radians) above which the
# plane-wave / carrier-referenced traced correction is flagged as invalid.
# ~0.02 rad (~1 deg) cleanly separates a collimated / carrier-matched beam
# (residual ~ 0) from an unreferenced divergent / emitter-array field
# (residual 0.1-0.2 rad); heuristic, tunable by the caller via the
# on_noncollimated policy.
_NONCOLLIMATED_RESID_THRESH = 0.02

# N5 (2026-07-19): when ``tilt_aware_rays=True`` and no explicit ``carrier`` is
# given, an auto-fit carrier eikonal is threaded through the carrier plumbing so
# the exit wavefront carries the input congruence (matching the carrier path's
# H6 entrance-eikonal fix).  It engages only when the fitted eikonal's peak phase
# over the bright support exceeds this floor, so a (near-)collimated tilt_aware
# input -- which fits ``W == 0`` exactly for a real / globally-phased field --
# keeps the byte-identical plane-wave-reference path.  1e-2 rad (~lambda/628 of
# OPD across the beam) sits far below any divergence that shifts the focus (a
# gently diverging R=10 m beam already fits tens of radians) yet safely above
# float round-off.
_TILT_EIKONAL_MIN_RAD = 1e-2

# R6 / audit F1 (2026-07-21): the ``carrier='auto'`` least-squares gradient fit
# silently degraded to ~inf (no carrier) on a strongly-diverging / coarsely-
# sampled spherical input -- exactly the input class it exists for.  Root cause:
# the nearest-neighbour phase-increment ``angle(E[i+1] conj(E[i]))`` tilt reading
# ALIASES (wraps past +-pi) at radii where the local carrier tilt exceeds the
# grid Nyquist tilt ``lambda/(2 dx)``.  On the 121's S5-S7 group (R_in=+153 mm,
# w=6 mm, dx=24.6 um) that boundary sits at r ~ 4 mm, well inside the beam, so
# MOST of the bright support fed the fit wrapped (near-zero-mean) tilt samples
# that pulled the fitted 1/R toward 0.  Fix: restrict the fit to the CONNECTED
# un-aliased core -- the region, contiguous with the phase-flat point, where the
# tilt reading stays below this fraction of the Nyquist tilt.  The wrapped rings
# beyond the first Nyquist crossing form SEPARATE connected components (a high-
# tilt annulus disconnects them from the core), so they are excluded and the
# central parabola alone -- which fully determines the spherical R -- drives the
# fit.  Recovers R to <~1% on S5-S7 and is byte-identical on well-sampled inputs
# (whole bright support is one un-aliased component -> same samples as before).
# The recovery is insensitive to this fraction over 0.35-0.7; 0.5 is a safe
# midpoint (masks before ANY grid axis component wraps, since gmag is the vector
# tilt magnitude).
_AUTO_CARRIER_NYQUIST_FRAC = 0.5
# Minimum un-aliased-core sample count below which the core restriction is
# abandoned and the full bright support is used (the historical behaviour): too
# few core samples cannot constrain the low-order fit, so a pathological /
# near-fully-aliased input falls back rather than fitting noise.
_AUTO_CARRIER_MIN_CORE = 64
# The un-aliased-core restriction engages ONLY when at least this fraction of
# the bright support reads as aliased (local tilt >= the Nyquist fraction).  A
# well-sampled input -- flat, mildly diverging, or a MULTI-EMITTER array whose
# beamlets are each Nyquist-sampled -- has ~no aliased samples, so the fit keeps
# the full bright support (byte-identical to the historical single-component-
# agnostic fit; critically, it does NOT collapse a disconnected multi-emitter
# field onto one beamlet's connected component).  The F1 strongly-diverging
# single carrier aliases the great majority of its support, far above this.
_AUTO_CARRIER_ALIAS_FRAC = 0.05


def _carrier_residual_rms(E_in, W_full, wavelength, dx):
    """RMS transverse angular spread (radians) of ``E_in`` AFTER removing
    the carrier wavefront ``W_full`` (length units; ``None`` -> no carrier).

    This is the discriminator for the F1 collimation guard: a beam that is
    a single smooth carrier plus a small angular residual (an emitter array,
    a diverging source) has a SMALL residual once the carrier is subtracted,
    even though its raw angular spread is large -- so it is well within the
    carrier-referenced traced model's validity.  Uses the wrapping-safe
    nearest-neighbour phase-increment estimator.
    """
    k0 = 2.0 * np.pi / wavelength
    E = np.asarray(E_in)
    if W_full is not None:
        E = E * np.exp(-1j * k0 * np.asarray(W_full))
    mag = np.abs(E)
    mx = mag.max()
    if not np.isfinite(mx) or mx <= 0:
        return 0.0
    mask = mag > 0.05 * mx
    del mag
    if not mask.any():
        return 0.0
    gx = E[:, 1:] * np.conj(E[:, :-1])
    lx = (np.angle(gx) / (k0 * dx))[mask[:, 1:] & mask[:, :-1]]
    del gx
    gy = E[1:, :] * np.conj(E[:-1, :])
    my = (np.angle(gy) / (k0 * dx))[mask[1:, :] & mask[:-1, :]]
    del gy
    if lx.size == 0 or my.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(lx ** 2) + np.mean(my ** 2)))


def _input_tilt_stats(E_in, wavelength, dx):
    """Wrapping-safe transverse tilt statistics of ``E_in`` over its bright
    support: ``(tilt_rms, coherence_ratio)`` (radians / dimensionless), or
    ``None`` when they cannot be formed (empty / degenerate field).

    ``tilt_rms`` is the SAME quantity two collimated-input diagnostics need and
    used to compute independently -- the ``carrier=None`` residual angular
    spread (``_carrier_residual_rms(E_in, None, ...)``) AND the
    ``tilt_aware_rays=False`` launch-warning discriminator -- so this single
    pass replaces the duplicate full-grid ``angle(E[i+1]*conj(E[i]))`` +
    ``np.angle`` computation that ran twice per call (~9.5% of the runtime at
    N=4096).  The arithmetic is byte-identical to both original sites (same
    ``0.05*max`` bright mask, same nearest-neighbour phase increments, same
    ``sqrt(mean(lx^2)+mean(my^2))`` / ``hypot(mean(lx),mean(my))/tilt_rms``)."""
    k0 = 2.0 * np.pi / wavelength
    E = np.asarray(E_in)
    mag = np.abs(E)
    mx = mag.max()
    if not np.isfinite(mx) or mx <= 0:
        return None
    mask = mag > 0.05 * mx
    del mag
    if not mask.any():
        return None
    gx = E[:, 1:] * np.conj(E[:, :-1])
    lx = (np.angle(gx) / (k0 * dx))[mask[:, 1:] & mask[:, :-1]]
    del gx
    gy = E[1:, :] * np.conj(E[:-1, :])
    my = (np.angle(gy) / (k0 * dx))[mask[1:, :] & mask[:-1, :]]
    del gy
    if lx.size == 0 or my.size == 0:
        return None
    tilt_rms = float(np.sqrt(np.mean(lx ** 2) + np.mean(my ** 2)))
    coherent = float(np.hypot(np.mean(lx), np.mean(my)))
    coherence_ratio = coherent / tilt_rms if tilt_rms > 0 else 1.0
    return (tilt_rms, coherence_ratio)


def _compute_carrier(carrier, E_in, wavelength, dx, X, Y, auto_degree=2):
    """Build the carrier reference wavefront ``W(x, y)`` (length units;
    reference phase = ``k0 * W``) and a callable giving its transverse
    gradient -- the ray direction cosines ``L = dW/dx``, ``M = dW/dy``.

    ``carrier`` accepts:

    * ``float`` -- an on-axis point-source conjugate at signed distance
      ``s`` (metres): paraxial spherical wavefront ``W = (x^2+y^2)/(2s)``
      (``s > 0`` for a diverging source in front of the plane).
    * ``ndarray`` -- an explicit wavefront (metres), same shape as ``E_in``.
    * ``'auto'`` -- a low-order (``auto_degree``) polynomial fit of the
      smooth carrier, obtained by least-squares matching the polynomial's
      GRADIENT to the wrapping-safe local tilt field of ``E_in`` over its
      bright support (never per-pixel gradients -- that is F4's failure).
      Curl-free by construction (a scalar potential is fit, not L/M
      separately).

    Returns ``(W_full, grad_fn, w_fn)`` where ``W_full`` is an ``(N, N)``
    array, ``grad_fn(xq, yq)`` returns ``(L, M)`` at the query positions,
    and ``w_fn(xq, yq)`` evaluates the carrier eikonal ``W`` (metres) at
    the query positions -- v5.25.1 (hammer H6): the per-ray OPL must be
    referenced to the carrier congruence by ADDING ``W(x_in)`` at the
    entrance plane; omitting it collapsed every diverging-input trace to
    the collimated focal plane.
    """
    N = X.shape[0]
    if isinstance(carrier, np.ndarray):
        W_full = np.asarray(carrier, dtype=np.float64)
        if W_full.shape != X.shape:
            raise ValueError(
                f"carrier ndarray shape {W_full.shape} != field shape "
                f"{X.shape}")
        gWy, gWx = np.gradient(W_full, dx, dx)

        def grad_fn(xq, yq):
            fx = np.clip(xq / dx + N / 2.0, 0, N - 1).astype(np.int64)
            fy = np.clip(yq / dx + N / 2.0, 0, N - 1).astype(np.int64)
            return gWx[fy, fx], gWy[fy, fx]

        def w_fn(xq, yq):
            fx = np.clip(xq / dx + N / 2.0, 0, N - 1).astype(np.int64)
            fy = np.clip(yq / dx + N / 2.0, 0, N - 1).astype(np.int64)
            return W_full[fy, fx]

        return W_full, grad_fn, w_fn

    if isinstance(carrier, str):
        if carrier != 'auto':
            raise ValueError(
                f"carrier string must be 'auto', got {carrier!r}")
        # Wrapping-safe local tilt field over the bright support.
        k0 = 2.0 * np.pi / wavelength
        E = np.asarray(E_in)
        mag = np.abs(E)
        mask = mag > 0.05 * mag.max()
        gx = E[:, 1:] * np.conj(E[:, :-1])
        Lx = np.angle(gx) / (k0 * dx)
        del gx
        gy = E[1:, :] * np.conj(E[:-1, :])
        My = np.angle(gy) / (k0 * dx)
        del gy
        # R6 / audit F1: build the CONNECTED un-aliased core mask so the
        # gradient fit sees only samples whose local tilt reading is below the
        # grid Nyquist tilt (i.e. NOT wrapped).  The per-pixel tilt magnitude is
        # the vector norm of the two nearest-neighbour phase increments; where it
        # exceeds ``_AUTO_CARRIER_NYQUIST_FRAC * (lambda/2dx)`` the reading is
        # (approaching) aliased.  The restriction engages only when a non-trivial
        # fraction (``_AUTO_CARRIER_ALIAS_FRAC``) of the bright support is
        # aliased; then connected-component labelling keeps only the component
        # containing the BRIGHTEST pixel (the beam centre): the central parabola
        # whose curvature fixes the spherical R.  Wrapped rings past the first
        # Nyquist crossing are separate components (a high-tilt annulus
        # disconnects them) and are excluded.  The brightest-pixel seed is
        # essential -- the min-tilt point can land on a wrapped-to-zero alias
        # ring on a coarse grid, seeding an off-centre blob that injects a
        # spurious tilt.  On a well-sampled input (~no aliasing) ``core`` stays
        # the full bright support so the fit is byte-identical to before and a
        # disconnected multi-emitter field is NOT collapsed onto one beamlet.
        core = mask
        if mask.any():
            _gphx = np.angle(np.roll(E, -1, axis=1) * np.conj(E)) / (k0 * dx)
            _gphy = np.angle(np.roll(E, -1, axis=0) * np.conj(E)) / (k0 * dx)
            _gmag = np.hypot(_gphx, _gphy)
            del _gphx, _gphy
            _nyq_tilt = wavelength / (2.0 * dx)
            _core_ok = mask & (_gmag < _AUTO_CARRIER_NYQUIST_FRAC * _nyq_tilt)
            del _gmag
            _n_bright = int(mask.sum())
            _n_aliased = _n_bright - int(_core_ok.sum())
            if (_n_aliased > _AUTO_CARRIER_ALIAS_FRAC * max(_n_bright, 1)
                    and _core_ok.any()):
                from scipy.ndimage import label as _ndlabel
                _lbl, _nlbl = _ndlabel(_core_ok)
                if _nlbl > 0:
                    _seed_lbl = int(_lbl.ravel()[int(mag.ravel().argmax())])
                    if _seed_lbl > 0:
                        _cand = _lbl == _seed_lbl
                        if int(_cand.sum()) >= _AUTO_CARRIER_MIN_CORE:
                            core = _cand
                del _lbl
            del _core_ok
        mxx = core[:, 1:] & core[:, :-1]
        myy = core[1:, :] & core[:-1, :]
        # sample coords at the increment midpoints
        xax = (np.arange(N) - N / 2.0) * dx
        Xg, Yg = np.meshgrid(xax, xax, indexing='xy')
        xL = 0.5 * (Xg[:, 1:] + Xg[:, :-1])[mxx]
        yL = Yg[:, 1:][mxx]
        Lv = Lx[mxx]
        xM = Xg[1:, :][myy]
        yM = 0.5 * (Yg[1:, :] + Yg[:-1, :])[myy]
        Mv = My[myy]
        # Intensity weights: on a fringed multi-source field the local tilt
        # is noisy, so weight each sample by the local |E| (bright regions
        # -- the imaged carrier -- dominate the low-order fit and the
        # fringe noise averages out).  This is why the fit is robust where
        # per-pixel tilts (F4) fail.
        magI = np.abs(E)
        wL = 0.5 * (magI[:, 1:] + magI[:, :-1])[mxx]
        wM = 0.5 * (magI[1:, :] + magI[:-1, :])[myy]
        del magI
        # Polynomial basis terms b_k(x,y)=x^i y^j (i+j<=deg, i+j>=1); fit the
        # scalar potential W = sum c_k b_k by matching grad(W) to (Lv, Mv).
        terms = [(i, j) for d in range(1, auto_degree + 1)
                 for i in range(d + 1) for j in [d - i]]
        nL, nM = xL.size, xM.size
        A = np.zeros((nL + nM, len(terms)))
        for k, (i, j) in enumerate(terms):
            # d/dx of x^i y^j = i x^(i-1) y^j ; d/dy = j x^i y^(j-1)
            A[:nL, k] = (i * xL ** (i - 1) * yL ** j) if i >= 1 else 0.0
            A[nL:, k] = (j * xM ** i * yM ** (j - 1)) if j >= 1 else 0.0
        rhs = np.concatenate([Lv, Mv])
        w = np.concatenate([wL, wM])
        A = A * w[:, None]
        rhs = rhs * w
        # B7: normal-equations solve (thread-safe; no gelsd/JAX-OpenMP deadlock).
        coef = _solve_lstsq_thread_safe(A, rhs)

        def _poly_and_grad(xq, yq):
            Wq = np.zeros_like(xq, dtype=np.float64)
            Lq = np.zeros_like(xq, dtype=np.float64)
            Mq = np.zeros_like(xq, dtype=np.float64)
            for k, (i, j) in enumerate(terms):
                Wq += coef[k] * xq ** i * yq ** j
                if i >= 1:
                    Lq += coef[k] * i * xq ** (i - 1) * yq ** j
                if j >= 1:
                    Mq += coef[k] * j * xq ** i * yq ** (j - 1)
            return Wq, Lq, Mq

        W_full, _, _ = _poly_and_grad(X, Y)

        def grad_fn(xq, yq):
            _, Lq, Mq = _poly_and_grad(xq, yq)
            return Lq, Mq

        def w_fn(xq, yq):
            Wq, _, _ = _poly_and_grad(xq, yq)
            return Wq

        return W_full, grad_fn, w_fn

    # scalar conjugate distance
    s = float(carrier)
    if s == 0.0:
        raise ValueError("carrier conjugate distance must be non-zero")
    W_full = (X ** 2 + Y ** 2) / (2.0 * s)

    def grad_fn(xq, yq):
        return xq / s, yq / s

    def w_fn(xq, yq):
        return (xq ** 2 + yq ** 2) / (2.0 * s)

    return W_full, grad_fn, w_fn


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
    # v5.17.0 lifetime hygiene (same pattern as the Newton-path upsample):
    # ii/jj are folded into coords -- free them before interpolating, and
    # free coords before the final mask combine.  Byte-identical.
    del ii, jj
    opl_map = map_coordinates(
        np.where(np.isnan(opl_coarse), 0.0, opl_coarse),
        coords, order=1, mode='nearest')
    nan_coarse = np.isnan(opl_coarse).astype(np.float64)
    nan_full = map_coordinates(
        nan_coarse, coords, order=1, mode='nearest')
    del coords
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
    carrier: Optional[Any] = None,
    on_noncollimated: str = 'warn',
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
    return_screen: bool = False,
    amplitude_model: str = 'screen',
    caustic: Optional[str] = None,
    output_plane_distance: float = 0.0,
    caustic_ray_subsample: int = 2,
    caustic_band: str = 'ludwig',
    caustic_min_area_ratio: float = 1e-6,
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

    Quick decision guide (revised per the 2026-07 wave-lens-models audit)
    --------------------
    * Collimated / MLA-relayed input, thick or cemented optics, sub-nm OPD
      -> ``apply_real_lens_traced`` (this function), ``carrier=None``.
    * SINGLE divergent / converging / tilted source through a multi-element
      train -> ``apply_real_lens_traced(carrier='auto')`` (or a known
      conjugate): the carrier drives the reference residual to ~0.
    * MULTI-source / emitter-array direct imaging (e.g. the no-MLA TX case):
      a SINGLE carrier is insufficient -- each source is its own congruence,
      so a per-lens residual survives (the ``on_noncollimated`` guard keeps
      firing even with ``carrier='auto'``) and the spots stay soft.  Use
      ``apply_real_lens`` (all angles via ASM legs; the validated choice for
      this family -- mind its ``sag*theta^2`` oblique floor on fast /
      asymmetric designs, see its Oblique validity boundary).  A future
      K-carrier decomposition would extend the traced model here.
    * Genuinely multi-congruence fields, planes at/near a caustic, or
      JAX-autodiff design loops -> ``apply_real_lens_maslov`` /
      ``apply_real_lens_maslov_jax`` (``integration_method='local_quadrature'``
      at production NA).
    * Aberration-free paraxial reference / isolating model vs geometry
      -> the thin-lens ABCD equivalent.

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
    * The DEFAULT (``carrier=None``) references the correction to a
      collimated plane wave (each pixel ray launched parallel to z), valid
      only when the input beam is ~collimated.  For a divergent / converging
      / tilted / emitter-array input, pass ``carrier=`` (a conjugate, an
      explicit wavefront, or ``'auto'``) to reference the beam's own
      congruence -- this generalises the model to those inputs (audit
      S5.1).  Without a carrier, such inputs blur; the ``on_noncollimated``
      guard warns or delegates to :func:`apply_real_lens` when it detects
      this regime.
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
    ray_subsample : int, default 8
        Compute the ray-trace OPL on every ``ray_subsample``-th pixel
        and bilinearly interpolate to the full grid.  OPL is a very
        smooth function of pupil position, so the default ``8`` (and
        ``ray_subsample=4``) typically loses < 1 nm of fidelity while
        cutting cost by ``ray_subsample**2``.  Set ``1`` to trace every
        pixel (no subsampling).  Recommended for production use on large
        grids.
    min_coarse_samples_per_aperture : int, default 32
        Guardrail against undersampled Newton inversion.  After
        ``ray_subsample`` is applied, the coarse output grid must have
        at least this many samples spanning the lens aperture,
        otherwise the cubic-spline interpolation of the wavefront will
        alias and the result will be wrong.  When the prescription has
        no ``aperture_diameter``, the effective pupil is the largest
        per-surface ``clear_aperture`` (capped at the launch diameter)
        if any surface carries one, else the launch diameter itself
        (= the grid extent ``N * dx``).  (v5.17.1, audit P3-08:
        previously the check was silently SKIPPED for apertureless
        prescriptions.)

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

        .. note::
           This knob is currently a **no-op for the default**
           ``newton_fit='polynomial'`` path: the Newton inversion always
           runs in-process (serial) for the polynomial fit, because the
           process worker rebuilds a SciPy spline
           (``RectBivariateSpline``) rather than the polynomial
           ``_Cheb2DEvaluator``.  ``n_workers`` only engages the process
           pool for ``newton_fit='spline'`` on the CPU path
           (``use_gpu=False``).  Polynomial Newton is cheap at the
           default subsampling, so the serial path is not a bottleneck
           in practice.
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
    carrier : float | ndarray | 'auto' | None, default None
        Reference congruence for the traced correction (audit S5.1).  The
        default (``None``) references the correction to a PLANE WAVE (unit
        input for ``phase_analytic_lens``; rays launched parallel to z),
        which is valid only when the input beam is ~collimated.  For a
        DIVERGENT / converging / tilted / emitter-array input (e.g. no-MLA
        direct imaging), supply the beam's smooth carrier wavefront so the
        reference matches the beam:

        * ``float`` -- an on-axis point-source conjugate at signed distance
          ``s`` metres (``W = (x^2+y^2)/(2s)``; ``s > 0`` diverging in front).
        * ``ndarray`` -- an explicit wavefront ``W(x, y)`` in metres,
          same shape as ``E_in`` (reference phase = ``k0 * W``).
        * ``'auto'`` -- fit a low-order polynomial carrier from ``E_in``'s
          intensity-weighted, wrapping-safe local tilt field (never
          per-pixel gradients -- that is the ``tilt_aware_rays`` failure
          mode).  Extracts the smooth COMMON wavefront; the correct choice
          for a single divergent source of unknown conjugate.

        With a carrier the exit reference is well-conditioned (it focuses
        where the real beam does) and the rays launch along the carrier
        normals, so the traced OPL is applied to the small angular RESIDUAL
        only.  ``carrier`` forces ``fast_analytic_phase=False`` (the fast
        geometric reference cannot carry the carrier congruence).

        Validity: a SINGLE carrier only helps when the residual after its
        removal is small.  It is INSUFFICIENT for genuinely multi-congruence
        fields -- an emitter array whose per-source residual (source spread
        / throw) is not small (e.g. the no-MLA TX imaging case; measured
        design-119 per-lens residual ~0.02-0.04 rad even with
        ``carrier='auto'``, so the ``on_noncollimated`` guard keeps firing
        and the spots stay soft), comparable-power beams at well-separated
        angles (post-DOE at large split), or planes at/near an intermediate
        focus.  Use :func:`apply_real_lens` (split-step, all angles) or
        :func:`apply_real_lens_maslov` there.
    on_noncollimated : {'warn', 'delegate', 'off'}, default 'warn'
        Policy when the input's residual angular spread (after removing any
        ``carrier``) exceeds the collimated-reference validity threshold --
        i.e. the plane-wave-referenced correction would blur (the silent
        regression class the audit was written for).  ``'warn'`` emits a
        ``RuntimeWarning`` pointing at ``carrier=`` / :func:`apply_real_lens`;
        ``'delegate'`` transparently falls back to :func:`apply_real_lens`;
        ``'off'`` disables the check (and its one-FFT-free cost).

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

    sag_dtype : {None, np.float32, np.float64}, default None
        v5.17.0 opt-in geometry dtype, forwarded to the internal
        :func:`apply_real_lens` amplitude legs.  ``None`` (default)
        resolves to the process-wide :func:`set_lens_sag_dtype` value
        (float64 by default -- byte-identical to prior releases).
        ``np.float32`` is ACCURACY-RISKY -- validate the prescription
        with :func:`lens_sag_float32_opd_error` first.  See
        :func:`apply_real_lens` for details.
    sag_chunk_rows : int or None, default None
        v5.17.0 row-band (chunked) memory mode: banded per-surface
        phase screens inside the internal :func:`apply_real_lens`
        amplitude legs AND banded OPL-upsample / exit-field assembly
        here (the latter on the ``ray_subsample > 1`` Newton path).
        ``None`` -> AUTO: row-banded (``max(256, N // 16)`` rows per
        band) when ``N >= 4096``, whole-grid below.  ``0`` forces the
        whole-grid path in BOTH stages; a positive int forces that
        band size.  Byte-identical to the whole-grid path.
    amplitude_model : {'screen', 'ray_density'}, default 'screen'
        Which model supplies the exit-plane AMPLITUDE (the phase is the
        ray-traced OPL either way).

        * ``'screen'`` (default) -- the historical hybrid amplitude: the
          magnitude of a single analytic :func:`apply_real_lens` call
          (ASM through glass).  **Byte-identical to prior releases.**  This
          amplitude is a single-plane phase-screen leg, so it carries no
          asymmetric ray-density redistribution: on a DECENTERED / tilted
          (generally aberrated) element the induced-coma SPOT is
          amplitude-limited (it does not broaden -- P9 / N10a).  Good for
          wavefront / pointing / on-axis work.
        * ``'ray_density'`` (opt-in, niche N12) -- geometric ray-tube energy
          conservation: with ``J = d(x_out,y_out)/d(x_in,y_in)`` the ray-map
          Jacobian (from the analytic gradient of the entrance->exit fit),
          the exit magnitude is ``|E_in(x_in)| / sqrt(|det J|)`` placed at the
          exit ray position with the traced OPL phase.  This ``1/sqrt(|det J|)``
          IS the asymmetric coma redistribution the screen leg lacks, so the
          decentered / aberrated SPOT broadens (usable for PSF / spot-size /
          EE metrics, not just wavefront).  Energy-conserving in the geometric
          limit (no silent renormalisation).

          **Caustic caveat.**  ``det J -> 0`` (or a sign change) at a fold, so
          the single-branch amplitude diverges there.  This mode DETECTS the
          fold (relative floor on ``|det J|`` + adjacent sign change), CAPS the
          amplitude (never returns inf/nan), and emits a one-time
          ``RuntimeWarning`` steering to :func:`apply_real_lens_gbd` /
          :func:`apply_real_lens_fga` -- it does NOT sum the multi-valued ray
          branches (no KMAH/Maslov phase); GBD/FGA remain the caustic reference.

          Requires ``inversion_method='newton'`` and the CPU path
          (``use_gpu=False``); incompatible with ``return_screen=True``.
    caustic : {None, 'single', 'multibranch', 'uniform'}, default None
        Opt-in MULTIBRANCH (KMAH / Maslov) refinement of the ``ray_density``
        amplitude (niche N13 / K1).  ``None`` / ``'single'`` (default) is the
        single-branch behaviour above -- BYTE-IDENTICAL to prior releases.

        ``'uniform'`` (niche N16 / K4; requires ``amplitude_model='ray_density'``
        + the CPU path) adds the Chester-Friedman-Ursell UNIFORM Airy DARK-side
        completion on top of the multibranch bright field so the traced field is
        diffraction-correct THROUGH a fold caustic.  The pure ``'multibranch'``
        geometric sum is identically ZERO on the DARK side of a fold (no real ray
        branch there) and so drops the exponentially-decaying Airy tail; the
        ``'uniform'`` mode meridional-ray-traces the fold to get the caustic
        radius ``r_c``, the fold parameter ``zeta(r) = kappa (r_c - r)`` and the
        mean phase, FITS the two smooth Airy coefficients to the bright field just
        inside ``r_c``, and continues the SAME ``uniform_fold_airy`` CFU kernel to
        ``zeta < 0`` to fill the dark tail -- closing the K1 fold-truth gap
        (windowed r2m -14.8% -> ~2%, energy 0.80 -> ~1.0 vs the direct
        Rayleigh-Sommerfeld ``caustic_fold_ref``).  It applies to a
        rotationally-symmetric SINGLE fold RING (collimated / rot-sym input,
        centred prescription); a decentered / astigmatic fold, a carrier tilt, a
        plane with no fold, or a CUSP / multiple rings (the Pearcey regime) are
        DETECTED and fall back to the plain multibranch field (finite, one-time
        warning).  Bright side ``r < r_c`` is byte-identical to ``'multibranch'``.

        ``'multibranch'`` (requires ``amplitude_model='ray_density'``) is the
        multi-valued generalisation: where the ray map FOLDS (``det J -> 0`` /
        sign change) it gathers ALL real ray branches reaching each output
        pixel, weights each ``|E_in(x_in^b)| / sqrt(|det J_b|)``, applies the
        Maslov phase ``exp(-i (pi/2) KMAH_b)`` (``KMAH_b`` = the number of
        ``det J`` sign changes -- astigmatic focal-line crossings -- along that
        branch's ray, counted ANALYTICALLY from the exact quadratic
        ``det Q(z)``), and SUMS COHERENTLY.  It reuses the existing
        :func:`apply_real_lens_traced_multibranch` branch-finder + det-Q KMAH
        counter (Ludwig uniform-Airy swap in the Kravtsov-Orlov caustic band),
        so the field is FINITE at the fold (never inf/nan / no ``sqrt``-blowup)
        and the sqrt-singularity resolves into the finite fold-diffraction
        profile.  Output is taken at ``output_plane_distance`` past the exit
        vertex, so a through-focus caustic plane is reached DIRECTLY (no
        separate ASM step).

        Scope / honest caveat.  The multibranch field is a GEOMETRIC (ART)
        construction: on the DARK side of a fold no real ray branches exist, so
        it carries no evanescent diffraction tail there.  On a fine, wave-
        resolved grid the single-branch ray-density exit field ASM-propagated
        to the fold plane (a genuine wave propagation) is therefore MORE
        accurate for the full caustic-ring r2m/EE than the pure multibranch
        sum; keep :func:`apply_real_lens_gbd` / :func:`apply_real_lens_fga` (or
        single-branch ``ray_density`` + ASM) as the quantitative caustic
        reference.  Multibranch is the tool when you need the coherent
        multi-arrival field / KMAH branch decomposition AT the caustic plane in
        one call (finite, no blow-up) rather than an aliasing-sensitive wave
        propagation.  See ``docs/plan_kmah_gpu_perf_2026_07_21.md`` (N13) and
        ``tests/unit/test_niche_k1_kmah_caustic.py`` for the measured envelope.
    output_plane_distance : float, default 0.0
        Observation-plane distance [m] past the last surface's exit vertex,
        honoured ONLY by ``caustic='multibranch'`` (the single-branch / screen
        paths always output at the exit vertex; a non-zero value with any other
        mode raises).  ``0.0`` = the exit vertex.
    caustic_ray_subsample : int, default 2
        ``caustic='multibranch'`` launch-grid spacing in units of ``dx`` (one
        ray per ``caustic_ray_subsample`` pixels); smaller = denser ray
        branches = finer caustic resolution.  Distinct from ``ray_subsample``
        (the Newton-inversion coarse grid), which is unused on the multibranch
        path.
    caustic_band : {'ludwig', 'plain'}, default 'ludwig'
        ``caustic='multibranch'`` fold-caustic band model: ``'ludwig'`` swaps a
        coalescing branch pair in the Kravtsov-Orlov band for the uniform
        Airy-fold field (finite at the fold); ``'plain'`` keeps the raw branch
        sum (diverges toward the fold).
    caustic_min_area_ratio : float, default 1e-6
        ``caustic='multibranch'`` degenerate-triangle skip threshold (mapped /
        launch area) -- the caustic set where ART is undefined.

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

    # ---- N12 (P11): opt-in ray-density (Jacobian) amplitude model -------
    # ``amplitude_model='screen'`` (default) is byte-identical to prior
    # releases -- none of the ray-density code below runs.  ``'ray_density'``
    # replaces the exit magnitude with the geometric ray-tube amplitude
    # ``|E_in| / sqrt(|det J|)`` (see the docstring), keeping the traced OPL
    # phase.  It is confined to the CPU Newton path so the entrance->exit fits
    # + Newton inverse it reuses are available in-process; the fold-detection
    # flag is a 1-element list so the nested amplitude closure can set it.
    if amplitude_model not in ('screen', 'ray_density'):
        raise ValueError(
            f"amplitude_model must be 'screen' or 'ray_density', got "
            f"{amplitude_model!r}.")
    _ray_density = (amplitude_model == 'ray_density')
    _rd_fold_detected = [False]
    # ---- N13 (K1): opt-in MULTIBRANCH (KMAH/Maslov) caustic refinement ----
    # ``caustic=None``/'single' (default) is byte-identical to prior releases.
    # ``'multibranch'`` routes the whole call to the existing
    # ``apply_real_lens_traced_multibranch`` (branch-finder + det-Q KMAH
    # counter); it is the multi-valued generalisation of the ray-density
    # amplitude, so it requires ``amplitude_model='ray_density'`` and the CPU
    # path.  The routing itself happens after the shared square-grid / dy / mirror
    # guards below (so it inherits them), via ``_multibranch``.
    if caustic is not None and caustic not in ('single', 'multibranch',
                                               'uniform'):
        raise ValueError(
            "caustic must be None, 'single', 'multibranch', or 'uniform', got "
            f"{caustic!r}.")
    _multibranch = (caustic == 'multibranch')
    # ---- N16 (K4): opt-in UNIFORM (Airy) dark-side completion --------------
    # ``caustic='uniform'`` runs the multibranch (bright side) and adds the
    # Chester-Friedman-Ursell dark-side Airy tail so the traced field is
    # diffraction-correct THROUGH a fold caustic; it shares the multibranch's
    # ray_density / CPU / output_plane_distance requirements (routed via
    # ``_uniform``).
    _uniform = (caustic == 'uniform')
    _mb_family = _multibranch or _uniform
    if _mb_family:
        _mode_name = 'multibranch' if _multibranch else 'uniform'
        if not _ray_density:
            raise ValueError(
                f"caustic={_mode_name!r} requires amplitude_model='ray_density' "
                "(it is the multi-valued generalisation of the ray-density "
                f"amplitude); got amplitude_model={amplitude_model!r}.")
        if use_gpu or amp_use_gpu:
            raise ValueError(
                f"caustic={_mode_name!r} requires the CPU path "
                "(use_gpu=amp_use_gpu=False): it reuses the CPU ray-trace "
                "branch-finder + analytic det-Q KMAH counter.")
    if float(output_plane_distance) != 0.0 and not _mb_family:
        raise ValueError(
            "output_plane_distance is only honoured by caustic='multibranch' / "
            "'uniform' (the single-branch / screen paths output at the exit "
            f"vertex); got output_plane_distance={output_plane_distance!r} with "
            f"caustic={caustic!r}.")
    if _ray_density:
        if return_screen:
            raise ValueError(
                "amplitude_model='ray_density' is incompatible with "
                "return_screen=True: the ray-density amplitude depends on "
                "|E_in| (and the traced phase), so it cannot be baked into an "
                "input-independent prepared screen.")
        if use_gpu:
            raise ValueError(
                "amplitude_model='ray_density' requires the CPU path "
                "(use_gpu=False): it re-uses the in-process entrance->exit "
                "fits + Newton inverse to build |E_in|/sqrt(|det J|).")
        if inversion_method != 'newton':
            raise ValueError(
                "amplitude_model='ray_density' requires "
                "inversion_method='newton' (it evaluates det J from the "
                f"Newton entrance->exit fits); got {inversion_method!r}.")
        # Full-grid Newton phase (no amp mask) so the reconstructed phasor
        # covers the WHOLE ray-valid region -- the ray-density amplitude has
        # energy (the coma tail) where |E_analytic| is small, which the amp
        # mask would otherwise drop.  Whole-grid final assembly (below) so the
        # magnitude swap sees the fully-built exit field.
        newton_amp_mask_rel = 0.0

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

    # ---- N13 (K1): dispatch the MULTIBRANCH (KMAH/Maslov) caustic sum ------
    # Route the whole call to the existing multibranch branch-finder (REUSE,
    # do not reimplement).  It gathers all real ray branches per output pixel,
    # weights each ``|E_in| / sqrt(|det J|)``, applies ``exp(-i (pi/2) KMAH)``,
    # and sums coherently -- finite at the fold (Ludwig uniform-Airy), output
    # at ``output_plane_distance`` past the exit vertex.  Bypasses the Newton
    # OPL machinery entirely (a ray-native construction; no phase unwrap, so
    # the ``on_undersample`` OPD-sampling check does not apply).
    if _mb_family:
        _mode_name = 'multibranch' if _multibranch else 'uniform'
        # ``carrier`` -> ``input_carrier``: the multibranch launch is one
        # tilted congruence taking a transverse carrier wavevector (rad/m) or
        # 'auto'; the traced None/'auto' vocabulary maps directly.  A scalar
        # conjugate / explicit-wavefront carrier is not representable as a
        # single launch tilt here.
        if carrier is None:
            _input_carrier = None
        elif isinstance(carrier, str) and carrier == 'auto':
            _input_carrier = 'auto'
        else:
            raise ValueError(
                f"caustic={_mode_name!r} supports carrier=None or "
                "carrier='auto' only (the launch is one tilted congruence); "
                f"got carrier={carrier!r}.  Use the single-branch ray_density "
                "path for a scalar-conjugate / explicit-wavefront carrier.")
        if _uniform:
            # N16 (K4): multibranch bright side + CFU uniform Airy dark tail
            # (rotationally-symmetric fold ring; falls back to plain
            # multibranch for cusp / non-symmetric / non-fold cases).
            from ._lens_traced_uniform import apply_real_lens_traced_uniform
            _mb = np.asarray(apply_real_lens_traced_uniform(
                E_in,
                prescription=prescription,
                wavelength=wavelength,
                dx=dx,
                output_plane_distance=float(output_plane_distance),
                ray_subsample=int(caustic_ray_subsample),
                min_area_ratio=float(caustic_min_area_ratio),
                caustic_band=caustic_band,
                input_carrier=_input_carrier,
            ))
        else:
            from ._lens_traced_multibranch import (
                apply_real_lens_traced_multibranch,
            )
            _mb = np.asarray(apply_real_lens_traced_multibranch(
                E_in,
                prescription=prescription,
                wavelength=wavelength,
                dx=dx,
                output_plane_distance=float(output_plane_distance),
                ray_subsample=int(caustic_ray_subsample),
                min_area_ratio=float(caustic_min_area_ratio),
                caustic_band=caustic_band,
                input_carrier=_input_carrier,
            ))
        _target_cdtype = (E_in.dtype if np.iscomplexobj(E_in)
                          else np.complex128)
        if _mb.dtype != _target_cdtype:
            _mb = _mb.astype(_target_cdtype)
        call_progress(progress, 'real_lens_traced', 1.0, 'done')
        return _mb

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
    # v5.17.0: sag_chunk_rows=None resolves to AUTO (banded when N >= 4096);
    # pass 0 to force the whole-grid path.  The caller's RAW kwarg also flows
    # to the apply_real_lens amp legs so both stages resolve -- and band --
    # consistently.
    # v5.17.1 (audit P2-05): forward the RAW kwarg, not the resolved value.
    # The resolver maps 0 -> None, and apply_real_lens re-resolves None ->
    # AUTO, so forwarding the resolved value silently re-enabled row-banding
    # in the amp legs when the caller passed the documented force-whole-grid
    # sentinel 0.  Both stages resolve the raw value against the same N, so
    # None / positive ints band identically in both stages and 0 now forces
    # whole-grid in BOTH.
    from ._lens_real import _resolve_sag_chunk_rows
    _sag_chunk_rows_raw = sag_chunk_rows
    sag_chunk_rows = _resolve_sag_chunk_rows(sag_chunk_rows, N)
    _chunk_assembly = (
        sag_chunk_rows is not None and int(sag_chunk_rows) > 0
        and max(1, int(ray_subsample)) > 1
        and inversion_method == 'newton'
        and not _ray_density   # ray-density does the magnitude swap on the
                               # whole-grid exit field (below), not per band
    )
    if _chunk_assembly:
        X = Y = None
    else:
        X, Y = np.meshgrid(x, x)

    # ----- Carrier-referenced correction (audit S5.1) -------------------
    # Traced's default correction is referenced to a PLANE WAVE (unit input
    # for phase_analytic_lens; rays launched parallel to z), valid only when
    # the input congruence is ~collimated.  For a divergent / tilted /
    # emitter-array input, supply the beam's own smooth CARRIER wavefront:
    # the reference then matches the beam (well-conditioned exit reference,
    # fixing N5) and the rays launch along the carrier normals, so the
    # traced correction is applied to the small residual only.  W is in
    # length units (reference phase = k0 * W); grad(W) gives the ray
    # direction cosines.
    _k0 = 2.0 * np.pi / wavelength
    _carrier_W = None
    _carrier_grad = None
    _carrier_W_fn = None
    # N5 (2026-07-19): tilt_aware_rays with NO explicit carrier still needs an
    # entrance-eikonal REFERENCE so the exit wavefront carries the input
    # congruence -- the same physics the carrier path's H6 fix restored.  On the
    # DEFAULT preserve_input_phase=True the plane-wave reference already works (a
    # diverging/tilted tilt_aware input focuses at its true image: E_analytic
    # carries the input eikonal and the plane-wave reference leg does not
    # subtract it, unlike the carrier path's exp(i*k0*W) leg that made the H6
    # collapse surface on the default path).  But preserve_input_phase=False
    # builds the exit phase from opl_traced ALONE, which the ray tracer
    # accumulates only from the entrance plane forward -- dropping k0*W(x_in) and
    # collapsing a diverging/tilted input to the collimated focal plane (the H6
    # class, here confined to the non-default mode).  Fix: auto-fit the input's
    # smooth carrier and thread it through the SAME carrier plumbing (reference
    # leg exp(i*k0*W) + the H6 entrance-eikonal OPL term); the per-pixel tilt
    # LAUNCH below is retained (the tilt_aware branch wins).  A (near-)collimated
    # input fits W == 0 exactly (real / globally-phased field -> zero tilt
    # samples), so it keeps the byte-identical plane-wave path.
    _carrier_src = carrier
    if _carrier_src is None and tilt_aware_rays:
        _carrier_src = 'auto'
    if _carrier_src is not None:
        if X is None:
            _cx = (np.arange(E_in.shape[0]) - E_in.shape[0] / 2) * dx
            _CX, _CY = np.meshgrid(_cx, _cx)
        else:
            _CX, _CY = X, Y
        _cW, _cGrad, _cWfn = _compute_carrier(
            _carrier_src, E_in, wavelength, dx, _CX, _CY)
        del _CX, _CY
        # Explicit carrier always engages.  The IMPLICIT tilt_aware auto-carrier
        # engages only when the fitted eikonal is non-trivial, so a flat-wavefront
        # input (fits W == 0) keeps the byte-identical plane-wave reference (pin:
        # collimated tilt_aware unchanged in both preserve_input_phase modes).
        _engage = True
        if carrier is None:
            _mag0 = np.abs(E_in)
            _pk0 = float(_mag0.max()) if _mag0.size else 0.0
            if _pk0 > 0:
                _bright0 = _mag0 > 0.05 * _pk0
                _peakW = (float(np.nanmax(np.abs(_cW[_bright0])))
                          if _bright0.any() else 0.0)
            else:
                _peakW = 0.0
            _engage = (_peakW * _k0) > _TILT_EIKONAL_MIN_RAD
        if _engage:
            _carrier_W, _carrier_grad, _carrier_W_fn = _cW, _cGrad, _cWfn
            # The fast_analytic_phase reference is the lens's on-axis geometric
            # phase (input-independent), which cannot carry the carrier
            # congruence; force the full wave reference when a carrier is set.
            if fast_analytic_phase:
                fast_analytic_phase = False

    # F1 (audit) collimation guard: measure the residual angular spread
    # (after removing any carrier) and warn / delegate when the input is
    # too far from the reference congruence for the traced correction to be
    # accurate.  With a carrier supplied the residual is small (the carrier
    # absorbs the divergence), so this only fires for an UNREFERENCED
    # non-collimated input -- exactly the silent-blur regression class.
    # C4 (perf): the carrier=None residual IS the raw input tilt RMS -- the SAME
    # quantity the tilt_aware_rays=False launch-warning block below computes.
    # Compute the wrapping-safe tilt stats ONCE here and reuse them there (saves
    # one full-grid phase-increment + np.angle pass, ~9.5% of runtime at N=4k).
    # None until computed; the launch-warning block computes it if we skipped.
    _input_tilt = None
    if on_noncollimated != 'off':
        try:
            if _carrier_W is None:
                _input_tilt = _input_tilt_stats(E_in, wavelength, dx)
                _resid = _input_tilt[0] if _input_tilt is not None else 0.0
            else:
                _resid = _carrier_residual_rms(E_in, _carrier_W, wavelength, dx)
        except (ValueError, RuntimeError, FloatingPointError):
            _resid = 0.0
        if _resid > _NONCOLLIMATED_RESID_THRESH:
            if on_noncollimated == 'delegate':
                return apply_real_lens(
                    E_in, prescription=lens_prescription,
                    wavelength=wavelength, dx=dx, bandlimit=bandlimit,
                    use_gpu=amp_use_gpu, wave_propagator=wave_propagator,
                    sag_dtype=sag_dtype, sag_chunk_rows=sag_chunk_rows,
                    progress=progress)
            else:  # 'warn'
                import warnings
                warnings.warn(
                    f"apply_real_lens_traced: input residual angular spread "
                    f"{_resid:.3f} rad exceeds the collimated-reference "
                    f"validity threshold ({_NONCOLLIMATED_RESID_THRESH} "
                    f"rad).  The plane-wave-referenced traced correction "
                    f"will be inaccurate (blurred).  Pass carrier= (a "
                    f"conjugate distance, an explicit wavefront, or 'auto') "
                    f"to reference the beam's own congruence, or use "
                    f"apply_real_lens.  Set on_noncollimated='delegate' to "
                    f"fall back automatically, or 'off' to silence.",
                    RuntimeWarning, stacklevel=2)

    # Reference input for the analytic lens-phase leg: the carrier
    # wavefront when supplied, else a unit plane wave (legacy default).
    def _reference_input():
        if _carrier_W is not None:
            return np.exp(1j * _k0 * _carrier_W).astype(E_in.dtype)
        return np.ones_like(E_in)

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

            # v5.17.2 (audit P2-21): honour a pinned set_max_ram() budget --
            # the doubled parallel working set must fit the effective
            # budget, not just physical free RAM (get_ram_budget() equals
            # the psutil read when no override is set).
            from ..memory import get_ram_budget
            _free_gb = min(int(_psutil.virtual_memory().available),
                           get_ram_budget()) / 1e9
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
                sag_dtype=sag_dtype, sag_chunk_rows=_sag_chunk_rows_raw,
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
            ones_input = _reference_input()  # carrier wavefront or plane wave

            def _amp_pw_call():
                return apply_real_lens(
                    ones_input, prescription=lens_prescription, wavelength=wavelength, dx=dx,
                    bandlimit=bandlimit, use_gpu=amp_use_gpu,
                    wave_propagator=wave_propagator,
                    sag_dtype=sag_dtype, sag_chunk_rows=_sag_chunk_rows_raw,
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
            sag_dtype=sag_dtype, sag_chunk_rows=_sag_chunk_rows_raw,
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
                    _reference_input(), prescription=lens_prescription, wavelength=wavelength, dx=dx,
                    bandlimit=bandlimit, use_gpu=amp_use_gpu,
                    wave_propagator=wave_propagator,
                    sag_dtype=sag_dtype, sag_chunk_rows=_sag_chunk_rows_raw,
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

    # ---- N10a: NO field-frame amplitude override (removed 2026-07-20) -----
    # The P9 build swapped the amplitude leg to the bare input envelope
    # (``E_out = E_in * exp(i k0 opl)``) for field-frame (decenter/tilt)
    # prescriptions, on the theory that coma is a pure exit-pupil PHASE
    # aberration.  The adversarial verifier REFUTED this: forcing the input-
    # envelope amplitude on CENTERED geometry (a 1e-7 decenter, or an exact-
    # conic ``sag_callable``) already widened the on-axis EE80 by ~8% with ZERO
    # decenter (grid-robust), so the reported "1.097 broadening / within 1.6% of
    # ZOS" was an amplitude-MODEL artefact (decentered override-amp compared to
    # centered analytic-amp), not induced coma.  Held to ONE amplitude model the
    # traced EE80 under decenter is unstable and wavelength/plane-dependent
    # (|E_analytic|: 0.88x @1.31 / 1.09x @0.633 at the paraxial image; |E_in|:
    # 0.99x @1.31 / 0.95x @0.633) -- because the traced hybrid's GRID-INDEXED
    # amplitude cannot carry the transverse walk-off (the coma flare is an
    # asymmetric ray-DENSITY redistribution the Newton-inverted OPL alone does
    # not put into |E|), and the singlet's paraxial plane is strongly defocused.
    # This is a genuine traced-model limit of the same class as the P3 single-
    # plane analytic limit.  The decenter GEOMETRY + OPL the traced model now
    # carries are correct (centroid / sign-mirror / tilt all oracle-matched),
    # but its decentered-spot EE is amplitude-limited, so the amplitude leg is
    # left as the standard self-consistent reconstruction here (no swap).  The
    # accurate decentered-coma EE reference is ``apply_real_lens_gbd`` (N10b):
    # its beamlets carry the walk-off amplitude, so it BROADENS matching ZOS
    # (ratio 1.035 @1.31um) and the geom-spot oracle (~1% on the ratio).  See
    # docs/audit_real_lens_displaced_2026_07_19.md (P9 / N10a) for the full
    # envelope + the routing to GBD.  ``_prescription_has_field_frame`` is kept
    # as the field-frame detector (used by the tests and available for routing).

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
    if min_coarse_samples_per_aperture:
        if aperture is not None:
            ap_diameter = float(aperture)
            _ap_label = 'aperture'
        else:
            # v5.17.1 (audit P3-08): the floor was documented as enforced
            # against the launch radius when no ``aperture_diameter`` is
            # set, but the guard was silently skipped for apertureless
            # prescriptions.  Derive the effective pupil from the largest
            # per-surface ``clear_aperture`` when present (the actual
            # pupil-limiting hardware, capped at the launch diameter the
            # coarse grid actually spans), else the launch diameter itself
            # (= the grid extent), so apertureless prescriptions get the
            # same aliasing protection.
            _cas = [float(s['clear_aperture'])
                    for s in (lens_prescription.get('surfaces') or [])
                    if isinstance(s, dict)
                    and s.get('clear_aperture') is not None]
            if _cas:
                ap_diameter = min(max(_cas), 2.0 * launch_radius)
                _ap_label = 'largest clear_aperture'
            else:
                ap_diameter = 2.0 * launch_radius
                _ap_label = 'launch diameter (grid extent)'
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
                f'the {ap_diameter*1e3:.2f}-mm {_ap_label} (threshold '
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
    elif _carrier_grad is not None:
        # Carrier-referenced launch (audit S5.1): rays follow the carrier
        # normals grad(W) at their entrance positions, so the ray-traced
        # OPL is referenced to the beam's own congruence (matching the
        # exp(i*k0*W) amplitude reference) rather than a plane wave.
        L_in, M_in = _carrier_grad(h_x, h_y)
        L_in = np.asarray(L_in, dtype=np.float64).ravel()
        M_in = np.asarray(M_in, dtype=np.float64).ravel()
    else:
        # 4.10: emit a one-time warning when the input field has a
        # measurable transverse tilt and tilt_aware_rays=False.  The
        # plane-wave reference OPD becomes inaccurate when the input
        # tilt is comparable to lambda / aperture.  Estimate the
        # transverse tilt as the RMS of grad(phase) / k0 over the
        # support of |E_in|; cap the check via a try-except so degenerate
        # input fields don't crash apply_real_lens_traced.
        # 4.10: emit a one-time warning when the input field has a measurable
        # transverse tilt and tilt_aware_rays=False -- the plane-wave reference
        # OPD becomes inaccurate when the input tilt is comparable to
        # lambda / aperture.  The tilt statistics (wrapping-safe nearest-
        # neighbour phase increments over the bright support; see
        # :func:`_input_tilt_stats`) are the SAME the noncollimated guard used
        # above, so we REUSE its result here (C4 perf) -- only computing them
        # when that guard was skipped (on_noncollimated='off').  The
        # coherence_ratio distinguishes a genuine single-beam tilt (~1, where
        # tilt_aware_rays=True would help) from a multi-beam / post-DOE
        # interference field (<<1, where it cannot -- F4 audit), so the two
        # branches point the user at the right fix.  Best-effort: a degenerate
        # field yields None / an exception, both silently skipping the warning.
        try:
            if _input_tilt is None:
                _input_tilt = _input_tilt_stats(E_in, wavelength, dx)
            if _input_tilt is not None:
                tilt_rms, coherence_ratio = _input_tilt
                if tilt_rms > 1e-4:
                    import warnings
                    if coherence_ratio >= 0.5:
                        warnings.warn(
                            "apply_real_lens_traced: tilt_aware_rays=False "
                            f"with a non-trivial single-beam input tilt "
                            f"(RMS = {tilt_rms:.2e} rad).  The plane-wave "
                            "reference OPD is off by an amount proportional "
                            "to (tilt * aperture); set tilt_aware_rays=True "
                            "for tilt-sensitive analyses.",
                            RuntimeWarning, stacklevel=3,
                        )
                    else:
                        warnings.warn(
                            "apply_real_lens_traced: tilt_aware_rays=False "
                            f"with a non-trivial input tilt of no single "
                            f"direction (RMS = {tilt_rms:.2e} rad, coherence "
                            f"{coherence_ratio:.2f}, i.e. INCOHERENT) -- a "
                            "divergent, multi-beam, or post-DOE interference "
                            "field.  Do NOT set tilt_aware_rays=True here "
                            "(per-pixel single-direction estimation fails on "
                            "such fields); pass carrier= (a conjugate, a "
                            "wavefront, or 'auto') to reference the beam's "
                            "congruence, or use apply_real_lens.",
                            RuntimeWarning, stacklevel=3,
                        )
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

    # ---- v5.25.1 (hammer audit H6): carrier entrance eikonal -----------
    # The ray tracer accumulates OPL only from the ENTRANCE plane forward.
    # When a carrier congruence is set, each ray belongs to a wavefront
    # whose phase AT the entrance plane is k0*W(x_in) -- that eikonal must
    # be added so the traced exit wavefront is referenced to the beam's
    # own diverging/converging sphere, CONSISTENT with the
    # exp(i*k0*W) reference leg used by preserve_input_phase.  Omitting it
    # imprinted a spurious -k0*W on the field, cancelling the input
    # divergence the wave model correctly carried: every diverging-input
    # trace collapsed to the COLLIMATED focal plane f and the true image
    # at z_img smeared by NA_exit*(z_img - f) (production exp22: energy
    # over +/-1.8 mm, EE(100um) = 0.9% -- reproduced to the digit; with
    # this term EE(100um) = 0.999 across the R_in = 300/150/100 mm scan
    # and per-group relay chains, no change for collimated input).
    if _carrier_W_fn is not None:
        final.opd = final.opd + _carrier_W_fn(h_x, h_y)

    # ---- v5.25.0 (hammer audit H3): exit-NA Nyquist guard --------------
    # The docstring's critical-sampling rule (dx <= lambda*f/aperture) was
    # documented but never ENFORCED, and violating it is silent: the exit
    # converging wavefront exceeds grid Nyquist (|sin theta| > lambda/2dx)
    # beyond some radius, the aliased annulus folds to WRONG positions,
    # and r^2-weighted far-halo metrics (r2m) read low while EE50/EE80
    # stay plausible -- measured on the dual-oracle f/5 singlet: r2m 40.9
    # vs 65.0 um at dx = 2.24x the limit, fully recovered (64.77, 99.7%)
    # at dx inside the limit.  Guard: the exact per-ray exit direction
    # cosines are already in hand; compare the beam's exit NA against the
    # grid Nyquist angle.  Amplitude-aware: only rays carrying input
    # amplitude >= e^-4 of peak count (a Gaussian's 99.97%-energy disc),
    # so zero-energy aperture-edge rays cannot over-fire the guard.
    # Policy: warn (RuntimeWarning) unless on_undersample == 'silent'.
    # Deliberately NOT an error even under on_undersample='error': the
    # returned field's core metrics remain valid; only far-halo moments
    # degrade -- erroring would break legitimate coarse-dx workflows.
    _ray_ix = np.clip(np.rint(xs_in / dx + E_in.shape[1] / 2).astype(int),
                      0, E_in.shape[1] - 1)
    _dy_eff = dy if dy is not None else dx
    _ray_iy = np.clip(np.rint(xs_in / _dy_eff + E_in.shape[0] / 2).astype(int),
                      0, E_in.shape[0] - 1)
    _amp = np.abs(E_in)[np.ix_(_ray_iy, _ray_ix)]      # (n_launch, n_launch)
    _sig = (_amp >= np.exp(-4.0) * _amp.max()).ravel() & final.alive
    if _sig.any():
        _na_exit = float(np.sqrt(final.L[_sig] ** 2
                                 + final.M[_sig] ** 2).max())
        _dx_eff = max(dx, _dy_eff)
        if _na_exit > 0 and _dx_eff > wavelength / (2.0 * _na_exit):
            _dx_need = wavelength / (2.0 * _na_exit)
            if on_undersample != 'silent':
                import warnings
                warnings.warn(
                    f'apply_real_lens_traced: the exit beam converges at '
                    f'NA_exit={_na_exit:.4f}, so the exit wavefront needs '
                    f'dx <= lambda/(2*NA_exit) = {_dx_need*1e6:.2f} um but '
                    f'the grid has dx = {_dx_eff*1e6:.2f} um.  The '
                    f'beyond-Nyquist annulus of the exit phase ALIASES: '
                    f'far-halo energy lands at wrong radii, so r^2-weighted '
                    f'spot metrics (r2m / second moments) read low while '
                    f'EE50/EE80 stay plausible.  Use a finer grid (dx <= '
                    f'{_dx_need*1e6:.2f} um) for halo-faithful results, or '
                    f'pass on_undersample="silent" to suppress.',
                    RuntimeWarning, stacklevel=2)

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

    # T-P2 (audit perf): optional DIRECT inverse-map fit.  Instead of Newton-
    # inverting the forward map per output pixel, fit ``opl`` as a smooth
    # function of the EXIT coordinates ``(x_out, y_out)`` by scattered
    # Chebyshev least squares from the already-traced ray samples, then
    # evaluate that polynomial on the exit grid -- one lstsq + one poly eval,
    # no per-pixel Newton.  A GLOBAL Chebyshev fit (vs the pre-3.x griddata
    # scatter this file replaced) avoids the Delaunay-edge spikes noted below,
    # while staying opt-in (``inversion_method='fit'``) so the thoroughly-
    # validated Newton path remains the default.  Output convention is
    # identical: on-axis-referenced OPL in metres, NaN outside the exit
    # sample hull.
    _use_fit = (inversion_method == 'fit')
    if _use_fit:
        from numpy.polynomial.chebyshev import chebvander as _chebvander
        from scipy.spatial import ConvexHull as _ConvexHull
        _fo = int(newton_poly_order)
        _xo_s = x_out_grid.ravel()
        _yo_s = y_out_grid.ravel()
        _op_s = opl_grid.ravel()
        _g = np.isfinite(_xo_s) & np.isfinite(_yo_s) & np.isfinite(_op_s)
        _xo_s, _yo_s, _op_s = _xo_s[_g], _yo_s[_g], _op_s[_g]
        _fx_c = 0.5 * (_xo_s.max() + _xo_s.min())
        _fx_h = 0.5 * (_xo_s.max() - _xo_s.min()) or 1.0
        _fy_c = 0.5 * (_yo_s.max() + _yo_s.min())
        _fy_h = 0.5 * (_yo_s.max() - _yo_s.min()) or 1.0
        # total-degree multi-index list, encoded as (P, 2) int for a
        # vectorized column-product Chebyshev design.
        _terms = np.array([[a, b] for a in range(_fo + 1)
                           for b in range(_fo + 1 - a)], dtype=np.intp)

        def _fit_design(ux, uy):
            Vx = _chebvander(ux, _fo)   # (K, _fo+1); col a = T_a(ux)
            Vy = _chebvander(uy, _fo)
            return Vx[:, _terms[:, 0]] * Vy[:, _terms[:, 1]]   # (K, M)

        _Afit = _fit_design((_xo_s - _fx_c) / _fx_h, (_yo_s - _fy_c) / _fy_h)
        # B7: normal-equations solve (thread-safe; no gelsd/JAX-OpenMP deadlock).
        _fit_coef = _solve_lstsq_thread_safe(_Afit, _op_s)
        # Domain: keep only exit pixels inside the convex hull of the ray
        # landing spots -- a vectorized half-plane test (A.x + b <= 0 for
        # every facet), far cheaper than a Delaunay simplex search over the
        # full output grid.  A lens exit region is convex (a disc), so the
        # hull is the exact coverage boundary.
        _heq = _ConvexHull(np.column_stack([_xo_s, _yo_s])).equations  # (F,3)
        _hA = np.ascontiguousarray(_heq[:, :2].T)   # (2, F)
        _hb = _heq[:, 2]

        def _invert_fit(Xw, Yw):
            _sh = np.asarray(Xw).shape
            xw = np.asarray(Xw).ravel()
            yw = np.asarray(Yw).ravel()
            val = _fit_design((xw - _fx_c) / _fx_h,
                              (yw - _fy_c) / _fy_h) @ _fit_coef
            pts = np.column_stack([xw, yw])
            inside = np.all(pts @ _hA + _hb <= 1e-12, axis=1)
            return np.where(inside, val, np.nan).reshape(_sh)

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

    # N12 (P11): the forward-map fits Sx/Sy expose a combined value+gradient on
    # the polynomial path (``_Cheb2DEvaluator.ev_value_and_grad``); the spline
    # path uses the ``.ev(dx=1)`` / ``.ev(dy=1)`` API.  The ray-density
    # amplitude closure needs the Jacobian d(x_out,y_out)/d(x_in,y_in), so it
    # dispatches on this flag.
    _has_combined_fits = (hasattr(Sx, 'ev_value_and_grad')
                          and hasattr(Sy, 'ev_value_and_grad'))

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

    def _invert_newton(Xw, Yw, sub_progress=None, _want_entrance=False):
        """Run Newton iteration to find (xe, ye) such that (Sx, Sy)
        evaluated at (xe, ye) equals (Xw, Yw).  Returns OPL at the
        converged entrance positions plus a validity mask.

        Fully vectorised over the input arrays -- ``Xw`` and ``Yw``
        may be any shape; result has the same shape.

        ``sub_progress`` is an optional ``ProgressScaler`` (or any
        callable ``f(frac, msg)``) driven once per Newton iteration.

        ``_want_entrance`` (N12/P11, internal): when True, ALSO return the
        converged entrance coordinates ``(xe, ye)`` (same shape as ``Xw``) as a
        3-tuple ``(opl, xe, ye)`` so the ray-density amplitude closure can
        evaluate ``det J`` and ``|E_in|`` at the entrance point.  Default False
        keeps the historical single-array return byte-identical.
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
            if _want_entrance:
                xe = cp.asnumpy(xe)
                ye = cp.asnumpy(ye)
        if _want_entrance:
            # N12 (P11): the ray-density amplitude closure needs the converged
            # entrance coordinates (to evaluate det J and |E_in| there).
            return (opl_flat.reshape(Xw.shape),
                    np.asarray(xe).reshape(Xw.shape),
                    np.asarray(ye).reshape(Xw.shape))
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

    def _ray_density_amp_grid(Xg, Yg):
        """N12 (P11): geometric ray-density exit amplitude ``|E_in(x_in)| /
        sqrt(|det J|)`` on the exit-position grid ``(Xg, Yg)``.

        Uses the SAME entrance->exit fits (``Sx``, ``Sy``) + Newton inverse the
        OPL phase uses, so the amplitude and phase are placed at consistent exit
        positions.  For each exit pixel Newton returns the entrance point
        ``(xe, ye)``; ``det J = d(x_out,y_out)/d(x_in,y_in)`` is the analytic
        gradient of the forward-map fits there (energy-conserving in the
        geometric limit -- the SAME ``1/sqrt(|det J|)`` Jacobian the ring-tube
        oracle uses), and ``|E_in|`` is bilinearly sampled at the entrance.  NaN
        where the ray map is out of domain.  ``|det J|`` is floored at a caustic
        (never inf/nan) and the fold is flagged in ``_rd_fold_detected``.
        """
        opl_f, xe_g, ye_g = _invert_newton(Xg, Yg, _want_entrance=True)
        sh = np.asarray(Xg).shape
        invalid = ~np.isfinite(np.asarray(opl_f))
        xef = np.asarray(xe_g, dtype=np.float64).ravel()
        yef = np.asarray(ye_g, dtype=np.float64).ravel()
        # Forward-map Jacobian J = d(x_out,y_out)/d(x_in,y_in) at the entrance.
        if _has_combined_fits:
            _fx, jxx, jxy = Sx.ev_value_and_grad(xef, yef)
            _fy, jyx, jyy = Sy.ev_value_and_grad(xef, yef)
        else:
            jxx = np.asarray(Sx.ev(xef, yef, dx=1))
            jxy = np.asarray(Sx.ev(xef, yef, dy=1))
            jyx = np.asarray(Sy.ev(xef, yef, dx=1))
            jyy = np.asarray(Sy.ev(xef, yef, dy=1))
        det_j = (np.asarray(jxx) * np.asarray(jyy)
                 - np.asarray(jxy) * np.asarray(jyx)).reshape(sh)
        # |E_in| at the entrance (bilinear); rays whose entrance falls outside
        # the input grid contribute zero amplitude.
        from scipy.ndimage import map_coordinates as _mc
        _absin = np.abs(np.asarray(E_in)).astype(np.float64)
        _col = xef / dx + N / 2.0
        _row = yef / dy + N / 2.0
        a_in = _mc(_absin, np.vstack([_row, _col]), order=1,
                   mode='constant', cval=0.0).reshape(sh)
        absdet = np.abs(det_j)
        fin = np.isfinite(absdet) & (~invalid)
        ref = float(np.median(absdet[fin])) if fin.any() else 0.0
        floor = _RAY_DENSITY_CAUSTIC_FLOOR_REL * ref
        # Caustic/fold detection: |det J| driven below the floor (det J -> 0), a
        # large |det J| dynamic range (near a focus/caustic the ray tube
        # collapses so |det J| spans orders of magnitude), or a det J sign
        # change between adjacent (valid) ray cells.
        if fin.any():
            _amin = float(np.min(absdet[fin]))
            _amax = float(np.max(absdet[fin]))
            if floor > 0.0 and _amin < floor:
                _rd_fold_detected[0] = True
            if _amin > 0.0 and _amax / _amin > _RAY_DENSITY_CAUSTIC_MAXMIN:
                _rd_fold_detected[0] = True
        _sd = np.sign(det_j)
        _mh = fin[:, 1:] & fin[:, :-1]
        _mv = fin[1:, :] & fin[:-1, :]
        if (bool(np.any((_sd[:, 1:] * _sd[:, :-1] < 0.0) & _mh))
                or bool(np.any((_sd[1:, :] * _sd[:-1, :] < 0.0) & _mv))):
            _rd_fold_detected[0] = True
        absdet_capped = np.maximum(absdet, floor) if floor > 0.0 else absdet
        with np.errstate(divide='ignore', invalid='ignore'):
            a_rd = a_in / np.sqrt(absdet_capped)
        a_rd = np.where(invalid | (~np.isfinite(a_rd)), np.nan, a_rd)
        # Aperture is a stop at the ENTRANCE: a ray whose entrance falls outside
        # the aperture is physically blocked, so it carries no energy.  Masking
        # on the entrance (vs the final exit-position mask, which for a
        # converging element admits rays whose entrance exceeds the stop) makes
        # the ray-density power exactly the aperture-transmitted input power.
        if aperture is not None:
            r_ent2 = (xef * xef + yef * yef).reshape(sh)
            a_rd = np.where(r_ent2 <= (0.5 * aperture) ** 2, a_rd, np.nan)
        return a_rd

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

    # K3 (N15 perf): the ray-density upsample below reuses the OPL upsample's
    # coarse->full (2, N, N) coordinate stack when their coarse resolutions
    # match (they always do -- both are ``X[::sub, ::sub]``).  Stashed here as
    # ``(_coords, Ns)`` by the OPL sub>1 branch; ``None`` means build a fresh
    # one.  Bounded to this call (freed after the ray-density upsample); no cache.
    _rd_upsample_coords = None
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
        if _use_fit:
            # T-P2: one polynomial evaluation over the whole coarse grid; no
            # amp mask (the fit is cheap everywhere, hull-masked to NaN).
            if preserve_input_phase:
                del amp
            opl_coarse = _invert_fit(Xs, Ys)
        else:
            amp_coarse = amp[::sub, ::sub]
            mask_coarse = _build_newton_mask(amp_coarse)
            if preserve_input_phase:
                # v5.17.1 (audit P3-09): on the sub>1 preserve_input_phase
                # path ``amp`` is never read again (Step 3 combines with
                # E_analytic, not amp) and ``amp_coarse`` is dead after the
                # Newton-mask build -- but amp_coarse is a VIEW, so the
                # full-grid float base (float64 for complex128 fields,
                # ~8.6 GB at N=32768) would otherwise stay resident through
                # the Newton inversion and the entire band assembly.  Free
                # both eagerly -- same lifetime-fix pattern as the v5.16.2
                # eager frees; values/outputs byte-identical.
                del amp_coarse, amp
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
            # K3 (N15 perf): hand the coordinate stack to the ray-density
            # upsample (identical ``Ns``) rather than let it rebuild
            # ``np.indices`` + a second (2, N, N) float64 array.  For the
            # screen path (no ray-density), free it now exactly as before.
            if _ray_density:
                _rd_upsample_coords = (_coords, Ns)
            else:
                del _coords
            opl_map = np.where(nan_full > 0.5, np.nan, opl_map)
            del nan_full
    elif _use_fit:
        # T-P2: full-grid inverse-map fit (no Newton, no amp mask).
        if X is None:
            X, Y = np.meshgrid(x, x)
        opl_map = _invert_fit(X, Y)
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

    # ---- N12 (P11): ray-density (Jacobian) exit amplitude on the wave grid ---
    # Built on the SAME coarse Newton grid the OPL used (or the full grid at
    # sub=1) and upsampled identically, so the ray-density magnitude and the
    # traced OPL phase share exit positions.  ``_chunk_assembly`` is forced off
    # for ray-density, so X/Y exist here.
    ard_map = None
    if _ray_density:
        if sub > 1:
            Xs_rd = X[::sub, ::sub]
            Ys_rd = Y[::sub, ::sub]
            ard_coarse = _ray_density_amp_grid(Xs_rd, Ys_rd)
            from scipy.ndimage import map_coordinates as _mc_rd
            Ns_rd = ard_coarse.shape[0]
            # K3 (N15 perf): reuse the OPL upsample's coordinate stack when it
            # matches this coarse resolution (it always does -- same
            # ``X[::sub, ::sub]`` grid, so ``ii*Ns/N`` / ``jj*Ns/N`` are the
            # SAME float64 array bit-for-bit); otherwise build a fresh one.
            if (_rd_upsample_coords is not None
                    and _rd_upsample_coords[1] == Ns_rd):
                _coords_rd = _rd_upsample_coords[0]
            else:
                ii_rd, jj_rd = np.indices((N, N), dtype=np.float64)
                _coords_rd = np.array([ii_rd * Ns_rd / N, jj_rd * Ns_rd / N])
                del ii_rd, jj_rd
            _a_rd = _mc_rd(np.where(np.isnan(ard_coarse), 0.0, ard_coarse),
                           _coords_rd, order=1, mode='nearest')
            _nan_rd = _mc_rd(np.isnan(ard_coarse).astype(np.float64),
                             _coords_rd, order=1, mode='nearest')
            del _coords_rd
            _rd_upsample_coords = None
            ard_map = np.where(_nan_rd > 0.5, np.nan, _a_rd)
        else:
            ard_map = _ray_density_amp_grid(X, Y)
        if _rd_fold_detected[0]:
            import warnings as _rd_warn
            _rd_warn.warn(
                "apply_real_lens_traced: amplitude_model='ray_density' "
                "detected a fold caustic (det J -> 0 or a sign change) in the "
                "ray map.  The single-branch ray-density amplitude is CAPPED "
                "there (finite, never inf/nan) but is UNRELIABLE near the fold "
                "-- this mode does NOT sum the multi-valued ray branches with "
                "the KMAH/Maslov phase.  Use apply_real_lens_gbd or "
                "apply_real_lens_fga for caustic-faithful amplitude.",
                RuntimeWarning, stacklevel=2)
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
    if return_screen:
        # T-P1 (prepared-traced): the entire traced leg -- ray trace, fits,
        # Newton inversion, phase_analytic_lens, opl_map, and the valid /
        # aperture masks -- is input-independent per
        # (prescription, wavelength, dx, N, carrier).  The ONLY
        # input-dependent factor is E_analytic = apply_real_lens(E_in), which
        # the assembly below multiplies in pointwise (with the masks folding
        # in multiplicatively).  Substituting ones for E_analytic here makes
        # the returned E_out equal the reusable "screen"
        # = mask(valid) * mask(aperture) * exp(1j*(k0*opl - phase_analytic)).
        # prepare_real_lens_traced() caches it; each subsequent call is then
        # one apply_real_lens(E_in) + one complex multiply.  Requires
        # preserve_input_phase=True (else the assembly uses |E_analytic|),
        # newton_amp_mask_rel=0 and tilt_aware_rays=False (else the valid
        # region / opl depend on E_in), and carrier != 'auto'; the factory
        # enforces all four.
        E_analytic = np.ones_like(E_analytic)
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
    # ---- N12 (P11): swap the exit MAGNITUDE to the ray-density amplitude -----
    # The screen-mode ``E_out`` above carries the correct traced OPL phase and
    # the valid / aperture masks (it is 0 outside the ray-covered pupil).  In
    # ray-density mode we keep that phase (its unit phasor) and replace the
    # magnitude with ``|E_in|/sqrt(|det J|)`` -- the geometric ray-tube energy
    # redistribution the screen amplitude lacks.  The unit phasor is 0 exactly
    # where the screen field is 0 (masked region), so the ray-density field
    # inherits the same support without a separate mask; NaN ray-density values
    # (out-of-domain) contribute 0.
    if _ray_density and ard_map is not None:
        _absE = np.abs(E_out)
        with np.errstate(divide='ignore', invalid='ignore'):
            _unit = np.divide(E_out, _absE,
                              out=np.zeros_like(E_out), where=_absE > 0)
        _ard = np.where(np.isfinite(ard_map), ard_map, 0.0)
        E_out = _ard * _unit
    if E_out.dtype != target_cdtype:
        E_out = E_out.astype(target_cdtype)
    call_progress(progress, 'real_lens_traced', 1.0, 'done')
    return E_out


def _carrier_reuse_key(carrier):
    """Hashable key for a carrier that is SAFE to share a prepared screen
    across emitters, or None if it must get its own trace.  'auto' fits the
    carrier from each field, so every emitter's 'auto' carrier is different ->
    NOT reusable.  An ndarray wavefront is per-emitter data -> not reusable.
    A float conjugate distance or None (plane wave) is a shared geometry ->
    reusable."""
    if carrier is None:
        return ('none',)
    if isinstance(carrier, str):
        return None            # 'auto' (or any string mode): never share
    if isinstance(carrier, np.ndarray):
        return None            # explicit per-field wavefront: never share
    try:
        return ('scalar', float(carrier))
    except (TypeError, ValueError):
        return None


def apply_real_lens_traced_multi(
    emitter_fields,
    *,
    prescription: Dict[str, Any],
    wavelength: float,
    dx: float,
    carriers: Any = 'auto',
    reuse_prepared: bool = True,
    **traced_kwargs,
) -> np.ndarray:
    """Coherently sum the traced lens applied to each emitter's field SEPARATELY.

    The traced model assigns one ray-traced OPL per output pixel (the dominant
    congruence), so it is **not linear**: when several emitter beams overlap on
    a pixel, ``traced(sum_k E_k)`` violates the single-OPL assumption.  Each
    emitter taken ALONE is a single congruence, so ``traced(E_k, carrier_k)`` is
    valid, and this returns their coherent sum::

        E_image = sum_k  apply_real_lens_traced(E_k, carrier=carrier_k, ...)

    which is the tractable form of carrier K-decomposition -- the K congruences
    are the *known* emitters, so no blind congruence segmentation is needed.
    It is exact for a single emitter and reproduces every per-emitter congruence
    correctly.

    **When this actually helps (read before using).**  The value is regime-
    dependent, and this is NOT a universal upgrade:

    * The analytic :func:`apply_real_lens` is **exactly linear**, so for it
      ``analytic(sum E_k) == sum analytic(E_k)`` -- there is *zero* benefit; just
      propagate the combined field once.
    * For a **well-corrected** lens the traced OPD correction is ~0, i.e.
      ``traced ~= analytic``, so ``traced(sum E_k)`` is already essentially exact
      and this per-emitter path only *adds* the per-emitter carrier-fit residual.
    * This mode earns its keep only when you genuinely need the traced ray-OPD
      **refinement** (a lens aberrated enough that analytic is insufficient) AND
      the scene is multi-emitter.  There the traced non-linearity is large
      (>100% between ``traced(sum)`` and this sum on strongly-aberrated lenses),
      and applying traced per emitter is the correct way to keep it valid.
    * Note the no-MLA multi-angle *direct-imaging* case was separately found to
      be modelled correctly by **analytic**, not traced -- so for that geometry
      prefer analytic on the combined field; this mode is for when traced's
      refinement is the thing you specifically want.

    Composes with T-P1: pass a shared explicit ``carrier`` (or ``None``) with
    ``reuse_prepared=True`` to pay the trace/fit/Newton cost once across all
    emitters that share it (see ``reuse_prepared``).

    Parameters
    ----------
    emitter_fields : sequence of complex ndarray
        Each ``E_k`` is the field AT THE LENS-INPUT PLANE from emitter ``k``
        alone (propagate each emitter to the lens plane first).  All must share
        the grid.
    carriers : 'auto' | None | float | ndarray | sequence
        Per-emitter carrier passed to :func:`apply_real_lens_traced`.  A scalar
        / string / single ndarray is broadcast to every emitter; a list/tuple of
        length ``len(emitter_fields)`` is used element-wise.  Default ``'auto'``
        fits each emitter's own congruence (drives its residual angular spread
        to ~0), which is what a divergent point-source array needs.
    reuse_prepared : bool
        When True and a carrier is a shared geometry (``None`` or a float
        conjugate distance), a :class:`PreparedTracedLens` screen is built once
        per distinct carrier and reused across emitters -- the trace/fit/Newton
        cost is paid once instead of per emitter.  ``'auto'`` and ndarray
        carriers are always full per-emitter passes (their screens differ).

    Returns
    -------
    E_image : complex ndarray
        The coherently-summed output field, dtype following the emitter fields.
    """
    fields = list(emitter_fields)
    if not fields:
        raise ValueError("apply_real_lens_traced_multi: emitter_fields is empty.")
    n = len(fields)
    shape0 = np.asarray(fields[0]).shape
    for k, E in enumerate(fields):
        if np.asarray(E).shape != shape0:
            raise ValueError(
                f"apply_real_lens_traced_multi: emitter_fields[{k}] shape "
                f"{np.asarray(E).shape} != emitter_fields[0] {shape0}.")

    if isinstance(carriers, (list, tuple)):
        if len(carriers) != n:
            raise ValueError(
                f"apply_real_lens_traced_multi: carriers list length "
                f"{len(carriers)} != number of emitters {n}.")
        carr_list = list(carriers)
    else:
        carr_list = [carriers] * n     # scalar / str / single ndarray broadcast

    # Each per-emitter pass runs full-grid Newton (no amp mask) so an emitter's
    # OWN dim regions are never clipped -- they may still contribute where a
    # later emitter is bright, and the reuse path (prepared screen) already
    # forces this.  tilt_aware_rays is off (the carrier carries the tilt) and
    # the phase is preserved.  These override any conflicting traced_kwargs.
    for _k in ('newton_amp_mask_rel', 'tilt_aware_rays', 'preserve_input_phase',
               'return_screen', 'parallel_amp'):
        traced_kwargs.pop(_k, None)

    N = int(shape0[0])
    prepared_cache = {}
    E_out = None
    for E_k, carrier_k in zip(fields, carr_list):
        E_k = np.asarray(E_k)
        key = _carrier_reuse_key(carrier_k) if reuse_prepared else None
        if key is not None:
            prep = prepared_cache.get(key)
            if prep is None:
                prep = prepare_real_lens_traced(
                    prescription=prescription, wavelength=wavelength, dx=dx,
                    N=N, carrier=carrier_k, **traced_kwargs)
                prepared_cache[key] = prep
            contrib = prep(E_k)
        else:
            contrib = apply_real_lens_traced(
                E_k, prescription=prescription, wavelength=wavelength, dx=dx,
                carrier=carrier_k, newton_amp_mask_rel=0.0,
                tilt_aware_rays=False, preserve_input_phase=True,
                parallel_amp=False, **traced_kwargs)
        E_out = contrib if E_out is None else E_out + contrib
    return E_out


def _flattop_partition_1d(u, cuts, halfwidth):
    """Flat-top cos^2-edge partition of unity over axis ``u`` split at ``cuts``.

    ``len(cuts)+1`` weight arrays (same shape as ``u``), each ~1 in its bin
    interior with a ``cos^2``/``sin^2`` transition of half-width ``halfwidth``
    centred on each cut, so adjacent bins hand off as ``cos^2 + sin^2 = 1`` and
    the whole set **sums to 1**.  Unlike a uniform partition, the transitions
    sit AT the cuts (spectral gaps between congruences), so each whole beam
    lands in one bin instead of being fragmented across bins -- which is what
    makes the per-segment traced pass valid (one congruence per segment).
    Requires ``halfwidth`` below half the smallest cut spacing for exact unity.
    """
    cuts = sorted(float(c) for c in cuts)
    K = len(cuts) + 1
    if K == 1:
        return [np.ones(np.shape(u), dtype=float)]
    hw = max(float(halfwidth), 1e-30)
    W = []
    for k in range(K):
        w = np.ones(np.shape(u), dtype=float)
        if k > 0:                       # rising edge at cuts[k-1]
            s = np.clip((u - (cuts[k - 1] - hw)) / (2.0 * hw), 0.0, 1.0)
            w = w * np.sin(np.pi * s / 2.0) ** 2
        if k < K - 1:                   # falling edge at cuts[k]
            s = np.clip((u - (cuts[k] - hw)) / (2.0 * hw), 0.0, 1.0)
            w = w * np.cos(np.pi * s / 2.0) ** 2
        W.append(w)
    return W


def _occupied_freq_support(power_1d, freqs, frac):
    """Frequency bounds ``(lo, hi)`` of the marginal-power support capturing
    ``frac`` of the total (the highest-power bins)."""
    total = float(power_1d.sum())
    if total <= 0.0:
        return float(freqs[0]), float(freqs[-1])
    order = np.argsort(power_1d)[::-1]
    cum = np.cumsum(power_1d[order]) / total
    klast = int(np.searchsorted(cum, frac)) + 1
    keep = order[:max(1, klast)]
    return float(freqs[keep].min()), float(freqs[keep].max())


def _spectral_gap_cuts(marginal, freqs, lo, hi, valley_frac, peak_frac):
    """Cut frequencies at deep valleys of the 1-D marginal angular power --
    the gaps that SEPARATE distinct beams (congruences).  A valley qualifies as
    a cut only if its power is below ``valley_frac`` of the marginal peak AND it
    is flanked (within the occupied ``[lo, hi]``) by peaks above ``peak_frac``,
    so a single (unimodal) congruence yields NO cuts (one segment = plain
    traced) and only genuinely separated beams are split."""
    p = np.asarray(marginal, dtype=float)
    pk = float(p.max())
    if pk <= 0.0:
        return []
    p = p / pk
    inband = (freqs >= lo) & (freqs <= hi)
    idx = np.where(inband)[0]
    cuts = []
    for i in idx:
        if i <= 0 or i >= len(p) - 1:
            continue
        if p[i] <= p[i - 1] and p[i] < p[i + 1] and p[i] < valley_frac:
            if p[:i].max() > peak_frac and p[i + 1:].max() > peak_frac:
                cuts.append(float(freqs[i]))
    # merge cuts that are closer than a few samples (same valley)
    if cuts:
        merged = [cuts[0]]
        df = float(abs(freqs[1] - freqs[0])) * 3.0
        for c in cuts[1:]:
            if c - merged[-1] > df:
                merged.append(c)
        cuts = merged
    return cuts


def _segment_field_by_angle(E, dx, dy, segments_x, segments_y,
                            power_frac, valley_frac, min_segment_power,
                            max_segments):
    """Partition ``E`` into angular sub-fields at the spectral GAPS between
    beams, so each sub-field is a single congruence.  With
    ``min_segment_power <= 0`` the segments sum to ``E`` EXACTLY.  Returns a
    single segment (the input) when the spectrum is unimodal (nothing to
    separate)."""
    E = np.asarray(E)
    Ny, Nx = E.shape[-2], E.shape[-1]
    F = np.fft.fftshift(np.fft.fft2(E))
    fx = np.fft.fftshift(np.fft.fftfreq(Nx, dx))
    fy = np.fft.fftshift(np.fft.fftfreq(Ny, dy))
    P = np.abs(F) ** 2
    lox, hix = _occupied_freq_support(P.sum(axis=0), fx, power_frac)
    loy, hiy = _occupied_freq_support(P.sum(axis=1), fy, power_frac)

    if segments_x == 'auto':
        cutx = _spectral_gap_cuts(P.sum(axis=0), fx, lox, hix, valley_frac, 0.25)
    else:
        nseg = max(1, int(segments_x))
        cutx = ([] if nseg == 1
                else list(np.linspace(lox, hix, nseg + 1)[1:-1]))
    if segments_y == 'auto':
        cuty = _spectral_gap_cuts(P.sum(axis=1), fy, loy, hiy, valley_frac, 0.25)
    else:
        nseg = max(1, int(segments_y))
        cuty = ([] if nseg == 1
                else list(np.linspace(loy, hiy, nseg + 1)[1:-1]))

    # cap total segments: drop the SHALLOWEST cuts first (D15) -- the valley
    # with the highest marginal power is the weakest separation, so removing it
    # keeps the deepest (best-separating) gaps.  (Previously popped the
    # last-listed = highest-frequency cut, contradicting the comment.)
    mx, my = P.sum(axis=0), P.sum(axis=1)

    def _valley_power(cut, marg, freqs):
        return float(marg[int(np.argmin(np.abs(freqs - cut)))])

    while (len(cutx) + 1) * (len(cuty) + 1) > max_segments:
        cand = ([("x", i, _valley_power(c, mx, fx))
                 for i, c in enumerate(cutx)]
                + [("y", i, _valley_power(c, my, fy))
                   for i, c in enumerate(cuty)])
        if not cand:
            break
        ax, idx, _ = max(cand, key=lambda t: t[2])   # shallowest = most power
        (cutx if ax == "x" else cuty).pop(idx)

    def _hw(cuts, lo, hi):
        # transition half-width: below half the smallest cut spacing (and to
        # the band edges) so the partition stays a partition of unity.
        edges = [lo] + sorted(cuts) + [hi]
        gaps = [edges[i + 1] - edges[i] for i in range(len(edges) - 1)]
        # narrow transition (sharp separation, since the cut sits in a near-
        # zero-power gap so Gibbs ringing is negligible), but < half the
        # smallest spacing so the partition stays a partition of unity.
        return 0.2 * min(gaps) if gaps else (hi - lo)

    hwx = _hw(cutx, lox, hix)
    hwy = _hw(cuty, loy, hiy)
    FX, FY = np.meshgrid(fx, fy)
    Wx = _flattop_partition_1d(FX, cutx, hwx)
    Wy = _flattop_partition_1d(FY, cuty, hwy)
    tot_power = float(np.sum(np.abs(E) ** 2)) + 1e-300
    segments = []
    for wi in Wx:
        for wj in Wy:
            Ej = np.fft.ifft2(np.fft.ifftshift((wi * wj) * F)).astype(E.dtype)
            if float(np.sum(np.abs(Ej) ** 2)) / tot_power > min_segment_power:
                segments.append(Ej)
    if not segments:
        segments = [E.copy()]
    return segments


def apply_real_lens_traced_segmented(
    E_in,
    *,
    prescription: Dict[str, Any],
    wavelength: float,
    dx: float,
    dy: Optional[float] = None,
    n_segments: Any = 'auto',
    valley_frac: float = 0.15,
    power_frac: float = 0.995,
    min_segment_power: float = 1e-3,
    max_segments: int = 32,
    carriers: Any = 'auto',
    return_segments: bool = False,
    **traced_kwargs,
):
    """Traced lens on a single, possibly MULTI-congruence field via blind
    angular segmentation.

    :func:`apply_real_lens_traced` assumes ONE ray congruence per output pixel,
    so it is invalid for a field that superposes several beams / an extended
    multi-angle source.  :func:`apply_real_lens_traced_multi` handles that when
    the emitters are *already separated*; this handles the case where you only
    have the **combined** field, by splitting its angular spectrum at the deep
    VALLEYS between beams (the gaps that separate distinct congruences), so each
    segment captures one whole beam -- single-congruence -> traced-valid.  The
    segments sum to the input EXACTLY when ``min_segment_power=0``; the
    per-segment traced results are coherently summed via
    :func:`apply_real_lens_traced_multi`.

    Splitting at the spectral GAP (not at uniform bin edges) is essential:
    traced is non-linear, so fragmenting ONE beam across bins would *add* error;
    splitting only at true gaps keeps each congruence intact.  A unimodal
    spectrum (a single congruence) yields one segment == plain traced, so this
    is safe to call unconditionally.

    Parameters
    ----------
    n_segments : 'auto' | int | (int, int)
        Segment count.  ``'auto'`` splits at detected spectral valleys (0 cuts
        for a unimodal field); an int forces that many uniform bins-per-axis; a
        pair is ``(n_x, n_y)``.  Total is capped at ``max_segments``.
    valley_frac : float
        A spectral valley counts as a beam-separating gap only if its marginal
        power is below this fraction of the peak (with a real peak on each side).
    min_segment_power : float
        Drop segments carrying less than this fraction of the input power (saves
        traced passes on empty bins); ``0`` keeps the exact partition.
    return_segments : bool
        If True, return the list of segment fields instead of applying the lens
        (for inspection / the partition-sums-to-input check).

    Returns
    -------
    complex ndarray, or list of complex ndarray if ``return_segments``.
    """
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E_in, 'apply_real_lens_traced_segmented')
    if dy is None:
        dy = dx
    if isinstance(n_segments, (tuple, list)) and len(n_segments) == 2:
        sx, sy = n_segments
    elif n_segments == 'auto':
        sx = sy = 'auto'
    else:
        sx = sy = int(n_segments)
    segments = _segment_field_by_angle(
        E_in, dx, dy, sx, sy, power_frac, valley_frac,
        min_segment_power, max_segments)
    if return_segments:
        return segments
    if len(segments) == 1:
        # single congruence -> the plain traced path (no per-segment overhead)
        return apply_real_lens_traced(
            segments[0], prescription=prescription, wavelength=wavelength,
            dx=dx, carrier=carriers, **traced_kwargs)
    return apply_real_lens_traced_multi(
        segments, prescription=prescription, wavelength=wavelength, dx=dx,
        carriers=carriers, **traced_kwargs)


class PreparedTracedLens:
    """A traced lens with its input-independent phase ``screen`` precomputed.

    Built by :func:`prepare_real_lens_traced`.  The entire traced leg (ray
    trace, Chebyshev/spline fits, Newton inversion, ``phase_analytic_lens``,
    the ``opl`` map and the valid / aperture masks) depends only on
    ``(prescription, wavelength, dx, N, carrier)``, so it is computed once and
    stored as ``screen``.  Each call is then just the input-dependent analytic
    leg plus one complex multiply::

        E_out = apply_real_lens(E_in, ...) * screen

    which drops the trace/fit/Newton stages from optimizer / tolerancing /
    multi-field loops entirely (>=2x per call).  Mirrors the library's
    ``PreparedRCWA2D`` / ``PreparedPMM2D`` precedent.

    Memory footprint
    ----------------
    The retained payload is the ``screen`` -- a single ``(N, N)`` complex128
    array of ``N*N*16`` bytes (**64 MB at N=2048, 256 MB at N=4096**); the
    other slots are tiny scalars / the prescription dict.  A prepared lens is
    a user-held object (not a module cache), so it is freed by normal garbage
    collection when it goes out of scope.  In a long-running optimizer /
    tolerancing loop that builds many prepared screens, call
    :meth:`release` to drop the screen deterministically (or reuse one prepared
    object).  There is no library-wide registry entry for these -- their
    lifetime is the caller's to manage.
    """

    __slots__ = ('screen', 'prescription', 'wavelength', 'dx', 'bandlimit',
                 'amp_use_gpu', 'wave_propagator', 'sag_dtype',
                 'sag_chunk_rows', 'N')

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    def release(self) -> None:
        """Free the precomputed ``screen`` (the ``N*N*16``-byte complex128
        array: 64 MB at N=2048, 256 MB at N=4096).

        After release the prepared lens can no longer be called (a subsequent
        ``prepared(E_in)`` raises).  Use in long-running optimizer / tolerancing
        loops to drop a prepared screen you are finished with, without waiting
        for garbage collection.  Idempotent.  ``clear`` is an alias.
        """
        self.screen = None

    clear = release

    def __call__(self, E_in: np.ndarray) -> np.ndarray:
        """Apply the prepared traced lens to ``E_in`` (shape must match N)."""
        if self.screen is None:
            raise RuntimeError(
                "PreparedTracedLens: the screen has been released (.release()/"
                ".clear() was called); rebuild it with prepare_real_lens_traced.")
        E_in = np.asarray(E_in)
        if E_in.shape != self.screen.shape:
            raise ValueError(
                f"PreparedTracedLens: E_in shape {E_in.shape} != prepared "
                f"grid {self.screen.shape}.")
        # Reproduce E_analytic EXACTLY as apply_real_lens_traced's internal
        # amp leg builds it (same 8 kwargs; note use_gpu=amp_use_gpu and the
        # raw sag_chunk_rows; dy is intentionally not forwarded there either).
        E_analytic = apply_real_lens(
            E_in, prescription=self.prescription, wavelength=self.wavelength,
            dx=self.dx, bandlimit=self.bandlimit, use_gpu=self.amp_use_gpu,
            wave_propagator=self.wave_propagator, sag_dtype=self.sag_dtype,
            sag_chunk_rows=self.sag_chunk_rows)
        out = E_analytic * self.screen
        tcd = E_in.dtype if np.iscomplexobj(E_in) else np.complex128
        return out.astype(tcd) if out.dtype != tcd else out


def prepare_real_lens_traced(
    *,
    prescription: Dict[str, Any],
    wavelength: float,
    dx: float,
    N: int,
    carrier: Optional[Any] = None,
    bandlimit: bool = True,
    ray_subsample: int = 8,
    min_coarse_samples_per_aperture: int = 32,
    on_undersample: str = 'error',
    on_noncollimated: str = 'warn',
    inversion_method: str = 'newton',
    newton_fit: str = 'polynomial',
    newton_poly_order: int = 6,
    newton_max_iters: Optional[int] = None,
    amp_use_gpu: bool = False,
    use_gpu: bool = False,
    wave_propagator: Optional[str] = None,
    sag_dtype: Optional[Any] = None,
    sag_chunk_rows: Optional[int] = None,
    n_workers: Optional[int] = None,
    progress: Optional[Any] = None,
) -> PreparedTracedLens:
    """Precompute the input-independent traced-lens screen for reuse (T-P1).

    The returned :class:`PreparedTracedLens` caches the whole trace/fit/Newton
    result, so every subsequent ``prepared(E_in)`` costs one analytic
    ``apply_real_lens`` + one complex multiply -- ideal for optimizer,
    tolerancing and multi-field loops that hold ``(prescription, wavelength,
    dx, N, carrier)`` fixed.

    The screen is exactly input-independent only for
    ``carrier in {None, <explicit wavefront / conjugate distance>}`` (NOT
    ``'auto'``, which fits the carrier from the field), so ``'auto'`` is
    rejected.  With H6 (v5.25.1) the carrier's entrance eikonal ``k0*W`` is
    baked into the cached ``opl`` map, so an explicit scalar-conjugate /
    ndarray carrier's screen focuses a diverging (or converging) input at the
    correct conjugate -- the 121-class per-group workflow, where every group
    has a KNOWN conjugate shared across many emitter fields, pays the
    trace/fit/Newton cost once and reuses the screen per field.
    ``tilt_aware_rays`` is forced False and the amplitude Newton mask is
    disabled (full coarse grid) so the cached ``valid`` region does not depend
    on any particular input; this makes the first (prepare) call a touch more
    expensive, amortized on the first reuse.  The screen is stored at float64
    complex precision; per-call output is cast back to the input dtype.

    ``on_noncollimated`` is honoured only for ``carrier=None`` (the plane-wave
    reference, where the ``ones`` placeholder the screen is built on is
    genuinely collimated).  For an explicit carrier the guard is forced
    ``'off'`` internally: the placeholder is a flat ``ones`` field, not the
    beam, so a scalar/ndarray carrier makes it LOOK strongly non-collimated
    (its residual is the whole carrier tilt) even though the actual reuse
    fields carry exactly that congruence -- the guard would either warn
    spuriously or, under ``'delegate'``, silently hand off to
    ``apply_real_lens`` (which ignores ``return_screen``) and cache a garbage
    screen.  The residual guard is the per-field caller's responsibility.
    """
    if isinstance(carrier, str) and carrier == 'auto':
        raise ValueError(
            "prepare_real_lens_traced cannot cache carrier='auto' (the "
            "carrier is fit from the field -> input-dependent).  Pass an "
            "explicit carrier (conjugate distance or wavefront ndarray) or "
            "None (plane-wave reference).")
    # The screen is built on a ``ones`` PLACEHOLDER (return_screen=True makes
    # it input-independent), so the collimation guard cannot judge it against a
    # carrier: force it off whenever a carrier is set.  For carrier=None the
    # placeholder IS the plane-wave reference, so the caller's value applies
    # (a correct, silent no-op there).
    _screen_noncol = 'off' if carrier is not None else on_noncollimated
    ones = np.ones((int(N), int(N)), dtype=np.complex128)
    screen = apply_real_lens_traced(
        ones, prescription=prescription, wavelength=wavelength, dx=dx,
        bandlimit=bandlimit, ray_subsample=ray_subsample,
        min_coarse_samples_per_aperture=min_coarse_samples_per_aperture,
        on_undersample=on_undersample, preserve_input_phase=True,
        tilt_aware_rays=False, carrier=carrier,
        on_noncollimated=_screen_noncol, parallel_amp=False,
        newton_amp_mask_rel=0.0, inversion_method=inversion_method,
        fast_analytic_phase=False, newton_fit=newton_fit,
        newton_poly_order=newton_poly_order, newton_max_iters=newton_max_iters,
        use_gpu=use_gpu, amp_use_gpu=amp_use_gpu,
        wave_propagator=wave_propagator, sag_dtype=sag_dtype,
        sag_chunk_rows=sag_chunk_rows, n_workers=n_workers, progress=progress,
        return_screen=True)
    return PreparedTracedLens(
        screen=screen, prescription=prescription, wavelength=wavelength,
        dx=dx, bandlimit=bandlimit, amp_use_gpu=amp_use_gpu,
        wave_propagator=wave_propagator, sag_dtype=sag_dtype,
        sag_chunk_rows=sag_chunk_rows, N=int(N))


__all__ = [
    'apply_real_lens_traced',
    'apply_real_lens_traced_multi',
    'prepare_real_lens_traced',
    'PreparedTracedLens',
    'close_worker_pool',
    'set_lens_parallel_amp',
    'get_lens_parallel_amp',
]
