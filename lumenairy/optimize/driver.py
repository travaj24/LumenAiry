"""
lumenairy.optimize.driver -- design_optimize and friends.

v5.1.0 split (Agent E): extracted from ``lumenairy/optimize/core.py``.
Hosts the wave-propagator registry, the finite-difference gradient
helper (:func:`_fd_grad_pure`), the per-method scipy dispatch logic,
and the main entry point :func:`design_optimize`.  Re-exported by
``optimize/core.py`` for bit-for-bit public-API preservation.
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# v5.3.2 (ROADMAP logging adoption sweep -- per-iteration telemetry):
# Module-level logger for design_optimize entry + per-scipy-iteration
# progress.  Default-quiet via the lumenairy root logger's NullHandler.
from .._logging import get_logger

logger = get_logger(__name__)

# Note: ``json`` (aliased to ``_json``) and ``os`` (aliased to ``_os``)
# are accessed by ``design_optimize``'s state-file helpers via
# ``lumenairy.optimize.core`` (lazy lookup at call time) so historical
# tests that monkey-patch ``core._json`` / ``core._os`` to count I/O
# calls continue to intercept the binding actually used by the
# implementation.  See ``test_v4_16_0_agent_c::test_state_save_every_throttles_io``.
from ..elements.lenses import apply_real_lens, apply_real_lens_traced

# 4.10.2: pull get_default_complex_dtype to module scope so merit-leg
# wave-propagation source fields (apply_real_lens inputs) can be
# allocated at the runtime-selected precision -- preserves the
# precision='single' choice through merit evaluations.
from .context import (
    _METHODS_SUPPORTING_CONSTRAINTS,
    Constraint,
    DesignResult,
    EvaluationContext,
    MeritTerm,
)
from .jax_merits import JaxMeritTerm
from .parameterizations import MultiPrescriptionParameterization
from .wrapper_merits import (
    MultiFieldMerit,
    MultiWavelengthMerit,
    ToleranceAwareMerit,
)

# v5.17.x (AUDIT_V5_17_0 P2-24): scipy.optimize.minimize methods that
# natively honour ``bounds=`` (scipy >= 1.7; verified against scipy
# 1.17.1 -- e.g. Powell on (x-2)^2 with bounds=[(0, 1)] converges to
# x=1.0).  Lowercase because scipy's own method dispatch is
# case-insensitive (``meth = method.lower()``).  Methods NOT listed
# here (CG / BFGS / Newton-CG / dogleg / trust-ncg / trust-exact /
# trust-krylov) merely emit an easy-to-miss RuntimeWarning inside
# scipy and IGNORE the bounds, so the driver warns loudly instead of
# forwarding them (see the generic-minimize branch below).
_MINIMIZE_METHODS_SUPPORTING_BOUNDS = frozenset({
    'nelder-mead', 'powell', 'l-bfgs-b', 'tnc', 'cobyla', 'cobyqa',
    'slsqp', 'trust-constr',
})

# =========================================================================
# Wave-propagator registry
# =========================================================================
#
# ``WAVE_PROPAGATOR_REGISTRY`` maps a string name to a callable with
# the signature
#
#     fn(E0, pres, *, wavelength, dx, N, wp_kwargs, opts) -> E_exit
#
# where ``opts`` is a dict of design_optimize-internal flags (e.g.
# ``wave_traced`` / ``ray_subsample``) so the registered function can
# honour them.  ``wp_kwargs`` is a mutable dict the function may
# ``.pop`` from to extract its own options.  Returns the post-lens
# field on the (Ny, Nx, dx) grid expected by the wave-leg merits.
#
# Users can register custom propagators with
# :func:`register_wave_propagator(name, fn)` and immediately use them
# via ``design_optimize(wave_propagator=name)``.  The default registry
# below is populated with the same five propagators that the legacy
# if/elif chain handled (real_lens, gbd, hf, hfpi, asymptotic) -- the
# refactor preserves behaviour byte-for-byte.

WAVE_PROPAGATOR_REGISTRY: Dict[str, Callable] = {}


def register_wave_propagator(name: str, fn: Callable) -> None:
    """Register a custom wave-leg propagator under ``name``.

    The function is called by :func:`design_optimize` when its
    ``wave_propagator=name``.  Signature:

    ``fn(E0, pres, *, wavelength, dx, N, wp_kwargs, opts) -> E_exit``

    where ``wp_kwargs`` is a mutable dict the function may pop from,
    and ``opts`` is a dict carrying ``design_optimize``-internal
    flags (``wave_traced``, ``ray_subsample``, ...).
    """
    WAVE_PROPAGATOR_REGISTRY[name] = fn


def unregister_wave_propagator(name: str) -> None:
    """Remove a wave propagator from the registry."""
    WAVE_PROPAGATOR_REGISTRY.pop(name, None)


def _wave_real_lens(E0, pres, *, wavelength, dx, N, wp_kwargs, opts):
    if opts.get('wave_traced', False):
        return apply_real_lens_traced(
            E0, prescription=pres, wavelength=wavelength, dx=dx,
            ray_subsample=opts.get('ray_subsample', 4), n_workers=1,
            **wp_kwargs)
    return apply_real_lens(E0, prescription=pres, wavelength=wavelength, dx=dx, **wp_kwargs)


def _wave_gbd(E0, pres, *, wavelength, dx, N, wp_kwargs, opts):
    from ..propagators.gbd import propagate_gbd_through_prescription
    return propagate_gbd_through_prescription(
        E0, dx, pres, wavelength=wavelength, **wp_kwargs)


def _wave_hf(E0, pres, *, wavelength, dx, N, wp_kwargs, opts):
    from ..propagators.hf import propagate_huygens_fresnel_through_prescription
    return propagate_huygens_fresnel_through_prescription(
        E0, dx, pres, wavelength=wavelength, **wp_kwargs)


def _wave_hfpi(E0, pres, *, wavelength, dx, N, wp_kwargs, opts):
    from ..propagators.hfpi import propagate_hfpi_through_prescription
    return propagate_hfpi_through_prescription(
        E0, dx, pres, wavelength=wavelength, **wp_kwargs)


def _wave_asymptotic(E0, pres, *, wavelength, dx, N, wp_kwargs, opts):
    from ..propagators.asymptotic import (
        fit_canonical_polynomials,
        propagate_modal_asymptotic,
    )
    # Fit the prescription (or reuse a pre-built fit if supplied).
    fit = wp_kwargs.pop('fit', None)
    fit_kw = wp_kwargs.pop('fit_kwargs', {}) or {}
    if fit is None:
        fit = fit_canonical_polynomials(
            pres, wavelength=wavelength, **fit_kw)
    # Sample the asymptotic propagator on the (Ny, Nx, dx) wave grid
    # so downstream merits see a usable field.  Centre at the chief
    # image of the on-axis source.
    # v4.12.1 (B1-10): pixel-centred `(arange(N) - N/2)*dx`, matches the
    # library-wide convention (ASM, Fresnel, RS, sources).  Merit
    # functions that compare wave-leg fields across propagator
    # families need a single shared grid convention; the previous
    # `+0.5` produced a half-pixel offset between the asymptotic leg
    # and the ASM / GBD / HF legs.
    _ax = (np.arange(N) - N / 2) * dx
    _x_grid = _ax + fit.s2x_centre
    _y_grid = _ax + fit.s2y_centre
    _X, _Y = np.meshgrid(_x_grid, _y_grid, indexing='xy')
    _wp = wp_kwargs.pop('w_p', 0.05)
    _ws = wp_kwargs.pop('w_s', 50e-6)
    return propagate_modal_asymptotic(
        fit,
        s2_grid_x=_X, s2_grid_y=_Y,
        w_s=_ws, w_p=_wp,
        v2_centre=(fit.v2x_centre, fit.v2y_centre),
        **wp_kwargs,
    )


# Populate the default registry.  Users can override these by
# re-registering, but all five preserve exact legacy behaviour.
register_wave_propagator('real_lens', _wave_real_lens)
register_wave_propagator('gbd', _wave_gbd)
register_wave_propagator('hf', _wave_hf)
register_wave_propagator('hfpi', _wave_hfpi)
register_wave_propagator('asymptotic', _wave_asymptotic)


# =========================================================================
# Finite-difference gradient helper
# =========================================================================

def _fd_bounds_arrays(
    bounds: Optional[Sequence[Optional[Tuple[float, float]]]],
    n: int,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """v5.17.x (AUDIT_V5_17_0 P3-47): normalise a driver-style bounds
    list (``[(lb, ub) or None, ...]`` with optional ``None`` endpoints)
    to a ``(lb, ub)`` pair of float64 arrays with ``+/-inf`` for the
    unbounded sides.  Returns ``None`` when ``bounds`` is ``None`` so
    callers can skip the clipping path entirely (byte-identical legacy
    behaviour)."""
    if bounds is None:
        return None
    lb = np.full(n, -np.inf, dtype=np.float64)
    ub = np.full(n, +np.inf, dtype=np.float64)
    for i, b in enumerate(bounds):
        if i >= n:
            break
        if b is None:
            continue
        if b[0] is not None:
            lb[i] = float(b[0])
        if b[1] is not None:
            ub[i] = float(b[1])
    return lb, ub


def _fd_grad_pure(
    f: Callable,
    x: np.ndarray,
    eps: float = 1e-7,
    scale_floor: Optional[np.ndarray] = None,
    f0: Optional[float] = None,
    scheme: str = 'central',
    validate_f0: bool = False,
    bounds: Optional[Sequence[Optional[Tuple[float, float]]]] = None,
) -> np.ndarray:
    """Finite-difference gradient of a scalar callable ``f(x)``.

    Pure-function helper extracted from :func:`design_optimize` so the
    FD-gradient core is testable without spinning up an optimization
    context.  Returns ``df/dx`` as an ndarray with the same shape as
    ``x``.

    Caller contract for ``f0``
    --------------------------
    When ``scheme='forward'`` and the caller supplies ``f0``, **the
    caller is responsible for the invariant ``f0 == f(x)`` at the
    same ``x`` passed in**.  A stale ``f0`` silently produces a
    wrong-direction gradient (the forward-difference quotient
    ``(f(x + h*e_i) - f0) / h`` reduces to junk plus the gradient if
    ``f0`` is off the true centre value).  This is by design -- the
    perf saving from skipping the centre evaluation is the whole
    reason ``f0`` exists.

    Set ``validate_f0=True`` to opt into a defensive ``f(x)`` re-
    evaluation that asserts the invariant; this halves the perf
    saving but catches stale-cache bugs immediately.  Off by default
    so the default path stays as cheap as the audit (P2 #11)
    expected.

    Parameters
    ----------
    f : callable
        Scalar function of the parameter vector.
    x : ndarray, shape (N,)
        Evaluation point.
    eps : float, default 1e-7
        Relative step parameter.  The actual step in coordinate ``i``
        is ``eps * max(|x[i]|, scale_floor[i])``.
    scale_floor : ndarray, shape (N,), optional
        Per-variable absolute step floor.  Defaults to 1 micron per
        variable (matches the legacy ``_fd_grad_for`` default for radii
        / thicknesses).  v5.4.6 (audit F-20): note that ``scale_floor``
        is consumed by THIS finite-difference helper, which
        ``design_optimize`` invokes only on the JAX-gradient-combined
        path (``jac='auto'`` WITH a JaxMeritTerm) and ``method='newton'``.
        On the default ``jac='auto'``-without-JAX path ``final_jac`` is
        ``None`` and scipy estimates the gradient with its own internal
        2-point step, which never sees ``scale_floor`` -- so the floor is
        inert there.  Pass an explicit ``jac`` / use the Newton path (or a
        JAX merit) to make the per-variable floor effective.
    f0 : float, optional
        Pre-computed ``f(x)``.  Only consulted when ``scheme='forward'``;
        ignored otherwise.  When supplied for the forward path, the
        central-point evaluation is skipped, saving one call to ``f``
        per gradient.  **The caller is responsible for ensuring
        ``f0 == f(x)``** (see the caller contract above); a stale
        value silently produces wrong gradients.
    scheme : {'central', 'forward'}, default 'central'
        Finite-difference scheme.  ``'central'`` evaluates ``f`` at
        ``x +/- h*e_i`` for each variable (2N evals, O(h^2) truncation
        error); this is the historical default and preserves bit-
        identical gradient values with pre-v4.13.0 behaviour.
        ``'forward'`` evaluates ``f`` at ``x`` and ``x + h*e_i`` (N+1
        evals, or N when ``f0`` is supplied) at the cost of O(h)
        truncation error.  Opt-in for perf-sensitive callers where
        the larger truncation is acceptable.
    validate_f0 : bool, default False
        Audit P2 #16 (v4.14): when ``True`` AND ``scheme='forward'``
        AND ``f0`` is supplied, re-evaluate ``f(x)`` once and raise
        ``ValueError`` if ``f0`` does not match within a tight
        tolerance.  Off by default because the validation costs one
        ``f`` call, which exactly cancels the saving from skipping
        the centre evaluation.  Useful for debugging stale-cache
        bugs in caller code.
    bounds : list of (lower, upper) tuples, optional
        v5.17.x (AUDIT_V5_17_0 P3-47): box constraints for the FD
        stencil, in the driver's bounds format (per-parameter
        2-tuples; ``None`` entries / endpoints mean unbounded).  When
        supplied, perturbed evaluation points are kept INSIDE the box
        (matching scipy's internal 2-point numdiff): a stencil leg
        that would cross a bound is clipped to it and the difference
        quotient uses the actual (shrunken) span; if a variable is
        pinned exactly at a bound, a one-sided difference pointing
        into the box is used instead.  ``None`` (default) preserves
        the historical unclipped stencil byte-for-byte.

    Returns
    -------
    g : ndarray, shape (N,)
        Finite-difference gradient.
    """
    if scheme not in ('central', 'forward'):
        raise ValueError(
            f"scheme must be 'central' or 'forward', got {scheme!r}")
    x = np.asarray(x, dtype=np.float64)
    N = x.size
    if scale_floor is None:
        scale_floor = np.full(N, 1e-6, dtype=np.float64)
    else:
        scale_floor = np.broadcast_to(
            np.asarray(scale_floor, dtype=np.float64), (N,))
    g = np.zeros(N, dtype=np.float64)
    # v5.17.x (AUDIT_V5_17_0 P3-47): normalised (lb, ub) arrays for
    # stencil clipping; None => legacy unclipped behaviour.
    lb_ub = _fd_bounds_arrays(bounds, N)
    if scheme == 'central':
        for i in range(N):
            step = eps * max(abs(x[i]), float(scale_floor[i]))
            hi = x[i] + step
            lo = x[i] - step
            clipped = False
            if lb_ub is not None:
                lb, ub = lb_ub
                if hi > ub[i]:
                    hi = ub[i]
                    clipped = True
                if lo < lb[i]:
                    lo = lb[i]
                    clipped = True
            if clipped and not (hi > lo):
                # Degenerate box (lb == ub): no room to difference.
                g[i] = 0.0
                continue
            xp_step = x.copy()
            xm_step = x.copy()
            xp_step[i] = hi
            xm_step[i] = lo
            fp = float(f(xp_step))
            fm = float(f(xm_step))
            # Use the actual (possibly shrunken) span only when a
            # clip occurred so the unclipped path stays byte-identical
            # with the historical (fp - fm) / (2 * step).
            g[i] = (fp - fm) / ((hi - lo) if clipped else (2.0 * step))
    else:  # forward
        if f0 is None:
            f0 = float(f(x))
        elif validate_f0:
            # v4.14 (audit P2 #16): opt-in stale-cache check.  Tight
            # but not exact tol to allow for benign re-evaluation
            # noise in non-deterministic merit functions.
            f0_check = float(f(x))
            tol = 1e-9 * max(abs(f0), abs(f0_check), 1.0)
            if not np.isfinite(f0_check) or abs(f0 - f0_check) > tol:
                raise ValueError(
                    f"_fd_grad_pure: validate_f0=True caught a stale "
                    f"f0 cache: f0={f0!r} but f(x) at the given x "
                    f"is {f0_check!r}.  Forward-FD with a stale f0 "
                    f"produces wrong gradients; the caller is "
                    f"responsible for the f0 == f(x) invariant.")
        for i in range(N):
            step = eps * max(abs(x[i]), float(scale_floor[i]))
            hi = x[i] + step
            if lb_ub is not None and hi > lb_ub[1][i]:
                # v5.17.x (AUDIT_V5_17_0 P3-47): the forward step
                # would cross the upper bound.  Clip it to the bound
                # (shrunken forward step) or, if x[i] is pinned
                # exactly at the bound, switch to a one-sided
                # BACKWARD difference pointing into the box -- scipy
                # _numdiff style.  (At a lower bound the forward step
                # already points into the box; nothing to do.)
                lb, ub = lb_ub
                hi = ub[i]
                if hi > x[i]:
                    xp_step = x.copy()
                    xp_step[i] = hi
                    fp = float(f(xp_step))
                    g[i] = (fp - f0) / (hi - x[i])
                else:
                    lo = max(x[i] - step, lb[i])
                    if not (lo < x[i]):
                        # Degenerate box (lb == ub): no room at all.
                        g[i] = 0.0
                        continue
                    xm_step = x.copy()
                    xm_step[i] = lo
                    fm = float(f(xm_step))
                    g[i] = (f0 - fm) / (x[i] - lo)
                continue
            xp_step = x.copy()
            xp_step[i] = hi
            fp = float(f(xp_step))
            g[i] = (fp - f0) / step
    return g


# =========================================================================
# Merit-sum helper
# =========================================================================

def _sum_merits(ctx, merit_terms):
    total = 0.0
    for m in merit_terms:
        total = total + m.evaluate(ctx)
    return total


def _clip_x0_to_bounds(x0: np.ndarray,
                       bounds: Optional[Sequence[Any]]) -> np.ndarray:
    """Clip a starting vector into the ``[lo, hi]`` bounds box.

    v5.24.x (audit S4-18): scipy's handling of an out-of-bounds ``x0``
    is method-dependent AND mostly silent -- ``TNC`` raises, ``L-BFGS-B``
    / ``Powell`` clip internally, but ``Nelder-Mead`` / ``CG`` / ``BFGS``
    happily start OUTSIDE the declared box (and the unconstrained methods
    ignore bounds entirely).  A resumed state-file ``x_best`` or a
    template whose values sit outside the user-declared bounds would
    otherwise seed the search at an infeasible point without any
    diagnostic.  Clipping once at the driver harmonises the start across
    every method.

    ``None`` bounds (no box at all) or ``None`` endpoints (the
    "unbounded on this side" idiom) leave the corresponding component
    untouched.  Returns a fresh ``float64`` array; when ``x0`` already
    lies inside the box the result is an exact copy (no numerical
    change on the feasible-start path).
    """
    x = np.array(x0, dtype=np.float64, copy=True)
    if bounds is None:
        return x
    for i, b in enumerate(bounds):
        if b is None or i >= x.size:
            continue
        lo, hi = b[0], b[1]
        if lo is not None and x[i] < lo:
            x[i] = float(lo)
        if hi is not None and x[i] > hi:
            x[i] = float(hi)
    return x


# =========================================================================
# Main entry point
# =========================================================================

def design_optimize(parameterization: Any,
                    merit_terms: Sequence[MeritTerm],
                    wavelength: float,
                    N: int = 512,
                    dx: float = 20e-6,
                    E_in: Optional[np.ndarray] = None,
                    method: str = 'L-BFGS-B',
                    max_iter: int = 100,
                    wave_traced: bool = False,
                    wave_propagator: str = 'real_lens',
                    wave_propagator_kwargs: Optional[Dict[str, Any]] = None,
                    ray_subsample: int = 4,
                    z_scan_range: Optional[Tuple[float, float]] = None,
                    z_scan_n: int = 31,
                    jac: Any = 'auto',
                    hess: Any = None,
                    precision: str = 'double',
                    plane_logger: Optional[Callable] = None,
                    verbose: bool = True,
                    progress: Optional[Callable] = None,
                    constraints: Optional[Sequence['Constraint']] = None,
                    state_file: Optional[str] = None,
                    state_save_every: int = 1,
                    seed: Optional[int] = 42) -> DesignResult:
    """Optimize a lens prescription against a set of merit terms.

    See ``lumenairy.optimize.core.design_optimize`` for the canonical
    docstring; the body lives here post-v5.1.0 split.  Parameter and
    behaviour contracts are unchanged.

    .. deprecated:: 5.30
       ``wave_traced=True`` is DEPRECATED (removal v5.32).  R-17
       (AUDIT_ADVERSARIAL_CODEBASE_2026_07_25) grep-verified ZERO
       callers anywhere in the repo -- library, tests, validation,
       examples, UI -- so the ``apply_real_lens_traced`` branch it
       selects in ``_wave_real_lens`` has never been exercised by CI.
       Register an explicit propagator that calls
       ``apply_real_lens_traced`` via
       :func:`register_wave_propagator` and select it with
       ``wave_propagator=<name>`` instead -- one dispatch mechanism
       rather than a boolean that mutates the meaning of another
       argument.  Passing a non-default value emits a
       :class:`DeprecationWarning`; the branch still runs.

    v5.24.x (audit S4-18): ``seed`` controls the RNG of the stochastic
    global methods (``differential_evolution`` / ``basin_hopping`` /
    ``dual_annealing``).  Defaults to ``42`` -- the historical hard-coded
    value -- so existing callers see byte-identical behaviour; pass a
    different int for an independent stochastic restart, or ``None`` to
    let scipy draw from the unseeded global RNG.
    """
    import scipy.optimize as so

    from ..progress import call_progress
    from ..propagators.propagation import (
        get_default_complex_dtype as _gddt,
    )
    from ..propagators.propagation import (
        set_default_complex_dtype,
    )

    # 3.5.6: ``precision`` knob.  'double' (default) preserves the
    # historical np.complex128 path; 'single' switches to np.complex64
    # for the duration of this design_optimize call (~2x FFT
    # throughput, ~2x memory headroom, ~80 dB cumulative dynamic-range
    # noise floor).  Restored to its prior value at the end via
    # try/finally.
    if precision not in ('double', 'single'):
        raise ValueError(
            f"design_optimize: precision must be 'double' or 'single', "
            f"got {precision!r}.")
    _orig_complex_dtype = _gddt()
    if precision == 'single':
        set_default_complex_dtype(np.complex64)
    # 4.10: register dtype restoration through a sentinel object so any
    # exception (scipy raise, KeyboardInterrupt, etc.) before the
    # success-path restore at the end still puts the global complex
    # dtype back.  Without this, an interrupted `precision='single'`
    # design_optimize() leaked complex64 to every subsequent call in
    # the process.
    #
    # v4.14 (audit P2 #10): the dominant restore path is now an
    # explicit ``try/finally`` around the optimization body so the
    # cleanup is deterministic under ``KeyboardInterrupt`` and any
    # exception (CPython refcount semantics in ``__del__`` are
    # implementation-defined and can be deferred under PyPy / when a
    # reference cycle survives garbage collection).  ``__del__`` is
    # retained as a defensive safety net only -- the ``_restored``
    # flag stops it firing twice on the normal path.
    class _RestoreDtype:
        def __init__(self, dtype):
            self.dtype = dtype
            self._restored = False
        def restore(self):
            """Explicitly restore the saved complex dtype.

            Idempotent: subsequent calls are no-ops.  Call from a
            ``finally:`` block to guarantee restoration on every exit
            path (normal return, exception, ``KeyboardInterrupt``).
            """
            if self._restored:
                return
            self._restored = True
            set_default_complex_dtype(self.dtype)
        def __del__(self):
            # Safety net only: the explicit ``restore()`` call in the
            # caller's ``finally:`` block is the primary path.
            # ``__del__`` runs at arbitrary points during interpreter
            # shutdown; broad-except is the standard pattern here
            # because the module-level globals may already be gone.
            if self._restored:
                return
            try:
                set_default_complex_dtype(self.dtype)
            except Exception:
                pass
    _dtype_restore_guard = _RestoreDtype(_orig_complex_dtype)

    # R-17 (AUDIT_ADVERSARIAL_CODEBASE_2026_07_25): ``wave_traced`` is a
    # documented public flag with a live branch in ``_wave_real_lens``
    # and ZERO callers anywhere (library / tests / validation / examples
    # / UI, grep-verified twice).  Deprecated rather than deleted --
    # out-of-repo callers may exist and the branch is harmless -- with
    # removal scheduled for v5.32.  The warning fires ONLY on a
    # non-default value, so every existing (default) call is silent.
    if wave_traced:
        from .._deprecation import warn_deprecated_kwarg
        warn_deprecated_kwarg(
            'wave_traced', "wave_propagator=<a propagator registered via "
                            "register_wave_propagator>",
            function='design_optimize',
            version_added='5.30', version_removed='5.32', stacklevel=3)

    # v4.14 (audit P2 #14): warn if the user selected a non-default
    # wave_propagator (e.g. 'gbd') AND any of the three Merit classes
    # that hard-code apply_real_lens for off-nominal legs is in use.
    # Threading the propagator through the sub-merit is a v4.14+
    # feature; this warning surfaces the silent inconsistency.
    if wave_propagator != 'real_lens':
        _SENSITIVE = (MultiWavelengthMerit, MultiFieldMerit, ToleranceAwareMerit)
        offenders = [type(m).__name__ for m in merit_terms
                     if isinstance(m, _SENSITIVE)]
        if offenders:
            warnings.warn(
                f"design_optimize: wave_propagator={wave_propagator!r} "
                f"selected but the following merit term(s) hard-code "
                f"apply_real_lens for off-nominal legs and will NOT use "
                f"that propagator: {offenders}.  The off-nominal "
                f"(wavelength / field / perturbation) wave-leg "
                f"penalties therefore exercise a different physical "
                f"model than the on-axis wave leg.  See audit P2 #14 "
                f"and the merit classes' docstrings; propagator "
                f"threading is a v4.14+ feature.",
                UserWarning, stacklevel=2)

    need_wave = any(m.needs_wave for m in merit_terms)
    # The ray-leg (system ABCD / EFL / BFL / Seidel) runs unless EVERY merit
    # opts out via needs_ray=False (default True preserves the historical
    # behavior).  Skipping it speeds up wave-only / rigorous-element merits and
    # lets a prescription with no sensible ABCD (a metasurface) optimize.
    need_ray = any(getattr(m, 'needs_ray', True) for m in merit_terms)
    n_params = parameterization.n_params
    x0 = parameterization.initial_values()
    bounds = parameterization.bounds

    # v4.16 (ROADMAP #9): hard-constraint validation + method-compat
    # check.  We accept zero or more :class:`Constraint` objects; each
    # is translated to scipy's :class:`NonlinearConstraint` and threaded
    # through ``scipy.optimize.minimize``.  Only SLSQP / trust-constr
    # honour hard constraints -- any other method raises here with a
    # clear hint instead of silently dropping the constraints.
    constraint_seq: Tuple[Constraint, ...] = tuple(constraints or ())
    if constraint_seq:
        for _ci, _c in enumerate(constraint_seq):
            if not isinstance(_c, Constraint):
                raise TypeError(
                    f"design_optimize: constraints[{_ci}] must be a "
                    f"Constraint instance, got {type(_c).__name__}.  "
                    f"Wrap your callable with "
                    f"`Constraint(fun=..., lb=..., ub=..., label=...)` "
                    f"from `lumenairy.optimize`.")
        if method not in _METHODS_SUPPORTING_CONSTRAINTS:
            raise ValueError(
                f"design_optimize: method={method!r} does not support "
                f"hard constraints via "
                f"scipy.optimize.NonlinearConstraint.  Use one of "
                f"{list(_METHODS_SUPPORTING_CONSTRAINTS)} "
                f"(SLSQP for small-to-medium problems with smooth "
                f"merits, trust-constr for larger / non-smooth ones), "
                f"or remove the constraints= argument and re-encode "
                f"them as soft penalties via MinThicknessMerit / "
                f"MinBackFocalLengthMerit / similar.")

    # v4.16 (ROADMAP #10): state-file checkpoint/resume.  Persist the
    # tuple ``(call_count, x_best, merit_best, history)`` to JSON so a
    # crashed multi-hour run can resume from disk on the next start.
    # When ``state_file`` points at an existing readable JSON file with
    # the right shape, we override ``x0`` with the persisted ``x_best``
    # and restore ``call_count``; otherwise we start from scratch and
    # create the file on the first eval.

    def _state_load() -> Optional[Dict[str, Any]]:
        if not state_file:
            return None
        # v5.1.0 split: look up _json / _os via lumenairy.optimize.core
        # so mock.patch-based tests that count JSON dumps via
        # ``core._json = counting_stub`` continue to work.
        from . import core as _core
        try:
            if not _core._os.path.isfile(state_file):
                return None
            with open(state_file, 'r', encoding='utf-8') as _fh:
                payload = _core._json.load(_fh)
        except (OSError, ValueError, TypeError):
            return None
        if not isinstance(payload, dict):
            return None
        if 'x_best' not in payload or 'merit_best' not in payload:
            return None
        try:
            x_best_arr = np.asarray(payload['x_best'], dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if x_best_arr.shape != (n_params,):
            # Shape mismatch -- silently start fresh; the new run will
            # overwrite the file with the correct shape after the first
            # save.
            return None
        payload['x_best'] = x_best_arr
        return payload

    _state_io_count = [0]

    def _state_save(*, force: bool = False) -> None:
        if not state_file:
            return
        _state_io_count[0] += 1
        if not force and (_state_io_count[0] % max(int(state_save_every), 1)) != 0:
            return
        payload = {
            'version': '4.16.0',
            'call_count': int(call_count[0]),
            'x_best': x_best[0].tolist() if x_best[0] is not None else None,
            'merit_best': (float(merit_best[0])
                           if np.isfinite(merit_best[0]) else None),
            'history': list(history),
        }
        # Atomic-ish replace: write to temp then rename.  Avoids
        # truncated files on a SIGKILL mid-write.  v5.1.0 split: look
        # up ``_json`` / ``_os`` via ``lumenairy.optimize.core`` so
        # tests that monkey-patch ``core._json`` to count writes
        # continue to intercept the actual call site.
        from . import core as _core
        tmp = state_file + '.tmp'
        try:
            with open(tmp, 'w', encoding='utf-8') as _fh:
                _core._json.dump(payload, _fh)
            _core._os.replace(tmp, state_file)
        except OSError:
            # I/O errors must not derail the optimisation.  The user
            # will see ``state_file`` empty / stale; the in-memory run
            # continues.  Warn once so silent persistence gaps are
            # visible (analog of plane_logger graceful-degrade).
            warnings.warn(
                f"design_optimize: failed to write state_file "
                f"{state_file!r}; continuing without checkpoint.",
                RuntimeWarning, stacklevel=2)

    # Track the best parameter vector seen so far for the state-file
    # plus best-merit-seen logic.  ``x_best`` is the actual returned
    # solution from the (potentially-resumed) optimisation.
    x_best: List[Optional[np.ndarray]] = [None]
    merit_best = [float('inf')]
    history: List[Dict[str, Any]] = []

    _resumed = _state_load()
    if _resumed is not None:
        x0 = _resumed['x_best']
        if _resumed.get('merit_best') is not None:
            merit_best[0] = float(_resumed['merit_best'])
        if _resumed.get('call_count') is not None:
            # Note: call_count is informational only; the optimisation
            # restart still does max_iter outer iterations.
            pass
        x_best[0] = np.asarray(x0, dtype=np.float64).copy()
        if isinstance(_resumed.get('history'), list):
            history.extend(_resumed['history'])

    call_count = [0]
    iter_count = [0]
    last_value = [float('inf')]
    last_efl = [0.0]
    last_frac = [0.0]  # monotonic guard: progress bar never moves backwards
    call_progress(progress, 'design_optimize', 0.0,
                  f'method={method}, {len(merit_terms)} merit term(s)')

    # v5.3.2 (ROADMAP logging adoption sweep -- per-iteration telemetry):
    # Entry log -- method + free-param count + merit-term count + iter
    # cap so an attached handler shows the design_optimize call shape
    # before the first scipy iteration fires.
    logger.info(
        "design_optimize: entry method=%s n_params=%d n_merits=%d "
        "max_iter=%d wave_propagator=%s",
        str(method), int(n_params), int(len(merit_terms)),
        int(max_iter), str(wave_propagator))

    multi_mode = isinstance(parameterization, MultiPrescriptionParameterization)

    # v5.6: a template-free parameterization (RawParameterization) builds a
    # prescription with no 'surfaces', so a ray-leg merit has nothing to
    # trace.  Catch that up front with a clear, actionable error instead of
    # silently degenerating to efl/bfl = 1e9 inside the per-iteration ray leg.
    if need_ray and not multi_mode:
        _built0 = parameterization.build(x0)
        if isinstance(_built0, dict) and 'surfaces' not in _built0:
            raise ValueError(
                "design_optimize: a ray-leg merit (needs_ray=True) needs a "
                "prescription with 'surfaces', but the parameterization built "
                "one without it (e.g. RawParameterization).  Set "
                "needs_ray=False on the merit terms for wave-only / "
                "rigorous-element (RCWA / coating) design, or use "
                "DesignParameterization with a real lens template.")

    def _emit_progress(frac: float, msg: str) -> None:
        # Clamp to [last_frac, 0.99] so the bar is monotonic.  The
        # eval-based and iter-based progress series can leapfrog each
        # other; we always take the larger value.
        frac = max(0.0, min(float(frac), 0.99))
        if frac < last_frac[0]:
            frac = last_frac[0]
        last_frac[0] = frac
        call_progress(progress, 'design_optimize', frac, msg)

    def _emit_iter_progress():
        # Fired from scipy's per-iteration callback (accurate iteration
        # counter, unlike merit_fn which fires on every FD gradient eval).
        iter_count[0] += 1
        frac = iter_count[0] / max(max_iter, 1)
        _emit_progress(
            frac,
            f'iter {iter_count[0]}: merit={last_value[0]:.4g}  '
            f'efl={last_efl[0]*1e3:.3f}mm')
        # v5.3.2 (ROADMAP logging adoption sweep -- per-iteration
        # telemetry): one INFO record per scipy iteration -- mirrors
        # the progress-callback message so an attached handler sees
        # the same merit/efl trace the GUI/CLI progress bar shows.
        logger.info(
            "design_optimize: iter %d/%d merit=%.4g efl=%.3fmm",
            int(iter_count[0]), int(max_iter),
            float(last_value[0]), float(last_efl[0] * 1e3))

    def evaluate(x):
        # Resolve names through ``lumenairy.optimize.core`` so the
        # historical ``mock.patch('lumenairy.optimize.core.X', ...)``
        # contract is preserved.  Lazy import on the first call avoids
        # the ``driver`` <-> ``core`` import cycle.
        from . import core as _core
        built = parameterization.build(x)
        if multi_mode:
            prescriptions = list(built)
            # Use the first prescription as the "primary" for backward-
            # compatible single-prescription merits (ABCD, Seidel, etc.).
            pres = prescriptions[0]
        else:
            pres = built
            prescriptions = [pres]
        ctx = EvaluationContext(
            prescription=pres, wavelength=wavelength, N=N, dx=dx,
            prescriptions=prescriptions,
            x=np.asarray(x, dtype=np.float64).copy())
        # Ray-leg (skipped if no merit needs it -- see need_ray above).
        if need_ray:
            surfs = _core.surfaces_from_prescription(pres)
            try:
                _, efl, bfl, _ = _core.system_abcd(surfs, wavelength)
            except (ValueError, RuntimeError, ZeroDivisionError, KeyError,
                    np.linalg.LinAlgError, IndexError, TypeError):
                # Degenerate / mirror-only / short prescription -- the
                # downstream sentinel filter (np.isfinite check) caps
                # these at 1e9.
                efl = bfl = float('inf')
            try:
                seidel_raw = _core.seidel_coefficients(surfs, wavelength)
                # seidel_coefficients returns (per-surface-dict, totals-dict)
                if (isinstance(seidel_raw, tuple) and len(seidel_raw) == 2
                        and isinstance(seidel_raw[0], dict)):
                    per_surf = seidel_raw[0]
                    # Sum each aberration coefficient over surfaces
                    seidel = np.array([
                        np.sum(per_surf.get(f'S{k}', np.zeros(1)))
                        for k in range(1, 6)], dtype=np.float64)
                else:
                    seidel = np.asarray(seidel_raw, dtype=np.float64)
            except (ValueError, RuntimeError, ZeroDivisionError, KeyError,
                    np.linalg.LinAlgError, IndexError, AttributeError,
                    TypeError):
                # Seidel sum unavailable -- zero-fill so the optimizer
                # sees no aberration contribution from this iteration.
                seidel = np.zeros(5)
            ctx.efl = float(efl) if np.isfinite(efl) else 1e9
            ctx.bfl = float(bfl) if np.isfinite(bfl) else 1e9
            ctx.seidel = np.asarray(seidel, dtype=np.float64).ravel()
        else:
            # No merit reads the ray-leg -- supply the same sentinels the
            # degenerate-prescription branch uses, so any incidental access is
            # well-defined.
            ctx.efl = 1e9
            ctx.bfl = 1e9
            ctx.seidel = np.zeros(5)
        # Wave leg (only if any merit term needs it)
        if need_wave:
            if E_in is None:
                # 4.11.2: honour precision='single' via
                # get_default_complex_dtype() so the wave-leg input
                # field matches the rest of the precision pipeline.
                E0 = np.ones((N, N), dtype=_core.get_default_complex_dtype())
            else:
                E0 = E_in
            wp_kwargs = dict(wave_propagator_kwargs or {})
            try:
                _wave_fn = WAVE_PROPAGATOR_REGISTRY[wave_propagator]
            except KeyError:
                raise ValueError(
                    f"design_optimize: unknown wave_propagator "
                    f"{wave_propagator!r}; expected one of "
                    f"{sorted(WAVE_PROPAGATOR_REGISTRY)} "
                    "(register custom propagators with "
                    "register_wave_propagator(name, fn)).")
            _opts = {
                'wave_traced': wave_traced,
                'ray_subsample': ray_subsample,
            }
            E_exit = _wave_fn(
                E0, pres,
                wavelength=wavelength, dx=dx, N=N,
                wp_kwargs=wp_kwargs, opts=_opts,
            )
            ctx.E_exit = E_exit
            # Through-focus scan
            if not np.isfinite(ctx.bfl) or abs(ctx.bfl) > 10:
                # 4.10: every other return path in evaluate() returns
                # (value, ctx) and every caller unpacks it that way.
                # Pre-4.10 this branch returned a bare scalar, so any
                # prescription that briefly drove BFL out of range
                # mid-optimisation raised "cannot unpack non-iterable
                # float object" and killed the run instead of being
                # penalised.
                return _sum_merits(ctx, merit_terms), ctx
            if z_scan_range is None:
                half = max(abs(ctx.bfl) / 20.0, 1e-3)
                z0, z1 = -half, +half
            else:
                z0, z1 = z_scan_range
            z_values = np.linspace(ctx.bfl + z0, ctx.bfl + z1, z_scan_n)
            ideal = _core.diffraction_limited_peak(
                E_exit, wavelength, ctx.bfl, dx)
            scan = _core.through_focus_scan(
                E_exit, dx, wavelength, z_values,
                ideal_peak=ideal, verbose=False)
            z_best_v, strehl_best_v = _core.find_best_focus(scan, 'strehl')
            ctx.z_best = float(z_best_v)
            # S4-5 (AUDIT_V5_24_2): ``find_best_focus`` returns ``nan``
            # when every through-focus slice is NaN (a fully-vignetted /
            # dark exit field).  A raw ``float(nan)`` then leaks into
            # ``StrehlMerit``, where ``max(0.0, min_strehl - nan) == 0.0``
            # REWARDS the failed design with a perfect score (the main
            # wave leg disagreed in the SIGN of its failure handling with
            # the wrapper merits, which write the 0.0 failed-scan
            # sentinel).  Coerce a non-finite best Strehl to 0.0 so a
            # degenerate wave leg is penalised, matching the wrapper.
            ctx.strehl_best = (float(strehl_best_v)
                               if np.isfinite(strehl_best_v) else 0.0)
            # v5.4.6 (audit F-5): NaN-safe argmax -- a single NaN
            # through-focus slice must not steal the argmax (np.argmax
            # treats NaN as the maximum).  Mirrors the wrapper-merit guard.
            if np.any(np.isfinite(scan.strehl)):
                i_best = int(np.nanargmax(scan.strehl))
                ctx.rms_radius_best = float(scan.rms_radius[i_best])
            # Build OPD map for Zernike fit
            ap = pres.get('aperture_diameter') or (0.4 * N * dx)
            try:
                _, _, opd_map = _core.wave_opd_2d(
                    E_exit, dx, wavelength, aperture=ap,
                    focal_length=ctx.bfl, f_ref=ctx.bfl)
                ctx.opd_map = opd_map
            except (ValueError, RuntimeError, ZeroDivisionError,
                    np.linalg.LinAlgError, IndexError, AttributeError,
                    TypeError):
                # Unwrap / least-squares fit failed in wave_opd_2d;
                # leave opd_map None so Zernike merits contribute 0.
                ctx.opd_map = None

        return _sum_merits(ctx, merit_terms), ctx

    # ------------------------------------------------------------------
    # JaxMeritTerm gradient routing.  When jac='auto' and at least one
    # JaxMeritTerm has build_args set, we combine analytic JAX
    # gradients for those terms with finite-difference partials for
    # everything else.  Pure-FD ('fd' or no JAX merits) is the legacy
    # path; user-callable jac is forwarded as-is.
    # ------------------------------------------------------------------
    jax_grad_terms = [
        m for m in merit_terms
        if isinstance(m, JaxMeritTerm) and m.supports_jax_grad()
    ]
    other_terms = [m for m in merit_terms if m not in jax_grad_terms]

    def _fd_grad_for(terms, x, eps=1e-7, f0=None, scheme='central'):
        """Finite-difference gradient of sum(t.evaluate(ctx)) for the
        selected ``terms`` only.

        4.10: step sized by parameter magnitude with a per-variable
        floor pulled from ``parameterization.scale_floor`` (or the
        magnitude itself for an absent floor).  Pre-4.10 the floor
        ``max(|x|, 1.0)`` pinned the step at 1e-7 for radii /
        thicknesses (~mm = ~0.01) AND indices (~1.5) alike, so
        relative perturbations varied by 100x across variable types
        and the Hessian estimate L-BFGS-B builds was biased.

        Parameters
        ----------
        terms : sequence of MeritTerm
        x : array-like
            Current parameter vector.
        eps : float
            Relative step parameter.
        f0 : float, optional
            Pre-computed ``sum(t.evaluate(ctx)) for t in terms`` at the
            current ``x``.  Only consulted when ``scheme='forward'``.
        scheme : {'central', 'forward'}, default 'central'
            Forwarded to :func:`_fd_grad_pure`.  Central differences
            (the default) preserve bit-identical gradient values with
            pre-v4.13.0 behaviour at 2N evaluations per gradient.
            Forward differences are an opt-in perf option (N+1 evals,
            or N with ``f0``) at the cost of O(h) truncation.
        """
        if not terms:
            return np.zeros_like(x, dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)
        # Per-variable scale floor: parameterization may expose it
        # explicitly; otherwise use a more conservative magnitude-
        # proportional floor that doesn't pin sub-mm radii / thicknesses
        # to 1e-7.
        scale_floor = getattr(parameterization, 'scale_floor', None)
        if scale_floor is None:
            scale_floor = np.full_like(x, 1e-6)  # 1 micron default for radii/thicknesses
        else:
            scale_floor = np.broadcast_to(
                np.asarray(scale_floor, dtype=np.float64), x.shape)

        def _f_terms(xv):
            _, ctx_ = evaluate(xv)
            return sum(t.evaluate(ctx_) for t in terms)

        # v5.17.x (AUDIT_V5_17_0 P3-47): forward the parameterization
        # bounds so the FD stencil never evaluates the merit outside
        # the box (scipy's own 2-point numdiff clips; the library FD
        # helpers did not, so a variable pinned at a bound produced a
        # spurious sentinel-driven gradient component).
        return _fd_grad_pure(_f_terms, x, eps=eps,
                             scale_floor=scale_floor, f0=f0,
                             scheme=scheme, bounds=bounds)

    def _merit_jac_auto(x):
        # Analytic part: sum gradient_at_x for JAX terms.
        g = np.zeros_like(np.asarray(x, dtype=np.float64))
        for t in jax_grad_terms:
            g = g + t.gradient_at_x(x)
        # Finite-difference part: gradient of the remaining terms.
        #
        # v4.14 (audit P2 #11): switch to forward-FD with a cached
        # ``f0`` for the other-terms sum.  scipy already evaluates
        # ``merit_fn(x)`` before calling ``jac`` at the same ``x`` (the
        # FULL merit, including JAX terms); we can't reuse that
        # directly because ``_fd_grad_for`` operates on the
        # ``other_terms`` subset only.  But evaluating once at ``x``
        # to capture ``f0_other`` costs one evaluate() call and then
        # saves ``N-1`` evaluations per gradient versus central FD
        # (2N -> N+1 evals; net (2N) - (1+N) = N-1 saved per gradient).
        #
        # For large N (>=10 free vars), this halves the FD cost of
        # the gradient and is the dominant runtime saving in design
        # optimisation when ANY non-JAX merit term is present.  The
        # O(h) truncation error of forward FD is well-tolerated by
        # quasi-Newton line searches (L-BFGS-B etc.) at h~1e-7.
        if other_terms:
            _, ctx_f0 = evaluate(x)
            f0_other = float(sum(t.evaluate(ctx_f0) for t in other_terms))
            g = g + _fd_grad_for(other_terms, x, f0=f0_other,
                                 scheme='forward')
        return g

    use_analytic_jac = (jac == 'auto' and len(jax_grad_terms) > 0)
    user_jac = jac if (callable(jac)) else None
    final_jac = (
        user_jac if user_jac is not None
        else (_merit_jac_auto if use_analytic_jac else None)
    )

    def _post_eval_bookkeeping(x, value, ctx):
        """OPT-3 (AUDIT_OPTIMIZE_SECOND_PASS): the shared per-eval bookkeeping
        -- plane_logger telemetry, best-merit tracking, the history rows, and
        the rolling ``_state_save()`` checkpoint.  Factored out so the
        ``method='lm'`` ``residuals`` path gets ALL of it too: pre-fix that
        path re-implemented only the eval counter + progress, so a multi-hour
        LM run with ``state_file=`` set wrote no checkpoint until the final
        force-save (a mid-run crash lost everything) and per-eval telemetry
        consumers silently received nothing."""
        if plane_logger is not None:
            try:
                plane_logger(call_count[0], ctx)
            except (TypeError, ValueError, RuntimeError, KeyError,
                    AttributeError, IndexError, OSError) as _exc:
                # logger errors must not derail optimization, but
                # warn once so silent telemetry gaps are visible.
                warnings.warn(
                    f"design_optimize: plane_logger callback failed "
                    f"({type(_exc).__name__}: {_exc}); continuing "
                    f"without telemetry for this iteration.",
                    RuntimeWarning, stacklevel=2)
        # v4.16 (ROADMAP #10): track best-merit-seen for checkpoint /
        # resume.  The optimiser is guaranteed to call merit_fn at the
        # actual converged x_opt at the end (in the final
        # evaluate() block), so x_best is monotonic-improving.
        # Gated on state_file being non-None so pre-v4.16 callers see
        # byte-identical behaviour (no per-eval bookkeeping cost).
        if state_file:
            if np.isfinite(value) and float(value) < merit_best[0]:
                merit_best[0] = float(value)
                x_best[0] = np.asarray(x, dtype=np.float64).copy()
            # Append a compact history row for the state file.  Limit
            # to the most-recent 1000 entries to bound the JSON
            # payload size on multi-hour runs.
            history.append({
                'call': int(call_count[0]),
                'merit': float(value) if np.isfinite(value) else None,
                'efl': (float(ctx.efl)
                        if np.isfinite(ctx.efl) else None),
            })
            if len(history) > 1000:
                del history[: len(history) - 1000]
        _state_save()

    def merit_fn(x):
        call_count[0] += 1
        value, ctx = evaluate(x)
        last_value[0] = float(value)
        last_efl[0] = float(ctx.efl) if np.isfinite(ctx.efl) else 0.0
        _post_eval_bookkeeping(x, value, ctx)
        # Constraint diagnostics for progress callback (v4.16 #9): show
        # the first constraint's label + current value in the eval msg
        # when constraints are active.  scipy enforces them via its own
        # machinery; this is purely diagnostic.
        _con_tag = ''
        if constraint_seq:
            try:
                _c0 = constraint_seq[0]
                _cv0 = float(_c0.fun(x))
                _label = _c0.label or 'constraint[0]'
                _con_tag = f'  [{_label}={_cv0:.4g}]'
            except (TypeError, ValueError, RuntimeError,
                    ZeroDivisionError, OverflowError):
                _con_tag = '  [constraint=err]'
        # (_state_save now runs inside _post_eval_bookkeeping above.)
        # Fallback eval-counter progress for methods without a per-
        # iteration callback hook (Powell, DE, dual_annealing, basin-
        # hopping).  For methods with a scipy callback we also emit
        # from there, which is more accurate iteration-wise; the
        # monotonic guard in _emit_progress prevents the bar from
        # going backwards when the two series leapfrog.
        frac = call_count[0] / max(max_iter * 5, 1)
        _emit_progress(
            frac,
            f'eval {call_count[0]}: merit={value:.4g}  '
            f'efl={ctx.efl*1e3:.3f}mm{_con_tag}')
        if verbose and call_count[0] % 5 == 1:
            print(f'  iter {call_count[0]}: merit = {value:.6g}  '
                  f'efl = {ctx.efl*1e3:.3f} mm  '
                  f'strehl = {ctx.strehl_best:.4f}')
        return value

    # v4.14 (audit P2 #13): honour the progress cancellation protocol
    # (``progress.should_stop``).  scipy stops the optimiser when the
    # callback returns ``True`` for L-BFGS-B / SLSQP / trust-constr /
    # Nelder-Mead etc.; differential_evolution interprets a True
    # return the same way; basin-hopping accepts True via its
    # ``callback`` arg.  See :mod:`lumenairy.progress`.
    from ..progress import is_cancelled

    def _scipy_cb_minimize(xk, *args, **kwargs):
        # Callback signature varies by method (some pass xk only,
        # trust-constr passes (xk, state), SLSQP passes xk).  Accept
        # anything.
        _emit_iter_progress()
        if is_cancelled(progress):
            return True
        return None

    def _scipy_cb_de(xk, convergence):
        _emit_iter_progress()
        if is_cancelled(progress):
            return True
        return None

    def _scipy_cb_basin(xk, f, accept):
        last_value[0] = float(f)
        _emit_iter_progress()
        if is_cancelled(progress):
            return True
        return None

    # v4.13.2 (P1-NEW-L): dual_annealing's callback was an inline
    # lambda that did NOT poll ``is_cancelled(progress)`` -- a Qt
    # ``Stop`` press during a dual_annealing run was silently
    # ignored.  Promote to a named callback matching the pattern of
    # the other three scipy callbacks; returning True asks
    # dual_annealing to terminate the run.
    def _scipy_cb_da(x, f, context):
        last_value[0] = float(f)
        _emit_iter_progress()
        if is_cancelled(progress):
            return True
        return None

    # v5.24.x (audit S4-18): clip the starting vector into the bounds
    # box before dispatch.  Harmonises the (method-dependent, mostly
    # silent) scipy handling of an out-of-bounds ``x0`` -- e.g. a
    # resumed state-file ``x_best`` or a template outside its declared
    # bounds -- so every method starts feasible.  No-op (exact copy)
    # when ``x0`` already satisfies the bounds.
    x0 = _clip_x0_to_bounds(np.asarray(x0, dtype=np.float64), bounds)

    # v4.14 (audit P2 #10): wrap the dispatch + final evaluation in
    # try/finally so the complex-dtype restore is deterministic even
    # under KeyboardInterrupt / scipy raise.  The ``_dtype_restore_guard``
    # ``__del__`` remains as a defensive safety net.
    t0 = time.time()
    try:
        if method == 'lm':
            # Gauss-Newton / Levenberg-Marquardt via least_squares.  No
            # per-iteration callback is available, so emit progress from
            # inside residuals() using the eval counter.
            # S4-5 (AUDIT_V5_24_2): one-shot flag so the negative-term
            # warning below fires at most once per design_optimize run.
            _lm_neg_warned = [False]

            def residuals(x):
                call_count[0] += 1
                value, ctx = evaluate(x)
                last_value[0] = float(value)
                last_efl[0] = float(ctx.efl) if np.isfinite(ctx.efl) else 0.0
                # OPT-3: same plane_logger / best-merit / history / checkpoint
                # bookkeeping every other method's merit_fn gets.
                _post_eval_bookkeeping(x, value, ctx)
                frac = call_count[0] / max(max_iter * 5, 1)
                _emit_progress(
                    frac,
                    f'eval {call_count[0]}: merit={value:.4g}  '
                    f'efl={ctx.efl*1e3:.3f}mm')
                # 4.10.2: LM residual = sqrt(m.evaluate(ctx)) is non-
                # differentiable at zero (sqrt'(0) = inf), which produces
                # inf/nan columns in the FD Jacobian near a converged
                # design.  Soft-floor with a tiny epsilon so the residual
                # is differentiable everywhere; the floor is well below
                # typical merit-term magnitudes so it doesn't affect the
                # converged solution.
                _LM_FLOOR = 1e-30
                # S4-5 (AUDIT_V5_24_2): the ``max(m, 0.0)`` clamp below
                # silently zeroes any NEGATIVE merit-term contribution --
                # so a reward-style ``CallableMerit`` that legitimately
                # returns a negative value is invisible to Levenberg-
                # Marquardt (it optimises a different objective than every
                # non-'lm' method, which sums the raw values).  Evaluate
                # each term ONCE, warn (at most once) when any term is
                # negative, then build the clamped residual from the same
                # values -- no extra merit evaluations, numerics unchanged.
                term_vals = [m.evaluate(ctx) for m in merit_terms]
                if not _lm_neg_warned[0] and any(v < 0.0 for v in term_vals):
                    _lm_neg_warned[0] = True
                    warnings.warn(
                        "design_optimize(method='lm'): a merit term "
                        "returned a NEGATIVE contribution, which the "
                        "least-squares residual sqrt(max(m, 0)) silently "
                        "clamps to 0 -- reward-style (negative) terms are "
                        "ignored under 'lm' only.  Use a non-'lm' method "
                        "(e.g. 'L-BFGS-B', 'Nelder-Mead') for reward-style "
                        "merits, or reformulate the term as a "
                        "non-negative penalty.",
                        RuntimeWarning, stacklevel=2)
                return np.array(
                    [np.sqrt(max(v, 0.0) + _LM_FLOOR) for v in term_vals],
                    dtype=np.float64)
            # v4.16.1 (AUDIT_V4_16_0_DEEP P1-DEEP-1-2): explicit
            # ``None``-aware unpacking.  Pre-v4.16.1 used
            # ``b[0] if b else -np.inf``, where ``b`` is a 2-tuple
            # ``(lb_i, ub_i)``; a non-empty tuple is ALWAYS truthy in
            # Python, so the conditional never fired even when either
            # endpoint was ``None`` (the "no bound on this side"
            # idiom).  ``None`` then leaked into ``np.array(...)``,
            # producing an object-dtype array that scipy's
            # ``least_squares`` rejects with an opaque downstream
            # error.  scipy's ``bounds=(lb, ub)`` spec requires each
            # endpoint to be either a finite float or +/-inf, not
            # ``None``.  Explicit per-endpoint check below honours the
            # idiom and produces a clean float64 array.
            # v4.16.2 (audit P3-NEW-F1-3): length guard.  Pre-v4.16.2
            # the helper silently picked ``b[0]`` / ``b[1]`` from
            # ANY indexable -- so a 3-tuple ``(lb, ub, extra)`` (a
            # genuine user mistake, e.g. mixing up scipy bounds /
            # least_squares bounds / DE bounds formats) would parse
            # cleanly with the 3rd element dropped.  Raise instead
            # so the user can fix the call shape.
            def _resolve_bound(b, i, default):
                if b is None:
                    return default
                if len(b) != 2:
                    raise ValueError(
                        f"design_optimize(method='lm'): each bound "
                        f"must be a 2-tuple (lower, upper), got "
                        f"{len(b)}-tuple {b!r}.  scipy's "
                        f"``least_squares(bounds=(lb, ub))`` API "
                        f"takes a 2-tuple of array-likes; only the "
                        f"first two endpoints are meaningful.")
                v = b[i]
                if v is None:
                    return default
                return float(v)
            _bounds_iter = bounds if bounds is not None else [None] * n_params
            lb = np.array([_resolve_bound(b, 0, -np.inf) for b in _bounds_iter],
                          dtype=np.float64)
            ub = np.array([_resolve_bound(b, 1, +np.inf) for b in _bounds_iter],
                          dtype=np.float64)
            # v4.16.2 (audit P3-NEW-F1-8): scipy's least_squares
            # contract is that ``method='lm'`` does NOT accept
            # bounds; passing both forces a silent switch to
            # ``'trf'``.  Pre-v4.16.2 the override was invisible to
            # the user: test names of the form
            # ``test_bug4_lm_bounds_*`` documented "lm" while the
            # production path actually ran "trf".  Warn at the
            # override point so the user knows.
            if bounds is not None:
                warnings.warn(
                    "design_optimize(method='lm', bounds=...): "
                    "scipy's least_squares does not accept bounds "
                    "under method='lm'.  Silently overriding to "
                    "method='trf' (Trust Region Reflective), which "
                    "supports box-constrained least squares.  Pass "
                    "method='trf' explicitly to silence this "
                    "warning, or drop the ``bounds`` argument to "
                    "use the actual Levenberg-Marquardt algorithm.",
                    UserWarning, stacklevel=2,
                )
            res = so.least_squares(
                residuals, x0, method='lm' if not (bounds is not None) else 'trf',
                bounds=(lb, ub) if bounds is not None else (-np.inf, np.inf),
                max_nfev=max_iter, verbose=1 if verbose else 0)
            x_opt = res.x
        elif method in ('differential_evolution', 'de', 'global'):
            # Differential evolution: stochastic global optimizer.
            # Requires bounds for all variables.
            if bounds is None:
                raise ValueError(
                    'differential_evolution requires bounds for all variables.')
            # S4-18: ``seed`` is user-controllable (default 42 preserves
            # the historical hard-coded value).  scipy's DE has no
            # gradient hook -- its optional ``polish`` step runs L-BFGS-B
            # with a NUMERICAL jacobian regardless -- so an analytic
            # ``final_jac`` cannot be threaded here.
            res = so.differential_evolution(
                merit_fn, bounds, maxiter=max_iter, seed=seed,
                tol=1e-8, disp=verbose, polish=True,
                callback=_scipy_cb_de)
            x_opt = res.x
        elif method == 'basin_hopping':
            # Basin-hopping: global optimizer with local minimisation steps.
            minimizer_kwargs = {
                'method': 'L-BFGS-B',
                'bounds': bounds,
                'options': {'maxiter': 50},
            }
            # S4-18: forward the analytic jacobian to the L-BFGS-B local
            # search when one is available (a JaxMeritTerm gradient or a
            # user-supplied ``jac`` callable).  Pre-fix the local search
            # always finite-differenced even when an exact gradient was
            # in hand.
            if final_jac is not None:
                minimizer_kwargs['jac'] = final_jac
            res = so.basinhopping(
                merit_fn, x0, niter=max_iter,
                minimizer_kwargs=minimizer_kwargs, seed=seed,
                disp=verbose, callback=_scipy_cb_basin)
            x_opt = res.x
        elif method == 'dual_annealing':
            if bounds is None:
                raise ValueError(
                    'dual_annealing requires bounds for all variables.')
            # S4-18: ``seed`` user-controllable; forward the analytic
            # jacobian to the L-BFGS-B local polish via
            # ``minimizer_kwargs`` when available (scipy >= 1.8 spelling).
            if final_jac is not None:
                try:
                    res = so.dual_annealing(
                        merit_fn, bounds, maxiter=max_iter, seed=seed,
                        callback=_scipy_cb_da,
                        minimizer_kwargs={'jac': final_jac})
                except TypeError:
                    # Older scipy spelled the local-search override
                    # differently (``local_search_options``); fall back
                    # to the no-jac call rather than fail.
                    res = so.dual_annealing(
                        merit_fn, bounds, maxiter=max_iter, seed=seed,
                        callback=_scipy_cb_da)
            else:
                res = so.dual_annealing(
                    merit_fn, bounds, maxiter=max_iter, seed=seed,
                    callback=_scipy_cb_da)
            x_opt = res.x
        elif method == 'newton':
            # v4.16 (ROADMAP #12): Hessian / Newton-step.  For small
            # (<30 free var) problems an FD-Hessian-based Newton step
            # converges in fewer outer evaluations than L-BFGS-B.  We
            # dispatch to scipy's 'trust-ncg' / 'Newton-CG' with an
            # FD-Jacobian-of-the-FD-gradient Hessian estimator.
            #
            # 'trust-ncg' demands an explicit Hessian callable; we
            # build one via _fd_grad_pure of the merit gradient.
            # 'Newton-CG' tolerates None (uses BFGS update internally),
            # which is the safer choice for larger problems.
            #
            # For >30 free vars, the FD-Hessian (O(N^2) merit-fn calls
            # per Newton step) becomes prohibitive; we warn but allow.
            if n_params > 30:
                warnings.warn(
                    f"design_optimize: method='newton' selected with "
                    f"n_params={n_params} > 30 free variables.  The "
                    f"FD-Hessian costs O(N^2) merit evaluations per "
                    f"Newton step, which scales poorly above ~30 "
                    f"variables.  Consider method='L-BFGS-B' "
                    f"(the default) for larger problems; it uses an "
                    f"L-BFGS Hessian approximation that costs O(N) "
                    f"per iteration.",
                    UserWarning, stacklevel=2)

            # Build merit-gradient and Hessian closures.
            def _grad_for_newton(xv):
                # Use the existing analytic-when-possible jac path
                # when available; otherwise FD the full merit.
                if final_jac is not None:
                    return np.asarray(final_jac(xv), dtype=np.float64)
                # v5.17.x (AUDIT_V5_17_0 P3-47): clip the FD stencil
                # to the parameterization bounds (trust-ncg itself
                # receives no bounds, but an x0 placed on a physical
                # wall must not push the merit outside the box).
                return _fd_grad_pure(
                    merit_fn,
                    np.asarray(xv, dtype=np.float64),
                    eps=1e-6, scheme='central', bounds=bounds)

            def _hess_for_newton(xv):
                # FD-Hessian = FD-Jacobian-of-FD-gradient.  Central
                # difference on each component of the gradient.  Costs
                # O(N) gradient evaluations; each gradient itself costs
                # O(N) merit-fn evals for the FD path, so the Hessian
                # is O(N^2) merit calls.  Justified for small N where
                # the faster Newton convergence rate amortises it.
                x_arr = np.asarray(xv, dtype=np.float64)
                Nv = x_arr.size
                H = np.zeros((Nv, Nv), dtype=np.float64)
                # Use a slightly coarser step on the outer FD to keep
                # the truncation error manageable -- the inner gradient
                # is already O(h) at h=1e-6, so the outer step should
                # be ~1e-4 for a balanced O(h)+O(h^2) trade-off.
                eps_outer = 1e-4
                # v5.17.x (AUDIT_V5_17_0 P3-47): clip the outer
                # central stencil to the parameterization bounds so a
                # variable sitting on a physical wall never drives a
                # merit evaluation outside the box.  Unclipped legs
                # keep the historical (gp - gm) / (2 * step) quotient
                # byte-for-byte.
                _lb_ub = _fd_bounds_arrays(bounds, Nv)
                for i in range(Nv):
                    step = eps_outer * max(abs(x_arr[i]), 1e-6)
                    hi = x_arr[i] + step
                    lo = x_arr[i] - step
                    clipped = False
                    if _lb_ub is not None:
                        if hi > _lb_ub[1][i]:
                            hi = _lb_ub[1][i]
                            clipped = True
                        if lo < _lb_ub[0][i]:
                            lo = _lb_ub[0][i]
                            clipped = True
                    if clipped and not (hi > lo):
                        # Degenerate box (lb == ub): leave column 0.
                        continue
                    xp_step = x_arr.copy()
                    xp_step[i] = hi
                    xm_step = x_arr.copy()
                    xm_step[i] = lo
                    gp = _grad_for_newton(xp_step)
                    gm = _grad_for_newton(xm_step)
                    H[:, i] = (gp - gm) / ((hi - lo) if clipped
                                           else (2.0 * step))
                # Symmetrise to suppress FD-asymmetry noise; the true
                # Hessian is symmetric.
                H = 0.5 * (H + H.T)
                return H

            _scipy_method = 'trust-ncg'
            _hess_fn = _hess_for_newton if (hess is None or hess == 'fd') else hess
            res = so.minimize(
                merit_fn, x0, method=_scipy_method,
                jac=_grad_for_newton,
                hess=_hess_fn,
                options={'maxiter': max_iter},
                callback=_scipy_cb_minimize)
            x_opt = res.x
        else:
            # v4.16 (ROADMAP #9): thread hard constraints through
            # scipy.optimize.minimize.  Method compatibility was
            # validated up-front; here we just translate each
            # :class:`Constraint` to a scipy NonlinearConstraint.
            _scipy_constraints = [c.to_scipy() for c in constraint_seq]
            # v5.17.x (AUDIT_V5_17_0 P2-24): forward ``bounds`` for
            # EVERY minimize method that honours them.  Pre-fix the
            # whitelist was ('L-BFGS-B', 'SLSQP', 'trust-constr'), so
            # user-supplied bounds were SILENTLY dropped for Powell /
            # Nelder-Mead / TNC / COBYLA / COBYQA -- methods scipy
            # box-constrains natively -- and the optimizer freely
            # walked outside the user's stated box (probe: bounded
            # Powell converged to x=2.0 with bounds [(0, 1)]).  For
            # methods that truly cannot handle bounds we keep passing
            # None but warn loudly (parity with the method='lm'
            # bounds warning above) instead of dropping silently.
            _supports_bounds = (
                isinstance(method, str)
                and method.lower() in _MINIMIZE_METHODS_SUPPORTING_BOUNDS)
            if bounds is not None and not _supports_bounds:
                warnings.warn(
                    f"design_optimize(method={method!r}, bounds=...): "
                    f"this scipy.optimize.minimize method cannot "
                    f"handle bounds, so the supplied bounds are being "
                    f"DROPPED and the optimizer may evaluate / "
                    f"converge outside them.  Use a bounds-capable "
                    f"method (e.g. 'L-BFGS-B', 'Powell', "
                    f"'Nelder-Mead', 'TNC', 'SLSQP', 'trust-constr') "
                    f"or remove the parameterization's bounds to "
                    f"silence this warning.",
                    UserWarning, stacklevel=2,
                )
            # v5.17.x (AUDIT_V5_17_0 wave-5 follow-up): TNC does not
            # take a 'maxiter' option -- its evaluation budget is
            # 'maxfun'.  Pre-fix scipy warned 'Unknown solver
            # options: maxiter' and ran with its DEFAULT budget, so
            # ``max_iter`` was silently ineffective for TNC.  Map the
            # option name per-method.
            #
            # v5.18.0: 'disp' is NO LONGER passed as a solver option.
            # scipy 1.18.0 tightened per-method option validation and now
            # rejects 'disp' for L-BFGS-B (and likely other methods),
            # emitting ``OptimizeWarning: Unknown solver options: disp``
            # under every generic-minimize call (scipy <= 1.17 accepted
            # it).  The driver already prints its own iteration progress
            # from the merit callback when ``verbose`` is set, so scipy's
            # internal ``disp`` was redundant; dropping it keeps the
            # generic path option-clean on scipy 1.17 AND 1.18.
            if isinstance(method, str) and method.lower() == 'tnc':
                _options: Dict[str, Any] = {'maxfun': max_iter}
            else:
                _options = {'maxiter': max_iter}
            _minimize_kwargs: Dict[str, Any] = {
                'jac': final_jac,
                'bounds': bounds if _supports_bounds else None,
                'options': _options,
                'callback': _scipy_cb_minimize,
            }
            if _scipy_constraints:
                _minimize_kwargs['constraints'] = _scipy_constraints
            if hess is not None and method in (
                    'Newton-CG', 'trust-ncg', 'trust-exact',
                    'trust-krylov'):
                _minimize_kwargs['hess'] = hess
            res = so.minimize(
                merit_fn, x0, method=method, **_minimize_kwargs)
            x_opt = res.x

        # Final evaluation for the returned context
        final_value, final_ctx = evaluate(x_opt)
        # v4.16 (ROADMAP #10): also force-save the final state so the
        # checkpoint file reflects the converged solution, regardless
        # of where state_save_every left the rolling-save counter.
        # Gated on state_file so pre-v4.16 callers see byte-identical
        # behaviour.
        if state_file:
            if np.isfinite(final_value) and float(final_value) < merit_best[0]:
                merit_best[0] = float(final_value)
                x_best[0] = np.asarray(x_opt, dtype=np.float64).copy()
            _state_save(force=True)
        dt = time.time() - t0
        iter_tag = (f'{iter_count[0]} iters, '
                    if iter_count[0] > 0 else '')
        call_progress(progress, 'design_optimize', 1.0,
                      f'converged: merit={final_value:.4g} '
                      f'({iter_tag}{call_count[0]} evals, {dt:.1f}s)')
    finally:
        # v4.14 (audit P2 #10): explicit deterministic restore.  Runs on
        # normal return AND on every exception path (scipy raise, user
        # KeyboardInterrupt, MemoryError from a huge FFT, etc.).
        _dtype_restore_guard.restore()

    return DesignResult(
        x=x_opt,
        prescription=final_ctx.prescription,
        merit=float(final_value),
        # OPT-nit (AUDIT_OPTIMIZE_SECOND_PASS): default converged to FALSE
        # when the scipy result lacks ``success`` (e.g. basin-hopping) --
        # reporting convergence we can't confirm is optimistic.
        converged=getattr(res, 'success', False),
        # OPT-nit: report the true ITERATION count (scipy's ``nit``, else our
        # per-iteration counter) -- ``call_count`` is the merit-EVAL count, a
        # mislabel for this field.
        iterations=int(getattr(res, 'nit', iter_count[0])),
        time_sec=dt,
        context_final=final_ctx,
        scipy_result=res,
        prescriptions=(final_ctx.prescriptions if multi_mode else None))
