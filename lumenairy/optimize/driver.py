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

def _fd_grad_pure(
    f: Callable,
    x: np.ndarray,
    eps: float = 1e-7,
    scale_floor: Optional[np.ndarray] = None,
    f0: Optional[float] = None,
    scheme: str = 'central',
    validate_f0: bool = False,
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
    if scheme == 'central':
        for i in range(N):
            step = eps * max(abs(x[i]), float(scale_floor[i]))
            xp_step = x.copy()
            xm_step = x.copy()
            xp_step[i] = x[i] + step
            xm_step[i] = x[i] - step
            fp = float(f(xp_step))
            fm = float(f(xm_step))
            g[i] = (fp - fm) / (2.0 * step)
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
            xp_step = x.copy()
            xp_step[i] = x[i] + step
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
                    state_save_every: int = 1) -> DesignResult:
    """Optimize a lens prescription against a set of merit terms.

    See ``lumenairy.optimize.core.design_optimize`` for the canonical
    docstring; the body lives here post-v5.1.0 split.  Parameter and
    behaviour contracts are unchanged.
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
        # Ray-leg (always)
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
            ctx.strehl_best = float(strehl_best_v)
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

        return _fd_grad_pure(_f_terms, x, eps=eps,
                             scale_floor=scale_floor, f0=f0,
                             scheme=scheme)

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

    def merit_fn(x):
        call_count[0] += 1
        value, ctx = evaluate(x)
        last_value[0] = float(value)
        last_efl[0] = float(ctx.efl) if np.isfinite(ctx.efl) else 0.0
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
        _state_save()
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
            def residuals(x):
                call_count[0] += 1
                value, ctx = evaluate(x)
                last_value[0] = float(value)
                last_efl[0] = float(ctx.efl) if np.isfinite(ctx.efl) else 0.0
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
                return np.array(
                    [np.sqrt(max(m.evaluate(ctx), 0.0) + _LM_FLOOR)
                     for m in merit_terms],
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
            res = so.differential_evolution(
                merit_fn, bounds, maxiter=max_iter, seed=42,
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
            res = so.basinhopping(
                merit_fn, x0, niter=max_iter,
                minimizer_kwargs=minimizer_kwargs, seed=42,
                disp=verbose, callback=_scipy_cb_basin)
            x_opt = res.x
        elif method == 'dual_annealing':
            if bounds is None:
                raise ValueError(
                    'dual_annealing requires bounds for all variables.')
            res = so.dual_annealing(
                merit_fn, bounds, maxiter=max_iter, seed=42,
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
                return _fd_grad_pure(
                    merit_fn,
                    np.asarray(xv, dtype=np.float64),
                    eps=1e-6, scheme='central')

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
                for i in range(Nv):
                    step = eps_outer * max(abs(x_arr[i]), 1e-6)
                    xp_step = x_arr.copy()
                    xp_step[i] = x_arr[i] + step
                    xm_step = x_arr.copy()
                    xm_step[i] = x_arr[i] - step
                    gp = _grad_for_newton(xp_step)
                    gm = _grad_for_newton(xm_step)
                    H[:, i] = (gp - gm) / (2.0 * step)
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
                options={'maxiter': max_iter, 'disp': verbose},
                callback=_scipy_cb_minimize)
            x_opt = res.x
        else:
            # v4.16 (ROADMAP #9): thread hard constraints through
            # scipy.optimize.minimize.  Method compatibility was
            # validated up-front; here we just translate each
            # :class:`Constraint` to a scipy NonlinearConstraint.
            _scipy_constraints = [c.to_scipy() for c in constraint_seq]
            _minimize_kwargs: Dict[str, Any] = {
                'jac': final_jac,
                'bounds': bounds if method in ('L-BFGS-B', 'SLSQP',
                                                'trust-constr') else None,
                'options': {'maxiter': max_iter, 'disp': verbose},
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
        converged=getattr(res, 'success', True),
        iterations=call_count[0],
        time_sec=dt,
        context_final=final_ctx,
        scipy_result=res,
        prescriptions=(final_ctx.prescriptions if multi_mode else None))
