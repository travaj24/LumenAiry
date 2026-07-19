"""
lumenairy.optimize.context -- evaluation context, sentinels, constraint.

v5.1.0 split (Agent E): this module hosts the data containers + sentinel
plumbing that the merit-term hierarchy and ``design_optimize`` depend
on.  Pre-v5.1.0 everything lived in ``lumenairy/optimize/core.py``; the
split is mechanical (no public API change) and ``optimize/core.py`` now
re-exports every previously-public name from this module.

Contents
--------
* :class:`MeritTerm` -- base class for every merit term.
* :func:`ctx_is_valid` -- ABCD-sentinel-aware validity check.
* :class:`EvaluationContext` -- ray/wave bundle passed to ``MeritTerm.evaluate``.
* :class:`DesignResult` -- ``design_optimize`` return value.
* :class:`Constraint` -- v4.16 hard nonlinear constraint.
* Sentinel classes / singletons:
    - :class:`_ZeroApertureMaskSentinel` / ``_ZERO_APERTURE_MASK``
    - :class:`_InvalidFocalLengthSentinel` / ``_INVALID_FL_SENTINEL_OBJ``
    - :class:`_FailedScanStrehlSentinel` / ``_FAILED_SCAN_STREHL_SENTINEL_OBJ``
* ``_INVALID_FL_SENTINEL = 1e9`` -- scalar magnitude flag used by ``ctx_is_valid``.

Sentinel pickle-safety contract
-------------------------------
The three ``_Sentinel`` subclasses register themselves in
``lumenairy._deprecation._SENTINEL_REGISTRY`` by their canonical
singleton name (``'_ZERO_APERTURE_MASK'`` / ``'_INVALID_FL_SENTINEL_OBJ'``
/ ``'_FAILED_SCAN_STREHL_SENTINEL_OBJ'``).  Pickle round-trip preserves
``is``-identity via the name-keyed lookup, independent of which module
declared the class.  Moving the class definitions from ``core.py`` to
this submodule changes ``__module__`` / ``__qualname__`` but does NOT
break pickling because the registry is keyed by NAME, not by module
path.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .._deprecation import _Sentinel as _Sentinel

# =========================================================================
# Shared Zernike-RMS quadrature (v5.24.x, audit S4-18 dedup)
# =========================================================================


def zernike_higher_order_rms_waves(coeffs, exclude_low_order, wavelength):
    """RMS (in waves) of the Zernike coefficients above the first
    ``exclude_low_order`` modes.

    Extracted (v5.24.x, audit S4-18) as the single implementation of the
    higher-order-RMS quadrature that was copy/pasted byte-for-byte at
    three sites: :meth:`EvaluationContext.rms_wavefront_waves` and the
    two OPD-matching merits (:class:`MatchIdealThinLensMerit`,
    :class:`MatchTargetOPDMerit`) in ``merit_terms.py``.  The formula is
    unchanged -- ``sqrt(sum(coeffs[k:] ** 2)) / wavelength`` -- so every
    former call site's numerical output is preserved to machine
    precision.

    Parameters
    ----------
    coeffs : ndarray
        Zernike coefficient vector (metres), OSA/ANSI-ordered, as
        returned by :func:`lumenairy.analysis.zernike_decompose`.
    exclude_low_order : int
        Number of leading modes to drop before taking the quadrature
        (e.g. piston / tilt / defocus).
    wavelength : float
        Vacuum wavelength [m]; the RMS is divided by it to convert
        metres -> waves.

    Returns
    -------
    float
        Higher-order wavefront RMS in waves.
    """
    higher = coeffs[exclude_low_order:]
    rms_m = float(np.sqrt(np.sum(higher ** 2)))
    return rms_m / wavelength


# =========================================================================
# Sentinels
# =========================================================================

# Sentinel used by EvaluationContext when ABCD extraction failed.  Merit
# terms that consume ``ctx.efl`` / ``ctx.bfl`` should route through
# :func:`ctx_is_valid` rather than blindly plugging the sentinel into
# their formulas (a naive ``(1e9 - target)^2`` is astronomical and drags
# the optimizer away from good regions).
_INVALID_FL_SENTINEL = 1e9


# v4.14.1 (P1-NEW-1): sentinel meaning "aperture explicitly zero, block
# all light."  Distinguished from ``mask is None`` ("no aperture
# specified, use full grid").  Pre-v4.14.0 a scalar
# ``aperture_diameter=0`` produced an all-False boolean mask, which
# downstream apply_real_lens treated as "block all light"; v4.14.0
# collapsed that branch into ``mask=None``, flipping the semantics so
# ``aperture_diameter=0`` instead produced a grid-filling plane wave.
# Callers compare ``mask is _ZERO_APERTURE_MASK`` to detect the
# deliberate-zero case and zero their field accordingly.
#
# v4.15.1 (Agent E): now inherits from ``_deprecation._Sentinel`` to
# share the singleton-name registry + pickle-safe ``__reduce__``
# protocol.  Pre-v4.15.1 this class duplicated the singleton plumbing
# in 3 places (here, ``_AngleUnsetSentinel`` in ``polarization.py``,
# and ``_Sentinel`` in ``_deprecation.py``); none carried a
# ``__reduce__``, so pickling a sentinel produced a NEW instance on
# the receiving side and broke ``is``-identity checks in distributed
# merit evaluation / joblib caches.
class _ZeroApertureMaskSentinel(_Sentinel):
    """Singleton sentinel for aperture explicitly zero / blocked."""

    __slots__ = ()

    def __init__(self) -> None:
        # Use the existing repr-friendly name as the singleton key.
        super().__init__('_ZERO_APERTURE_MASK')


_ZERO_APERTURE_MASK = _ZeroApertureMaskSentinel()


# v4.15.2 (Agent E, AUDIT_V4_15_1 P2): three additional pre-existing
# sentinel patterns in this module are promoted to ``_Sentinel``
# subclasses for pickle-safety + ``is``-identity discoverability.  Pre-
# v4.15.2 these were bare scalar fallbacks (``1e9`` for invalid focal
# length, ``0.0`` for failed-scan Strehl, and a "fall-back-to-nominal"
# marker for perturbed-ABCD failures).  Scalar storage is preserved at
# the call sites for arithmetic compatibility; the dedicated sentinel
# classes here register in ``_SENTINEL_REGISTRY`` so downstream consumers
# can perform identity checks (``ctx.efl is _INVALID_FL_SENTINEL_OBJ``)
# without breaking the existing magnitude-based ``ctx_is_valid`` path.
# Each carries a ``.value`` attribute holding its canonical scalar
# fallback so call sites that want the numeric form can ``float(s)`` or
# ``s.value``.  All three inherit ``__bool__ -> False`` from the base
# ``_Sentinel`` (matching ``_ZeroApertureMaskSentinel`` semantics).
#
# Naming convention: ``_<Concept>Sentinel`` for the class +
# ``_<CONCEPT>_SENTINEL_OBJ`` for the singleton.  ``_OBJ`` suffix
# distinguishes the new class-instance singletons from the pre-existing
# ``_INVALID_FL_SENTINEL = 1e9`` scalar at module top (kept for
# arithmetic uses in ``ctx_is_valid``).


class _InvalidFocalLengthSentinel(_Sentinel):
    """Identity-checkable singleton for "ABCD extraction failed -- focal
    length collapsed to the ``1e9`` magnitude-flag fallback".

    Used at the wave-leg ABCD failure branch.  Pre-v4.15.2 that branch
    wrote a bare scalar ``efl = bfl = 1e9``; the magnitude-check
    downstream (``ctx_is_valid``) recovered the "invalid" semantics by
    comparing ``abs(v) >= _INVALID_FL_SENTINEL * 0.5``.  v4.15.2 keeps
    the scalar write (arithmetic stability) and adds this singleton so
    a future caller wanting a strict identity check
    (``ctx.efl is _INVALID_FL_SENTINEL_OBJ``) can opt in without
    breaking the existing magnitude path.
    """
    __slots__ = ()

    value: float = 1e9

    def __init__(self) -> None:
        super().__init__('_INVALID_FL_SENTINEL_OBJ')

    def __float__(self) -> float:
        return float(self.value)


_INVALID_FL_SENTINEL_OBJ = _InvalidFocalLengthSentinel()


class _FailedScanStrehlSentinel(_Sentinel):
    """Identity-checkable singleton for "through-focus Strehl scan
    failed -- Strehl collapsed to the safe ``0.0`` fallback".

    Used at the through-focus-scan exception branches.  Pre-v4.15.2 the
    branch wrote ``sub_ctx.strehl_best = 0.0``; the optimizer treats
    ``0.0`` as "very bad design" so the merit-leg contribution sinks
    into the noise floor without dragging the parameter vector further
    than the dispatcher's adaptive-step safeguards allow.  v4.15.2
    keeps the scalar write and adds this singleton for identity
    discoverability.
    """
    __slots__ = ()

    value: float = 0.0

    def __init__(self) -> None:
        super().__init__('_FAILED_SCAN_STREHL_SENTINEL_OBJ')

    def __float__(self) -> float:
        return float(self.value)


_FAILED_SCAN_STREHL_SENTINEL_OBJ = _FailedScanStrehlSentinel()


# v4.15.4 (audit AUDIT_V4_15_3 P2-NEW-F1-B option a): the previously
# defined ``_PerturbedABCDFallbackSentinel`` class and its singleton
# ``_PERTURBED_ABCD_FALLBACK_SENTINEL_OBJ`` were dead code -- never
# wired at the intended callsite (the tolerance-perturbation ABCD
# failure branch at ``ToleranceAwareMerit.evaluate``).  The branch
# writes a 2-tuple fallback ``(efl_p, bfl_p) = (ctx.efl, ctx.bfl)``
# rather than a single scalar, and wrapping the tuple in a single
# sentinel singleton would break downstream unpacking.  v4.15.4 deletes
# the class + singleton outright; see the historical comment in the
# v4.15.4 release notes.


# =========================================================================
# MeritTerm base + validity check
# =========================================================================

def ctx_is_valid(ctx: Any, field: str) -> bool:
    """Return True if ``ctx.<field>`` holds a usable physical value.

    Guards against the sentinels set when the ray-leg failed (``1e9``
    for focal lengths) and against NaN/Inf from downstream computations.
    """
    try:
        v = getattr(ctx, field)
    except AttributeError:
        return False
    if v is None:
        return False
    if not np.isfinite(v):
        return False
    if abs(v) >= _INVALID_FL_SENTINEL * 0.5:
        return False
    return True


class MeritTerm:
    """Base class for a single term in the merit function.

    Each merit term takes the full ``EvaluationContext`` (ray-trace
    results, wave field, etc.) and returns a scalar contribution
    (already weighted).  Concrete subclasses override
    :meth:`evaluate`.

    Attributes
    ----------
    weight : float
        Multiplier applied to the raw term value.  Squared residuals
        in a least-squares sense, or additive penalty in a general
        minimize sense.
    needs_wave : bool, default False
        If True, the optimizer will run the wave-optics pipeline
        (``apply_real_lens_traced`` + through-focus) for each
        evaluation.  Set False for pure-geometric terms -- the
        optimizer will skip the expensive wave leg if NO merit
        terms need it.
    needs_ray : bool, default True
        If True, the optimizer runs the ray-leg (system ABCD / EFL / BFL /
        Seidel) each evaluation.  Set False for a merit that reads neither
        ``ctx.efl/bfl`` nor ``ctx.seidel`` -- e.g. a rigorous-element
        (RCWA / metasurface / coatings) merit with no imaging prescription --
        so the optimizer skips the ray-leg (a speed win, and the unblocker for
        optimizing a prescription that has no sensible ABCD).  The ray-leg is
        skipped only if NO merit term needs it.
    """

    weight: float = 1.0
    needs_wave: bool = False
    needs_ray: bool = True
    name: str = 'MeritTerm'

    def evaluate(self, ctx: Any) -> float:
        raise NotImplementedError


# =========================================================================
# EvaluationContext + DesignResult
# =========================================================================

@dataclass
class EvaluationContext:
    prescription: Dict[str, Any]
    wavelength: float
    N: int
    dx: float
    efl: float = 0.0
    bfl: float = 0.0
    seidel: np.ndarray = field(default_factory=lambda: np.zeros(5))
    E_exit: Optional[np.ndarray] = None  # wave leg output
    strehl_best: float = 0.0
    rms_radius_best: float = np.inf
    z_best: float = 0.0
    opd_map: Optional[np.ndarray] = None
    efls_per_wavelength: Optional[np.ndarray] = None
    # Populated when a MultiPrescriptionParameterization is used.
    # ``prescription`` stays == ``prescriptions[0]`` for backward
    # compatibility so single-prescription merit terms keep working.
    prescriptions: Optional[List[Dict[str, Any]]] = None
    # Current parameter vector (populated by design_optimize).  Lets
    # JaxMeritTerm route through its build_args(x) for analytic
    # gradient propagation.  Standalone evaluations may leave this
    # None; merits should fall back to ctx-based code paths in that
    # case.
    x: Optional[np.ndarray] = None
    # Per-eval cache for canonical-polynomial fits.  Used by
    # LGAberrationMerit so a CompositeMerit with multiple LG terms
    # (centre / edge / corner emitter classes) builds the fit ONCE
    # per merit eval instead of once per term.  Lives only for the
    # lifetime of a single merit_fn(x) call -- the next eval gets a
    # fresh context with an empty cache.
    _canonical_fit_cache: Dict[Any, Any] = field(default_factory=dict)

    def rms_wavefront_waves(self, n_modes: int = 21,
                             exclude_low_order: int = 3) -> float:
        """RMS wavefront error in waves, excluding the first
        ``exclude_low_order`` Zernike modes (default: piston, tilt X,
        tilt Y).  Computed from the current OPD map.
        """
        if self.opd_map is None:
            return np.inf
        ap = self.prescription.get('aperture_diameter')
        if ap is None:
            return np.inf
        from ..analysis import zernike_decompose
        coeffs, _ = zernike_decompose(
            self.opd_map, self.dx, ap, n_modes=n_modes)
        # rms of higher-order modes, in waves (v5.24.x: shared helper).
        return zernike_higher_order_rms_waves(
            coeffs, exclude_low_order, self.wavelength)


@dataclass
class DesignResult:
    x: np.ndarray
    prescription: Dict[str, Any]
    merit: float
    converged: bool
    iterations: int
    time_sec: float
    context_final: EvaluationContext
    scipy_result: Any = None
    # Populated when a MultiPrescriptionParameterization was used.
    # Otherwise None (use ``prescription`` for the single-lens case).
    prescriptions: Optional[List[Dict[str, Any]]] = None


# =========================================================================
# Constraint (v4.16 #9)
# =========================================================================

# Methods that support scipy.optimize.NonlinearConstraint via
# scipy.optimize.minimize.  Any OTHER method passed with non-empty
# ``constraints=`` raises a clear ValueError pointing to these two.
_METHODS_SUPPORTING_CONSTRAINTS = ('SLSQP', 'trust-constr')


# v4.16.3 (audit P2-NEW-F1-1): one-cycle DeprecationWarning latched at
# module level, pattern parity with v4.16.2 MultiWavelengthMerit.
# v4.16.1 shipped a ``Constraint.__post_init__`` auto-probe that called
# ``fun(np.zeros(1))`` to shape-check the return.  v4.16.2 silently
# removed it (the probe was expensive for BFL-style ``fun`` callables
# that internally ran a full ray-trace) and moved the contract to an
# opt-in :meth:`Constraint.validate` method.  Emit a one-cycle
# DeprecationWarning so callers that came to rely on the v4.16.1
# auto-probe notice the change and call ``.validate()`` explicitly.
# Latched at module level so an optimisation loop that builds many
# ``Constraint(...)`` objects doesn't flood the warning channel.
_CONSTRAINT_AUTOPROBE_DEPRECATION_WARNED = False


@dataclass
class Constraint:
    """Hard nonlinear constraint for :func:`design_optimize` (v4.16 #9).

    Each ``Constraint`` wraps a SCALAR callable
    ``f(x) -> float`` plus inclusive lower/upper bounds
    ``(lb, ub)``.  The optimiser enforces ``lb <= f(x) <= ub`` as a
    hard constraint via :class:`scipy.optimize.NonlinearConstraint`,
    rather than via the soft ``max(0, x - threshold)**2`` penalty
    pattern used by the merit terms.

    Hard constraints are honoured exactly by SciPy methods
    ``'SLSQP'`` and ``'trust-constr'``.  They are NOT supported by
    ``'L-BFGS-B'`` / Powell / Nelder-Mead / the global methods (DE,
    basin-hopping, dual_annealing) and ``design_optimize`` will raise
    a clear ``ValueError`` recommending SLSQP / trust-constr in those
    cases.

    Scalar-only contract (v4.16.1)
    ------------------------------
    ``fun`` MUST return a scalar (Python ``float`` / ``int`` / 0-d
    ndarray).  Vector-valued constraints are not supported by the
    pymoo wrapper (``design_optimize_multi_objective``) because the
    wrapper coerces each constraint result via ``float(...)``, which
    raises ``TypeError: only length-1 arrays can be converted to
    Python scalars`` on a ``(K,)``-shaped ndarray.

    v4.16.2 (audit P2-NEW-F1-1): the automatic ``fun(np.zeros(1))``
    probe formerly run in :meth:`__post_init__` has been REMOVED.
    For a BFL-style ``fun`` that internally calls
    :func:`system_abcd` / :func:`apply_real_lens`, the probe ran an
    entire trace on every ``Constraint(...)`` instantiation (e.g.
    once per optimisation set-up and once per parallel-worker fork).
    Caught exceptions were swallowed silently so users couldn't even
    see the wasted work.  Call :meth:`validate` explicitly after
    construction if you want the (best-effort) shape check.

    Pickle-safety contract (v4.16.1 / v4.16.2)
    -------------------------------------------
    Lambdas (``lambda x: ...``) are NOT picklable, so a
    ``Constraint(fun=lambda x: ...)`` instance breaks
    :func:`scipy.optimize.differential_evolution(..., workers>1)`
    and the joblib-parallelised FD-gradient path
    (``PicklingError: Can't pickle <function <lambda>>``).
    v4.16.2 (audit P2-NEW-F1-2) replaces the v4.16.1
    ``getattr(fun, '__name__', None) == '<lambda>'`` heuristic with
    a direct :func:`pickle.dumps` probe so closures
    (``def inner(x): ...``) and ``functools.partial(lambda x: ...,
    ...)`` -- neither of which has ``__name__ == '<lambda>'`` --
    also raise the warning.  Module-level functions (including
    :func:`functools.partial` of a module-level function) pickle
    cleanly and do NOT warn.

    Attributes
    ----------
    fun : Callable[[np.ndarray], float]
        ``f(x) -> float``.  Receives the current parameter vector
        and returns the constraint quantity as a scalar.  Must
        return a 0-d array / Python float / int; ndarrays of shape
        ``(K,)`` are rejected at construction time.
    lb : float or None
        Inclusive lower bound on ``f(x)``.  ``None`` means ``-inf``.
    ub : float or None
        Inclusive upper bound on ``f(x)``.  ``None`` means ``+inf``.
    label : str, default ''
        Human-readable label for the constraint (e.g.
        ``'BFL >= 5 mm'``).  Appears in progress-callback messages.
    jac : callable, optional
        Analytic Jacobian of ``f`` w.r.t. ``x`` if available.  Passed
        through to ``NonlinearConstraint`` as-is.  Default ``None``
        uses scipy's FD Jacobian.

    Example
    -------
    >>> from lumenairy.optimize import Constraint
    >>> # Require sum(x) <= 1 exactly.  v4.16.2 (audit P3-NEW-F1-7):
    >>> # docstring example now uses a module-level function so
    >>> # copy-paste users don't trigger the v4.16.1 lambda warning.
    >>> def my_constraint(x):
    ...     return float(x[0] + x[1] - 1.0)
    >>> sum_constraint = Constraint(
    ...     fun=my_constraint, lb=0.0, ub=np.inf,
    ...     label='sum_to_one')
    """

    fun: Callable
    lb: Optional[float] = None
    ub: Optional[float] = None
    label: str = ''
    jac: Optional[Callable] = None

    def __post_init__(self) -> None:
        if not callable(self.fun):
            raise TypeError(
                f"Constraint.fun must be callable, got "
                f"{type(self.fun).__name__}")
        if self.lb is None and self.ub is None:
            raise ValueError(
                f"Constraint(label={self.label!r}): at least one of "
                f"lb / ub must be supplied (both None is unbounded "
                f"and the constraint is a no-op).")

        # v4.16.2 (audit P2-NEW-F1-2): pickle-probe instead of the
        # v4.16.1 ``__name__ == '<lambda>'`` heuristic.  The
        # ``__name__`` check missed closures (``def inner(x): ...``
        # has ``__name__ == 'inner'``) and
        # ``functools.partial(lambda x: ..., ...)`` (whose
        # ``__name__`` attribute does not exist at all) -- both are
        # genuinely unpicklable and both will fail under
        # ``differential_evolution(workers>1)`` / joblib-parallelised
        # FD-gradients with PicklingError at the first parallel
        # eval, which is exactly the failure mode the v4.16.1 closure
        # was meant to prevent.  A direct ``pickle.dumps(self.fun)``
        # probe catches all three patterns at construction time
        # cheaply (most ``fun`` callables are tens of bytes pickled).
        #
        # v4.16.3 (audit P2-NEW-F1-2): widen the catch list from
        # ``(pickle.PicklingError, AttributeError, TypeError)`` to
        # ``Exception``.  The narrow tuple missed ``RecursionError``
        # (deep object graph), ``RuntimeError`` (raised by a custom
        # ``__reduce__``), ``MemoryError`` (huge object), and arbitrary
        # exceptions from ``__reduce__`` / ``__getstate__``.  A ``fun``
        # whose ``__reduce__`` raises ``RuntimeError`` propagates out of
        # ``Constraint(...)`` construction, defeating the warning's
        # "friendlier than rejecting" intent.  Pickling is a
        # best-effort heuristic; any failure is a "not safely
        # picklable" signal.  ``BaseException`` (``KeyboardInterrupt`` /
        # ``SystemExit``) is intentionally left to propagate.
        import pickle
        try:
            pickle.dumps(self.fun)
        except Exception as e:  # noqa: BLE001 -- best-effort probe
            warnings.warn(
                f"Constraint(label={self.label!r}): ``fun`` is not "
                f"picklable ({type(e).__name__}: {e!s}); this will "
                f"fail under "
                f"``differential_evolution(workers>1)`` / "
                f"joblib-parallelised FD-gradient with PicklingError. "
                f"Define ``fun`` as a module-level function (or use "
                f"functools.partial on a module-level function) for "
                f"parallel-workers compatibility.  Single-process "
                f"SLSQP / trust-constr works with non-picklable "
                f"callables without issue.",
                UserWarning, stacklevel=2,
            )

        # v4.16.2 (audit P2-NEW-F1-1): the v4.16.1 ``fun(np.zeros(1))``
        # auto-probe used to live here -- removed because for a
        # BFL-style ``fun`` that calls ``system_abcd(...)`` internally
        # it would run an entire trace on every ``Constraint(...)``
        # instantiation, swallowing the exception silently.  Users
        # who want the shape check can call ``self.validate()``
        # explicitly after construction; see :meth:`validate` for
        # the same scalar-only contract.

    def validate(self) -> None:
        """Best-effort scalar-shape check by probing ``fun(np.zeros(1))``.

        Pre-v4.16.2 this ran automatically in :meth:`__post_init__`
        but the probe was expensive for the canonical BFL / EFL
        constraint pattern (runs a full ray-trace) and the caught
        exceptions were swallowed silently.  Now opt-in -- call
        explicitly after construction if you want the shape check.

        Raises
        ------
        TypeError
            If ``fun(np.zeros(1))`` returns an ndarray with shape
            != ``()`` (i.e. a vector or higher-rank array, which
            ``design_optimize_multi_objective``'s ``float(...)``
            coercion would later reject).

        Notes
        -----
        If ``fun`` raises while evaluating the synthetic
        ``np.zeros(1)`` probe (e.g. because it requires a specific
        parameter-vector shape), the exception is caught and the
        check is skipped -- this is best-effort, matching the
        v4.16.1 behaviour.  No warning is emitted on the silent
        skip.
        """
        _probe_x = np.zeros(1, dtype=np.float64)
        try:
            _probe_result = self.fun(_probe_x)
        except Exception:  # noqa: BLE001 -- probe is best-effort
            # ``fun`` rejected our synthetic probe.  We can't
            # pre-validate the return shape; skip the check (same
            # contract as the v4.16.1 auto-probe).
            return
        if isinstance(_probe_result, np.ndarray):
            if _probe_result.shape != ():
                raise TypeError(
                    f"Constraint(label={self.label!r}): ``fun`` "
                    f"returned an ndarray of shape "
                    f"{_probe_result.shape!r}; only scalar (0-d) "
                    f"returns are supported.  The pymoo wrapper "
                    f"(design_optimize_multi_objective) coerces "
                    f"each constraint via float(...) which raises "
                    f"TypeError on (K,)-shaped returns.  Either:\n"
                    f"  1. Reduce the vector return to a single "
                    f"scalar (e.g. ``np.max(...)`` or "
                    f"``np.sum(...)``), OR\n"
                    f"  2. Split into K separate Constraint "
                    f"objects, one per component.")

    def to_scipy(self):
        """Return a :class:`scipy.optimize.NonlinearConstraint`."""
        import scipy.optimize as so
        lb = -np.inf if self.lb is None else float(self.lb)
        ub = +np.inf if self.ub is None else float(self.ub)
        kwargs: Dict[str, Any] = {'fun': self.fun, 'lb': lb, 'ub': ub}
        if self.jac is not None:
            kwargs['jac'] = self.jac
        return so.NonlinearConstraint(**kwargs)
