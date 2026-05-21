"""v4.16.2 / Agent B -- optimisation subsystem closures.

Pins the 1 P1 + 3 P2 + 3 P3 audit items from
``docs/audits/AUDIT_V4_16_1_2026_05_19.md`` concentrated in
``lumenairy/optimize/core.py``:

* **P1-NEW-F1-3** -- ``MultiWavelengthMerit`` SUM->AVG silent
  behaviour change.  v4.16.1 fixed the wrong arithmetic but shipped
  without a migration warning.  v4.16.2 emits a one-cycle
  ``FutureWarning`` (latched at module level) so existing
  weight-calibrations notice the silent 3x drop on a 3-wavelength
  configuration.
* **P2-NEW-F1-1** -- ``Constraint.__post_init__`` probe runs real
  user code (a BFL constraint runs an entire ray-trace on every
  ``Constraint(...)`` instantiation, swallowing exceptions
  silently).  v4.16.2 moves the probe to an opt-in
  :meth:`Constraint.validate` method.
* **P2-NEW-F1-2** -- v4.16.1 lambda detector
  (``getattr(fun, '__name__', None) == '<lambda>'``) misses
  closures (``def inner(x): ...``) and
  ``functools.partial(lambda x: ..., ...)`` -- both genuinely
  unpicklable.  v4.16.2 replaces the heuristic with a
  :func:`pickle.dumps` probe that catches all three patterns.
* **P2-NEW-F1-3** -- the v4.16.0 ``test_v4_16_0_agent_c.py``
  Constraint sites at lines 169, 197, 219, 237 used lambdas and
  emitted the v4.16.1 warning on every construction.  Updated to
  module-level functions.  Pinned negatively here: ``Constraint``
  built with a module-level function must NOT emit any warning.
* **P3-NEW-F1-3** -- ``_resolve_bound`` accepted truncated 3-tuples
  silently; raises ``ValueError`` with a clear hint.
* **P3-NEW-F1-7** -- ``Constraint.fun`` docstring example used a
  lambda (copy-paste users hit the warning).  Updated to a
  module-level function; pinned by reading the docstring at run
  time.
* **P3-NEW-F1-8** -- ``method='lm'`` with non-None ``bounds``
  silently overrides to ``method='trf'`` per scipy's
  ``least_squares`` contract.  v4.16.2 emits a ``UserWarning`` at
  the override point.

Test taxonomy:
  * P1-NEW-F1-3: 3 tests -- (a) the warning fires once with >1
    wavelength; (b) the latch suppresses re-emission; (c)
    single-wavelength configs do NOT warn.
  * P2-NEW-F1-1: 3 tests -- (a) auto-probe gone (no fun() call at
    construction); (b) explicit ``validate()`` shape check; (c)
    ``validate()`` swallows arbitrary ``fun()`` exceptions
    silently.
  * P2-NEW-F1-2: 4 tests -- (a) lambda still warns; (b) closure
    now warns (used to slip); (c) functools.partial(lambda, ...)
    now warns; (d) module-level function does NOT warn.
  * P2-NEW-F1-3: 1 test -- module-level function on Constraint
    does NOT emit UserWarning (negative pin).
  * P3-NEW-F1-3: 2 tests -- (a) 3-tuple raises ValueError;
    (b) 2-tuple still parses cleanly.
  * P3-NEW-F1-7: 1 test -- Constraint docstring example uses
    ``def my_constraint`` not ``lambda``.
  * P3-NEW-F1-8: 1 test -- ``method='lm'`` + bounds emits the
    ``UserWarning`` and the run still completes.

Total: 15 tests.

Author: Andrew Traverso -- v4.16.2 / Agent B
"""
from __future__ import annotations

import functools
import warnings
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Module-level callables used in pickle-probe tests.  Each one is a
# named def at module scope so pickle.dumps succeeds.
# ---------------------------------------------------------------------------


def _module_level_sum(x):
    """sum(x).  Picklable: module-level def at the top of this file."""
    return float(np.sum(x))


def _module_level_first(x):
    """float(x[0]).  Picklable companion to ``_module_level_sum``."""
    return float(x[0])


def _module_level_vector_returner(x):
    """Returns a (3,) ndarray -- intentionally vector-shaped for
    the validate() shape-rejection test."""
    return np.array([float(x[0]), 2 * float(x[0]), 3 * float(x[0])])


def _module_level_raiser(x):
    """Raises RuntimeError when probed -- exercises validate()'s
    silent-skip on caught exceptions."""
    raise RuntimeError(
        "intentional probe failure: this fun requires a specific "
        "x shape and rejects np.zeros(1)")


# ---------------------------------------------------------------------------
# P1-NEW-F1-3: MultiWavelengthMerit SUM->AVG one-cycle FutureWarning
# ---------------------------------------------------------------------------


@pytest.fixture
def reset_multiwl_warn_latch():
    """Reset the module-level ``_MULTIWL_AVG_WARNED`` latch around
    each test so the warning's one-shot semantic is observable
    independently in each call.

    The latch is sticky by design (one notice per process), but
    that makes the warning untestable without a reset hook.  We
    set the module-global directly via ``setattr`` so the test
    environment matches a fresh-process state.
    """
    import lumenairy.optimize.core as _core
    _saved = _core._MULTIWL_AVG_WARNED
    _core._MULTIWL_AVG_WARNED = False
    try:
        yield
    finally:
        _core._MULTIWL_AVG_WARNED = _saved


class _ConstantMerit:
    """Trivial sub-merit returning a constant for every call.
    Avoids per-wavelength wave-leg propagation so we isolate the
    AVG/SUM accounting + warning behaviour.
    """
    name = 'Constant'
    needs_wave = False
    weight = 1.0

    def __init__(self, value: float):
        self.value = float(value)

    def evaluate(self, ctx) -> float:
        return self.value


def _build_singlet_ctx(wl: float):
    """Minimal EvaluationContext with a working singlet
    prescription so MultiWavelengthMerit.evaluate hits the
    standard path (not the ABCD sentinel branch).
    """
    import lumenairy as la
    from lumenairy.optimize.core import EvaluationContext
    pres = la.make_singlet(
        R1=60e-3, R2=float('inf'), d=4e-3, glass='N-BK7',
        aperture=12e-3)
    return EvaluationContext(
        prescription=pres, wavelength=wl,
        N=16, dx=10e-6, efl=0.1, bfl=0.1)


def test_p1_multiwavelength_future_warning_fires_on_multiwl(
        reset_multiwl_warn_latch):
    """audit closure: P1-NEW-F1-3.

    A 3-wavelength ``MultiWavelengthMerit.evaluate`` call emits
    exactly one ``FutureWarning`` describing the SUM->AVG
    transition.  The warning cites v4.16.1 as the change-version,
    advises that the new semantics are CORRECT, and tells users
    how to silence via ``warnings.filterwarnings``.
    """
    from lumenairy.optimize.core import MultiWavelengthMerit

    sub = _ConstantMerit(value=0.5)
    merit = MultiWavelengthMerit(
        wavelengths=[1.30e-6, 1.55e-6, 1.064e-6],
        sub_merit=sub, weight=1.0)
    ctx = _build_singlet_ctx(1.30e-6)

    with warnings.catch_warnings(record=True) as w_list:
        warnings.simplefilter('always')
        result = merit.evaluate(ctx)

    # One FutureWarning, content includes the migration call-out
    # (v4.16.1 change version, the AVG-vs-SUM phrasing, and the
    # silence-hint).
    future_warnings = [w for w in w_list
                       if issubclass(w.category, FutureWarning)]
    assert len(future_warnings) == 1, (
        f"expected 1 FutureWarning on first-call multi-wavelength "
        f"evaluate, got {len(future_warnings)}: "
        f"{[str(w.message) for w in future_warnings]}")
    msg = str(future_warnings[0].message)
    assert 'v4.16.1' in msg, (
        f"FutureWarning must cite v4.16.1 as the change-version; "
        f"got: {msg!r}")
    assert 'AVG' in msg or 'SUM' in msg, (
        f"FutureWarning must reference the SUM->AVG transition; "
        f"got: {msg!r}")
    assert 'filterwarnings' in msg, (
        f"FutureWarning must include the silence hint via "
        f"warnings.filterwarnings; got: {msg!r}")
    # AVG semantics: 3-wavelength merit equals the constant value
    # (weight=1).  Pre-v4.16.1 would have returned 3 * 0.5 = 1.5.
    assert result == pytest.approx(0.5, rel=1e-12)


def test_p1_multiwavelength_future_warning_latched_once_per_process(
        reset_multiwl_warn_latch):
    """audit closure: P1-NEW-F1-3 (latch pin).

    A SECOND call to evaluate() in the same process must NOT
    re-emit the FutureWarning.  Without the latch, an
    optimisation loop would flood the warning channel (hundreds
    of evaluations per design_optimize call).
    """
    from lumenairy.optimize.core import MultiWavelengthMerit

    sub = _ConstantMerit(value=0.5)
    merit = MultiWavelengthMerit(
        wavelengths=[1.30e-6, 1.55e-6, 1.064e-6],
        sub_merit=sub, weight=1.0)
    ctx = _build_singlet_ctx(1.30e-6)

    with warnings.catch_warnings(record=True) as w_list_1:
        warnings.simplefilter('always')
        merit.evaluate(ctx)
    # First call: latch trips; FutureWarning emitted.
    n1 = sum(1 for w in w_list_1
             if issubclass(w.category, FutureWarning))
    assert n1 == 1, (
        f"latch did not record first-call FutureWarning; got "
        f"{n1} FutureWarning(s)")

    with warnings.catch_warnings(record=True) as w_list_2:
        warnings.simplefilter('always')
        merit.evaluate(ctx)
        merit.evaluate(ctx)
        merit.evaluate(ctx)
    # Subsequent calls: latch is hot; NO FutureWarning.
    n2 = sum(1 for w in w_list_2
             if issubclass(w.category, FutureWarning))
    assert n2 == 0, (
        f"latch failed to suppress re-emission; got {n2} "
        f"FutureWarning(s) on 3 follow-up calls: "
        f"{[str(w.message) for w in w_list_2 if issubclass(w.category, FutureWarning)]}")


def test_p1_multiwavelength_no_future_warning_on_single_wavelength(
        reset_multiwl_warn_latch):
    """audit closure: P1-NEW-F1-3 (single-wavelength negative pin).

    A 1-wavelength ``MultiWavelengthMerit`` evaluation is
    arithmetically identical pre- and post-v4.16.1 (the SUM->AVG
    transition only matters for >1 wavelength).  The
    FutureWarning must therefore NOT fire for single-wavelength
    configurations, even with a fresh latch.
    """
    from lumenairy.optimize.core import MultiWavelengthMerit

    sub = _ConstantMerit(value=0.5)
    merit = MultiWavelengthMerit(
        wavelengths=[1.30e-6],  # single wavelength
        sub_merit=sub, weight=1.0)
    ctx = _build_singlet_ctx(1.30e-6)

    with warnings.catch_warnings(record=True) as w_list:
        warnings.simplefilter('always')
        merit.evaluate(ctx)

    future_warnings = [w for w in w_list
                       if issubclass(w.category, FutureWarning)]
    assert len(future_warnings) == 0, (
        f"single-wavelength MultiWavelengthMerit must NOT emit "
        f"the SUM->AVG FutureWarning (the transition is a no-op "
        f"for len(wavelengths) == 1); got "
        f"{[str(w.message) for w in future_warnings]}")


# ---------------------------------------------------------------------------
# P2-NEW-F1-1: Constraint.__post_init__ probe removed; opt-in validate()
# ---------------------------------------------------------------------------


def test_p2_constraint_post_init_does_not_call_fun():
    """audit closure: P2-NEW-F1-1.

    Pre-v4.16.2 ``Constraint(...)`` ran a probe call
    ``fun(np.zeros(1))`` from ``__post_init__`` -- expensive for
    user-defined constraint callables that internally run a full
    ray-trace.  Post-v4.16.2 the auto-probe is gone; the
    ``fun`` must not be called at construction time.

    Sentinel: a counter ``fun`` that increments on every call.
    Construction must leave the counter at zero.
    """
    from lumenairy.optimize import Constraint

    call_count = {'n': 0}

    def _counting_fun(x):
        call_count['n'] += 1
        return float(x[0])

    c = Constraint(fun=_counting_fun, lb=0.0, ub=None,
                   label='counting_probe')
    assert call_count['n'] == 0, (
        f"Constraint.__post_init__ called fun {call_count['n']} "
        f"time(s); post-v4.16.2 should call fun ZERO times at "
        f"construction (audit P2-NEW-F1-1 fix).")
    # Sanity: a later explicit call still works.
    assert c.fun(np.zeros(1)) == 0.0


def test_p2_constraint_validate_rejects_vector_return():
    """audit closure: P2-NEW-F1-1.

    The shape-check the v4.16.1 ``__post_init__`` probe used to
    do is preserved in :meth:`Constraint.validate`.  Calling
    ``validate()`` on a Constraint whose ``fun`` returns a
    non-0-d ndarray must raise ``TypeError`` with the same
    contract phrasing.
    """
    from lumenairy.optimize import Constraint

    c = Constraint(fun=_module_level_vector_returner,
                   lb=0.0, ub=None, label='vector')
    # Construction does NOT raise (probe removed).
    # Explicit validate() catches the contract violation.
    with pytest.raises(TypeError, match=r'(scalar|0-d|shape)'):
        c.validate()


def test_p2_constraint_validate_silent_on_probe_exception():
    """audit closure: P2-NEW-F1-1.

    A fun() that raises on the synthetic ``np.zeros(1)`` probe
    (e.g. because it needs a specific parameter-vector shape) is
    best-effort -- ``validate()`` swallows the exception silently
    and returns without complaint.  Matches the v4.16.1
    auto-probe behaviour.
    """
    from lumenairy.optimize import Constraint

    c = Constraint(fun=_module_level_raiser, lb=0.0, ub=None,
                   label='raises')
    # Should NOT raise; the probe exception is swallowed.
    c.validate()  # returns None, no exception, no warning


# ---------------------------------------------------------------------------
# P2-NEW-F1-2: pickle-probe lambda detector catches closures + partial
# ---------------------------------------------------------------------------


def test_p2_lambda_still_warns():
    """audit closure: P2-NEW-F1-2.

    A bare lambda is unpicklable; the v4.16.2 pickle-probe must
    still catch it (regression for the v4.16.1 closure).
    """
    from lumenairy.optimize import Constraint

    with pytest.warns(UserWarning, match=r'(picklable|pickle)'):
        Constraint(fun=lambda x: float(x[0]),
                   lb=0.0, ub=None, label='lambda')


def test_p2_closure_now_warns():
    """audit closure: P2-NEW-F1-2.

    Closures (``def inner(x): ...`` returned from an outer
    factory) are NOT picklable and were missed by the v4.16.1
    ``__name__ == '<lambda>'`` heuristic (closures have
    ``__name__ == 'inner'``, not ``<lambda>``).  The v4.16.2
    pickle-probe catches them.
    """
    from lumenairy.optimize import Constraint

    def _make_closure(threshold):
        def inner(x):  # __name__ == 'inner', NOT '<lambda>'
            return float(x[0] - threshold)
        return inner

    closure = _make_closure(0.5)
    # Sanity-check the precondition: __name__ is NOT '<lambda>'.
    assert closure.__name__ == 'inner', (
        f"precondition violated: closure.__name__ = "
        f"{closure.__name__!r}; expected 'inner' so the test "
        f"actually exercises the v4.16.2 pickle-probe (not the "
        f"v4.16.1 __name__ heuristic).")

    with pytest.warns(UserWarning, match=r'(picklable|pickle)'):
        Constraint(fun=closure, lb=0.0, ub=None, label='closure')


def test_p2_functools_partial_of_lambda_now_warns():
    """audit closure: P2-NEW-F1-2.

    ``functools.partial(lambda x: ..., ...)`` is NOT picklable
    (the underlying lambda is not picklable, and partial's
    pickle protocol pickles the underlying callable).  The
    v4.16.1 ``__name__`` heuristic missed this because partial
    objects do not expose ``__name__`` at all.  The v4.16.2
    pickle-probe catches it.
    """
    from lumenairy.optimize import Constraint

    # functools.partial(lambda, ...) -- the lambda's identity is
    # preserved inside partial; pickling the partial requires
    # pickling the lambda, which fails.
    p = functools.partial(lambda x, c: float(x[0] - c), c=0.5)

    with pytest.warns(UserWarning, match=r'(picklable|pickle)'):
        Constraint(fun=p, lb=0.0, ub=None, label='partial')


def test_p2_module_level_function_does_not_warn():
    """audit closure: P2-NEW-F1-2 (negative pin).

    A module-level function pickles cleanly.  The v4.16.2
    pickle-probe must NOT emit a UserWarning for this case --
    that would be a false-positive that defeats the whole point
    of the heuristic.
    """
    from lumenairy.optimize import Constraint

    with warnings.catch_warnings(record=True) as w_list:
        warnings.simplefilter('always')
        Constraint(fun=_module_level_sum, lb=0.0, ub=None,
                   label='module_fn')

    user_warnings = [w for w in w_list
                     if issubclass(w.category, UserWarning)]
    assert len(user_warnings) == 0, (
        f"module-level function must NOT emit UserWarning; got "
        f"{[str(w.message) for w in user_warnings]}")


# ---------------------------------------------------------------------------
# P2-NEW-F1-3: existing v4.16.0 tests updated to module-level functions
# ---------------------------------------------------------------------------


def test_p2_v4_16_0_constraint_test_no_userwarning_in_construct():
    """audit closure: P2-NEW-F1-3 (negative pin).

    Build the Constraint objects from ``test_v4_16_0_agent_c.py``
    -- now using module-level functions per the v4.16.2 update
    -- and verify they don't emit ANY UserWarning.  Regressing
    one of those four sites back to a lambda would re-trigger
    the warning on every construction and trip this test.
    """
    from lumenairy.optimize import Constraint

    # Use the same module-level helpers that the updated
    # test_v4_16_0_agent_c.py now imports.
    from tests.unit.test_v4_16_0_agent_c import (
        _first_coord,
        _sum_constraint,
    )

    with warnings.catch_warnings(record=True) as w_list:
        warnings.simplefilter('always')
        # The four v4.16.0 Constraint sites (now module-level fn).
        Constraint(fun=_sum_constraint, lb=15.0, ub=None,
                   label='sum_constraint_a')
        Constraint(fun=_sum_constraint, lb=15.0, ub=None,
                   label='sum_constraint_b')
        # The "both bounds None" site -- legitimately raises
        # ValueError; we catch and verify the lambda-warning
        # didn't ALSO fire.
        with pytest.raises(ValueError):
            Constraint(fun=_sum_constraint, lb=None, ub=None,
                       label='unbounded')
        Constraint(fun=_sum_constraint, lb=15.0, ub=None,
                   label='sum_constraint_d')
        # Plus the pymoo-test site.
        Constraint(fun=_first_coord, lb=0.5, ub=None,
                   label='first_coord')

    user_warnings = [w for w in w_list
                     if issubclass(w.category, UserWarning)]
    assert len(user_warnings) == 0, (
        f"the 5 v4.16.0 Constraint sites (now using module-level "
        f"functions) must NOT emit any UserWarning; got "
        f"{[str(w.message) for w in user_warnings]}")


# ---------------------------------------------------------------------------
# P3-NEW-F1-3: _resolve_bound 3-tuple guard
# ---------------------------------------------------------------------------


def test_p3_resolve_bound_rejects_3tuple():
    """audit closure: P3-NEW-F1-3.

    A 3-tuple bound (genuine user mistake -- e.g. confusing
    scipy bounds, ``least_squares`` bounds, and DE bounds
    formats) MUST raise ``ValueError`` with a clear message.
    Pre-v4.16.2 the comprehension silently dropped the 3rd
    element.

    Drive the failure via ``design_optimize(method='lm', ...)``
    since ``_resolve_bound`` is defined as a local function
    inside the ``method == 'lm'`` branch.
    """
    from lumenairy.optimize import (
        CallableMerit,
        DesignParameterization,
        design_optimize,
    )

    template = {
        'params': [0.5, 0.5],
        'surfaces': [{'radius': np.inf, 'aperture': 5e-3,
                      'glass_before': 'air', 'glass_after': 'air'}],
        'thicknesses': [],
        'aperture_diameter': 5e-3,
    }
    free_vars: List[Tuple[Any, ...]] = [('params', 0), ('params', 1)]
    # 3-tuple -- one valid bound + one mistakenly truncated.
    bounds_bad = [(0.0, 1.0, 999.0), (0.0, 1.0)]
    param = DesignParameterization(
        template=template, free_vars=free_vars, bounds=bounds_bad)

    def _q(ctx) -> float:
        x = np.asarray(ctx.x, dtype=np.float64)
        return float(np.sum(x * x))

    merits = [CallableMerit(_q, weight=1.0, name='q')]

    # Pre-v4.16.2: silently accepted, third element dropped.
    # Post-v4.16.2: ValueError with a clear hint.
    with pytest.raises(ValueError, match=r'(2-tuple|3-tuple|lower, upper)'):
        design_optimize(
            param, merits, wavelength=1.31e-6,
            method='lm', max_iter=5, verbose=False)


def test_p3_resolve_bound_2tuple_still_parses_cleanly():
    """audit closure: P3-NEW-F1-3 (negative pin).

    The 2-tuple guard is a length check -- it must NOT regress
    the existing 2-tuple path.  A vanilla ``method='lm'`` run
    with proper 2-tuple bounds must still complete (with the
    new lm->trf override warning, which we explicitly catch).
    """
    from lumenairy.optimize import (
        CallableMerit,
        DesignParameterization,
        design_optimize,
    )

    template = {
        'params': [0.5, 0.5],
        'surfaces': [{'radius': np.inf, 'aperture': 5e-3,
                      'glass_before': 'air', 'glass_after': 'air'}],
        'thicknesses': [],
        'aperture_diameter': 5e-3,
    }
    free_vars: List[Tuple[Any, ...]] = [('params', 0), ('params', 1)]
    bounds_good = [(-1.0, 1.0), (-1.0, 1.0)]
    param = DesignParameterization(
        template=template, free_vars=free_vars, bounds=bounds_good)

    def _q(ctx) -> float:
        x = np.asarray(ctx.x, dtype=np.float64)
        return float(np.sum(x * x))

    merits = [CallableMerit(_q, weight=1.0, name='q')]

    # The new lm->trf UserWarning will fire; we swallow it so the
    # test exercises only the bound-parsing path.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        result = design_optimize(
            param, merits, wavelength=1.31e-6,
            method='lm', max_iter=10, verbose=False)
    assert result is not None
    assert len(result.x) == 2
    assert np.all(np.isfinite(result.x))


# ---------------------------------------------------------------------------
# P3-NEW-F1-7: Constraint.fun docstring example uses def, not lambda
# ---------------------------------------------------------------------------


def test_p3_constraint_docstring_example_uses_def_not_lambda():
    """audit closure: P3-NEW-F1-7.

    The ``Constraint`` docstring example in v4.16.1 used
    ``fun=lambda x: ...`` -- so copy-paste users hit the new
    UserWarning at the first construction.  v4.16.2 rewrites
    the example to use ``def my_constraint(x): ...`` and the
    module-level form.  Pin this textually so future docstring
    edits don't silently revert.
    """
    from lumenairy.optimize import Constraint

    doc = Constraint.__doc__ or ''
    # The new example uses ``def my_constraint(x):``.
    assert 'def my_constraint(x)' in doc, (
        f"Constraint docstring example must use ``def "
        f"my_constraint(x)`` (audit P3-NEW-F1-7).  Current "
        f"docstring excerpt: "
        f"{doc[doc.find('Example'):doc.find('Example')+400] if 'Example' in doc else doc[:400]!r}")
    # And the constructor call must reference the new fn, not lambda.
    assert 'fun=my_constraint' in doc, (
        "Constraint docstring example must use "
        "``fun=my_constraint`` (the module-level function).  "
        "Found a lambda reference instead in the example.")
    # Defensive: no ``fun=lambda`` in the docstring example block.
    if 'Example' in doc:
        example_block = doc[doc.find('Example'):]
        assert 'fun=lambda' not in example_block, (
            "Constraint docstring example still contains "
            "``fun=lambda`` -- audit P3-NEW-F1-7 fix incomplete.")


# ---------------------------------------------------------------------------
# P3-NEW-F1-8: method='lm' + bounds emits the lm->trf override warning
# ---------------------------------------------------------------------------


def test_p3_lm_bounds_emits_override_userwarning():
    """audit closure: P3-NEW-F1-8.

    ``design_optimize(method='lm', bounds=...)`` silently
    overrides scipy's ``least_squares`` method to ``'trf'``
    (per scipy's contract: ``method='lm'`` does NOT accept
    bounds).  Pre-v4.16.2 this happened with no notice.
    v4.16.2 emits a UserWarning at the override point; the run
    still completes successfully (existing v4.16.1 tests pass).
    """
    from lumenairy.optimize import (
        CallableMerit,
        DesignParameterization,
        design_optimize,
    )

    template = {
        'params': [0.5],
        'surfaces': [{'radius': np.inf, 'aperture': 5e-3,
                      'glass_before': 'air', 'glass_after': 'air'}],
        'thicknesses': [],
        'aperture_diameter': 5e-3,
    }
    param = DesignParameterization(
        template=template, free_vars=[('params', 0)],
        bounds=[(0.0, 1.0)])

    def _q(ctx) -> float:
        return float(np.asarray(ctx.x, dtype=np.float64)[0] ** 2)

    merits = [CallableMerit(_q, weight=1.0, name='q')]

    with pytest.warns(UserWarning, match=r"(lm.*trf|trf.*lm|method='trf')"):
        result = design_optimize(
            param, merits, wavelength=1.31e-6,
            method='lm', max_iter=10, verbose=False)
    assert result is not None
    assert len(result.x) == 1
    assert np.all(np.isfinite(result.x))


# ---------------------------------------------------------------------------
# Meta-pin: fix-line markers (defensive against accidental revert)
# ---------------------------------------------------------------------------


def test_v4_16_2_agent_b_fix_lines_present():
    """audit closure: meta-pin.

    Each of the 6 in-scope audit items must show its signature
    line in ``lumenairy/optimize/core.py`` at run time.  Casual
    reverts would silently re-introduce the bugs; the functional
    tests above catch the behaviour, but a textual marker pin
    surfaces the regression earlier in CI.
    """
    repo = Path(__file__).resolve().parents[2]
    core_py = (repo / 'lumenairy' / 'optimize' / 'core.py').read_text(
        encoding='cp1252')

    # P1-NEW-F1-3: module-level latch + FutureWarning emission.
    assert '_MULTIWL_AVG_WARNED' in core_py, (
        "P1-NEW-F1-3 fix-line missing: '_MULTIWL_AVG_WARNED' "
        "module-level latch in optimize/core.py.")
    assert 'FutureWarning' in core_py, (
        "P1-NEW-F1-3 fix-line missing: 'FutureWarning' emission "
        "in optimize/core.py (MultiWavelengthMerit.evaluate).")
    # P2-NEW-F1-1: opt-in validate() method on Constraint.
    assert 'def validate(self)' in core_py, (
        "P2-NEW-F1-1 fix-line missing: 'def validate(self)' on "
        "Constraint in optimize/core.py.")
    # P2-NEW-F1-2: pickle-probe.
    assert 'pickle.dumps(self.fun)' in core_py, (
        "P2-NEW-F1-2 fix-line missing: 'pickle.dumps(self.fun)' "
        "in Constraint.__post_init__.")
    # P3-NEW-F1-3: 2-tuple length guard.
    assert 'must be a 2-tuple' in core_py, (
        "P3-NEW-F1-3 fix-line missing: '2-tuple (lower, upper)' "
        "phrasing in _resolve_bound.")
    # P3-NEW-F1-7: module-level fn docstring example.
    assert 'def my_constraint(x)' in core_py, (
        "P3-NEW-F1-7 fix-line missing: 'def my_constraint(x)' "
        "in Constraint docstring example.")
    # P3-NEW-F1-8: lm->trf override warning.
    assert "method='lm', bounds=" in core_py, (
        "P3-NEW-F1-8 fix-line missing: lm->trf override warning "
        "phrasing in design_optimize.")
