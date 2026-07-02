"""Wave-5 audit fixes: optimize-driver bounds passthrough.

Pins the fix for AUDIT_V5_17_0_2026_07_01_DEEP finding:

* **P2-24** -- the generic scipy-minimize branch of ``design_optimize``
  forwarded ``bounds`` only for ``('L-BFGS-B', 'SLSQP', 'trust-constr')``
  and SILENTLY dropped them for every other local method -- including
  Powell / Nelder-Mead / TNC / COBYLA / COBYQA, all of which
  scipy >= 1.7 box-constrains natively (probe on scipy 1.17.1: bounded
  Powell on (x-2)^2 with bounds=[(0, 1)] converged to x=2.0 with 15
  merit evaluations outside the box and zero warnings).  Post-fix the
  driver forwards bounds for every method in
  ``_MINIMIZE_METHODS_SUPPORTING_BOUNDS`` (matched case-insensitively,
  like scipy's own dispatch) and warns loudly -- parity with the
  ``method='lm'`` bounds warning -- when a truly bounds-incapable
  method (BFGS / CG / Newton-CG / trust-*) is paired with bounds.

  Non-regression (A/B vs the pre-fix driver): unbounded runs are
  byte-identical (full eval traces compared for Powell / Nelder-Mead /
  BFGS / L-BFGS-B / CG / COBYQA), and BFGS+bounds numerics are
  unchanged (only the new warning is added).

Author: Wave-5 audit implementer -- v5.17.x
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from lumenairy.optimize import RawParameterization, design_optimize
from lumenairy.optimize.driver import _MINIMIZE_METHODS_SUPPORTING_BOUNDS
from lumenairy.optimize.merit_terms import CallableMerit


def _bounded_quadratic_setup():
    """Merit (x-2)^2 with box [0, 1]: unconstrained optimum x=2 lies
    OUTSIDE the box; the box-constrained optimum is the boundary x=1."""
    evals: list = []

    def merit(ctx):
        x = float(ctx.prescription['_raw_params'][0])
        evals.append(x)
        return (x - 2.0) ** 2

    m = CallableMerit(merit, needs_wave=False)
    m.needs_ray = False
    param = RawParameterization(x0=[0.5], bounds=[(0.0, 1.0)])
    return param, m, evals


# ---------------------------------------------------------------------------
# P2-24: bounds are forwarded for every bounds-capable minimize method
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('method', ['Powell', 'Nelder-Mead', 'TNC',
                                    'COBYLA', 'COBYQA'])
def test_p2_24_bounds_respected(method):
    """Pre-fix: all of these converged to x=2.0 (outside the user's
    [0, 1] box, silently).  Post-fix: on-boundary optimum x=1.0 and
    ZERO merit evaluations outside the box."""
    param, m, evals = _bounded_quadratic_setup()
    try:
        res = design_optimize(param, [m], wavelength=1.55e-6, N=32,
                              method=method, max_iter=200, verbose=False)
    except ImportError:  # pragma: no cover - COBYQA needs the cobyqa pkg
        pytest.skip(f'{method} unavailable in this scipy install')
    x_opt = float(res.x[0])
    assert 0.0 <= x_opt <= 1.0, (
        f'{method}: bounded optimum {x_opt} escaped the [0, 1] box '
        f'(bounds dropped?)')
    assert x_opt == pytest.approx(1.0, abs=1e-4), (
        f'{method}: expected the on-boundary optimum x=1.0, got {x_opt}')
    # Powell / Nelder-Mead / TNC / COBYQA clip every trial point into
    # the box; COBYLA treats bounds as linear constraints that MAY be
    # violated at intermediate iterates (scipy-documented behaviour),
    # so only the converged optimum is pinned for it.
    if method != 'COBYLA':
        outside = [x for x in evals if not (0.0 <= x <= 1.0)]
        assert not outside, (
            f'{method}: {len(outside)} merit evaluations outside the box '
            f'(max {max(evals)})')


def test_p2_24_lowercase_method_gets_bounds():
    """scipy's method dispatch is case-insensitive; the driver's
    bounds-capability check must be too (pre-fix even 'l-bfgs-b'
    lowercase dropped bounds)."""
    param, m, evals = _bounded_quadratic_setup()
    res = design_optimize(param, [m], wavelength=1.55e-6, N=32,
                          method='powell', max_iter=200, verbose=False)
    assert float(res.x[0]) == pytest.approx(1.0, abs=1e-4)
    assert all(0.0 <= x <= 1.0 for x in evals)


# ---------------------------------------------------------------------------
# P2-24: bounds-incapable methods warn LOUDLY instead of dropping silently
# ---------------------------------------------------------------------------

def test_p2_24_incapable_method_warns_and_drops():
    """BFGS cannot handle bounds: the driver must now emit a UserWarning
    (parity with the method='lm' bounds warning) while preserving the
    pre-fix numerics (unconstrained optimum x=2.0)."""
    param, m, _ = _bounded_quadratic_setup()
    with pytest.warns(UserWarning, match='cannot handle bounds'):
        res = design_optimize(param, [m], wavelength=1.55e-6, N=32,
                              method='BFGS', max_iter=200, verbose=False)
    # Numerics unchanged: BFGS still finds the unconstrained optimum.
    assert float(res.x[0]) == pytest.approx(2.0, abs=1e-4)


def test_p2_24_unbounded_run_emits_no_bounds_warning():
    """No bounds supplied -> no bounds warning, any method."""
    def merit(ctx):
        return (float(ctx.prescription['_raw_params'][0]) - 2.0) ** 2

    for method in ('Powell', 'BFGS'):
        m = CallableMerit(merit, needs_wave=False)
        m.needs_ray = False
        param = RawParameterization(x0=[0.5], bounds=None)
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter('always')
            res = design_optimize(param, [m], wavelength=1.55e-6, N=32,
                                  method=method, max_iter=200,
                                  verbose=False)
        bounds_warns = [w for w in wlist
                        if 'bounds' in str(w.message).lower()
                        and 'cannot handle' in str(w.message)]
        assert not bounds_warns, (
            f'{method}: spurious bounds warning on an unbounded run: '
            f'{[str(w.message) for w in bounds_warns]}')
        assert float(res.x[0]) == pytest.approx(2.0, abs=1e-3)


# ---------------------------------------------------------------------------
# P2-24: the capability table matches the INSTALLED scipy
# ---------------------------------------------------------------------------

def test_p2_24_capability_table_matches_installed_scipy():
    """Every method the driver treats as bounds-capable must actually
    honour bounds in the installed scipy (no 'cannot handle bounds'
    RuntimeWarning, optimum inside the box); this catches drift if a
    future scipy changes its support matrix."""
    import scipy.optimize as so

    f = lambda x: (x[0] - 2.0) ** 2  # noqa: E731

    for meth in sorted(_MINIMIZE_METHODS_SUPPORTING_BOUNDS):
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter('always')
            try:
                res = so.minimize(f, [0.5], method=meth,
                                  bounds=[(0.0, 1.0)])
            except ImportError:  # pragma: no cover - optional COBYQA dep
                continue
        cannot = [w for w in wlist
                  if 'cannot handle bounds' in str(w.message)]
        assert not cannot, (
            f'{meth}: installed scipy says it cannot handle bounds but '
            f'the driver table claims it can')
        assert 0.0 <= float(res.x[0]) <= 1.0 + 1e-9, (
            f'{meth}: scipy returned an out-of-box optimum {res.x[0]} '
            f'despite bounds')
