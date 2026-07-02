"""Wave-6 audit fixes: optimize cluster (P3-47, P3-48, P3-50 + TNC options).

Pins the fixes for AUDIT_V5_17_0_2026_07_01_DEEP findings:

* **P3-47** -- the library's custom FD gradient paths (``jac='auto'``
  with a JAX merit, and ``method='newton'``) perturbed ``x[i]`` by
  ``+/-step`` with NO clipping to ``parameterization.bounds``, so a
  variable pinned exactly at a bound evaluated the merit OUTSIDE the
  box (probe: ``_fd_grad_pure(f, [1.0], scheme='forward')`` evaluated
  at x=1.0000001 with ub=1.0).  Post-fix ``_fd_grad_pure`` accepts a
  ``bounds`` argument, clips stencil legs to the box, and switches to
  a one-sided difference pointing into the box at an active bound
  (scipy _numdiff style).  Unclipped stencils stay byte-identical.

* **P3-48** -- ``MaxFNumberMerit.evaluate`` used
  ``prescription.get('aperture_diameter', 1e-3)``, whose default does
  NOT fire when the key exists with value ``None`` (a codebase-legal
  "no aperture" state every sibling merit guards); ``None > 0`` then
  raised TypeError out of scipy.minimize, aborting the run.  Post-fix
  ``None`` routes to the existing ``ap <= 0`` fnum=1e9 fallback.

* **P3-50** -- ``DesignParameterization`` lacked the duplicate-
  free_vars guard v4.14 added to MultiPrescriptionParameterization:
  duplicate paths silently over-parameterised the design (last write
  wins, dead x-slot, split FD gradient).  Post-fix construction raises
  a clear ValueError, with numpy-int/py-int key normalisation.

* **TNC maxiter** (wave-5 follow-up) -- the generic minimize branch
  passed ``options={'maxiter': max_iter}`` to TNC, which takes
  ``maxfun``; scipy warned 'Unknown solver options: maxiter' and ran
  on its DEFAULT budget.  Post-fix the option name is mapped
  per-method so ``max_iter`` is effective for TNC.

(P3-49 is a doc-only fix in multiconfig.py -- no test.)

Author: Wave-6 audit implementer -- v5.17.x
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from lumenairy.optimize import RawParameterization, design_optimize
from lumenairy.optimize.driver import _fd_bounds_arrays, _fd_grad_pure
from lumenairy.optimize.merit_terms import CallableMerit, MaxFNumberMerit
from lumenairy.optimize.parameterizations import DesignParameterization

# ---------------------------------------------------------------------------
# P3-47: FD stencil clipped to the bounds box
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('scheme', ['forward', 'central'])
def test_p3_47_fd_stencil_stays_inside_box_at_upper_bound(scheme):
    """Pre-fix: x pinned at ub=1.0 evaluated f at 1.0000001 (outside
    the box).  Post-fix: every evaluation satisfies lb <= x <= ub and
    the gradient is still accurate (one-sided difference)."""
    evals: list = []

    def f(x):
        evals.append(float(x[0]))
        return float(x[0]) ** 2

    g = _fd_grad_pure(f, np.array([1.0]), scheme=scheme,
                      bounds=[(0.0, 1.0)])
    assert max(evals) <= 1.0, (
        f'{scheme}: FD stencil evaluated at {max(evals)} > ub=1.0')
    assert min(evals) >= 0.0
    assert g[0] == pytest.approx(2.0, rel=1e-5)


def test_p3_47_fd_stencil_stays_inside_box_at_lower_bound():
    """Central FD with x pinned at lb=0.0 must not evaluate below 0."""
    evals: list = []

    def f(x):
        evals.append(float(x[0]))
        return (float(x[0]) - 1.0) ** 2

    g = _fd_grad_pure(f, np.array([0.0]), scheme='central',
                      bounds=[(0.0, 1.0)])
    assert min(evals) >= 0.0, (
        f'FD stencil evaluated at {min(evals)} < lb=0.0')
    # At x=0 the default step is eps*scale_floor = 1e-13, so the
    # one-sided quotient carries ~1e-3 relative cancellation noise.
    assert g[0] == pytest.approx(-2.0, rel=1e-2)


def test_p3_47_sentinel_discontinuity_above_bound_not_sampled():
    """The audit's failure mode: a merit that degenerates to a 1e9
    sentinel immediately above ub.  Pre-fix the forward stencil sampled
    the sentinel and produced a ~1e16 spurious gradient component;
    post-fix the gradient is the true one-sided value."""
    def f(x):
        v = float(x[0])
        if v > 1.0:
            return 1e9  # infeasible-region sentinel
        return (v - 2.0) ** 2

    g = _fd_grad_pure(f, np.array([1.0]), scheme='forward',
                      bounds=[(0.0, 1.0)])
    assert abs(g[0]) < 10.0, (
        f'spurious sentinel-driven gradient {g[0]:.3g} (pre-fix ~1e16)')
    assert g[0] == pytest.approx(-2.0, rel=1e-4)


def test_p3_47_unclipped_paths_byte_identical():
    """bounds=None and non-binding bounds must reproduce the historical
    stencil bit-for-bit (both schemes)."""
    rng = np.random.default_rng(0)
    x = rng.uniform(0.2, 0.8, 5)

    def f(v):
        return float(np.sum(np.sin(v) * v ** 2))

    for scheme in ('central', 'forward'):
        g_none = _fd_grad_pure(f, x, scheme=scheme)
        g_wide = _fd_grad_pure(f, x, scheme=scheme,
                               bounds=[(0.0, 1.0)] * 5)
        assert np.array_equal(g_none, g_wide), scheme


def test_p3_47_degenerate_box_gives_zero_gradient():
    """lb == ub leaves no room to difference: gradient component 0
    rather than a divide-by-zero nan."""
    def f(x):
        return float(x[0]) ** 2

    for scheme in ('central', 'forward'):
        g = _fd_grad_pure(f, np.array([0.5]), scheme=scheme,
                          bounds=[(0.5, 0.5)])
        assert g[0] == 0.0, scheme


def test_p3_47_none_bound_entries_mean_unbounded():
    """Driver bounds format allows None entries / endpoints; they must
    normalise to +/-inf (no clipping)."""
    lb_ub = _fd_bounds_arrays([None, (0.0, None), (None, 1.0)], 3)
    lb, ub = lb_ub
    assert lb[0] == -np.inf and ub[0] == np.inf
    assert lb[1] == 0.0 and ub[1] == np.inf
    assert lb[2] == -np.inf and ub[2] == 1.0
    assert _fd_bounds_arrays(None, 3) is None


def test_p3_47_jax_auto_jac_path_respects_bounds():
    """End-to-end (audit probe shape): JaxMeritTerm(build_args) +
    a non-JAX CallableMerit forces the jac='auto' combined path, whose
    FD half pre-fix evaluated the merit ABOVE ub when L-BFGS-B pinned
    x at the bound (probe: 20 evals at x>1.0).  Post-fix: zero merit
    evaluations outside the box."""
    jax = pytest.importorskip('jax')
    jnp = jax.numpy
    from lumenairy.optimize.jax_merits import JaxMeritTerm

    evals: list = []

    def other_merit(ctx):
        x = float(ctx.prescription['_raw_params'][0])
        evals.append(x)
        if x > 1.0:
            return 1e9  # infeasible-region sentinel
        return 0.1 * (x - 1.5) ** 2

    m_jax = JaxMeritTerm(lambda a: (a - 1.5) ** 2,
                         build_args=lambda x: (jnp.asarray(x[0]),),
                         needs_ray=False, real_part=True)
    m_other = CallableMerit(other_merit, needs_wave=False)
    m_other.needs_ray = False
    param = RawParameterization(x0=[0.5], bounds=[(0.0, 1.0)])
    res = design_optimize(param, [m_jax, m_other], wavelength=1.55e-6,
                          N=32, method='L-BFGS-B', max_iter=100,
                          jac='auto', verbose=False)
    outside = [x for x in evals if x > 1.0 + 1e-12]
    assert not outside, (
        f'{len(outside)} merit evaluations above ub=1.0 '
        f'(max {max(evals)}); FD stencil escaped the box')
    assert float(res.x[0]) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# P3-48: MaxFNumberMerit with aperture_diameter=None
# ---------------------------------------------------------------------------

class _Ctx:
    def __init__(self, efl, prescription):
        self.efl = efl
        self.prescription = prescription


def test_p3_48_none_aperture_routes_to_fallback_penalty():
    """Pre-fix: TypeError ('>' not supported between NoneType and int)
    aborted the whole optimization.  Post-fix: the existing ap<=0
    fnum=1e9 fallback penalty fires."""
    m = MaxFNumberMerit(max_f_number=2.0, weight=1.0)
    v = m.evaluate(_Ctx(0.1, {'aperture_diameter': None}))
    assert v == pytest.approx((1e9 - 2.0) ** 2)


def test_p3_48_absent_key_and_valid_aperture_unchanged():
    """Non-regression: the key-absent 1e-3 default and a real aperture
    give byte-identical pre-fix values."""
    m = MaxFNumberMerit(max_f_number=2.0, weight=1.0)
    # key absent -> 1e-3 default -> fnum = 0.1/1e-3 = 100, excess 98
    assert m.evaluate(_Ctx(0.1, {})) == 98.0 ** 2
    # real aperture 0.05 -> fnum = 2.0 -> excess 0
    assert m.evaluate(_Ctx(0.1, {'aperture_diameter': 0.05})) == 0.0
    # zero aperture -> existing fallback (unchanged)
    assert m.evaluate(_Ctx(0.1, {'aperture_diameter': 0.0})) == \
        pytest.approx((1e9 - 2.0) ** 2)


# ---------------------------------------------------------------------------
# P3-50: DesignParameterization duplicate-free_vars guard
# ---------------------------------------------------------------------------

_TMPL = {'surfaces': [{'radius': 0.05}], 'thicknesses': [1e-3]}


def test_p3_50_duplicate_free_vars_rejected():
    """Pre-fix: dup accepted with n_params=2, build() last-write-wins.
    Post-fix: clear ValueError at construction."""
    with pytest.raises(ValueError, match='duplicate path entries'):
        DesignParameterization(
            _TMPL, free_vars=[('thicknesses', 0), ('thicknesses', 0)])


def test_p3_50_numpy_int_duplicate_normalised():
    """numpy int / Python int path components must compare equal in the
    dedup key (mirrors the MultiPrescription guard's normalisation)."""
    with pytest.raises(ValueError, match='duplicate path entries'):
        DesignParameterization(
            _TMPL,
            free_vars=[('thicknesses', 0), ('thicknesses', np.int64(0))])


def test_p3_50_distinct_free_vars_still_accepted():
    p = DesignParameterization(
        _TMPL, free_vars=[('thicknesses', 0), ('surfaces', 0, 'radius')])
    out = p.build(np.array([2e-3, 0.07]))
    assert out['thicknesses'][0] == pytest.approx(2e-3)
    assert out['surfaces'][0]['radius'] == pytest.approx(0.07)


# ---------------------------------------------------------------------------
# TNC option mapping (wave-5 P2-24 follow-up)
# ---------------------------------------------------------------------------

def _quadratic_setup(bounds):
    evals: list = []

    def merit(ctx):
        x = float(ctx.prescription['_raw_params'][0])
        evals.append(x)
        return (x - 2.0) ** 2

    m = CallableMerit(merit, needs_wave=False)
    m.needs_ray = False
    return RawParameterization(x0=[0.5], bounds=bounds), m, evals


def test_tnc_gets_maxfun_not_maxiter():
    """Pre-fix: scipy warned 'Unknown solver options: maxiter' and TNC
    ran on its DEFAULT budget (max_iter silently ineffective).
    Post-fix: no unknown-option warning and max_iter caps the eval
    budget via TNC's maxfun."""
    param, m, evals = _quadratic_setup(bounds=[(0.0, 10.0)])
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter('always')
        design_optimize(param, [m], wavelength=1.55e-6, N=32,
                        method='TNC', max_iter=2, verbose=False)
    unknown = [w for w in wlist
               if 'Unknown solver options' in str(w.message)]
    assert not unknown, [str(w.message) for w in unknown]
    # maxfun=2 budget is now effective: scipy stops within a few evals
    # (a small overshoot past maxfun is scipy-documented; pre-fix the
    # default budget ran ~14 raw evals on this problem).
    n_budget = len(evals)
    evals.clear()
    param2, m2, evals2 = _quadratic_setup(bounds=[(0.0, 10.0)])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        design_optimize(param2, [m2], wavelength=1.55e-6, N=32,
                        method='TNC', max_iter=200, verbose=False)
    assert n_budget < len(evals2), (
        f'max_iter=2 ({n_budget} evals) did not run fewer merit '
        f'evaluations than max_iter=200 ({len(evals2)} evals) -- '
        f'budget option still ineffective for TNC')


def test_non_tnc_methods_still_get_maxiter():
    """Non-regression: L-BFGS-B keeps options={'maxiter': ...} with no
    unknown-option warning either."""
    param, m, _ = _quadratic_setup(bounds=[(0.0, 10.0)])
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter('always')
        res = design_optimize(param, [m], wavelength=1.55e-6, N=32,
                              method='L-BFGS-B', max_iter=100,
                              verbose=False)
    unknown = [w for w in wlist
               if 'Unknown solver options' in str(w.message)]
    assert not unknown
    assert float(res.x[0]) == pytest.approx(2.0, abs=1e-5)
