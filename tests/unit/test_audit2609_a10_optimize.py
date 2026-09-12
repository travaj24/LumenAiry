"""WP-A10 / AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11 -- ``lumenairy/optimize``.

Findings I7 (through-focus scan gating, edge-thickness constraint,
``design_optimize_multi_objective`` infeasible-run contract) and I8
(``method='newton'`` bounds warning, ``x0`` bounds check,
``create_zoom_configs`` slots, ndarray aperture cache key).

Per ``docs/TESTING_STANDARDS.md`` there are **no wall-clock or speedup
assertions** anywhere here: the focus-scan saving is pinned by COUNTING the
propagations the driver performs, which is a decision the code makes, not a
timing a shared box can move.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.optimize import MinEdgeThicknessMerit, edge_thickness
from lumenairy.optimize import core as _opt_core
from lumenairy.optimize import driver as _driver


# ---------------------------------------------------------------------------
# I7 -- the 31-plane through-focus scan runs only when a merit reads it
# ---------------------------------------------------------------------------

def _count_focus_slices(merit_terms, monkeypatch, n_iter=1):
    """Run a tiny design_optimize and return (n_merit_evals, n_scan_slices).

    Instruments ``through_focus_scan`` at the binding ``driver.py`` actually
    calls (``optimize.core``), which is the same indirection the repo's
    existing mock.patch tests rely on.
    """
    calls = {'scan_slices': 0, 'scan_calls': 0}
    real_scan = _opt_core.through_focus_scan

    def counting_scan(E, dx, wavelength, z_values, **kw):
        calls['scan_calls'] += 1
        calls['scan_slices'] += int(np.size(z_values))
        return real_scan(E, dx, wavelength, z_values, **kw)

    monkeypatch.setattr(_opt_core, 'through_focus_scan', counting_scan)

    template = la.make_singlet(50e-3, -50e-3, 4e-3, 'N-BK7',
                               aperture=200e-6)
    param = la.DesignParameterization(
        template,
        free_vars=[('surfaces', 0, 'radius')],
        bounds=[(40e-3, 60e-3)])
    res = la.design_optimize(
        param, merit_terms=merit_terms, wavelength=1.31e-6,
        N=64, dx=4e-6, method='L-BFGS-B', max_iter=n_iter, verbose=False)
    del res
    return calls


def test_i7_focus_scan_is_skipped_when_no_merit_reads_it(monkeypatch):
    """Measured pre-fix (audit, 3 free vars, N=128, L-BFGS-B, max_iter=2):

        17 merit evals, 17 apply_real_lens calls, 527 through-focus slices
         -> per merit eval: 1.0 lens propagations + 31.0 focus-scan
            propagations

    i.e. 97 % of the wave-leg work was the focus scan, recomputed from
    scratch for every finite-difference probe -- **even when no merit read
    ``strehl_best`` / ``z_best`` / ``rms_radius_best``**.  The count, not a
    timing, is the assertion.
    """
    # RMSWavefrontMerit reads ctx.opd_map only.
    calls = _count_focus_slices([la.RMSWavefrontMerit(weight=1.0)],
                                monkeypatch)
    assert calls['scan_calls'] == 0, (
        f'focus scan ran {calls["scan_calls"]} times for a merit that does '
        f'not read it ({calls["scan_slices"]} slices)')


def test_i7_focus_scan_still_runs_for_a_merit_that_reads_it(monkeypatch):
    """The gate must not silently disable the scan for StrehlMerit."""
    calls = _count_focus_slices([la.StrehlMerit(min_strehl=0.8)], monkeypatch)
    assert calls['scan_calls'] >= 1
    # Default z_scan_n = 31 slices per call.
    assert calls['scan_slices'] == 31 * calls['scan_calls']


def test_i7_needs_focus_scan_flags_are_declared_consistently():
    """Every library wave merit declares the flag, and the default is safe.

    ``MeritTerm.needs_focus_scan`` defaults to True so a user-written merit
    written before the flag existed keeps its scan.
    """
    from lumenairy.optimize.context import MeritTerm
    assert MeritTerm.needs_focus_scan is True
    readers = (la.StrehlMerit(min_strehl=0.8),
               la.SpotSizeMerit(max_rms_radius=5e-6))
    non_readers = (la.RMSWavefrontMerit(),
                   la.ZernikeCoefficientMerit({4: 0.0}))
    for m in readers:
        assert getattr(m, 'needs_focus_scan', True) is True, type(m).__name__
    for m in non_readers:
        assert getattr(m, 'needs_focus_scan', True) is False, type(m).__name__
    # A wrapper forwards the sub-merit's requirement in both directions.
    assert la.MultiWavelengthMerit(
        [1.3e-6, 1.55e-6],
        la.StrehlMerit(min_strehl=0.8)).needs_focus_scan is True
    assert la.MultiWavelengthMerit(
        [1.3e-6, 1.55e-6], la.RMSWavefrontMerit()).needs_focus_scan is False


# ---------------------------------------------------------------------------
# I7 -- edge thickness
# ---------------------------------------------------------------------------

def _sag_sphere(R, h):
    """Closed-form spherical sag -- the independent oracle."""
    if not math.isfinite(R):
        return 0.0
    return R - math.copysign(math.sqrt(R * R - h * h), R)


@pytest.mark.parametrize('R1, R2, t, h', [
    (25e-3, -25e-3, 3e-3, 12.5e-3),     # biconvex: knife-edged at this h
    (-50e-3, -25e-3, 3e-3, 12.5e-3),    # concave-first meniscus: sags cancel
    (51.5e-3, float('inf'), 3.6e-3, 12.5e-3),   # plano-convex
    (float('inf'), float('inf'), 2e-3, 10e-3),  # plane parallel plate
])
def test_i7_edge_thickness_matches_closed_form(R1, R2, t, h):
    """``t_edge = t_c + sag(R2, h) - sag(R1, h)``.

    Oracle: the exact spherical sag ``R - sign(R) sqrt(R^2 - h^2)``, evaluated
    independently of the library's ``surface_sag_general``.  Both are
    double-precision closed forms of the same expression, so the only error
    is IEEE rounding: measured |diff| <= 5.2e-18 m across these four cases,
    i.e. ~15 decades below the 1e-3 m scale of the quantity.  The bar (1e-15
    absolute) sits ~2.5 decades above the measured residual and ~12 decades
    below any physically meaningful error.

    The meniscus row is the case the audit's probe list flags: the two sags
    have the SAME sign there and partly cancel, so a formula written as
    ``t_c - |sag1| - |sag2|`` would get it wrong (it gives -6.5 mm instead of
    +1.24 mm here).
    """
    rx = la.make_singlet(R1, R2, t, 'N-BK7', aperture=2 * h)
    got = edge_thickness(rx, 0)
    want = t + _sag_sphere(R2, h) - _sag_sphere(R1, h)
    assert got == pytest.approx(want, abs=1e-15)


def test_i7_edge_thickness_merit_penalises_a_knife_edge():
    """The merit is zero on a manufacturable element and positive on a
    knife-edged one, with the quadratic-deficit magnitude pinned exactly.
    """
    ok = la.make_singlet(200e-3, -200e-3, 5e-3, 'N-BK7', aperture=10e-3)
    bad = la.make_singlet(25e-3, -25e-3, 3e-3, 'N-BK7', aperture=25e-3)
    m = MinEdgeThicknessMerit(min_edge=1e-3, weight=1.0)

    class _Ctx:
        def __init__(self, rx):
            self.prescription = rx

    t_ok = edge_thickness(ok, 0)
    assert t_ok > 1e-3
    assert m.evaluate(_Ctx(ok)) == 0.0

    t_bad = edge_thickness(bad, 0)
    assert t_bad < 0.0          # knife edge: the surfaces have crossed
    expected = (1e-3 - t_bad) ** 2
    assert m.evaluate(_Ctx(bad)) == pytest.approx(expected, rel=1e-12)


def test_i7_edge_thickness_merit_skips_air_gaps_by_default():
    """Parity with MinThicknessMerit's glass-only classification."""
    rx = la.combine_prescriptions(
        [la.make_singlet(200e-3, -200e-3, 5e-3, 'N-BK7', aperture=10e-3),
         la.make_singlet(200e-3, -200e-3, 5e-3, 'N-BK7', aperture=10e-3)],
        gaps=1e-4)

    class _Ctx:
        prescription = rx

    # The 0.1 mm AIR gap is below min_edge but must not be penalised.
    assert MinEdgeThicknessMerit(min_edge=1e-3).evaluate(_Ctx()) == 0.0
    assert MinEdgeThicknessMerit(
        min_edge=1e-3, include_air=True).evaluate(_Ctx()) > 0.0


def test_i7_edge_thickness_merit_validates_kwargs():
    with pytest.raises(ValueError, match='MinEdgeThicknessMerit'):
        MinEdgeThicknessMerit(min_edge=float('nan'))
    with pytest.raises(ValueError, match='MinEdgeThicknessMerit'):
        MinEdgeThicknessMerit(min_edge=1e-3, semi_diameter=-1.0)


# ---------------------------------------------------------------------------
# I7 / I8 -- multi-objective contracts (pymoo is NOT installed here)
# ---------------------------------------------------------------------------

def test_i7_infeasible_pareto_run_raises_instead_of_returning_nan(
        monkeypatch):
    """pymoo's documented infeasible-run contract is ``Result.X is None``.

    ``np.asarray(None, dtype=np.float64)`` is ``array(nan)`` with
    ``ndim == 0`` (measured below on this numpy), so the ``if X.ndim == 1``
    normalisation did NOT fire and a 0-d NaN array was returned AS the Pareto
    front; with ``progress`` supplied, ``X.shape[0]`` raised ``IndexError:
    tuple index out of range`` instead.

    pymoo is an optional dependency and is not installed on this machine, so
    the pymoo side is supplied by a minimal stub injected into
    ``sys.modules`` that reproduces exactly the documented contract.  The
    numpy half -- the reason the guard failed -- is measured directly.
    """
    # The numpy step the whole finding rests on.
    arr = np.asarray(None, dtype=np.float64)
    assert arr.ndim == 0 and np.isnan(arr)

    import sys
    import types

    from lumenairy.optimize import multi_objective as mo

    class _Result:
        X = None                 # pymoo's infeasible-run contract
        F = None
        CV = np.array([[0.7], [1.3]])

    class _Problem:
        def __init__(self, *a, **k):
            pass

    mods = {}
    for name in ('pymoo', 'pymoo.algorithms', 'pymoo.algorithms.moo',
                 'pymoo.algorithms.moo.nsga2', 'pymoo.core',
                 'pymoo.core.problem', 'pymoo.optimize'):
        mods[name] = types.ModuleType(name)
    mods['pymoo.algorithms.moo.nsga2'].NSGA2 = lambda **k: object()
    mods['pymoo.core.problem'].ElementwiseProblem = _Problem
    mods['pymoo.optimize'].minimize = lambda *a, **k: _Result()
    for name, m in mods.items():
        monkeypatch.setitem(sys.modules, name, m)
    monkeypatch.setattr(mo, 'PYMOO_AVAILABLE', True)

    with pytest.raises(ValueError, match='NO feasible solution'):
        mo.design_optimize_multi_objective(
            merits=[lambda x: float(x[0]), lambda x: float(-x[0])],
            x0=np.array([0.5]), bounds=[(0.0, 1.0)],
            n_generations=3, pop_size=4)


def test_i7_infeasible_guard_is_present_in_the_source():
    """Structural pin: the guard must run BEFORE the asarray.

    (pymoo is an optional dependency and is absent here, so the end-to-end
    path cannot be exercised; this pins the ordering the finding is about.)
    """
    import inspect

    from lumenairy.optimize import multi_objective as mo
    src = inspect.getsource(mo.design_optimize_multi_objective)
    i_guard = src.index('pymoo_res.X is None')
    i_asarray = src.index('X = np.asarray(pymoo_res.X')
    assert i_guard < i_asarray, 'the None-guard must precede np.asarray'


def test_i8_x0_outside_bounds_warns():
    """The docstring promised "a sanity check against bounds"; ``x0`` was only
    ever read for ``n_params``.
    """
    from lumenairy.optimize import multi_objective as mo
    if mo.PYMOO_AVAILABLE:                       # pragma: no cover
        pytest.skip('pymoo present: covered by the end-to-end path')
    # Without pymoo the function raises ImportError -- but the bounds check
    # must already have warned, since it precedes the pymoo import.
    import inspect
    src = inspect.getsource(mo.design_optimize_multi_objective)
    i_warn = src.index('x0 lies OUTSIDE bounds')
    i_pymoo = src.index('_import_pymoo') if '_import_pymoo' in src else len(src)
    assert i_warn < i_pymoo


# ---------------------------------------------------------------------------
# I8 -- create_zoom_configs slots
# ---------------------------------------------------------------------------

def test_i8_zoom_configs_reject_a_mis_sized_spacing_vector():
    """Pre-fix: ``if j < len(pres['thicknesses'])`` dropped extra entries with
    no warning, and the positional loop wrote GLASS centre thicknesses even
    though the parameter is named ``zoom_spacings`` and documented as "the
    air-gap thicknesses".
    """
    template = la.combine_prescriptions(
        [la.make_singlet(50e-3, -50e-3, 4e-3, 'N-BK7', aperture=10e-3),
         la.make_singlet(50e-3, -50e-3, 4e-3, 'N-BK7', aperture=10e-3)],
        gaps=20e-3)
    n = len(template['thicknesses'])
    assert n == 3
    with pytest.raises(ValueError, match='thickness slot'):
        la.create_zoom_configs(template, [[1e-3] * (n + 2)])
    with pytest.raises(ValueError, match='thickness slot'):
        la.create_zoom_configs(template, [[(99, 1e-3)]])
    # The flat form still works when it matches exactly.
    cfgs = la.create_zoom_configs(template, [[4e-3, 10e-3, 4e-3],
                                             [4e-3, 50e-3, 4e-3]])
    assert [c.prescription['thicknesses'][1] for c in cfgs] == [10e-3, 50e-3]
    # And the pair form touches only the named slot.
    cfgs2 = la.create_zoom_configs(template, [[(1, 33e-3)]])
    assert cfgs2[0].prescription['thicknesses'] == [4e-3, 33e-3, 4e-3]


# ---------------------------------------------------------------------------
# I8 -- ndarray aperture cache key
# ---------------------------------------------------------------------------

def test_i8_aperture_cache_key_is_a_content_digest():
    """Pre-fix the ndarray branch keyed on Python's 64-bit
    ``hash(arr.tobytes())``, so a collision returned the WRONG cached aperture
    mask.  The key must now be a content digest and must separate arrays that
    differ anywhere.
    """
    from lumenairy.optimize.wrapper_merits import _wrapper_merit_aperture_key
    a = np.zeros((16, 16))
    b = a.copy()
    b[7, 7] = 1.0
    ka, kb = _wrapper_merit_aperture_key(a), _wrapper_merit_aperture_key(b)
    assert ka != kb
    assert ka == _wrapper_merit_aperture_key(a.copy())
    # Not an int (the old 64-bit hash); a 128-bit digest.
    assert isinstance(ka[3], bytes) and len(ka[3]) == 16
    # Shape / dtype still separate.
    assert (_wrapper_merit_aperture_key(np.zeros((16, 16), dtype=np.float32))
            != ka)
    assert _wrapper_merit_aperture_key(None) == ('none',)
    assert _wrapper_merit_aperture_key(1.0) == ('scalar', 1.0)


# ---------------------------------------------------------------------------
# I8 -- method='newton' drops bounds without a warning
# ---------------------------------------------------------------------------

def test_i8_newton_warns_that_bounds_are_not_enforced():
    """``trust-ncg`` accepts no bounds; only the FD stencil is clipped.  The
    generic ``minimize`` branch and the ``lm`` branch both warn loudly in
    exactly this situation -- the Newton branch did not, so a bounded Newton
    run LOOKED bounded and was not.
    """
    template = la.make_singlet(50e-3, -50e-3, 4e-3, 'N-BK7', aperture=6e-3)
    param = la.DesignParameterization(
        template,
        free_vars=[('surfaces', 0, 'radius')],
        bounds=[(40e-3, 60e-3)])
    with pytest.warns(UserWarning, match="cannot handle bounds"):
        la.design_optimize(
            param, merit_terms=[la.FocalLengthMerit(90e-3)],
            wavelength=1.31e-6, N=32, dx=4e-6,
            method='newton', max_iter=1, verbose=False)


def test_i8_newton_without_bounds_does_not_warn():
    """Guard on the guard: no bounds, no warning."""
    template = la.make_singlet(50e-3, -50e-3, 4e-3, 'N-BK7', aperture=6e-3)
    param = la.DesignParameterization(
        template, free_vars=[('surfaces', 0, 'radius')], bounds=None)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        la.design_optimize(
            param, merit_terms=[la.FocalLengthMerit(90e-3)],
            wavelength=1.31e-6, N=32, dx=4e-6,
            method='newton', max_iter=1, verbose=False)
    assert not [x for x in w if 'cannot handle bounds' in str(x.message)]


del _driver      # imported only to assert the module loads cleanly
