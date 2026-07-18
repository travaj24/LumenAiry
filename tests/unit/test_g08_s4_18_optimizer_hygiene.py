"""G08 / S4-18 (AUDIT_V5_24_2) -- optimizer hygiene.

Covers the concrete, safe sub-fixes:

* ``_clip_x0_to_bounds`` clips the starting vector into the bounds box
  before dispatch (scipy's out-of-bounds x0 handling is method-dependent
  and mostly silent).
* ``seed`` is a real parameter (default 42) threaded into the stochastic
  global methods instead of a hard-coded literal.
* the analytic jacobian is forwarded to the basin-hopping local search.
* the higher-order Zernike-RMS quadrature is a single shared helper
  (byte-identical dedup of 3 former copies).
* ``MaxThicknessMerit`` skips AIR gaps (docstring parity with the
  ``MinThicknessMerit`` sibling).

Pure-NumPy; no JAX / GUI needed.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy
from lumenairy.optimize.context import (
    EvaluationContext,
    zernike_higher_order_rms_waves,
)
from lumenairy.optimize.core import (
    DesignParameterization,
    FocalLengthMerit,
    MaxThicknessMerit,
    MinThicknessMerit,
    design_optimize,
)
from lumenairy.optimize.driver import _clip_x0_to_bounds

_UNSET = object()


def _singlet_template():
    return lumenairy.make_singlet(
        R1=60e-3, R2=float('inf'), d=4e-3, glass='N-BK7', aperture=12e-3)


# =========================================================================
# x0 clip into bounds
# =========================================================================

class TestClipX0:

    def test_clip_above_upper_matches_np_clip(self):
        x0 = np.array([60e-3])
        out = _clip_x0_to_bounds(x0, [(40e-3, 50e-3)])
        # Independent oracle: np.clip for a fully-finite box.
        assert out == pytest.approx(np.clip(x0, 40e-3, 50e-3))
        assert out[0] == pytest.approx(50e-3)

    def test_clip_below_lower(self):
        out = _clip_x0_to_bounds(np.array([10e-3]), [(40e-3, 90e-3)])
        assert out[0] == pytest.approx(40e-3)

    def test_feasible_start_is_unchanged(self):
        x0 = np.array([55e-3, 3e-3])
        out = _clip_x0_to_bounds(x0, [(40e-3, 90e-3), (2e-3, 10e-3)])
        assert np.array_equal(out, x0)
        assert out is not x0  # fresh array

    def test_none_endpoints_clip_only_the_bounded_side(self):
        # (None, hi): only the upper side clips.  (lo, None): only lower.
        # x0=[10, -3] -> [5 (upper clip), 2 (lower clip)].
        out = _clip_x0_to_bounds(
            np.array([10.0, -3.0]), [(None, 5.0), (2.0, None)])
        assert out[0] == pytest.approx(5.0)   # upper bound clips 10 -> 5
        assert out[1] == pytest.approx(2.0)   # lower bound clips -3 -> 2

    def test_free_side_never_clips(self):
        # (None, hi) with a value far below hi -> unchanged (lower free).
        # (lo, None) with a value far above lo -> unchanged (upper free).
        out = _clip_x0_to_bounds(
            np.array([-1e6, 1e6]), [(None, 5.0), (2.0, None)])
        assert out[0] == pytest.approx(-1e6)
        assert out[1] == pytest.approx(1e6)

    def test_none_bounds_is_noop_copy(self):
        x0 = np.array([1.0, 2.0])
        out = _clip_x0_to_bounds(x0, None)
        assert np.array_equal(out, x0)

    def test_dispatch_receives_clipped_x0(self, monkeypatch):
        """End-to-end: an out-of-bounds template start must reach scipy
        already clipped into the box.  Pre-fix the raw (infeasible) x0
        was handed to scipy."""
        import scipy.optimize as so_mod
        captured = {}

        class _R:
            x = np.array([50e-3])
            success = True
            nit = 1

        def _spy_minimize(fun, x0, **kw):
            captured['x0'] = np.array(x0, dtype=float)
            return _R()

        monkeypatch.setattr(so_mod, 'minimize', _spy_minimize)
        param = DesignParameterization(
            template=_singlet_template(),
            free_vars=[('surfaces', 0, 'radius')],
            bounds=[(40e-3, 50e-3)])  # template radius 60mm is ABOVE 50mm
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            design_optimize(
                parameterization=param,
                merit_terms=[FocalLengthMerit(target=0.1)],
                wavelength=1.30e-6, N=16, dx=10e-6,
                method='L-BFGS-B', max_iter=1, verbose=False)
        assert captured['x0'][0] == pytest.approx(50e-3), (
            'S4-18: out-of-bounds x0 was not clipped before dispatch.')


# =========================================================================
# seed threading
# =========================================================================

class TestSeedThreading:

    def _run_de(self, monkeypatch, seed_kwarg):
        import scipy.optimize as so_mod
        captured = {}

        class _R:
            x = np.array([55e-3])
            success = True
            nit = 1

        def _spy_de(func, bounds, **kw):
            captured['seed'] = kw.get('seed')
            return _R()

        monkeypatch.setattr(so_mod, 'differential_evolution', _spy_de)
        param = DesignParameterization(
            template=_singlet_template(),
            free_vars=[('surfaces', 0, 'radius')],
            bounds=[(40e-3, 90e-3)])
        kwargs = {} if seed_kwarg is _UNSET else {'seed': seed_kwarg}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            design_optimize(
                parameterization=param,
                merit_terms=[FocalLengthMerit(target=0.1)],
                wavelength=1.30e-6, N=16, dx=10e-6,
                method='de', max_iter=1, verbose=False, **kwargs)
        return captured['seed']

    def test_seed_default_is_42(self, monkeypatch):
        assert self._run_de(monkeypatch, _UNSET) == 42

    def test_seed_is_user_controllable(self, monkeypatch):
        assert self._run_de(monkeypatch, 123) == 123

    def test_seed_none_passes_through(self, monkeypatch):
        assert self._run_de(monkeypatch, None) is None


# =========================================================================
# analytic jac -> basin-hopping local search
# =========================================================================

def test_basinhopping_forwards_analytic_jac(monkeypatch):
    import scipy.optimize as so_mod
    captured = {}

    class _R:
        x = np.array([55e-3])
        success = True
        nit = 1

    def _spy_basin(func, x0, **kw):
        captured['minimizer_kwargs'] = kw.get('minimizer_kwargs')
        return _R()

    monkeypatch.setattr(so_mod, 'basinhopping', _spy_basin)

    def _user_jac(x):
        return np.zeros_like(np.asarray(x, dtype=float))

    param = DesignParameterization(
        template=_singlet_template(),
        free_vars=[('surfaces', 0, 'radius')],
        bounds=[(40e-3, 90e-3)])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        design_optimize(
            parameterization=param,
            merit_terms=[FocalLengthMerit(target=0.1)],
            wavelength=1.30e-6, N=16, dx=10e-6,
            method='basin_hopping', max_iter=1, verbose=False,
            jac=_user_jac)
    mk = captured['minimizer_kwargs']
    assert mk is not None and mk.get('jac') is _user_jac, (
        'S4-18: analytic jacobian was not forwarded to the basin-hopping '
        'local search.')


# =========================================================================
# Zernike-RMS quadrature dedup -- byte-identical to each former call site.
# =========================================================================

def test_zernike_rms_helper_reproduces_former_inline_formula():
    """The shared helper must reproduce the exact inline quadrature that
    lived byte-for-byte at the three former call sites, to machine
    precision, across representative coefficient vectors / exclusions /
    wavelengths."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = int(rng.integers(6, 30))
        coeffs = rng.standard_normal(n) * 1e-7
        exclude = int(rng.integers(0, 5))
        wl = float(rng.uniform(0.4e-6, 1.6e-6))
        # Former inline formula (context.py / merit_terms.py, verbatim):
        higher = coeffs[exclude:]
        rms_m = float(np.sqrt(np.sum(higher ** 2)))
        expected = rms_m / wl
        got = zernike_higher_order_rms_waves(coeffs, exclude, wl)
        assert got == expected  # exact bit-for-bit, not approx


# =========================================================================
# MaxThicknessMerit air handling.
# =========================================================================

class TestMaxThicknessAir:

    def _ctx(self, thicknesses, glasses):
        surfaces = [{'glass_after': g} for g in glasses]
        pres = {'thicknesses': thicknesses, 'surfaces': surfaces}
        return EvaluationContext(prescription=pres, wavelength=1.3e-6,
                                 N=16, dx=10e-6)

    def test_air_gap_not_penalised(self):
        # A 100mm air gap, max 20mm: air is skipped -> zero penalty.
        ctx = self._ctx([100e-3], ['air'])
        assert MaxThicknessMerit(max_thickness=20e-3).evaluate(ctx) == 0.0

    def test_glass_gap_is_penalised(self):
        # A 100mm N-BK7 gap, max 20mm -> penalised.
        ctx = self._ctx([100e-3], ['N-BK7'])
        merit = MaxThicknessMerit(max_thickness=20e-3, weight=1.0)
        excess = 100e-3 - 20e-3
        assert merit.evaluate(ctx) == pytest.approx(excess * excess)

    def test_include_air_restores_penalty(self):
        ctx = self._ctx([100e-3], ['air'])
        merit = MaxThicknessMerit(max_thickness=20e-3, include_air=True)
        excess = 100e-3 - 20e-3
        assert merit.evaluate(ctx) == pytest.approx(excess * excess)

    def test_min_thickness_air_skip_unchanged(self):
        # Sibling parity: MinThickness still skips air (dedup preserved).
        ctx = self._ctx([0.0], ['air'])
        assert MinThicknessMerit(min_thickness=1e-3).evaluate(ctx) == 0.0
        ctx2 = self._ctx([0.0], ['N-BK7'])
        assert MinThicknessMerit(min_thickness=1e-3).evaluate(ctx2) == \
            pytest.approx((1e-3) ** 2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
