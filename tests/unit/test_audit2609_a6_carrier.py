"""WP-A6 -- regression tests for findings C1-C5 of the 2026-09-11 exhaustive
adversarial audit (``docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/
CARRIER.md``), on ``lumenairy/propagators/carrier.py`` and ``carrier_field.py``.

Every bar below carries its derivation, its oracle and the measured pre-fix
value in a comment, per ``docs/TESTING_STANDARDS.md`` S1-S5.  Nothing here
asserts wall-clock time or a speed-up; the two performance findings (C4) are
pinned on the properties that make them safe -- bit-identity where the change
is a re-association, and a derived agreement bound where it is a regrouping --
and on the ALLOCATION count, which is a deterministic property of the code and
not of the machine.

THE ORACLES, all independent of the code under test:

* C1 -- a converging Gaussian whose focal field is known analytically
  (``w0 = lambda |R| / (pi w)``), read back through the public readout; the
  fail-before is the shipped resolver with its beam term reverted IN PROCESS,
  which reproduces the audit's measured table row for row.
* C2 -- a field built from a closed-form decentred parabola / a closed-form
  uniform tilt, where the true radius is known by construction.
* C3 -- ``numpy``'s own promotion rules, and the dtype of the returned array.
* C4 -- whole-grid ``meshgrid`` + ``np.exp`` expressions written out here, and
  ``tracemalloc``.
* C5 -- ``np.fft.fftfreq`` for the frequency axes, a hand-built 1-D angular
  spectrum for the band-limit mask, and a hand-built eikonal for ``dy``.
"""

import dataclasses
import math
import tracemalloc
import warnings

import numpy as np
import pytest

from lumenairy.propagators import carrier as C
from lumenairy.propagators import carrier_field as CF
from lumenairy.propagators.carrier_field import (
    CarrierField,
    CarrierSpec,
    FieldGrid,
)

LAM = 1.31e-6
K0 = 2.0 * np.pi / LAM


# ===========================================================================
# helpers
# ===========================================================================
def _gauss(n, dx, w, xc=0.0, yc=0.0):
    g = (np.arange(n, dtype=np.float64) - n / 2) * dx
    return np.exp(-(((g[None, :] - xc) ** 2 + (g[:, None] - yc) ** 2) / w ** 2))


def _c1_fixture(n=1024, w_in=1.0e-3, na=0.05, ext=4.0):
    """The audit's own C1 fixture: ONE physical converging Gaussian; only the
    REFERENCE carrier is varied between rows."""
    r0 = -w_in / na
    dx = 2.0 * ext * w_in / n
    g = (np.arange(n, dtype=np.float64) - n / 2) * dx
    r2 = g[None, :] ** 2 + g[:, None] ** 2
    e_phys = np.exp(-r2 / w_in ** 2) * np.exp(1j * K0 * r2 / (2.0 * r0))
    w0 = LAM * abs(r0) / (np.pi * w_in)
    return e_phys, r0, dx, w0


def _pre_fix_standoff(env, r, z, dx):
    """The SHIPPED-before-C1 leg: the resolver with its beam term reverted in
    process.  That term is the only change to the resolver, so this is the
    exact pre-fix number (and it reproduces the audit's table)."""
    real = C._beam_containment_standoff
    try:
        C._beam_containment_standoff = lambda *a, **k: 0.0
        return C._default_focus_standoff(env, r, z, LAM, dx)
    finally:
        C._beam_containment_standoff = real


# ===========================================================================
# C1 -- the focus readout sized its stop grid from the CARRIER, not the beam
# ===========================================================================
class TestC1FocusReadoutContainment:
    """Pre-fix, ``carrier_referenced_focus_readout`` returned a 4-40x LOW focal
    peak, silently, whenever the reference carrier was a few percent off the
    beam's own wavefront -- which is the chain's structural case, since it
    takes that carrier from a paraxial ABCD.  Measured (shipped defaults,
    ``on_replica='error'``, no other knobs; CARRIER/p6c_mismatch.py):

        R/R0    containment at the stop plane    peak vs truth   warnings
        1.00              3.21                      1.000000        0
        0.99              1.96                      0.986188        0
        0.98              1.39                      0.745432        0
        0.95              0.91                      0.187913        0
        0.90              0.87                      0.026309        0
    """

    @pytest.mark.parametrize('frac,pre_peak,pre_cont', [
        # (carrier/truth, measured pre-fix peak ratio, pre-fix containment)
        (0.99, 0.986188, 1.963),
        (0.98, 0.745432, 1.387),
        (0.95, 0.187913, 0.909),
        (0.90, 0.026309, 0.863),
    ])
    def test_a_mismatched_carrier_no_longer_costs_the_peak(
            self, frac, pre_peak, pre_cont):
        """ORACLE: the same PHYSICAL field read against its OWN carrier, which
        the audit scored independently against an analytic Gaussian-ABCD focal
        oracle (piston-free relL2 1.05e-03, peak ratio 0.99992 at NA 0.05).
        The readout must return the SAME focus whatever reference carrier the
        caller happens to hold, because only the reference changed.

        BAR 0.98: post-fix the worst of these rows measures 0.985236 and the
        best 0.999514 (the residual is the hand-off model error L(f) ~ 0.155
        NA^3 f^1.6 of the longer leg the fix resolves: 7.7e-3 relL2 at the
        worst row, i.e. ~1.5 % of peak).  Pre-fix the same rows measure
        0.986188 / 0.745432 / 0.187913 / 0.026309, so the bar is clear of the
        post-fix values by 1.005x and of the three failing pre-fix values by
        1.31x / 5.2x / 37x.  It is deliberately NOT tighter: this pins the
        DEFECT, not the 1.5 % model residual."""
        e_phys, r0, dx, w0 = _c1_fixture()
        z = -r0
        kw = dict(dx_out=w0 / 8.0, N_out=64)

        env_ref = C.carrier_referenced_envelope(e_phys, r0, LAM, dx)
        f_ref = C.carrier_referenced_focus_readout(env_ref, r0, z, LAM, dx, **kw)

        r = frac * r0
        env = C.carrier_referenced_envelope(e_phys, r, LAM, dx)
        pd = {}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            f_off = C.carrier_referenced_focus_readout(
                env, r, z, LAM, dx, _period_out=pd, **kw)
        peak = float((np.abs(f_off).max() / np.abs(f_ref).max()) ** 2)

        assert peak > 0.98, (frac, peak)
        assert not w, [str(x.message)[:80] for x in w]
        # and the leg now reaches the margin the resolver's derivation asks
        # for: post-fix every row measures 3.200 against _FOCUS_STANDOFF_MARGIN
        # = 3.2 (the resolver solves for equality), pre-fix 1.96 .. 0.86.
        assert pd['containment'] > 0.99 * C._FOCUS_STANDOFF_MARGIN, pd

    def test_the_pre_fix_leg_really_was_that_bad(self):
        """FAIL-BEFORE, in process: revert the resolver's beam term (its only
        change) and confirm the audit's measured column reappears -- so the
        test above is pinning a real defect and not a fixture artefact."""
        e_phys, r0, dx, w0 = _c1_fixture()
        z = -r0
        kw = dict(dx_out=w0 / 8.0, N_out=64, on_replica='ignore',
                  on_focus_containment='ignore')
        env_ref = C.carrier_referenced_envelope(e_phys, r0, LAM, dx)
        f_ref = C.carrier_referenced_focus_readout(
            env_ref, r0, z, LAM, dx,
            standoff=_pre_fix_standoff(env_ref, r0, z, dx), **kw)
        got = {}
        for frac in (0.98, 0.95, 0.90):
            r = frac * r0
            env = C.carrier_referenced_envelope(e_phys, r, LAM, dx)
            s_old = _pre_fix_standoff(env, r, z, dx)
            f = C.carrier_referenced_focus_readout(
                env, r, z, LAM, dx, standoff=s_old, **kw)
            got[frac] = float((np.abs(f).max() / np.abs(f_ref).max()) ** 2)
        # measured pre-fix: 0.745432 / 0.187913 / 0.026309 (CARRIER/p6c.out).
        # Bars at 1.2x above each measured value and comfortably under the
        # post-fix 0.985+.
        assert got[0.98] < 0.90, got
        assert got[0.95] < 0.23, got
        assert got[0.90] < 0.04, got

    def test_a_flat_envelope_resolves_the_identical_leg(self):
        """The beam term must be EXACTLY zero when the carrier already is the
        beam's own wavefront, so no shipped configuration moves.  Checked over
        the resolver's own calibration matrix (6 NA x 10 extents = 60 cells,
        the grid its derivation was fitted on): measured 0 cells with a
        non-zero beam term."""
        rmag, n = 20e-3, 512
        bad = []
        for na in (0.03, 0.05, 0.10, 0.15, 0.278, 0.35):
            for ext in (1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0, 6.0, 10.0):
                w = na * rmag
                dx = 2.0 * ext * w / n
                env = _gauss(n, dx, w).astype(complex)
                cen = C._envelope_amp_centroid(env, dx, dx)
                w_env = C._envelope_amp_radius(env, dx, dx, centre=cen)
                half = 0.5 * n * dx - max(abs(cen[0]), abs(cen[1]))
                s_beam = C._beam_containment_standoff(
                    env, -rmag, rmag, LAM, dx, w_env, cen, half)
                if s_beam != 0.0:
                    bad.append((na, ext, s_beam))
                # ... and the resolved leg is the pre-fix one, bit for bit
                assert (C._default_focus_standoff(env, -rmag, rmag, LAM, dx)
                        == _pre_fix_standoff(env, -rmag, rmag, dx)), (na, ext)
        assert bad == [], bad

    def test_the_beam_term_agrees_with_the_shipped_closed_form(self):
        """ORACLE: the shipped carrier-referenced law ``f = M/sqrt(ext^2-M^2)``
        evaluated here.  At ``R_eff == R`` the beam-referenced quadratic must
        reproduce it -- the two are the same containment condition, differing
        only in evaluating the diffraction term at the stop plane rather than
        at the carrier focus.

        BARS, both derived: the ABCD width is the SMALLER of the two at every
        ``zeta <= zeta_cf``, so ``s_beam <= s_law`` is an INEQUALITY the
        algebra guarantees (and it is what makes ``max(s_shipped, s_beam)``
        continuous at the flat-envelope short-circuit).  The size of the gap is
        ``2 v/zeta_cf`` on the fraction of ``w^2`` the diffraction term
        carries; measured 1.1 % / 0.5 % / 0.2 % on these three cells, so the
        2 % bar is ~2x over the largest and 50x under the 4-40x errors this
        branch exists to remove.

        The helper short-circuits on an exactly-flat envelope, so the
        comparison is made through the quadratic itself with an envelope
        carrying a deliberately tiny (1e-12 /m) residual curvature: at that
        level ``R_eff`` and ``R`` differ by 5e-14 relative, i.e. by nothing."""
        n, rmag = 512, 20e-3
        m = C._FOCUS_STANDOFF_MARGIN
        for na, ext in ((0.05, 4.0), (0.10, 6.0), (0.03, 10.0)):
            w = na * rmag
            dx = 2.0 * ext * w / n
            g = (np.arange(n) - n / 2) * dx
            r2 = g[None, :] ** 2 + g[:, None] ** 2
            env = (_gauss(n, dx, w) * np.exp(1j * K0 * r2 * 0.5 * 1e-12)
                   ).astype(complex)
            cen = C._envelope_amp_centroid(env, dx, dx)
            w_env = C._envelope_amp_radius(env, dx, dx, centre=cen)
            half = 0.5 * n * dx - max(abs(cen[0]), abs(cen[1]))
            ext_eff = half / w_env
            w0 = LAM * rmag / (np.pi * w_env)
            z_r = np.pi * w0 * w0 / LAM
            s_law = (m / math.sqrt(ext_eff * ext_eff - m * m)) * z_r
            s_beam = C._beam_containment_standoff(
                env, -rmag, rmag, LAM, dx, w_env, cen, half)
            assert s_beam <= s_law, (na, ext, s_beam, s_law)
            assert s_beam == pytest.approx(s_law, rel=2e-2), (na, ext)

    def test_the_containment_guard_refuses_the_pre_fix_landing(self):
        """The guard is the second, independent half of the fix: it measures
        the beam that ACTUALLY landed.  Fed the pre-fix leg it must refuse the
        rows whose peak collapsed.

        FLOOR (``_FOCUS_READOUT_CONTAINMENT_MIN`` = 1.0 beam radii of
        co-moving half-width): measured containment on those rows is 0.909 and
        0.863 -- 1.10x and 1.16x under the floor -- while the narrowest leg
        the resolver ever DELIBERATELY chooses across its own 60-cell
        calibration matrix measures 1.257, i.e. 1.26x over it.  That two-sided
        clearance is ~1.2x, not decades, and is why the floor is a refusal of
        the unfittable rather than a quality bar."""
        e_phys, r0, dx, w0 = _c1_fixture()
        z = -r0
        kw = dict(dx_out=w0 / 8.0, N_out=64, on_replica='ignore')
        for frac in (0.95, 0.90):
            env = C.carrier_referenced_envelope(e_phys, frac * r0, LAM, dx)
            s_old = _pre_fix_standoff(env, frac * r0, z, dx)
            with pytest.raises(RuntimeError, match='does not fit the co-moving'):
                C.carrier_referenced_focus_readout(
                    env, frac * r0, z, LAM, dx, standoff=s_old, **kw)
            # 'warn' still returns the (bad) field, 'ignore' is silent
            with pytest.warns(RuntimeWarning, match='does not fit the co-moving'):
                out = C.carrier_referenced_focus_readout(
                    env, frac * r0, z, LAM, dx, standoff=s_old,
                    on_focus_containment='warn', **kw)
            assert np.isfinite(out).all()
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                C.carrier_referenced_focus_readout(
                    env, frac * r0, z, LAM, dx, standoff=s_old,
                    on_focus_containment='ignore', **kw)
            assert not [x for x in w if 'co-moving' in str(x.message)]

    def test_the_guard_is_silent_on_the_whole_shipped_matrix(self):
        """No false positives where the resolver's premise holds: 60 cells of
        its own calibration matrix, default dispositions, zero warnings.
        Measured worst containment there 1.257 against the 1.0 floor."""
        rmag, n = 20e-3, 512
        worst = np.inf
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            for na in (0.03, 0.05, 0.10, 0.15, 0.278, 0.35):
                for ext in (1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0, 6.0, 10.0):
                    wb = na * rmag
                    dx = 2.0 * ext * wb / n
                    env = _gauss(n, dx, wb).astype(complex)
                    pd = {}
                    C.carrier_referenced_focus_readout(
                        env, -rmag, rmag, LAM, dx,
                        dx_out=(LAM * rmag / (np.pi * wb)) / 8.0, N_out=32,
                        on_replica='ignore', _period_out=pd)
                    worst = min(worst, pd['containment'])
        assert worst > C._FOCUS_READOUT_CONTAINMENT_MIN, worst
        assert not [x for x in w if 'co-moving' in str(x.message)
                    or 'stop plane' in str(x.message)]

    def test_the_window_energy_diagnostic_is_bounded_by_one(self):
        """A Bluestein window is a SUB-window of one period, so its power
        cannot exceed the stop plane's; above 1 the transform folded replicas
        in.  Measured across the same 60 cells: worst 0.999863, i.e. 1.0e-4
        under 1 and 1.0e-2 under the
        ``1 + _FOCUS_READOUT_WINDOW_ENERGY_TOL`` tripwire."""
        rmag, n = 20e-3, 512
        worst = -np.inf
        for na in (0.03, 0.10, 0.35):
            for ext in (1.5, 2.5, 4.0, 10.0):
                wb = na * rmag
                dx = 2.0 * ext * wb / n
                env = _gauss(n, dx, wb).astype(complex)
                pd = {}
                C.carrier_referenced_focus_readout(
                    env, -rmag, rmag, LAM, dx,
                    dx_out=(LAM * rmag / (np.pi * wb)) / 8.0, N_out=32,
                    on_replica='ignore', _period_out=pd)
                worst = max(worst, pd['window_energy_frac'])
        assert worst <= 1.0 + 1e-6, worst

    def test_the_matched_readout_still_matches_the_analytic_waist(self):
        """Nothing the fix touches may move the case the audit verified
        correct.  ORACLE: the analytic focal amplitude ``exp(-r^2/w0^2)`` with
        ``w0 = lambda|R|/(pi w)``.  Measured relL2 4.728e-05 and radius ratio
        1.000016; bars at 5x and 1e-3, against a fixture whose own grid
        truncation floor is ~1e-7."""
        rmag, n, na, ext = 20e-3, 512, 0.05, 4.0
        wb = na * rmag
        dx = 2.0 * ext * wb / n
        env = _gauss(n, dx, wb).astype(complex)
        w0 = LAM * rmag / (np.pi * wb)
        n_out, dx_out = 64, w0 / 8.0
        f = C.carrier_referenced_focus_readout(env, -rmag, rmag, LAM, dx,
                                               dx_out=dx_out, N_out=n_out)
        u = (np.arange(n_out) - n_out / 2) * dx_out
        orc = np.exp(-(u[None, :] ** 2 + u[:, None] ** 2) / w0 ** 2)
        a = np.abs(f) / np.abs(f).max()
        assert float(np.linalg.norm(a - orc) / np.linalg.norm(orc)) < 2.5e-4
        assert (C._envelope_amp_radius(f, dx_out, dx_out) / w0
                == pytest.approx(1.0, abs=1e-3))


# ===========================================================================
# C2 -- the fit radius was taken about the GRID ORIGIN
# ===========================================================================
class TestC2FitRadiusCentre:
    """``carrier_referenced_fit_radius`` fitted the parabola about the grid
    origin.  Because the estimator is a moment, a beam decentred by ``x0``
    carrying a perfect parabola about its OWN centre read
    ``R*(1 + 2 x0^2/w^2)`` (measured 1.5x / 3.0x / 9.0x at 0.5 / 1 / 2 waists)
    and a FLAT wavefront carrying only a uniform tilt read a finite radius
    (measured R = 75.0 mm for L = 0.002 at x0 = 50 um; truth ``inf``)."""

    N, DX, W, R = 512, 2e-6, 100e-6, 50e-3

    def _decentred_parabola(self, x0):
        g = (np.arange(self.N, dtype=np.float64) - self.N / 2) * self.DX
        xx = g[None, :] - x0
        yy = g[:, None]
        r2 = xx * xx + yy * yy
        return np.exp(-r2 / self.W ** 2) * np.exp(1j * K0 * r2 / (2.0 * self.R))

    def _decentred_tilt(self, x0, tilt):
        g = (np.arange(self.N, dtype=np.float64) - self.N / 2) * self.DX
        xx = g[None, :] - x0
        yy = g[:, None]
        amp = np.exp(-(xx * xx + yy * yy) / self.W ** 2)
        return amp * np.exp(1j * K0 * tilt * xx)

    @pytest.mark.parametrize('waists,pre_ratio', [(0.5, 1.5), (1.0, 3.0),
                                                  (2.0, 9.0)])
    @pytest.mark.parametrize('est', ['increment', 'gradient'])
    def test_a_decentred_beam_reads_its_own_radius(self, waists, pre_ratio,
                                                   est):
        """ORACLE: the radius is known BY CONSTRUCTION (the field is built from
        ``exp(i k (r-r0)^2/2R)``), and the analytic pre-fix error is
        ``1 + 2 x0^2/w^2`` -- 1.5 / 3.0 / 9.0 here, measured to 4 digits.

        BAR 2e-3: 'increment' measures 1.000000 post-fix at every decentre and
        'gradient' 1.000200, the latter being the documented amplitude-
        curvature bias ``0.5 (dx/w)^2 = 0.5*(2/100)^2 = 2.0e-4`` of the central
        difference, which is a GRID artefact and not a centring one.  The bar
        is 10x over that residual and 250x-4500x under the pre-fix errors."""
        e = self._decentred_parabola(waists * self.W)
        got = C.carrier_referenced_fit_radius(
            e, LAM, self.DX, estimator=est, on_aliased='silent')
        assert got / self.R == pytest.approx(1.0, abs=2e-3), (waists, est, got)
        # the pre-fix answer, reproduced through the escape hatch
        old = C.carrier_referenced_fit_radius(
            e, LAM, self.DX, estimator=est, on_aliased='silent',
            centre='origin')
        assert old / self.R == pytest.approx(pre_ratio, rel=2e-3)

    @pytest.mark.parametrize('tilt,x0,pre_R', [(0.002, 50e-6, 0.0750),
                                               (0.020, 50e-6, 0.0075),
                                               (0.020, 200e-6, 0.01125)])
    @pytest.mark.parametrize('est', ['increment', 'gradient'])
    def test_a_decentred_pure_tilt_is_not_curvature(self, tilt, x0, pre_R,
                                                    est):
        """ORACLE: a uniform tilt on a flat wavefront has ``R = inf`` by
        construction, and the analytic pre-fix reading is
        ``1/R_fit = L x0/(x0^2 + 2 sigma^2)`` -- 75.0 / 7.50 / 11.25 mm here.

        BAR |R_fit| > 1e6 m: post-fix the worst of these six cases measures
        1.18e8 m, i.e. 118x over the bar, while the pre-fix values are
        1.3e7x-1.5e8x UNDER it.  An absolute bar (rather than a ratio to
        ``inf``) is the only falsifiable form for a quantity whose truth is
        infinite; 1e6 m is 5e7 beam radii of radius, i.e. flat to 2e-8 waves
        of sag across the beam."""
        e = self._decentred_tilt(x0, tilt)
        got = C.carrier_referenced_fit_radius(
            e, LAM, self.DX, estimator=est, on_aliased='silent')
        assert abs(got) > 1e6, (tilt, x0, est, got)
        old = C.carrier_referenced_fit_radius(
            e, LAM, self.DX, estimator=est, on_aliased='silent',
            centre='origin')
        # 'increment' reads the analytic pre-fix value exactly; 'gradient'
        # carries its own documented 0.5 (dx/w)^2 = 2e-4 central-difference
        # bias plus the sin(h)/h term, measured 0.6 % here.
        assert abs(old) == pytest.approx(
            pre_R, rel=(2e-3 if est == 'increment' else 1.5e-2))

    def test_the_on_axis_answer_is_byte_identical(self):
        """The centroid sub-pixel-snaps to exactly ``(0, 0)``, so a centred
        field must take the historical origin arithmetic BIT for bit -- the
        new default changes no on-axis answer."""
        for est in ('gradient', 'increment'):
            for r in (50e-3, -20e-3, 1e9, np.inf):
                g = (np.arange(self.N) - self.N / 2) * self.DX
                r2 = g[None, :] ** 2 + g[:, None] ** 2
                ph = (0.0 if np.isinf(r) else K0 * r2 / (2.0 * r))
                e = np.exp(-r2 / self.W ** 2) * np.exp(1j * ph)
                a = C.carrier_referenced_fit_radius(
                    e, LAM, self.DX, estimator=est, on_aliased='silent')
                b = C.carrier_referenced_fit_radius(
                    e, LAM, self.DX, estimator=est, on_aliased='silent',
                    centre='origin')
                assert a == b, (est, r, a, b)
            # ... and the astigmatic pair too
            e = self._decentred_parabola(0.0)
            assert (C.carrier_referenced_fit_radius(
                        e, LAM, self.DX, astigmatic=True, estimator=est,
                        on_aliased='silent')
                    == C.carrier_referenced_fit_radius(
                        e, LAM, self.DX, astigmatic=True, estimator=est,
                        on_aliased='silent', centre='origin'))

    @pytest.mark.parametrize('stride', [1, 2, 4, 8])
    def test_the_diagnostic_stride_does_not_move_the_fit(self, stride):
        """The C1 resolver and guard fit this envelope on EVERY readout, and
        the fit is the most expensive non-FFT reduction on that path (measured
        457 ms at N = 2048).  The internal diagnostic therefore reads every
        ``stride``-th line -- which changes how many samples the weighted
        average runs over but NOT the pitch along the differenced axis, on
        which the increment estimator's exactness depends.

        BARS.  ``stride == 1`` must be BIT-identical (it is the same arrays).
        Above that, two fixtures with two derived bars:

        * a SMOOTH aberrated envelope (curvature + 3 rad of r^4 + coma +
          astigmatism, decentred by 0.6 waists) -- 1e-4 relative.  An envelope
          is by construction the smooth residual (the carrier holds all the
          fast phase), so this is the representative case: measured 6.5e-9
          centred, 1.6e-5 on THIS fixture and 1.8e-4 at a 1.5-waist decentre,
          all at stride 8, so the bar is 6x over this fixture's measurement.
        * the same envelope with 30 % PER-PIXEL UNCORRELATED amplitude noise
          -- 1e-2 relative.  White noise is the adversarial limit (striding
          then samples a different draw) and is not an envelope; measured
          worst 2.6e-3 at stride 8, so the bar is ~4x over it.

        Why 0.3 % is harmless where it lands: the fit supplies ``1/R_env`` in
        ``1/R_eff = 1/R + 1/R_env``, and it is the SMALL term -- on the C1
        fixture 0.3 % of it is 0.03 % of ``1/R_eff`` -- against a standoff
        plateau on which the derivation's own matrix has M = 2.8..3.6 all
        landing inside 6.1e-3..9.2e-3 of readout error."""
        n, dx = 512, 4e-6
        g = (np.arange(n, dtype=np.float64) - n / 2) * dx
        xx, yy = g[None, :], g[:, None]
        r2 = xx * xx + yy * yy
        w = 0.4e-3
        phase = (K0 * r2 / (2.0 * 0.2) + 3.0 * (r2 / w ** 2) ** 2
                 + 2.5 * xx * r2 / w ** 3
                 + 2.0 * (xx * xx - yy * yy) / w ** 2)
        smooth = np.exp(-((xx - 0.6 * w) ** 2 + yy * yy) / w ** 2)
        rng = np.random.default_rng(23)
        noisy = smooth * (1.0 + 0.3 * rng.standard_normal((n, n)))
        for amp, bar in ((smooth, 1e-4), (noisy, 1e-2)):
            e = amp * np.exp(1j * phase)
            cen = C._envelope_amp_centroid(e, dx, dx)
            full = C._fit_carrier_inv(e, LAM, dx, dx, estimator='increment',
                                      centre=cen)
            got = C._fit_carrier_inv(e, LAM, dx, dx, estimator='increment',
                                     centre=cen, stride=stride)
            if stride == 1:
                assert got == full
            else:
                assert got == pytest.approx(full, rel=bar), (stride, bar)
        # the stride schedule keeps <= 512-line grids on stride 1, so no
        # shipped fit at or under 512^2 changes at all
        assert C._fit_carrier_diag_stride((512, 512)) == 1
        assert C._fit_carrier_diag_stride((1024, 1024)) == 2
        assert C._fit_carrier_diag_stride((2048, 2048)) == 4
        # and it is refused where it would change the arithmetic
        with pytest.raises(ValueError, match="only defined for estimator"):
            C._fit_carrier_inv(e, LAM, dx, dx, estimator='gradient', stride=2)

    def test_an_explicit_centre_is_honoured_and_validated(self):
        e = self._decentred_parabola(1.0 * self.W)
        good = C.carrier_referenced_fit_radius(
            e, LAM, self.DX, estimator='increment', on_aliased='silent',
            centre=(self.W, 0.0))
        assert good / self.R == pytest.approx(1.0, abs=2e-3)
        for bad in ('middle', 3.0):
            with pytest.raises(ValueError, match='centre must be'):
                C.carrier_referenced_fit_radius(
                    e, LAM, self.DX, on_aliased='silent', centre=bad)
        with pytest.raises(ValueError, match='centre components must be'):
            C.carrier_referenced_fit_radius(
                e, LAM, self.DX, on_aliased='silent', centre=(np.nan, 0.0))


# ===========================================================================
# C3 -- complex64 chains were promoted to complex128
# ===========================================================================
class TestC3Complex64:

    def test_a_numpy_complex_scalar_is_what_promoted(self):
        """The mechanism, stated as an executable fact rather than asserted in
        prose: under NEP 50 a numpy complex128 SCALAR is strong and promotes a
        complex64 array, while a Python ``complex`` is weak and does not.  If
        numpy ever changes this, the fixes below stop being needed and this
        test says so first."""
        a = np.ones((4, 4), dtype=np.complex64)
        assert (a * np.exp(1j * 0.3)).dtype == np.complex128
        assert (a * complex(np.exp(1j * 0.3))).dtype == np.complex64

    def test_a_carrier_leg_keeps_complex64(self):
        n, dx = 128, 2e-6
        env = _gauss(n, dx, 40e-6).astype(np.complex64)
        for r in (np.inf, 50e-3, -20e-3):
            out = C.propagate_carrier_referenced(env, r, 1e-3, LAM, dx)
            assert np.asarray(out.env).dtype == np.complex64, r

    def test_the_carrier_field_layer_keeps_complex64(self):
        """Pre-fix, ``CarrierField.full_field()`` and ``from_full_field`` both
        returned complex128 for a complex64 envelope because ``phasor_on``
        hard-coded the whole-grid complex128 build (measured with
        CARRIER/p7_odd_dtype.py).  At N = 16384 each avoidable complex128 grid
        is 4.29 GB."""
        n, dx = 128, 2e-6
        g = FieldGrid((n, n), dx)
        car = CarrierSpec(R=-20e-3, tilt=(0.01, -0.005))
        f = CarrierField(_gauss(n, dx, 40e-6).astype(np.complex64), g, car, LAM)
        assert f.full_field().dtype == np.complex64
        assert car.phasor_on(g, LAM, dtype=np.complex64).dtype == np.complex64
        assert car.phasor_on(g, LAM).dtype == np.complex128      # default kept
        back = CarrierField.from_full_field(f.full_field(), g, car, LAM)
        assert back.envelope.dtype == np.complex64

    def test_the_c64_phasor_is_the_narrowed_c128_one(self):
        """PRECISION BOUNDARY: the ARGUMENT stays float64 and only the finished
        unit phasor is narrowed, so the complex64 build must agree with the
        complex128 one to ONE float32 rounding.  Bar 1e-6: eps32 = 1.2e-7, and
        the measured difference on this fixture is <= 1.2e-07 -- flat in the
        argument, which is the property that matters (a float32-ARGUMENT build
        reads 9.8e-04 at 3.3e4 rad)."""
        n, dx = 256, 2e-6
        g = FieldGrid((n, n), dx)
        car = CarrierSpec(R=-20e-3, tilt=(0.01, -0.005), piston=3e-3)
        a = car.phasor_on(g, LAM, dtype=np.complex64)
        b = car.phasor_on(g, LAM).astype(np.complex64)
        assert float(np.abs(a - b).max()) <= 1e-6

    def test_the_aggregate_accumulator_follows_the_stored_dtype(self):
        n, dx = 64, 2e-6
        g = FieldGrid((n, n), dx)
        car = CarrierSpec(R=-1e-2)
        env = _gauss(n, dx, 20e-6)
        f64 = CarrierField(env.astype(np.complex64), g, car, LAM)
        f128 = CarrierField(env.astype(np.complex128), g, car, LAM)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            assert CF.aggregate([f64, f64], car, g).field.envelope.dtype \
                == np.complex64
            # any complex128 member widens the sum, as result_type requires
            assert CF.aggregate([f64, f128], car, g).field.envelope.dtype \
                == np.complex128


# ===========================================================================
# C4 -- separable phases, the TF build and the in-place rescale
# ===========================================================================
class TestC4SeparablePhases:
    """The screens are separable and were not built that way: measured 6.9x
    (N = 2048) and 11.5x (N = 4096) on ``_radial_carrier_phase`` at 3.50 ->
    1.00 complex128 grids of peak, 4.9x on ``_tilt_ramp``, and 4.00 -> 2.00
    grids on the exact TF step.  Timings are NOT asserted (S1); what is
    asserted is the pair of properties that make the change safe -- the
    agreement bound, and the ALLOCATION count, which is a property of the code."""

    @pytest.mark.parametrize('n', [64, 65, 256])
    @pytest.mark.parametrize('R', [50e-3, -20e-3])
    @pytest.mark.parametrize('centre', [(0.0, 0.0), (13e-6, -7e-6)])
    def test_radial_phase_matches_the_whole_grid_oracle(self, n, R, centre):
        """ORACLE: the whole-grid ``meshgrid`` + ``np.exp`` expression written
        out here, which is also the shipped fallback behind
        ``_SEPARABLE_CARRIER_PHASE = False``.

        BAR ``4 eps max|arg|``, RELATIVE TO THE SCREEN'S OWN ARGUMENT.  The
        identity is exact in exact arithmetic; the float64 regrouping rounds
        the two half-arguments apart, so the difference scales with the
        argument -- re-measured (VERIFY-A6, 2026-09-12) at 1.02 / 1.27 / 1.59
        times ``eps max|arg|`` from 6.3 rad to 1.6e3 rad and 1.53x at 5.2e5 rad,
        i.e. AT the floor the whole-grid build itself has, not below it.  (The
        original bar here was a FIXED 1e-11, justified as the representation
        noise of the ~1e5-1e6 rad arguments these screens carry; that noise is
        1.1e-11..1.1e-10, so the fixed bar does not hold over the stated range
        -- measured 1.75e-10 at 5.2e5 rad.  The relative form does, with 2.5x
        of headroom, and a regrouping blunder is O(|arg|) -- 15 decades up.)"""
        dx = 2e-6
        x = (np.arange(n, dtype=np.float64) - n / 2) * dx - centre[0]
        y = (np.arange(n, dtype=np.float64) - n / 2) * dx - centre[1]
        Y, X = np.meshgrid(y, x, indexing='ij')
        orc = np.exp(1j * K0 * (X * X + Y * Y) / (2.0 * R))
        arg_max = float(np.abs(K0 * (X * X + Y * Y) / (2.0 * R)).max())
        got = C._radial_carrier_phase((n, n), dx, dx, LAM, R, +1,
                                      centre=centre)
        assert float(np.abs(got - orc).max()) <= (
            4.0 * float(np.finfo(np.float64).eps) * arg_max)

    def test_radial_phase_allocates_one_grid(self):
        """The memory claim, as a deterministic allocation count rather than a
        wall-clock number.  Measured 3.50 -> 1.01 complex128 full grids of
        tracemalloc peak at N = 2048 (234.9 MB -> 67.4 MB); at N = 16384 that
        is 15 GB -> 4.3 GB.  Bar 1.5 grids: the separable build allocates the
        one grid it returns plus two length-N vectors (1 + 2/N grids), and the
        whole-grid build cannot get under 3.0 (meshgrid X, Y, r2, then exp)."""
        n, dx, R = 512, 2e-6, 50e-3
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        out = C._radial_carrier_phase((n, n), dx, dx, LAM, R, +1)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert out.shape == (n, n)
        assert (peak - base) / (16.0 * n * n) < 1.5

    @pytest.mark.parametrize('n', [64, 65, 128])
    def test_tilt_ramp_matches_the_whole_grid_oracle(self, n):
        """Same identity, same argument-relative bar and the same reasoning as
        the radial screen; measured 8.5e-14 at N = 2048 (argument 1.9e3 rad,
        i.e. 0.2x ``eps|arg|`` -- the ramp's argument is LINEAR in the
        coordinate, so the two half-arguments round even more closely than the
        radial screen's)."""
        dx, L, M = 2e-6, 0.03, -0.02
        x = (np.arange(n, dtype=np.float64) - n / 2) * dx - 5e-6
        y = (np.arange(n, dtype=np.float64) - n / 2) * dx + 3e-6
        orc = np.exp(1j * K0 * (L * x[None, :] + M * y[:, None]))
        arg_max = float(np.abs(K0 * (L * x[None, :]
                                     + M * y[:, None])).max())
        got = C._tilt_ramp((n, n), dx, LAM, L, M, 5e-6, -3e-6, +1)
        assert float(np.abs(got - orc).max()) <= (
            4.0 * float(np.finfo(np.float64).eps) * arg_max)
        assert C._tilt_ramp((n, n), dx, LAM, 0.0, 0.0, 0.0, 0.0, +1) is None

    def test_rereference_matches_the_whole_grid_oracle(self):
        """ORACLE: the difference-of-parabolas screen written out here.

        BAR 1e-12 relative: measured 2.5e-14 on this fixture.  The quantity is
        a unit-modulus screen applied to an O(1) envelope, so relative and
        absolute agree; the bar is 40x over the measurement and 4 decades
        under the 1e-8 at which a re-referenced envelope stops being flat."""
        n, dx = 256, 2e-6
        rng = np.random.default_rng(3)
        env = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
        g = (np.arange(n, dtype=np.float64) - n / 2) * dx
        Y, X = np.meshgrid(g, g, indexing='ij')
        r_old, r_new = (0.05, 0.05), (0.08, -0.03)
        dphi = 0.5 * K0 * ((X * X) * (1 / r_old[0] - 1 / r_new[0])
                           + (Y * Y) * (1 / r_old[1] - 1 / r_new[1]))
        orc = env * np.exp(1j * dphi)
        got = C._rereference(env, r_old, r_new, LAM, dx, dx)
        assert (float(np.abs(got - orc).max()) / float(np.abs(orc).max())
                < 1e-12)
        # a collimated re-reference is still an exact no-op, not a NaN
        assert np.array_equal(
            C._rereference(env, (np.inf, np.inf), (np.inf, np.inf),
                           LAM, dx, dx), env)

    def test_the_separable_flag_restores_the_whole_grid_build(self):
        """The fail-before switch, exactly as ``_EXACT_READOUT_SEPARABLE_
        BLUESTEIN`` is for the readout: flipping it must change the answer (or
        the flag is not the switch it claims to be) and restore the historical
        build."""
        n, dx, R = 128, 2e-6, 50e-3
        old = C._SEPARABLE_CARRIER_PHASE
        try:
            C._SEPARABLE_CARRIER_PHASE = True
            a = C._radial_carrier_phase((n, n), dx, dx, LAM, R, +1)
            C._SEPARABLE_CARRIER_PHASE = False
            b = C._radial_carrier_phase((n, n), dx, dx, LAM, R, +1)
        finally:
            C._SEPARABLE_CARRIER_PHASE = old
        g = (np.arange(n, dtype=np.float64) - n / 2) * dx
        Y, X = np.meshgrid(g, g, indexing='ij')
        # the byte-identity arm keeps the SHIPPED association verbatim
        # ((1j*k)*r2)/(2R) -- regrouping it as 1j*(k*r2/(2R)) moves the last
        # bits and the equality is the point of this arm
        assert np.array_equal(
            b, np.exp(1j * K0 * (X * X + Y * Y) / (2.0 * R)))
        assert not np.array_equal(a, b)      # the flag is live
        arg_max = float(np.abs(K0 * (X * X + Y * Y) / (2.0 * R)).max())
        assert float(np.abs(a - b).max()) <= (
            4.0 * float(np.finfo(np.float64).eps) * arg_max)


class TestC4TransferFunction:

    @staticmethod
    def _tf_oracle(E, z_eff, dx, dy, tilt):
        """The SHIPPED-before-C4 arithmetic, verbatim: whole-grid ``ax``/``ay``
        and ``np.exp(1j*phase)``.  Independent of the code under test."""
        from lumenairy.propagators.fft_infra import _fft2, _ifft2
        E = np.ascontiguousarray(E, dtype=np.complex128)
        ny, nx = E.shape[-2], E.shape[-1]
        k = K0
        kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
        ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=(dy if dy else dx))
        L, M = float(tilt[0]), float(tilt[1])
        s2 = L * L + M * M
        nz = float(np.sqrt(1.0 - s2))
        KX, KY = kx[None, :], ky[:, None]
        ax, ay = k * L + KX, k * M + KY
        rad = k * k - (ax * ax + ay * ay)
        np.maximum(rad, 0.0, out=rad)
        root = np.sqrt(rad)
        root0 = float(np.sqrt(max(k * k * (1.0 - s2), 0.0)))
        lin = (L * KX + M * KY) / nz
        phase = (k * z_eff) + z_eff * (root - root0 + lin)
        return _ifft2(_fft2(E) * np.exp(1j * phase)).copy()

    @pytest.mark.parametrize('n', [63, 64, 65, 128, 256])
    @pytest.mark.parametrize('tilt', [(0.0, 0.0), (0.03, -0.02)])
    def test_the_tf_step_is_bit_identical(self, n, tilt):
        """The untilted fast path re-associates by ADDITION and MULTIPLICATION
        only, both commutative to the bit in IEEE-754, and ``cos``/``sin`` into
        ``H.real``/``H.imag`` is what ``np.exp`` of a pure-imaginary argument
        computes.  So the bar is not a tolerance: it is EQUALITY of the raw
        bytes, measured 0.000e+00 at every shape and both tilts."""
        rng = np.random.default_rng(11)
        e = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        got = C._exact_envelope_tf_step(e, 5e-3, LAM, 2e-6, 2e-6, tilt=tilt)
        orc = self._tf_oracle(e, 5e-3, 2e-6, 2e-6, tilt)
        assert np.array_equal(got.view(np.float64), orc.view(np.float64))

    def test_the_tf_step_allocates_fewer_grids(self):
        """Measured 4.00 -> 2.00 complex128 full grids of tracemalloc peak at
        N = 2048 (268.5 MB -> 134.2 MB); at N = 16384 that is 17 GB -> 8.6 GB.
        Bar 3.0 grids: the new path allocates ``phase`` (0.5), ``H`` (1) and
        the returned copy (1), so 2.5 is its ceiling with the float64 phase
        counted, while the oracle's extra complex128 ``1j*phase`` puts it at
        4.0.  The bar sits between, 1.5x over the measurement."""
        n = 512
        e = _gauss(n, 2e-6, 200e-6).astype(np.complex128)
        C._exact_envelope_tf_step(e, 5e-3, LAM, 2e-6, 2e-6)   # warm FFT plans
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        C._exact_envelope_tf_step(e, 5e-3, LAM, 2e-6, 2e-6)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert (peak - base) / (16.0 * n * n) < 3.0

    def test_tf_phase_to_H_is_bit_identical_to_the_exponential(self):
        rng = np.random.default_rng(5)
        for shape in ((129,), (65, 127), (256, 256)):
            arg = rng.standard_normal(shape) * 1e5
            got = C._tf_phase_to_H(arg, np.complex128, np, False, np)
            assert np.array_equal(got.view(np.float64),
                                  np.exp(1j * arg).view(np.float64))

    def test_the_fine_crop_rescale_is_in_place_and_unchanged(self):
        """``out *= scale`` instead of ``out = out * scale`` removes one FULL
        FINE-GRID temporary (4.29 GB at the shipped ``n_fine_cap = 16384``).
        The product is elementwise, so the VALUES are unchanged; the oracle is
        a raw-numpy transform written here.

        BAR 1e-12 relL2: measured 5.3e-16, and the gap to the bar is the FFT
        backend difference (pyFFTW vs pocketfft) this function's own note
        bounds, not the rescale."""
        n = 128
        rng = np.random.default_rng(13)
        e = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        n_crop, n_fine = 64, 256
        got = C._fourier_upsample_crop(e, n_crop, n_fine)
        c0 = n // 2 - n_crop // 2
        ec = np.ascontiguousarray(e[c0:c0 + n_crop, c0:c0 + n_crop])
        f = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(ec)))
        pad = np.zeros((n_fine, n_fine), complex)
        o = n_fine // 2 - n_crop // 2
        pad[o:o + n_crop, o:o + n_crop] = f
        orc = (np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(pad)))
               * (float(n_fine) / float(n_crop)) ** 2)
        assert (float(np.linalg.norm(got - orc) / np.linalg.norm(orc))
                < 1e-12)


# ===========================================================================
# C5 -- the P3 cluster
# ===========================================================================
class TestC5SmallerItems:

    def test_the_dead_frequency_builder_is_gone(self):
        """``_freq_sq_1d`` had NO call sites and still carried the ``- N/2``
        offset that defect D7 fixed everywhere else, while
        ``_freq_sq_1d_bld``'s docstring asserted the two agreed.  Measured at
        N = 5 they did not: ``[9.87 3.55 0.39 0.39 3.55]`` against
        ``[6.32 1.58 0. 1.58 6.32]``."""
        assert not hasattr(C, '_freq_sq_1d')

    @pytest.mark.parametrize('n', [1, 4, 5, 7, 64, 65])
    def test_the_surviving_builders_are_fftfreq_at_both_parities(self, n):
        """ORACLE: ``numpy.fft.fftfreq`` itself, for BOTH parities -- the
        property the deleted twin violated (its ``- N/2`` offset made
        ``ifftshift`` of its axis a half-integer shift of ``fftfreq`` at odd
        ``N``, so every transfer function built on it multiplied the wrong
        spectral bin; measured relL2 1.239 at N = 65 on ``_exact_tf_2d_xp``).

        BAR 4 ulp relative: the two are the same integers over the same
        ``N*d``, differing only in that ``fftfreq`` multiplies by the
        reciprocal where these divide -- one rounding, and exactly zero
        whenever ``1/(N d)`` is representable (N = 1, 4, 5, 64 here).
        Measured worst 1 ulp.  A structural error is O(1), 15 decades up."""
        d = 2e-6
        np.testing.assert_allclose(
            np.fft.ifftshift(C._freq_sq_1d_bld(n, d, np)),
            (2.0 * np.pi * np.fft.fftfreq(n, d)) ** 2,
            rtol=4 * np.finfo(np.float64).eps, atol=0.0)
        np.testing.assert_allclose(
            np.fft.ifftshift(C._freq_1d_bld(n, d, np)),
            2.0 * np.pi * np.fft.fftfreq(n, d),
            rtol=4 * np.finfo(np.float64).eps, atol=0.0)

    @pytest.mark.parametrize('n', [64, 65, 127])
    def test_the_asm_axis_band_limit_is_in_register_at_odd_n(self, n):
        """``H`` is built on the ``N//2`` axis but the Matsushima band-limit
        mask used ``N/2``, so at ODD ``N`` the mask sat half a bin out of
        register with the transfer function it masks (measured 3.85e3 1/m at
        N = 65, dx = 2 um): one bin too many kept on one side, one too many
        dropped on the other.

        ORACLE: the 1-D band-limited angular spectrum built by hand here on
        ``fftfreq``.  Bar: relL2 < 1e-14 -- the two expressions are the same
        arithmetic, measured 0.000e+00 at all three parities; pre-fix N = 65
        and N = 127 differed at O(1) on the two mis-masked bins."""
        d, z = 2e-6, 1e-3
        e = np.zeros((n, n), complex)
        e[n // 2, n // 2] = 1.0
        f = (np.arange(n, dtype=np.float64) - (n // 2)) / (n * d)
        kz_sq = K0 ** 2 - (2.0 * np.pi * f) ** 2
        prop = kz_sq > 0
        kz = np.where(prop, np.sqrt(np.maximum(kz_sq, 0.0)), 0.0)
        h = np.where(prop, np.exp(1j * z * kz), 0.0)
        h = np.where(np.abs(f) < (n * d) / (2.0 * LAM * abs(z)), h, 0.0)
        orc = np.fft.ifft(np.fft.fft(e, axis=1)
                          * np.fft.ifftshift(h)[None, :], axis=1)
        got = C._asm_axis(e, z, LAM, d, axis=1, bandlimit=True)
        assert (float(np.linalg.norm(got - orc) / np.linalg.norm(orc))
                < 1e-14)

    def test_the_sphere_parabola_conversion_honours_dy(self):
        """``_sphere_parab_conversion`` built its y axis on ``dx``, so a
        non-square-pixel call -- which ``propagate_carrier_referenced`` and
        ``carrier_referenced_reconstruct`` both accept -- converted the y axis
        against the wrong pitch.  Every shipped call site is square, so this
        was latent; the square call must stay BIT-identical."""
        sh, dx, r = (64, 96), 2e-6, -20e-3
        a = C._sphere_parab_conversion(sh, dx, LAM, r, +1)
        assert np.array_equal(
            a, C._sphere_parab_conversion(sh, dx, LAM, r, +1, dy=dx))
        dy = 3.0 * dx
        got = C._sphere_parab_conversion(sh, dx, LAM, r, +1, dy=dy)
        x = (np.arange(sh[1], dtype=np.float64) - sh[1] / 2) * dx
        y = (np.arange(sh[0], dtype=np.float64) - sh[0] / 2) * dy
        r2 = x[None, :] ** 2 + y[:, None] ** 2
        s = np.sign(r) * (r2 / (np.sqrt(r2 + r * r) + abs(r)))
        orc = np.exp(1j * K0 * (s - r2 / (2.0 * r)))
        assert np.array_equal(got, orc)          # same arithmetic, exactly
        # and it really is a different answer from the dy-blind one
        assert float(np.abs(got - a).max()) > 1e-6

    def test_the_readout_forwards_the_gap_kernel_and_the_tilt(self):
        """The readout's internal carrier leg took its OWN ``'auto'`` default,
        so a chain asked for ``gap_kernel='fresnel'`` -- whose documented
        purpose is to be pinned FP-identical to prior releases -- got a MIXED
        chain; and the readout had no ``tilt`` at all, so a tilted congruence's
        final leg ran an untilted kernel while the chain's own documentation
        claimed the exact kernel carried the tilt to all orders.

        Both are pinned COMPARATIVELY: the argument must CHANGE the answer (or
        it is not reaching the kernel) and the change must be small enough to
        be the kernel difference and not a blunder.  Measured here: the
        fresnel-vs-exact relL2 is 5e-6..1e-2 and the tilt one 1e-4..1e-1,
        against 0 for an argument that never arrives."""
        n, dx, w, rmag = 512, 2e-6 * 4, 200e-6, 20e-3
        env = _gauss(n, dx, w).astype(complex)
        kw = dict(dx_out=(LAM * rmag / (np.pi * w)) / 8.0, N_out=48,
                  on_replica='ignore')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            a = C.carrier_referenced_focus_readout(
                env, -rmag, rmag, LAM, dx, gap_kernel='exact', **kw)
            b = C.carrier_referenced_focus_readout(
                env, -rmag, rmag, LAM, dx, gap_kernel='fresnel', **kw)
            c = C.carrier_referenced_focus_readout(
                env, -rmag, rmag, LAM, dx, tilt=(0.03, -0.02), **kw)
        assert not np.array_equal(a, b)
        assert float(np.linalg.norm(a - b) / np.linalg.norm(a)) < 0.2
        assert not np.array_equal(a, c)
        assert float(np.linalg.norm(a - c) / np.linalg.norm(a)) < 0.5
        # 'auto' still resolves to 'exact' -- the default is unchanged
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            assert np.array_equal(a, C.carrier_referenced_focus_readout(
                env, -rmag, rmag, LAM, dx, gap_kernel='auto', **kw))

    def test_mutating_a_built_carrier_field_is_announced(self):
        """``CarrierField`` is the one MUTABLE dataclass among frozen siblings
        (``CarrierSpec``, ``FieldGrid``), so an assignment bypasses every
        ``__post_init__`` invariant.  Pre-fix each of these was SILENT and
        could leave a field whose grid no longer described its array.

        The hard freeze the audit asks for is scheduled rather than taken in
        one step because a live consumer accumulates in place through the
        attribute (``validation/pipeline/driver.py``): this is the
        announcement half, so the assignment still works and the invariant
        bypass is no longer silent.  ``dataclasses.FrozenInstanceError`` is
        what it becomes at ``_CARRIER_FIELD_FROZEN_IN``."""
        n, dx = 32, 2e-6
        f = CarrierField(_gauss(n, dx, 8e-6).astype(complex),
                         FieldGrid((n, n), dx), CarrierSpec(R=-1e-2), LAM)
        with pytest.warns(DeprecationWarning, match='built CarrierField'):
            f.wavelength = 1.55e-6
        assert f.wavelength == 1.55e-6          # ... and it still takes effect
        # construction itself must be silent, or every caller drowns
        with warnings.catch_warnings():
            warnings.simplefilter('error', DeprecationWarning)
            CarrierField(_gauss(n, dx, 8e-6).astype(complex),
                         FieldGrid((n, n), dx), CarrierSpec(R=-1e-2), LAM)
        # the siblings ARE already hard-frozen -- the convention this joins
        for sib in (CarrierSpec(R=-1e-2), FieldGrid((n, n), dx)):
            assert dataclasses.fields(sib)
            with pytest.raises(dataclasses.FrozenInstanceError):
                sib.R = 1.0 if isinstance(sib, CarrierSpec) else None

    def test_the_backend_twins_agree_with_the_numpy_paths(self):
        """COMMON §9: every NumPy path changed here that has a backend twin is
        checked against it.  x64 is enabled so the comparison is
        precision-matched (the established convention in
        ``test_niche_k2_carrier_backends``); CuPy is not installed on this
        machine and is desk-checked instead -- the separable builds use only
        ``bld.exp`` and broadcasting, and the two host-side reductions the C1
        guard adds pull through ``backend.to_numpy`` rather than
        ``np.asarray``, which is what a CuPy device array requires.

        BAR 1e-12 relative: these are the same float64 arithmetic on two
        implementations of the same primitives, so the expectation is
        round-off; measured 3.5e-16 on the screens, 0.0 on ``_rereference``
        and 4.7e-16 .. 5.6e-16 on a whole carrier leg.  A genuine backend
        divergence (a screen built on the wrong grid, a dtype collapse) is
        O(1), 12 decades up."""
        jax = pytest.importorskip('jax')
        import jax.numpy as jnp
        jax.config.update('jax_enable_x64', True)
        n, dx, R = 128, 2e-6, 50e-3
        for centre in ((0.0, 0.0), (7e-6, -3e-6)):
            a = C._radial_carrier_phase((n, n), dx, dx, LAM, R, +1, bld=np,
                                        centre=centre)
            b = np.asarray(C._radial_carrier_phase(
                (n, n), dx, dx, LAM, R, +1, bld=jnp, centre=centre))
            assert float(np.abs(a - b).max()) < 1e-12
        env = _gauss(n, dx, 40e-6).astype(np.complex128)
        a = C._rereference(env, (0.05, 0.05), (0.08, -0.03), LAM, dx, dx)
        b = np.asarray(C._rereference(jnp.asarray(env), (0.05, 0.05),
                                      (0.08, -0.03), LAM, dx, dx,
                                      bld=np, xp=jnp, is_jax=True))
        assert float(np.abs(a - b).max()) / float(np.abs(a).max()) < 1e-12
        for r in (np.inf, 0.05, -20e-3):
            a = C.propagate_carrier_referenced(env, r, 1e-3, LAM, dx)
            b = C.propagate_carrier_referenced(jnp.asarray(env), r, 1e-3,
                                               LAM, dx)
            assert (float(np.abs(np.asarray(b.env) - a.env).max())
                    / float(np.abs(a.env).max()) < 1e-12), r
            assert float(b.dx) == pytest.approx(float(a.dx), rel=1e-14)
        # the odd-N band-limit fix is on the ``bld`` axis, so both arms move
        for nn in (64, 65):
            e = np.zeros((nn, nn), complex)
            e[nn // 2, nn // 2] = 1.0
            a = C._asm_axis(e, 1e-3, LAM, dx, axis=1)
            b = np.asarray(C._asm_axis(jnp.asarray(e), 1e-3, LAM, dx, axis=1))
            assert (float(np.abs(a - b).max()) / float(np.abs(a).max())
                    < 1e-12), nn

    def test_the_gap_kernel_comment_no_longer_contradicts_itself(self):
        """``_carrier_step_fast`` carried two statements four lines apart --
        "'auto' ... resolves by BACKEND" and "'auto' resolves to 'exact'
        everywhere" -- and ``propagate_carrier_referenced`` claimed its fast
        path was byte-identical "on the default gap_kernel='fresnel'" when the
        default is 'auto'.  The behaviour is pinned here so the prose and the
        code cannot drift apart again."""
        n, dx = 64, 2e-6
        env = _gauss(n, dx, 20e-6).astype(complex)
        a = C.propagate_carrier_referenced(env, 50e-3, 1e-3, LAM, dx,
                                           gap_kernel='auto').env
        b = C.propagate_carrier_referenced(env, 50e-3, 1e-3, LAM, dx,
                                           gap_kernel='exact').env
        c = C.propagate_carrier_referenced(env, 50e-3, 1e-3, LAM, dx,
                                           gap_kernel='fresnel').env
        assert np.array_equal(a, b)          # 'auto' IS 'exact', on numpy too
        assert not np.array_equal(a, c)
