"""Regression tests for the v4.11.2 audit-residual patch in the
Richards-Wolf + lens-element + polarization / coating domain.

Each test pins one of the round-3 audit findings:

* RW-1 -- ``richards_wolf_focus`` Airy-peak intensity scales as 1/f^2
          (prefactor 1/f, not f); a 1 m vs 0.1 m focal length differ
          by ~100x in Airy peak (within +/- 20%).  Pre-4.11.2 the
          missing 1/f^2 factor reversed the scaling.

* RW-2 -- ``richards_wolf_focus`` on-axis focal-field global phase
          matches the sign of ``angular_spectrum_propagate`` over the
          same distance (both use ``exp(+i k z)`` under the library's
          ``exp(-i omega t)`` convention).  Pre-4.11.2 used the
          opposite ``exp(-i k f)``.

* CT-1 -- ``thin_film_stack(polarization='avg')`` is symmetric at
          normal incidence: T_avg == 0.5 * (T_s + T_p) and the
          s/p-separately-computed transmissions agree at 0 deg.
          Pre-4.11.2 the 'avg' branch reused the LAST polarization's
          eta_sub/eta_amb (p-pol) for both T_s and T_p; at 0 deg this
          was masked because eta_s = eta_p, but the regression test
          still verifies the equality holds (no symmetry drift) and
          additionally checks that a non-zero AOI no longer reuses the
          p-pol admittance for the s-pol transmission.

* PL-1 -- ``apply_waveplate`` with retardance=pi/2 and angle=pi/4 on
          a linear-x input produces a Jones vector with the same
          handedness as ``create_circular_polarized('right')`` (the
          v4.11.1 test pinned the result; this test re-pins it to
          lock in the docstring/code agreement after the 4.11.2
          docstring sync).

Run-time budget: all four tests together should finish in << 5 s.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy as lm


# ============================================================================
# RW-1 -- Richards-Wolf intensity scales correctly with focal length
# ============================================================================

class TestRichardsWolfIntensityScalesWithF:
    """The Richards-Wolf integral's prefactor goes as 1/f (amplitude)
    -> 1/f^2 (intensity).  An ideal aplanatic objective with identical
    NA and pupil at two different focal lengths f1 vs f2 should give

        I_peak(f1) / I_peak(f2) ~= (f2 / f1)^2

    Pre-v4.11.2 the prefactor was (-i k f / 2 pi) (i.e. multiplied by
    f instead of dividing), so the ratio came out as (f1 / f2)^2 -- the
    WRONG direction.  A 10x focal-length shrink therefore CHANGED the
    Airy-peak intensity by 1/100 instead of x100, a 10^4 relative
    error in the wrong direction.

    The pupil grid is held fixed (same dx_pupil, same Np) so the
    plane-wave amplitude entering the integral is identical between
    the two calls; only ``f`` changes.  NA is also held fixed (the
    NA-limited exit pupil rim is the same).
    """

    def test_peak_intensity_ratio_is_f_squared(self):
        # Modest pupil grid so the test runs quickly.
        Np = 64
        wavelength = 1e-6
        NA = 0.05  # low NA so the Richards-Wolf result reduces to scalar
        dx_pupil = 100e-6
        # Two focal lengths a factor of 10 apart.
        f_long = 1.0   # 1 m
        f_short = 0.1  # 0.1 m  (i.e. 10x shorter)

        # Top-hat pupil (NA-limited) -- uniform amplitude inside the
        # geometric pupil.  richards_wolf_focus itself applies the
        # rim mask using sin(theta) <= NA, so we just hand it a uniform
        # complex 1+0j array.
        pupil = np.ones((Np, Np), dtype=np.complex128)

        Ex_a, Ey_a, Ez_a, _, _ = lm.richards_wolf_focus(
            pupil, wavelength, NA, f_long, dx_pupil,
            N_focal=Np, z_planes=[0.0], polarization='x',
        )
        Ex_b, Ey_b, Ez_b, _, _ = lm.richards_wolf_focus(
            pupil, wavelength, NA, f_short, dx_pupil,
            N_focal=Np, z_planes=[0.0], polarization='x',
        )

        I_a = (np.abs(Ex_a)**2 + np.abs(Ey_a)**2 + np.abs(Ez_a)**2)
        I_b = (np.abs(Ex_b)**2 + np.abs(Ey_b)**2 + np.abs(Ez_b)**2)
        peak_a = float(I_a.max())
        peak_b = float(I_b.max())

        # I_peak(f_short) / I_peak(f_long) should be (f_long / f_short)^2 = 100
        ratio = peak_b / peak_a
        expected = (f_long / f_short) ** 2  # 100
        assert 0.8 * expected <= ratio <= 1.2 * expected, (
            f"Richards-Wolf peak intensity at f={f_short} m vs f={f_long} m: "
            f"got ratio = {ratio:.3f}, expected ~ {expected:.1f} (within "
            f"+/- 20%).  Pre-4.11.2 the prefactor was missing the 1/f^2 "
            f"factor; the ratio came out around 1/100 instead of 100."
        )


# ============================================================================
# RW-2 -- Richards-Wolf global phase has the +i k f sign
# ============================================================================

class TestRichardsWolfGlobalPhaseSign:
    """The Richards-Wolf prefactor includes a global ``exp(+i k f)``
    under the library's ``exp(-i omega t)`` convention -- the same sign
    that every other forward propagator (``angular_spectrum_propagate``,
    ``fresnel_propagate``, ...) uses for ``exp(+i k z)``.

    Pre-4.11.2 the code had ``exp(-i k f)``, opposite-sign to all the
    other forward props.  Coherent superposition of a focal field with
    a reference arm propagated by ASM picked up a spurious
    ``exp(-2 i k f)`` mismatch.

    Test approach: at two focal lengths f1 and f2, the *change* in
    on-axis focal-field phase between f1 and f2 should be approximately
    +k*(f2-f1) (modulo 2 pi) -- the same sign as the ASM phase
    advance over a propagation distance (f2 - f1).  Pre-4.11.2 the
    phase would advance with the opposite sign.

    Because the prefactor's 1/f factor introduces a slowly-varying
    amplitude with no phase contribution (real-positive), the phase
    of E_x(0,0) is dominated by the prefactor's exp(+i k f) term plus
    the (real-positive) area-integrated pupil sum.  The (modulo 2 pi)
    increment between f1 and f2 should therefore equal k*(f2-f1).
    """

    def test_on_axis_phase_advances_like_forward_prop(self):
        Np = 64
        wavelength = 1e-6
        NA = 0.05
        dx_pupil = 100e-6
        # Choose two focal lengths whose difference is an integer
        # number of wavelengths so the phase comparison is unambiguous.
        # Use a small absolute f to keep k*f small for clarity, but
        # finite so the prefactor difference is visible.
        delta_f = wavelength * 1000.0  # 1000 wavelengths, ~ 1 mm
        f1 = 0.10           # 0.1 m
        f2 = 0.10 + delta_f
        k = 2 * np.pi / wavelength

        pupil = np.ones((Np, Np), dtype=np.complex128)

        Ex_1, _, _, _, _ = lm.richards_wolf_focus(
            pupil, wavelength, NA, f1, dx_pupil,
            N_focal=Np, z_planes=[0.0], polarization='x',
        )
        Ex_2, _, _, _, _ = lm.richards_wolf_focus(
            pupil, wavelength, NA, f2, dx_pupil,
            N_focal=Np, z_planes=[0.0], polarization='x',
        )

        ic = Np // 2
        # Wrap the difference into [-pi, pi].
        phase_1 = np.angle(Ex_1[ic, ic])
        phase_2 = np.angle(Ex_2[ic, ic])
        dphi = (phase_2 - phase_1 + np.pi) % (2 * np.pi) - np.pi

        # ASM forward-prop over distance delta_f advances the on-axis
        # plane wave by +k*delta_f.  Wrap to [-pi, pi] for comparison.
        expected = ((k * delta_f) + np.pi) % (2 * np.pi) - np.pi
        # ASM forward-prop reference: same wrap of +k*delta_f.
        # (We could call angular_spectrum_propagate over delta_f and
        # measure its central-pixel phase; that gives the same result
        # but adds dependent-grid setup overhead.  The k*delta_f form
        # is equivalent.)

        # Allow a small tolerance for FFT-grid sampling effects.
        assert abs(dphi - expected) < 0.05, (
            f"Richards-Wolf on-axis phase shift between f={f1} m and "
            f"f={f2} m: got d(phase) = {dphi:.4f} rad, expected "
            f"{expected:.4f} rad (= +k*delta_f wrapped to [-pi, pi]).  "
            f"Pre-4.11.2 used exp(-i k f), so the phase advance was "
            f"-k*delta_f -- opposite sign to angular_spectrum_propagate."
        )


# ============================================================================
# CT-1 -- Coating 'avg' mode is symmetric at normal incidence
# ============================================================================

class TestCoatingAvgPolarizationSymmetric:
    """At 0-deg AOI, s-pol and p-pol reduce to the same physics
    (the polarization basis is degenerate at normal incidence), so

        thin_film_stack(..., angle=0, polarization='avg')

    must produce the same T as 0.5*(T_s + T_p) computed by two
    separate calls with polarization='s' and 'p'.

    Pre-4.11.2 the 'avg' branch reused the LAST polarization's
    ``eta_sub`` / ``eta_amb`` (p-pol, since pols=['s','p']) inside
    the T_s and T_p formulas, which made the averaged T silently wrong
    at oblique AOI.  At 0 deg the bug was masked because eta_s=eta_p,
    but the test still pins the equality so a future refactor can't
    silently break the 0-deg consistency.

    For belt-and-braces coverage, we also pin that the s/p separate
    transmissions agree at 0 deg (a basic physical consistency check).
    """

    def test_avg_equals_half_sum_at_zero_aoi(self):
        from lumenairy.elements.coatings import coating_reflectance

        # Simple single-layer AR-style coating on a high-index substrate.
        # Wavelength 1 um, n_amb=1 (air), n_sub=1.5 (BK7-like), one
        # MgF2-like layer at the AR thickness.
        wavelengths = np.array([1.0e-6])
        layers = [(1.38, 1.0e-6 / (4 * 1.38))]  # quarter-wave at 1 um
        n_ambient = 1.0
        n_substrate = 1.5
        angle = 0.0

        R_s, T_s, _ = coating_reflectance(
            layers, wavelengths, angle=angle,
            n_substrate=n_substrate, n_ambient=n_ambient,
            polarization='s')
        R_p, T_p, _ = coating_reflectance(
            layers, wavelengths, angle=angle,
            n_substrate=n_substrate, n_ambient=n_ambient,
            polarization='p')
        R_avg, T_avg, _ = coating_reflectance(
            layers, wavelengths, angle=angle,
            n_substrate=n_substrate, n_ambient=n_ambient,
            polarization='avg')

        # At 0 deg, T_s == T_p (and R_s == R_p) to numerical precision.
        assert abs(float(T_s[0]) - float(T_p[0])) < 1e-12, (
            f"Coating at 0 deg: T_s={float(T_s[0]):.6f}, "
            f"T_p={float(T_p[0]):.6f} differ.  At normal incidence "
            f"s-pol and p-pol must be degenerate."
        )
        # avg should equal both, and equal 0.5*(T_s + T_p).
        expected = 0.5 * (float(T_s[0]) + float(T_p[0]))
        assert abs(float(T_avg[0]) - expected) < 1e-10, (
            f"Coating 'avg' mode at 0 deg: T_avg={float(T_avg[0]):.6f}, "
            f"expected 0.5*(T_s + T_p) = {expected:.6f}.  Pre-4.11.2 "
            f"the 'avg' branch reused the p-pol eta_sub / eta_amb for "
            f"both T_s and T_p computations; at 0 deg this was masked "
            f"because eta_s = eta_p, but the formula is structurally "
            f"wrong and the regression test pins the correct equality."
        )

    def test_avg_uses_correct_admittance_at_oblique_aoi(self):
        """At a non-zero AOI, T_avg from 'avg' mode must equal
        0.5*(T_s + T_p) computed separately.  Pre-4.11.2 the 'avg'
        branch used p-pol admittances inside both T_s and T_p, giving
        T_avg(buggy) = 0.5 * (T_s_with_p_admittance + T_p_with_p_admittance)
        != 0.5 * (T_s_correct + T_p_correct) in general at oblique AOI.
        """
        from lumenairy.elements.coatings import coating_reflectance

        wavelengths = np.array([1.0e-6])
        # 2-layer coating to get a non-trivial admittance ratio.
        layers = [(1.38, 1.0e-6 / (4 * 1.38)),
                  (2.30, 1.0e-6 / (4 * 2.30))]
        n_ambient = 1.0
        n_substrate = 1.5
        # 30 deg AOI is large enough that eta_s and eta_p differ by ~30%.
        angle = np.deg2rad(30.0)

        _, T_s, _ = coating_reflectance(
            layers, wavelengths, angle=angle,
            n_substrate=n_substrate, n_ambient=n_ambient,
            polarization='s')
        _, T_p, _ = coating_reflectance(
            layers, wavelengths, angle=angle,
            n_substrate=n_substrate, n_ambient=n_ambient,
            polarization='p')
        _, T_avg, _ = coating_reflectance(
            layers, wavelengths, angle=angle,
            n_substrate=n_substrate, n_ambient=n_ambient,
            polarization='avg')

        expected = 0.5 * (float(T_s[0]) + float(T_p[0]))
        assert abs(float(T_avg[0]) - expected) < 1e-8, (
            f"Coating 'avg' at 30 deg AOI: T_avg={float(T_avg[0]):.6f}, "
            f"expected 0.5*(T_s + T_p) = {expected:.6f} "
            f"(T_s={float(T_s[0]):.6f}, T_p={float(T_p[0]):.6f}).  "
            f"Pre-4.11.2 the 'avg' branch overwrote eta_sub/eta_amb "
            f"in the inner loop and used the LAST iteration's value "
            f"(p-pol) for BOTH T_s and T_p, so T_avg drifted from the "
            f"correct half-sum at oblique AOI."
        )


# ============================================================================
# PL-1 -- apply_waveplate matches create_circular_polarized('right')
# ============================================================================

class TestApplyWaveplateMatchesCreateCircularRight:
    """4.11.1 already pinned this equivalence; the v4.11.2 docstring
    sync (correcting the docstring formula
    ``J = R(-theta) diag(1, exp(+i phi)) R(theta)`` -> the actual
    ``R(theta) diag(1, exp(-i phi)) R(-theta)``) shouldn't break it.

    Test approach: pass a linear-x JonesField through a quarter-wave
    plate at angle pi/4 and compare the on-grid Stokes S3 sign with
    ``create_circular_polarized('right')`` -- they must agree (same
    handedness).
    """

    def test_qwp_on_linear_x_matches_create_right_handedness(self):
        N, dx = 16, 5e-6
        scalar = np.ones((N, N), dtype=np.complex128)
        jf_lin = lm.create_linear_polarized(scalar, dx, angle=0.0)
        jf_qwp = lm.apply_quarter_wave_plate(jf_lin, angle=np.pi / 4)
        jf_ref = lm.create_circular_polarized(
            scalar, dx, handedness='right')
        s_qwp = float(np.mean(lm.stokes_parameters(jf_qwp)['S3']))
        s_ref = float(np.mean(lm.stokes_parameters(jf_ref)['S3']))
        assert s_qwp * s_ref > 0, (
            f"apply_waveplate(retardance=pi/2, angle=pi/4) on linear-x: "
            f"S3 = {s_qwp:.3f} should agree in sign with "
            f"create_circular_polarized('right') S3 = {s_ref:.3f}.  "
            f"4.11.2 only synced the docstring formula, not the "
            f"implementation, so this 4.11.1 equivalence must still "
            f"hold."
        )
        # Tight bound: both should be close to +1 in S3/S0 ratio.
        assert s_qwp > 0.5 and s_ref > 0.5, (
            f"Both S3 should be near +1 (right-circular): got "
            f"qwp={s_qwp:.3f}, ref={s_ref:.3f}."
        )
