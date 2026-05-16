"""Pinning tests for the v4.11.2 raytrace / Seidel audit fixes.

The audit (``AUDIT_ROUND3_2026_05_16.md``) identified three correctness
bugs in the ray-trace / Seidel-analysis domain that 4.11.2 fixes:

* **CRIT** Chained-mirror Seidel parity not tracked.  The v4.10 mirror
  fix used ``n2 = -n1`` at the mirror but the next surface
  re-queried ``glass_before='air'`` and got ``n=+1`` instead of
  ``n=-1`` -- so Cassegrain / Schwarzschild / any 2-mirror catadioptric
  produced wrong Seidel sums beyond the first mirror.
* **CRIT** ``system_abcd`` and ``seidel_coefficients`` disagreed on the
  mirror power sign convention.  For a single concave mirror with
  R = -100 mm, ``system_abcd`` returned EFL = -0.05 m while the
  Seidel branch (which uses Welford's ``n2 = -n1``) computed the
  equivalent of +0.05 m.  4.11.2 reconciles both to the +0.05 m
  Welford convention.
* **CRIT** ``seidel_wfe`` was missing the field-curvature DC term
  ``(1/4) * S3 * rho^2``.  Hopkins / Welford eq. 7.11 has BOTH the
  ``(1/2) S3 rho^2 cos^2(theta)`` astigmatism term AND a rotationally
  symmetric ``(1/4) (S3 + S4 H^2) rho^2`` field-curvature DC term;
  pre-4.11.2 the docstring and the code agreed -- both were wrong.

Each test below corresponds to one finding and asserts the magnitude
direction of the fix (not just "non-zero") so a future regression
that reverts to the pre-4.11.2 sign / formula is caught immediately.
"""
from __future__ import annotations

import numpy as np
import pytest

import lumenairy as lm
from lumenairy.raytrace import (
    Surface, seidel_coefficients, seidel_wfe, system_abcd,
)


# ============================================================================
# CRIT-2 -- Chained-mirror Seidel parity
# ============================================================================

class TestChainedMirrorSeidelParity:
    """Two-mirror Cassegrain-style geometry: the second mirror must
    see the post-first-mirror index (n = -1, not n = +1).  Without
    the v4.11.2 parity tracking, Welford's ``S4_i = -(n2-n1) c /
    (n1 n2)`` flips sign on the secondary because n1 is still +1
    instead of -1, doubling the absolute |S4_total| in the wrong
    direction.

    Hand calculation for the two-mirror geometry below (both
    mirrors, n=1 input, in vacuum):

        Primary:  n1=+1, n2=-1, c1=1/R1 = 1/-0.5 = -2 m^-1
            S4_1 = -(1/(n2 n1)) * c1 * (n2 - n1)
                 = -(1/(-1))    * (-2) * (-2)
                 = -(+4)
                 = -4
            ... wait -- let me re-do carefully.
            (1/(n2 n1)) = 1/((-1)(+1)) = -1
            (n2 - n1)   = -1 - 1 = -2
            c1          = -2
            S4_1 = -( -1 * -2 * -2 ) = -( -4 ) = +4

        Secondary: n1=-1, n2=+1, c2=1/R2 = 1/-0.1 = -10 m^-1
            (1/(n2 n1)) = 1/((+1)(-1)) = -1
            (n2 - n1)   = +1 - (-1) = +2
            c2          = -10
            S4_2 = -( -1 * -10 * 2 ) = -( 20 ) = -20

    Total (with parity fix):  S4 = +4 + (-20) = -16
    Total (without parity, pre-fix): the secondary would compute
    with n1=+1 instead of -1, giving
            (1/(n2 n1)) = 1/((-1)(+1)) = -1
            (n2 - n1)   = -1 - 1 = -2
            S4_2_bug    = -( -1 * -10 * -2 ) = -( -20 ) = +20
        Total = +4 + 20 = +24.

    A factor-of-1.5 magnitude AND a sign flip -- a strong fingerprint.
    """

    def _make_two_mirror_system(self):
        R1 = -0.5     # primary radius (concave, |R| = 500 mm)
        R2 = -0.1     # secondary radius (convex toward primary)
        # Thickness from primary to secondary (between-mirrors leg).
        # Sign is positive because the seidel branch handles direction
        # via the n_after sign internally.
        d12 = 0.2     # 200 mm primary-to-secondary
        surfaces = [
            Surface(radius=R1, thickness=d12,
                    glass_before='air', glass_after='air',
                    is_mirror=True, is_stop=True,
                    semi_diameter=50e-3),
            Surface(radius=R2, thickness=0.0,
                    glass_before='air', glass_after='air',
                    is_mirror=True,
                    semi_diameter=15e-3),
        ]
        return surfaces, R1, R2

    def test_two_mirror_S4_matches_welford_handcalc(self):
        """The hand-computed Welford S4 sum with parity tracking is
        -16 (for R1=-0.5m, R2=-0.1m).  Without parity tracking the
        pre-4.11.2 code returned +24 (factor 1.5 AND a sign flip).
        Tolerance 1e-9 -- the formula is exact to roundoff."""
        surfaces, R1, R2 = self._make_two_mirror_system()
        wavelength = 0.55e-6
        result, _ = seidel_coefficients(
            surfaces, wavelength=wavelength,
            field_angle=0.001,
        )
        S4_total = float(result['total']['S4'])
        S4_per_surf = np.asarray(result['S4'])

        # Hand-calculated S4 contributions:
        # primary (parity 0 -> 1):   +4
        # secondary (parity 1 -> 0): -20
        expected_S4_primary = +4.0
        expected_S4_secondary = -20.0
        expected_total = expected_S4_primary + expected_S4_secondary

        assert abs(float(S4_per_surf[0]) - expected_S4_primary) < 1e-9, (
            f"Primary mirror S4 = {S4_per_surf[0]!r}; expected "
            f"{expected_S4_primary} (Welford with n2=-n1)."
        )
        assert abs(float(S4_per_surf[1]) - expected_S4_secondary) < 1e-9, (
            f"Secondary mirror S4 = {S4_per_surf[1]!r}; expected "
            f"{expected_S4_secondary} (Welford parity-tracked: n1=-1 "
            f"at the second mirror).  Pre-4.11.2 the secondary saw "
            f"n1=+1 and gave +20 with the WRONG sign."
        )
        assert abs(S4_total - expected_total) < 1e-9, (
            f"Two-mirror total S4 = {S4_total!r}; expected "
            f"{expected_total} (sum of Welford per-surface S4 with "
            f"mirror_parity tracking).  Pre-4.11.2: total ~ +24."
        )

    def test_two_mirror_round_trip_returns_to_positive_index(self):
        """After two mirrors the mirror_parity returns to 0 and any
        subsequent (refractive) surface should see n1 = +1 again.
        We can't check that directly without an extra surface, but we
        CAN check that the secondary mirror's S4 has the
        parity-tracked sign (verified by the handcalc above) -- which
        is only possible if mirror_parity flipped at the primary.

        Also verify the per-surface S1, S2, S3 are non-zero on BOTH
        mirrors (pre-4.10 the mirror branch never wrote them; pre-4.11.2
        the second-mirror values were silently wrong via the parity
        bug).
        """
        surfaces, _, _ = self._make_two_mirror_system()
        wavelength = 0.55e-6
        result, _ = seidel_coefficients(
            surfaces, wavelength=wavelength,
            field_angle=np.radians(0.5),
        )
        for k in ('S1', 'S2', 'S3', 'S4'):
            arr = np.asarray(result[k])
            assert arr.shape == (2,)
            assert abs(float(arr[0])) > 1e-12, (
                f"Primary mirror {k} = {arr[0]!r}: expected non-zero.")
            assert abs(float(arr[1])) > 1e-12, (
                f"Secondary mirror {k} = {arr[1]!r}: expected non-zero.")


# ============================================================================
# CRIT-3 -- system_abcd & seidel_coefficients agree on mirror EFL sign
# ============================================================================

class TestSystemAbcdMirrorSignAgreement:
    """For a single concave mirror R = -100 mm, the Welford convention
    (n2 = -n1) gives EFL = +50 mm = +0.05 m.  Pre-4.11.2:

    * ``system_abcd`` used ``phi = +2*n1/R`` and returned EFL = -0.05 m
    * ``seidel_coefficients`` used Welford ``phi = (n2-n1)/R = -2*n1/R``
      and returned (implicitly via its embedded ABCD) +0.05 m.

    4.11.2 reconciles both to the Welford +0.05 m convention.
    """

    def test_concave_mirror_efl_is_positive_50mm(self):
        R = -100e-3
        wavelength = 0.55e-6
        surfaces = [
            Surface(radius=R, thickness=0.0,
                    glass_before='air', glass_after='air',
                    is_mirror=True, is_stop=True,
                    semi_diameter=10e-3),
        ]
        _, efl, _, _ = system_abcd(surfaces, wavelength)
        # Welford / mirror convention: concave mirror has positive
        # focal length |R|/2 (with the post-mirror "axis" pointing
        # back toward the object).
        assert abs(efl - 0.05) < 1e-9, (
            f"system_abcd EFL for concave mirror R=-100mm = {efl!r}; "
            f"expected +0.05 m (Welford convention, agrees with "
            f"seidel_coefficients).  Pre-4.11.2 system_abcd returned "
            f"-0.05 m (opposite sign), disagreeing with the Seidel "
            f"branch on the same prescription."
        )

    def test_system_abcd_and_seidel_agree_on_efl(self):
        """Both code paths must produce identical ABCD matrices for
        a concave mirror.  ``seidel_coefficients`` embeds the ABCD
        at the end of its return; ``system_abcd`` returns it
        directly.  They share the (now reconciled) Welford
        convention.
        """
        R = -100e-3
        wavelength = 0.55e-6
        surfaces = [
            Surface(radius=R, thickness=0.0,
                    glass_before='air', glass_after='air',
                    is_mirror=True, is_stop=True,
                    semi_diameter=10e-3),
        ]
        M_abcd, efl_abcd, _, _ = system_abcd(surfaces, wavelength)
        _, M_seidel = seidel_coefficients(
            surfaces, wavelength=wavelength, field_angle=0.001)
        # The two ABCD matrices should be bit-identical (or within
        # floating-point roundoff).
        diff = float(np.max(np.abs(np.asarray(M_abcd) -
                                    np.asarray(M_seidel))))
        assert diff < 1e-12, (
            f"system_abcd and seidel_coefficients return different "
            f"ABCD matrices for the same single-mirror system; "
            f"max element-wise diff = {diff:.3e}.  This means the "
            f"two code paths use different mirror-power sign "
            f"conventions, which is exactly the CRIT-3 finding."
        )
        # Sanity: paraxial focal length from the recovered ABCD's
        # C-element matches the Welford value.
        C_seidel = float(np.asarray(M_seidel)[1, 0])
        efl_recovered = -1.0 / C_seidel
        assert abs(efl_recovered - 0.05) < 1e-9
        assert abs(efl_abcd - efl_recovered) < 1e-12


# ============================================================================
# CRIT-FC -- seidel_wfe field-curvature DC term
# ============================================================================

class TestSeidelWfeFieldCurvatureDcTerm:
    """The standard Hopkins / Welford third-order WFE has both:

    * astigmatism      ``(1/2) S3 rho^2 cos^2(theta)``
    * field-curvature  ``(1/4) (S3 + S4 H^2) rho^2`` (rotationally symmetric DC)

    Pre-4.11.2 the ``(1/4) S3 rho^2`` companion was missing from both
    the docstring and the implementation.  With S3=1, S4=0, rho=1,
    theta=0:

        pre-4.11.2 result:  (1/2)(1)(1)(1) + 0 = 0.5
        v4.11.2 result:     0.5 + (1/4)(1)(1) = 0.75
    """

    def test_wfe_includes_field_curvature_dc_companion(self):
        totals = {'S1': 0.0, 'S2': 0.0, 'S3': 1.0, 'S4': 0.0, 'S5': 0.0}
        # field_angle is required for the bare-totals path; for
        # S4=0 it's irrelevant to the numeric result.
        with pytest.warns(RuntimeWarning):
            # Triggers the bare-sigma^2 fallback warning since we
            # passed a bare totals dict.  Expected; doesn't affect
            # the (S3-only) result.
            W = lm.seidel_wfe(totals, rho=1.0, theta=0.0,
                              field_angle=0.0)
        W = float(W)
        # Expected: (1/2)*1*1*1 + (1/4)*1*1 = 0.5 + 0.25 = 0.75
        assert abs(W - 0.75) < 1e-15, (
            f"seidel_wfe(S3=1, S4=0, rho=1, theta=0) = {W!r}; "
            f"expected 0.75 (= 0.5 from the (1/2) S3 rho^2 cos^2 theta "
            f"astigmatism term plus 0.25 from the (1/4) S3 rho^2 "
            f"field-curvature DC companion that was missing pre-4.11.2)."
        )

    def test_wfe_pre_fix_value_no_longer_returned(self):
        """Direct regression: the value 0.5 (pre-fix) must no longer
        come out of this exact input."""
        totals = {'S1': 0.0, 'S2': 0.0, 'S3': 1.0, 'S4': 0.0, 'S5': 0.0}
        with pytest.warns(RuntimeWarning):
            W = lm.seidel_wfe(totals, rho=1.0, theta=0.0,
                              field_angle=0.0)
        W = float(W)
        assert abs(W - 0.5) > 0.1, (
            f"seidel_wfe returned the pre-4.11.2 value {W!r} ~ 0.5; "
            f"expected 0.75 after the field-curvature DC fix.")

    def test_wfe_field_curvature_dc_scales_with_rho_squared(self):
        """The new (1/4) S3 rho^2 DC term scales as rho^2; with S3=4
        and other Sk=0, theta=pi/2 (cos=0 so astigmatism term
        vanishes), the only contribution is (1/4)*4*rho^2 = rho^2.
        """
        totals = {'S1': 0.0, 'S2': 0.0, 'S3': 4.0, 'S4': 0.0, 'S5': 0.0}
        rho = np.array([0.0, 0.5, 1.0])
        theta = np.full_like(rho, np.pi / 2.0)
        with pytest.warns(RuntimeWarning):
            W = lm.seidel_wfe(totals, rho=rho, theta=theta,
                              field_angle=0.0)
        expected = rho ** 2  # (1/4) * 4 * rho^2
        max_err = float(np.max(np.abs(W - expected)))
        assert max_err < 1e-15, (
            f"seidel_wfe with S3=4, theta=pi/2 should give "
            f"(1/4)*4*rho^2 = rho^2; got W = {W!r}, max-err = "
            f"{max_err:.2e}.")


# ============================================================================
# CRIT-Bundles -- bundles.py conversion helpers no longer raise
# ============================================================================

class TestBundleConversionHelpers:
    """Pre-4.11.2 ``ray_to_path``, ``ray_to_beamlet``, ``path_to_ray``
    accessed ``RayBundle.positions`` and ``.directions`` -- attributes
    that the dataclass doesn't expose.  Every call raised
    ``AttributeError`` on the first attribute lookup.  4.11.2 stacks
    the per-component (x, y, z) / (L, M, N) arrays into (N, 3) views
    so the helpers actually work.
    """

    def _make_ray_bundle(self, n=4):
        from lumenairy.raytrace import RayBundle
        return RayBundle(
            x=np.linspace(-1e-3, 1e-3, n),
            y=np.zeros(n),
            z=np.zeros(n),
            L=np.zeros(n),
            M=np.zeros(n),
            N=np.ones(n),
            wavelength=633e-9,
            alive=np.ones(n, dtype=bool),
            opd=np.arange(n, dtype=float) * 1e-3,
        )

    def test_ray_to_path_succeeds(self):
        from lumenairy.raytrace.bundles import ray_to_path
        rb = self._make_ray_bundle()
        pb = ray_to_path(rb)
        # Stacked positions / directions shape: (N, 3)
        assert np.asarray(pb.positions).shape == (4, 3)
        assert np.asarray(pb.directions).shape == (4, 3)
        # OPL maps from RayBundle.opd to PathBundle.opl
        assert np.allclose(np.asarray(pb.opl),
                            np.arange(4, dtype=float) * 1e-3)
        # Default weights are unit complex
        assert np.allclose(np.asarray(pb.weights),
                            np.ones(4, dtype=np.complex128))

    def test_ray_to_beamlet_succeeds(self):
        from lumenairy.raytrace.bundles import ray_to_beamlet
        rb = self._make_ray_bundle()
        bb = ray_to_beamlet(rb, wavelength=633e-9, waist0=1e-3)
        assert np.asarray(bb.positions).shape == (4, 3)
        assert np.asarray(bb.directions).shape == (4, 3)
        # Q at the waist plane is purely imaginary (= -i / z_R)
        Q = np.asarray(bb.Q)
        assert np.all(np.real(Q) == 0.0)
        assert np.all(np.imag(Q) < 0.0)

    def test_path_to_ray_round_trip(self):
        from lumenairy.raytrace.bundles import ray_to_path, path_to_ray
        rb = self._make_ray_bundle()
        pb = ray_to_path(rb)
        rb2 = path_to_ray(pb)
        # Geometry preserved through the round trip
        assert np.allclose(rb2.x, rb.x)
        assert np.allclose(rb2.y, rb.y)
        assert np.allclose(rb2.z, rb.z)
        assert np.allclose(rb2.L, rb.L)
        assert np.allclose(rb2.M, rb.M)
        assert np.allclose(rb2.N, rb.N)
        # opd preserved through opl<->opd mapping
        assert np.allclose(rb2.opd, rb.opd)


# ============================================================================
# HIGH-Airy -- spot_diagram / trace_summary Airy radius includes f_eff
# ============================================================================

class TestAiryRadiusIncludesFeff:
    """Pre-4.11.2 ``trace_summary`` and ``spot_diagram`` printed/drew
    the Airy radius as ``1.22 * lambda / (2 * semi_diameter)`` -- a
    half-angle in radians, not an image-plane length.  4.11.2 includes
    the ``f_eff`` factor so the radius is in image-plane metres
    (consistent with the spot-RMS axis).

    For a 100mm-EFL plano-convex BK7 singlet at 587.6 nm, f/8:
        D       = 12.5 mm  (semi = 6.25 mm)
        f_eff   ~ 100 mm
        r_Airy  = 1.22 * 587.6e-9 * 0.1 / 0.0125  ~  5.73 um

    The pre-fix value would have been 1.22 * 587.6e-9 / 0.0125
    = 5.73e-5 rad (printed as 5.73e-5 m -- the metric scale was off
    by f_eff [m^-1]).
    """

    def test_trace_summary_airy_uses_feff(self, capsys):
        # 100mm-EFL plano-convex BK7 singlet, f/8
        presc = lm.make_singlet(R1=51.5e-3, R2=float('inf'),
                                d=3.0e-3, glass='N-BK7',
                                aperture=12.5e-3)
        wavelength = 587.6e-9
        surfaces = lm.surfaces_from_prescription(presc)
        # Trace a tiny on-axis fan
        fan = lm.make_fan('y', 6.25e-3, n_rays=11, field_angle=0.0,
                          wavelength=wavelength)
        result = lm.trace(fan, surfaces, wavelength)
        lm.trace_summary(result, units='um')
        captured = capsys.readouterr().out
        # Parse the "Airy radius:" line
        for line in captured.splitlines():
            if 'Airy radius' in line:
                # value is between the colon and the unit suffix
                tail = line.split(':', 1)[1].strip()
                value_str = tail.split()[0]
                airy_um = float(value_str)
                break
        else:
            pytest.fail("No 'Airy radius' line in trace_summary output")
        # Expect ~5.7 um (f_eff factor included).  Without it: ~5.7e-5 um.
        assert 1.0 < airy_um < 50.0, (
            f"Airy radius printed as {airy_um!r} um; expected O(5-10) "
            f"um for a 100mm/f8 singlet at 587 nm.  Pre-4.11.2 the "
            f"missing f_eff factor produced ~5.7e-5 um (a half-angle "
            f"in radians, mis-labelled as metres)."
        )
