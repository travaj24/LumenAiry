"""Ground-truth tests for ``seidel_coefficients`` -- the 4.9 audit-response
addition.

The pre-4.9 test suite only smoke-tested (`return_named_data` checks for
a non-None dict) and scaling-relation-tested (S1 field-independent,
S2 ∝ h, etc.) -- both of which are preserved by a uniform multiplicative
error in the per-surface coefficient formula.  This file adds
**absolute-magnitude** assertions by fitting a known ray-traced OPD and
asserting the recovered coefficient matches Seidel theory.

Reference: Welford, *Aberrations of Optical Systems* (Adam Hilger, 1986),
chapter 8 -- specifically Eq. 8.46 (per-surface S_I) and the standard
relation ``W_spherical(rho) = (1/8) · S_I · rho^4`` for the third-order
spherical wavefront error.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

import lumenairy as lm
from lumenairy.raytrace import Surface, first_order_data, make_fan, trace

# Test geometry: plano-convex BK7 singlet matched to the audit's hand
# calc so the numerical comparison is one-to-one with the audit report.
PRESCRIPTION = {
    'wavelength': 587.6e-9,
    'aperture_diameter': 25.4e-3,
    'surfaces': [
        {'radius': 51.5e-3, 'thickness': 1.0e-3,
         'glass_before': 'air', 'glass_after': 'N-BK7',
         'semi_diameter': 12.7e-3},
        {'radius': np.inf, 'thickness': 0.0,
         'glass_before': 'N-BK7', 'glass_after': 'air',
         'semi_diameter': 12.7e-3},
    ],
    'thicknesses': [1.0e-3],
    'stop_index': 0,
}


def _traced_third_order_S1(surfaces, wavelength, r_stop, n_rays=201):
    """Independent, ray-traced estimate of the total third-order
    spherical Seidel sum ``S1`` -- the absolute-magnitude oracle the
    S5-1 audit finding said the flagship test was missing.

    Third-order theory gives, at the paraxial image plane, a purely odd
    transverse ray aberration for the tangential (y) fan::

        dy(rho) = -(S1 / (2 n' u'_k)) * rho^3   (+ higher-order rho^5, ...)

    with ``rho = y_launch / r_stop`` the fractional pupil coordinate and
    ``n' u'_k = -r_stop / efl`` the image-space reduced marginal-ray
    angle for an object at infinity with the stop at surface 0.  The
    transverse ray aberration is just the local slope of the wavefront
    OPD (``dW/d rho``), so this is the audit's promised "ray-trace OPD"
    check -- measured as ray displacement rather than a raw OPD-to-plane
    rho^4 fit, which is corrupted at f/4 by the plane-vs-reference-sphere
    quartic (~3 um, comparable to the ~7 um of spherical itself).

    Reading the ``rho^3`` coefficient of the traced ``dy`` therefore
    recovers ``S1`` *without* touching the paraxial Buchdahl-Hopkins code
    in :func:`seidel_coefficients`: it is a full geometric trace (exact
    Snell refraction + propagation), a genuinely independent oracle.
    Only the sign convention differs from the library's ``S1`` (the two
    use opposite transverse-aberration / ray-angle conventions); the
    magnitude is the cross-check.  Returns the signed traced value.
    """
    fod = first_order_data(surfaces, wavelength)
    efl, bfl = fod.efl, fod.bfl
    # Evaluate at the paraxial focus.  The tracer applies transfers only
    # BETWEEN surfaces, so the last surface's own thickness is never
    # propagated -- append an explicit flat image surface ``bfl`` away.
    surf = [copy.copy(s) for s in surfaces]
    surf[-1].thickness = bfl
    surf.append(Surface(radius=np.inf, thickness=0.0,
                        glass_before='air', glass_after='air'))
    for s in surf:
        s.semi_diameter = np.inf  # never clip the marginal ray at the edge
    fan = make_fan('y', r_stop, n_rays, field_angle=0.0, wavelength=wavelength)
    img = trace(fan, surf, wavelength).image_rays
    rho = np.linspace(-1.0, 1.0, n_rays)
    dy = np.asarray(img.y)
    good = np.asarray(img.alive) & np.isfinite(dy)
    rho, dy = rho[good], dy[good]
    # Odd-power fit; the rho^3 term is the third-order transverse
    # spherical (rho^5/rho^7 absorb higher-order spherical at f/4).
    basis = np.vstack([rho ** 3, rho ** 5, rho ** 7]).T
    c3 = np.linalg.lstsq(basis, dy, rcond=None)[0][0]
    nu_prime = -r_stop / efl  # image-space reduced marginal-ray angle
    return -2.0 * nu_prime * c3


@pytest.fixture
def seidel_result():
    """Run the Seidel calculator once per test session for our reference
    singlet so each test reads from the same numbers."""
    surfaces = lm.surfaces_from_prescription(PRESCRIPTION)
    return lm.seidel_coefficients(
        surfaces, wavelength=PRESCRIPTION['wavelength'],
        field_angle=0.001,  # near on-axis -- isolates the S1 term
    )


class TestSeidelGroundTruth:
    """Each test compares a Seidel-derived quantity to an independent
    reference value (ray trace OPD fit, Welford hand calc, etc.).
    Tests that only check internal self-consistency live elsewhere; this
    file is specifically for cross-implementation checks the audit
    flagged as missing."""

    def test_seidel_returns_lagrange_invariant(self, seidel_result):
        """4.9: the ``seidel_coefficients`` dict must carry the
        Lagrange invariant ``H`` so that ``seidel_wfe`` can apply
        the correct ``H²`` Petzval scaling."""
        result, _abcd = seidel_result
        assert 'lagrange_invariant' in result
        H = float(result['lagrange_invariant'])
        # For a stop-at-surface-0 collimated input: H = -r_stop * sigma
        # (sign-convention-dependent).  |H| = r_stop * |sigma|.
        expected_abs_H = (PRESCRIPTION['aperture_diameter'] / 2.0) * 0.001
        assert abs(abs(H) - expected_abs_H) / expected_abs_H < 0.05

    def test_flat_surface_contributes_to_S1(self, seidel_result):
        """4.9 fix: pre-4.9 zeroed S1[i] on the flat-refracting branch.
        For a plano-convex singlet (curved-front + flat-back) the back
        surface is flat but Δ(u/n) is still non-zero, so it should
        contribute a non-zero spherical-aberration term."""
        result, _abcd = seidel_result
        S1 = np.asarray(result['S1'])
        # Surface 0 is the curved front; surface 1 is the flat back.
        # Both should be non-zero under the corrected formula.
        assert abs(S1[1]) > 1e-9, (
            f"S1[1] (flat back surface) = {S1[1]:.3e}, expected non-zero. "
            f"This was the audit's #2.1 finding: pre-4.9 zeroed S1/S2/S3 "
            f"on flat refracting surfaces, which is wrong because "
            f"Δ(u/n) ≠ 0 at non-normal incidence."
        )

    def test_S1_matches_ray_trace_OPD_fit(self):
        """The headline ground-truth test the audit recommended:
        ray-trace a marginal-ray bundle through the singlet, extract the
        on-axis paraxial-focus wavefront, and assert the analytic Seidel
        S1 matches an INDEPENDENT ray-traced value.

        S5-1 fix: the pre-4.9 assertion was only the range window
        ``1e-5 < |S1| < 5e-4``, which admitted BOTH the correct ~5.8e-5
        AND the pre-4.9 buggy ~2.6e-4 -- a tautology that caught nothing.
        It is replaced here by a genuine cross-check against a full
        geometric ray trace via :func:`_traced_third_order_S1`.  That
        oracle reads the ``rho^3`` transverse-ray-aberration coefficient
        at the paraxial focus (the local slope ``dW/d rho`` of the
        wavefront OPD), which recovers ``S1`` while sharing no code with
        the paraxial Buchdahl-Hopkins formula -- and, unlike a raw
        OPD-to-plane ``rho^4`` fit, is free of the plane-vs-sphere
        quartic that contaminates the OPD at f/4.
        """
        # On-axis Seidel run -- pure spherical, no other aberrations.
        surfaces = lm.surfaces_from_prescription(PRESCRIPTION)
        seidel, abcd = lm.seidel_coefficients(
            surfaces, wavelength=PRESCRIPTION['wavelength'],
            field_angle=0.0,
        )
        S1_total = float(seidel['total']['S1'])

        r_stop = PRESCRIPTION['aperture_diameter'] / 2.0
        S1_traced = _traced_third_order_S1(
            surfaces, PRESCRIPTION['wavelength'], r_stop)

        # Independent-oracle magnitude match.  Empirically the traced
        # value is 5.755e-5 vs the analytic 5.7545e-5 (|ratio| = 1.0000,
        # stable across 81-401 rays); the pre-4.9 bug produced ~2.6e-4
        # (|ratio| ~4.5), comfortably rejected by this +-15% band.
        ratio = abs(S1_total) / abs(S1_traced)
        assert 0.85 < ratio < 1.15, (
            f"Analytic Seidel S1 = {S1_total:.4e} disagrees with the "
            f"independent ray-traced transverse-aberration oracle "
            f"{S1_traced:.4e} (|ratio| = {ratio:.4f}).  Expected |ratio| "
            f"~1.0; the pre-4.9 bug (~2.6e-4) gives |ratio| ~4.5."
        )

    def test_S1_traced_oracle_rejects_5x_inflated_value(self):
        """S5-1 regression: prove the traced oracle actually
        DISCRIMINATES -- it must accept the correct S1 *and* reject the
        documented pre-4.9 buggy value.  The old range-window assertion
        (``1e-5 < |S1| < 5e-4``) passed on BOTH, so it caught nothing;
        this guards against silently re-introducing such a slack gate.

        The oracle (:func:`_traced_third_order_S1`) shares no code with
        the Seidel formula -- it is a full geometric ray trace.
        """
        surfaces = lm.surfaces_from_prescription(PRESCRIPTION)
        seidel, _ = lm.seidel_coefficients(
            surfaces, wavelength=PRESCRIPTION['wavelength'],
            field_angle=0.0)
        S1_total = float(seidel['total']['S1'])
        r_stop = PRESCRIPTION['aperture_diameter'] / 2.0
        S1_traced = _traced_third_order_S1(
            surfaces, PRESCRIPTION['wavelength'], r_stop)

        band = 0.15  # the +-15% band used by the flagship test
        # (a) the CORRECT analytic value is accepted ...
        assert abs(abs(S1_total) / abs(S1_traced) - 1.0) < band, (
            f"correct S1 = {S1_total:.4e} rejected by the traced oracle "
            f"{S1_traced:.4e}")
        # (b) ... while the documented pre-4.9 buggy magnitude (~2.6e-4,
        # ~4.5x too large) is REJECTED by that very same band.  If this
        # ever stops holding, the magnitude gate has gone slack and the
        # tautology the audit flagged (S5-1) has crept back in.
        S1_buggy = 2.6e-4
        assert abs(S1_buggy / abs(S1_traced) - 1.0) >= band, (
            f"the traced oracle FAILED to reject the pre-4.9 buggy "
            f"S1 = {S1_buggy:.3e} (traced = {S1_traced:.4e}); the "
            f"magnitude gate is too loose to be a real regression guard.")

    def test_seidel_wfe_petzval_uses_H_squared(self, seidel_result):
        """4.9 fix #4.6: seidel_wfe scales S4 by |H|² (the Lagrange
        invariant squared), not bare sigma².  Verify by reconstructing
        the WFE map and checking the Petzval term's magnitude.
        """
        result, _abcd = seidel_result
        # Build a small (rho, theta) grid to evaluate seidel_wfe on.
        rho = np.linspace(0.0, 1.0, 33)
        theta = np.zeros_like(rho)
        W = lm.seidel_wfe(result, rho, theta)
        # The WFE at the marginal-ray edge (rho=1) for a near-on-axis
        # field (sigma = 0.001 rad) on a 100 mm BK7 singlet at f/4
        # should be sub-micron (a few hundred nm of WFE max).  Pre-4.9
        # the buggy Petzval term produced ~100 mm of phantom WFE.
        W_edge_abs = abs(float(W[-1]))
        assert W_edge_abs < 1e-3, (
            f"WFE at pupil edge = {W_edge_abs:.3e} m.  Expected "
            f"sub-millimetre for a near-on-axis singlet evaluation; "
            f"pre-4.9 Petzval bug produced ~100 mm.  4.9 fix uses "
            f"|H|² = (r_stop·sigma)² instead of bare sigma² for the "
            f"S4 term."
        )

    def test_S1_S2_S3_S4_S5_finite_no_NaN(self, seidel_result):
        """Defensive: every per-surface coefficient must be finite."""
        result, _abcd = seidel_result
        for key in ('S1', 'S2', 'S3', 'S4', 'S5'):
            arr = np.asarray(result[key])
            assert np.all(np.isfinite(arr)), (
                f"{key} contains NaN or Inf: {arr!r}")

    def test_stop_at_thin_lens_distortion_vanishes(self):
        """S3-1 INDEPENDENT gate: third-order distortion (S5) must
        vanish for a thin lens with the aperture stop in its plane
        (Welford / Kingslake: a stop AT a thin lens gives zero
        distortion).  For a biconvex singlet with the stop at the
        front surface, ``|S5|`` must therefore scale *linearly toward
        zero* as the centre thickness ``t -> 0`` (the stop approaches
        the lens plane).

        This oracle does NOT come from the library's own Seidel
        formula -- it is the physical invariant the S3-1 sign bug
        violated.  Pre-fix the Petzval sum S4 sat on the OPPOSITE sign
        convention to S1-S3, so the ``S3`` and ``H^2 S4`` terms in the
        Schwarzschild ``S5 = (A_c/A_m)(S3 + H^2 S4)`` added instead of
        cancelling: ``|S5|`` PLATEAUED (~1.2e-5 relative) and actually
        grew as the lens thinned, instead of vanishing.
        """
        wl = 0.55e-6

        def biconvex_stop_at_lens(t):
            # Stop AT the front surface; as t -> 0 the two surfaces
            # merge into a thin lens in the stop plane.
            return [
                Surface(radius=50e-3, thickness=t, glass_before='air',
                        glass_after='N-BK7', is_stop=True,
                        semi_diameter=10e-3),
                Surface(radius=-50e-3, thickness=0.0,
                        glass_before='N-BK7', glass_after='air',
                        semi_diameter=10e-3),
            ]

        thicknesses = [4e-3, 2e-3, 1e-3, 0.5e-3]
        S5s, S1s = [], []
        for t in thicknesses:
            r, _ = lm.seidel_coefficients(
                biconvex_stop_at_lens(t), wavelength=wl, field_angle=0.01)
            S5s.append(abs(float(r['total']['S5'])))
            S1s.append(abs(float(r['total']['S1'])))

        # (1) |S5| shrinks monotonically toward 0 as the lens thins.
        # The bug made distortion grow/plateau instead.
        for a, b in zip(S5s[:-1], S5s[1:]):
            assert b < a, (
                f"|S5| must shrink as the stop approaches the lens "
                f"plane; got {S5s} for t={thicknesses}.  The S3-1 bug "
                f"made distortion plateau/grow as the lens thinned.")

        # (2) Linear-in-thickness scaling: halving t halves |S5|, so
        # |S5(thinnest)| / |S5(thickest)| ~ t_thin / t_thick = 0.125.
        ratio = S5s[-1] / S5s[0]
        t_ratio = thicknesses[-1] / thicknesses[0]
        assert 0.6 * t_ratio < ratio < 1.6 * t_ratio, (
            f"|S5| must scale ~linearly toward 0 with thickness: |S5| "
            f"ratio {ratio:.3f} vs thickness ratio {t_ratio:.3f}.  "
            f"Pre-S3-1 this ratio was ~1 (plateau).")

        # (3) At the near-thin limit the relative distortion is tiny.
        # Pre-fix |S5/S1| floored near ~1.2e-5; the fix drives it to
        # ~5e-7 and continues toward 0.
        rel = S5s[-1] / S1s[-1]
        assert rel < 2e-6, (
            f"|S5/S1| at the near-thin stop-at-lens limit = {rel:.2e}; "
            f"expected << 1e-6 (distortion -> 0).  The S3-1 bug left "
            f"this at ~1.2e-5.")

    def test_petzval_sum_matches_independent_analytic_formula(self):
        """S3-1 INDEPENDENT magnitude gate (shares NO code with the
        library's Seidel formula): the stored Petzval sum ``S4`` is the
        RAW Petzval sum ``P = sum_k c_k (n'_k - n_k)/(n'_k n_k)`` (surface
        curvature ``c`` and the indices before/after each surface), a
        closed form from the Petzval theorem itself.  Matching it pins
        BOTH the sign AND the magnitude of S4 against an oracle the
        original self-referential S3-1 test never had -- and it is the
        gate that would have caught the shipped ``-P`` sign directly.

        Ground truth (by-hand + rayoptics cross-checked): a biconvex
        N-BK7 (R = +/-50 mm) singlet has P = +13.6287 /m,
        |r_petzval| = 73.4 mm.
        """
        from lumenairy.glass import get_glass_index
        wl = 587.5618e-9
        n = get_glass_index('N-BK7', wl)
        R1, R2 = 50e-3, -50e-3
        surfs = [
            Surface(radius=R1, thickness=5e-3, glass_before='air',
                    glass_after='N-BK7', is_stop=True, semi_diameter=5e-3),
            Surface(radius=R2, thickness=47.6e-3, glass_before='N-BK7',
                    glass_after='air', semi_diameter=5e-3),
        ]
        r, _ = lm.seidel_coefficients(surfs, wavelength=wl,
                                      object_distance=np.inf, stop_index=0,
                                      field_angle_deg=1.0)
        S4 = float(r['total']['S4'])
        # Independent closed form (Petzval theorem) -- not the library's.
        P = ((1.0 / R1) * (n - 1.0) / (n * 1.0)
             + (1.0 / R2) * (1.0 - n) / (1.0 * n))
        assert abs(S4 - P) <= 1e-6 * abs(P), (
            f"library S4 (raw Petzval sum) = {S4:+.6e} /m must equal the "
            f"independent analytic Petzval theorem P = {P:+.6e} /m "
            f"(diff {abs(S4 - P):.2e}).  The S3-1 sign bug put S4 at -P.")
        # The pre-fix opposite convention (-P) must be firmly REJECTED.
        assert abs(S4 - (-P)) > 0.5 * abs(P), (
            "S4 must NOT sit on the -P convention (the S3-1 bug).")

    def test_seidel_signs_match_independent_rayoptics(self):
        """S3-1 fully-independent cross-LIBRARY gate: every third-order
        Seidel sum must share a SINGLE global sign convention with
        rayoptics (an independent optical-design package).  The S3-1 bug
        was a RELATIVE sign error -- S4 on the opposite convention to
        S1-S3 -- which surfaces here as S4's sign disagreeing with
        rayoptics once the (S1-anchored) global convention factor is
        removed.  rayoptics is optional; skip cleanly where absent so the
        gate never blocks a minimal-deps CI leg, but locks the convention
        wherever the external oracle is installed.
        """
        pytest.importorskip('rayoptics')
        import rayoptics.parax.thirdorder as _to
        from rayoptics.environment import (
            FieldSpec,
            OpticalModel,
            PupilSpec,
            WvlSpec,
        )

        opm = OpticalModel()
        sm = opm['seq_model']
        osp = opm['optical_spec']
        osp['pupil'] = PupilSpec(osp, key=['object', 'epd'], value=10.0)
        osp['fov'] = FieldSpec(osp, key=['object', 'angle'], value=[1.0],
                               is_relative=False)
        osp['wvls'] = WvlSpec([(587.5618, 1.0)], ref_wl=0)
        opm.radius_mode = True
        sm.gaps[0].thi = 1e10
        sm.add_surface([50.0, 5.0, 'N-BK7', 'Schott'])
        sm.add_surface([-50.0, 100.0])
        sm.stop_surface = 1
        opm.update_model()
        fod = opm['analysis_results']['parax_data'].fod
        sm.gaps[-1].thi = fod.bfl
        opm.update_model()
        df = _to.compute_third_order(opm)
        ray = {k: float(df[c].sum())
               for k, c in zip(('S1', 'S2', 'S3', 'S4', 'S5'), df.columns)}

        surfs = [
            Surface(radius=50e-3, thickness=5e-3, glass_before='air',
                    glass_after='N-BK7', is_stop=True, semi_diameter=5e-3),
            Surface(radius=-50e-3, thickness=fod.bfl * 1e-3,
                    glass_before='N-BK7', glass_after='air',
                    semi_diameter=5e-3),
        ]
        r, _ = lm.seidel_coefficients(surfs, wavelength=587.5618e-9,
                                      object_distance=np.inf, stop_index=0,
                                      field_angle_deg=1.0)
        lum = {k: float(r['total'][k]) for k in ('S1', 'S2', 'S3', 'S4', 'S5')}

        # Global convention factor from S1 (spherical -- sign unambiguous).
        g = np.sign(lum['S1'] / ray['S1'])
        for k in ('S1', 'S2', 'S3', 'S4', 'S5'):
            assert np.sign(lum[k]) == g * np.sign(ray[k]), (
                f"Seidel {k} sign {np.sign(lum[k]):+.0f} disagrees with "
                f"the independent rayoptics oracle {g * np.sign(ray[k]):+.0f} "
                f"(global convention factor g={g:+.0f}).  The S3-1 bug put "
                f"S4 on the opposite convention; a regression re-introduces "
                f"it.")

    def test_seidel_scaling_with_field_angle(self):
        """Cross-check: S1 is independent of field angle, S2 ∝ field,
        S3 ∝ field², S5 ∝ field³.  The audit pointed out these
        scaling tests are blind to a uniform multiplicative bug, but
        they SHOULD still hold after the fix -- so this is a sanity
        check that we didn't break the scaling while fixing the
        magnitudes."""
        surfaces = lm.surfaces_from_prescription(PRESCRIPTION)
        rs1, _ = lm.seidel_coefficients(
            surfaces, wavelength=PRESCRIPTION['wavelength'],
            field_angle=0.001)
        rs2, _ = lm.seidel_coefficients(
            surfaces, wavelength=PRESCRIPTION['wavelength'],
            field_angle=0.002)
        T1 = rs1['total']
        T2 = rs2['total']
        # S1 (spherical) field-independent
        np.testing.assert_allclose(T1['S1'], T2['S1'], rtol=1e-9)
        # S2 (coma) ∝ field
        np.testing.assert_allclose(T2['S2'] / T1['S2'], 2.0, rtol=1e-9)
        # S3 (astigmatism) ∝ field²
        np.testing.assert_allclose(T2['S3'] / T1['S3'], 4.0, rtol=1e-9)
        # S5 distortion gets two contributions through the Schwarzschild
        # relation S_V = -(A_c/A_m)·(S3 + H²·S4): one ∝ field³ from S3
        # and one ∝ field from S4·H² (H ∝ field for object at infinity).
        # The ratio is therefore NOT a clean power of (field_ratio); it
        # falls between field¹ = 2 and field³ = 8 depending on which
        # term dominates.  Just assert it grew substantially with field.
        ratio_S5 = abs(T2['S5'] / T1['S5'])
        assert 1.5 < ratio_S5 < 10.0, (
            f"S5 ratio across 2× field bump: {ratio_S5:.3f} "
            f"(expected 2-8 range; depends on S3 vs S4·H² mix).")
