"""Regression tests for the 4.11.1 audit-residual patch.

Each test pins one of the round-2-verification findings.  These tests
are intentionally small and runtime-cheap (each <1 s); their purpose is
to fail loudly if a future refactor reintroduces the specific bug, not
to exercise the broader physics.

Findings covered (per ``AUDIT_VERIFICATION_2026_05_16.md`` / round-2):

* C-OP-1 / N1  -- ``MultiWavelengthMerit`` per-wavelength wave-leg
                   keyword-call (was positional, raised silently).
* M-LR-1       -- decentered-stop lookup on dict (was ``getattr`` on
                   wrong keys, fix dead-on-arrival pre-4.11.1).
* C-PL-1       -- circular-polarisation handedness consistency.
* H-PR-4       -- ``create_point_source`` central pixel bounded.
* N5           -- subaperture output-grid unpack imports.
* C-RT-1/C-AB-1 -- mirror Seidel coefficients are non-zero.
* C-SC-1       -- tilted-ASM band-limit produces a non-zero output.
* H-RT-2       -- ``trace_jax`` raises on a mirror surface.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy as lm


# ============================================================================
# C-OP-1 / N1 -- MultiWavelengthMerit per-wavelength wave-leg
# ============================================================================

class TestMultiWavelengthMeritWaveLegRuns:
    """``MultiWavelengthMerit`` is supposed to re-evaluate the wave leg
    (apply_real_lens + through-focus scan) at each wavelength.  Pre-
    4.11.1 it called ``apply_real_lens(E, ctx.prescription, wl, dx)``
    positionally, which raised ``TypeError`` because every argument
    after ``E_in`` is keyword-only since 4.7.  A bare
    ``except Exception: pass`` swallowed the error.  4.11.1 makes
    the call keyword and narrows the except to typed warnings.

    This test passes iff the per-wavelength wave leg DOES NOT trigger
    the typed ``"MultiWavelengthMerit: per-wavelength wave-leg
    propagation failed"`` warning.  Pre-4.11.1 the positional call
    would have raised TypeError on every iteration; 4.11.1 the wave
    leg runs cleanly.
    """

    def test_per_wavelength_wave_leg_does_not_silently_fail(self):
        from lumenairy.optimize.core import EvaluationContext
        wls = [1.0e-6, 1.55e-6]
        N = 32
        dx = 5e-6
        # Single thin lens: gives a real propagating wave-leg path.
        prescription = lm.make_singlet(
            R1=20e-3, R2=-20e-3, d=2e-3,
            glass='N-BK7', aperture=50e-6)
        # A simple plane-wave-like field so ctx.E_exit is non-None
        # and the needs_wave branch fires.
        E_seed = np.ones((N, N), dtype=np.complex128)
        ctx = EvaluationContext(
            prescription=prescription, wavelength=wls[0],
            N=N, dx=dx, efl=1.0, bfl=1.0,
            E_exit=E_seed,
            opd_map=None, strehl_best=0.0,
            rms_radius_best=1e-6, z_best=1.0)
        sub = lm.StrehlMerit(min_strehl=0.8, weight=1.0)
        merit = lm.MultiWavelengthMerit(
            wavelengths=wls, sub_merit=sub, weight=1.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            _ = merit.evaluate(ctx)
        wave_leg_warns = [
            w for w in caught
            if 'per-wavelength wave-leg propagation failed' in str(w.message)
        ]
        assert not wave_leg_warns, (
            f"MultiWavelengthMerit silently fell back to the parent "
            f"context's wave-leg values at "
            f"{len(wave_leg_warns)} wavelength(s).  Pre-4.11.1 every "
            f"call to apply_real_lens inside the merit raised "
            f"TypeError (positional call to a keyword-only function), "
            f"which was swallowed by a bare except.  4.11.1 makes the "
            f"call keyword; this test asserts no failure warning fires."
        )


# ============================================================================
# M-LR-1 -- Decentered stop lookup on dict
# ============================================================================

class TestDecenteredStopHonoured:
    """The aperture stop's decenter is stored as a tuple under the
    ``'decenter'`` key on the surface dict.  Pre-4.11.1 the lookup
    used ``getattr(surf, 'decenter_x_m', 0.0)`` -- getattr on a dict
    for a non-attribute name silently returns the default, so the
    fix landed in 4.10.2 was inoperative for the entire v4.10 series.
    """

    def test_decentered_stop_clips_offset_disk(self):
        from lumenairy.elements._lens_real import apply_real_lens
        N, dx, wl = 64, 2e-6, 1.55e-6
        E_in = np.ones((N, N), dtype=np.complex128)
        ap_diam = 30e-6
        dx_decenter = 0.5 * ap_diam  # decenter stop by half its diameter in +x
        # Two-surface flat prescription with the stop on surface 0.
        prescription = {
            'aperture_diameter': ap_diam,
            'stop_index': 0,
            'surfaces': [
                {'radius': float('inf'), 'conic': 0.0,
                 'aspheric_coeffs': None,
                 'radius_y': None, 'conic_y': None,
                 'aspheric_coeffs_y': None,
                 'glass_before': 'air', 'glass_after': 'air',
                 'decenter': (dx_decenter, 0.0)},
                {'radius': float('inf'), 'conic': 0.0,
                 'aspheric_coeffs': None,
                 'radius_y': None, 'conic_y': None,
                 'aspheric_coeffs_y': None,
                 'glass_before': 'air', 'glass_after': 'air'},
            ],
            'thicknesses': [1e-6],  # tiny propagation so we can still observe clipping
        }
        E_out = apply_real_lens(
            E_in, prescription=prescription,
            wavelength=wl, dx=dx, bandlimit=True)
        x = (np.arange(N) - N / 2) * dx
        iy_mid = N // 2
        # Pixel inside the +x-decentered stop (near stop centre at
        # +dx_decenter) should survive with |E| ~ 1.
        ix_inside = int(np.argmin(np.abs(x - dx_decenter)))
        v_inside = abs(E_out[iy_mid, ix_inside])
        # Pixel outside the +x-decentered stop (on the -x side,
        # well past the optical axis from the stop) should be ~zero.
        ix_outside = int(np.argmin(np.abs(x - (-1.5 * dx_decenter))))
        v_outside = abs(E_out[iy_mid, ix_outside])
        assert v_inside > 0.5, (
            f"Decentered stop fix dead: pixel inside the +x-shifted "
            f"aperture had |E|={v_inside:.3e} (expected ~1)."
        )
        assert v_outside < 0.1, (
            f"Decentered stop fix dead: pixel at -1.5*decenter (well "
            f"outside the +x-shifted aperture) survived with "
            f"|E|={v_outside:.3e} (expected ~0).  Pre-4.11.1 the "
            f"lookup was getattr(dict, 'decenter_x_m', 0.0) which "
            f"always returned 0.0, so the stop stayed on-axis."
        )


# ============================================================================
# C-PL-1 -- Circular polarisation handedness consistency
# ============================================================================

class TestCircularPolarisationHandednessConsistent:
    """Three sites must agree under the library's
    ``S3 = -2 Im(Ex Ey*)`` convention with "right = S3 > 0":

    1. ``create_circular_polarized('right')`` -> S3 > 0
    2. ``apply_quarter_wave_plate`` on a linear x-polarised input
       at fast-axis angle pi/4 -> matches handedness of 1.
    3. (Documented match with ``vector_diffraction.py:147`` is by
       inspection: ``(1, +1j)/sqrt(2)`` is hard-coded for
       ``polarization='circular'``.)

    Pre-4.11.1 the three sites disagreed (the 4.10 patch flipped
    ``create_circular_polarized`` and inadvertently broke parity).
    """

    def test_create_right_has_positive_s3(self):
        N, dx = 16, 5e-6
        scalar = np.ones((N, N), dtype=np.complex128)
        jf = lm.create_circular_polarized(scalar, dx, handedness='right')
        s = lm.stokes_parameters(jf)
        s3 = float(np.mean(s['S3']))
        assert s3 > 0.5, (
            f"create_circular_polarized('right') gave S3 = {s3:.3f}; "
            f"expected ~+1 under the library's S3 = -2 Im(Ex Ey*) "
            f"convention with 'right'-positive.  Pre-4.11.1 the 4.10 "
            f"patch flipped 'right' to (1, -i)/sqrt(2) which has "
            f"S3 = -1, contradicting the docstring and apply_waveplate."
        )

    def test_qwp_on_linear_x_matches_create_right_handedness(self):
        N, dx = 16, 5e-6
        scalar = np.ones((N, N), dtype=np.complex128)
        jf_lin = lm.create_linear_polarized(scalar, dx, angle=0.0)
        jf_qwp = lm.apply_quarter_wave_plate(jf_lin, angle=np.pi / 4)
        jf_ref = lm.create_circular_polarized(
            scalar, dx, handedness='right')
        # Both should land at the same handedness (same sign of S3).
        s_qwp = float(np.mean(lm.stokes_parameters(jf_qwp)['S3']))
        s_ref = float(np.mean(lm.stokes_parameters(jf_ref)['S3']))
        assert s_qwp * s_ref > 0, (
            f"apply_quarter_wave_plate on linear-x at angle=pi/4 "
            f"gave S3={s_qwp:.3f}; create_circular_polarized('right') "
            f"gave S3={s_ref:.3f}.  Signs must agree for the two "
            f"recipes to produce the same physical handedness.  "
            f"Pre-4.11.1 these disagreed."
        )


# ============================================================================
# H-PR-4 -- create_point_source central-pixel amplitude is bounded
# ============================================================================

class TestPointSourceCentralPixelBounded:
    """Pre-4.11.1 ``r`` was clamped to 1e-30 so the central pixel had
    ``|E| = amplitude / 1e-30 = 1e30``, dominating every integrated
    power calc.  4.11.1 clamps ``r`` to the pixel half-diagonal,
    capping ``|E_central|`` to ~amplitude / dx.
    """

    def test_central_pixel_finite_and_bounded(self):
        N, dx, wl = 32, 5e-6, 1.55e-6
        # |z0| < dx triggers the regularised regime + RuntimeWarning.
        with pytest.warns(RuntimeWarning):
            E, _, _ = lm.create_point_source(
                N, dx, wavelength=wl, x0=0.0, y0=0.0, z0=0.5 * dx,
                amplitude=1.0)
        peak = float(np.max(np.abs(E)))
        # 4.11.1 cap: |E_central| <= amplitude / (sqrt(2)/2 * dx)
        # ~= 2.83e5 for dx=5e-6.  Allow a few orders of margin.
        assert peak < 1e7, (
            f"create_point_source central-pixel |E| = {peak:.3e}; "
            f"expected bounded by ~amplitude/dx (~{1/dx:.3e}).  "
            f"Pre-4.11.1 the 1e-30 floor on r produced |E| ~ 1e30."
        )


# ============================================================================
# N5 -- subaperture grid unpack import
# ============================================================================

class TestSubapertureImports:
    """``subaperture.py`` constructed ``output_grid_xy`` as
    ``np.stack([OX, OY], axis=-1)`` (ndim=3) and then tried to unpack
    it ``sgx, sgy = output_grid_xy``, which raised ``ValueError``
    for any Ny != 2.  4.11.1 simplifies to ``sgx, sgy = OX, OY``.
    Smoke-test: the public function imports without error.
    """

    def test_propagate_subaperture_asymptotic_importable(self):
        from lumenairy.propagators.subaperture import (
            propagate_subaperture_asymptotic)
        assert callable(propagate_subaperture_asymptotic)


# ============================================================================
# C-RT-1 / C-AB-1 -- Mirror Seidel coefficients non-zero
# ============================================================================

class TestMirrorSeidelNotZero:
    """Pre-4.10 the mirror branch in ``seidel_coefficients`` updated
    paraxial ray heights but never wrote ``S1..S5`` (always zero).
    4.10 added the Welford form with ``n2 = -n1``.  This pins it:
    a concave mirror has non-zero Petzval (S4).
    """

    def test_concave_mirror_has_nonzero_petzval(self):
        from lumenairy.raytrace import seidel_coefficients, Surface
        R = -100e-3
        surfaces = [
            Surface(radius=R, thickness=0.0,
                    glass_before='air', glass_after='air',
                    is_mirror=True, is_stop=True,
                    semi_diameter=10e-3),
        ]
        wavelength = 0.55e-6
        result, _ = seidel_coefficients(
            surfaces, wavelength=wavelength,
            field_angle=np.radians(0.5))
        S4_total = float(result['total']['S4'])
        # For a mirror, Welford gives
        #   S4 = -(1/(n1 n2)) c (n2 - n1) ; n2 = -n1
        #      = -(1/(-n1^2)) c (-2 n1)
        #      =   -2 c / n1
        # With n1 = 1 (air) and c = 1/R = 1/-0.1 = -10 m^-1, S4 = +20.
        assert abs(S4_total) > 1.0, (
            f"Mirror Seidel S4 (Petzval total) = {S4_total!r}; "
            f"expected O(1/|R|) ~ {abs(1.0/R):.1f}.  Pre-4.10 the "
            f"mirror branch never wrote S1..S5, so every catadioptric "
            f"design reported S4=0 by silent oversight."
        )


# ============================================================================
# C-SC-1 -- Tilted-ASM band-limit produces non-zero output
# ============================================================================

class TestTiltedAsmBandlimitNonZero:
    """Pre-4.10 the default ``bandlimit=True`` mask was centred on the
    baseband (``|FX| < fx_max``) instead of the tilt-shifted band
    (``|FX + fx0| < fx_max``).  For any non-trivial tilt this killed
    the energy-bearing modes and zeroed the output.  4.10 centres
    the mask on ``FX + fx0``.  Pin: a tilted plane-wave produces
    a non-zero output.
    """

    def test_nonzero_output_for_tilted_plane_wave(self):
        N, dx, wl = 64, 5e-6, 1.55e-6
        E_in = np.ones((N, N), dtype=np.complex128)
        tilt_x = np.radians(5.0)
        z = 1e-3
        E_out = lm.angular_spectrum_propagate_tilted(
            E_in, z=z, wavelength=wl, dx=dx,
            tilt_x=tilt_x, tilt_y=0.0, bandlimit=True)
        rms = float(np.sqrt(np.mean(np.abs(E_out) ** 2)))
        assert rms > 1e-3, (
            f"angular_spectrum_propagate_tilted(tilt=5 deg, "
            f"bandlimit=True) rms = {rms:.3e}; expected O(1).  "
            f"Pre-4.10 the baseband-centred mask killed the "
            f"carrier-bearing modes."
        )


# ============================================================================
# H-RT-2 -- trace_jax raises on unsupported surface types
# ============================================================================

class TestTraceJaxRaisesOnMirror:
    """Pre-4.10 ``trace_jax`` silently treated mirror / coord-break /
    biconic / freeform surfaces as flat refractives.  4.10 added an
    explicit ``NotImplementedError`` at trace-build time.
    """

    def test_trace_jax_raises_on_mirror(self):
        jax = pytest.importorskip('jax')
        from lumenairy.raytrace.jax_trace import (
            trace_jax, make_jax_ray_state)
        prescription = {
            'aperture_diameter': 20e-3,
            'surfaces': [
                {'radius': -100e-3, 'conic': 0.0,
                 'aspheric_coeffs': None,
                 'radius_y': None, 'conic_y': None,
                 'aspheric_coeffs_y': None,
                 'is_mirror': True,
                 'glass_before': 'air', 'glass_after': 'air'},
                {'radius': float('inf'), 'conic': 0.0,
                 'aspheric_coeffs': None,
                 'radius_y': None, 'conic_y': None,
                 'aspheric_coeffs_y': None,
                 'glass_before': 'air', 'glass_after': 'air'},
            ],
            'thicknesses': [1e-3],
        }
        # Five rays starting on-axis, parallel.
        n_rays = 5
        zero = np.zeros(n_rays, dtype=np.float64)
        ones = np.ones(n_rays, dtype=np.float64)
        state = make_jax_ray_state(
            zero, zero, zero, zero, zero, ones)
        with pytest.raises(NotImplementedError):
            _ = trace_jax(state, prescription, wavelength=0.55e-6)
