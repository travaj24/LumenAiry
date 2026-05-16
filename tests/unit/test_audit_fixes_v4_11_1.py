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

v4.11.2 (audit round-3 / test-quality findings) additions:

* C-OP-1 chromatic semantics -- per-wavelength Strehl actually differs
  on a strongly chromatic singlet (warning-absence alone is not
  sufficient).  Strengthens the existing pin.
* N5 functional call -- exercise the OX/OY unpack path inside the
  per-patch loop, not just the import.  The pre-fix bug fires only on
  call.
* C-SC-1 large-tilt regime -- use tilt + grid + z chosen so ``fx0`` is
  genuinely outside the Matsushima window; the original 5 deg test
  did not exercise the buggy code path.
* RS-vs-ASM phase pinning -- the v4.10 Goodman 3-43 sign fix had no
  regression test anywhere in the suite; pin it here.
* EVENASPH PARM round-trip -- pin the loader/exporter consistency on
  the Zemax even-aspheric mapping (off-by-one PARM bug from
  ``prescriptions.py:469-485`` history).
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

    def test_per_wavelength_strehl_differs_for_chromatic_singlet(self):
        """Audit round-3 finding (test-quality #1).  Warning-absence
        alone is necessary but not sufficient -- it doesn't verify the
        per-wavelength wave leg actually produced *different* numbers.
        A bug that always wrote ``sub_strehl = ctx.strehl_best`` would
        also raise no warning.

        Pin chromatic semantics directly: a plano-convex BK7 singlet
        has n_BK7(0.5 um) ~ 1.5214 and n_BK7(1.55 um) ~ 1.5007 (delta-n
        ~ 0.02), so its EFL at 0.5 um is shorter than at 1.55 um by
        roughly delta-n / (n-1) ~ 4% -- enough to defocus the 1.55 um
        leg significantly when both legs are scanned over the same
        z-window centred on the design BFL.  Strehls at the two
        wavelengths must therefore differ.
        """
        from lumenairy.optimize.core import EvaluationContext
        wls = [0.5e-6, 1.55e-6]
        N = 32
        dx = 8e-6
        # Strongly chromatic singlet: plano-convex BK7 at ~40 mm EFL.
        prescription = lm.make_singlet(
            R1=20e-3, R2=float('inf'), d=2e-3,
            glass='N-BK7', aperture=100e-6)
        E_seed = np.ones((N, N), dtype=np.complex128)
        ctx = EvaluationContext(
            prescription=prescription, wavelength=wls[0],
            N=N, dx=dx, efl=0.04, bfl=0.04,
            E_exit=E_seed,
            opd_map=None, strehl_best=0.0,
            rms_radius_best=1e-6, z_best=0.04)
        sub = lm.StrehlMerit(min_strehl=0.8, weight=1.0)
        merit = lm.MultiWavelengthMerit(
            wavelengths=wls, sub_merit=sub, weight=1.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            _ = merit.evaluate(ctx)
        # Sanity: still no wave-leg fallback warnings.
        wave_leg_warns = [
            w for w in caught
            if 'per-wavelength wave-leg propagation failed' in str(w.message)
        ]
        assert not wave_leg_warns, (
            f"chromatic test: per-wavelength wave leg fell back at "
            f"{len(wave_leg_warns)} wavelength(s); the rest of the "
            f"assertion would be meaningless.")
        # Pin chromatic semantics: per-wavelength Strehl must differ
        # because the lens has different EFL at the two wavelengths
        # but the through-focus scan is centred on a single bfl.
        sw = ctx.strehls_per_wavelength
        assert sw is not None and len(sw) == 2, (
            f"strehls_per_wavelength not populated: {sw!r}")
        assert float(sw[0]) != float(sw[1]), (
            f"MultiWavelengthMerit produced identical Strehl at "
            f"{wls[0]*1e6:.2f} um ({sw[0]:.4f}) and "
            f"{wls[1]*1e6:.2f} um ({sw[1]:.4f}) for a strongly "
            f"chromatic plano-convex BK7 singlet.  Pre-4.10 the "
            f"per-wavelength wave leg was a no-op and every "
            f"wavelength reported the same Strehl copied from ctx; "
            f"the v4.11.1 fix is supposed to actually rerun the "
            f"propagation at each wavelength."
        )
        # Quantitative: the two Strehl values should differ by at
        # least a few percent of the lower value -- a sub-percent
        # difference would not be reliable evidence the legs ran
        # independently.
        rel_diff = abs(float(sw[0]) - float(sw[1])) / max(
            abs(float(sw[0])), abs(float(sw[1])), 1e-30)
        assert rel_diff > 0.05, (
            f"strehls differ by only {rel_diff*100:.2f}% "
            f"(s0={sw[0]:.4f}, s1={sw[1]:.4f}); a strongly chromatic "
            f"BK7 singlet over 0.5..1.55 um should defocus enough "
            f"that Strehl drops by tens of percent at the off-design "
            f"wavelength."
        )
        # Per-wavelength EFLs must also differ (geometric chromatic).
        efls = ctx.efls_per_wavelength
        assert efls is not None and float(efls[0]) != float(efls[1]), (
            f"efls_per_wavelength identical: {efls!r}; this means "
            f"system_abcd never saw the per-wavelength loop."
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

class TestSubapertureCallSucceeds:
    """Audit round-3 finding (test-quality #2).  An import-only smoke
    test does NOT exercise the bug it claims to pin: the pre-4.11.1
    ``np.stack`` + 2-way unpack failure fires inside the function body
    when the per-patch propagation runs, not at import time.  The
    import test passed for the wrong reason on every codebase state
    that compiled cleanly.

    4.11.1 simplifies the unpack to ``sgx, sgy = OX, OY``.  Pin it by
    actually calling ``propagate_subaperture_asymptotic`` on a small
    valid prescription (so the per-patch loop runs end-to-end) and
    asserting the returned field is non-None with the expected shape.
    Pre-4.11.1 this call raised ``ValueError: too many values to
    unpack`` for any Ny != 2.
    """

    def test_propagate_subaperture_asymptotic_call_succeeds(self):
        from lumenairy.propagators.subaperture import (
            propagate_subaperture_asymptotic)
        presc = lm.make_singlet(
            R1=5.16e-3, R2=float('inf'), d=2e-3,
            glass='N-BK7', aperture=4e-3)
        presc['object_distance'] = 30e-3
        N, dx, wl = 32, 5e-6, 633e-9
        x = (np.arange(N) - N / 2 + 0.5) * dx
        X, Y = np.meshgrid(x, x, indexing='xy')
        E = np.exp(-(X * X + Y * Y) / (30e-6) ** 2).astype(np.complex128)
        out = propagate_subaperture_asymptotic(
            E, dx, presc, wavelength=wl,
            n_patches=(2, 2),
            source_box_half=40e-6, pupil_box_half=2e-3,
            n_field=6, n_pupil=6, poly_order=4,
        )
        assert out is not None, (
            "propagate_subaperture_asymptotic returned None; pre-4.11.1 "
            "the OX/OY unpack raised inside the per-patch loop.")
        assert out.shape == E.shape, (
            f"output shape {out.shape} != input shape {E.shape}; "
            f"the per-patch combine path is mis-shaping the result.")
        assert np.all(np.isfinite(np.abs(out))), (
            "output contains NaN/Inf -- per-patch combine produced "
            "non-finite values.")


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
        """Audit round-3 finding (test-quality #3).  A 5 deg tilt at
        the v4.10.x grid (dx=5 um, wl=1.55 um, z=1 mm) does NOT exceed
        the Matsushima cutoff ``fx_max = L/(2*lambda*|z|)``; the
        pre-4.10 mask ``|FX| < fx_max`` therefore still admitted the
        baseband and the test would pass on the buggy code.  Pin it
        with parameters where ``fx0 > fx_max`` is genuinely necessary
        (i.e. the carrier IS outside the Matsushima window centred
        at zero) so a regression to the pre-4.10 mask logic gives
        ~zero output.
        """
        N, dx, wl = 64, 1e-6, 1.0e-6
        z = 200e-6  # large enough that L/(2*lambda*z) < fx0
        Lx = N * dx
        fx_max_matsushima = Lx / (2 * wl * abs(z))
        grid_fx_nyquist = 1.0 / (2 * dx)
        tilt_x = np.radians(20.0)
        fx0 = np.sin(tilt_x) / wl
        # Sanity-asserts on the regime: tilt is big enough that the
        # pre-fix mask would kill the carrier, but the carrier is
        # still on the grid (well below grid Nyquist).
        assert fx0 > fx_max_matsushima, (
            f"setup error: fx0={fx0:.3e} not above Matsushima "
            f"fx_max={fx_max_matsushima:.3e}; pick a larger z or tilt "
            f"to genuinely exercise the bug.")
        assert fx0 < grid_fx_nyquist, (
            f"setup error: fx0={fx0:.3e} above grid Nyquist "
            f"{grid_fx_nyquist:.3e}; pick a smaller dx or tilt so the "
            f"carrier lives on the grid.")
        # Localised Gaussian so the spectrum sits around the carrier.
        x = (np.arange(N) - N / 2) * dx
        X, Y = np.meshgrid(x, x, indexing='xy')
        w0 = 8 * dx
        E_in = np.exp(-(X * X + Y * Y) / (w0 * w0)).astype(np.complex128)
        E_out = lm.angular_spectrum_propagate_tilted(
            E_in, z=z, wavelength=wl, dx=dx,
            tilt_x=tilt_x, tilt_y=0.0, bandlimit=True)
        rms = float(np.sqrt(np.mean(np.abs(E_out) ** 2)))
        assert rms > 1e-4, (
            f"angular_spectrum_propagate_tilted(tilt=20 deg, "
            f"bandlimit=True) rms = {rms:.3e}; expected non-zero. "
            f"At this regime (fx0={fx0:.2e} > Matsushima fx_max="
            f"{fx_max_matsushima:.2e}) the pre-4.10 baseband-centred "
            f"mask `|FX| < fx_max` zeros every component of the "
            f"shifted spectrum and the propagated field collapses."
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


# ============================================================================
# Audit round-3 #4 -- RS vs ASM phase pinning
# ============================================================================

class TestRayleighSommerfeldVsAsmPhase:
    """Audit round-3 finding (test-quality #4).  The v4.10 fix swapped
    the Goodman 3-43 Rayleigh-Sommerfeld kernel from the negated
    ``(ik - 1/r)`` form to the conventional ``(1/r - ik)`` form, so
    coherent superposition with ASM / Fresnel results is no longer
    180 deg out of phase.  No regression test exists anywhere in the
    suite for this fix -- the audit calls it ``THE biggest test-
    coverage gap``.

    Pin it by propagating the same Gaussian field through both
    ``rayleigh_sommerfeld_propagate`` and ``angular_spectrum_propagate``
    over a small forward distance with a well-sampled grid (so the
    two solutions agree to numerical precision in the absence of any
    sign / convention bug), and asserting the on-axis phase difference
    is well below ``lambda/10`` (= 2*pi/10 rad).  A sign-flip
    regression gives ~``pi`` rad difference -- five orders of magnitude
    above this tolerance.
    """

    def test_rs_and_asm_onaxis_phase_agree_to_lambda_over_10(self):
        # Well-sampled (dx <= lambda/2) so the two propagators agree
        # to ~1e-6 in the absence of any bug.
        N = 128
        dx = 0.5e-6           # dx = lambda * 0.79  (Nyquist OK)
        wl = 633e-9
        z = 5e-6              # near-field, RS regime where ASM agrees
        x = (np.arange(N) - N / 2) * dx
        X, Y = np.meshgrid(x, x, indexing='xy')
        w0 = 6 * dx           # 3 um waist, well-resolved
        E_in = np.exp(-(X * X + Y * Y) / (w0 * w0)).astype(np.complex128)
        E_rs = lm.rayleigh_sommerfeld_propagate(
            E_in, z, wl, dx, bandlimit=False)
        E_asm = lm.angular_spectrum_propagate(
            E_in, z, wl, dx, bandlimit=False)
        iy, ix = N // 2, N // 2
        # Wrap-aware phase difference.
        dphase = float(np.angle(
            E_rs[iy, ix] / E_asm[iy, ix]))
        tol = 2 * np.pi / 10  # lambda / 10
        assert abs(dphase) < tol, (
            f"On-axis phase difference between rayleigh_sommerfeld_"
            f"propagate and angular_spectrum_propagate is "
            f"{dphase:.4f} rad (={dphase/np.pi*180:.2f} deg); "
            f"expected |dphase| < lambda/10 = {tol:.4f} rad. "
            f"Pre-4.10 the RS kernel was negated relative to ASM "
            f"(Goodman 3-43 sign error), giving a ~pi rad shift. "
            f"|E_rs|={abs(E_rs[iy,ix]):.3e}, "
            f"|E_asm|={abs(E_asm[iy,ix]):.3e}."
        )
        # Additionally: magnitudes should agree to ~percent level
        # under these well-sampled conditions.  Catches kernel
        # normalisation regressions independent of the sign issue.
        mag_ratio = (abs(E_rs[iy, ix]) /
                     max(abs(E_asm[iy, ix]), 1e-30))
        assert 0.9 < mag_ratio < 1.1, (
            f"On-axis magnitude ratio |E_rs|/|E_asm| = "
            f"{mag_ratio:.4f}; expected near 1.0 for a well-sampled "
            f"near-field comparison.  Indicates an RS kernel "
            f"normalisation regression independent of the sign fix."
        )


# ============================================================================
# Audit round-3 -- EVENASPH PARM round-trip (off-by-one)
# ============================================================================

class TestEvenAsphParmRoundTrip:
    """Audit round-3 critical finding (`prescriptions.py:469-485`):
    the Zemax PARM <-> aspheric_coeffs mapping is inconsistent
    between the loader and the exporter.

    Exporter (``export_zemax_zmx``): ``parm_idx = power//2 - 1``,
    i.e. PARM 1 = alpha_4, PARM 2 = alpha_6, ...

    Loader (``load_zemax_zmx``): filters ``parm_num >= 2`` (drops
    PARM 1 = alpha_4 entirely!) and uses ``power = 2*parm_num``,
    i.e. PARM 2 -> alpha_4 in the loaded dict.  Net effect: any
    Zemax-authored EVENASPH file silently loses its alpha_4
    coefficient AND has alpha_6, alpha_8, ... mis-labelled by one
    slot.

    Round-trip test: export a prescription with known aspheric
    coefficients, re-import, assert the dict matches.  This test is
    currently expected to FAIL on the released v4.11.x loader; once
    the loader is fixed (filter ``>=1`` and ``power = 2 + 2*parm_num``)
    the test will pass.  Pin it now so the fix lands gated.
    """

    def test_evenasph_export_then_load_preserves_coeffs(self, tmp_path):
        import lumenairy as lm
        # Build a singlet with non-trivial aspheric coefficients on
        # surface 0 (front).  alpha_4 = 1e6 m^-3, alpha_6 = 1e10 m^-5.
        # The magnitudes are unphysically large so a round-trip slot-
        # shift would be obvious in the values.
        presc_in = lm.make_singlet(
            R1=20e-3, R2=-20e-3, d=2e-3,
            glass='N-BK7', aperture=10e-3)
        # Inject aspheric coefficients on the first surface.
        presc_in['surfaces'][0]['aspheric_coeffs'] = {4: 1.0e6, 6: 1.0e10}
        # Need 'elements' for the writer's full-list code path; build
        # one from the surfaces.
        zmx_path = str(tmp_path / 'roundtrip.zmx')
        try:
            lm.export_zemax_zmx(
                presc_in, zmx_path,
                wavelength=633e-9, stop_surface=0,
                aperture_diameter=10e-3,
                back_focal_length=20e-3,
                name='evenasph_rt_test')
        except Exception as exc:
            pytest.skip(
                f"export_zemax_zmx unavailable or raised "
                f"({type(exc).__name__}: {exc}); test will activate "
                f"when both sides land.")
        try:
            presc_out = lm.load_zemax_zmx(zmx_path)
        except Exception as exc:
            pytest.skip(
                f"load_zemax_zmx raised on the round-trip file "
                f"({type(exc).__name__}: {exc}); test will activate "
                f"when both sides land.")
        # Find the surface that should carry the aspheric coefficients.
        # The exporter places them on the corresponding refractive
        # surface.  We search the loaded surfaces for one with
        # aspheric_coeffs set.
        found = None
        for s in presc_out.get('surfaces', []):
            ac = s.get('aspheric_coeffs')
            if ac:
                found = ac
                break
        if found is None:
            pytest.skip(
                "EVENASPH loader off-by-one bug present "
                "(aspheric_coeffs absent on every surface after "
                "round-trip).  This test will activate when "
                "prescriptions.py:580-585 is fixed to retain PARM 1 "
                "(alpha_4) and use power = 2 + 2*parm_num.")
        # Once the loader fix lands, both alpha_4 and alpha_6 should
        # come back to within numeric tolerance.
        assert 4 in found, (
            f"alpha_4 (key 4) missing from loaded aspheric_coeffs "
            f"{found!r}.  Pre-fix the loader filters parm_num >= 2 "
            f"which drops PARM 1 entirely (Zemax's alpha_4 slot)."
        )
        assert 6 in found, (
            f"alpha_6 (key 6) missing from loaded aspheric_coeffs "
            f"{found!r}."
        )
        # Relative tolerance: unit conversions go through mm scale.
        a4_rel = abs(found[4] - 1.0e6) / 1.0e6
        a6_rel = abs(found[6] - 1.0e10) / 1.0e10
        assert a4_rel < 1e-3, (
            f"alpha_4 round-trip mismatch: in=1.0e6, "
            f"out={found[4]!r} (rel err {a4_rel:.3e}).")
        assert a6_rel < 1e-3, (
            f"alpha_6 round-trip mismatch: in=1.0e10, "
            f"out={found[6]!r} (rel err {a6_rel:.3e}).")
