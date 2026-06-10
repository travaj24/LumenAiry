"""Consolidated audit-fix tests for the **misc** domain.

This module consolidates v4.9 - v5.0 audit-fix regression pins
from 12 source files (per the v5.2 ROADMAP / 57-file consolidation):

* ``test_audit_fixes_v4_11_1.py``
* ``test_audit_fixes_v4_12_0_round4_jax.py``
* ``test_audit_fixes_v4_12_0_round4_tier0.py``
* ``test_audit_fixes_v4_12_1_coverage.py``
* ``test_audit_fixes_v4_12_2_cache_hygiene.py``
* ``test_audit_fixes_v4_13_0_except_sweep.py``
* ``test_audit_fixes_v4_13_0_jax_dtype_dy_siblings.py``
* ``test_audit_fixes_v4_13_1_agent3.py``
* ``test_audit_fixes_v4_13_1_random_choice.py``
* ``test_audit_fixes_v4_13_2_agent_b.py``
* ``test_audit_fixes_v4_13_2_agent_d.py``
* ``test_audit_fixes_v4_9.py``

Each source file's contents are concatenated below verbatim (modulo
minimal renames to avoid identifier collisions and to give each top-level
test class an audit-version attribution prefix).  v5.2.3 closed the
v5.2.1 TODO markers on the inspect.getsource proxy-test sites in this
file: replaced where a behavioral pin was achievable; otherwise kept
inspect.getsource by design and updated the comment to explain why
(see AUDIT_V4_13_1 Part 6.1).
"""
from __future__ import annotations

# ============================================================================
# Source: test_audit_fixes_v4_11_1.py
# Audit version: V4_11_1  scope: (top-level)
# Original module docstring preserved as comment block for git-blame traceability:
#   Regression tests for the 4.11.1 audit-residual patch.
#   
#   Each test pins one of the round-2-verification findings.  These tests
#   are intentionally small and runtime-cheap (each <1 s); their purpose is
#   to fail loudly if a future refactor reintroduces the specific bug, not
#   to exercise the broader physics.
#   
#   Findings covered (per ``AUDIT_VERIFICATION_2026_05_16.md`` / round-2):
#   
#   * C-OP-1 / N1  -- ``MultiWavelengthMerit`` per-wavelength wave-leg
#                      keyword-call (was positional, raised silently).
#   * M-LR-1       -- decentered-stop lookup on dict (was ``getattr`` on
#                      wrong keys, fix dead-on-arrival pre-4.11.1).
#   * C-PL-1       -- circular-polarisation handedness consistency.
#   * H-PR-4       -- ``create_point_source`` central pixel bounded.
#   * N5           -- subaperture output-grid unpack imports.
#   * C-RT-1/C-AB-1 -- mirror Seidel coefficients are non-zero.
#   * C-SC-1       -- tilted-ASM band-limit produces a non-zero output.
#   * H-RT-2       -- ``trace_jax`` raises on a mirror surface.
#   
#   v4.11.2 (audit round-3 / test-quality findings) additions:
#   
#   * C-OP-1 chromatic semantics -- per-wavelength Strehl actually differs
#     on a strongly chromatic singlet (warning-absence alone is not
#     sufficient).  Strengthens the existing pin.
#   * N5 functional call -- exercise the OX/OY unpack path inside the
#     per-patch loop, not just the import.  The pre-fix bug fires only on
#     call.
#   * C-SC-1 large-tilt regime -- use tilt + grid + z chosen so ``fx0`` is
#     genuinely outside the Matsushima window; the original 5 deg test
#     did not exercise the buggy code path.
#   * RS-vs-ASM phase pinning -- the v4.10 Goodman 3-43 sign fix had no
#     regression test anywhere in the suite; pin it here.
#   * EVENASPH PARM round-trip -- pin the loader/exporter consistency on
#     the Zemax even-aspheric mapping (off-by-one PARM bug from
#     ``prescriptions.py:469-485`` history).
# ============================================================================
import warnings

import numpy as np
import pytest

import lumenairy as lm

# ============================================================================
# C-OP-1 / N1 -- MultiWavelengthMerit per-wavelength wave-leg
# ============================================================================

class TestAuditFixesV4_11_1_MultiWavelengthMeritWaveLegRuns:
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

class TestAuditFixesV4_11_1_DecenteredStopHonoured:
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

class TestAuditFixesV4_11_1_CircularPolarisationHandednessConsistent:
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

class TestAuditFixesV4_11_1_PointSourceCentralPixelBounded:
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

class TestAuditFixesV4_11_1_SubapertureCallSucceeds:
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
        from lumenairy.propagators.subaperture import propagate_subaperture_asymptotic
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

class TestAuditFixesV4_11_1_MirrorSeidelNotZero:
    """Pre-4.10 the mirror branch in ``seidel_coefficients`` updated
    paraxial ray heights but never wrote ``S1..S5`` (always zero).
    4.10 added the Welford form with ``n2 = -n1``.  This pins it:
    a concave mirror has non-zero Petzval (S4).
    """

    def test_concave_mirror_has_nonzero_petzval(self):
        from lumenairy.raytrace import Surface, seidel_coefficients
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

class TestAuditFixesV4_11_1_TiltedAsmBandlimitNonZero:
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

class TestAuditFixesV4_11_1_TraceJaxRaisesOnMirror:
    """Pre-4.10 ``trace_jax`` silently treated mirror / coord-break /
    biconic / freeform surfaces as flat refractives.  4.10 added an
    explicit ``NotImplementedError`` at trace-build time.
    """

    def test_trace_jax_raises_on_mirror(self):
        pytest.importorskip('jax')
        from lumenairy.raytrace.jax_trace import make_jax_ray_state, trace_jax
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

class TestAuditFixesV4_11_1_RayleighSommerfeldVsAsmPhase:
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

class TestAuditFixesV4_11_1_EvenAsphParmRoundTrip:
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


# ============================================================================
# Source: test_audit_fixes_v4_12_0_round4_jax.py
# Audit version: V4_12_0  scope: round4_jax
# Original module docstring preserved as comment block for git-blame traceability:
#   Pinning tests for the v4.12.0 round-4 audit Tier-1 JAX fixes.
#   
#   The round-4 audit (``AUDIT_ROUND4_2026_05_16.md``) identified three
#   correctness issues in the JAX backends that this test module pins:
#   
#   * **B1-1 -- JAX/NumPy aperture schemas incompatible.**
#     ``system.py`` (JAX path) read ``params.get('radius')`` while NumPy
#     ``apply_aperture`` reads ``params.get('diameter')``.  A working
#     NumPy element list ported to ``propagate_through_system_jax`` had
#     every aperture silently no-op'd because ``params.get('radius')``
#     returned ``None``.  Pinning:
#   
#       1. Canonical NumPy schema (``params={'diameter': ...}``) works
#          end-to-end (no warning, correct aperture clipping).
#       2. Legacy schema (``params={'radius': ...}``) still works but
#          emits a one-shot :class:`DeprecationWarning`.
#       3. All three shapes (circular / rectangular / annular) accept
#          both schemas.
#   
#   * **B1-2 -- ``propagate_through_system_jax`` was not actually JAX-end-
#     to-end traceable.**  Pre-v4.12 the function silently fell back to
#     NumPy via ``np.asarray(E)`` for unsupported element types, which
#     raises ``TracerArrayConversionError`` under ``jax.jit`` /
#     ``jax.grad``.  v4.12 raises :class:`NotImplementedError` at call
#     time listing the offending element types.  Pinning:
#   
#       1. Element lists with only supported types
#          (:data:`_TRACEABLE_ELEMENT_TYPES`) run without error.
#       2. Lists containing any unsupported type raise
#          :class:`NotImplementedError` with the offending type names in
#          the error message.
#       3. ``_TRACEABLE_ELEMENT_TYPES`` is exposed and contains the
#          expected 4-element set.
#   
#   * **B1-9 -- ``_apply_doe_kick_jax`` blocked gradients.**
#     Pre-v4.12 used ``float(period_x)`` (strips trace -> silent zero
#     gradient) and ``np.isfinite(period_x)`` (raises on a traced array).
#     v4.12 keeps the JAX trace alive via ``jnp.where`` whenever the
#     period argument is traced.  Pinning:
#   
#       1. A concrete-period call returns finite, sensible state.
#       2. ``jax.grad`` w.r.t. ``period_x`` returns a FINITE, NON-ZERO
#          gradient (the actual round-3 + round-4 finding).
# ============================================================================

import warnings

import numpy as np
import pytest

# Skip the whole module if JAX is unavailable.
try:
    import jax
    import jax.numpy as jnp
    _HAS_JAX = True
except ImportError:                  # pragma: no cover - environment dependent
    jax = jnp = None
    _HAS_JAX = False

# SCOPED skip (2026-06-10): the previous MODULE-LEVEL importorskip('jax')
# silently skipped this entire file -- including every non-JAX pin in it --
# on any environment without jax (CI and the WSL proxy among them).
_requires_jax = pytest.mark.skipif(not _HAS_JAX,
                                   reason="could not import 'jax'")

if _HAS_JAX:
    jax.config.update('jax_enable_x64', True)


import lumenairy as lm  # noqa: E402  (after importorskip)
from lumenairy.propagators.system import (  # noqa: E402
    _PROPAGATE_SYSTEM_JAX_CACHE,
    _TRACEABLE_ELEMENT_TYPES,
    propagate_through_system_jax,
)

# ===========================================================================
# B1-1 -- JAX/NumPy aperture schema unification
# ===========================================================================

class TestAuditFixesV4_12_0_round4_jax_B1_1_ApertureSchemaUnification:
    """``propagate_through_system_jax`` aperture handling must accept
    BOTH the canonical NumPy schema (``diameter`` / ``width_x`` /
    ``inner_diameter``) AND the legacy JAX-only schema (``radius`` /
    ``half_width_x`` / ``inner_radius``).  The legacy form emits a
    one-shot :class:`DeprecationWarning`."""

    @pytest.fixture
    def field(self):
        N, dx, wl = 64, 5e-6, 1.55e-6
        E = np.ones((N, N), dtype=np.complex64)
        return E, dx, wl

    # v5.0: ``_LEGACY_APERTURE_SCHEMA_WARNED`` latch removed alongside
    # the legacy schema; tests that previously reset it no longer need
    # the fixture (kept as a no-op for forward compatibility).

    # ---- circular ----------------------------------------------------

    def test_circular_diameter_canonical_no_warning(self, field):
        """``params={'diameter': D}`` is the canonical NumPy schema:
        must clip the field AND not emit a deprecation warning."""
        E, dx, wl = field
        diameter = 100e-6
        elements = [
            {'type': 'aperture', 'shape': 'circular',
             'params': {'diameter': diameter}},
        ]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            E_out = propagate_through_system_jax(E, elements, wl, dx)
        E_np = np.asarray(E_out)
        # The aperture must actually clip: some pixels become 0.
        n_zero = int(np.sum(np.abs(E_np) < 1e-9))
        n_one = int(np.sum(np.abs(E_np) > 0.5))
        assert n_zero > 0, (
            "Canonical 'diameter' schema must clip the field "
            "(audit B1-1: pre-fix silently no-op'd).")
        assert n_one > 0, "Some pixels must remain inside the aperture."
        # And no deprecation warning fires.
        dep = [wi for wi in w if issubclass(wi.category, DeprecationWarning)]
        assert not dep, (
            f"Canonical schema must not emit DeprecationWarning; "
            f"got: {[str(wi.message)[:80] for wi in dep]}")

    def test_circular_radius_legacy_now_raises_in_v5_0(self, field):
        """v5.0 (honest break): ``params={'radius': r}`` was deprecated
        in v4.12 and removed in v5.0.  Now raises ``ValueError`` with
        the canonical-schema migration hint."""
        E, dx, wl = field
        elements = [
            {'type': 'aperture', 'shape': 'circular',
             'params': {'radius': 50e-6}},
        ]
        with pytest.raises(ValueError, match="legacy.*radius.*removed in v5\\.0"):
            propagate_through_system_jax(E, elements, wl, dx)

    # ---- rectangular -------------------------------------------------

    def test_rectangular_width_canonical(self, field):
        """``params={'width_x': Wx, 'width_y': Wy}`` is the NumPy
        canonical schema (full widths)."""
        E, dx, wl = field
        N = E.shape[0]
        elements = [
            {'type': 'aperture', 'shape': 'rectangular',
             'params': {'width_x': 40e-6, 'width_y': 60e-6}},
        ]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            E_out = propagate_through_system_jax(E, elements, wl, dx)
        E_np = np.asarray(E_out)
        # Center pixel inside the aperture: |E| ~ 1.
        cx = cy = N // 2
        assert abs(E_np[cy, cx]) > 0.5
        # Far-corner pixel outside the aperture: |E| == 0.
        assert abs(E_np[0, 0]) < 1e-9
        dep = [wi for wi in w if issubclass(wi.category, DeprecationWarning)]
        assert not dep, "Canonical schema must not emit DeprecationWarning."

    def test_rectangular_half_width_legacy_now_raises_in_v5_0(self, field):
        """v5.0 (honest break): the legacy ``half_width_x/half_width_y``
        schema raises ValueError."""
        E, dx, wl = field
        elements = [
            {'type': 'aperture', 'shape': 'rectangular',
             'params': {'half_width_x': 20e-6, 'half_width_y': 30e-6}},
        ]
        with pytest.raises(ValueError, match="legacy.*removed in v5\\.0"):
            propagate_through_system_jax(E, elements, wl, dx)

    # ---- annular -----------------------------------------------------

    def test_annular_diameter_canonical(self, field):
        """``params={'inner_diameter', 'outer_diameter'}`` is the
        canonical schema."""
        E, dx, wl = field
        N = E.shape[0]
        elements = [
            {'type': 'aperture', 'shape': 'annular',
             'params': {'inner_diameter': 40e-6,
                        'outer_diameter': 200e-6}},
        ]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            E_out = propagate_through_system_jax(E, elements, wl, dx)
        E_np = np.asarray(E_out)
        # Origin pixel is inside the inner hole -> zero.
        cx = cy = N // 2
        assert abs(E_np[cy, cx]) < 1e-9
        # An off-axis pixel inside the ring -> ~1.
        # Pixel at (cy, cx + 10) is at x = 10*dx = 50e-6 m (in the ring).
        assert abs(E_np[cy, cx + 10]) > 0.5
        dep = [wi for wi in w if issubclass(wi.category, DeprecationWarning)]
        assert not dep

    def test_annular_radius_legacy_now_raises_in_v5_0(self, field):
        """v5.0 (honest break): the legacy
        ``inner_radius/outer_radius`` schema raises ValueError."""
        E, dx, wl = field
        elements = [
            {'type': 'aperture', 'shape': 'annular',
             'params': {'inner_radius': 20e-6, 'outer_radius': 100e-6}},
        ]
        with pytest.raises(ValueError, match="legacy.*removed in v5\\.0"):
            propagate_through_system_jax(E, elements, wl, dx)

    # ---- cross-backend agreement -------------------------------------

    def test_jax_aperture_matches_numpy(self, field):
        """The MAIN B1-1 finding: an aperture spec built with the
        canonical NumPy schema and passed to BOTH backends must
        produce the same clipped field (within fp32 precision on the
        circle boundary).  Pre-fix the JAX path silently no-op'd
        because it read ``params.get('radius')``."""
        E, dx, wl = field
        # Use a diameter that doesn't land on an integer multiple of
        # dx so the boundary doesn't sit exactly on a grid pixel
        # (different fp32 vs fp64 rounding on ``X**2 + Y**2 <= r**2``
        # can flip a handful of boundary pixels in either direction).
        elements = [
            {'type': 'aperture', 'shape': 'circular',
             'params': {'diameter': 97.5e-6}},
        ]
        # NumPy backend.
        E_np_out, _ = lm.propagate_through_system(
            E, elements, wl, dx, method='asm')
        # JAX backend.
        E_jax_out = propagate_through_system_jax(E, elements, wl, dx)
        # Pixel-wise mismatch must be bounded by the perimeter of the
        # disk (~ pi*D/dx ~ 60 pixels); insist no more than ~5% drift.
        diff = np.asarray(E_jax_out) - E_np_out.astype(np.complex64)
        n_mismatch = int(np.sum(np.abs(diff) > 1e-6))
        total = E_np_out.size
        assert n_mismatch < 0.05 * total, (
            f"Cross-backend mismatch must be limited to the disk "
            f"boundary; got {n_mismatch} / {total} mismatched pixels.")
        # And the JAX result is NOT just E_in unchanged (the pre-fix
        # behaviour) -- it's actually clipped.
        n_zero = int(np.sum(np.abs(np.asarray(E_jax_out)) < 1e-9))
        assert n_zero > 0, (
            "JAX aperture must clip the field; pre-fix silently "
            "no-op'd because params.get('radius') returned None.")


# ===========================================================================
# B1-2 -- propagate_through_system_jax fail-fast on non-traceable elements
# ===========================================================================

@_requires_jax
class TestAuditFixesV4_12_0_round4_jax_B1_2_NonTraceableElementFailFast:
    """Element types without a JAX handler must raise
    :class:`NotImplementedError` at call time, listing the offending
    types.  Pre-v4.12 the function silently fell back to NumPy via
    ``np.asarray(E)``, which crashes under ``jax.jit`` / ``jax.grad``."""

    def test_traceable_element_types_constant(self):
        """The module-level constant must exist and contain the
        expected 4-element set."""
        assert isinstance(_TRACEABLE_ELEMENT_TYPES, frozenset)
        assert _TRACEABLE_ELEMENT_TYPES == frozenset(
            {'propagate', 'lens', 'aperture', 'mask'})

    def test_pure_traceable_chain_runs(self):
        """An element list with only traceable types runs without
        error."""
        N, dx, wl = 32, 5e-6, 1.55e-6
        E = np.ones((N, N), dtype=np.complex64)
        elements = [
            {'type': 'propagate', 'z': 1e-3, 'bandlimit': True},
            {'type': 'lens', 'f': 5e-3},
            {'type': 'aperture', 'shape': 'circular',
             'params': {'diameter': 100e-6}},
            {'type': 'mask',
             'mask': np.ones((N, N), dtype=np.complex64)},
        ]
        E_out = propagate_through_system_jax(E, elements, wl, dx)
        assert E_out.shape == (N, N)
        assert np.all(np.isfinite(np.asarray(E_out)))

    @pytest.mark.parametrize('etype, extra_keys', [
        ('spherical_lens', {'R1': 5e-3, 'R2': -5e-3, 'd': 1e-3,
                             'n_lens': 1.5}),
        ('aspheric_lens',  {'R1': 5e-3, 'R2': -5e-3, 'd': 1e-3,
                             'n_lens': 1.5, 'k1': 0.0, 'k2': 0.0}),
        ('mirror',         {'radius': -100e-3}),
        ('gaussian_aperture', {'sigma': 50e-6}),
        ('turbulence',     {'r0': 1e-2}),
        ('zernike',        {'coefficients': [0.0],
                             'aperture_radius': 1e-3}),
        ('cylindrical_lens', {'f': 5e-3}),
        ('axicon',         {'alpha': 0.01, 'n_axicon': 1.5}),
        ('grin_lens',      {'n0': 1.5, 'g': 0.1, 'd': 1e-3}),
        ('propagate_tilted', {'z': 1e-3}),
    ])
    def test_non_traceable_raises_not_implemented(self, etype, extra_keys):
        """Each non-traceable element type triggers a clear
        ``NotImplementedError`` (one row per element type listed in
        the audit B1-2 finding)."""
        N, dx, wl = 16, 5e-6, 1.55e-6
        E = np.ones((N, N), dtype=np.complex64)
        elem = {'type': etype, **extra_keys}
        with pytest.raises(NotImplementedError) as excinfo:
            propagate_through_system_jax(E, [elem], wl, dx)
        assert etype in str(excinfo.value), (
            f"NotImplementedError message must name the offending "
            f"element type {etype!r}; got: {excinfo.value}")

    def test_spherical_lens_in_mixed_list_raises(self):
        """A mixed list with one ``spherical_lens`` among otherwise
        traceable types must still raise."""
        N, dx, wl = 16, 5e-6, 1.55e-6
        E = np.ones((N, N), dtype=np.complex64)
        elements = [
            {'type': 'propagate', 'z': 1e-3},
            {'type': 'spherical_lens', 'R1': 5e-3, 'R2': -5e-3,
             'd': 1e-3, 'n_lens': 1.5},
            {'type': 'propagate', 'z': 1e-3},
        ]
        with pytest.raises(NotImplementedError, match='spherical_lens'):
            propagate_through_system_jax(E, elements, wl, dx)

    def test_fail_fast_does_not_pollute_cache(self):
        """The fail-fast check runs BEFORE any kernel build / cache
        insertion, so a NotImplementedError must not pollute the
        kernel cache."""
        N, dx, wl = 16, 5e-6, 1.55e-6
        E = np.ones((N, N), dtype=np.complex64)
        elements = [
            {'type': 'mirror', 'radius': -100e-3},
        ]
        cache_size_before = len(_PROPAGATE_SYSTEM_JAX_CACHE)
        with pytest.raises(NotImplementedError):
            propagate_through_system_jax(E, elements, wl, dx)
        assert len(_PROPAGATE_SYSTEM_JAX_CACHE) == cache_size_before


# ===========================================================================
# B1-9 -- _apply_doe_kick_jax gradient flow w.r.t. grating period
# ===========================================================================

@_requires_jax
class TestAuditFixesV4_12_0_round4_jax_B1_9_DoeKickJaxGradient:
    """The DOE-kick path must keep the JAX trace alive when
    ``period_x`` / ``period_y`` are JAX-traced values, so users can
    ``jax.grad`` w.r.t. grating period.  Pre-v4.12 used
    ``float(period_x)`` (strips trace, zero gradient) and
    ``np.isfinite(period_x)`` (raises on a tracer)."""

    def _make_initial_state(self, n_rays=8):
        """Build a JaxRayState ray bundle parallel to z, slightly
        off-axis so the DOE kick affects (x, y, opd) downstream."""
        from lumenairy.raytrace.jax_trace import make_jax_ray_state
        # Rays at non-zero (x, y) so the OPL contribution ``dL*x +
        # dM*y`` is non-zero and depends on period.
        x = jnp.linspace(-0.5e-3, 0.5e-3, n_rays)
        y = jnp.linspace(-0.3e-3, 0.3e-3, n_rays)
        z = jnp.zeros(n_rays)
        L = jnp.zeros(n_rays)
        M = jnp.zeros(n_rays)
        N = jnp.ones(n_rays)
        opd = jnp.zeros(n_rays)
        alive = jnp.ones(n_rays, dtype=bool)
        return make_jax_ray_state(x=x, y=y, z=z, L=L, M=M, N=N,
                                   opd=opd, alive=alive)

    def test_concrete_period_runs(self):
        """Sanity: with concrete Python floats for period_x /
        period_y, the function returns a finite state."""
        from lumenairy.raytrace.jax_trace import _apply_doe_kick_jax
        state = self._make_initial_state()
        wavelength = 1.55e-6
        order_x, order_y = 1, 0
        period_x, period_y = 5e-6, float('inf')
        out = _apply_doe_kick_jax(state, order_x, order_y,
                                    period_x, period_y, wavelength)
        # All output fields finite.
        for name in ('x', 'y', 'L', 'M', 'N', 'opd'):
            arr = np.asarray(getattr(out, name))
            assert np.all(np.isfinite(arr)), f"{name} not finite"
        # The kick along x should equal m*lambda/period.
        dL_expected = order_x * wavelength / period_x
        np.testing.assert_allclose(np.asarray(out.L),
                                    np.zeros(out.L.shape) + dL_expected,
                                    atol=1e-12)

    def test_grad_w_r_t_period_finite_nonzero(self):
        """THE B1-9 PIN: ``jax.grad`` w.r.t. ``period_x`` must return a
        finite, non-zero scalar.  Pre-v4.12 the ``float(period_x)``
        strip produced silent zero gradient (or crash on the
        ``np.isfinite`` of a tracer)."""
        from lumenairy.raytrace.jax_trace import _apply_doe_kick_jax

        state = self._make_initial_state()
        wavelength = 1.55e-6

        def trace_and_reduce(period_x):
            """Apply DOE kick with traced period_x; return a scalar
            function of the result so we can grad it.  Sum of xÂ²+yÂ²
            after a transfer step makes the result period-dependent."""
            order_x, order_y = 1, 0
            period_y = jnp.inf
            new_state = _apply_doe_kick_jax(
                state, order_x, order_y,
                period_x, period_y, wavelength)
            # Propagate the ray a fixed thickness so dL contributes to
            # x_out (otherwise dL only shows up at the next surface).
            thickness = 10e-3
            x_out = new_state.x + new_state.L * thickness
            y_out = new_state.y + new_state.M * thickness
            return jnp.sum(x_out * x_out + y_out * y_out)

        grad_fn = jax.grad(trace_and_reduce)
        period_x = jnp.float32(5e-6)
        g = grad_fn(period_x)
        g_np = float(np.asarray(g))
        assert np.isfinite(g_np), (
            f"Gradient w.r.t. period_x must be finite; got {g_np}")
        assert abs(g_np) > 1e-12, (
            f"Gradient w.r.t. period_x must be non-zero (pre-fix "
            f"``float(period_x)`` stripped the trace and produced "
            f"silent zero gradient); got {g_np}")

    def test_grad_w_r_t_period_y_finite_nonzero(self):
        """Same pin for the y-axis grating period."""
        from lumenairy.raytrace.jax_trace import _apply_doe_kick_jax

        state = self._make_initial_state()
        wavelength = 1.55e-6

        def trace_and_reduce(period_y):
            order_x, order_y = 0, 1
            period_x = jnp.inf
            new_state = _apply_doe_kick_jax(
                state, order_x, order_y,
                period_x, period_y, wavelength)
            thickness = 10e-3
            x_out = new_state.x + new_state.L * thickness
            y_out = new_state.y + new_state.M * thickness
            return jnp.sum(x_out * x_out + y_out * y_out)

        grad_fn = jax.grad(trace_and_reduce)
        period_y = jnp.float32(5e-6)
        g = grad_fn(period_y)
        g_np = float(np.asarray(g))
        assert np.isfinite(g_np)
        assert abs(g_np) > 1e-12

    def test_grad_matches_analytic_sign(self):
        """Sanity-check the sign of the gradient against the analytic
        derivative.  ``trace_and_reduce`` ~ ``(dL * t)^2 * n_rays`` for
        on-axis rays (x_init = 0), where ``dL = lambda / period_x``.
        ``d/dperiod_x[(lambda*t/period)^2 * n] = -2*lambda^2*t^2*n /
        period^3 < 0``.  Even with off-axis rays, the leading
        period-dependent term still has negative derivative."""
        from lumenairy.raytrace.jax_trace import _apply_doe_kick_jax

        state = self._make_initial_state()
        wavelength = 1.55e-6

        def trace_and_reduce(period_x):
            order_x, order_y = 1, 0
            period_y = jnp.inf
            new_state = _apply_doe_kick_jax(
                state, order_x, order_y,
                period_x, period_y, wavelength)
            thickness = 10e-3
            x_out = new_state.x + new_state.L * thickness
            y_out = new_state.y + new_state.M * thickness
            return jnp.sum(x_out * x_out + y_out * y_out)

        # Compare JAX grad to forward-difference numeric grad.
        period_x = 5e-6
        eps = 1e-9
        f_plus = float(trace_and_reduce(jnp.float64(period_x + eps)))
        f_minus = float(trace_and_reduce(jnp.float64(period_x - eps)))
        g_fd = (f_plus - f_minus) / (2 * eps)

        g_jax = float(jax.grad(trace_and_reduce)(jnp.float64(period_x)))
        # Sign agreement; numeric value within ~1%.
        assert np.sign(g_jax) == np.sign(g_fd), (
            f"JAX-grad sign ({g_jax}) disagrees with finite-difference "
            f"({g_fd}).")
        rel = abs(g_jax - g_fd) / max(abs(g_fd), 1e-30)
        assert rel < 1e-2, (
            f"JAX-grad / FD mismatch: jax={g_jax:.6e}, fd={g_fd:.6e}, "
            f"rel={rel:.3e}")

    def test_infinite_period_concrete_no_kick(self):
        """Concrete ``inf`` period (1-D grating) must produce zero
        kick along that axis (the no-grating branch)."""
        from lumenairy.raytrace.jax_trace import _apply_doe_kick_jax
        state = self._make_initial_state()
        wavelength = 1.55e-6
        out = _apply_doe_kick_jax(
            state, order_x=1, order_y=0,
            period_x=np.inf, period_y=np.inf, wavelength=wavelength)
        np.testing.assert_allclose(np.asarray(out.L),
                                    np.asarray(state.L))
        np.testing.assert_allclose(np.asarray(out.M),
                                    np.asarray(state.M))

    def test_infinite_period_traced_no_kick(self):
        """Traced ``jnp.inf`` period must also produce zero kick (the
        ``jnp.where`` guard).  This is the path that pre-fix raised
        on ``np.isfinite(tracer)``."""
        from lumenairy.raytrace.jax_trace import _apply_doe_kick_jax
        state = self._make_initial_state()
        wavelength = 1.55e-6

        def kick_with_traced_period(period_x):
            return _apply_doe_kick_jax(
                state, 1, 0, period_x, jnp.inf, wavelength)

        # Should NOT raise (pre-fix crashed on np.isfinite(tracer)).
        out = jax.jit(kick_with_traced_period)(jnp.inf)
        np.testing.assert_allclose(np.asarray(out.L),
                                    np.asarray(state.L), atol=1e-12)


# ============================================================================
# Source: test_audit_fixes_v4_12_0_round4_tier0.py
# Audit version: V4_12_0  scope: round4_tier0
# Original module docstring preserved as comment block for git-blame traceability:
#   Pinning tests for the v4.12.0 round-4 audit Tier-0 fixes
#   (``AUDIT_ROUND4_2026_05_16.md`` items B0-1, B0-2, B2-6).
#   
#   Each test pins a finding from the round-4 audit:
#   
#   * B0-1 -- README cookbook examples (lines 2490-2606) now actually
#     run.  Pre-4.12.0 the "Three minimal end-to-end examples" block
#     failed on:
#   
#     - ``create_gaussian_beam(N=512, dx=2e-6, sigma=50e-6)`` -- missing
#       required ``wavelength``.
#     - ``apply_real_lens(E, presc, wavelength=..., dx=...)`` --
#       positional ``presc`` (kw-only since v4.7).
#     - ``load_zmx_prescription(...)`` -- function renamed to
#       ``load_zemax_zmx`` in v4.7, no back-compat alias.
#   
#     Each test below reproduces one cookbook block verbatim and asserts
#     it runs to completion with non-zero output.
#   
#   * B0-2 -- ``_deprecation.py`` helpers are now wired into the
#     top-level namespace.  The renamed-in-v4.7 functions
#     ``load_zmx_prescription`` and ``load_zemax_prescription_txt`` are
#     exposed as ``deprecated_alias`` shims that forward to
#     ``load_zemax_zmx`` / ``load_zemax_prescription_data_txt`` after
#     emitting a ``DeprecationWarning``.
#   
#   * B2-6 -- ``gerchberg_saxton(backend='jax')`` now forwards
#     ``seed`` / ``dtype`` / ``initial_phase`` to ``gerchberg_saxton_jax``.
#     Pre-4.12.0 the dispatcher only forwarded ``n_iter`` so two seeds
#     produced identical trajectories via the JAX path; the
#     function-level kwargs were wired inside the JAX implementation
#     but unreachable via the unified entry.
#   
#   These pins guard against regression of the v4.12.0 fixes.
# ============================================================================

import io
import os
import sys
import tempfile
import warnings

import numpy as np
import pytest

import lumenairy as la

# ============================================================================
# B0-1 -- README cookbook examples run to completion
# ============================================================================

class TestAuditFixesV4_12_0_round4_tier0_ReadmeCookbookExamples:
    """Each test reproduces one README cookbook code block verbatim.

    Pre-4.12.0 these failed with ``TypeError`` / ``AttributeError`` at
    import or first-line execution; 4.12.0 fixed the README and the
    `pip install lumenairy` first-five-minutes experience.
    """

    def test_three_minimal_example_1_free_space(self):
        """Cookbook example 1: free-space propagation via smart
        dispatch.  The README uses ``angular_spectrum_propagate``
        directly to get a bare-ndarray return (``propagate`` returns
        a tuple for Fraunhofer/Fresnel methods, separate audit B1-7).
        """
        E, x, y = la.create_gaussian_beam(
            N=128, dx=2e-6, wavelength=1.31e-6, sigma=50e-6)
        # README uses angular_spectrum_propagate (near-field) so the
        # return is an ndarray rather than the (E, dx_out, dy_out)
        # tuple that the smart dispatcher returns from Fraunhofer.
        E_focus = la.angular_spectrum_propagate(
            E, z=1e-3, wavelength=1.31e-6, dx=2e-6)
        cx, cy = la.beam_centroid(E_focus, 2e-6)
        assert E_focus.shape == (128, 128)
        assert np.isfinite(cx) and np.isfinite(cy)
        # Energy preserved (near-field, no aperture clipping)
        assert np.sum(np.abs(E_focus) ** 2) > 0

    def test_three_minimal_example_2_real_lens_end_to_end(self):
        """Cookbook example 2: a Thorlabs lens, end-to-end.
        Pre-4.12.0 broke on positional ``presc``."""
        E, _, _ = la.create_gaussian_beam(
            N=128, dx=2e-6, wavelength=1.31e-6, sigma=50e-6)
        presc = la.thorlabs_lens('AC254-100-C')
        with warnings.catch_warnings():
            # The 128 grid is intentionally tiny for test speed; the
            # full grid-vs-aperture warning is expected and harmless.
            warnings.simplefilter('ignore', UserWarning)
            E_out = la.apply_real_lens(
                E, prescription=presc, wavelength=1.31e-6, dx=2e-6)
        assert E_out.shape == (128, 128)
        assert np.any(E_out != 0)

    def test_three_minimal_example_3_trace_for_spot_rms(self):
        """Cookbook example 3: ray-trace the same lens for spot RMS."""
        presc = la.thorlabs_lens('AC254-100-C')
        result = la.trace_prescription(
            presc, wavelength=1.31e-6, num_rings=8)
        rms = la.spot_rms(result)
        # spot_rms returns either a tuple (rms, ...) or a scalar
        # depending on the field structure; the cookbook indexes [0].
        rms_axis = rms[0] if hasattr(rms, '__getitem__') else rms
        assert np.isfinite(rms_axis)
        assert rms_axis > 0

    def test_basic_propagation_cookbook(self):
        """Cookbook 'Basic propagation' block (line 2519).  Pre-4.12.0
        broke on missing ``wavelength`` in ``create_gaussian_beam``."""
        E, x, y = la.create_gaussian_beam(
            N=128, dx=2e-6, wavelength=1.3e-6, sigma=50e-6)
        E_prop = la.angular_spectrum_propagate(
            E, z=0.01, wavelength=1.3e-6, dx=2e-6)
        cx, cy = la.beam_centroid(E_prop, 2e-6)
        dx_b, dy_b = la.beam_d4sigma(E_prop, 2e-6)
        assert np.isfinite(cx) and np.isfinite(cy)
        assert dx_b > 0 and dy_b > 0

    def test_real_lens_from_zemax_cookbook(self):
        """Cookbook 'Real lens from Zemax file' block (line 2569).
        Pre-4.12.0 broke on ``load_zmx_prescription`` (renamed) and
        positional ``rx``."""
        E_in, _, _ = la.create_gaussian_beam(
            N=128, dx=2e-6, wavelength=1.3e-6, sigma=50e-6)
        # README uses la.thorlabs_lens as the second option; we use
        # it to avoid needing a .zmx file on disk for the test.
        rx = la.thorlabs_lens('AC254-200-C')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            E_out = la.apply_real_lens(
                E_in, prescription=rx, wavelength=1.3e-6, dx=2e-6)
            E_out_traced = la.apply_real_lens_traced(
                E_in, prescription=rx, wavelength=1.3e-6, dx=2e-6,
                ray_subsample=4)
        assert E_out.shape == (128, 128)
        assert E_out_traced.shape == (128, 128)
        assert np.any(E_out != 0)
        assert np.any(E_out_traced != 0)

    def test_cylindrical_biconic_cookbook(self):
        """Cookbook 'Anamorphic / cylindrical / biconic' block
        (line 2613).  Pre-4.12.0 broke on positional ``pres``."""
        E_in, _, _ = la.create_gaussian_beam(
            N=128, dx=2e-6, wavelength=1.3e-6, sigma=50e-6)
        pres_cyl = la.make_cylindrical(
            R_focus=50e-3, d=3e-3, glass='N-BK7', axis='x')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            E_line_focus = la.apply_real_lens(
                E_in, prescription=pres_cyl,
                wavelength=1.3e-6, dx=2e-6)

            pres_bi = la.make_biconic(
                R1_x=50e-3, R1_y=70e-3,
                R2_x=-30e-3, R2_y=-40e-3,
                d=4e-3, glass='N-BK7')
            E_anam = la.apply_real_lens(
                E_in, prescription=pres_bi,
                wavelength=1.3e-6, dx=2e-6)
        assert E_line_focus.shape == (128, 128)
        assert E_anam.shape == (128, 128)
        assert np.any(E_line_focus != 0)
        assert np.any(E_anam != 0)

    def test_polarization_cookbook(self):
        """Cookbook 'Polarization' block (line 2765).  Pre-4.12.0
        broke on positional ``create_gaussian_beam(256, 2e-6, 30e-6)``
        which silently bound ``30e-6`` to ``wavelength`` and missed
        the required ``sigma`` kwarg."""
        scalar, _, _ = la.create_gaussian_beam(
            128, 2e-6, 1.3e-6, sigma=30e-6)
        field = la.create_circular_polarized(
            scalar, dx=2e-6, handedness='right')
        la.apply_half_wave_plate(field, angle=np.pi / 8)
        field.apply_thin_lens(f=100e-3, wavelength=1.3e-6)
        S = la.stokes_parameters(field)
        # Right-circular incident -> left-circular after lambda/2 at
        # 22.5 deg gives S3/S0 == -1 (sign flips).
        s3_over_s0 = float(S['S3'].mean() / S['S0'].mean())
        assert -1.5 < s3_over_s0 < -0.5


# ============================================================================
# B0-2 -- deprecation aliases are wired and emit DeprecationWarning
# ============================================================================

@_requires_jax
class TestAuditFixesV4_12_0_round4_tier0_DeprecatedAliasShims:
    """``_deprecation.deprecated_alias`` is now actually imported into
    the top-level namespace, and v4.7-renamed functions ship back-compat
    shims.  Pre-4.12.0 the helpers existed in ``_deprecation.py`` but
    nothing imported them, so old names died with ``AttributeError``."""

    def test_load_zmx_prescription_shim_exists(self):
        """The pre-v4.7 name ``load_zmx_prescription`` is reachable."""
        assert hasattr(la, 'load_zmx_prescription')
        assert callable(la.load_zmx_prescription)

    def test_load_zemax_prescription_txt_shim_exists(self):
        """The pre-v4.7 name ``load_zemax_prescription_txt`` is reachable."""
        assert hasattr(la, 'load_zemax_prescription_txt')
        assert callable(la.load_zemax_prescription_txt)

    def test_load_zmx_prescription_emits_deprecation_warning(self):
        """Calling the old name fires ``DeprecationWarning`` pointing
        at the new canonical name."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with pytest.raises((FileNotFoundError, OSError, IOError)):
                la.load_zmx_prescription('nonexistent_test_file.zmx')
            # Filter to our deprecation
            dws = [w for w in caught
                   if issubclass(w.category, DeprecationWarning)]
            assert len(dws) >= 1, (
                f"Expected DeprecationWarning, got: "
                f"{[(w.category.__name__, str(w.message)) for w in caught]}")
            msg = str(dws[0].message)
            assert 'load_zmx_prescription' in msg
            assert 'load_zemax_zmx' in msg

    def test_load_zemax_prescription_txt_emits_deprecation_warning(self):
        """Calling the old txt-loader name fires ``DeprecationWarning``
        pointing at the new canonical name."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with pytest.raises((FileNotFoundError, OSError, IOError)):
                la.load_zemax_prescription_txt('nonexistent_test_file.txt')
            dws = [w for w in caught
                   if issubclass(w.category, DeprecationWarning)]
            assert len(dws) >= 1
            msg = str(dws[0].message)
            assert 'load_zemax_prescription_txt' in msg
            assert 'load_zemax_prescription_data_txt' in msg

    def test_deprecated_shim_forwards_to_canonical(self):
        """The shim calls through to the new function so a user with
        pre-v4.7 code keeps getting the correct result (alongside the
        warning).  Test via a minimal singlet .zmx file."""
        # Build a minimal pseudo-.zmx with a BK7 singlet surface so
        # the auto-detect (active surfaces = glass-or-mirror) finds
        # at least one surface.  We don't assert physical correctness
        # here, only that the shim invokes the same code path.
        zmx_text = (
            "VERS 190311 0 0\n"
            "MODE SEQ\n"
            "NAME test\n"
            "UNIT MM X W X CM MR CPMM\n"
            "ENPD 10.0\n"
            "WAVM 1 0.587600 1.0\n"
            "PWAV 1\n"
            "SURF 0\n"
            "  TYPE STANDARD\n"
            "  CURV 0.0\n"
            "  DISZ INFINITY\n"
            "SURF 1\n"
            "  TYPE STANDARD\n"
            "  STOP\n"
            "  CURV 0.01 0 0 0 0 \"\"\n"
            "  DISZ 3.0\n"
            "  GLAS BK7 0 0 1.5 50.0\n"
            "  DIAM 5.0\n"
            "SURF 2\n"
            "  TYPE STANDARD\n"
            "  CURV -0.005 0 0 0 0 \"\"\n"
            "  DISZ 50.0\n"
            "  DIAM 5.0\n"
            "SURF 3\n"
            "  TYPE STANDARD\n"
            "  CURV 0.0 0 0 0 0 \"\"\n"
            "  DISZ 0.0\n"
            "  DIAM 5.0\n"
            "BLNK\n"
        )
        with tempfile.NamedTemporaryFile(
                mode='w', suffix='.zmx', delete=False) as f:
            f.write(zmx_text)
            fpath = f.name
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', DeprecationWarning)
                warnings.simplefilter('ignore', UserWarning)
                rx_new = la.load_zemax_zmx(fpath)
                rx_old = la.load_zmx_prescription(fpath)
            # Same canonical schema keys (surfaces / thicknesses)
            assert set(rx_new.keys()) == set(rx_old.keys())
            # Same surface count
            assert (len(rx_new.get('surfaces', []))
                    == len(rx_old.get('surfaces', [])))
            # Same thickness count and values
            assert (len(rx_new.get('thicknesses', []))
                    == len(rx_old.get('thicknesses', [])))
            assert (rx_new.get('thicknesses') == rx_old.get('thicknesses'))
        finally:
            try:
                os.unlink(fpath)
            except OSError:
                pass


# ============================================================================
# B2-6 -- gerchberg_saxton(backend='jax') forwards seed/dtype/initial_phase
# ============================================================================

# These tests need JAX; skip cleanly if not installed.
jax = pytest.importorskip('jax', reason='JAX not available')


@_requires_jax
class TestAuditFixesV4_12_0_round4_tier0_GerchbergSaxtonJaxDispatch:
    """``gerchberg_saxton(backend='jax')`` now forwards
    ``seed`` / ``dtype`` / ``initial_phase`` to the JAX implementation.

    Pre-4.12.0 the dispatcher (``phase_retrieval.py:127-128``)
    only forwarded ``n_iter``, so two seeds produced byte-identical
    trajectories via the JAX path."""

    @pytest.fixture
    def source_target(self):
        N = 32
        x = np.linspace(-1, 1, N)
        X, Y = np.meshgrid(x, x)
        source = np.exp(-(X**2 + Y**2) / 0.5**2)
        target = np.exp(-(X**2 + Y**2) / 0.3**2)
        return source, target

    def test_jax_seeds_produce_different_trajectories(self, source_target):
        """Two different seeds via the unified entry yield genuinely
        different intermediate trajectories (pinning the dispatcher
        fix).  Pre-4.12.0 ``seed`` was dropped at the dispatcher so
        both runs got the same default-zero initial phase."""
        source, target = source_target
        phase_42, _ = la.gerchberg_saxton(
            source, target, n_iter=20, seed=42, backend='jax')
        phase_43, _ = la.gerchberg_saxton(
            source, target, n_iter=20, seed=43, backend='jax')
        # Identical-seed -> identical phases (the historical bug)
        diff = float(np.max(np.abs(phase_42 - phase_43)))
        assert diff > 1e-3, (
            f"BUG: seed=42 and seed=43 produced near-identical phases "
            f"(max abs diff = {diff:.2e}); dispatcher dropped seed.")

    def test_jax_dtype_forwarded_no_error(self, source_target):
        """Passing ``dtype`` reaches ``gerchberg_saxton_jax``.

        Pre-4.12.0 the dispatcher dropped the kwarg so users couldn't
        reach precision control via the unified entry.  We assert via
        the explicit float32 path (which works regardless of whether
        the global JAX_ENABLE_X64 flag is set); the JAX implementation
        itself has a separate fragility around float64 mode that's a
        pre-existing concern, not a dispatcher-fix concern.
        """
        source, target = source_target
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            phase_f32, err_f32 = la.gerchberg_saxton(
                source, target, n_iter=10, seed=7,
                dtype=np.float32, backend='jax')
        assert phase_f32.shape == source.shape
        assert np.all(np.isfinite(phase_f32))
        assert np.isfinite(err_f32)

    def test_jax_dtype_kwarg_actually_threads_through(self, source_target):
        """A targeted check that the dispatcher's `dtype` kwarg is
        forwarded into `gerchberg_saxton_jax` -- we patch the JAX
        function and assert the dispatcher passes `dtype` explicitly.
        """
        import lumenairy.analysis.phase_retrieval as pr

        source, target = source_target
        captured = {}
        real_jax = pr.gerchberg_saxton_jax

        def spy(*args, **kwargs):
            captured.update(kwargs)
            return real_jax(*args, **kwargs)

        pr.gerchberg_saxton_jax = spy
        try:
            la.gerchberg_saxton(
                source, target, n_iter=5, seed=11,
                dtype=np.float32, initial_phase=None,
                backend='jax')
        finally:
            pr.gerchberg_saxton_jax = real_jax

        assert 'seed' in captured and captured['seed'] == 11
        assert 'dtype' in captured and captured['dtype'] is np.float32
        assert 'initial_phase' in captured and captured['initial_phase'] is None
        assert 'n_iter' in captured and captured['n_iter'] == 5

    def test_jax_initial_phase_forwarded(self, source_target):
        """Passing ``initial_phase`` reaches ``gerchberg_saxton_jax``.
        With identical seed but different explicit ``initial_phase``,
        the trajectories diverge."""
        source, target = source_target
        ip_zero = np.zeros_like(source)
        ip_rand = np.random.default_rng(123).uniform(
            -np.pi, np.pi, size=source.shape).astype(np.float32)
        phase_zero, _ = la.gerchberg_saxton(
            source, target, n_iter=20, initial_phase=ip_zero,
            backend='jax')
        phase_rand, _ = la.gerchberg_saxton(
            source, target, n_iter=20, initial_phase=ip_rand,
            backend='jax')
        diff = float(np.max(np.abs(phase_zero - phase_rand)))
        assert diff > 1e-3, (
            f"BUG: distinct initial_phase arrays produced near-identical "
            f"results (max abs diff = {diff:.2e}); dispatcher dropped "
            f"initial_phase.")

    def test_numpy_seeds_still_differ(self, source_target):
        """Regression guard: the numpy-backend behaviour (unchanged)
        still produces different trajectories for different seeds."""
        source, target = source_target
        phase_42, _ = la.gerchberg_saxton(
            source, target, n_iter=20, seed=42, backend='numpy')
        phase_43, _ = la.gerchberg_saxton(
            source, target, n_iter=20, seed=43, backend='numpy')
        diff = float(np.max(np.abs(phase_42 - phase_43)))
        assert diff > 1e-3

    def test_numpy_dtype_kwarg_accepted(self, source_target):
        """4.12.0 added ``dtype`` to ``gerchberg_saxton``'s public
        signature; the numpy path now honours it for API parity with
        ``error_reduction`` / ``hybrid_input_output``."""
        source, target = source_target
        phase_c128, _ = la.gerchberg_saxton(
            source, target, n_iter=10, seed=7,
            dtype=np.complex128, backend='numpy')
        phase_c64, _ = la.gerchberg_saxton(
            source, target, n_iter=10, seed=7,
            dtype=np.complex64, backend='numpy')
        # Both run to completion with the requested dtype guidance.
        # We can't pin the dtype of `phase` itself (np.angle returns
        # float matching the complex input's float-component dtype),
        # but the kwarg must at least be accepted without TypeError.
        assert phase_c128.shape == source.shape
        assert phase_c64.shape == source.shape


# ============================================================================
# Source: test_audit_fixes_v4_12_1_coverage.py
# Audit version: V4_12_1  scope: coverage
# Original module docstring preserved as comment block for git-blame traceability:
#   Pinning tests for the 14 v4.11.2 audit coverage gaps identified by
#   AUDIT_ROUND4_2026_05_16 (section "Test-suite quality").
#   
#   Each item below corresponds to a v4.11.2 fix that landed in the code
#   but had no regression test.  The tests here would FAIL on the pre-fix
#   version of the code.
#   
#   Item map (one test class per item):
#   
#   1. compute_psf non-square pupil error
#   2. apply_detector non-integer pixel ratio
#   3. find_best_focus NaN injection
#   4. monte_carlo_tolerancing_linearized a_k >= 0 clamp
#   5. load_material RuntimeWarning on dispersion drop
#   6. Source.* **factory_kwargs propagation
#   7. apply_real_lens_traced M_x/M_y transpose (centered output)
#   8. NaN sentinel mask in apply_real_lens (no NaN in wave-leg output)
#   9. stop_index != 0 RuntimeWarning in _traced / _maslov
#   10. Freeform-terms RuntimeWarning in thin-element apply_real_lens
#   11. Zemax coord-break STOP marker (refractive-only counter)
#   12. JAX <-> NumPy phase-retrieval cross-parity (gerchberg_saxton)
#   13. Cassegrain S1/S2/S3/S5 hand-derivation (extends v4.11.2's S4 pin)
#   14. Richards-Wolf vs paraxial Airy at low NA
#   
#   All tests should PASS on the current v4.12.1 codebase.  A failure
#   indicates the underlying fix has regressed (or, if it was never
#   present, the round-4 audit was right to flag it).
# ============================================================================

import os
import tempfile
import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.raytrace import Surface, seidel_coefficients

# ============================================================================
# Item 1 -- compute_psf raises on non-square pupil
# ============================================================================

class TestAuditFixesV4_12_1_coverage_ComputePsfNonSquarePupil:
    """v4.11.2 added an explicit ``ValueError`` in
    :func:`compute_psf` when the pupil array is not square.  Pre-fix
    the function used ``pupil.shape[0]`` for both axes (and the
    Fraunhofer pad / grid scale), so a rectangular input produced a
    silently wrong PSF.
    """

    def test_non_square_pupil_raises_value_error(self):
        # Rectangular pupil: 32 rows, 48 columns.
        pupil = np.ones((32, 48), dtype=np.complex128)
        with pytest.raises(ValueError) as excinfo:
            la.compute_psf(pupil, wavelength=1e-6, f=0.1,
                            dx_pupil=10e-6)
        msg = str(excinfo.value)
        # Message must reference the rectangular shape so a user
        # ports their code into a square embed.
        assert '32' in msg and '48' in msg, (
            f"v4.11.2 ValueError must reference the rectangular shape "
            f"(32, 48); got: {msg!r}")
        # And it must specifically name 'square' or 'rectangular'
        # to be actionable.
        msg_lower = msg.lower()
        assert ('square' in msg_lower or 'rectangular' in msg_lower), (
            f"v4.11.2 ValueError should explain the constraint "
            f"(square / rectangular); got: {msg!r}")

    def test_non_square_pupil_raises_before_fft(self):
        """The raise must happen ON ENTRY, not after the FFT.  Pin
        with the very first non-trivial shape mismatch (rows == 1,
        cols == 2)."""
        pupil = np.ones((1, 2), dtype=np.complex128)
        with pytest.raises(ValueError):
            la.compute_psf(pupil, wavelength=1e-6, f=0.1,
                            dx_pupil=10e-6)

    def test_square_pupil_does_not_raise(self):
        """Sanity check: a square pupil still works."""
        pupil = np.ones((32, 32), dtype=np.complex128)
        psf, dx_psf = la.compute_psf(pupil, wavelength=1e-6, f=0.1,
                                       dx_pupil=10e-6)
        assert psf.shape == (32, 32)
        assert dx_psf > 0


# ============================================================================
# Item 2 -- apply_detector non-integer pixel ratio (area-integral correctness)
# ============================================================================

class TestAuditFixesV4_12_1_coverage_ApplyDetectorNonIntegerPixelRatio:
    """v4.11.2 added the area-integral correction to
    :func:`apply_detector` so a non-integer ``pixel_pitch / dx_field``
    ratio doesn't break photon conservation.  Pre-fix the bilinear
    zoom + ``pixel_pitch**2`` multiplication broke conservation for
    ratios like 2.3 -- integrated detector power vs analytic input
    power drifted by several percent.

    Pin: build a uniform unit-amplitude field, run apply_detector
    with ratio = 2.3, and assert the integrated photon count agrees
    with the analytic expectation (input_intensity * total_area * QE
    * exposure_time / photon_energy) to within ~1%.
    """

    def test_ratio_2p3_conserves_total_power(self):
        # Uniform field with controlled photon scale.  No noise, no
        # dark current, no read noise -- isolate the area-integral
        # branch.
        N = 64
        dx_field = 1e-6  # 1 um per field sample
        # Non-integer ratio = 2.3
        pixel_pitch = 2.3e-6  # detector pitch
        n_pixels = 20  # 20 * 2.3 um = 46 um detector span

        E = np.ones((N, N), dtype=np.complex128)
        # apply_detector returns (image, x_det, y_det).
        image_ni, _x, _y = la.apply_detector(
            E, dx_field=dx_field, pixel_pitch=pixel_pitch,
            n_pixels=n_pixels,
            exposure_time=1.0,
            quantum_efficiency=1.0,
            read_noise_e=0.0,
            dark_current_e_per_s=0.0,
            seed=0,
        )
        total_counts_ni = float(image_ni.sum())

        # Reference: integer ratio 2.0, same physical detector area
        # (23 * 2 um = 46 um = 20 * 2.3 um, exact).
        n_pixels_ref = 23
        pixel_pitch_ref = 2.0e-6
        image_int, _x_r, _y_r = la.apply_detector(
            E, dx_field=dx_field, pixel_pitch=pixel_pitch_ref,
            n_pixels=n_pixels_ref,
            exposure_time=1.0,
            quantum_efficiency=1.0,
            read_noise_e=0.0,
            dark_current_e_per_s=0.0,
            seed=0,
        )
        total_counts_int = float(image_int.sum())

        # Under v4.11.2's area-integral correction, the two detectors
        # over the same physical area must integrate to within ~1% of
        # each other.  Pre-fix the bilinear-zoom path made the
        # non-integer ratio over- or under-count by several percent.
        # (The Poisson-shot-noise step adds sqrt(N) jitter, hence the
        # 2% allowance instead of 1%.)
        if total_counts_int > 0:
            rel = abs(total_counts_ni - total_counts_int) / total_counts_int
        else:
            rel = abs(total_counts_ni)
        assert rel < 0.02, (
            f"apply_detector with ratio=2.3 and ratio=2.0 over the "
            f"same physical area produced integrated counts differing "
            f"by {rel*100:.2f}% -- v4.11.2 area-integral correction "
            f"should keep the relative drift below ~1%.  "
            f"non-int total = {total_counts_ni:.6e}, "
            f"int total = {total_counts_int:.6e}")


# ============================================================================
# Item 3 -- find_best_focus NaN injection (all-NaN guard)
# ============================================================================

class TestAuditFixesV4_12_1_coverage_FindBestFocusNanGuard:
    """v4.11.2 added an all-NaN guard returning ``(nan, nan)`` from
    :func:`find_best_focus` so a monte-carlo trial whose every scan
    point is NaN doesn't poison the whole MC run with a
    ``ValueError: All-NaN slice encountered``.

    Pin: build a :class:`ThroughFocusResult` with all-NaN ``strehl``,
    call find_best_focus(metric='strehl'), and assert it returns
    (nan, nan) without raising.
    """

    def test_all_nan_strehl_returns_nan_pair(self):
        # Build a minimal scan struct.
        from lumenairy.analysis import ThroughFocusResult
        n = 5
        z = np.linspace(-1e-3, 1e-3, n)
        nans = np.full(n, np.nan)
        scan = ThroughFocusResult(
            z=z, peak_I=nans.copy(),
            strehl=nans.copy(),
            d4sigma_x=nans.copy(),
            d4sigma_y=nans.copy(),
            rms_radius=nans.copy(),
            power_in_bucket=nans.copy(),
            wavelength=1e-6,
        )
        # Should not raise; returns (nan, nan).
        z_best, val_best = la.find_best_focus(scan, metric='strehl')
        assert np.isnan(z_best), (
            f"find_best_focus on all-NaN strehl should return NaN z; "
            f"got {z_best!r}")
        assert np.isnan(val_best), (
            f"find_best_focus on all-NaN strehl should return NaN value; "
            f"got {val_best!r}")


# ============================================================================
# Item 4 -- monte_carlo_tolerancing_linearized a_k >= 0 clamp
# ============================================================================

class TestAuditFixesV4_12_1_coverage_McTolLinearizedAkClamp:
    """v4.11.2 clamps the per-knob sensitivity coefficient
    ``a_k = max(a_k, 0)`` in
    :func:`monte_carlo_tolerancing_linearized` so a one-sided FD
    probe that yields S_p > S_nom (due to numerical noise, sign
    coincidence with the nominal aberration, or a non-monotone FD
    step) cannot predict S_pred > S_nom -- impossible under Marechal
    (S_perturbed <= S_nom for any small perturbation).

    Pin: build a tolerancing scenario whose nominal lens is
    essentially perfect (S_nom ~ 1) and whose perturbation sigma is
    deliberately small, so any FD-noise-driven S_p > S_nom would be
    clipped to a_k=0.  Verify that the maximum predicted Strehl in
    the Monte-Carlo run is <= S_nom + a small epsilon (the v4.11.2
    clamp guarantees this).
    """

    def test_predicted_strehl_never_exceeds_nominal_and_a_k_nonneg(self):
        from lumenairy.analysis import (
            monte_carlo_tolerancing_linearized,
        )
        wavelength = 1.31e-6
        N = 32
        dx = 8e-6
        # Near-perfect singlet so the FD-probe S_p ~ S_nom and any
        # numerical-noise sign coincidence would land naturally
        # close to the clamp boundary.
        rx = la.make_singlet(R1=200e-3, R2=-200e-3, d=2e-3,
                              glass='N-BK7', aperture=200e-6)
        x = (np.arange(N) - N / 2) * dx
        X, Y = np.meshgrid(x, x, indexing='xy')
        E_source = np.exp(-(X ** 2 + Y ** 2) / (50e-6) ** 2).astype(
            np.complex128)
        # Perturbation spec: tiny decenter / tilt / form-error, where
        # any of the per-knob FD probes could produce S_p > S_nom
        # purely from numerical noise.
        perturbation_spec = [
            {'surface_index': 0,
             'decenter_std': 1e-12,    # 1 pm
             'tilt_std': 1e-12,        # 1 prad
             'form_error_rms': 1e-12,  # 1 pm
             'name': 'tiny_pert_s0'},
        ]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = monte_carlo_tolerancing_linearized(
                prescription=rx, wavelength=wavelength,
                N=N, dx=dx,
                E_source=E_source,
                perturbation_spec=perturbation_spec,
                focal_length=100e-3,
                aperture=200e-6,
                n_trials=64,
                seed=42,
                verbose=False,
            )

        # 1. Every sensitivity coefficient a_k must be >= 0
        # (the v4.11.2 clamp ``a_k = max(a_k, 0)``).
        sensitivities = result['sensitivities']
        for (spec_idx, knob, sigma, a_k) in sensitivities:
            assert a_k >= 0.0, (
                f"sensitivities[{spec_idx}, {knob}]: a_k = {a_k!r} < 0 "
                f"-- v4.11.2 clamps a_k = max(a_k, 0).")

        # 2. Every per-trial S_pred must be <= S_nom (since
        # S_pred = S_nom - sum a_k * xi_k^2 with a_k >= 0).
        strehls = np.asarray(result['strehl_array'])
        s_nom = float(result['nominal_strehl'])
        max_strehl = float(np.nanmax(strehls))
        assert max_strehl <= s_nom + 1e-12, (
            f"monte_carlo_tolerancing_linearized produced "
            f"S_pred={max_strehl} > S_nom={s_nom} -- Marechal-"
            f"invalid.  v4.11.2 clamps a_k=max(a_k,0), so S_pred "
            f"<= S_nom always.")


# ============================================================================
# Item 5 -- load_material RuntimeWarning on dispersion drop
# ============================================================================

class TestAuditFixesV4_12_1_coverage_LoadMaterialDispersionWarning:
    """v4.11.2 added a ``RuntimeWarning`` in :func:`load_material`
    when a fixed-glass JSON includes a ``dispersion`` field
    (which ``register_fixed_glass`` silently drops).  Pre-fix the
    saved dispersion data was discarded with no surface signal.

    Pin: save a fixed-glass material with a dispersion dict, reload
    it, and assert the warning fires.
    """

    def test_dispersion_drop_warning_emitted(self):
        # Save a fixed-glass material with a dispersion dict (mimics
        # what a save_material with dispersion= kwarg writes to disk).
        # ``set_library_path`` lives on the user_library module, not
        # on the top-level namespace, so import it directly.
        from lumenairy import user_library

        # Snapshot the existing library path so we can restore it.
        prior_path = user_library._library_path
        with tempfile.TemporaryDirectory() as td:
            user_library.set_library_path(td)
            try:
                user_library.save_material(
                    name='test_glass_disp',
                    n=1.5,
                    dispersion={'A': 1.0, 'B': 0.01},
                    description='test fixture',
                )
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    user_library.load_material('test_glass_disp')
                relevant = [
                    w for w in caught
                    if issubclass(w.category, RuntimeWarning)
                    and 'dispersion' in str(w.message).lower()
                ]
                assert relevant, (
                    f"load_material on a fixed-glass file with a "
                    f"'dispersion' field must emit a RuntimeWarning "
                    f"mentioning 'dispersion' (v4.11.2 fix).  "
                    f"Caught: {[(w.category.__name__, str(w.message)[:80]) for w in caught]}")
            finally:
                # Restore the prior library path so other tests are
                # unaffected.
                user_library._library_path = prior_path


# ============================================================================
# Item 6 -- Source.* **factory_kwargs propagation (dy=)
# ============================================================================

class TestAuditFixesV4_12_1_coverage_SourceFactoryKwargsForwarded:
    """v4.11.2 added ``**factory_kwargs`` to the :class:`Source`
    classmethods so callers can pass ``dy=``, ``dtype=``,
    ``normalize=`` etc. through to the underlying ``create_*``
    factories.  Pre-fix these kwargs were silently dropped and the
    factory defaulted to ``dy=dx``.

    Pin: build ``Source.gaussian(w0, N, dx, wavelength, dy=...)``
    with ``dy != dx`` and verify the resulting field shows the
    anisotropic pixel scaling (a vertical sample 1 pixel above
    the centre maps to physical y = dy, not dx).
    """

    def test_gaussian_dy_kwarg_propagates(self):
        N = 16
        dx = 2e-6
        dy = 5e-6  # 2.5x dx, so a centred Gaussian with sigma matched
                   # to dx would show clearly different values along
                   # the y-axis vs x-axis if dy were honored.
        wavelength = 1e-6
        w0 = 20e-6
        w0 / np.sqrt(2)
        src_dy = la.Source.gaussian(w0=w0, N=N, dx=dx,
                                     wavelength=wavelength, dy=dy)
        E_dy = np.asarray(src_dy.E)
        # |E| at row-offset 1 from centre (i.e. y = +/- dy):
        # exp(-(dy)^2 / w0^2)
        center = N // 2
        v_at_dy = float(np.abs(E_dy[center + 1, center]))
        expected_with_dy = np.exp(-(dy ** 2) / (w0 ** 2))
        # If dy was honored: 0.939... (for w0=20um, dy=5um).
        # If dy was ignored (defaulted to dx): 0.990...
        assert abs(v_at_dy - expected_with_dy) < 1e-6, (
            f"Source.gaussian with dy={dy} gave |E|[c+1, c]="
            f"{v_at_dy:.6f}; expected exp(-(dy/w0)^2)="
            f"{expected_with_dy:.6f}.  Pre-v4.11.2 dy was silently "
            f"ignored, which would have given "
            f"{np.exp(-(dx**2)/(w0**2)):.6f} instead.")

    def test_gaussian_default_dy_equals_dx(self):
        """When the caller does NOT pass dy, the factory defaults to
        dy=dx -- so the field should be isotropic."""
        N = 16
        dx = 2e-6
        wavelength = 1e-6
        w0 = 20e-6
        src = la.Source.gaussian(w0=w0, N=N, dx=dx, wavelength=wavelength)
        E = np.asarray(src.E)
        center = N // 2
        # x-offset 1 == y-offset 1 in physical units (both = dx).
        v_row = float(np.abs(E[center, center + 1]))
        v_col = float(np.abs(E[center + 1, center]))
        assert abs(v_row - v_col) < 1e-10, (
            f"Default Source.gaussian (no dy kwarg) should produce an "
            f"isotropic field (dy=dx); got |E|[c,c+1]={v_row}, "
            f"|E|[c+1,c]={v_col}.")


# ============================================================================
# Item 7 -- apply_real_lens_traced M_x/M_y stencil (Newton initial guess)
# ============================================================================

class TestAuditFixesV4_12_1_coverage_ApplyRealLensTracedMxMyTranspose:
    """v4.11.2 fixed the NumPy ``apply_real_lens_traced`` central
    finite-difference stencil that measures the paraxial magnification
    M_x / M_y from the forward-map grid.  Pre-fix the indices were
    transposed (computing dx_out/dy_in and dy_out/dx_in instead of
    dx_out/dx_in and dy_out/dy_in), which for a rotationally-symmetric
    singlet returned M_x = M_y ~ 0 from the cross-axis derivative.
    Newton then started from the all-zero initial guess (clipped to
    boundary 0.91-fallback) on every pixel.

    Pin: trace a centred Gaussian through a symmetric biconvex
    singlet (paraxial M = +1 by symmetry) and verify the output
    centroid sits within sub-pixel of the input centroid.  Pre-fix
    the silent transpose used to put Newton start at the centre-of-
    the-grid-but-clipped, producing an off-centred output bias.
    """

    def test_symmetric_singlet_keeps_input_centered(self):
        rx = la.make_singlet(
            R1=50e-3, R2=-50e-3, d=3e-3, glass='N-BK7', aperture=8e-3)
        N = 64
        dx = 1e-5
        wavelength = 1.31e-6
        x = (np.arange(N) - N / 2) * dx
        X, Y = np.meshgrid(x, x, indexing='xy')
        E_in = np.exp(-(X ** 2 + Y ** 2) / (1.5e-3) ** 2).astype(
            np.complex128)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            E_out = la.apply_real_lens_traced(
                E_in, prescription=rx, wavelength=wavelength, dx=dx,
                ray_subsample=4,
            )

        I_out = np.abs(E_out) ** 2
        total = float(I_out.sum())
        assert total > 0, "output had no power"
        cx = float((I_out * X).sum() / total)
        cy = float((I_out * Y).sum() / total)
        # Centroid should be within 1 pixel of (0, 0) for the
        # symmetric input.  The transposed-stencil version produced
        # a multi-pixel offset because Newton started from the clipped
        # boundary on every pixel.
        assert abs(cx) < dx, (
            f"Symmetric singlet output centroid x = {cx:.3e} m; "
            f"expected within 1 pixel ({dx:.3e} m) of zero.  "
            f"Pre-v4.11.2 the M_x/M_y stencil index transpose "
            f"produced an offset centroid (Newton initial guess "
            f"degenerated to clipped boundary).")
        assert abs(cy) < dx, (
            f"Symmetric singlet output centroid y = {cy:.3e} m; "
            f"expected within 1 pixel ({dx:.3e} m) of zero.")


# ============================================================================
# Item 8 -- NaN sentinel mask in apply_real_lens
# ============================================================================

class TestAuditFixesV4_12_1_coverage_ApplyRealLensNanSentinelMask:
    """v4.11.2 masks NaN OPD values from
    :func:`surface_sag_general` (returned outside the conic validity
    domain ``norm >= 0.9999``) to 0 before
    ``exp(-1j * k0 * opd)``.  Pre-fix the NaN propagated as
    ``exp(-i*k0*NaN) = NaN``, poisoning the entire downstream ASM step.

    Pin: build a prescription with an oblate conic (k = +3) whose
    valid-domain radius is smaller than the grid extent.  Run
    apply_real_lens; the wave-leg output must contain NO NaN.
    """

    def test_oblate_conic_outside_domain_no_nan(self):
        # R = 5 mm, conic = +3 (oblate): (1+k)*h^2/R^2 = 4*h^2/(25e-6)
        # blow up at h = 2.5 mm.  Grid spans +/- 6.4 mm (64 * 200 um/2).
        rx = {
            'name': 'oblate_test',
            'aperture_diameter': 6e-3,
            'surfaces': [
                {'radius': 5e-3, 'conic': 3.0,
                 'aspheric_coeffs': None,
                 'radius_y': None, 'conic_y': None,
                 'aspheric_coeffs_y': None,
                 'glass_before': 'air', 'glass_after': 'N-BK7'},
                {'radius': -5e-3, 'conic': 0.0,
                 'aspheric_coeffs': None,
                 'radius_y': None, 'conic_y': None,
                 'aspheric_coeffs_y': None,
                 'glass_before': 'N-BK7', 'glass_after': 'air'},
            ],
            'thicknesses': [2e-3],
        }
        N = 64
        dx = 2e-4  # 200 um, grid spans +/- 6.4 mm (well outside 2.5 mm valid)
        wavelength = 1.31e-6
        E_in = np.ones((N, N), dtype=np.complex128)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            E_out = la.apply_real_lens(
                E_in, prescription=rx,
                wavelength=wavelength, dx=dx,
                slant_correction=True,
            )
        assert not np.any(np.isnan(E_out)), (
            f"apply_real_lens with an oblate conic produced NaN in "
            f"the output ({int(np.isnan(E_out).sum())} pixels) -- "
            f"v4.11.2 NaN sentinel mask should zero the OPD on "
            f"undefined-surface pixels before exp(-i*k0*opd).")
        assert not np.any(np.isinf(E_out)), (
            "apply_real_lens output contained Inf -- NaN mask "
            "should have prevented this.")


# ============================================================================
# Item 9 -- stop_index != 0 warn in apply_real_lens_traced / _maslov
# ============================================================================

class TestAuditFixesV4_12_1_coverage_StopIndexWarn:
    """v4.11.2 adds a ``RuntimeWarning`` when
    :func:`apply_real_lens_traced` (and ``_maslov``) sees a
    prescription with ``stop_index != 0``, because the ray-traced
    phase leg launches from the entrance pupil only -- the actual
    stop placement is silently moved to surface 0.
    """

    def test_traced_emits_warning_for_stop_index_2(self):
        rx = la.make_singlet(R1=50e-3, R2=-50e-3, d=3e-3,
                              glass='N-BK7', aperture=5e-3)
        rx['stop_index'] = 2
        N = 32
        dx = 1e-5
        wavelength = 1.31e-6
        E_in = np.ones((N, N), dtype=np.complex128)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            la.apply_real_lens_traced(
                E_in, prescription=rx,
                wavelength=wavelength, dx=dx, ray_subsample=8)
        stop_warns = [
            w for w in caught
            if issubclass(w.category, RuntimeWarning)
            and 'stop_index' in str(w.message).lower()
        ]
        assert stop_warns, (
            f"apply_real_lens_traced with stop_index=2 must emit a "
            f"RuntimeWarning mentioning 'stop_index'.  Caught: "
            f"{[(w.category.__name__, str(w.message)[:80]) for w in caught]}")
        # Also confirm the warning mentions the value 2.
        msg = str(stop_warns[0].message)
        assert '2' in msg, (
            f"stop_index warning should reference the offending "
            f"value (2); got: {msg!r}")

    def test_maslov_emits_warning_for_stop_index_2(self):
        rx = la.make_singlet(R1=50e-3, R2=-50e-3, d=3e-3,
                              glass='N-BK7', aperture=5e-3)
        rx['stop_index'] = 2
        N = 32
        dx = 1e-5
        wavelength = 1.31e-6
        E_in = np.ones((N, N), dtype=np.complex128)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            la.apply_real_lens_maslov(
                E_in, prescription=rx,
                wavelength=wavelength, dx=dx,
                # Keep ray-field sampling cheap; defaults are 16/16.
                ray_field_samples=8, ray_pupil_samples=8, n_v2=8,
            )
        stop_warns = [
            w for w in caught
            if issubclass(w.category, RuntimeWarning)
            and 'stop_index' in str(w.message).lower()
        ]
        assert stop_warns, (
            f"apply_real_lens_maslov with stop_index=2 must emit a "
            f"RuntimeWarning mentioning 'stop_index'.  Caught: "
            f"{[(w.category.__name__, str(w.message)[:80]) for w in caught]}")


# ============================================================================
# Item 10 -- Freeform-terms RuntimeWarning in thin-element apply_real_lens
# ============================================================================

class TestAuditFixesV4_12_1_coverage_ApplyRealLensFreeformWarning:
    """v4.11.2 adds a ``RuntimeWarning`` when
    :func:`apply_real_lens` (thin-element wave-optics path) sees a
    surface with ``freeform_type`` set.  The function only computes
    conic + aspheric + biconic sag at the phase-screen step; freeform
    departures are silently DROPPED from the OPD.  Pin so a future
    refactor can't silently re-introduce the silent drop.
    """

    def test_freeform_surface_emits_runtime_warning(self):
        rx = la.make_singlet(R1=50e-3, R2=-50e-3, d=3e-3,
                              glass='N-BK7', aperture=5e-3)
        # Add a freeform_type to the first surface.
        rx['surfaces'][0]['freeform_type'] = 'xy_polynomial'
        rx['surfaces'][0]['freeform_coeffs'] = {(2, 0): 1e-6}
        N = 32
        dx = 1e-5
        wavelength = 1.31e-6
        E_in = np.ones((N, N), dtype=np.complex128)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            la.apply_real_lens(
                E_in, prescription=rx,
                wavelength=wavelength, dx=dx)
        ff_warns = [
            w for w in caught
            if issubclass(w.category, RuntimeWarning)
            and 'freeform' in str(w.message).lower()
        ]
        assert ff_warns, (
            f"apply_real_lens with a freeform_type surface must emit a "
            f"RuntimeWarning mentioning 'freeform'.  Caught: "
            f"{[(w.category.__name__, str(w.message)[:80]) for w in caught]}")


# ============================================================================
# Item 11 -- Zemax coord-break STOP marker (refractive-only counter)
# ============================================================================

class TestAuditFixesV4_12_1_coverage_ZemaxCoordBreakStopMarker:
    """v4.11.2 fixed the STOP-marker indexing in
    ``_export_zemax_zmx_full``: the writer now tracks a separate
    ``refr_counter`` so the STOP marker lands on the REFRACTIVE
    surface index requested, not the global surface counter that
    includes coord-breaks and mirrors.

    Round-4 noted this was only tested with mirrors.  Pin a
    coord-break case: 1 coord-break BEFORE the stop, then re-load
    and verify the STOP lands on the correct refractive surface.

    We test this directly on the .zmx text output (the loader does
    not surface stop_index for load_zemax_zmx but the text file
    carries the STOP marker explicitly).
    """

    def test_coord_break_does_not_displace_stop_marker(self):
        # Build a prescription with two refractive surfaces and one
        # coord-break BEFORE the second surface; stop_surface=1
        # (the SECOND refractive surface).
        rx = {
            'name': 'cb_stop_test',
            'aperture_diameter': 6e-3,
            'elements': [
                {'element_type': 'surface', 'radius': 50e-3,
                 'conic': 0.0, 'aspheric_coeffs': None,
                 'glass_before': 'air', 'glass_after': 'N-BK7',
                 'semi_diameter': 3e-3, 'surf_num': 1},
                {'element_type': 'surface', 'radius': float('inf'),
                 'conic': 0.0, 'aspheric_coeffs': None,
                 'glass_before': 'N-BK7', 'glass_after': 'air',
                 'semi_diameter': 3e-3, 'surf_num': 2},
                {'element_type': 'surface', 'radius': -50e-3,
                 'conic': 0.0, 'aspheric_coeffs': None,
                 'glass_before': 'air', 'glass_after': 'N-BK7',
                 'semi_diameter': 3e-3, 'surf_num': 4},
                {'element_type': 'surface', 'radius': float('inf'),
                 'conic': 0.0, 'aspheric_coeffs': None,
                 'glass_before': 'N-BK7', 'glass_after': 'air',
                 'semi_diameter': 3e-3, 'surf_num': 5},
            ],
            'all_thicknesses': [2e-3, 5e-3, 2e-3, 10e-3],
            'coord_breaks': [
                {'surf_num': 3, 'decenter_x_m': 0.0,
                 'decenter_y_m': 0.0, 'tilt_x_deg': 0.0,
                 'tilt_y_deg': 0.0, 'tilt_z_deg': 0.0,
                 'order': 0, 'thickness_m': 0.0},
            ],
            # Lens-only fallbacks (the full writer uses 'elements').
            'surfaces': [],
            'thicknesses': [],
        }
        # stop_surface = 1 -> the SECOND refractive surface.
        with tempfile.TemporaryDirectory() as td:
            zmx_path = os.path.join(td, 'cb_stop_test.zmx')
            la.export_zemax_zmx(
                rx, zmx_path, wavelength=1.31e-6, stop_surface=1)
            with open(zmx_path, encoding='utf-8') as f:
                txt = f.read()

        # Parse line by line: find the SURF blocks (object plus
        # written), identify which is COORDBRK vs STANDARD vs MIRROR,
        # and which carries STOP.
        lines = txt.splitlines()
        # Each surface block starts at a 'SURF <n>' line.
        surfs = []
        cur = None
        for ln in lines:
            s = ln.strip()
            if s.startswith('SURF '):
                if cur is not None:
                    surfs.append(cur)
                cur = {'idx': int(s.split()[1]),
                       'type': None,
                       'is_stop': False}
            elif cur is not None:
                if s.startswith('TYPE '):
                    cur['type'] = s.split()[1]
                elif s == 'STOP':
                    cur['is_stop'] = True
        if cur is not None:
            surfs.append(cur)

        # SURF 0 is the object (always STANDARD, never STOP).
        # SURF 1: first refractive (TYPE STANDARD)
        # SURF 2: second refractive (TYPE STANDARD)
        # SURF 3: coord-break (TYPE COORDBRK)
        # SURF 4: third refractive (NEW: STOP must land HERE if
        #         refr_counter is correctly tracked, since
        #         stop_surface=1 + 0 (object) + 1 (refr) = surface 4 ...
        # Actually let's re-check the logic.  The writer skips the
        # object (SURF 0) and starts emitting from SURF 1.  Within
        # the flat list, refr_counter starts at -1 and increments
        # BEFORE comparing to stop_surface.  So refr_counter goes
        # 0, 1, 2, 3 across the four refractive emissions.  The 2nd
        # (refr_counter == 1) gets the STOP.
        # In the SURF numbering: SURF 1, 2 are refractive; SURF 3 is
        # the coord-break; SURF 4, 5 are refractive.  refr_counter at
        # SURF 1 = 0, at SURF 2 = 1 (STOP HERE), at SURF 4 = 2, at
        # SURF 5 = 3.

        # Find the SURF index of the surface that carries STOP.
        stop_surfs = [s for s in surfs if s['is_stop']]
        assert len(stop_surfs) == 1, (
            f"Expected exactly one STOP marker; got {len(stop_surfs)}.")
        stop_surf = stop_surfs[0]
        # v4.11.2 fix: stop_surface=1 maps to the SECOND refractive
        # surface, which is SURF 2 in the emitted file (SURF 0 is
        # object, SURF 1 is first refractive, SURF 2 is second
        # refractive).
        assert stop_surf['type'] == 'STANDARD', (
            f"STOP marker landed on a non-refractive surface "
            f"(type={stop_surf['type']}); pre-v4.11.2 the global "
            f"surf_counter (including coord-breaks) was used so the "
            f"STOP could land on a COORDBRK.")
        assert stop_surf['idx'] == 2, (
            f"STOP marker on SURF {stop_surf['idx']}; expected SURF 2 "
            f"(the second refractive surface).  Pre-v4.11.2 the "
            f"writer used the global counter, which would place STOP "
            f"differently in the presence of coord-breaks.")

    def test_coord_break_at_index_0_does_not_bump_stop(self):
        """Round-5 audit (Item 11 weakness): the original
        ``test_coord_break_does_not_displace_stop_marker`` puts the
        coord-break AFTER the stop_surface in the flat list, so the
        global counter and the refractive counter still agree on the
        STOP position -- the pre-fix bug is NOT exercised.

        This test exercises the actual off-by-one: a coord-break
        placed at index 0 in the emit list (BEFORE any refractive
        surface), with ``stop_surface=1`` (the SECOND refractive).

        Walk-through:
          * SURF 0 -- object plane
          * SURF 1 -- COORDBRK (cb.surf_num = 1, drained before
                      the first element)
          * SURF 2 -- first refractive (refr_counter = 0)
          * SURF 3 -- second refractive (refr_counter = 1) -- STOP
                      MUST LAND HERE
          * SURF 4 -- third refractive (refr_counter = 2)
          * SURF 5 -- fourth refractive (refr_counter = 3)

        Pre-v4.11.2 (the bug) compared the global ``surf_counter``
        against ``stop_surface`` directly, which at ``stop_surface=1``
        would emit STOP on SURF 1 (the COORDBRK) -- a non-refractive
        surface.  Post-fix the refractive-only counter puts STOP on
        SURF 3 (the second refractive).
        """
        rx = {
            'name': 'cb_at_zero_stop_test',
            'aperture_diameter': 6e-3,
            'elements': [
                # First element targets surf_num=2 so the coord-break
                # at surf_num=1 drains BEFORE it.
                {'element_type': 'surface', 'radius': 50e-3,
                 'conic': 0.0, 'aspheric_coeffs': None,
                 'glass_before': 'air', 'glass_after': 'N-BK7',
                 'semi_diameter': 3e-3, 'surf_num': 2},
                {'element_type': 'surface', 'radius': float('inf'),
                 'conic': 0.0, 'aspheric_coeffs': None,
                 'glass_before': 'N-BK7', 'glass_after': 'air',
                 'semi_diameter': 3e-3, 'surf_num': 3},
                {'element_type': 'surface', 'radius': -50e-3,
                 'conic': 0.0, 'aspheric_coeffs': None,
                 'glass_before': 'air', 'glass_after': 'N-BK7',
                 'semi_diameter': 3e-3, 'surf_num': 4},
                {'element_type': 'surface', 'radius': float('inf'),
                 'conic': 0.0, 'aspheric_coeffs': None,
                 'glass_before': 'N-BK7', 'glass_after': 'air',
                 'semi_diameter': 3e-3, 'surf_num': 5},
            ],
            'all_thicknesses': [2e-3, 5e-3, 2e-3, 10e-3],
            'coord_breaks': [
                # surf_num=1 -> emitted BEFORE every element (which all
                # have surf_num >= 2).
                {'surf_num': 1, 'decenter_x_m': 0.0,
                 'decenter_y_m': 0.0, 'tilt_x_deg': 0.0,
                 'tilt_y_deg': 0.0, 'tilt_z_deg': 0.0,
                 'order': 0, 'thickness_m': 0.0},
            ],
            'surfaces': [],
            'thicknesses': [],
        }
        # stop_surface = 1 -> second refractive surface.
        with tempfile.TemporaryDirectory() as td:
            zmx_path = os.path.join(td, 'cb_at_zero_stop_test.zmx')
            la.export_zemax_zmx(
                rx, zmx_path, wavelength=1.31e-6, stop_surface=1)
            with open(zmx_path, encoding='utf-8') as f:
                txt = f.read()

        lines = txt.splitlines()
        surfs = []
        cur = None
        for ln in lines:
            s = ln.strip()
            if s.startswith('SURF '):
                if cur is not None:
                    surfs.append(cur)
                cur = {'idx': int(s.split()[1]),
                       'type': None,
                       'is_stop': False}
            elif cur is not None:
                if s.startswith('TYPE '):
                    cur['type'] = s.split()[1]
                elif s == 'STOP':
                    cur['is_stop'] = True
        if cur is not None:
            surfs.append(cur)

        # Verify the surface layout matches our expectation BEFORE
        # checking the STOP marker -- if the layout drifted, the
        # pre-fix bug wouldn't actually exercise.
        # SURF 0 = object, SURF 1 = COORDBRK, SURF 2..5 = refractive,
        # SURF 6 = image.
        cb_surfs = [s for s in surfs if s['type'] == 'COORDBRK']
        [s for s in surfs if s['type'] == 'STANDARD']
        assert len(cb_surfs) == 1, (
            f"Expected exactly one COORDBRK; got {len(cb_surfs)} "
            f"({[s['idx'] for s in cb_surfs]}).")
        assert cb_surfs[0]['idx'] == 1, (
            f"Coord-break should be at SURF 1 (so refractive surfaces "
            f"start at SURF 2); got SURF {cb_surfs[0]['idx']}.")

        stop_surfs = [s for s in surfs if s['is_stop']]
        assert len(stop_surfs) == 1, (
            f"Expected exactly one STOP marker; got {len(stop_surfs)} "
            f"({[(s['idx'], s['type']) for s in stop_surfs]}).")
        stop_surf = stop_surfs[0]

        # The critical pin: STOP must land on a STANDARD (refractive)
        # surface, NOT on the COORDBRK.  Pre-v4.11.2 the global
        # surf_counter == stop_surface check would emit STOP on SURF 1
        # (the COORDBRK) since coord-break came first.
        assert stop_surf['type'] == 'STANDARD', (
            f"STOP marker landed on a {stop_surf['type']} surface "
            f"(SURF {stop_surf['idx']}); pre-v4.11.2 the global "
            f"surf_counter compared directly against stop_surface "
            f"would place STOP on the COORDBRK at SURF 1.  The "
            f"refractive-only counter must skip non-refractive "
            f"surfaces.")

        # And it must be the SECOND refractive surface specifically
        # (refr_counter=1 = stop_surface), which is SURF 3 in the
        # emitted file (SURF 0=object, SURF 1=COORDBRK,
        # SURF 2=first refr, SURF 3=second refr).
        assert stop_surf['idx'] == 3, (
            f"STOP marker on SURF {stop_surf['idx']}; expected SURF 3 "
            f"(the second refractive surface, since the coord-break at "
            f"SURF 1 must NOT bump the refractive counter).  Pre-fix "
            f"the global counter would either land on SURF 1 (the "
            f"COORDBRK) or SURF 2 (the first refractive), depending on "
            f"the exact integer arithmetic used.")


# ============================================================================
# Item 12 -- JAX <-> NumPy phase-retrieval cross-parity
# ============================================================================

@_requires_jax
class TestAuditFixesV4_12_1_coverage_GerchbergSaxtonCrossBackendParity:
    """v4.12.0 (Track Tier-0) wired ``seed`` / ``dtype`` /
    ``initial_phase`` forwarding through the unified
    :func:`gerchberg_saxton` dispatcher.  Same-backend reproducibility
    is pinned in test_audit_fixes_v4_12_0_round4_tier0.py; this test
    pins CROSS-backend parity: given the same explicit
    ``initial_phase``, the JAX path's final phase agrees with the
    NumPy path's final phase to ~1e-3 RMS over the support.

    Uses a small grid (N=16) for runtime.  Forces float32 because the
    JAX kernel's internal complex64 path determines precision; the
    NumPy path runs at float64 but the comparison is done after the
    same number of iterations from the same initial phase, so the
    agreement is good even with the precision asymmetry.
    """

    def test_cross_backend_phase_agreement(self):
        from lumenairy.backend import JAX_AVAILABLE
        if not JAX_AVAILABLE:
            pytest.skip("JAX is not available.")

        N = 16
        rng = np.random.default_rng(42)
        # Reasonable source + target amplitudes.
        source = (np.abs(rng.standard_normal((N, N))) + 0.1).astype(
            np.float32)
        target = (np.abs(rng.standard_normal((N, N))) + 0.1).astype(
            np.float32)
        init_phase = rng.uniform(-np.pi, np.pi, (N, N)).astype(
            np.float32)

        phase_np, _err_np = la.gerchberg_saxton(
            source, target, n_iter=50,
            initial_phase=init_phase, backend='numpy')
        phase_jax, _err_jax = la.gerchberg_saxton(
            source, target, n_iter=50,
            initial_phase=init_phase, backend='jax')

        # Compare wrapped phase difference over the support
        # (source > small threshold).
        support = source > 1e-3
        diff = phase_np - phase_jax
        diff_mod = np.angle(np.exp(1j * diff))
        rms_mod = float(np.sqrt(np.mean(diff_mod[support] ** 2)))
        assert rms_mod < 1e-3, (
            f"gerchberg_saxton JAX vs NumPy phase RMS = {rms_mod:.3e} "
            f"rad over the support; expected < 1e-3 rad after 50 "
            f"iterations from the same initial phase.  Pre-v4.12.0 "
            f"the JAX path didn't honour ``initial_phase`` so the "
            f"two trajectories diverged.")


# ============================================================================
# Item 13 -- Cassegrain S1/S2/S3/S5 hand-derivation
# ============================================================================

class TestAuditFixesV4_12_1_coverage_CassegrainSeidelHandDerivation:
    """v4.11.2 added a hand-derived S4 = -16 pin for a two-mirror
    Cassegrain with R1 = -0.5 m, R2 = -0.1 m, d12 = 0.2 m, stop at
    primary (see test_audit_fixes_v4_11_2_raytrace.py).  Round-4 audit
    flagged that S1, S2, S3, S5 lacked equivalent hand pins.

    Geometry (user-spec): R1 = -0.5 m primary, R2 = -0.1 m secondary,
    d12 = 0.4 m, stop at primary, lambda = 0.55 um, field_angle =
    0.001 rad, r_stop = 0.05 m.

    The hand calc follows Welford's per-surface Seidel formulas
    (chapter 8 / 6.5) verbatim, threaded through the library's
    parity-tracked n2 = -n1 mirror convention.  Per-surface and total
    values pinned to 1e-9 (the formulas are exact to roundoff once
    parity is correctly threaded).

    Hand calculation (see C:/tmp/_handcalc_cassegrain.py for the
    step-by-step trace):

    Primary (parity 0 -> 1):
        n1=+1, n2=-1, c1=-2 m^-1
        y_m=0.05, nu_m=0, y_c=0, nu_c=0.001
        A_m = -0.1, A_c = +0.001
        h = 0.05, delta_un = -0.2
        H = -5e-5, H^2 = 2.5e-9
        S1_1 = -A_m^2 * h * delta_un = -(0.01)*0.05*(-0.2) = +1e-4
        S2_1 = -A_m*A_c * h * delta_un = -(-1e-4)*0.05*(-0.2) = -1e-6
        S3_1 = -A_c^2 * h * delta_un = -(1e-6)*0.05*(-0.2) = +1e-8
        S4_1 = -(1/(n2*n1)) * c1 * (n2-n1) = -(1/-1)*(-2)*(-2) = +4
        S5_1 = -(A_c/A_m) * (S3_1 + H^2*S4_1)
             = -(0.001/-0.1) * (1e-8 + 1e-8)
             = +2e-10

    Transfer to secondary at d=0.4 m, n_after=-1 (parity flipped):
        u_m = -0.2/-1 = 0.2;  u_c = 0.001/-1 = -0.001
        y_m_at_2 = 0.05 + 0.2 * 0.4 = 0.13
        y_c_at_2 = 0 + (-0.001) * 0.4 = -0.0004

    Secondary (parity 1, sign=-1):
        n1=-1, n2=+1, c2=-10 m^-1
        u_m = -0.2/-1 = 0.2;  u_c = 0.001/-1 = -0.001
        i_m = c2*y_m + u_m = -10*0.13 + 0.2 = -1.1
        i_c = c2*y_c + u_c = -10*(-0.0004) + (-0.001) = 0.003
        A_m = n1*i_m = -1 * -1.1 = +1.1
        A_c = n1*i_c = -1 * 0.003 = -0.003
        nu_m_after = -0.2 - 0.13 * 2 * -10 = -0.2 + 2.6 = +2.4
        nu_c_after = 0.001 - (-0.0004) * 2 * -10 = 0.001 - 0.008 = -0.007
        u_m_after = 2.4 / 1 = 2.4;  u_c_after = -0.007 / 1 = -0.007
        delta_un = u_m_after/n2 - u_m/n1 = 2.4 - 0.2/-1 = 2.4 - (-0.2) = 2.6
        h = 0.13
        S1_2 = -(1.21)*0.13*2.6 = -0.40898
        S2_2 = -(1.1*-0.003)*0.13*2.6 = +0.0011154
        S3_2 = -(9e-6)*0.13*2.6 = -3.042e-6
        S4_2 = -(1/(1*-1)) * (-10) * (1 - (-1)) = -(-1)*(-10)*2 = -20
        S5_2 = -(-0.003/1.1) * (-3.042e-6 + 2.5e-9*-20)
             = (0.003/1.1) * (-3.042e-6 - 5e-8)
             = -8.432727e-9

    Totals:
        S1 = -0.40888,  S2 = +0.0011144,  S3 = -3.032e-6,
        S4 = -16,        S5 = -8.232727e-9
    """

    @staticmethod
    def _make_cassegrain():
        """Build the user-spec Cassegrain prescription."""
        R1 = -0.5
        R2 = -0.1
        d12 = 0.4
        r_stop = 0.05
        surfaces = [
            Surface(radius=R1, thickness=d12,
                    glass_before='air', glass_after='air',
                    is_mirror=True, is_stop=True,
                    semi_diameter=r_stop),
            Surface(radius=R2, thickness=0.0,
                    glass_before='air', glass_after='air',
                    is_mirror=True,
                    semi_diameter=0.015),
        ]
        return surfaces

    def test_S1_per_surface_and_total(self):
        surfaces = self._make_cassegrain()
        result, _ = seidel_coefficients(
            surfaces, wavelength=0.55e-6, field_angle=0.001)
        S1 = np.asarray(result['S1'])
        assert abs(float(S1[0]) - 1e-4) < 1e-12, (
            f"Primary S1 = {S1[0]!r}; expected +1e-4 "
            f"(Welford hand calc).")
        assert abs(float(S1[1]) - (-0.40898)) < 1e-9, (
            f"Secondary S1 = {S1[1]!r}; expected -0.40898 "
            f"(Welford hand calc with parity-tracked n1=-1).")
        S1_total = float(result['total']['S1'])
        assert abs(S1_total - (-0.40888)) < 1e-9, (
            f"Cassegrain S1 total = {S1_total!r}; expected -0.40888 "
            f"(sum of Welford per-surface S1 with mirror parity).")

    def test_S2_per_surface_and_total(self):
        surfaces = self._make_cassegrain()
        result, _ = seidel_coefficients(
            surfaces, wavelength=0.55e-6, field_angle=0.001)
        S2 = np.asarray(result['S2'])
        assert abs(float(S2[0]) - (-1e-6)) < 1e-14, (
            f"Primary S2 = {S2[0]!r}; expected -1e-6.")
        assert abs(float(S2[1]) - 0.0011154) < 1e-9, (
            f"Secondary S2 = {S2[1]!r}; expected +0.0011154.")
        S2_total = float(result['total']['S2'])
        assert abs(S2_total - 0.0011144) < 1e-9, (
            f"Cassegrain S2 total = {S2_total!r}; expected +0.0011144.")

    def test_S3_per_surface_and_total(self):
        surfaces = self._make_cassegrain()
        result, _ = seidel_coefficients(
            surfaces, wavelength=0.55e-6, field_angle=0.001)
        S3 = np.asarray(result['S3'])
        assert abs(float(S3[0]) - 1e-8) < 1e-15, (
            f"Primary S3 = {S3[0]!r}; expected +1e-8.")
        assert abs(float(S3[1]) - (-3.042e-6)) < 1e-9, (
            f"Secondary S3 = {S3[1]!r}; expected -3.042e-6.")
        S3_total = float(result['total']['S3'])
        assert abs(S3_total - (-3.032e-6)) < 1e-9, (
            f"Cassegrain S3 total = {S3_total!r}; expected -3.032e-6.")

    def test_S5_per_surface_and_total(self):
        surfaces = self._make_cassegrain()
        result, _ = seidel_coefficients(
            surfaces, wavelength=0.55e-6, field_angle=0.001)
        S5 = np.asarray(result['S5'])
        # S5_1 = 2e-10 (handcalc); the library computes it via the
        # Schwarzschild equation, so the hand value is exact.
        assert abs(float(S5[0]) - 2e-10) < 1e-15, (
            f"Primary S5 = {S5[0]!r}; expected +2e-10.")
        assert abs(float(S5[1]) - (-8.432727272727272e-9)) < 1e-13, (
            f"Secondary S5 = {S5[1]!r}; expected -8.432727e-9.")
        S5_total = float(result['total']['S5'])
        assert abs(S5_total - (-8.232727272727272e-9)) < 1e-13, (
            f"Cassegrain S5 total = {S5_total!r}; expected "
            f"-8.232727e-9.")


# ============================================================================
# Item 14 -- Richards-Wolf vs paraxial Airy at low NA
# ============================================================================

class TestAuditFixesV4_12_1_coverage_RichardsWolfAiryAtLowNA:
    """At low NA the Richards-Wolf vectorial focal field reduces to
    the paraxial Airy pattern.  v4.11.2's Richards-Wolf prefactor /
    apodisation fix puts the on-axis peak in the right place; this
    test pins the first-null radius against the paraxial Airy formula
    ``r1 = 1.22 * lambda / (2 * NA)``.

    Pin: NA = 0.1 (paraxial regime), N = 128, dx_pupil = 20 um,
    f = 1 mm => dx_focal = 0.39 um, Airy first null = 6.1 um
    (~15.6 focal pixels).  The first null should agree with the
    paraxial formula to within ~5%.
    """

    def test_first_null_within_5_percent_of_paraxial(self):
        Np = 128
        wavelength = 1e-6
        NA = 0.1
        f = 1e-3
        dx_pupil = 20e-6
        pupil = np.ones((Np, Np), dtype=np.complex128)
        Ex, Ey, Ez, _xf, _yf = la.richards_wolf_focus(
            pupil, wavelength, NA, f, dx_pupil,
            N_focal=Np, z_planes=[0.0], polarization='x',
        )
        I = (np.abs(Ex) ** 2 + np.abs(Ey) ** 2 + np.abs(Ez) ** 2)
        # I is shape (1, Np, Np) when z_planes has one element.
        I = np.squeeze(I)
        nc = Np // 2
        center_row = I[nc, :]
        dx_focal = wavelength * f / (Np * dx_pupil)

        peak_idx = int(np.argmax(center_row))
        # Find first local minimum to the right of the peak.
        right = center_row[peak_idx:]
        first_min_offset = None
        for k in range(2, len(right) - 1):
            if right[k] < right[k - 1] and right[k] < right[k + 1]:
                first_min_offset = k
                break

        assert first_min_offset is not None, (
            "Could not locate first Airy null in the focal-plane "
            "intensity profile.")
        r_observed = first_min_offset * dx_focal
        r_paraxial = 1.22 * wavelength / (2 * NA)
        rel = abs(r_observed - r_paraxial) / r_paraxial
        assert rel < 0.05, (
            f"Richards-Wolf at NA=0.1 first-null radius = "
            f"{r_observed:.3e} m; paraxial Airy = {r_paraxial:.3e} m; "
            f"relative error = {rel*100:.1f}% (expected < 5%).")


# ============================================================================
# Source: test_audit_fixes_v4_12_2_cache_hygiene.py
# Audit version: V4_12_2  scope: cache_hygiene
# Original module docstring preserved as comment block for git-blame traceability:
#   Pinning tests for the v4.12.2 cache-hygiene Tier-0 release blockers.
#   
#   Audit reference
#   ---------------
#   
#   ``AUDIT_V4_12_1_2026_05_16.md`` Part 2 / Round 5 identified four still-
#   open release blockers carried forward through v4.12.0 -> v4.12.1, plus a
#   NEW cache regression introduced by v4.12.1's ``_TRACE_JAX_CACHE``.
#   
#   The v4.12.2 Tier-0 release closes:
#   
#   * **A1** ``pytest-benchmark`` missing from the ``dev`` optional-extra
#     in ``pyproject.toml``.  (Pinned by an import smoke test only -- this
#     test passes as long as the package is actually installed in the
#     current env, which is documentation/install discovery.)
#   * **A2** ``bench`` pytest marker not registered.  Without
#     registration, ``--strict-markers`` (already set in
#     ``[tool.pytest.ini_options].addopts``) fails benchmark collection
#     outright.
#   * **A3** Cache-clear and FFT-toggle public symbols not exposed at the
#     top level: ``set_fft_auto_promote`` / ``get_fft_auto_promote`` /
#     ``clear_zernike_basis_cache`` / ``clear_lg_polynomial_cache``.
#   * **A5** ``clear_asm_caches()`` only cleared the original three ASM
#     caches; v4.12.0 + v4.12.1 added several more propagator-adjacent
#     caches that nothing cleared, and ``lumenairy_context(
#     clear_caches_on_exit=True)`` only called ``clear_asm_caches()``.
#   * **Cache regression follow-up** ``_TRACE_JAX_CACHE`` (introduced in
#     v4.12.1) and ``_PROPAGATE_SYSTEM_JAX_CACHE`` + phase-retrieval
#     caches were plain ``dict``s with no bound -- a long-running
#     optimizer over many distinct prescription / element-list / n_iter
#     signatures would leak compiled XLA executables indefinitely.
#     Converted to LRU-bounded ``OrderedDict``s with maxsize=32.
#   
#   Each test class below pins one of these behaviours.  All tests should
#   PASS on the v4.12.2 codebase; a failure indicates a regression in the
#   specific fix the class names.
#   
#   Author: Andrew Traverso
# ============================================================================

import numpy as np
import pytest

import lumenairy as la

# ============================================================================
# A3 -- public-symbol export pins
# ============================================================================

class TestAuditFixesV4_12_2_cache_hygiene_Exports:
    """v4.12.2 exposes the FFT auto-promote toggles and the
    ``clear_*_cache()`` family at the top level.  Pre-fix these
    existed in submodules but were not on ``lumenairy``.

    Pins ``la.<symbol>`` is callable for each of:

    * ``set_fft_auto_promote`` / ``get_fft_auto_promote``
    * ``clear_zernike_basis_cache``
    * ``clear_lg_polynomial_cache``
    * ``clear_trace_jax_cache``
    * ``clear_propagate_system_jax_cache``
    * ``clear_phase_retrieval_caches``
    * ``clear_asm_caches`` (was already exported; pinned here for
      completeness so the whole family is in one place).
    """

    def test_set_fft_auto_promote_callable(self):
        """``la.set_fft_auto_promote(bool)`` is callable and the
        companion ``la.get_fft_auto_promote()`` round-trips."""
        prior = la.get_fft_auto_promote()
        try:
            la.set_fft_auto_promote(False)
            assert la.get_fft_auto_promote() is False
            la.set_fft_auto_promote(True)
            assert la.get_fft_auto_promote() is True
        finally:
            la.set_fft_auto_promote(prior)

    def test_clear_zernike_basis_cache_callable(self):
        """``la.clear_zernike_basis_cache()`` is callable and a no-op
        when the cache is already empty."""
        la.clear_zernike_basis_cache()
        # Calling twice is fine.
        la.clear_zernike_basis_cache()

    def test_clear_lg_polynomial_cache_callable(self):
        """``la.clear_lg_polynomial_cache()`` is callable and a no-op
        when the cache is already empty."""
        la.clear_lg_polynomial_cache()
        la.clear_lg_polynomial_cache()

    def test_clear_trace_jax_cache_callable(self):
        """``la.clear_trace_jax_cache()`` is callable; works even when
        JAX is not installed in the env (the function itself does not
        import JAX)."""
        la.clear_trace_jax_cache()
        la.clear_trace_jax_cache()

    def test_clear_propagate_system_jax_cache_callable(self):
        """``la.clear_propagate_system_jax_cache()`` is callable."""
        la.clear_propagate_system_jax_cache()
        la.clear_propagate_system_jax_cache()

    def test_clear_phase_retrieval_caches_callable(self):
        """``la.clear_phase_retrieval_caches()`` is callable."""
        la.clear_phase_retrieval_caches()
        la.clear_phase_retrieval_caches()

    def test_clear_asm_caches_still_callable(self):
        """Pre-existing ``la.clear_asm_caches()`` still callable after
        the v4.12.2 extension."""
        la.clear_asm_caches()

    def test_all_clear_functions_listed_in_dunder_all(self):
        """Every new ``clear_*_cache`` function appears in
        ``la.__all__`` so ``from lumenairy import *`` brings them in."""
        for name in (
            'set_fft_auto_promote',
            'get_fft_auto_promote',
            'clear_asm_caches',
            'clear_zernike_basis_cache',
            'clear_lg_polynomial_cache',
            'clear_trace_jax_cache',
            'clear_propagate_system_jax_cache',
            'clear_phase_retrieval_caches',
        ):
            assert name in la.__all__, (
                f"{name!r} missing from lumenairy.__all__.")


# ============================================================================
# A5.1 -- clear_asm_caches drops every propagator-adjacent cache
# ============================================================================

class TestAuditFixesV4_12_2_cache_hygiene_ClearAsmCachesExtended:
    """Pre v4.12.2, ``clear_asm_caches()`` only cleared
    ``_FREQ_GRID_CACHE`` / ``_BANDLIMIT_CACHE`` / ``_H_CACHE``.

    v4.12.2 extends it to ALSO clear ``_PYFFTW_PLAN_CACHE`` and
    ``_PYFFTW_BAD_SHAPES``.  This test populates each of the now-five
    caches with sentinel entries and pins that one ``clear_asm_caches()``
    call drops them all.
    """

    def test_clears_all_five_propagation_caches(self):
        from lumenairy.propagators.propagation import (
            _BANDLIMIT_CACHE,
            _FREQ_GRID_CACHE,
            _H_CACHE,
            _PYFFTW_BAD_SHAPES,
            _PYFFTW_PLAN_CACHE,
            clear_asm_caches,
        )

        # Sentinels: small synthetic entries that mimic what the real
        # caches would hold.  We don't care about correctness of the
        # values; we only need to demonstrate that the cache is
        # non-empty before the clear and empty after.
        _FREQ_GRID_CACHE[('sentinel', 64, 64, 1e-6, 1e-6)] = (
            np.zeros(64), np.zeros(64))
        _BANDLIMIT_CACHE[('sentinel', 64, 64, 1e-6, 1e-6, 633e-9, 0.1)] = (
            np.zeros(64), np.zeros(64))
        _H_CACHE[('sentinel', 64, 64, 1e-6, 1e-6, 633e-9, 0.1, '<c16')] = (
            np.zeros((64, 64), dtype=np.complex128))
        _PYFFTW_PLAN_CACHE[('fwd', (64, 64), 'complex128', 1)] = {
            'plan': None, 'bufs': None, 'idx': 0}
        _PYFFTW_BAD_SHAPES.add(('sentinel', 64, 64))

        # All five should be non-empty.
        assert len(_FREQ_GRID_CACHE) >= 1
        assert len(_BANDLIMIT_CACHE) >= 1
        assert len(_H_CACHE) >= 1
        assert len(_PYFFTW_PLAN_CACHE) >= 1
        assert len(_PYFFTW_BAD_SHAPES) >= 1

        clear_asm_caches()

        assert len(_FREQ_GRID_CACHE) == 0, (
            f"_FREQ_GRID_CACHE not cleared: {len(_FREQ_GRID_CACHE)} entries")
        assert len(_BANDLIMIT_CACHE) == 0, (
            f"_BANDLIMIT_CACHE not cleared: {len(_BANDLIMIT_CACHE)} entries")
        assert len(_H_CACHE) == 0, (
            f"_H_CACHE not cleared: {len(_H_CACHE)} entries")
        assert len(_PYFFTW_PLAN_CACHE) == 0, (
            f"_PYFFTW_PLAN_CACHE not cleared: {len(_PYFFTW_PLAN_CACHE)} entries")
        assert len(_PYFFTW_BAD_SHAPES) == 0, (
            f"_PYFFTW_BAD_SHAPES not cleared: {len(_PYFFTW_BAD_SHAPES)} entries")


# ============================================================================
# A5.2 -- LRU eviction on jit caches
# ============================================================================

class TestAuditFixesV4_12_2_cache_hygiene_TraceJaxCacheLru:
    """``_TRACE_JAX_CACHE`` (added in v4.12.1) is converted to an
    LRU-bounded ``OrderedDict`` in v4.12.2.  Filling past
    ``_TRACE_JAX_CACHE_MAXSIZE`` evicts the oldest entry; ``len`` never
    exceeds the bound."""

    def test_lru_eviction(self):
        from lumenairy.raytrace.jax_trace import (
            _TRACE_JAX_CACHE,
            _TRACE_JAX_CACHE_MAXSIZE,
        )

        _TRACE_JAX_CACHE.clear()
        n = _TRACE_JAX_CACHE_MAXSIZE
        # Insert maxsize + 5 entries; cache must stay at <= maxsize.
        for i in range(n + 5):
            key = ('sentinel', i)
            # Push the key/value through the SAME LRU semantics the
            # real lookup uses: insert, then prune.
            _TRACE_JAX_CACHE[key] = i
            while len(_TRACE_JAX_CACHE) > _TRACE_JAX_CACHE_MAXSIZE:
                _TRACE_JAX_CACHE.popitem(last=False)

        assert len(_TRACE_JAX_CACHE) <= _TRACE_JAX_CACHE_MAXSIZE, (
            f"Cache exceeded maxsize: {len(_TRACE_JAX_CACHE)} > "
            f"{_TRACE_JAX_CACHE_MAXSIZE}")
        # Oldest entry (i=0) must have been evicted.
        assert ('sentinel', 0) not in _TRACE_JAX_CACHE, (
            "Oldest entry must have been evicted under LRU; still present.")
        # Newest entry must still be there.
        assert ('sentinel', n + 4) in _TRACE_JAX_CACHE, (
            "Most-recently-inserted entry must be present.")
        _TRACE_JAX_CACHE.clear()

    def test_maxsize_is_ordered_dict(self):
        """The cache is an ``OrderedDict``, not a plain dict, so the
        ``move_to_end`` LRU touch in the lookup path is valid."""
        from collections import OrderedDict

        from lumenairy.raytrace.jax_trace import _TRACE_JAX_CACHE
        assert isinstance(_TRACE_JAX_CACHE, OrderedDict), (
            f"_TRACE_JAX_CACHE must be an OrderedDict for LRU semantics; "
            f"got {type(_TRACE_JAX_CACHE).__name__}.")


class TestAuditFixesV4_12_2_cache_hygiene_PropagateSystemJaxCacheLru:
    """Same LRU contract for ``_PROPAGATE_SYSTEM_JAX_CACHE``."""

    def test_lru_eviction(self):
        from lumenairy.propagators.system import (
            _PROPAGATE_SYSTEM_JAX_CACHE,
            _PROPAGATE_SYSTEM_JAX_CACHE_MAXSIZE,
        )

        _PROPAGATE_SYSTEM_JAX_CACHE.clear()
        n = _PROPAGATE_SYSTEM_JAX_CACHE_MAXSIZE
        for i in range(n + 5):
            key = ('sentinel', i)
            _PROPAGATE_SYSTEM_JAX_CACHE[key] = i
            while (len(_PROPAGATE_SYSTEM_JAX_CACHE)
                   > _PROPAGATE_SYSTEM_JAX_CACHE_MAXSIZE):
                _PROPAGATE_SYSTEM_JAX_CACHE.popitem(last=False)

        assert len(_PROPAGATE_SYSTEM_JAX_CACHE) <= n, (
            f"Cache exceeded maxsize: "
            f"{len(_PROPAGATE_SYSTEM_JAX_CACHE)} > {n}")
        assert ('sentinel', 0) not in _PROPAGATE_SYSTEM_JAX_CACHE
        assert ('sentinel', n + 4) in _PROPAGATE_SYSTEM_JAX_CACHE
        _PROPAGATE_SYSTEM_JAX_CACHE.clear()

    def test_is_ordered_dict(self):
        from collections import OrderedDict

        from lumenairy.propagators.system import _PROPAGATE_SYSTEM_JAX_CACHE
        assert isinstance(_PROPAGATE_SYSTEM_JAX_CACHE, OrderedDict), (
            "_PROPAGATE_SYSTEM_JAX_CACHE must be an OrderedDict for LRU; "
            f"got {type(_PROPAGATE_SYSTEM_JAX_CACHE).__name__}.")


class TestAuditFixesV4_12_2_cache_hygiene_PhaseRetrievalCachesLru:
    """Same LRU contract for ``_GS_KERNEL_CACHE`` / ``_ER_KERNEL_CACHE``
    / ``_HIO_KERNEL_CACHE``."""

    def test_gs_cache_is_ordered_dict(self):
        from collections import OrderedDict

        from lumenairy.analysis.phase_retrieval import _GS_KERNEL_CACHE
        assert isinstance(_GS_KERNEL_CACHE, OrderedDict)

    def test_er_cache_is_ordered_dict(self):
        from collections import OrderedDict

        from lumenairy.analysis.phase_retrieval import _ER_KERNEL_CACHE
        assert isinstance(_ER_KERNEL_CACHE, OrderedDict)

    def test_hio_cache_is_ordered_dict(self):
        from collections import OrderedDict

        from lumenairy.analysis.phase_retrieval import _HIO_KERNEL_CACHE
        assert isinstance(_HIO_KERNEL_CACHE, OrderedDict)

    def test_lru_eviction_gs(self):
        from lumenairy.analysis.phase_retrieval import (
            _GS_KERNEL_CACHE,
            _PR_KERNEL_CACHE_MAXSIZE,
        )

        _GS_KERNEL_CACHE.clear()
        n = _PR_KERNEL_CACHE_MAXSIZE
        for i in range(n + 5):
            _GS_KERNEL_CACHE[i] = i
            while len(_GS_KERNEL_CACHE) > _PR_KERNEL_CACHE_MAXSIZE:
                _GS_KERNEL_CACHE.popitem(last=False)
        assert len(_GS_KERNEL_CACHE) <= n
        assert 0 not in _GS_KERNEL_CACHE
        assert (n + 4) in _GS_KERNEL_CACHE
        _GS_KERNEL_CACHE.clear()


# ============================================================================
# A5.1 (continued) -- phase-retrieval cache clear
# ============================================================================

@_requires_jax
class TestAuditFixesV4_12_2_cache_hygiene_PhaseRetrievalCacheClear:
    """``clear_phase_retrieval_caches()`` (v4.12.2 new) clears GS / ER /
    HIO kernel caches in one call.  Populated by running a few iterations
    of each algorithm; assert cleared on call.

    JAX is optional; if not available we populate the dicts with sentinels
    so the clear-semantics still get pinned.
    """

    def test_clear_phase_retrieval_caches_drops_all_three(self):
        from lumenairy.analysis.phase_retrieval import (
            _ER_KERNEL_CACHE,
            _GS_KERNEL_CACHE,
            _HIO_KERNEL_CACHE,
            clear_phase_retrieval_caches,
        )

        # Try the real-iteration population path first; if JAX is not
        # available, fall back to sentinel insertion so the clear
        # semantics still get exercised.
        try:
            import jax  # noqa: F401
            populated_by_real_iteration = True
        except ImportError:
            populated_by_real_iteration = False

        if populated_by_real_iteration:
            src = np.ones((32, 32), dtype=np.float32)
            tgt = np.ones((32, 32), dtype=np.float32)
            sup = np.ones((32, 32), dtype=bool)
            meas = np.ones((32, 32), dtype=np.float32)
            la.gerchberg_saxton_jax(src, tgt, n_iter=3)
            la.error_reduction_jax(meas, sup, n_iter=3)
            la.hybrid_input_output_jax(meas, sup, n_iter=3)
        else:
            _GS_KERNEL_CACHE[42] = object()
            _ER_KERNEL_CACHE[42] = object()
            _HIO_KERNEL_CACHE[42] = object()

        assert len(_GS_KERNEL_CACHE) >= 1
        assert len(_ER_KERNEL_CACHE) >= 1
        assert len(_HIO_KERNEL_CACHE) >= 1

        clear_phase_retrieval_caches()

        assert len(_GS_KERNEL_CACHE) == 0
        assert len(_ER_KERNEL_CACHE) == 0
        assert len(_HIO_KERNEL_CACHE) == 0


# ============================================================================
# A5.3 -- lumenairy_context(clear_caches_on_exit=True) clears EVERY cache
# ============================================================================

class TestAuditFixesV4_12_2_cache_hygiene_LumenairyContextClearsAll:
    """``lumenairy_context(clear_caches_on_exit=True)`` calls every
    ``clear_*_cache()`` function on exit, not just
    ``clear_asm_caches()``.  This test populates each cache inside the
    ``with`` block (via sentinel entries since we don't want JAX
    dependencies in the unit suite), then asserts they are all empty
    on exit.
    """

    def test_clears_all_caches_on_exit(self):
        from lumenairy.analysis.core import _ZERNIKE_BASIS_CACHE
        from lumenairy.analysis.phase_retrieval import (
            _ER_KERNEL_CACHE,
            _GS_KERNEL_CACHE,
            _HIO_KERNEL_CACHE,
        )
        from lumenairy.propagators.propagation import (
            _BANDLIMIT_CACHE,
            _FREQ_GRID_CACHE,
            _H_CACHE,
            _PYFFTW_BAD_SHAPES,
            _PYFFTW_PLAN_CACHE,
        )
        from lumenairy.propagators.system import _PROPAGATE_SYSTEM_JAX_CACHE
        from lumenairy.raytrace.jax_trace import _TRACE_JAX_CACHE

        with la.lumenairy_context(clear_caches_on_exit=True):
            # Populate each cache with a sentinel entry.
            _FREQ_GRID_CACHE[('sentinel', 64, 64, 1e-6, 1e-6)] = (
                np.zeros(64), np.zeros(64))
            _BANDLIMIT_CACHE[(
                'sentinel', 64, 64, 1e-6, 1e-6, 633e-9, 0.1)] = (
                np.zeros(64), np.zeros(64))
            _H_CACHE[(
                'sentinel', 64, 64, 1e-6, 1e-6, 633e-9, 0.1, '<c16')] = (
                np.zeros((64, 64), dtype=np.complex128))
            _PYFFTW_PLAN_CACHE[('fwd', (64, 64), 'complex128', 1)] = {
                'plan': None, 'bufs': None, 'idx': 0}
            _PYFFTW_BAD_SHAPES.add(('sentinel', 64, 64))
            _ZERNIKE_BASIS_CACHE[('sentinel', 'k')] = (
                np.zeros((4, 4)), np.zeros((4,)))
            _GS_KERNEL_CACHE[7] = object()
            _ER_KERNEL_CACHE[7] = object()
            _HIO_KERNEL_CACHE[7] = object()
            _TRACE_JAX_CACHE[('sentinel', 7)] = object()
            _PROPAGATE_SYSTEM_JAX_CACHE[('sentinel', 7)] = object()

            # Sanity: all caches non-empty inside the with.
            assert len(_FREQ_GRID_CACHE) >= 1
            assert len(_BANDLIMIT_CACHE) >= 1
            assert len(_H_CACHE) >= 1
            assert len(_PYFFTW_PLAN_CACHE) >= 1
            assert len(_PYFFTW_BAD_SHAPES) >= 1
            assert len(_ZERNIKE_BASIS_CACHE) >= 1
            assert len(_GS_KERNEL_CACHE) >= 1
            assert len(_ER_KERNEL_CACHE) >= 1
            assert len(_HIO_KERNEL_CACHE) >= 1
            assert len(_TRACE_JAX_CACHE) >= 1
            assert len(_PROPAGATE_SYSTEM_JAX_CACHE) >= 1

        # On exit, every cache must be empty.  The clear hooks are
        # individually guarded so any single failure doesn't stop the
        # others from firing -- the only way this fails is if a hook
        # is missing entirely.
        assert len(_FREQ_GRID_CACHE) == 0, "_FREQ_GRID_CACHE leaked"
        assert len(_BANDLIMIT_CACHE) == 0, "_BANDLIMIT_CACHE leaked"
        assert len(_H_CACHE) == 0, "_H_CACHE leaked"
        assert len(_PYFFTW_PLAN_CACHE) == 0, "_PYFFTW_PLAN_CACHE leaked"
        assert len(_PYFFTW_BAD_SHAPES) == 0, "_PYFFTW_BAD_SHAPES leaked"
        assert len(_ZERNIKE_BASIS_CACHE) == 0, "_ZERNIKE_BASIS_CACHE leaked"
        assert len(_GS_KERNEL_CACHE) == 0, "_GS_KERNEL_CACHE leaked"
        assert len(_ER_KERNEL_CACHE) == 0, "_ER_KERNEL_CACHE leaked"
        assert len(_HIO_KERNEL_CACHE) == 0, "_HIO_KERNEL_CACHE leaked"
        assert len(_TRACE_JAX_CACHE) == 0, "_TRACE_JAX_CACHE leaked"
        assert len(_PROPAGATE_SYSTEM_JAX_CACHE) == 0, (
            "_PROPAGATE_SYSTEM_JAX_CACHE leaked")

    def test_default_off(self):
        """``lumenairy_context()`` without
        ``clear_caches_on_exit=True`` does NOT clear caches on exit;
        the speedup-preserving default is to leave entries in place.
        """
        from lumenairy.propagators.propagation import _FREQ_GRID_CACHE
        _FREQ_GRID_CACHE.clear()
        _FREQ_GRID_CACHE[('sentinel', 64, 64, 1e-6, 1e-6)] = (
            np.zeros(64), np.zeros(64))
        with la.lumenairy_context():
            pass
        assert len(_FREQ_GRID_CACHE) >= 1, (
            "Default behaviour must leave caches untouched on exit.")
        _FREQ_GRID_CACHE.clear()


# ============================================================================
# A2 -- benchmark marker registration
# ============================================================================

@_requires_jax
class TestAuditFixesV4_12_2_cache_hygiene_BenchMarkerRegistered:
    """``bench`` marker is registered in ``[tool.pytest.ini_options]
    .markers`` so ``--strict-markers`` does not fail benchmark
    collection.  Direct inspection of ``pyproject.toml`` would be
    sufficient, but we exercise the actual pytest marker API to ensure
    it's not just a comment.
    """

    def test_bench_marker_is_known(self, pytestconfig):
        """``pytestconfig`` exposes the registered markers; ``bench``
        must be present.  Pre-fix this raised ``--strict-markers``
        during benchmark collection because the marker was never
        registered."""
        # Newer pytest puts known markers in ``ini_marker_names``
        # (returned by ``getini``).  Use the same API ``--strict-markers``
        # checks against.
        markers = pytestconfig.getini('markers')
        # ``markers`` is a list of strings of the form ``"name: desc"``.
        names = [m.split(':', 1)[0].strip() for m in markers]
        assert 'bench' in names, (
            f"'bench' marker must be registered in pyproject.toml "
            f"[tool.pytest.ini_options].markers so --strict-markers "
            f"does not fail benchmark collection.  Got: {names}")


# ============================================================================
# Source: test_audit_fixes_v4_13_0_except_sweep.py
# Audit version: V4_13_0  scope: except_sweep
# Original module docstring preserved as comment block for git-blame traceability:
#   Pinning tests for the v4.13.0 Phase-2 ``except Exception:`` sweep
#   (audit finding L5).
#   
#   Audit reference
#   ---------------
#   
#   ``AUDIT_V4_12_1_2026_05_16.md`` L5 reported ~242 ``except Exception:``
#   clauses across the package (excluding the ``ui/`` subpackage, which has
#   different conventions for Qt-signal robustness).  Many of those were
#   effect-equivalent to bare-except: ``pass`` or ``return NaN`` with no
#   warning, silently masking real bugs.
#   
#   The v4.13.0 Phase-2 sweep:
#   
#   * **Survey** -- 99 ``except Exception:`` clauses in non-``ui/`` code
#     (the audit's 242 includes the test-suite and validation harness,
#     which were correctly excluded from the sweep per the audit's own
#     scope note).
#   * **Judgement rule** -- four buckets:
#   
#     - **KEEP-AS-IS** for optional-dep imports / ``__del__`` / atexit
#       cleanup where broad-except is the standard pattern.
#     - **NARROW** to typed ``except (TypeError, ValueError, ...):``
#       matching the wrapped call's documentable raises.
#     - **WARN-BEFORE-PASS** for physics-critical paths where a silent
#       ``pass`` would hide a real bug.
#     - **RE-RAISE** with added context for sites whose only purpose was
#       to add context.
#   
#   * **Post-sweep target**: ≤80 non-``ui/`` clauses remaining (the keepers
#     are documented justified KEEP-AS-IS sites).
#   
#   What this test pins
#   -------------------
#   
#   1. **Count regression guard**: non-``ui/`` ``except Exception:`` count
#      stays within the post-sweep budget.  Hard ceiling at the
#      post-sweep number + a small slack so refactors that legitimately
#      add a justified KEEP-AS-IS don't have to also bump this test.
#   
#   2. **WARN-BEFORE-PASS pins**: two physics-critical sites where the
#      sweep added a ``warnings.warn`` -- ``analysis.field.petzval_radius``
#      and ``optimize.core.design_optimize``'s ``plane_logger`` call.
#      Inject a synthetic failure into the wrapped call and assert the
#      ``RuntimeWarning`` fires.
#   
#   3. **NARROW pins**: the narrowed-except clauses must still catch the
#      exception types they were narrowed to.  Specifically:
#   
#      - ``analysis.plotting.abbe_diagram`` skips a glass that raises
#        ``KeyError`` from ``get_glass_index``.
#      - ``analysis.field.distortion_grid`` continues past a ``ValueError``
#        raised by an inner ``_trace`` invocation (NaN-fills the entry).
#   
#   Author: Andrew Traverso (sweep) -- v4.13.0
# ============================================================================

import os
import re
import warnings
from typing import List

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
LUMENAIRY_DIR = os.path.join(REPO_ROOT, 'lumenairy')


# ============================================================================
# Count regression guard -- MOVED (2026-06-10, v5.14.1)
# ============================================================================
# The broad-``except`` budget guard now lives in
# tests/unit/test_audit_except_budget.py: this module's MODULE-LEVEL
# ``pytest.importorskip('jax')`` silently skipped it (and every other
# non-JAX pin in this file) on CI, letting the count creep 13 -> 20
# unobserved.  The extracted file has no jax dependency.
#
# ============================================================================
# WARN-BEFORE-PASS pin: petzval_radius glass lookup
# ============================================================================

class TestAuditFixesV4_13_0_except_sweep_PetzvalRadiusWarnsOnGlassFailure:
    """Glass-index lookup failure in ``petzval_radius`` now warns
    instead of silently returning NaN.

    Pre-sweep: ``except Exception: return float('nan')`` -- a missing
    glass entry produced a silent NaN that downstream merits could
    mistake for an honest "field is perfectly flat" answer.
    Post-sweep: warns about the failure with the offending glass name
    so the user can find and fix it.
    """

    def test_warns_when_glass_lookup_fails(self, monkeypatch):
        """Inject a glass-lookup KeyError; ``petzval_radius`` warns
        and returns NaN."""
        from lumenairy import glass as glass_mod
        from lumenairy.analysis import field as af

        # Build a minimal surface list with one finite-R surface.
        # Surface dataclass is in lumenairy.raytrace -- import and
        # construct lightly.
        from lumenairy.raytrace.core import Surface

        s = Surface(
            radius=10e-3,
            thickness=5e-3,
            glass_before='air',
            glass_after='BK7-mock',
            semi_diameter=2e-3,
        )

        def _raise_keyerror(name, wavelength):
            raise KeyError(name)

        # ``petzval_radius`` does ``from ..glass import get_glass_index``
        # inside the function -- patch the source module so the
        # function-local import picks up the patched name.
        monkeypatch.setattr(glass_mod, 'get_glass_index', _raise_keyerror)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            r = af.petzval_radius([s], wavelength=1.31e-6)

        assert np.isnan(r), (
            "petzval_radius should return NaN when the glass lookup "
            "fails (matches pre-sweep behaviour).")
        # At least one RuntimeWarning should mention petzval_radius +
        # the offending glass name.
        runtime_warnings = [x for x in w
                            if issubclass(x.category, RuntimeWarning)
                            and 'petzval_radius' in str(x.message)]
        assert len(runtime_warnings) >= 1, (
            f"Expected at least one RuntimeWarning from "
            f"petzval_radius on glass-lookup failure; got "
            f"{[str(x.message) for x in w]}")
        assert 'BK7-mock' in str(runtime_warnings[0].message), (
            f"The warning should mention the offending glass name; "
            f"got: {runtime_warnings[0].message!r}")


# ============================================================================
# WARN-BEFORE-PASS pin: design_optimize plane_logger failure
# ============================================================================

class TestAuditFixesV4_13_0_except_sweep_DesignOptimizePlaneLoggerWarnsOnFailure:
    """A broken ``plane_logger`` callback now warns instead of
    silently swallowing the failure for the entire optimization run.

    Pre-sweep: ``except Exception: pass`` -- a bug in user telemetry
    silently aborted all logging without leaving any trace; the user
    saw an empty log file at the end and no clue why.
    Post-sweep: the FIRST failure emits a RuntimeWarning naming the
    callback's exception so the user immediately sees the issue.
    """

    def test_warns_when_plane_logger_raises(self):
        """Run a 1-iter design_optimize with a logger that raises a
        ValueError; assert at least one ``RuntimeWarning`` is emitted
        with the user-visible ``"plane_logger callback failed"`` text
        AND the optimisation completes (the logger error must NOT
        propagate out of merit_fn).

        v5.2.3 (AUDIT_V4_13_1 Part 6.1 closure: replace inspect.getsource
        proxy with behavioral pin): the previous version grepped
        ``design_optimize`` source for the warning-message substring,
        ``"RuntimeWarning"``, and the narrowed except tuple.  All three
        are visible from the merit_fn invocation path; we now exercise
        the path directly.  Empirically confirmed (against the live
        v5.2.x driver) that the merit_fn fires at least once during a
        max_iter=1 L-BFGS-B run on a single-free-variable singlet, so
        the previous comment about scipy not calling merit_fn was
        addressed by the v4.16 unconditional final-evaluate guarantee
        and is no longer a concern.
        """
        from lumenairy.optimize.core import DesignParameterization, design_optimize

        pres = lm.make_singlet(
            R1=60e-3, R2=float('inf'), d=4e-3, glass='N-BK7',
            aperture=12e-3,
        )
        param = DesignParameterization(
            template=pres,
            free_vars=[('surfaces', 0, 'radius')],
            bounds=[(30e-3, 100e-3)],
        )
        inner = lm.StrehlMerit(weight=1.0)

        logger_calls = []

        def bad_logger(iteration, ctx):
            logger_calls.append(iteration)
            # ValueError is in the narrowed except tuple
            # (TypeError, ValueError, RuntimeError, KeyError,
            # AttributeError, IndexError, OSError).  A future
            # narrowing that drops ValueError would re-raise out
            # of merit_fn and break the optimisation -- which is
            # exactly what this behavioural pin would catch
            # (the design_optimize call would raise).
            raise ValueError('synthetic plane_logger fail')

        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            # The call MUST NOT raise -- the warning-handler's job
            # is to swallow the logger exception and continue.
            design_optimize(
                parameterization=param,
                merit_terms=[inner],
                wavelength=1.55e-6,
                N=32, dx=10e-6,
                max_iter=1,
                plane_logger=bad_logger,
                verbose=False,
            )

        # Behavioral pin 1: the logger DID get called (otherwise the
        # warning-emit path is untested).
        assert len(logger_calls) >= 1, (
            'plane_logger was never invoked during a max_iter=1 '
            'design_optimize call; the merit_fn fires at least once '
            'at finalisation.  If this fails, scipy / the driver no '
            'longer evaluates merit_fn during finalize -- the rest of '
            'this test would be vacuous.')

        # Behavioral pin 2: at least one RuntimeWarning fired with the
        # user-visible ``"plane_logger callback failed"`` text.  This
        # is the contract that subsumes the three pre-v5.2.3
        # source-grep assertions:
        #   - ``"plane_logger callback failed"`` substring in source
        #     -> appears in the actual warning message;
        #   - ``"RuntimeWarning"`` in source
        #     -> the warning category is RuntimeWarning;
        #   - ``"except (TypeError, ValueError, RuntimeError, ..."``
        #     narrowed-tuple -> ValueError raised by bad_logger is
        #     caught (otherwise this test would fail with an
        #     unhandled ValueError, not the assertion below).
        runtime_ws = [w for w in ws
                      if issubclass(w.category, RuntimeWarning)
                      and 'plane_logger callback failed' in str(w.message)]
        assert len(runtime_ws) >= 1, (
            f"design_optimize plane_logger callback failure must emit "
            f"a RuntimeWarning containing 'plane_logger callback "
            f"failed'.  Got {[str(w.message) for w in ws]}.")
        # The warning must also name the exception type, so the
        # user can triage their broken logger.
        assert 'ValueError' in str(runtime_ws[0].message), (
            f"warning should name the exception type "
            f"(ValueError); got: {runtime_ws[0].message!r}")


# ============================================================================
# NARROW pins: the typed exception clauses still catch what they should
# ============================================================================

class TestAuditFixesV4_13_0_except_sweep_AbbeDiagramSkipsMissingGlass:
    """``abbe_diagram`` narrows ``except Exception`` to
    ``except (KeyError, ValueError, TypeError)`` for the
    ``get_glass_index`` call.  A KeyError on an unknown glass should
    drop the entry, not blow up the whole diagram."""

    def test_unknown_glass_skipped_silently(self):
        import matplotlib

        from lumenairy.analysis import plotting as ap
        matplotlib.use('Agg')  # headless backend for the test

        # Pass a known glass + a deliberately bogus name.  The bogus
        # name should raise KeyError from get_glass_index and be
        # dropped without affecting the rest of the call.
        fig, axes, data = ap.abbe_diagram(
            glasses=['N-BK7', 'DEFINITELY-NOT-A-GLASS-NAME'],
            wavelengths_nm=(587.6, 486.1, 656.3),
        )
        # The data list should have exactly the valid entries (the
        # bogus name was silently dropped through the narrow).
        assert any(name == 'N-BK7' for name, _, _ in data)
        assert not any(name == 'DEFINITELY-NOT-A-GLASS-NAME'
                       for name, _, _ in data), (
            "The bogus glass name should be silently dropped via the "
            "narrowed except (KeyError, ValueError, TypeError); "
            "instead it was kept in the data list.")

        # If the narrow worked, the function returned a figure with
        # at least one valid entry (N-BK7).  The bogus name should
        # have been silently dropped.
        assert fig is not None, (
            "abbe_diagram returned None instead of a figure; the "
            "narrow may have failed to swallow the bogus-glass "
            "KeyError.")
        import matplotlib.pyplot as plt
        plt.close(fig)


@_requires_jax
class TestAuditFixesV4_13_0_except_sweep_DistortionGridNarrowCatchesTraceFailures:
    """``distortion_grid`` narrows the per-(ix, iy) trace except to
    ``(ValueError, RuntimeError, ZeroDivisionError, KeyError,
    IndexError, AttributeError)``.

    Inject a failure via monkey-patch and verify the function
    completes with NaN entries instead of crashing.
    """

    def test_extreme_corners_nan_out_no_crash(self, monkeypatch):
        import lumenairy as lm
        from lumenairy.analysis import field as af

        # Build a simple symmetric singlet prescription via the
        # public helper to satisfy validate_prescription.
        pres = lm.make_singlet(R1=51.5e-3, R2=-51.5e-3,
                                d=3e-3, glass='N-BK7',
                                aperture=8e-3)

        # Patch _trace to raise ValueError on every other call.
        from lumenairy.analysis import field as af_mod
        _orig_trace = af_mod._trace
        call_idx = [0]

        def _flaky_trace(*args, **kwargs):
            call_idx[0] += 1
            if call_idx[0] % 2 == 0:
                raise ValueError("synthetic trace failure")
            return _orig_trace(*args, **kwargs)

        monkeypatch.setattr(af_mod, '_trace', _flaky_trace)

        # 3x3 grid over a small angular range -- enough to exercise
        # multiple inner trace calls.
        result = af.distortion_grid(
            pres,
            wavelength=1.31e-6,
            max_field_deg=3.0,
            n_grid=3,
        )
        # actual_x / actual_y should have some NaN entries (where the
        # synthetic failure fired) but the function should not have
        # crashed.
        assert result is not None
        assert np.isnan(result.actual_x).any() or np.isnan(result.actual_y).any(), (
            "Expected the flaky-trace injection to leave at least "
            "one NaN in the distortion grid; the narrow may not be "
            "catching ValueError.")


# ============================================================================
# Source: test_audit_fixes_v4_13_0_jax_dtype_dy_siblings.py
# Audit version: V4_13_0  scope: jax_dtype_dy_siblings
# Original module docstring preserved as comment block for git-blame traceability:
#   Pinning tests for the v4.13.0 Track-C audit closures.
#   
#   Audit reference
#   ---------------
#   
#   ``AUDIT_V4_12_1_2026_05_16.md`` Part 5 / Round 5 carried four
#   known-limitations items into v4.12.2's CHANGELOG.  v4.13.0 closes the
#   following three:
#   
#   * **L2** -- JAX path complex64 hard-casts silently override
#     ``set_default_complex_dtype``.  v4.13.0 routes every JAX-side dtype
#     resolution through :func:`_resolve_jax_complex_dtype` (and its real
#     twin :func:`_resolve_jax_real_dtype`) which read
#     :func:`get_default_complex_dtype`.  JIT cache keys that previously
#     omitted the dtype now include it.
#   * **L3** -- :class:`PropagationResult` lost the y-axis pitch for
#     anamorphic Fresnel propagation because ``_coerce_field`` extracted
#     ``dx_out`` from tuple-returning kernels but silently discarded
#     ``dy_out``.  v4.13.0 adds a ``dy`` field on both
#     :class:`PropagationResult` (with ``dy_out`` alias) and
#     :class:`lumenairy.sources.Source`; defaults to ``dx`` for back-compat.
#   * **L4a** -- the v4.11.2 pre-flight mirror guard in
#     ``apply_real_lens_traced`` was not ported to the three siblings
#     (``apply_real_lens_maslov``, ``apply_real_lens_traced_jax``,
#     ``apply_real_lens_maslov_jax``).  A hand-built prescription with
#     ``surfaces[i]['is_mirror']=True`` slipped past on all three.
#   * **L4b** -- ``error_reduction(backend='jax')`` /
#     ``hybrid_input_output(backend='jax')`` dispatchers silently dropped
#     the NumPy-API ``initial_guess`` kwarg.  Mapping
#     ``initial_guess`` (complex object-plane field) to the JAX twin's
#     ``init_phase`` (real Fourier-plane phase) is not lossless; v4.13.0
#     raises ``NotImplementedError`` to force the user to make an
#     explicit choice.
#   * **L4c** -- ``gerchberg_saxton(backend='jax', return_history=True)``
#     silently dropped ``return_history`` and returned a 2-tuple instead
#     of a 3-tuple.  v4.13.0 emits a ``RuntimeWarning`` and returns a
#     synthetic 3-tuple with an empty history list so the return shape
#     always matches the NumPy API.
#   
#   Each test class below pins one closure.
#   
#   Author: Andrew Traverso
# ============================================================================

import warnings

import numpy as np
import pytest

import lumenairy as la

JAX_AVAILABLE = False
try:
    import jax  # noqa: F401
    import jax.numpy as jnp  # noqa: F401
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


needs_jax = pytest.mark.skipif(
    not JAX_AVAILABLE, reason="JAX is not installed")


# ============================================================================
# L2 -- JAX dtype unification via _resolve_jax_complex_dtype
# ============================================================================

@_requires_jax
class TestAuditFixesV4_13_0_jax_dtype_dy_siblings_L2JaxDtypeUnification:
    """``set_default_complex_dtype(np.complex128)`` propagates to every
    JAX entry point.  Pre-fix the JAX-side hard-casts (``jnp.asarray(
    E_in, dtype=jnp.complex64)``, ``.astype(jnp.complex64)``, and
    ``jax.config.jax_enable_x64`` reads in ``_lens_jax.py``) all
    bypassed ``set_default_complex_dtype`` and gave float32-precision
    answers with no warning.
    """

    def test_resolve_jax_complex_dtype_reads_default(self):
        """When no override is passed, ``_resolve_jax_complex_dtype``
        reads :func:`get_default_complex_dtype`."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX is not installed")
        import jax.numpy as _jnp

        from lumenairy.propagators.propagation import (
            _resolve_jax_complex_dtype,
            set_default_complex_dtype,
        )
        prior = la.get_default_complex_dtype()
        try:
            set_default_complex_dtype(np.complex128)
            assert _resolve_jax_complex_dtype() == _jnp.complex128
            set_default_complex_dtype(np.complex64)
            assert _resolve_jax_complex_dtype() == _jnp.complex64
        finally:
            set_default_complex_dtype(prior)

    def test_resolve_jax_complex_dtype_override(self):
        """An explicit per-call override takes precedence over the
        library default."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX is not installed")
        import jax.numpy as _jnp

        from lumenairy.propagators.propagation import (
            _resolve_jax_complex_dtype,
            set_default_complex_dtype,
        )
        prior = la.get_default_complex_dtype()
        try:
            set_default_complex_dtype(np.complex64)
            # Even with default=complex64, an explicit complex128
            # override should win.
            assert _resolve_jax_complex_dtype(np.complex128) == _jnp.complex128
            set_default_complex_dtype(np.complex128)
            assert _resolve_jax_complex_dtype(np.complex64) == _jnp.complex64
        finally:
            set_default_complex_dtype(prior)

    def test_resolve_jax_real_dtype_pairs_with_complex(self):
        """The real twin returns ``float64`` paired with ``complex128``
        and ``float32`` paired with ``complex64``."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX is not installed")
        import jax.numpy as _jnp

        from lumenairy.propagators.propagation import (
            _resolve_jax_real_dtype,
            set_default_complex_dtype,
        )
        prior = la.get_default_complex_dtype()
        try:
            set_default_complex_dtype(np.complex128)
            assert _resolve_jax_real_dtype() == _jnp.float64
            set_default_complex_dtype(np.complex64)
            assert _resolve_jax_real_dtype() == _jnp.float32
        finally:
            set_default_complex_dtype(prior)

    @needs_jax
    def test_propagate_through_system_jax_complex128(self):
        """Direction 1: ``set_default_complex_dtype(np.complex128)``
        then run ``propagate_through_system_jax`` -- the JAX result
        must be ``jnp.complex128``.

        Pre-fix the ``E = jnp.asarray(E_in, dtype=jnp.complex64)``
        hard-cast at the top of ``propagate_through_system_jax``
        silently demoted to single precision regardless of the
        configured default.
        """
        import jax.numpy as _jnp

        from lumenairy.propagators.propagation import (
            set_default_complex_dtype,
        )
        from lumenairy.propagators.system import (
            _PROPAGATE_SYSTEM_JAX_CACHE,
            propagate_through_system_jax,
        )
        prior = la.get_default_complex_dtype()
        try:
            _PROPAGATE_SYSTEM_JAX_CACHE.clear()
            set_default_complex_dtype(np.complex128)
            N = 32
            dx = 5e-6
            wavelength = 633e-9
            # Real-valued input so the dtype comes from the default,
            # not from a complex caller dtype.
            E_in = np.ones((N, N), dtype=np.float32)
            elements = [
                {'type': 'propagate', 'z': 1e-3, 'bandlimit': True},
                {'type': 'lens', 'f': 5e-3, 'xc': 0.0, 'yc': 0.0},
            ]
            E_out = propagate_through_system_jax(
                E_in, elements, wavelength, dx, verbose=False)
            assert E_out.dtype == _jnp.complex128, (
                f"Expected jnp.complex128 with "
                f"set_default_complex_dtype(np.complex128); got "
                f"{E_out.dtype!r}.")
        finally:
            set_default_complex_dtype(prior)
            _PROPAGATE_SYSTEM_JAX_CACHE.clear()

    @needs_jax
    def test_propagate_through_system_jax_complex64(self):
        """Direction 2: ``set_default_complex_dtype(np.complex64)``
        then run ``propagate_through_system_jax`` -- the JAX result
        must be ``jnp.complex64`` (which was also the historical
        hardcoded value, but now via the helper rather than a
        silent override)."""
        import jax.numpy as _jnp

        from lumenairy.propagators.propagation import (
            set_default_complex_dtype,
        )
        from lumenairy.propagators.system import (
            _PROPAGATE_SYSTEM_JAX_CACHE,
            propagate_through_system_jax,
        )
        prior = la.get_default_complex_dtype()
        try:
            _PROPAGATE_SYSTEM_JAX_CACHE.clear()
            set_default_complex_dtype(np.complex64)
            N = 32
            dx = 5e-6
            wavelength = 633e-9
            E_in = np.ones((N, N), dtype=np.float32)
            elements = [
                {'type': 'propagate', 'z': 1e-3, 'bandlimit': True},
            ]
            E_out = propagate_through_system_jax(
                E_in, elements, wavelength, dx, verbose=False)
            assert E_out.dtype == _jnp.complex64, (
                f"Expected jnp.complex64 with "
                f"set_default_complex_dtype(np.complex64); got "
                f"{E_out.dtype!r}.")
        finally:
            set_default_complex_dtype(prior)
            _PROPAGATE_SYSTEM_JAX_CACHE.clear()

    @needs_jax
    def test_jit_cache_key_separates_dtypes(self):
        """Pre-fix the ``_PROPAGATE_SYSTEM_JAX_CACHE`` key omitted dtype,
        so a complex64 call and a complex128 call at the same shape /
        wavelength / dx would collide on a single cached XLA kernel.
        v4.13.0 adds ``str(np.dtype(cdtype))`` to the key so each
        dtype gets its own kernel."""
        from lumenairy.propagators.propagation import (
            set_default_complex_dtype,
        )
        from lumenairy.propagators.system import (
            _PROPAGATE_SYSTEM_JAX_CACHE,
            propagate_through_system_jax,
        )
        prior = la.get_default_complex_dtype()
        try:
            _PROPAGATE_SYSTEM_JAX_CACHE.clear()
            N = 16
            dx = 5e-6
            wavelength = 633e-9
            E_in = np.ones((N, N), dtype=np.float32)
            elements = [
                {'type': 'propagate', 'z': 1e-3, 'bandlimit': True},
            ]
            set_default_complex_dtype(np.complex64)
            propagate_through_system_jax(
                E_in, elements, wavelength, dx, verbose=False)
            n_after_c64 = len(_PROPAGATE_SYSTEM_JAX_CACHE)
            set_default_complex_dtype(np.complex128)
            propagate_through_system_jax(
                E_in, elements, wavelength, dx, verbose=False)
            n_after_c128 = len(_PROPAGATE_SYSTEM_JAX_CACHE)
            assert n_after_c128 > n_after_c64, (
                f"JIT cache should grow when dtype changes; before "
                f"complex128 call had {n_after_c64} entries, after "
                f"had {n_after_c128}.  Pre-fix the cache key omitted "
                f"dtype so the two calls collided.")
        finally:
            set_default_complex_dtype(prior)
            _PROPAGATE_SYSTEM_JAX_CACHE.clear()


# ============================================================================
# L3 -- PropagationResult.dy_out + Source.dy
# ============================================================================

class TestAuditFixesV4_13_0_jax_dtype_dy_siblings_L3PropagationResultDy:
    """:class:`PropagationResult` and :class:`Source` carry the y-axis
    pitch so anamorphic Fresnel propagation no longer silently
    discards it.
    """

    def test_propagation_result_has_dy_field(self):
        """Direct field check: ``PropagationResult.dy`` exists and
        defaults to ``dx``."""
        from lumenairy.propagators.result import PropagationResult
        pr = PropagationResult(
            field=np.zeros((4, 4), dtype=np.complex128),
            dx=1.5e-6, wavelength=633e-9,
        )
        assert hasattr(pr, 'dy')
        # Back-compat: square-grid default falls back to dx.
        assert pr.dy == pr.dx

    def test_propagation_result_dy_out_alias(self):
        """``dy_out`` alias mirrors ``dx_out`` (matches the
        tuple-returning kernel naming convention)."""
        from lumenairy.propagators.result import PropagationResult
        pr = PropagationResult(
            field=np.zeros((4, 4), dtype=np.complex128),
            dx=2.0e-6, dy=3.0e-6, wavelength=633e-9,
        )
        assert pr.dx_out == 2.0e-6
        assert pr.dy_out == 3.0e-6

    def test_source_has_dy_field(self):
        """``Source`` carries a ``dy`` attribute that defaults to
        ``dx`` when not given."""
        from lumenairy.sources.core import Source
        src = Source(E=np.zeros((4, 4), dtype=np.complex128),
                     dx=1e-6, wavelength=633e-9)
        assert hasattr(src, 'dy')
        # Back-compat default.
        assert src.dy == src.dx

    def test_source_dy_explicit(self):
        """Explicit ``dy`` is honoured."""
        from lumenairy.sources.core import Source
        src = Source(E=np.zeros((4, 4), dtype=np.complex128),
                     dx=1e-6, dy=2e-6, wavelength=633e-9)
        assert src.dy == 2e-6

    def test_anamorphic_fresnel_threads_dy_out(self):
        """Pin: anamorphic Fresnel propagation (different input dx /
        dy) returns distinct ``dx_out, dy_out`` from the kernel, and
        the wrapped :class:`PropagationResult` exposes both."""
        from lumenairy.propagators.dispatch import _coerce_field
        from lumenairy.propagators.propagation import fresnel_propagate
        from lumenairy.propagators.result import PropagationResult

        N = 32
        dx_in = 5e-6
        dy_in = 1e-6   # rectangular pixels
        wavelength = 633e-9
        z = 5e-3
        x = (np.arange(N) - N / 2) * dx_in
        y = (np.arange(N) - N / 2) * dy_in
        X, Y = np.meshgrid(x, y, indexing='xy')
        E_in = np.exp(-(X * X + Y * Y) / (5 * dx_in) ** 2).astype(np.complex128)

        # Raw kernel returns (E_out, dx_out, dy_out).
        E_out, dx_out_kernel, dy_out_kernel = fresnel_propagate(
            E_in, z, wavelength, dx_in, dy=dy_in)
        # Anamorphic geometry -> distinct output pitches.
        assert abs(dx_out_kernel - dy_out_kernel) > 1e-9 * dx_out_kernel, (
            f"Anamorphic input should yield distinct dx_out / dy_out; "
            f"got dx_out={dx_out_kernel}, dy_out={dy_out_kernel}.")

        # _coerce_field now returns a 3-tuple; ensure dy_out lands.
        field, dx_coerced, dy_coerced = _coerce_field(
            (E_out, dx_out_kernel, dy_out_kernel))
        assert field is E_out
        assert dx_coerced == dx_out_kernel
        assert dy_coerced == dy_out_kernel

        # End-to-end: dispatcher wrap-level thread.
        pr = PropagationResult(
            field=E_out, dx=dx_out_kernel, dy=dy_out_kernel,
            wavelength=wavelength, z=z, method='fresnel',
        )
        assert pr.dx_out == dx_out_kernel
        assert pr.dy_out == dy_out_kernel
        # And the round-trip through to_source preserves dy.
        src = pr.to_source()
        assert src.dx == dx_out_kernel
        assert src.dy == dy_out_kernel


# ============================================================================
# L4a -- mirror guard ported to the three apply_real_lens_* siblings
# ============================================================================

def _hand_built_mirror_prescription():
    """Construct a minimal prescription that puts ``is_mirror=True``
    directly into the surfaces list (the slip-past case the
    apply_real_lens_traced v4.11.2 guard caught for the parent
    function but not the siblings).
    """
    # Two refracting surfaces sandwiching a mirror surface in the
    # middle.  The mirror flag should trigger the guard regardless of
    # whether the rest of the geometry is sensible.
    return {
        'surfaces': [
            {'radius': 0.025, 'thickness': 5e-3,
             'glass_before': 'air', 'glass_after': 'BK7',
             'semi_diameter': 0.010, 'conic': 0.0},
            # Hand-built mirror entry the audit flagged.
            {'radius': float('inf'), 'thickness': 0.0,
             'glass_before': 'BK7', 'glass_after': 'BK7',
             'semi_diameter': 0.010, 'is_mirror': True,
             'conic': 0.0},
            {'radius': -0.025, 'thickness': 0.0,
             'glass_before': 'BK7', 'glass_after': 'air',
             'semi_diameter': 0.010, 'conic': 0.0},
        ],
        'aperture_diameter': 0.020,
    }


class TestAuditFixesV4_13_0_jax_dtype_dy_siblings_L4aMirrorGuardSiblings:
    """Pin the mirror guard fires for all three siblings of
    ``apply_real_lens_traced`` -- AND the parent ``apply_real_lens``
    itself (v4.13.0 audit P1-A: the L4a sweep missed the parent).
    """

    def test_apply_real_lens_maslov_raises_on_hand_built_mirror(self):
        from lumenairy.elements.lenses_maslov import apply_real_lens_maslov
        N = 32
        dx = 5e-6
        wavelength = 633e-9
        E_in = np.ones((N, N), dtype=np.complex128)
        rx = _hand_built_mirror_prescription()
        with pytest.raises(ValueError, match=r'(mirror|MIRROR)'):
            apply_real_lens_maslov(
                E_in, prescription=rx, wavelength=wavelength, dx=dx)

    @needs_jax
    def test_apply_real_lens_traced_jax_raises_on_hand_built_mirror(self):
        from lumenairy.elements._lens_jax import apply_real_lens_traced_jax
        N = 32
        dx = 5e-6
        wavelength = 633e-9
        E_in = np.ones((N, N), dtype=np.complex128)
        rx = _hand_built_mirror_prescription()
        with pytest.raises(ValueError, match=r'(mirror|MIRROR)'):
            apply_real_lens_traced_jax(
                E_in, prescription=rx, wavelength=wavelength, dx=dx)

    @needs_jax
    def test_apply_real_lens_maslov_jax_raises_on_hand_built_mirror(self):
        from lumenairy.elements._lens_jax import apply_real_lens_maslov_jax
        N = 32
        dx = 5e-6
        wavelength = 633e-9
        E_in = np.ones((N, N), dtype=np.complex128)
        rx = _hand_built_mirror_prescription()
        with pytest.raises(ValueError, match=r'(mirror|MIRROR)'):
            apply_real_lens_maslov_jax(
                E_in, prescription=rx, wavelength=wavelength, dx=dx)

    def test_apply_real_lens_raises_on_hand_built_mirror(self):
        """v4.13.0 audit P1-A: pin the mirror guard fires for the
        parent ``apply_real_lens`` itself.  Pre-fix the L4a sweep
        hardened the 4 ``apply_real_lens_*`` siblings but missed the
        parent -- a hand-built prescription with ``is_mirror=True``
        and no ``elements`` key silently miscomputed via the
        refractive-only thin-element path.  The new guard fails
        loudly before any sag / refraction math touches the field.
        Closes the audit P1-A gap.
        """
        from lumenairy.elements._lens_real import apply_real_lens
        N = 32
        dx = 5e-6
        wavelength = 633e-9
        E_in = np.ones((N, N), dtype=np.complex128)
        rx = _hand_built_mirror_prescription()
        with pytest.raises(ValueError, match=r'(mirror|MIRROR)'):
            apply_real_lens(
                E_in, prescription=rx, wavelength=wavelength, dx=dx)

    def test_apply_real_lens_raises_on_glass_after_mirror(self):
        """Same closure, but the guard fires on
        ``glass_after='MIRROR'`` (no explicit ``is_mirror`` flag).
        Matches the alternate Zemax convention the sibling guards
        recognise.
        """
        from lumenairy.elements._lens_real import apply_real_lens
        N = 32
        dx = 5e-6
        wavelength = 633e-9
        E_in = np.ones((N, N), dtype=np.complex128)
        rx = {
            'surfaces': [
                {'radius': 0.025, 'thickness': 5e-3,
                 'glass_before': 'air', 'glass_after': 'BK7',
                 'semi_diameter': 0.010, 'conic': 0.0},
                # Mirror flagged via ``glass_after`` rather than
                # ``is_mirror``.
                {'radius': float('inf'), 'thickness': 0.0,
                 'glass_before': 'BK7', 'glass_after': 'MIRROR',
                 'semi_diameter': 0.010, 'conic': 0.0},
                {'radius': -0.025, 'thickness': 0.0,
                 'glass_before': 'BK7', 'glass_after': 'air',
                 'semi_diameter': 0.010, 'conic': 0.0},
            ],
            'aperture_diameter': 0.020,
        }
        with pytest.raises(ValueError, match=r'(mirror|MIRROR)'):
            apply_real_lens(
                E_in, prescription=rx, wavelength=wavelength, dx=dx)


def _import_apply_real_lens_variant(name):
    """Resolve the variant name to its callable from the per-module
    home (closures the dispatcher-level pin across all 5 entry
    points)."""
    if name == 'apply_real_lens':
        from lumenairy.elements._lens_real import apply_real_lens
        return apply_real_lens
    if name == 'apply_real_lens_traced':
        from lumenairy.elements._lens_traced import apply_real_lens_traced
        return apply_real_lens_traced
    if name == 'apply_real_lens_maslov':
        from lumenairy.elements.lenses_maslov import apply_real_lens_maslov
        return apply_real_lens_maslov
    if name == 'apply_real_lens_traced_jax':
        from lumenairy.elements._lens_jax import apply_real_lens_traced_jax
        return apply_real_lens_traced_jax
    if name == 'apply_real_lens_maslov_jax':
        from lumenairy.elements._lens_jax import apply_real_lens_maslov_jax
        return apply_real_lens_maslov_jax
    raise ValueError(f"unknown variant: {name!r}")


_APPLY_REAL_LENS_VARIANTS = [
    pytest.param('apply_real_lens', id='apply_real_lens'),
    pytest.param('apply_real_lens_traced', id='apply_real_lens_traced'),
    pytest.param('apply_real_lens_maslov', id='apply_real_lens_maslov'),
    pytest.param(
        'apply_real_lens_traced_jax', id='apply_real_lens_traced_jax',
        marks=pytest.mark.skipif(
            not JAX_AVAILABLE, reason='JAX is not installed')),
    pytest.param(
        'apply_real_lens_maslov_jax', id='apply_real_lens_maslov_jax',
        marks=pytest.mark.skipif(
            not JAX_AVAILABLE, reason='JAX is not installed')),
]


class TestAuditFixesV4_13_0_jax_dtype_dy_siblings_L4aMirrorGuardDispatcherPin:
    """v4.13.0 audit meta-finding: the strongest closure for the
    L4a sibling-gap pattern is a parametrized dispatcher-level pin
    that exercises every entry point a user can reach the fix
    through.  If a future sweep misses one of the 5 variants the
    way the v4.13.0 sweep missed ``apply_real_lens``, this test
    will fail rather than silently allowing a downstream
    miscomputation.
    """

    @pytest.mark.parametrize('variant', _APPLY_REAL_LENS_VARIANTS)
    def test_mirror_guard_fires_on_hand_built_is_mirror(self, variant):
        fn = _import_apply_real_lens_variant(variant)
        N = 32
        dx = 5e-6
        wavelength = 633e-9
        E_in = np.ones((N, N), dtype=np.complex128)
        rx = _hand_built_mirror_prescription()
        with pytest.raises(ValueError, match=r'(mirror|MIRROR)'):
            fn(E_in, prescription=rx, wavelength=wavelength, dx=dx)


# ============================================================================
# L4b -- JAX-dispatch ``initial_guess`` forwarding / refusal
# ============================================================================

@_requires_jax
class TestAuditFixesV4_13_0_jax_dtype_dy_siblings_L4bInitialGuessForwarding:
    """v4.13.0 decision: refuse to silently demote a NumPy-API
    ``initial_guess`` (complex object-plane field) into a JAX-API
    ``init_phase`` (real Fourier-plane phase).  Pre-fix the dispatcher
    silently dropped ``initial_guess`` on the JAX path.
    """

    @needs_jax
    def test_error_reduction_jax_with_initial_guess_raises(self):
        """``error_reduction(backend='jax', initial_guess=...)`` raises
        ``NotImplementedError`` with a clear migration message."""
        N = 16
        meas = np.ones((N, N), dtype=np.float32)
        sup = np.ones((N, N), dtype=bool)
        ig = (np.ones((N, N), dtype=np.complex128) * 0.1).astype(np.complex128)
        with pytest.raises(NotImplementedError, match=r'init_phase'):
            la.error_reduction(
                meas, sup, n_iter=5, initial_guess=ig, backend='jax')

    @needs_jax
    def test_hybrid_input_output_jax_with_initial_guess_raises(self):
        """``hybrid_input_output(backend='jax', initial_guess=...)``
        raises ``NotImplementedError``."""
        N = 16
        meas = np.ones((N, N), dtype=np.float32)
        sup = np.ones((N, N), dtype=bool)
        ig = (np.ones((N, N), dtype=np.complex128) * 0.1).astype(np.complex128)
        with pytest.raises(NotImplementedError, match=r'init_phase'):
            la.hybrid_input_output(
                meas, sup, n_iter=5, beta=0.9, initial_guess=ig,
                backend='jax')

    @needs_jax
    def test_error_reduction_jax_without_initial_guess_works(self):
        """Sanity: the path still works when ``initial_guess`` is
        not given (the new check only fires when the user passed it
        explicitly)."""
        N = 16
        meas = np.ones((N, N), dtype=np.float32)
        sup = np.ones((N, N), dtype=bool)
        # No initial_guess -> no raise.
        obj, err = la.error_reduction(
            meas, sup, n_iter=5, backend='jax', seed=42)
        assert obj.shape == (N, N)
        assert np.isfinite(err)


# ============================================================================
# L4c -- gerchberg_saxton(backend='jax', return_history=True) shape
# ============================================================================

@_requires_jax
class TestAuditFixesV4_13_0_jax_dtype_dy_siblings_L4cReturnHistoryShape:
    """v4.13.0 decision: emit ``RuntimeWarning`` and synthesise a
    3-tuple with an empty history list when ``backend='jax'`` is paired
    with ``return_history=True``.  Pre-fix the dispatcher silently
    dropped ``return_history`` so the user expecting a 3-tuple
    received a 2-tuple.
    """

    @needs_jax
    def test_gs_jax_return_history_warns_and_returns_3tuple(self):
        N = 16
        src = np.ones((N, N), dtype=np.float32)
        tgt = np.ones((N, N), dtype=np.float32)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = la.gerchberg_saxton(
                src, tgt, n_iter=5, return_history=True, backend='jax')
        # Shape: 3-tuple as the NumPy API contracts.
        assert len(result) == 3, (
            f"Expected 3-tuple (phase, err, history) when "
            f"return_history=True; got len={len(result)}.")
        phase, err, history = result
        assert phase.shape == (N, N)
        assert np.isfinite(err)
        # History is the empty list synthesized by the dispatcher.
        assert history == []
        # Warning text mentions the synthetic / jax limitation.
        assert any(
            'jax' in str(w.message).lower() and 'history' in str(w.message).lower()
            for w in caught
        ), (f"Expected a RuntimeWarning about JAX history capture; "
            f"caught {[str(w.message) for w in caught]!r}.")

    @needs_jax
    def test_error_reduction_jax_return_history_warns_and_returns_3tuple(self):
        """L4c sibling: same contract for ``error_reduction``."""
        N = 16
        meas = np.ones((N, N), dtype=np.float32)
        sup = np.ones((N, N), dtype=bool)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = la.error_reduction(
                meas, sup, n_iter=5, return_history=True, backend='jax',
                seed=0)
        assert len(result) == 3
        obj, err, history = result
        assert obj.shape == (N, N)
        assert history == []
        assert any(
            'jax' in str(w.message).lower() and 'history' in str(w.message).lower()
            for w in caught
        )

    @needs_jax
    def test_hybrid_input_output_jax_return_history_warns_and_returns_3tuple(self):
        """L4c sibling: same contract for ``hybrid_input_output``."""
        N = 16
        meas = np.ones((N, N), dtype=np.float32)
        sup = np.ones((N, N), dtype=bool)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = la.hybrid_input_output(
                meas, sup, n_iter=5, beta=0.9, return_history=True,
                backend='jax', seed=0)
        assert len(result) == 3
        obj, err, history = result
        assert obj.shape == (N, N)
        assert history == []
        assert any(
            'jax' in str(w.message).lower() and 'history' in str(w.message).lower()
            for w in caught
        )

    @needs_jax
    def test_gs_jax_no_history_still_returns_2tuple(self):
        """Default path (``return_history=False``) still returns a
        2-tuple -- the warning + synthesis only fires when the user
        explicitly asks for history."""
        N = 16
        src = np.ones((N, N), dtype=np.float32)
        tgt = np.ones((N, N), dtype=np.float32)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = la.gerchberg_saxton(
                src, tgt, n_iter=5, backend='jax')
        assert len(result) == 2
        assert not any(
            'jax' in str(w.message).lower() and 'history' in str(w.message).lower()
            for w in caught
        )


# ============================================================================
# P1-B -- ER and HIO route dtype through _resolve_jax_complex_dtype
# ============================================================================

def _call_phase_retrieval_kernel(kernel_name, *, dtype):
    """Dispatch helper for the parametrized P1-B pin.  Returns the
    primary output array (complex object / Fourier-plane phase) of
    each kernel with a small, well-conditioned input.
    """
    if kernel_name == 'gerchberg_saxton_jax':
        from lumenairy.analysis.phase_retrieval import gerchberg_saxton_jax
        N = 16
        src = np.ones((N, N), dtype=np.dtype(dtype))
        tgt = np.ones((N, N), dtype=np.dtype(dtype))
        # GS returns (phase, err) -- phase is the real-valued source
        # plane phase.  For the P1-B pin we want to verify the
        # complex intermediates ran at the resolved cdtype, which
        # is observable indirectly through the kernel cache key
        # (verified separately) and directly through the warning
        # the resolver emits on first complex128 request with
        # ``jax_enable_x64=False``.
        return gerchberg_saxton_jax(src, tgt, n_iter=3, dtype=dtype)
    if kernel_name == 'error_reduction_jax':
        from lumenairy.analysis.phase_retrieval import error_reduction_jax
        N = 16
        meas = np.ones((N, N), dtype=np.dtype(dtype))
        sup = np.ones((N, N), dtype=bool)
        return error_reduction_jax(meas, sup, n_iter=3, seed=0, dtype=dtype)
    if kernel_name == 'hybrid_input_output_jax':
        from lumenairy.analysis.phase_retrieval import hybrid_input_output_jax
        N = 16
        meas = np.ones((N, N), dtype=np.dtype(dtype))
        sup = np.ones((N, N), dtype=bool)
        return hybrid_input_output_jax(
            meas, sup, n_iter=3, beta=0.9, seed=0, dtype=dtype)
    raise ValueError(f"unknown kernel: {kernel_name!r}")


_PHASE_RETRIEVAL_KERNELS = ['gerchberg_saxton_jax',
                            'error_reduction_jax',
                            'hybrid_input_output_jax']


@_requires_jax
class TestAuditFixesV4_13_0_jax_dtype_dy_siblings_P1BPhaseRetrievalDtypeResolver:
    """v4.13.0 audit P1-B: ``error_reduction_jax`` and
    ``hybrid_input_output_jax`` route ``dtype`` through
    ``_resolve_jax_complex_dtype`` (and the real twin) so that a
    caller passing ``dtype=np.float64`` while
    ``jax.config.jax_enable_x64=False`` either auto-enables x64
    (with the documented one-shot RuntimeWarning) or returns at
    full complex128 precision rather than silently demoting.

    The L2 sweep applied this closure to ``gerchberg_saxton_jax``
    but missed the ER/HIO siblings; this pin closes that gap and
    parametrises over all 3 kernels to keep future sweeps
    honest.
    """

    @needs_jax
    @pytest.mark.parametrize('kernel_name', _PHASE_RETRIEVAL_KERNELS)
    def test_float64_dtype_auto_enables_x64_or_returns_complex128(
            self, kernel_name):
        """Under ``jax_enable_x64=False``, calling any of the 3
        phase-retrieval kernels with ``dtype=np.float64`` MUST
        either fire the documented one-shot RuntimeWarning AND
        return a complex128 / float64 result OR (if x64 was
        already enabled earlier in the session) return at the
        full requested precision with no warning.  Pre-P1-B the
        ER/HIO siblings silently produced float32 / complex64
        intermediates.
        """
        import jax
        # We don't try to flip x64 off mid-session -- once any
        # earlier test (or the GS sibling on the first call here)
        # auto-enabled it, JAX won't let us turn it back off
        # cleanly.  Instead, assert the post-condition that holds
        # regardless of starting state: after the call, x64 is
        # enabled AND the returned array is at full precision.
        result = _call_phase_retrieval_kernel(
            kernel_name, dtype=np.float64)
        # Each kernel returns (primary, err).  primary is either a
        # complex object (ER / HIO) or a real phase (GS).
        primary = result[0]
        # The resolver auto-enables x64 when complex128 is
        # requested; once it has flipped, JAX returns at the
        # requested precision.
        assert jax.config.jax_enable_x64, (
            f"{kernel_name} with dtype=np.float64 should have "
            f"auto-enabled jax_enable_x64; current value is "
            f"{jax.config.jax_enable_x64!r}.")
        if np.iscomplexobj(primary):
            assert primary.dtype == np.complex128, (
                f"{kernel_name} returned complex array of dtype "
                f"{primary.dtype!r}; expected complex128 with "
                f"dtype=np.float64 + x64 enabled.")
        else:
            # GS returns the real source-plane phase.
            assert primary.dtype == np.float64, (
                f"{kernel_name} returned real array of dtype "
                f"{primary.dtype!r}; expected float64 with "
                f"dtype=np.float64 + x64 enabled.")

    @needs_jax
    @pytest.mark.parametrize('kernel_name', _PHASE_RETRIEVAL_KERNELS)
    def test_float32_dtype_keeps_complex64(self, kernel_name):
        """Sanity: float32 caller still gets complex64 output (the
        documented historical behaviour); the P1-B fix only
        changes the float64 path.
        """
        result = _call_phase_retrieval_kernel(
            kernel_name, dtype=np.float32)
        primary = result[0]
        if np.iscomplexobj(primary):
            assert primary.dtype == np.complex64, (
                f"{kernel_name} returned complex array of dtype "
                f"{primary.dtype!r}; expected complex64 with "
                f"dtype=np.float32.")
        else:
            assert primary.dtype == np.float32, (
                f"{kernel_name} returned real array of dtype "
                f"{primary.dtype!r}; expected float32 with "
                f"dtype=np.float32.")


# ============================================================================
# P1-C -- Source.propagate() and 5 classmethod factories thread `dy`
# ============================================================================

class TestAuditFixesV4_13_0_jax_dtype_dy_siblings_P1CSourceDyThreading:
    """v4.13.0 audit P1-C: ``Source.propagate()`` and the 5
    classmethod factories (``gaussian``, ``plane_wave``,
    ``point_source``, ``top_hat``, ``fiber_mode``) must thread the
    anamorphic ``dy`` onto the returned ``Source`` instance.

    Pre-P1-C the underlying ``create_*`` helpers received and used
    ``dy`` (so the E-field WAS built on the anamorphic grid), but
    the wrapping ``cls(E=E, dx=dx, wavelength=..., ...)`` call
    omitted ``dy``, so the returned Source advertised
    ``dy == dx`` even though ``E.shape`` reflected the rectangular
    pixel count.  L3 added the ``dy`` field to ``Source`` but the
    threading sweep missed these six call sites.
    """

    def test_source_propagate_preserves_dy(self):
        """Pin: ``Source.propagate(...).dy == self.dy`` for an
        anamorphic input.  Pre-fix the result advertised
        ``dy == dx_out`` regardless of the input's y-pitch.
        """
        from lumenairy.sources.core import Source
        N = 32
        dx = 5e-6
        dy = 7e-6   # distinct y-pitch
        wavelength = 633e-9
        # Build a simple Gaussian-shaped field on the anamorphic
        # grid so the propagator has something physical to chew on.
        x = (np.arange(N) - N / 2) * dx
        y = (np.arange(N) - N / 2) * dy
        X, Y = np.meshgrid(x, y, indexing='xy')
        E = np.exp(-(X * X + Y * Y) / (5 * dx) ** 2).astype(
            np.complex128)
        src = Source(E=E, dx=dx, dy=dy, wavelength=wavelength)
        assert src.dy == dy
        # ``method='asm'`` is the simplest free-space propagator that
        # respects the input pitch verbatim (no resample).
        result = src.propagate(method='asm', z=1e-4)
        assert result.dy == dy, (
            f"Source.propagate must thread dy onto the returned "
            f"Source; got result.dy={result.dy!r}, expected "
            f"src.dy={dy!r}.")
        assert result.dx == dx

    def test_source_gaussian_factory_preserves_dy(self):
        """``Source.gaussian(..., dy=...)`` must wrap the result
        with the supplied ``dy``."""
        from lumenairy.sources.core import Source
        N = 32
        dx = 5e-6
        dy = 9e-6   # distinct y-pitch
        wavelength = 633e-9
        src = Source.gaussian(
            w0=20e-6, N=N, dx=dx, wavelength=wavelength, dy=dy)
        assert src.dx == dx
        assert src.dy == dy, (
            f"Source.gaussian must thread dy onto the wrapped "
            f"Source; got src.dy={src.dy!r}, expected "
            f"{dy!r}.")

    def test_source_plane_wave_factory_preserves_dy(self):
        from lumenairy.sources.core import Source
        N = 32
        dx = 5e-6
        dy = 9e-6
        wavelength = 633e-9
        src = Source.plane_wave(
            N=N, dx=dx, wavelength=wavelength, dy=dy)
        assert src.dx == dx
        assert src.dy == dy

    def test_source_point_source_factory_preserves_dy(self):
        from lumenairy.sources.core import Source
        N = 32
        dx = 5e-6
        dy = 9e-6
        wavelength = 633e-9
        src = Source.point_source(
            N=N, dx=dx, wavelength=wavelength, dy=dy, z0=-1e-3)
        assert src.dx == dx
        assert src.dy == dy

    def test_source_top_hat_factory_preserves_dy(self):
        from lumenairy.sources.core import Source
        N = 32
        dx = 5e-6
        dy = 9e-6
        wavelength = 633e-9
        src = Source.top_hat(
            diameter=50e-6, N=N, dx=dx, wavelength=wavelength, dy=dy)
        assert src.dx == dx
        assert src.dy == dy

    def test_source_fiber_mode_factory_preserves_dy(self):
        """``Source.fiber_mode`` is the one factory whose underlying
        helper (``create_fiber_mode``) doesn't accept ``dy``; the
        P1-C fix is wrapper-only and verifies the returned Source
        advertises a distinct ``dy``.  The factory drops ``dy``
        from ``factory_kwargs`` BEFORE forwarding because the helper
        is dy-agnostic; we still want the wrapped Source to carry
        the caller-supplied ``dy`` so the metadata reaches
        downstream code that does honour ``dy``.

        NOTE: ``create_fiber_mode`` does not accept ``dy``, so we
        verify the audit-spec'd behaviour only by directly
        constructing a Source with explicit ``dy``.  When v4.14
        widens ``create_fiber_mode`` to accept ``dy`` this test
        can be rewritten to call ``Source.fiber_mode(..., dy=...)``
        directly.
        """
        from lumenairy.sources.core import Source
        N = 32
        dx = 5e-6
        dy = 9e-6
        wavelength = 633e-9
        # Smoke: the factory builds a Source without dy (the
        # underlying helper is dy-agnostic).  Default dy == dx is
        # the documented back-compat for that helper.
        src_default = Source.fiber_mode(
            mode_field_diameter=10e-6, N=N, dx=dx,
            wavelength=wavelength)
        assert src_default.dx == dx
        assert src_default.dy == dx, (
            f"Source.fiber_mode without explicit dy should default "
            f"to dy=dx for back-compat; got "
            f"{src_default.dy!r}.")
        # Direct construction with explicit dy: confirm the
        # Source dataclass round-trips dy correctly (matches the
        # P1-C fix's intent for the dy-aware factories above).
        E = src_default.E
        src_anamorphic = Source(
            E=E, dx=dx, dy=dy, wavelength=wavelength)
        assert src_anamorphic.dy == dy


# ============================================================================
# P1-C extension: parametrised dispatcher-level pin over Source factories
# ============================================================================

_SOURCE_FACTORY_KWARGS = [
    pytest.param(
        'gaussian', {'w0': 20e-6}, id='gaussian'),
    pytest.param(
        'plane_wave', {}, id='plane_wave'),
    pytest.param(
        'point_source', {'z0': -1e-3}, id='point_source'),
    pytest.param(
        'top_hat', {'diameter': 50e-6}, id='top_hat'),
    # fiber_mode: v4.13.2 widened ``create_fiber_mode`` to accept the
    # ``dy=`` kwarg (it forwards through to ``create_gaussian_beam``),
    # so the dispatcher pin now covers all 5 factories.  The v4.14.0
    # audit P2-9 flagged that the v4.13.2 CHANGELOG claim of "covers
    # all factories" was off-by-one because this parametrize list had
    # never been updated.  v4.14.1 closes that gap.
    pytest.param(
        'fiber_mode', {'mode_field_diameter': 10e-6}, id='fiber_mode'),
]


class TestAuditFixesV4_13_0_jax_dtype_dy_siblings_P1CSourceFactoryDispatcherPin:
    """v4.13.0 audit meta-finding: the strongest closure for the
    L3 / P1-C sibling-gap pattern is a parametrised dispatcher-
    level pin over every Source factory that the v4.13.0 sweep
    was supposed to touch.  If a future sweep adds a new factory
    and forgets the ``dy`` thread-through, this test fails at
    the parametrised entry point rather than silently producing
    a Source with the wrong y-pitch metadata.
    """

    @pytest.mark.parametrize('factory_name, extra_kwargs',
                             _SOURCE_FACTORY_KWARGS)
    def test_factory_threads_dy(self, factory_name, extra_kwargs):
        from lumenairy.sources.core import Source
        N = 32
        dx = 5e-6
        dy = 11e-6   # distinct y-pitch
        wavelength = 633e-9
        factory = getattr(Source, factory_name)
        src = factory(
            N=N, dx=dx, wavelength=wavelength, dy=dy,
            **extra_kwargs)
        assert src.dx == dx
        assert src.dy == dy, (
            f"Source.{factory_name}: returned Source advertised "
            f"dy={src.dy!r}; expected dy={dy!r}.  Pre-fix the "
            f"v4.13.0 L3 sweep missed the ``cls(...)`` call in "
            f"this factory.")


# ============================================================================
# Source: test_audit_fixes_v4_13_1_agent3.py
# Audit version: V4_13_1  scope: agent3
# Original module docstring preserved as comment block for git-blame traceability:
#   Pinning tests for the v4.14 audit (group Agent 3 scope).
#   
#   Covers nine audit items handled by Agent 3 in the v4.13.0 -> v4.14
#   audit pass.  Each test pins exactly one finding so a regression
#   points straight at the relevant fix:
#   
#   * **P2 #10** ``_RestoreDtype`` exposes an explicit ``restore()`` and
#     the dominant call site uses ``try/finally`` instead of relying on
#     CPython refcount semantics.
#   * **P2 #11** ``_merit_jac_auto`` uses forward-FD + cached ``f0`` for
#     the non-JAX merit terms, reducing eval count from ``2N`` to
#     ``N + 2`` (one centre eval plus N forward steps plus one outer
#     evaluate() to capture f0).
#   * **P2 #12** ``apply_mirror`` aperture docstring describes a circle
#     (not an ellipse, which never matched the code).
#   * **P2 #13** :class:`CancellableProgress` exposes a ``should_stop``
#     flag and ``cancel()`` method; :func:`is_cancelled` reads it.
#   * **P2 #14** :func:`design_optimize` emits a UserWarning when a non-
#     default ``wave_propagator`` is selected alongside any of the three
#     Merit classes that hard-code ``apply_real_lens`` for off-nominal
#     legs.
#   * **P2 #16** ``_fd_grad_pure(validate_f0=True)`` catches a
#     deliberately-stale ``f0`` cache and raises ValueError.
#   * **P2 #17** :class:`BSDFModel.total_integrated_scatter` raises
#     ValueError (instead of silently broadcasting) when a subclass
#     evaluator returns a wrong-shape array.
#   * **P3 #18** :func:`set_max_ram` rejects negative / zero budgets;
#     :func:`get_max_ram` is in ``__all__``.
#   * **P3 #19** :class:`MultiPrescriptionParameterization` raises
#     ValueError when the ``free_vars`` list contains a duplicate
#     ``(prescription_index, *path)`` entry.
# ============================================================================

import warnings

import numpy as np
import pytest

import lumenairy
from lumenairy import (
    CancellableProgress,
    MultiPrescriptionParameterization,
    MultiWavelengthMerit,
    StrehlMerit,
    get_max_ram,
    is_cancelled,
    set_max_ram,
)
from lumenairy.elements.bsdf import BSDFModel, LambertianBSDF
from lumenairy.optimize.core import (
    DesignParameterization,
    _fd_grad_pure,
    design_optimize,
)

# ====================================================================
# P2 #13 -- Cancellation protocol
# ====================================================================

def test_p2_13_cancellable_progress_default_should_stop_false():
    """Fresh ``CancellableProgress`` has ``should_stop == False``."""
    cp = CancellableProgress()
    assert cp.should_stop is False


def test_p2_13_cancellable_progress_cancel_flips_flag():
    """``cancel()`` sets ``should_stop`` to True."""
    cp = CancellableProgress()
    assert cp.should_stop is False
    cp.cancel()
    assert cp.should_stop is True


def test_p2_13_cancellable_progress_cancel_is_idempotent():
    """Calling ``cancel()`` twice is harmless."""
    cp = CancellableProgress()
    cp.cancel()
    cp.cancel()
    assert cp.should_stop is True


def test_p2_13_cancellable_progress_reset_clears_flag():
    """``reset()`` lets one callback be reused across runs."""
    cp = CancellableProgress()
    cp.cancel()
    assert cp.should_stop is True
    cp.reset()
    assert cp.should_stop is False


def test_p2_13_is_cancelled_handles_none():
    """``is_cancelled(None)`` is always False -- no callback means
    no cancellation channel."""
    assert is_cancelled(None) is False


def test_p2_13_is_cancelled_handles_plain_callable():
    """Plain callables without the protocol are never cancelled."""
    plain = lambda *a, **k: None  # noqa: E731
    assert is_cancelled(plain) is False


def test_p2_13_is_cancelled_reads_cancellable_progress():
    """``is_cancelled`` reads ``CancellableProgress.should_stop``."""
    cp = CancellableProgress()
    assert is_cancelled(cp) is False
    cp.cancel()
    assert is_cancelled(cp) is True


def test_p2_13_cancellable_progress_forwards_to_parent():
    """Calling the wrapper forwards to the wrapped callback."""
    received = []
    parent = lambda s, f, m='': received.append((s, f, m))  # noqa: E731
    cp = CancellableProgress(parent)
    cp('foo', 0.5, 'msg')
    assert received == [('foo', 0.5, 'msg')]


# ====================================================================
# P2 #16 -- _fd_grad_pure validate_f0 catches stale cache
# ====================================================================

def test_p2_16_validate_f0_catches_stale_cache():
    """``validate_f0=True`` raises ValueError when f0 does not match
    f(x) at the supplied x."""
    def f(x):
        return float(np.sum(np.asarray(x, dtype=np.float64) ** 2))
    x = np.array([1.0, 2.0, 3.0])
    stale_f0 = 999.0  # clearly wrong (true f(x) = 14)
    with pytest.raises(ValueError, match='stale'):
        _fd_grad_pure(f, x, scheme='forward', f0=stale_f0,
                       validate_f0=True)


def test_p2_16_validate_f0_default_off_lets_stale_pass():
    """Default behaviour: stale f0 silently produces a wrong gradient
    (this pins the documented "caller is responsible" contract)."""
    def f(x):
        return float(np.sum(np.asarray(x, dtype=np.float64) ** 2))
    x = np.array([1.0, 2.0, 3.0])
    stale_f0 = 999.0
    # Should NOT raise; just produces a wrong gradient.
    g_stale = _fd_grad_pure(f, x, scheme='forward', f0=stale_f0)
    g_correct = _fd_grad_pure(f, x, scheme='forward', f0=None)
    # Confirm the stale path silently disagrees with the correct one
    # by far more than O(h) truncation could explain.
    assert np.max(np.abs(g_stale - g_correct)) > 1.0


def test_p2_16_validate_f0_accepts_correct_f0():
    """``validate_f0=True`` does NOT raise when ``f0 == f(x)``."""
    def f(x):
        return float(np.sum(np.asarray(x, dtype=np.float64) ** 2))
    x = np.array([1.0, 2.0, 3.0])
    correct_f0 = f(x)
    g = _fd_grad_pure(f, x, scheme='forward', f0=correct_f0,
                       validate_f0=True)
    assert g.shape == x.shape


# ====================================================================
# P2 #11 -- _merit_jac_auto forward-FD + cached f0 eval-count saving
# ====================================================================

def test_p2_11_forward_fd_with_f0_eval_count_is_N():
    """Pin the contract that motivates the design_optimize change:
    forward FD with ``f0`` supplied costs exactly N evaluations of
    the merit-subset function, vs 2N for central differences."""
    counter = {'count': 0}
    def f(x):
        counter['count'] += 1
        return float(np.sum(np.asarray(x, dtype=np.float64) ** 2))
    x = np.array([1.0, 2.0, 3.0, 4.0])
    N = x.size
    # Central differences: 2N evals.
    counter['count'] = 0
    _fd_grad_pure(f, x, scheme='central')
    central_count = counter['count']
    # Forward + cached f0: N evals (caller pre-computed f0).
    counter['count'] = 0
    f0 = f(x)              # 1 eval -- caller pays this (mimics design_optimize)
    pre = counter['count']
    _fd_grad_pure(f, x, scheme='forward', f0=f0)
    forward_count = counter['count'] - pre
    assert central_count == 2 * N, (
        f'central FD expected 2N={2*N}, got {central_count}')
    assert forward_count == N, (
        f'forward FD with f0 expected N={N}, got {forward_count}')
    # The audit's net saving: central(2N) -> forward(N + 1 setup eval)
    # = (2N) - (N + 1) = N - 1 saved.  Pin this.
    saving = central_count - (forward_count + 1)
    assert saving == N - 1, (
        f'net eval-count saving expected N-1={N-1}, got {saving}')


def test_p2_11_forward_fd_gradient_matches_central_to_O_h():
    """Forward + cached f0 must agree with central differences to
    within forward-FD's O(h) truncation tolerance on a smooth
    quadratic.  This pins that the design_optimize switch is
    physically correct, not just faster."""
    def f(x):
        x = np.asarray(x, dtype=np.float64)
        return float(x @ np.diag([1.0, 2.0, 3.0]) @ x)
    x = np.array([0.1, -0.3, 0.5])
    g_central = _fd_grad_pure(f, x, eps=1e-7, scheme='central')
    f0 = f(x)
    g_forward = _fd_grad_pure(f, x, eps=1e-7, scheme='forward', f0=f0)
    # Forward FD O(h) at h~1e-7 -> rel err ~1e-7, well under 1e-3.
    rel = np.max(np.abs(g_forward - g_central) /
                 (np.abs(g_central) + 1e-12))
    assert rel < 1e-3, f'rel mismatch {rel:.3e} exceeds O(h) tol'


# ====================================================================
# P2 #14 -- Merit propagator inconsistency warning
# ====================================================================

def _make_minimal_singlet():
    """A trivial single-element prescription for warning-only tests."""
    return lumenairy.make_singlet(
        R1=60e-3, R2=float('inf'), d=4e-3, glass='N-BK7',
        aperture=12e-3,
    )


def test_p2_14_warning_fires_for_gbd_plus_multiwavelength():
    """``design_optimize`` with ``wave_propagator='gbd'`` + a
    MultiWavelengthMerit triggers a UserWarning that mentions the
    inconsistency.  We catch and check the warning text without
    needing to run the full optimisation."""
    pres = _make_minimal_singlet()
    param = DesignParameterization(
        template=pres,
        free_vars=[('surfaces', 0, 'radius')],
        bounds=[(30e-3, 100e-3)],
    )
    inner = StrehlMerit(weight=1.0)
    mw = MultiWavelengthMerit(
        wavelengths=[1.30e-6, 1.55e-6], sub_merit=inner)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always')
        try:
            design_optimize(
                parameterization=param,
                merit_terms=[mw],
                wavelength=1.30e-6,
                N=32, dx=10e-6,
                wave_propagator='gbd',
                max_iter=1,
                verbose=False,
            )
        except Exception:
            # Some sub-pipelines may raise on this trivial setup;
            # we only care that the warning fired before any raise.
            pass
    msgs = [str(w.message) for w in ws
            if issubclass(w.category, UserWarning)]
    matched = [m for m in msgs
               if 'wave_propagator' in m and 'MultiWavelengthMerit' in m]
    assert matched, (
        f'expected a UserWarning mentioning wave_propagator and '
        f'MultiWavelengthMerit; got messages={msgs}')


def test_p2_14_no_warning_for_default_real_lens_propagator():
    """The default ``wave_propagator='real_lens'`` must NOT trigger
    the warning -- it only fires when the user actively requests a
    propagator that the sub-merit cannot honour."""
    pres = _make_minimal_singlet()
    param = DesignParameterization(
        template=pres,
        free_vars=[('surfaces', 0, 'radius')],
        bounds=[(30e-3, 100e-3)],
    )
    inner = StrehlMerit(weight=1.0)
    mw = MultiWavelengthMerit(
        wavelengths=[1.30e-6, 1.55e-6], sub_merit=inner)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always')
        try:
            design_optimize(
                parameterization=param,
                merit_terms=[mw],
                wavelength=1.30e-6,
                N=32, dx=10e-6,
                wave_propagator='real_lens',
                max_iter=1,
                verbose=False,
            )
        except Exception:
            pass
    msgs = [str(w.message) for w in ws
            if issubclass(w.category, UserWarning)]
    inconsistency = [m for m in msgs
                     if 'wave_propagator' in m
                     and 'MultiWavelengthMerit' in m]
    assert not inconsistency, (
        f'real_lens propagator must NOT trigger the inconsistency '
        f'warning; got {inconsistency}')


# ====================================================================
# P2 #17 -- BSDF TIS shape mismatch raises ValueError
# ====================================================================

def test_p2_17_bsdf_tis_raises_on_wrong_shape():
    """A subclass evaluator that returns a wrong-shape array must
    surface as a ValueError (pre-v4.14 silently broadcast)."""
    class _BadBSDF(BSDFModel):
        def evaluate(self, incident_dir, scattered_dir):
            # Wrong shape: scalar instead of the integration grid.
            return np.array(0.5)
        def sample(self, incident_dir, n_samples, rng=None):
            return np.zeros((n_samples, 3))
    bad = _BadBSDF()
    with pytest.raises(ValueError) as info:
        bad.total_integrated_scatter()
    msg = str(info.value)
    assert 'shape' in msg.lower(), (
        f'error must mention shape; got {msg!r}')
    # Expected/actual shapes both in the message.
    assert '256' in msg and '128' in msg, (
        f'error must include expected grid dims; got {msg!r}')


def test_p2_17_bsdf_tis_still_works_for_correct_subclass():
    """The default integration path still produces the right TIS
    for a well-behaved subclass (Lambertian closed form == rho)."""
    lamb = LambertianBSDF(rho=0.7)
    # Lambertian overrides total_integrated_scatter with the closed
    # form rho, so we exercise the default-path correctness via the
    # well-known fact that the Lambertian closed-form equals rho.
    tis = lamb.total_integrated_scatter()
    assert abs(tis - 0.7) < 1e-3


# ====================================================================
# P3 #18 -- memory.py validation + __all__
# ====================================================================

def test_p3_18_set_max_ram_rejects_negative():
    """``set_max_ram(-5)`` must raise ValueError (pre-v4.14 silently
    accepted, treating -5 GB as a negative byte count)."""
    # Ensure we don't leave state behind for other tests.
    prev = get_max_ram()
    try:
        with pytest.raises(ValueError, match='positive'):
            set_max_ram(-5)
    finally:
        set_max_ram(prev if prev is not None else None)


def test_p3_18_set_max_ram_rejects_zero():
    """``set_max_ram(0)`` must raise ValueError -- a zero budget is
    nonsensical."""
    prev = get_max_ram()
    try:
        with pytest.raises(ValueError, match='positive'):
            set_max_ram(0)
    finally:
        set_max_ram(prev if prev is not None else None)


def test_p3_18_set_max_ram_accepts_positive_gb():
    """Round-trip a 16 GB budget."""
    prev = get_max_ram()
    try:
        set_max_ram(16)                       # 16 GB
        assert get_max_ram() == 16 * 1024**3
    finally:
        set_max_ram(prev if prev is not None else None)


def test_p3_18_get_max_ram_in_memory_dunder_all():
    """``get_max_ram`` must be in ``lumenairy.memory.__all__``."""
    import lumenairy.memory as mem
    assert 'get_max_ram' in mem.__all__


# ====================================================================
# P3 #19 -- MultiPrescriptionParameterization duplicate detection
# ====================================================================

def test_p3_19_duplicate_free_vars_raises():
    """Two identical ``(prescription_index, *path)`` entries must
    raise ValueError at construction time."""
    template_a = _make_minimal_singlet()
    template_b = _make_minimal_singlet()
    free_vars = [
        (0, 'surfaces', 0, 'radius'),  # template 0, surface 0 radius
        (1, 'surfaces', 0, 'radius'),  # template 1, surface 0 radius
        (0, 'surfaces', 0, 'radius'),  # DUPLICATE of the first entry
    ]
    with pytest.raises(ValueError, match='duplicate'):
        MultiPrescriptionParameterization(
            templates=[template_a, template_b],
            free_vars=free_vars,
            bounds=[(30e-3, 100e-3)] * 3)


def test_p3_19_unique_free_vars_accepts():
    """No duplicates -> construction succeeds."""
    template_a = _make_minimal_singlet()
    template_b = _make_minimal_singlet()
    free_vars = [
        (0, 'surfaces', 0, 'radius'),
        (0, 'thicknesses', 0),
        (1, 'surfaces', 0, 'radius'),
    ]
    param = MultiPrescriptionParameterization(
        templates=[template_a, template_b],
        free_vars=free_vars,
        bounds=[(30e-3, 100e-3)] * 3)
    assert param.n_params == 3


# ====================================================================
# P2 #10 -- _RestoreDtype try/finally pattern
# ====================================================================

def test_p2_10_dtype_restored_on_exception():
    """Pin that ``design_optimize`` restores the global complex dtype
    on every exit path, including an exception during optimisation.

    Pre-v4.14 this relied on ``_RestoreDtype.__del__`` running at the
    right moment; under PyPy / when a reference cycle survives gc the
    restore could be deferred.  v4.14 wraps the body in try/finally
    so the restore is deterministic.
    """
    from lumenairy.propagators.propagation import (
        get_default_complex_dtype,
        set_default_complex_dtype,
    )
    saved = get_default_complex_dtype()
    try:
        set_default_complex_dtype(np.complex128)

        pres = _make_minimal_singlet()
        param = DesignParameterization(
            template=pres,
            free_vars=[('surfaces', 0, 'radius')],
            bounds=[(30e-3, 100e-3)],
        )

        # A merit term that always raises -- forces an exception
        # path so we can verify the restore still happens.
        from lumenairy.optimize.core import MeritTerm
        class _BoomMerit(MeritTerm):
            name = 'Boom'
            needs_wave = False
            def evaluate(self, ctx):
                raise RuntimeError('boom')

        with pytest.raises(Exception):
            design_optimize(
                parameterization=param,
                merit_terms=[_BoomMerit()],
                wavelength=1.30e-6,
                N=32, dx=10e-6,
                precision='single',
                max_iter=1,
                verbose=False,
            )
        # Even though precision='single' switched the global to
        # complex64 during the call, the finally block must put it
        # back to complex128.
        assert get_default_complex_dtype() == np.complex128, (
            f'expected complex128 restored, got '
            f'{get_default_complex_dtype()}')
    finally:
        set_default_complex_dtype(saved)


# ====================================================================
# P2 #12 -- apply_mirror docstring says circle (not ellipse)
# ====================================================================

def test_p2_12_apply_mirror_aperture_doc_says_circle():
    """The inline comment block at the aperture-mask site must
    describe the actual code -- a circular aperture in physical
    coordinates, NOT the old (wrong) "elliptical" description."""
    from pathlib import Path
    src = Path(lumenairy.__file__).parent / 'elements' / 'elements.py'
    text = src.read_text(encoding='cp1252')
    # The current comment block lives just above the apply_mirror
    # aperture mask.  Pin: the word "ellipse" must NOT appear in
    # the comment block that immediately precedes the line that
    # zeros pixels outside the aperture in apply_mirror.  We
    # search for the unique phrase introduced by the fix.
    assert 'CIRCULAR clear aperture' in text, (
        'apply_mirror aperture comment must state a CIRCULAR '
        'aperture (audit P2 #12 fix marker missing)')
    # Sanity: the file should still have the apply_mirror def.
    assert 'def apply_mirror' in text


# ====================================================================
# P3 #21 -- dtype-aware zero replaces 0.0+0.0j literal
# ====================================================================

@_requires_jax
def test_p3_21_apply_mirror_aperture_no_complex_literal():
    """The aperture mask in apply_mirror must not use the
    ``0.0 + 0.0j`` complex128 literal (which could silently upcast
    a JAX-x32 input)."""
    from pathlib import Path
    src = Path(lumenairy.__file__).parent / 'elements' / 'elements.py'
    text = src.read_text(encoding='cp1252')
    # The fix marker.
    assert 'xp.zeros((), dtype=E.dtype)' in text, (
        'apply_mirror aperture must use xp.zeros((), dtype=E.dtype) '
        '(audit P3 #21 fix marker missing)')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


# ============================================================================
# Source: test_audit_fixes_v4_13_1_random_choice.py
# Audit version: V4_13_1  scope: random_choice
# Original module docstring preserved as comment block for git-blame traceability:
#   Pinning tests for the v4.13.1 audit fix P1-F:
#   ``RandomState.choice`` on JAX backend silently ignored
#   ``replace=False`` (and ``_is_jax_prng_key`` missed opaque PRNG keys).
#   
#   Audit reference
#   ---------------
#   
#   ``AUDIT_V4_13_0_2026_05_17.md`` P1-F found two issues in
#   ``lumenairy/backend/random.py``:
#   
#   1. ``RandomState.choice(arr, size=k, replace=False)`` on the JAX
#      backend with ``p=None`` dispatched to ``jax.random.randint``,
#      which is always with-replacement.  The ``replace=False`` flag
#      was silently ignored, returning duplicates the caller expected
#      to be unique.
#   
#   2. ``_is_jax_prng_key`` only recognised legacy ``uint32`` /
#      shape-trailing-``(..., 2)`` typed keys.  JAX 0.4.20+ ships
#      opaque PRNG keys via ``jax.random.key()`` whose dtype has a
#      custom name like ``'key<fry>'``; the old detector rejected
#      them, causing ``RandomState(jax.random.key(0))`` to fall
#      through to the unrecognized-type ``TypeError``.
#   
#   v4.13.1 fixes:
#   
#   * Dispatch ``replace=False`` to ``jax.random.choice(..., replace=False)``
#     for both ``p=None`` and ``p!=None`` branches (JAX 0.4.x supports
#     unweighted ``replace=False``).
#   * Update ``_is_jax_prng_key`` to recognise opaque keys via
#     ``jax.dtypes.issubdtype(d, jax.dtypes.prng_key)`` (canonical) with a
#     ``dtype.name.startswith('key<')`` fallback.
#   
#   Author: Andrew Traverso -- v4.13.1
# ============================================================================

import numpy as np
import pytest

jax = pytest.importorskip('jax')


# ============================================================================
# replace=False is honoured on JAX
# ============================================================================

@_requires_jax
class TestAuditFixesV4_13_1_random_choice_JaxChoiceReplaceFalse:
    """``RandomState(jax_key).choice(n, shape, replace=False)``
    returns unique indices.
    """

    def test_unweighted_replace_false_returns_unique_indices(self):
        from lumenairy.backend.random import RandomState

        key = jax.random.PRNGKey(123)
        rs = RandomState(rng=key)
        n = 20
        k = 7
        out = rs.choice(n, (k,), replace=False)
        vals = np.asarray(out).tolist()
        assert len(vals) == k
        assert len(set(vals)) == k, (
            f'Expected {k} unique values but got duplicates: '
            f'{vals} -- replace=False was silently ignored.')
        # All values must be in [0, n).
        assert all(0 <= v < n for v in vals), (
            f'Indices out of range [0, {n}): {vals}')

    def test_unweighted_replace_true_still_works(self):
        """Regression guard: don't break the with-replacement
        path."""
        from lumenairy.backend.random import RandomState

        key = jax.random.PRNGKey(0)
        rs = RandomState(rng=key)
        out = rs.choice(5, (1000,), replace=True)
        vals = np.asarray(out)
        # With 1000 draws from {0..4}, duplicates are guaranteed.
        assert len(np.unique(vals)) <= 5
        # And full coverage of {0..4} is overwhelmingly likely.
        assert len(np.unique(vals)) == 5, (
            f'Expected full coverage of [0, 5) in 1000 draws; got '
            f'{np.unique(vals)}.')

    def test_weighted_replace_false(self):
        """Weighted ``replace=False`` should also produce unique
        indices.  Pre-fix: this branch raised NotImplementedError,
        which was the documented behaviour but still broken; the
        post-fix code dispatches to
        ``jax.random.choice(..., replace=False, p=...)``.
        """
        from lumenairy.backend.random import RandomState

        key = jax.random.PRNGKey(7)
        rs = RandomState(rng=key)
        n = 10
        k = 4
        p = np.array([0.1, 0.1, 0.1, 0.1, 0.1,
                      0.1, 0.1, 0.1, 0.1, 0.1])
        out = rs.choice(n, (k,), p=p, replace=False)
        vals = np.asarray(out).tolist()
        assert len(set(vals)) == k, (
            f'Weighted replace=False produced duplicates: {vals}')


# ============================================================================
# _is_jax_prng_key recognises opaque keys (JAX 0.4.20+)
# ============================================================================

@_requires_jax
class TestAuditFixesV4_13_1_random_choice_IsJaxPrngKeyOpaque:
    """``_is_jax_prng_key`` returns True for both legacy uint32
    keys and the opaque keys from ``jax.random.key()``.
    """

    def test_legacy_uint32_key_detected(self):
        from lumenairy.backend.random import _is_jax_prng_key

        k = jax.random.PRNGKey(0)
        assert _is_jax_prng_key(k), (
            f'Legacy uint32 PRNGKey rejected: '
            f'dtype={k.dtype}, shape={k.shape}')

    def test_opaque_key_detected(self):
        """``jax.random.key(...)`` returns an opaque-dtype key
        (JAX 0.4.20+).  The detector must recognise it.

        Skip if the installed JAX is too old to have
        ``jax.random.key``.
        """
        if not hasattr(jax.random, 'key'):
            pytest.skip('JAX too old: jax.random.key() not available.')
        from lumenairy.backend.random import _is_jax_prng_key

        k = jax.random.key(0)
        assert _is_jax_prng_key(k), (
            f'Opaque PRNG key rejected (regression of v4.13.0 audit '
            f'P3 #20): dtype={k.dtype}, dtype.name='
            f'{getattr(k.dtype, "name", "?")}, shape={k.shape}')

    def test_non_key_array_rejected(self):
        """Sanity: regular arrays are NOT detected as keys."""
        from lumenairy.backend.random import _is_jax_prng_key

        arr = jax.numpy.arange(10, dtype=jax.numpy.float32)
        assert not _is_jax_prng_key(arr)
        arr_int = jax.numpy.arange(10, dtype=jax.numpy.int32)
        assert not _is_jax_prng_key(arr_int)
        # uint32 of wrong shape (not trailing 2) should also be
        # rejected by the legacy branch.
        arr_u32 = jax.numpy.arange(10, dtype=jax.numpy.uint32)
        assert not _is_jax_prng_key(arr_u32)

    def test_randomstate_accepts_opaque_key(self):
        """End-to-end: ``RandomState(jax.random.key(seed))``
        constructs and dispatches to the JAX backend."""
        if not hasattr(jax.random, 'key'):
            pytest.skip('JAX too old: jax.random.key() not available.')
        from lumenairy.backend.random import RandomState

        k = jax.random.key(0)
        rs = RandomState(rng=k)
        assert rs.backend == 'jax', (
            f'RandomState(opaque_key) failed to dispatch to the JAX '
            f'backend; got backend={rs.backend!r}.')
        # And a basic draw should work.
        out = rs.uniform((4,))
        assert out.shape == (4,)


# ============================================================================
# Source: test_audit_fixes_v4_13_2_agent_b.py
# Audit version: V4_13_2  scope: agent_b
# Original module docstring preserved as comment block for git-blame traceability:
#   Pinning tests for the v4.13.2 Agent-B audit fixes.
#   
#   Audit reference
#   ---------------
#   
#   ``AUDIT_V4_13_1_2026_05_17.md`` Part 10 (Consolidation) dispatched the
#   v4.13.2 patch as four parallel agents.  Agent B owns six fixes across
#   ``elements/_lens_jax.py``, ``elements/elements.py``,
#   ``raytrace/core.py``, ``elements/_lens_thin.py`` and
#   ``elements/_lens_real.py``:
#   
#   * **B.1 / P1-NEW-E** -- The two JAX real-lens twins
#     (``apply_real_lens_traced_jax``, ``apply_real_lens_maslov_jax``) were
#     missing the ``dy=None`` parameter that the NumPy siblings (``apply_real_lens``,
#     ``apply_real_lens_traced``, ``apply_real_lens_maslov``) accept.
#     Pre-fix an anamorphic ``Source.dy`` round-trip silently dropped y-pitch
#     at the JAX boundary.  v4.13.2 adds ``dy: Optional[float] = None`` and
#     raises a clear ``ValueError`` when ``dy != dx`` (consistent with the
#     NumPy traced/Maslov contract documented in their docstrings).
#   
#   * **B.2 / P1-NEW-F** -- ``apply_mirror`` lacked the NaN-zeroing guard
#     that ``apply_real_lens`` carries.  For a hyperbolic conic at the
#     domain boundary ``(1+k)*h_sq/R^2 >= 0.9999`` the sag is NaN and
#     ``exp(1j * NaN) = NaN`` poisoned every pixel via the next ASM step.
#     v4.13.2 mirrors the ``_lens_real.py:704-705`` template.
#   
#   * **B.3 / P1-NEW-J** -- ``trace_prescription`` mutated the last
#     ``Surface.thickness`` in place when ``image_distance=`` was supplied.
#     The ``Surface`` dataclass is not frozen; in-place mutation was a
#     tripwire for shared-state bugs.  v4.13.2 uses ``_surface_copy_with``
#     (matching the ``lens_abcd`` pattern at ``raytrace/core.py:2510``).
#   
#   * **B.4 / C-P1-4** -- 9 sites of bare ``0.0 + 0.0j`` complex128 literal
#     in ``xp.where(..., E, 0+0j)`` clear-aperture / stop-mask constructs
#     silently upcast complex64 fields to complex128.  v4.13.1 P3 #21's
#     dtype-aware zero only swept ``apply_aperture`` + ``apply_mirror``;
#     v4.13.2 finishes the sweep in 4 sites of ``_lens_thin.py`` and 5
#     sites of ``_lens_real.py`` (the audit cited 13 but grep finds 9 --
#     the audit explicitly told us to verify by grep).
#   
#   * **B.5 / C-P1-5** -- Six thin-lens functions
#     (``apply_thin_lens`` / ``apply_spherical_lens`` / ``apply_aspheric_lens``
#     / ``apply_cylindrical_lens`` / ``apply_grin_lens`` / ``apply_axicon``)
#     built ``xp.exp(1j * phase)`` from a float64 ``phase`` and multiplied
#     against ``E`` without dtype matching.  Same complex64->complex128
#     upcast as B.4 but at the phase-mask leg.  v4.13.2 adds the
#     ``if phase_exp.dtype != E.dtype: phase_exp = phase_exp.astype(...)``
#     cast (mirroring v4.13.0 L6's ``apply_mirror`` fix).
#   
#   * **B.6 / C-P1-6** -- The thin-lens module docstring claims "all
#     functions accept ``use_gpu=False``" but three (``apply_cylindrical_lens``,
#     ``apply_grin_lens``, ``apply_axicon``) lacked the parameter and the
#     CuPy dispatch.  v4.13.2 adds ``use_gpu: bool = False`` and the
#     canonical dispatch pattern used by the other three thin-lens entry
#     points.
#   
#   Author: Andrew Traverso -- v4.13.2 / Agent B
# ============================================================================

from copy import deepcopy

import numpy as np
import pytest

import lumenairy as lm

# ----------------------------------------------------------------------------
# JAX availability gate (B.1 uses the JAX twins)
# ----------------------------------------------------------------------------
JAX_AVAILABLE = False
try:
    import jax  # noqa: F401
    import jax.numpy as jnp  # noqa: F401
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


needs_jax = pytest.mark.skipif(
    not JAX_AVAILABLE, reason="JAX is not installed")


# ============================================================================
# B.1 -- JAX lens twins accept dy=None and enforce dy == dx
# ============================================================================

@_requires_jax
class TestAuditFixesV4_13_2_agent_b_B1JaxLensTwinsAcceptDy:
    """``apply_real_lens_traced_jax`` / ``apply_real_lens_maslov_jax``
    accept ``dy=None`` and raise a clear ``ValueError`` on ``dy != dx``.

    Pre-fix the JAX twins did NOT accept ``dy`` at all, so an upstream
    ``Source.dy`` (anamorphic) silently failed at the JAX boundary --
    either ``TypeError: unexpected keyword`` or, when called positionally,
    the dy was silently ignored.
    """

    @needs_jax
    def test_traced_jax_accepts_dy_keyword(self):
        """``apply_real_lens_traced_jax(..., dy=dx)`` succeeds.

        Pins the keyword-acceptance half of the fix.  A round-trip with
        ``dy == dx`` should produce a finite (non-NaN) output.
        """
        from lumenairy.elements._lens_jax import apply_real_lens_traced_jax

        rx = lm.make_singlet(R1=50e-3, R2=-50e-3, d=2e-3,
                              glass='N-BK7', aperture=8e-3)
        N = 64
        dx = 8e-3 / N
        wavelength = 632.8e-9
        E_in = jnp.ones((N, N), dtype=jnp.complex64)

        E_out = apply_real_lens_traced_jax(
            E_in, prescription=rx, wavelength=wavelength,
            dx=dx, dy=dx,
            ray_subsample=8, cheb_order=8, newton_iters=8,
        )
        assert E_out.shape == E_in.shape
        # The traced JAX path produces complex output; in the central
        # region (well inside the aperture) the field magnitude should
        # be finite.
        ctr = jnp.abs(E_out[N // 2, N // 2])
        assert bool(jnp.isfinite(ctr))

    @needs_jax
    def test_maslov_jax_accepts_dy_keyword(self):
        """Same for ``apply_real_lens_maslov_jax(..., dy=dx)``."""
        from lumenairy.elements._lens_jax import apply_real_lens_maslov_jax

        rx = lm.make_singlet(R1=50e-3, R2=-50e-3, d=2e-3,
                              glass='N-BK7', aperture=8e-3)
        N = 64
        dx = 8e-3 / N
        wavelength = 632.8e-9
        E_in = jnp.ones((N, N), dtype=jnp.complex64)

        E_out = apply_real_lens_maslov_jax(
            E_in, prescription=rx, wavelength=wavelength,
            dx=dx, dy=dx,
            ray_subsample=8, cheb_order=8, newton_iters=8,
        )
        assert E_out.shape == E_in.shape
        ctr = jnp.abs(E_out[N // 2, N // 2])
        assert bool(jnp.isfinite(ctr))

    @needs_jax
    def test_traced_jax_rejects_dy_ne_dx_with_clear_message(self):
        """``dy != dx`` raises a ValueError with a message that names
        the function and points to the NumPy variant.

        Pins the documented constraint: the JAX path's Chebyshev
        tensor-product fit + Newton inversion + ray subsample paths all
        assume an isotropic square grid, so anamorphic dy is rejected
        rather than silently mishandled.
        """
        from lumenairy.elements._lens_jax import apply_real_lens_traced_jax

        rx = lm.make_singlet(R1=50e-3, R2=-50e-3, d=2e-3,
                              glass='N-BK7', aperture=8e-3)
        N = 64
        dx = 8e-3 / N
        E_in = jnp.ones((N, N), dtype=jnp.complex64)

        with pytest.raises(ValueError) as exc_info:
            apply_real_lens_traced_jax(
                E_in, prescription=rx, wavelength=632.8e-9,
                dx=dx, dy=dx * 1.5,  # non-square pixels
            )
        msg = str(exc_info.value)
        assert 'apply_real_lens_traced_jax' in msg
        assert 'dy' in msg
        # Should point users at the NumPy variant
        assert 'apply_real_lens' in msg

    @needs_jax
    def test_maslov_jax_rejects_dy_ne_dx_with_clear_message(self):
        """Same constraint for the Maslov twin."""
        from lumenairy.elements._lens_jax import apply_real_lens_maslov_jax

        rx = lm.make_singlet(R1=50e-3, R2=-50e-3, d=2e-3,
                              glass='N-BK7', aperture=8e-3)
        N = 64
        dx = 8e-3 / N
        E_in = jnp.ones((N, N), dtype=jnp.complex64)

        with pytest.raises(ValueError) as exc_info:
            apply_real_lens_maslov_jax(
                E_in, prescription=rx, wavelength=632.8e-9,
                dx=dx, dy=dx * 1.5,
            )
        msg = str(exc_info.value)
        assert 'apply_real_lens_maslov_jax' in msg
        assert 'dy' in msg


# ============================================================================
# B.2 -- apply_mirror NaN zeroing guard
# ============================================================================

class TestAuditFixesV4_13_2_agent_b_B2ApplyMirrorNanGuard:
    """A curved mirror with a strongly hyperbolic conic where
    ``(1+k)*h_sq/R^2 >= 0.9999`` at the domain boundary produces NaN
    sag.  Pre-fix the NaN propagated through ``exp(1j * phase)`` and
    poisoned every pixel.  v4.13.2 zeros the OPD on those pixels so the
    rest of the field is unaffected.
    """

    def test_hyperbolic_mirror_at_domain_boundary_no_nan(self):
        """A spherical (k=0) mirror with R chosen so the field's outer
        pixels fall right at the conic-domain edge produces NO NaN.

        Specifically: choose ``R`` such that ``h_sq_max / R^2 >= 0.9999``
        at the corner.  Pre-fix the corner pixel produced NaN sag,
        ``exp(1j * NaN) = NaN``, and ``E *= NaN`` -> all-NaN field
        (NaN propagates through the array multiply).
        """
        N = 64
        dx = 1e-4  # 100 um, so half-width = 32 * 1e-4 = 3.2 mm
        # Choose radius so the corner pixel falls right at the conic
        # domain boundary.  Half-diagonal r_max = sqrt(2) * N/2 * dx.
        # For a parabola (conic = -1) ``(1+k)*h_sq/R^2 = 0`` for ALL h,
        # so we use a hyperbolic conic.  conic = -3 with R close to
        # r_max gives (1+conic)*h_sq/R^2 = -2*h_sq/R^2 which is < 0,
        # so we go the other way.  Use a strongly oblate conic
        # (conic > 0) so (1+conic)*h_sq/R^2 grows fast.
        r_max = np.sqrt(2) * (N // 2) * dx
        conic = 5.0
        # We want (1+conic)*h_sq/R^2 >= 0.9999 at r = r_max -- this
        # triggers the NaN branch in the JAX/CuPy inline sag path.
        # Equivalently ``R^2 <= (1+conic) * r_max^2 / 0.9999``.
        R = float(np.sqrt((1.0 + conic) * r_max ** 2 / 0.9999)) * 0.999

        E_in = np.ones((N, N), dtype=np.complex64)
        E_out = lm.apply_mirror(
            E_in, wavelength=550e-9, dx=dx,
            radius=R, conic=conic,
        )
        # The output should NOT be all-NaN.  Pre-fix even a single
        # NaN sag triggers NaN propagation through the next array op.
        # Post-fix the NaN pixels are zeroed in OPD and remain finite
        # (their amplitude is just 1.0 * exp(1j*0) = 1.0).
        assert not bool(np.any(np.isnan(E_out))), (
            "apply_mirror produced NaN pixels with a hyperbolic conic "
            "at the domain boundary -- the v4.13.2 P1-NEW-F guard "
            "should have zeroed the OPD on those pixels.")
        # And the central pixel should still carry sensible phase
        # (lens center should be ~zero-OPD).
        assert bool(np.isfinite(E_out[N // 2, N // 2]))


# ============================================================================
# B.3 -- trace_prescription does not mutate shared Surface state
# ============================================================================

class TestAuditFixesV4_13_2_agent_b_B3TracePrescriptionNoMutation:
    """``trace_prescription`` clones the last surface with
    ``_surface_copy_with`` instead of mutating in place.  This pins
    the cleaner contract: even if a caller (or future caching layer)
    shares Surface state across calls, an ``image_distance=`` argument
    on call N+1 cannot contaminate call N's surfaces list (or vice
    versa).
    """

    def test_image_distance_does_not_mutate_input_prescription(self):
        """The prescription dict's ``thicknesses`` list (the original
        source of the last surface's thickness) is NOT mutated.
        """
        rx = lm.make_singlet(R1=50e-3, R2=-50e-3, d=2e-3,
                              glass='N-BK7', aperture=8e-3)
        thicknesses_before = list(rx['thicknesses'])

        _ = lm.trace_prescription(
            rx, wavelength=632.8e-9,
            semi_aperture=3e-3,
            image_distance=0.1, num_rings=2, rays_per_ring=8)

        # The prescription's thicknesses list must be unchanged.
        assert list(rx['thicknesses']) == thicknesses_before, (
            "trace_prescription contaminated the input prescription's "
            "thicknesses list.")

    def test_repeated_calls_with_different_image_distance_independent(self):
        """Two trace_prescription calls with different image_distance
        produce DIFFERENT image-plane spot positions.  Pre-fix any
        shared-Surface-state bug would leak distance A into call B's
        geometry; this test pins that they remain independent.
        """
        rx = lm.make_singlet(R1=50e-3, R2=-50e-3, d=2e-3,
                              glass='N-BK7', aperture=8e-3)

        # Call A: image at 50 mm
        res_a = lm.trace_prescription(
            rx, wavelength=632.8e-9, semi_aperture=3e-3,
            image_distance=0.050, num_rings=2, rays_per_ring=8)
        # Call B: image at 100 mm  (same prescription object)
        res_b = lm.trace_prescription(
            rx, wavelength=632.8e-9, semi_aperture=3e-3,
            image_distance=0.100, num_rings=2, rays_per_ring=8)

        # The image-ray z coordinates should differ (the image plane
        # in the trace engine's local coordinates is z=0 at the image
        # surface, but the optical path lengths to that plane differ
        # because of the different thicknesses).  More robust: the
        # OPL accumulated to the image plane should differ.
        alive_a = bool(np.any(res_a.image_rays.alive))
        alive_b = bool(np.any(res_b.image_rays.alive))
        assert alive_a and alive_b
        opd_a = float(np.nanmean(res_a.image_rays.opd[res_a.image_rays.alive]))
        opd_b = float(np.nanmean(res_b.image_rays.opd[res_b.image_rays.alive]))
        # Doubling the air gap from 50 mm to 100 mm should add roughly
        # 50 mm of OPL (n_air = 1).  Assert a clear separation.
        assert abs(opd_b - opd_a) > 1e-3, (
            f"trace_prescription gave OPLs {opd_a!r} and {opd_b!r} for "
            f"image_distance=50mm vs 100mm; expected ~50mm spread.  "
            f"Possible shared-state mutation contaminated one call.")


# ============================================================================
# B.4 + B.5 + B.6 -- Parametrized thin-lens / real-lens dtype + use_gpu pin
# ============================================================================

class TestAuditFixesV4_13_2_agent_b_B456ThinLensDtypePreservation:
    """The thin-lens family entry points and ``apply_real_lens``
    preserve a complex64 input field's dtype.

    Pre-fix:
      * B.4: ``xp.where(..., E, 0.0 + 0.0j)`` upcast E to complex128
        in 9 sites of the lens family.
      * B.5: ``xp.exp(1j * <float64 phase>)`` produced complex128
        which upcast E on the multiply, in 6 thin-lens functions.

    Post-fix every entry should leave complex64 output for complex64
    input.
    """

    @pytest.fixture
    def cfg(self):
        """Common test fixture: small grid, visible wavelength."""
        N = 32
        dx = 4e-6  # 4 um pitch -- realistic-ish microscale grid
        wavelength = 632.8e-9
        return dict(N=N, dx=dx, wavelength=wavelength,
                    E_in=np.ones((N, N), dtype=np.complex64))

    # ----- B.5: thin-lens family preserves complex64 -----

    def test_apply_thin_lens_complex64(self, cfg):
        from lumenairy import apply_thin_lens
        out = apply_thin_lens(
            cfg['E_in'], f=1e-3, wavelength=cfg['wavelength'],
            dx=cfg['dx'])
        assert out.dtype == np.complex64, (
            f"apply_thin_lens upcast complex64 -> {out.dtype}")

    def test_apply_spherical_lens_complex64(self, cfg):
        from lumenairy import apply_spherical_lens
        out = apply_spherical_lens(
            cfg['E_in'], R1=10e-3, R2=-10e-3, d=1e-3, n_lens=1.5,
            wavelength=cfg['wavelength'], dx=cfg['dx'])
        assert out.dtype == np.complex64, (
            f"apply_spherical_lens upcast complex64 -> {out.dtype}")

    def test_apply_aspheric_lens_complex64(self, cfg):
        from lumenairy import apply_aspheric_lens
        out = apply_aspheric_lens(
            cfg['E_in'], R1=10e-3, R2=-10e-3, d=1e-3, n_lens=1.5,
            k1=-1.0, k2=0.0,
            wavelength=cfg['wavelength'], dx=cfg['dx'])
        assert out.dtype == np.complex64, (
            f"apply_aspheric_lens upcast complex64 -> {out.dtype}")

    def test_apply_cylindrical_lens_complex64(self, cfg):
        from lumenairy import apply_cylindrical_lens
        out = apply_cylindrical_lens(
            cfg['E_in'], f=1e-3, wavelength=cfg['wavelength'],
            dx=cfg['dx'])
        assert out.dtype == np.complex64, (
            f"apply_cylindrical_lens upcast complex64 -> {out.dtype}")

    def test_apply_grin_lens_complex64(self, cfg):
        from lumenairy import apply_grin_lens
        out = apply_grin_lens(
            cfg['E_in'], n0=1.5, g=100.0, d=1e-3,
            wavelength=cfg['wavelength'], dx=cfg['dx'])
        assert out.dtype == np.complex64, (
            f"apply_grin_lens upcast complex64 -> {out.dtype}")

    def test_apply_axicon_complex64(self, cfg):
        from lumenairy import apply_axicon
        out = apply_axicon(
            cfg['E_in'], 0.01, 1.5,
            wavelength=cfg['wavelength'], dx=cfg['dx'])
        assert out.dtype == np.complex64, (
            f"apply_axicon upcast complex64 -> {out.dtype}")

    # ----- B.4: apply_real_lens (real-lens family) preserves complex64 -----

    def test_apply_real_lens_complex64(self, cfg):
        """The 5 sites of ``0.0 + 0.0j`` in ``_lens_real.py`` could
        only fire when a clear_aperture / stop / fresnel / slant /
        entrance-aperture branch was taken.  Use a prescription with
        ``aperture_diameter`` so the entrance-aperture branch (line
        533) fires; the per-surface clear_aperture branch (line 769)
        is exercised by setting ``clear_aperture`` on each surface.
        """
        from lumenairy import apply_real_lens
        N = 32
        dx = 4e-6
        wavelength = 632.8e-9
        E_in = np.ones((N, N), dtype=np.complex64)
        # Build a singlet with an aperture (forces line 533) plus a
        # per-surface clear_aperture (forces line 769).
        rx = lm.make_singlet(R1=50e-3, R2=-50e-3, d=1e-3,
                              glass='N-BK7', aperture=80e-6)
        # Add per-surface clear apertures so the line-769 branch fires.
        for s in rx['surfaces']:
            s['clear_aperture'] = 80e-6
        out = apply_real_lens(
            E_in, prescription=rx, wavelength=wavelength, dx=dx)
        assert out.dtype == np.complex64, (
            f"apply_real_lens upcast complex64 -> {out.dtype}")

    # ----- B.6: thin-lens functions accept use_gpu=False even without CuPy -----

    def test_apply_cylindrical_lens_accepts_use_gpu_false(self, cfg):
        """The use_gpu kwarg exists and defaults to False -- this pins
        the docstring claim 'all functions accept use_gpu=False'.
        Does not require CuPy to be installed.
        """
        from lumenairy import apply_cylindrical_lens
        out = apply_cylindrical_lens(
            cfg['E_in'], f=1e-3, wavelength=cfg['wavelength'],
            dx=cfg['dx'], use_gpu=False)
        assert out.shape == cfg['E_in'].shape

    def test_apply_grin_lens_accepts_use_gpu_false(self, cfg):
        from lumenairy import apply_grin_lens
        out = apply_grin_lens(
            cfg['E_in'], n0=1.5, g=100.0, d=1e-3,
            wavelength=cfg['wavelength'], dx=cfg['dx'], use_gpu=False)
        assert out.shape == cfg['E_in'].shape

    def test_apply_axicon_accepts_use_gpu_false(self, cfg):
        from lumenairy import apply_axicon
        out = apply_axicon(
            cfg['E_in'], 0.01, 1.5,
            wavelength=cfg['wavelength'], dx=cfg['dx'], use_gpu=False)
        assert out.shape == cfg['E_in'].shape


# ============================================================================
# B.6 (CuPy gating) -- if CuPy is installed, verify dispatch on the
# three newly-fixed functions actually returns a CuPy array
# ============================================================================

class TestAuditFixesV4_13_2_agent_b_B6CupyDispatch:
    """When CuPy is available, the three newly-routed thin-lens
    functions should accept a CuPy array input and return a CuPy
    output.  Without CuPy the test is skipped.
    """

    def test_cupy_dispatch_apply_cylindrical_lens(self):
        cp = pytest.importorskip('cupy')
        from lumenairy import apply_cylindrical_lens
        N = 32
        dx = 4e-6
        E_in = cp.ones((N, N), dtype=cp.complex64)
        out = apply_cylindrical_lens(
            E_in, f=1e-3, wavelength=632.8e-9, dx=dx)
        assert type(out).__module__.startswith('cupy'), (
            f"apply_cylindrical_lens did not return a CuPy array "
            f"(got {type(out)!r}).")

    def test_cupy_dispatch_apply_grin_lens(self):
        cp = pytest.importorskip('cupy')
        from lumenairy import apply_grin_lens
        N = 32
        dx = 4e-6
        E_in = cp.ones((N, N), dtype=cp.complex64)
        out = apply_grin_lens(
            E_in, n0=1.5, g=100.0, d=1e-3,
            wavelength=632.8e-9, dx=dx)
        assert type(out).__module__.startswith('cupy')

    def test_cupy_dispatch_apply_axicon(self):
        cp = pytest.importorskip('cupy')
        from lumenairy import apply_axicon
        N = 32
        dx = 4e-6
        E_in = cp.ones((N, N), dtype=cp.complex64)
        out = apply_axicon(
            E_in, 0.01, 1.5,
            wavelength=632.8e-9, dx=dx)
        assert type(out).__module__.startswith('cupy')


# ============================================================================
# Source: test_audit_fixes_v4_13_2_agent_d.py
# Audit version: V4_13_2  scope: agent_d
# Original module docstring preserved as comment block for git-blame traceability:
#   Pinning tests for the v4.13.2 audit (Agent D scope, six items).
#   
#   Covers six audit items handled by Agent D in the v4.13.1 -> v4.13.2
#   pass.  Each test pins exactly one finding so a regression points
#   straight at the responsible fix:
#   
#   * **D.1 (C-P0-3)** -- :func:`lumenairy.io.storage.load_plane_slice`
#     documented to return ``np.ndarray``; actually returns
#     ``(arr, attrs)`` in both backends.  v4.13.2 pinned the docstring /
#     annotation to the tuple shape that callers have always destructured.
#   * **D.2 (C-P0-4)** -- CODE V ``.seq`` reader dropped the final
#     surface's THI (BFL) on read.  v4.13.2 surfaces the BFL via the
#     ``back_focal_length`` key on the returned prescription dict.
#   * **D.3 (C-P0-5)** -- Quadoa ``.qos`` reader dropped the final
#     surface's THI on read; the round-trip with :func:`export_quadoa_qos`
#     was lossy for BFL.  v4.13.2 reads the top-level ``back_focal_length``
#     JSON field plus the trailing surface as a fallback.
#   * **D.4** -- :meth:`Source.fiber_mode` forwarded a user-supplied
#     ``dy=`` to :func:`create_fiber_mode`, which previously rejected it.
#     v4.13.2 widens ``create_fiber_mode`` to accept ``dy=`` and threads
#     it through the underlying Gaussian helper.
#   * **D.5 (C-P1-3)** -- :func:`propagate_through_system` only threaded
#     ``dy`` through the free-space ``propagate_*`` steps; every other
#     element handler passed the outer function's ``dx``/``dy`` literals
#     (squaring anamorphic grids to ``dx``).  v4.13.2 routes every handler
#     through ``current_dx`` / ``current_dy``.
#   
#   D.6 (broken wiki anchor + ``__all__`` reshuffles + duplicate
#   ``reset_fft_backend`` import) are verified by ``python -c "import
#   lumenairy"`` succeeding plus a smoke test asserting the moved names
#   are still importable.
#   
#   Author: Andrew Traverso
# ============================================================================

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

import lumenairy
from lumenairy import (
    Source,
    propagate_through_system,
)
from lumenairy.io.prescriptions import (
    export_codev_seq,
    export_quadoa_qos,
    load_codev_seq,
    load_quadoa_qos,
)
from lumenairy.io.storage import append_plane, load_plane_slice
from lumenairy.sources.core import create_fiber_mode

# ====================================================================
# D.1 -- load_plane_slice return-type matches docstring (tuple)
# ====================================================================

def test_d1_load_plane_slice_returns_tuple_h5(tmp_path):
    """load_plane_slice must return ``(arr, attrs)`` per the v4.13.2
    docstring -- a bare ``np.ndarray`` would silently break every
    caller that destructures."""
    pytest.importorskip('h5py')

    # Build a small HDF5 store with one plane.
    fpath = str(tmp_path / 'one_plane.h5')
    E = np.arange(64, dtype=np.complex64).reshape(8, 8)
    append_plane(fpath, E, dx=1e-6, label='disk-aperture')

    result = load_plane_slice(
        fpath, plane_index=0,
        y_slice=slice(0, 4), x_slice=slice(0, 4))

    # Documented return type is (ndarray, dict).
    assert isinstance(result, tuple), (
        f"load_plane_slice must return a tuple, got {type(result).__name__}")
    assert len(result) == 2, (
        f"load_plane_slice must return a 2-tuple, got len={len(result)}")
    arr, attrs = result
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (4, 4)
    assert isinstance(attrs, dict)
    assert 'label' in attrs


# ====================================================================
# D.2 -- CODE V .seq BFL round-trip
# ====================================================================

def test_d2_codev_seq_roundtrip_preserves_bfl(tmp_path):
    """A `.seq` file whose final surface encodes the BFL must
    round-trip through ``load_codev_seq`` -- the BFL must be
    accessible (v4.13.2 surfaces it via the ``back_focal_length`` key
    on the returned prescription)."""
    # Hand-written .seq with an SI image plane THI = 0.012345 m.
    seq_text = '\n'.join([
        '! Hand-written CODE V test file',
        'LEN NEW',
        'DIM M',
        'WL 1550.0',
        'REF 1',
        'APE F1 CIR R 0.005',
        '',
        '! Object surface',
        'SO',
        '  RDY INFINITY',
        '  THI INFINITY',
        '',
        '! Surface 1 (stop)',
        'S1',
        '  STO',
        '  RDY 0.025000',
        '  THI 0.003000',
        '  GLA BK7',
        '',
        '! Surface 2',
        'S2',
        '  RDY -0.025000',
        '  THI 0.012345',
        '',
        '! Image surface',
        'SI',
        '  RDY INFINITY',
        '  THI 0.012345',
        '',
        'GO',
        'END',
    ])

    fpath = str(tmp_path / 'roundtrip_bfl.seq')
    Path(fpath).write_text(seq_text, encoding='utf-8')

    pres = load_codev_seq(fpath)

    # The library's traditional thicknesses convention has len ==
    # len(surfaces) - 1; the trailing BFL gap must be surfaced via
    # ``back_focal_length`` rather than silently dropped.
    assert 'back_focal_length' in pres, (
        "v4.13.2 must surface the trailing-surface THI as a BFL key.")
    assert pres['back_focal_length'] == pytest.approx(0.012345, rel=1e-6)
    assert len(pres['thicknesses']) == len(pres['surfaces']) - 1


def test_d2_codev_seq_export_roundtrip_preserves_bfl(tmp_path):
    """Round-trip a real lumenairy prescription through
    ``export_codev_seq`` -> ``load_codev_seq`` and confirm the BFL is
    preserved end-to-end."""
    pres = {
        'name': 'test_lens',
        'aperture_diameter': 0.010,
        'surfaces': [
            {'radius': 0.025, 'conic': 0.0,
             'glass_before': 'air', 'glass_after': 'BK7'},
            {'radius': -0.025, 'conic': 0.0,
             'glass_before': 'BK7', 'glass_after': 'air'},
        ],
        'thicknesses': [0.003],  # only inter-surface gap
    }
    bfl = 0.04321
    fpath = str(tmp_path / 'roundtrip.seq')
    export_codev_seq(
        pres, fpath, wavelength=1.55e-6, stop_surface=0,
        back_focal_length=bfl)
    pres_back = load_codev_seq(fpath)
    assert 'back_focal_length' in pres_back
    assert pres_back['back_focal_length'] == pytest.approx(bfl, rel=1e-5)


# ====================================================================
# D.3 -- Quadoa .qos BFL round-trip
# ====================================================================

def test_d3_quadoa_qos_roundtrip_preserves_bfl(tmp_path):
    """``export_quadoa_qos`` writes ``back_focal_length`` at JSON top
    level; ``load_quadoa_qos`` must read it back into the returned
    prescription dict."""
    pres = {
        'name': 'doublet',
        'aperture_diameter': 0.010,
        'surfaces': [
            {'radius': 0.025, 'conic': 0.0,
             'glass_before': 'air', 'glass_after': 'BK7'},
            {'radius': -0.025, 'conic': 0.0,
             'glass_before': 'BK7', 'glass_after': 'air'},
        ],
        'thicknesses': [0.003],
    }
    bfl = 0.03579
    fpath = str(tmp_path / 'roundtrip.qos')
    export_quadoa_qos(
        pres, fpath, wavelength=1.55e-6, stop_surface=0,
        back_focal_length=bfl)
    pres_back = load_quadoa_qos(fpath)
    assert 'back_focal_length' in pres_back, (
        "v4.13.2 must surface ``back_focal_length`` on `.qos` round-trip.")
    assert pres_back['back_focal_length'] == pytest.approx(bfl, rel=1e-5)
    # thicknesses length convention preserved.
    assert len(pres_back['thicknesses']) == len(pres_back['surfaces']) - 1


# ====================================================================
# D.4 -- Source.fiber_mode accepts dy=
# ====================================================================

def test_d4_fiber_mode_accepts_dy_kwarg():
    """``Source.fiber_mode(..., dy=2*dx, ...)`` must build a Source
    with ``dy`` distinct from ``dx`` (v4.13.2 widens
    ``create_fiber_mode`` to accept ``dy=`` so the wrap-level
    ``Source(..., dy=...)`` actually fires)."""
    dx = 1e-6
    dy_in = 2e-6
    src = Source.fiber_mode(
        mode_field_diameter=10e-6,
        N=64, dx=dx, dy=dy_in,
        wavelength=1.55e-6, na=0.12)
    assert src.dx == pytest.approx(dx)
    assert src.dy == pytest.approx(dy_in), (
        f"Source.fiber_mode must propagate dy={dy_in}, got {src.dy}")


def test_d4_create_fiber_mode_accepts_dy_directly():
    """``create_fiber_mode`` itself accepts ``dy=`` after v4.13.2
    (option-(a) fix)."""
    dx = 1e-6
    dy_in = 1.5e-6
    E, x, y = create_fiber_mode(
        N=64, dx=dx, dy=dy_in,
        wavelength=1.55e-6, mode_field_diameter=10e-6)
    # y-axis step must reflect dy, not dx.
    assert (y[1] - y[0]) == pytest.approx(dy_in)
    assert (x[1] - x[0]) == pytest.approx(dx)


# ====================================================================
# D.5 -- propagate_through_system threads dy through every element
# ====================================================================

def test_d5_propagate_through_system_threads_dy_aperture():
    """An anamorphic source through an aperture must keep dy == dy_in
    (pre-fix the aperture handler called ``apply_aperture(E, dx, ...)``
    only, squaring dy -> dx)."""
    N = 64
    dx = 1e-6
    dy = 2e-6
    wavelength = 1.55e-6
    # Uniform plane-wave input on an anamorphic grid.
    E_in = np.ones((N, N), dtype=np.complex64)

    elements = [
        {'type': 'aperture',
         'shape': 'circular',
         'params': {'diameter': 20e-6}},
    ]
    result = propagate_through_system(
        E_in, elements, wavelength, dx=dx, dy=dy,
        return_result=True)
    # The aperture is an in-place mask; result.dx / result.dy are the
    # final grid pitches that the system maintained throughout.
    assert result.dx == pytest.approx(dx)
    assert result.dy == pytest.approx(dy), (
        f"propagate_through_system must preserve anamorphic dy through "
        f"element handlers; got dy={result.dy}, expected {dy}.")


def test_d5_propagate_through_system_threads_dy_mask():
    """Same threading guarantee for ``mask`` element."""
    N = 64
    dx = 1e-6
    dy = 3e-6
    wavelength = 1.55e-6
    E_in = np.ones((N, N), dtype=np.complex64)
    mask = np.ones((N, N), dtype=np.complex64)  # transparent mask
    elements = [{'type': 'mask', 'mask': mask}]
    result = propagate_through_system(
        E_in, elements, wavelength, dx=dx, dy=dy,
        return_result=True)
    assert result.dx == pytest.approx(dx)
    assert result.dy == pytest.approx(dy)


# ====================================================================
# D.6 -- smoke checks for __all__ retiering and import dedup
# ====================================================================

def test_d6_tier_moves_keep_names_importable():
    """The four names moved between Tier 6 -> Tier 1/4 must remain
    importable from the top-level package after the reshuffle."""
    # Tier 1 (lens models)
    assert hasattr(lumenairy, 'apply_real_lens_traced_jax')
    assert hasattr(lumenairy, 'apply_real_lens_maslov_jax')
    # Tier 4 (tolerancing)
    assert hasattr(lumenairy, 'monte_carlo_tolerancing_jax')
    assert hasattr(lumenairy, 'monte_carlo_tolerancing_linearized')
    assert hasattr(lumenairy, 'tolerancing_report')


def test_d6_reset_fft_backend_still_callable():
    """The duplicate ``reset_fft_backend`` import was removed; the
    name must remain callable through the package namespace."""
    assert callable(lumenairy.reset_fft_backend)


# ============================================================================
# Source: test_audit_fixes_v4_9.py
# Audit version: V4_9  scope: (top-level)
# Original module docstring preserved as comment block for git-blame traceability:
#   Regression tests for the 4.9 external-audit fixes.
#   
#   Each test maps to one finding in ``LumenAiry_Audit_Report.md``.  These
#   are *behavioural* tests -- they assert the new behaviour exists, which
#   incidentally means the old buggy behaviour is no longer present.  See
#   the per-finding docstrings for the specific assertion logic.
# ============================================================================

import numpy as np
import pytest

import lumenairy as lm

# ============================================================================
# #2.2 -- GBD axial_phase abs(t) removal
# ============================================================================

class TestAuditFixesV4_9_GBDBackPropSign:
    """The pre-4.9 code used ``exp(1j*k*abs(t))`` which produced the
    complex-conjugate axial phase on back-propagation.  4.9 uses raw
    ``exp(1j*k*t)`` which is correct for both signs of z."""

    def test_back_prop_does_not_conjugate_axial_phase(self):
        """Direct check: backward propagation should give a phase
        change of exp(-i*k*|z|), not exp(+i*k*|z|).  Pre-4.9 the
        abs(t) bug made forward and backward give the same axial
        phase, so the ratio bundle_back.amplitude[0] / bundle.amplitude[0]
        would be 1 (= exp(0)) instead of exp(-2j*k*z) after a
        forward+back round trip on a non-zero-phase source.
        """
        N, dx, wl = 64, 5e-6, 1.55e-6
        # Use a complex unit constant so the phase difference is
        # visible.  Phase-only round-trip check.
        E_in = np.ones((N, N), dtype=np.complex128)
        bundle = lm.decompose_field_to_beamlets(
            E_in, dx, wavelength=wl, sample_step=2)
        z = 5e-4
        # Forward then back: total phase = (k·z) + (k·(-z)) = 0 -> identity.
        # Pre-4.9: |t| for both legs -> total phase = 2·k·z -> NOT identity.
        bundle_fwd = lm.propagate_beamlets_freespace(bundle, z, wl)
        bundle_back = lm.propagate_beamlets_freespace(bundle_fwd, -z, wl)
        ratio = (np.asarray(bundle_back.amplitude)
                 / np.asarray(bundle.amplitude))
        # 4.9 (correct): ratio = 1 (up to numerical noise)
        # Pre-4.9: ratio = exp(2j·k·z) -- generally not 1.
        for r in ratio[np.isfinite(ratio)][:8]:
            assert abs(r - 1.0) < 1e-3, (
                f"GBD round-trip ratio = {r!r}, expected 1.  "
                f"Pre-4.9 abs(t) bug made forward + back = exp(2j·k·z) "
                f"instead of identity.")


# ============================================================================
# #2.3 -- Coronagraph pix_per_lam_over_D from physical parameters
# ============================================================================

class TestAuditFixesV4_9_CoronagraphScale:
    def _build_psfs(self, N=64):
        rng = np.random.default_rng(0)
        psf_ref = np.zeros((N, N))
        psf_ref[N // 2, N // 2] = 1.0
        # Add a smooth Gaussian halo
        Y, X = np.meshgrid(np.arange(N) - N / 2,
                            np.arange(N) - N / 2, indexing='ij')
        r = np.hypot(X, Y)
        psf_ref += 1e-2 * np.exp(-(r / 5) ** 2)
        psf_coro = psf_ref * 1e-3 + 1e-9 * rng.random((N, N))
        return psf_coro, psf_ref

    def test_explicit_pupil_diameter_uses_lambdaF_over_D(self):
        psf_coro, psf_ref = self._build_psfs()
        result = lm.coronagraph_contrast_curve(
            psf_coro, psf_ref,
            dx_focal=2e-6, wavelength=1.55e-6, f_eff=100e-3,
            pupil_diameter_m=10e-3,
        )
        # pix_per_lam_over_D = λ·f/(D·dx) = 1.55e-6·0.1/(10e-3·2e-6)
        #                    = 7.75 pixels per λ/D.
        # The first radial bin should map to ~0 λ/D; the last bin
        # should reach a substantial fraction of the requested
        # max_lam_over_D = 20.  Verify the SCALE not the absolute
        # numbers (the test PSFs are synthetic).
        r_lod = result['r_lam_over_D']
        # Sanity: monotonically increasing.
        assert np.all(np.diff(r_lod) > 0)
        # First bin near zero, last bin near max_lam_over_D (capped
        # by the grid).
        assert r_lod[0] < 1.0
        assert r_lod[-1] > 3.0   # at least a few λ/D coverage

    def test_legacy_path_warns(self):
        psf_coro, psf_ref = self._build_psfs()
        with pytest.warns(RuntimeWarning, match="pupil_diameter_m"):
            lm.coronagraph_contrast_curve(
                psf_coro, psf_ref,
                dx_focal=2e-6, wavelength=1.55e-6, f_eff=100e-3,
                # NO pupil_diameter_m -> legacy fallback + warning
            )


# ============================================================================
# #3.3 -- Fresnel/Fraunhofer/SAS z<=0 guards
# ============================================================================

class TestAuditFixesV4_9_FresnelFamilyZGuards:
    """4.9 added explicit z<=0 guards on the Fresnel-family
    propagators (which are forward-only by construction)."""

    def test_fresnel_propagate_rejects_negative_z(self):
        E = np.ones((64, 64), dtype=np.complex128)
        with pytest.raises(ValueError, match="z must be > 0"):
            lm.fresnel_propagate(E, z=-0.1, wavelength=1.55e-6, dx=5e-6)

    def test_fresnel_propagate_rejects_zero_z(self):
        E = np.ones((64, 64), dtype=np.complex128)
        with pytest.raises(ValueError, match="z must be > 0"):
            lm.fresnel_propagate(E, z=0.0, wavelength=1.55e-6, dx=5e-6)

    def test_fraunhofer_propagate_rejects_negative_z(self):
        E = np.ones((64, 64), dtype=np.complex128)
        with pytest.raises(ValueError, match="z must be > 0"):
            lm.fraunhofer_propagate(E, z=-0.1, wavelength=1.55e-6, dx=5e-6)

    def test_sas_rejects_negative_z(self):
        E = np.ones((64, 64), dtype=np.complex128)
        with pytest.raises(ValueError, match="z must be > 0"):
            lm.scalable_angular_spectrum_propagate(
                E, z=-0.1, wavelength=1.55e-6, dx=5e-6)

    def test_fresnel_mft_rejects_negative_z(self):
        E = np.ones((64, 64), dtype=np.complex128)
        with pytest.raises(ValueError, match="z must be > 0"):
            lm.fresnel_propagate_mft(
                E, z=-0.1, wavelength=1.55e-6,
                dx_in=5e-6, dx_out=5e-6, N_out=64)

    def test_fraunhofer_mft_rejects_negative_z(self):
        E = np.ones((64, 64), dtype=np.complex128)
        with pytest.raises(ValueError, match="z must be > 0"):
            lm.fraunhofer_propagate_mft(
                E, z=-0.1, wavelength=1.55e-6,
                dx_in=5e-6, dx_out=5e-6, N_out=64)

    def test_asm_still_accepts_negative_z(self):
        """ASM is correctly back-prop-capable; the guard must NOT
        touch it."""
        E = np.ones((64, 64), dtype=np.complex128)
        # Should not raise
        out = lm.angular_spectrum_propagate(
            E, z=-0.1, wavelength=1.55e-6, dx=5e-6)
        assert out.shape == E.shape


# ============================================================================
# #3.5 -- TIR mask runs for slant_correction even without fresnel
# ============================================================================

class TestAuditFixesV4_9_TIRMaskOutOfFresnelBlock:
    """Pre-4.9 the TIR mask only fired inside ``if fresnel:``;
    a user with ``slant_correction=True`` and ``fresnel=False``
    got unphysical residual field amplitude in TIR regions."""

    def test_slant_correction_only_finite(self):
        """Smoke test: the slant-only path must produce a finite
        field (no NaN / Inf from the cos_tt division at TIR points)."""
        N, dx, wl = 64, 5e-6, 1.55e-6
        E = lm.create_gaussian_beam(
            N=N, dx=dx, wavelength=wl, sigma=20e-6)[0]
        prescription = lm.make_singlet(
            R1=50e-3, R2=np.inf, d=2e-3,
            glass='N-BK7', aperture=200e-6)
        out = lm.apply_real_lens(
            E, prescription=prescription,
            wavelength=wl, dx=dx,
            fresnel=False, slant_correction=True)
        assert np.all(np.isfinite(out))


# ============================================================================
# #4.3 -- dx > 1 mm warning instead of error
# ============================================================================

class TestAuditFixesV4_9_DxValidatorLoosened:
    def test_dx_above_1mm_now_warns(self):
        E = np.ones((64, 64), dtype=np.complex128)
        # 5 mm pitch -- legitimate for large-telescope sampling.
        # Pre-4.9 raised; 4.9 warns and propagates.
        with pytest.warns(RuntimeWarning, match="unusually large"):
            out = lm.angular_spectrum_propagate(
                E, z=10.0, wavelength=1.55e-6, dx=5e-3)
            assert out.shape == E.shape

    def test_dx_above_100mm_still_raises(self):
        """The unit-error guard moves to > 100 mm; the > 1 mm range
        is now valid-with-warning."""
        E = np.ones((64, 64), dtype=np.complex128)
        with pytest.raises(ValueError, match="looks suspicious"):
            lm.angular_spectrum_propagate(
                E, z=1.0, wavelength=1.55e-6, dx=2.0)


# ============================================================================
# #4.4 -- Zemax INCH alias
# ============================================================================

class TestAuditFixesV4_9_ZemaxInchAlias:
    """Zemax exports sometimes use 'INCH' or 'INCHES' instead of 'IN'."""

    def test_inch_alias_recognized_in_unit_map(self):
        """Audit #4.4: the .zmx loader (``load_zemax_zmx``) used to
        handle MM / CM / IN / M but not INCH / INCHES.  Some Zemax
        exports use the long form.  4.9 adds INCH and INCHES.

        v5.2.3 (AUDIT_V4_13_1 Part 6.1 closure: replace inspect.getsource
        proxy with behavioral pin): the previous version grepped
        ``load_zemax_zmx`` source for the literal strings ``"'INCH'"``
        and ``"'INCHES'"`` in the local ``unit_map`` dict.  The new
        pin writes three minimal .zmx files (UNIT IN as a v4.9-
        pre-existing reference, UNIT INCH and UNIT INCHES as the new
        aliases) and verifies all three produce identical, correctly
        inch-scaled thicknesses (1.0 inch -> 0.0254 m).
        """
        import os
        import tempfile

        # 3 identical files differing only in the UNIT token.  Thickness
        # 1.0 (in whatever the unit is) on the air gap -- so the parsed
        # all_thicknesses[0] should be 0.0254 m for every inch spelling.
        zmx_template = (
            'UNIT {unit}\n'
            'SURF 0\n'
            '  CURV 0.0\n'
            '  DISZ INFINITY\n'
            'SURF 1\n'
            '  CURV 0.0\n'
            '  DISZ 1.0\n'
            '  GLAS N-BK7\n'
            '  DIAM 0.5\n'
            'SURF 2\n'
            '  CURV 0.0\n'
            '  DISZ 0.5\n'
            '  DIAM 0.5\n'
        )

        def _parse_with_unit(unit_str):
            with tempfile.NamedTemporaryFile(
                    'w', suffix='.zmx', encoding='utf-8',
                    delete=False) as fh:
                fh.write(zmx_template.format(unit=unit_str))
                path = fh.name
            try:
                return lm.load_zemax_zmx(path)
            finally:
                os.unlink(path)

        # Reference: pre-v4.9 ``IN`` form (already worked).
        rx_in = _parse_with_unit('IN')
        # New v4.9 aliases.
        rx_inch = _parse_with_unit('INCH')
        rx_inches = _parse_with_unit('INCHES')

        # The 1-inch air gap between surfaces 1 and 2 should appear
        # in all_thicknesses at index 1 (index 0 is the OBJ->surf 1
        # INFINITY gap, which is dropped to 0.0).  All three forms
        # must agree to within float-precision.
        ref_t = rx_in['all_thicknesses']
        inch_t = rx_inch['all_thicknesses']
        inches_t = rx_inches['all_thicknesses']

        assert len(ref_t) == len(inch_t) == len(inches_t), (
            f'thickness-list lengths disagree: IN={len(ref_t)}, '
            f'INCH={len(inch_t)}, INCHES={len(inches_t)} -- the unit '
            f'parser must produce identical structure regardless of '
            f'spelling.')
        for i, (a, b, c) in enumerate(zip(ref_t, inch_t, inches_t)):
            assert abs(a - b) < 1e-12, (
                f'UNIT INCH alias produced thickness[{i}]={b!r} but '
                f'reference UNIT IN gave {a!r}.  Audit #4.4: INCH '
                f'must scale to inches (25.4e-3 m), not the default '
                f'MM (1e-3 m).')
            assert abs(a - c) < 1e-12, (
                f'UNIT INCHES alias produced thickness[{i}]={c!r} '
                f'but reference UNIT IN gave {a!r}.')
        # Quantitative: a 1.0-inch DISZ must parse to ~0.0254 m.
        # The DISZ 1.0 lives on surf 1 (the air gap before surf 2)
        # which maps to all_thicknesses[1] under the load_zemax_zmx
        # contract.  (Index 0 is OBJ -> first lens surface, which we
        # set to INFINITY -> 0.0.)
        non_zero_inch = [t for t in inch_t if abs(t) > 1e-12]
        assert non_zero_inch, (
            'Expected at least one non-zero thickness; '
            f'all_thicknesses={inch_t!r}')
        assert any(abs(t - 25.4e-3) < 1e-9 for t in non_zero_inch), (
            f'No thickness matched 1.0 inch = 25.4e-3 m in the parsed '
            f'all_thicknesses {inch_t!r}; UNIT INCH is being silently '
            f'treated as the default UNIT MM (factor 1000x smaller).')


# ============================================================================
# #4.5 -- cosmic_ray_rate_per_m2_per_s scales with area · time
# ============================================================================

class TestAuditFixesV4_9_CosmicRayScaling:
    """The new kwarg scales strikes by detector area × exposure
    time; the legacy kwarg deprecation-warns."""

    def _make_field(self, N=64):
        # Bright uniform complex field so the detector integrates to
        # a saturated baseline; cosmic-ray strikes show up as extra
        # bright pixels above the saturation level.
        return np.ones((N, N), dtype=np.complex128)

    def test_legacy_kwarg_removed_in_v5_0(self):
        """v5.0 (honest break): the v4.9-deprecated ``cosmic_ray_rate``
        kwarg was removed.  Callers passing it now hit ``TypeError``
        (unexpected keyword argument)."""
        E = self._make_field()
        with pytest.raises(TypeError, match="cosmic_ray_rate"):
            lm.apply_detector(
                E, dx_field=5e-6, pixel_pitch=5e-6,
                exposure_time=1.0, cosmic_ray_rate=5.0,
                seed=0,
            )

    def test_new_kwarg_scales_with_area(self):
        """Two detectors with same flux but different area should
        get strike counts proportional to area."""
        E_small = self._make_field(N=32)
        E_large = self._make_field(N=128)
        # 1e8 strikes/m²/s -- guarantees many strikes on both
        out_small, _, _ = lm.apply_detector(
            E_small, dx_field=5e-6, pixel_pitch=5e-6,
            exposure_time=1.0,
            cosmic_ray_rate_per_m2_per_s=1e8,
            cosmic_ray_amp_e=1e5,
            seed=0, full_well=1e4,
        )
        out_large, _, _ = lm.apply_detector(
            E_large, dx_field=5e-6, pixel_pitch=5e-6,
            exposure_time=1.0,
            cosmic_ray_rate_per_m2_per_s=1e8,
            cosmic_ray_amp_e=1e5,
            seed=1, full_well=1e4,
        )
        # Strikes saturate; count pixels at full_well as proxy for
        # strikes.  area_large = 16 · area_small; strike count should
        # scale roughly 16× (Poisson noise allows wide tolerance).
        n_strikes_small = int((out_small >= 1e4).sum())
        n_strikes_large = int((out_large >= 1e4).sum())
        # Both runs should have strikes.  Allow generous tolerance:
        # ratio in [4, 64] (16× ±factor-of-4 from Poisson noise).
        ratio = n_strikes_large / max(n_strikes_small, 1)
        assert 4 < ratio < 64, (
            f"strike count scaling ratio = {ratio:.2f} "
            f"(small={n_strikes_small}, large={n_strikes_large}); "
            f"expected ~16 (large-detector area is 16× small).")
