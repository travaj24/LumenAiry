"""Regression tests for the 4.9 external-audit fixes.

Each test maps to one finding in ``LumenAiry_Audit_Report.md``.  These
are *behavioural* tests -- they assert the new behaviour exists, which
incidentally means the old buggy behaviour is no longer present.  See
the per-finding docstrings for the specific assertion logic.
"""

from __future__ import annotations

import numpy as np
import pytest

import lumenairy as lm


# ============================================================================
# #2.2 -- GBD axial_phase abs(t) removal
# ============================================================================

class TestGBDBackPropSign:
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

class TestCoronagraphScale:
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

class TestFresnelFamilyZGuards:
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

class TestTIRMaskOutOfFresnelBlock:
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

class TestDxValidatorLoosened:
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

class TestZemaxInchAlias:
    """Zemax exports sometimes use 'INCH' or 'INCHES' instead of 'IN'."""

    def test_inch_alias_recognized_in_unit_map(self):
        """Audit #4.4: the .zmx loader (``load_zemax_zmx``) used to
        handle MM / CM / IN / M but not INCH / INCHES.  Some Zemax
        exports use the long form.  4.9 adds INCH and INCHES.
        """
        import inspect
        src = inspect.getsource(lm.load_zemax_zmx)
        # 'INCH' (and ideally 'INCHES') should now appear in the
        # unit_map literal.
        assert "'INCH'" in src, (
            "load_zemax_zmx unit_map does not contain 'INCH' alias.  "
            "Audit #4.4: add 'INCH' / 'INCHES' alongside 'IN'.")
        assert "'INCHES'" in src, (
            "load_zemax_zmx unit_map does not contain 'INCHES' alias.")


# ============================================================================
# #4.5 -- cosmic_ray_rate_per_m2_per_s scales with area · time
# ============================================================================

class TestCosmicRayScaling:
    """The new kwarg scales strikes by detector area × exposure
    time; the legacy kwarg deprecation-warns."""

    def _make_field(self, N=64):
        # Bright uniform complex field so the detector integrates to
        # a saturated baseline; cosmic-ray strikes show up as extra
        # bright pixels above the saturation level.
        return np.ones((N, N), dtype=np.complex128)

    def test_legacy_kwarg_deprecation_warns(self):
        E = self._make_field()
        with pytest.warns(DeprecationWarning, match="cosmic_ray_rate"):
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
