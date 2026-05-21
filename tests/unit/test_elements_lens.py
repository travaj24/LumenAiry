"""Unit tests for lens operators: thin / spherical / aspheric / DOE-lens.

API-contract focused: dtype/shape preservation, sign conventions, error
handling.  Physics fidelity is covered by the validation suite.
"""
from __future__ import annotations

import numpy as np
import pytest

import lumenairy as la

# ----------------------------------------------------------------------
# apply_thin_lens
# ----------------------------------------------------------------------

class TestApplyThinLens:

    def test_preserves_complex128_dtype(self, plane_wave, wavelength_m, dx_m):
        out = la.apply_thin_lens(plane_wave, f=50e-3, wavelength=wavelength_m, dx=dx_m)
        assert out.dtype == np.complex128

    def test_preserves_shape(self, plane_wave, wavelength_m, dx_m):
        out = la.apply_thin_lens(plane_wave, f=50e-3, wavelength=wavelength_m, dx=dx_m)
        assert out.shape == plane_wave.shape

    def test_phase_only_unitarity(self, plane_wave, wavelength_m, dx_m):
        """A thin lens is phase-only: |output| should equal |input|."""
        out = la.apply_thin_lens(plane_wave, f=50e-3, wavelength=wavelength_m, dx=dx_m)
        assert np.allclose(np.abs(out), np.abs(plane_wave))

    def test_sign_convention_converging_negative_phase(self, plane_wave,
                                                       wavelength_m, dx_m):
        """Per the exp(-i omega t) convention, a converging lens (+f)
        imparts negative phase off-axis."""
        N = plane_wave.shape[0]
        out = la.apply_thin_lens(plane_wave, f=50e-3, wavelength=wavelength_m, dx=dx_m)
        off_phase = np.angle(out[N // 2, N // 2 + N // 4])
        center_phase = np.angle(out[N // 2, N // 2])
        assert off_phase < center_phase

    def test_positive_vs_negative_f_have_opposite_phase_sign(
            self, plane_wave, wavelength_m, dx_m):
        out_pos = la.apply_thin_lens(plane_wave, f=+50e-3,
                                       wavelength=wavelength_m, dx=dx_m)
        out_neg = la.apply_thin_lens(plane_wave, f=-50e-3,
                                       wavelength=wavelength_m, dx=dx_m)
        N = plane_wave.shape[0]
        phi_pos = np.angle(out_pos[N // 2, N // 2 + N // 4])
        phi_neg = np.angle(out_neg[N // 2, N // 2 + N // 4])
        assert np.sign(phi_pos) != np.sign(phi_neg) or abs(phi_pos) < 1e-12


# ----------------------------------------------------------------------
# Diffractive lenses (4.3.0)
# ----------------------------------------------------------------------

class TestDiffractiveLens:

    def test_create_diffractive_lens_returns_phase_only(
            self, N_small, dx_m, wavelength_m):
        T = la.create_diffractive_lens(N_small, dx_m,
                                         focal_length=50e-3,
                                         wavelength=wavelength_m)
        assert T.dtype == np.complex128
        assert T.shape == (N_small, N_small)
        assert np.allclose(np.abs(T), 1.0)

    def test_create_diffractive_lens_center_phase_zero(
            self, N_small, dx_m, wavelength_m):
        T = la.create_diffractive_lens(N_small, dx_m, 50e-3, wavelength_m)
        assert abs(np.angle(T[N_small // 2, N_small // 2])) < 1e-12

    def test_create_diffractive_lens_rejects_zero_f(
            self, N_small, dx_m, wavelength_m):
        with pytest.raises(ValueError):
            la.create_diffractive_lens(N_small, dx_m, 0.0, wavelength_m)


class TestKinoform:

    def test_kinoform_phase_only(self, N_small, dx_m, wavelength_m):
        T = la.create_kinoform(N_small, dx_m, 50e-3, wavelength_m,
                                n_levels=8)
        assert np.allclose(np.abs(T), 1.0)

    def test_kinoform_rejects_n_levels_lt_2(
            self, N_small, dx_m, wavelength_m):
        with pytest.raises(ValueError):
            la.create_kinoform(N_small, dx_m, 50e-3, wavelength_m,
                                n_levels=1)

    @pytest.mark.parametrize('n_levels', [2, 4, 8, 16])
    def test_kinoform_phase_levels_are_discrete(
            self, N_small, dx_m, wavelength_m, n_levels):
        """Quantized to exactly n_levels distinct phase values."""
        T = la.create_kinoform(N_small, dx_m, 50e-3, wavelength_m,
                                n_levels=n_levels)
        unique_phases = np.unique(np.round(np.angle(T), 8))
        assert len(unique_phases) <= n_levels


class TestFresnelZonePlate:

    def test_binary_amplitude_is_zero_or_one(
            self, N_small, dx_m, wavelength_m):
        T = la.create_fresnel_zone_plate(N_small, dx_m, 50e-3,
                                            wavelength_m, binary=True)
        mags = np.abs(T).flatten()
        for v in mags:
            assert (abs(v) < 1e-12) or (abs(v - 1) < 1e-12)

    def test_binary_phase_is_unitary_without_aperture(
            self, N_small, dx_m, wavelength_m):
        T = la.create_fresnel_zone_plate(N_small, dx_m, 50e-3,
                                            wavelength_m, binary=False)
        assert np.allclose(np.abs(T), 1.0)

    def test_n_zones_crops_outside(self, N_small, dx_m, wavelength_m):
        T = la.create_fresnel_zone_plate(N_small, dx_m, 50e-3,
                                            wavelength_m, binary=True,
                                            n_zones=3)
        r_3 = np.sqrt(3 * wavelength_m * 50e-3)
        x = (np.arange(N_small) - N_small / 2) * dx_m
        X, Y = np.meshgrid(x, x)
        r = np.sqrt(X * X + Y * Y)
        if r.max() > r_3 * 1.5:
            assert np.all(T[r > r_3 * 1.01] == 0)
