"""Unit tests for beam-analysis helpers (Strehl, centroid, power, PSF).

Contract focus: dtype/shape, mathematical identities (Strehl(E, E) = 1,
centroid of a symmetric beam is at origin, power scales as |E|^2), and
sensible defaults.
"""
from __future__ import annotations

import numpy as np
import pytest

import lumenairy as la


# ----------------------------------------------------------------------
# Strehl ratio
# ----------------------------------------------------------------------

class TestStrehlRatio:

    def test_strehl_of_self_is_unity(self, gaussian_beam, dx_m):
        s = la.strehl_ratio(gaussian_beam, gaussian_beam, dx_m)
        assert np.isclose(s, 1.0, atol=1e-9)

    def test_strehl_of_aberrated_less_than_unity(self, gaussian_beam, dx_m):
        N = gaussian_beam.shape[0]
        # Add tip-tilt aberration to the reference: should reduce Strehl
        x = (np.arange(N) - N / 2) * dx_m
        X, _Y = np.meshgrid(x, x)
        tilt_phase = np.exp(1j * 0.5 * X / dx_m)  # noticeable tilt
        E_ab = gaussian_beam * tilt_phase
        s = la.strehl_ratio(E_ab, gaussian_beam, dx_m)
        # Should still be a non-negative real number; tilt reduces Strehl.
        assert 0.0 <= s < 1.0


# ----------------------------------------------------------------------
# Beam centroid
# ----------------------------------------------------------------------

class TestBeamCentroid:

    def test_centroid_returns_tuple(self, gaussian_beam, dx_m):
        c = la.beam_centroid(gaussian_beam, dx_m)
        assert isinstance(c, tuple)
        assert len(c) == 2

    def test_symmetric_beam_centroid_near_zero(self, gaussian_beam, dx_m):
        cx, cy = la.beam_centroid(gaussian_beam, dx_m)
        # A symmetric centred Gaussian has centroid at the origin.
        assert abs(cx) < dx_m
        assert abs(cy) < dx_m


# ----------------------------------------------------------------------
# Beam power
# ----------------------------------------------------------------------

class TestBeamPower:

    def test_power_is_positive_real(self, gaussian_beam, dx_m):
        p = la.beam_power(gaussian_beam, dx_m)
        assert isinstance(p, float)
        assert p > 0

    def test_power_scales_with_amplitude_squared(self, gaussian_beam, dx_m):
        p1 = la.beam_power(gaussian_beam, dx_m)
        p2 = la.beam_power(2.0 * gaussian_beam, dx_m)
        # power ~ |E|^2, so doubling amplitude quadruples power.
        assert np.isclose(p2 / p1, 4.0, rtol=1e-9)


# ----------------------------------------------------------------------
# Zernike decomposition (uses an aberrated OPD map)
# ----------------------------------------------------------------------

class TestZernikeDecompose:
    """``zernike_decompose`` returns ``(coeffs, mode_names)``."""

    def test_decompose_returns_coeffs_and_names(self, N_small, dx_m):
        # Construct a pure-defocus OPD map and decompose.
        x = (np.arange(N_small) - N_small / 2) * dx_m
        X, Y = np.meshgrid(x, x)
        aperture = (N_small // 2 - 2) * dx_m
        rho = np.sqrt(X**2 + Y**2) / aperture
        opd = np.where(rho <= 1.0, 0.1 * (2 * rho**2 - 1), 0.0)
        coeffs, names = la.zernike_decompose(opd, dx_m, aperture,
                                              n_modes=10)
        assert coeffs.shape == (10,)
        assert np.all(np.isfinite(coeffs))
        assert len(names) == 10
        # 'Defocus' should be one of the named modes
        assert any('focus' in n.lower() for n in names)
