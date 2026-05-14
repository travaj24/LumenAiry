"""Unit tests for source generators.

Every source factory in lumenairy returns ``(E, x, y)`` where ``E``
is the complex field on an ``N x N`` grid and ``(x, y)`` are the
1-D spatial coordinate axes.  The unit tests check the contract
for each common source.
"""
from __future__ import annotations

import numpy as np
import pytest

import lumenairy as la


# ----------------------------------------------------------------------
# Gaussian beam
# ----------------------------------------------------------------------

class TestGaussianBeam:

    def test_shape_and_dtype(self, N_small, dx_m, wavelength_m):
        E, x, y = la.create_gaussian_beam(N_small, dx_m, wavelength_m,
                                            sigma=30e-6)
        assert E.shape == (N_small, N_small)
        assert np.iscomplexobj(E)
        assert x.shape == (N_small,)
        assert y.shape == (N_small,)

    def test_peak_at_center(self, N_small, dx_m, wavelength_m):
        E, _x, _y = la.create_gaussian_beam(N_small, dx_m, wavelength_m,
                                              sigma=30e-6)
        I = np.abs(E) ** 2
        iy, ix = np.unravel_index(int(np.argmax(I)), I.shape)
        assert abs(iy - N_small // 2) <= 1
        assert abs(ix - N_small // 2) <= 1

    def test_finite_power(self, N_small, dx_m, wavelength_m):
        E, _x, _y = la.create_gaussian_beam(N_small, dx_m, wavelength_m,
                                              sigma=30e-6)
        power = float(np.sum(np.abs(E) ** 2) * dx_m ** 2)
        assert power > 0
        assert np.isfinite(power)


# ----------------------------------------------------------------------
# Tilted plane wave
# ----------------------------------------------------------------------

class TestTiltedPlaneWave:

    def test_shape(self, N_small, dx_m, wavelength_m):
        E, _x, _y = la.create_tilted_plane_wave(N_small, dx_m,
                                                  wavelength_m,
                                                  angle_x=0.01,
                                                  angle_y=0.0)
        assert E.shape == (N_small, N_small)

    def test_uniform_intensity(self, N_small, dx_m, wavelength_m):
        E, _x, _y = la.create_tilted_plane_wave(N_small, dx_m,
                                                  wavelength_m,
                                                  angle_x=0.01,
                                                  angle_y=0.0)
        I = np.abs(E) ** 2
        assert np.allclose(I, I.mean(), atol=1e-9)

    def test_zero_tilt_is_uniform_phase(self, N_small, dx_m, wavelength_m):
        E, _x, _y = la.create_tilted_plane_wave(N_small, dx_m,
                                                  wavelength_m,
                                                  angle_x=0.0,
                                                  angle_y=0.0)
        ph = np.angle(E)
        assert np.allclose(ph, ph.mean(), atol=1e-9)


# ----------------------------------------------------------------------
# Point source
# ----------------------------------------------------------------------

class TestPointSource:

    def test_shape_and_finite(self, N_small, dx_m, wavelength_m):
        E, _x, _y = la.create_point_source(N_small, dx_m, wavelength_m,
                                              z0=100e-3)
        assert E.shape == (N_small, N_small)
        assert np.all(np.isfinite(E))


# ----------------------------------------------------------------------
# Top-hat
# ----------------------------------------------------------------------

class TestTopHat:

    def test_top_hat_zero_outside_diameter(self, N_small, dx_m,
                                            wavelength_m):
        D = (N_small // 4) * dx_m  # quarter-grid disk
        E, _x, _y = la.create_top_hat_beam(N_small, dx_m, wavelength_m,
                                              diameter=D)
        x = (np.arange(N_small) - N_small / 2) * dx_m
        X, Y = np.meshgrid(x, x)
        r = np.sqrt(X * X + Y * Y)
        outside = r > D / 2 + dx_m
        assert np.allclose(np.abs(E[outside]), 0, atol=1e-9)


# ----------------------------------------------------------------------
# Annular
# ----------------------------------------------------------------------

class TestAnnularBeam:

    def test_annular_zero_at_center(self, N_small, dx_m, wavelength_m):
        E, _x, _y = la.create_annular_beam(N_small, dx_m, wavelength_m,
                                              outer_diameter=80e-6,
                                              inner_diameter=20e-6)
        c = N_small // 2
        assert abs(E[c, c]) < 1e-9
