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

    def test_diverging_vs_converging_sign_flips(self, N_small, dx_m,
                                                  wavelength_m):
        """Regression for the sign-convention bug in 4.7.x where the
        formula ``exp(+i k r)/r`` with ``r = sqrt(... + z0**2)`` was
        sign-independent in ``z0`` and always produced a diverging wave.

        z0 < 0 (source before grid) -> diverging (positive curvature,
        phase grows outward); z0 > 0 (focus after grid) -> converging
        (negative curvature, phase shrinks outward).
        """
        z = 100e-3
        E_div, _, _ = la.create_point_source(
            N_small, dx_m, wavelength_m, z0=-z)
        E_conv, _, _ = la.create_point_source(
            N_small, dx_m, wavelength_m, z0=+z)
        i_c = N_small // 2
        i_off = i_c + N_small // 4
        ph_div = np.unwrap(np.angle(E_div[i_c, :]))
        ph_conv = np.unwrap(np.angle(E_conv[i_c, :]))
        delta_div = ph_div[i_off] - ph_div[i_c]
        delta_conv = ph_conv[i_off] - ph_conv[i_c]
        # |r| is sign-independent in z0 so magnitudes are equal but signs
        # must be opposite.
        assert np.isclose(delta_div, -delta_conv, rtol=1e-9, atol=1e-9), (
            f"diverging and converging phase curvatures should be "
            f"opposite-signed; got {delta_div=:.4f}, {delta_conv=:.4f}"
        )
        assert delta_div > 0, (
            f"z0 < 0 must give diverging (positive) phase curvature, "
            f"got {delta_div=:.4f}"
        )
        assert delta_conv < 0, (
            f"z0 > 0 must give converging (negative) phase curvature, "
            f"got {delta_conv=:.4f}"
        )

    def test_converging_wave_focuses_after_propagation(
            self, N_small, dx_m, wavelength_m):
        """A converging wave (z0 > 0) propagated through z = z0 with ASM
        should concentrate energy near the on-axis pixel."""
        z = 5e-3
        E, _, _ = la.create_point_source(
            N_small, dx_m, wavelength_m, z0=z, amplitude=1.0)
        E_focal = la.angular_spectrum_propagate(
            E, z=z, wavelength=wavelength_m, dx=dx_m, bandlimit=True)
        I = np.abs(E_focal) ** 2
        peak = int(np.argmax(I))
        peak_iy, peak_ix = divmod(peak, N_small)
        assert abs(peak_ix - N_small // 2) <= 2, (
            f"converging wave should focus near centre, got peak at "
            f"({peak_iy}, {peak_ix}) for {N_small=}"
        )
        assert abs(peak_iy - N_small // 2) <= 2


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
