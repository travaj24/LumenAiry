"""Unit tests for free-space propagators (ASM, Fresnel, resample).

Contracts checked:
* dtype / shape preservation
* z=0 returns the input
* round-trip (forward then backward) recovers the input
* resample_field returns the field at the new pixel size
"""
from __future__ import annotations

import numpy as np
import pytest

import lumenairy as la


# ----------------------------------------------------------------------
# Angular spectrum (ASM)
# ----------------------------------------------------------------------

class TestAngularSpectrum:
    """``angular_spectrum_propagate`` returns the propagated field
    directly (no tuple)."""

    def test_preserves_shape_and_dtype(self, plane_wave, wavelength_m, dx_m):
        out = la.angular_spectrum_propagate(plane_wave, 10e-3,
                                              wavelength_m, dx_m)
        assert out.shape == plane_wave.shape
        assert out.dtype == plane_wave.dtype

    def test_zero_distance_is_identity(self, gaussian_beam, wavelength_m, dx_m):
        out = la.angular_spectrum_propagate(gaussian_beam, 0.0,
                                              wavelength_m, dx_m)
        assert np.allclose(out, gaussian_beam, atol=1e-12)

    def test_round_trip_recovers_input(self, gaussian_beam, wavelength_m, dx_m):
        """Forward then backward should return the input field for a
        short, near-field propagation (the ASM bandlimit filter
        introduces small losses over longer paths)."""
        z_near = 100 * dx_m  # ~ 500 um -- well inside the near-field
        forward = la.angular_spectrum_propagate(gaussian_beam, z_near,
                                                  wavelength_m, dx_m)
        back = la.angular_spectrum_propagate(forward, -z_near,
                                               wavelength_m, dx_m)
        # Compare on the field magnitude; bandlimit clipping costs
        # a small amount of total amplitude.
        peak_in = float(np.max(np.abs(gaussian_beam)))
        max_dev = float(np.max(np.abs(back - gaussian_beam)))
        assert max_dev < 0.01 * peak_in, \
            f'round-trip max deviation = {max_dev:.2e} vs peak {peak_in:.2e}'


# ----------------------------------------------------------------------
# Fresnel
# ----------------------------------------------------------------------

class TestFresnelPropagate:
    """``fresnel_propagate`` returns ``(E, dx_out, dy_out)``."""

    def test_preserves_shape(self, plane_wave, wavelength_m, dx_m):
        result = la.fresnel_propagate(plane_wave, 50e-3, wavelength_m, dx_m)
        E_out = result[0]
        assert E_out.shape == plane_wave.shape

    def test_dtype_is_complex(self, plane_wave, wavelength_m, dx_m):
        result = la.fresnel_propagate(plane_wave, 50e-3, wavelength_m, dx_m)
        E_out = result[0]
        assert np.iscomplexobj(E_out)

    def test_returns_grid_spacing(self, plane_wave, wavelength_m, dx_m):
        result = la.fresnel_propagate(plane_wave, 50e-3, wavelength_m, dx_m)
        # Fresnel uses an MFT-like grid update; the second element
        # should be a positive output grid spacing.
        dx_out = result[1]
        assert dx_out > 0


# ----------------------------------------------------------------------
# Resample field
# ----------------------------------------------------------------------

class TestResampleField:
    """``resample_field`` returns ``(E, dx_out)``."""

    def test_resample_to_same_grid(self, gaussian_beam, dx_m, N_small):
        result = la.resample_field(gaussian_beam, dx_m, dx_m,
                                     N_out=N_small)
        E_out, dx_out = result
        assert E_out.shape == (N_small, N_small)
        assert np.isclose(dx_out, dx_m)

    def test_resample_to_larger_grid(self, gaussian_beam, dx_m, N_med):
        new_dx = dx_m * 0.5
        result = la.resample_field(gaussian_beam, dx_m, new_dx,
                                     N_out=N_med)
        E_out, dx_out = result
        assert E_out.shape == (N_med, N_med)
        assert np.isclose(dx_out, new_dx)


# ----------------------------------------------------------------------
# Smart-method dispatch
# ----------------------------------------------------------------------

class TestPropagateDispatch:

    def test_propagate_smoke(self, gaussian_beam, wavelength_m, dx_m):
        """High-level dispatch should run for reasonable inputs.

        The auto-selector may route to ``asm`` (bare ndarray return)
        or to ``sas`` / ``fresnel`` (``(E, dx_out, dy_out)`` tuple
        return) depending on the grid Fresnel ratio ``Q``.  Accept
        either form."""
        out = la.propagate(gaussian_beam, z=10e-3,
                             wavelength=wavelength_m, dx=dx_m)
        assert out is not None
        if isinstance(out, tuple):
            field = out[0]
        else:
            field = out
        assert field.shape == gaussian_beam.shape

    def test_valid_methods_nonempty(self):
        """VALID_METHODS registry should be non-empty."""
        methods = la.VALID_METHODS
        assert methods is not None
        assert len(methods) > 0
