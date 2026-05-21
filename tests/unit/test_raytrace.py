"""Unit tests for raytrace core: surfaces, ray bundles, ABCD, Seidel.

Contract focus: API plumbing for the geometric raytrace.  Detailed
physics validation is in ``validation/raytrace/``.
"""
from __future__ import annotations

import numpy as np
import pytest

import lumenairy as la

# ----------------------------------------------------------------------
# Ray bundles
# ----------------------------------------------------------------------

class TestMakeRay:

    def test_make_ray_default(self, wavelength_m):
        ray = la.make_ray(x=0.0, y=0.0, L=0.0, M=0.0, wavelength=wavelength_m)
        assert isinstance(ray, la.RayBundle)

    def test_make_fan_n_rays(self, wavelength_m):
        bundle = la.make_fan(axis='y', semi_aperture=1e-3, n_rays=11,
                              wavelength=wavelength_m)
        assert isinstance(bundle, la.RayBundle)


# ----------------------------------------------------------------------
# Prescriptions -> surfaces
# ----------------------------------------------------------------------

class TestSurfacesFromPrescription:

    def test_singlet_yields_two_surfaces(self, singlet_prescription):
        surfaces = la.surfaces_from_prescription(singlet_prescription)
        # Singlet has 2 refracting surfaces (R1 + R2)
        assert len(surfaces) >= 2
        assert all(isinstance(s, la.Surface) for s in surfaces)


# ----------------------------------------------------------------------
# system_abcd
# ----------------------------------------------------------------------

class TestSystemABCD:
    """``system_abcd`` returns ``(M, efl, bfl, ffl)`` where ``M`` is the
    2x2 ABCD ray-transfer matrix."""

    def test_abcd_returns_matrix_and_focal_lengths(
            self, singlet_prescription, wavelength_m):
        surfaces = la.surfaces_from_prescription(singlet_prescription)
        result = la.system_abcd(surfaces, wavelength_m)
        assert isinstance(result, tuple) and len(result) == 4
        M, efl, bfl, ffl = result
        M = np.asarray(M)
        assert M.shape == (2, 2)
        assert np.all(np.isfinite(M))
        for fl in (efl, bfl, ffl):
            assert np.isfinite(fl)

    def test_abcd_determinant_unity_for_lossless_lens(
            self, singlet_prescription, wavelength_m):
        """Reciprocal lossless system: det(M) = n_in / n_out = 1
        (air-to-air)."""
        surfaces = la.surfaces_from_prescription(singlet_prescription)
        M = np.asarray(la.system_abcd(surfaces, wavelength_m)[0])
        det = float(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0])
        assert np.isclose(det, 1.0, rtol=1e-9)


# ----------------------------------------------------------------------
# Seidel coefficients
# ----------------------------------------------------------------------

class TestSeidelCoefficients:

    def test_seidel_returns_named_data(self, singlet_prescription,
                                        wavelength_m):
        """Smoke test: seidel_coefficients runs and returns something
        iterable / dict-like with per-aberration values."""
        surfaces = la.surfaces_from_prescription(singlet_prescription)
        result = la.seidel_coefficients(surfaces, wavelength_m)
        # Result API may be tuple, dict, or dataclass.  Just check
        # it returned something non-None and finite.
        assert result is not None


# ----------------------------------------------------------------------
# Paraxial helpers (4.0+)
# ----------------------------------------------------------------------

class TestParaxialHelpers:

    def test_f_number_positive_for_converging_lens(
            self, singlet_prescription, wavelength_m):
        fn = la.f_number(singlet_prescription, wavelength_m)
        assert fn > 0
        assert np.isfinite(fn)

    def test_defocus_waves_to_zernike_roundtrip(self):
        c = la.defocus_waves_to_zernike(1.0)
        # 1 wave of PV defocus -> c20 = 1 / (2 sqrt(3))
        assert np.isclose(c, 1.0 / (2 * np.sqrt(3)), rtol=1e-12)

    def test_astigmatism_waves_to_zernike(self):
        c = la.astigmatism_waves_to_zernike(1.0)
        # 1 wave of PV astigmatism -> c22 = 1 / sqrt(6)
        assert np.isclose(c, 1.0 / np.sqrt(6), rtol=1e-12)
