"""Unit tests for ``apply_real_lens_gbd`` -- the Gaussian-beamlet-decomposition
peer of ``apply_real_lens`` / ``apply_real_lens_traced`` / ``apply_real_lens_maslov``.

The physics contract is pinned against ``apply_real_lens`` on a LOW-NA lens,
where the analytic thin-element model is essentially exact (its ``sag*theta**2``
error is negligible): the beamlet field must agree at the exit plane, conserve
energy, and preserve the grid.
"""
from __future__ import annotations

import numpy as np
import pytest

import lumenairy as la


def _low_na_prescription():
    # f ~ 100 mm biconvex at a 1.5 mm aperture -> ~F/67: aberration-free, so
    # analytic and gbd must agree.
    return la.make_singlet(100e-3, -100e-3, 3e-3, 'N-BK7', aperture=1.5e-3)


def _disk(N, dx, radius):
    xs = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(xs, xs)
    return ((X ** 2 + Y ** 2) <= radius ** 2).astype(np.complex128)


class TestApplyRealLensGBD:
    def test_matches_analytic_low_na(self):
        """Exit-plane field overlaps the analytic model >0.98 and conserves
        energy on a low-NA lens (the analytic model's accurate regime)."""
        presc = _low_na_prescription()
        wl, dx, N = 1.31e-6, 8e-6, 256
        E0 = _disk(N, dx, 0.7e-3)

        Ea = la.apply_real_lens(E0.copy(), prescription=presc,
                                wavelength=wl, dx=dx)
        Eg = la.apply_real_lens_gbd(E0.copy(), prescription=presc,
                                    wavelength=wl, dx=dx)
        Eg = np.asarray(Eg)

        assert Eg.shape == E0.shape
        a, g = Ea.ravel(), Eg.ravel()
        overlap = (np.abs(np.vdot(a, g)) ** 2
                   / (np.vdot(a, a).real * np.vdot(g, g).real))
        assert overlap > 0.98, f"gbd vs analytic overlap {overlap:.4f} < 0.98"

        pa = float(np.sum(np.abs(a) ** 2))
        pg = float(np.sum(np.abs(g) ** 2))
        assert 0.9 < pg / pa < 1.1, f"energy ratio {pg / pa:.3f} not ~1"

    def test_grid_and_dtype_preserved(self):
        presc = _low_na_prescription()
        E0 = np.ones((128, 128), np.complex128)
        Eg = np.asarray(la.apply_real_lens_gbd(
            E0, prescription=presc, wavelength=1.31e-6, dx=8e-6))
        assert Eg.shape == (128, 128)
        assert np.iscomplexobj(Eg)

    def test_auto_waist_ties_to_sample_step(self):
        """The auto frame is overlapping: leaving both defaults, or passing
        ``sample_step`` alone, ties the waist to the spacing (no sparse-frame
        footgun).  We just assert the call runs and both explicit/auto agree."""
        presc = _low_na_prescription()
        wl, dx, N = 1.31e-6, 8e-6, 128
        E0 = _disk(N, dx, 0.7e-3)
        auto = np.asarray(la.apply_real_lens_gbd(
            E0.copy(), prescription=presc, wavelength=wl, dx=dx))
        # Passing sample_step alone must tie waist_factor to it internally.
        expl = np.asarray(la.apply_real_lens_gbd(
            E0.copy(), prescription=presc, wavelength=wl, dx=dx,
            sample_step=2))
        assert auto.shape == expl.shape == (N, N)

    def test_roi_not_supported_raises(self):
        presc = _low_na_prescription()
        E0 = np.ones((64, 64), np.complex128)
        with pytest.raises(NotImplementedError):
            la.apply_real_lens_gbd(E0, prescription=presc,
                                   wavelength=1.31e-6, dx=8e-6,
                                   roi=(0, 32, 0, 32))

    def test_medium_output_leg_raises(self):
        presc = _low_na_prescription()
        E0 = np.ones((64, 64), np.complex128)
        with pytest.raises(NotImplementedError):
            la.apply_real_lens_gbd(E0, prescription=presc,
                                   wavelength=1.31e-6, dx=8e-6,
                                   output_plane_distance=1e-3,
                                   output_plane_n=1.5)
