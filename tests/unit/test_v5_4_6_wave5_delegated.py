"""v5.4.6 Wave 5 regression pins: behavior-changing delegated findings
(F-16 aperture, F-19 biconic NaN, F-22 BSDF batched, F-23 grating order,
F-34 richards_wolf dead-code smoke).  Docstring-only findings (P3-8/9,
F-4/20/21/24/25/26/27/37) carry no behavioral pin.
"""
from __future__ import annotations

import numpy as np
import pytest

# ---- F-16: annular aperture default inner-diameter --------------------

def test_annular_aperture_default_inner_is_half_diameter():
    from lumenairy.algebra.apertures import Aperture
    a = Aperture(diameter=10e-3, shape='annular')
    assert np.isclose(a.inner_diameter, 5e-3), (
        "default annular aperture must have a D/2 central obstruction")


def test_aperture_inner_ge_diameter_raises():
    from lumenairy.algebra.apertures import Aperture
    with pytest.raises(ValueError):
        Aperture(diameter=10e-3, shape='annular', inner_diameter=10e-3)


# ---- F-19: biconic sag is NaN (not 0) outside the conic domain --------

def test_biconic_sag_nan_outside_conic_domain():
    from lumenairy.elements.lenses import surface_sag_biconic
    # Sphere R=10 mm; sample a point at h=20 mm so (1+K) h^2 / R^2 = 4 > 1.
    X = np.array([[0.0, 20e-3]])
    Y = np.array([[0.0, 0.0]])
    sag = surface_sag_biconic(X, Y, R_x=10e-3)
    assert np.isfinite(sag[0, 0]), "on-axis sag must be finite"
    assert np.isnan(sag[0, 1]), (
        "beyond the conic domain biconic sag must be NaN, not a silent 0.0")


# ---- F-22: GaussianBSDF.evaluate batched-incidence path ---------------

def test_gaussian_bsdf_batched_evaluate():
    from lumenairy.elements.bsdf import GaussianBSDF
    g = GaussianBSDF(sigma_rad=0.02, scattered_fraction=0.05)
    single = g.evaluate(np.array([0.0, 0.0, -1.0]), np.array([0.0, 0.0, 1.0]))
    assert np.ndim(single) == 0 or single.shape == ()
    inc = np.tile(np.array([0.0, 0.3, -np.sqrt(1 - 0.09)]), (4, 1))
    sca = np.tile(np.array([0.0, 0.0, 1.0]), (4, 1))
    out = np.asarray(g.evaluate(inc, sca))
    assert out.shape == (4,), f"batched evaluate must return (4,), got {out.shape}"


# ---- F-23: out-of-range grating order raises --------------------------

def test_grating_efficiency_out_of_range_order_raises():
    from lumenairy.elements.thin_grating import grating_efficiency_vs_wavelength
    with pytest.raises(ValueError, match='outside the retained range'):
        grating_efficiency_vs_wavelength(
            period=2e-6, n_ridge=1.5, n_groove=1.0, n_substrate=1.5,
            n_superstrate=1.0, depth=1e-6, duty_cycle=0.5,
            wavelengths=550e-9, order=20, n_orders=11)


# ---- F-34: richards_wolf_focus still runs after dead-code removal -----

def test_richards_wolf_focus_smoke():
    from lumenairy.propagators.vector_diffraction import richards_wolf_focus
    pupil = np.ones((32, 32), dtype=np.complex128)
    out = richards_wolf_focus(pupil, wavelength=633e-9, NA=0.5, f=1e-3,
                              dx_pupil=20e-6, N_focal=32)
    assert len(out) == 5  # (Ex, Ey, Ez, x_f, y_f)
