"""Fixtures and shared helpers for the WP-B7c multibranch / zeta probes.

Three optics, none of them shared with the tests they inform:

* ``V`` -- VERIFY-B7b section 4.1/4.2's own fold fixture (N-BAF10 biconvex
  R = +/-2.6 mm, t = 0.70 mm, 0.90 mm aperture, lambda = 1.064 um, N = 512,
  dx = 2.20 um, w0 = 330 um).  Reproduced verbatim so the blow-up window of
  VERIFY-B7b section 4.2 can be re-measured on the same geometry.
* ``C`` -- a plano-convex N-BK7 singlet at 780 nm on a different grid.
* ``D`` -- an N-SF11 biconvex at 1.55 um with a smaller aperture:beam ratio.

Every optic is described by a plain dict so the probes never import a test
module.  ``gauss`` builds the collimated input.
"""
from __future__ import annotations

import numpy as np


def gauss(N, dx, w0, dtype=np.complex128):
    x = (np.arange(N) - N / 2.0) * dx
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(dtype)


def _biconvex(r1, r2, thickness, semi, glass, wl, aperture):
    return {'wavelength': wl, 'aperture_diameter': aperture, 'surfaces': [
        {'radius': r1, 'thickness': thickness, 'glass_before': 'air',
         'glass_after': glass, 'semi_diameter': semi},
        {'radius': r2, 'thickness': 0.0, 'glass_before': glass,
         'glass_after': 'air', 'semi_diameter': semi}],
        'thicknesses': [thickness], 'stop_index': 0}


# --------------------------------------------------------------------------
# Optic V -- VERIFY-B7b section 4.1's fold fixture, verbatim.
# --------------------------------------------------------------------------
FIX_V = dict(
    name='V',
    note='VERIFY-B7b sec 4.1 fold fixture: N-BAF10 biconvex R=+/-2.6 mm',
    prescription=_biconvex(2.6e-3, -2.6e-3, 0.70e-3, 0.45e-3,
                           'N-BAF10', 1.064e-6, 0.90e-3),
    wavelength=1.064e-6, N=512, dx=2.20e-6, w0=330e-6,
)

# --------------------------------------------------------------------------
# Optic C -- N-BK7 plano-convex, convex first, 780 nm, finer grid.
# --------------------------------------------------------------------------
FIX_C = dict(
    name='C',
    note='N-BK7 plano-convex R=-2.0 mm, FLAT side first (high-SA orientation), '
         '780 nm',
    prescription={'wavelength': 780e-9, 'aperture_diameter': 0.90e-3,
                  'surfaces': [
                      {'radius': float('inf'), 'thickness': 0.80e-3,
                       'glass_before': 'air', 'glass_after': 'N-BK7',
                       'semi_diameter': 0.45e-3},
                      {'radius': -2.0e-3, 'thickness': 0.0,
                       'glass_before': 'N-BK7', 'glass_after': 'air',
                       'semi_diameter': 0.45e-3}],
                  'thicknesses': [0.80e-3], 'stop_index': 0},
    wavelength=780e-9, N=512, dx=2.00e-6, w0=300e-6,
)

# --------------------------------------------------------------------------
# Optic D -- N-SF11 biconvex at 1.55 um, wider aperture:beam ratio.
# --------------------------------------------------------------------------
FIX_D = dict(
    name='D',
    note='N-SF11 biconvex R=+/-3.4 mm, 1.55 um',
    prescription=_biconvex(3.4e-3, -3.4e-3, 0.90e-3, 0.60e-3,
                           'N-SF11', 1.55e-6, 1.20e-3),
    wavelength=1.55e-6, N=512, dx=3.00e-6, w0=430e-6,
)


# --------------------------------------------------------------------------
# Optic W1 -- WP-B7b's OWN fast singlet at its marginal focus (the docstring's
# ``zeta_extrapolation = 0.16`` row, where it records `uniform` beating
# `ray_density` 0.9435 vs 0.9395).  Reproduced verbatim from WP-B7b_REPORT.md
# section 3.1.
# --------------------------------------------------------------------------
FIX_W1 = dict(
    name='W1',
    note="WP-B7b sec 3.1 fast singlet: N-LAK22 biconvex R=+/-3.0 mm, "
         "marginal focus 2058.98 um",
    prescription=_biconvex(3.0e-3, -3.0e-3, 0.55e-3, 0.60e-3,
                           'N-LAK22', 850e-9, 1.20e-3),
    wavelength=850e-9, N=640, dx=2.10e-6, w0=300e-6,
)

# --------------------------------------------------------------------------
# Optic W2 -- WP-B7b's "fixture F" (the docstring's ``453.6`` row, where it
# records `ray_density` 0.9996 against `uniform` 0.9297).  WP-B7b_REPORT.md
# section 3.2.
# --------------------------------------------------------------------------
FIX_W2 = dict(
    name='W2',
    note='WP-B7b sec 3.2 fixture F: N-LAK22 biconvex R=+/-6.0 mm, 1.55 um, '
         'z = 4.400 mm',
    prescription=_biconvex(6.0e-3, -6.0e-3, 0.90e-3, 0.55e-3,
                           'N-LAK22', 1.55e-6, 1.10e-3),
    wavelength=1.55e-6, N=320, dx=3.85e-6, w0=400e-6,
)

FIXTURES = {'V': FIX_V, 'C': FIX_C, 'D': FIX_D,
            'W1': FIX_W1, 'W2': FIX_W2}


def input_field(fx):
    return gauss(fx['N'], fx['dx'], fx['w0'])


def build_env():
    """Return (lumenairy module, its __file__) after asserting the tree."""
    import lumenairy
    return lumenairy, lumenairy.__file__
