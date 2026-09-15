"""Fixtures for WP-B7c round 2 -- VERIFY-WP-B7c's four oracle-valid optics
plus three the campaign has never had.

VERIFY-WP-B7c re-derived the band on S (N-SK16 plano-convex convex-first /
532 nm), M (N-SF6 positive meniscus / 2.0 um), Q (cemented N-BK7 / N-SF6
doublet / 1.31 um) and F_alt (N-LASF9 biconvex stopped to f/2.0 / 633 nm), and
excluded P (f/1.2) as past the shared oracle's Debye ``J0`` ceiling.  Those
four are imported verbatim, so the round-2 population CONTAINS the population
the round-1 bar was refuted on.  Three more are added here:

* ``A`` -- an AIR-SPACED doublet (four surfaces with a real air gap between
  two elements), N-BK7 positive + N-SF11 negative at 1.064 um.  Neither study
  had an air gap: the ray map crosses two glass-air boundaries between the
  elements, and the second element's negative power moves the marginal focus
  independently of the first's aberration.
* ``K`` -- a CONIC first surface (prolate ellipsoid, ``conic = -0.55``) on an
  N-LAK22 biconvex at 1.31 um.  Every optic in WP-B7b / VERIFY-B7b / WP-B7c /
  VERIFY-WP-B7c is all-spherical; a conic PARTIALLY corrects the spherical
  aberration, so the fold is much tighter and its two-branch band much
  narrower at the same aperture -- the regime the linear ``zeta`` fit is worst
  in.
* ``G`` -- a fast N-SF11 biconvex stopped to NA 0.33, i.e. ABOVE the 0.12-0.29
  envelope every published fixture sits in and above the 0.3 the round-2 brief
  asks for.  The shared oracle's Debye ``J0`` azimuthal form does not hold
  there (VERIFY-WP-B7c D5), so ``G`` is scored against the EXACT azimuthal
  quadrature added in ``oracle.py`` -- see the ceiling measurement there.

``*_alt`` are the same optics on a different grid (``N`` and ``dx`` both
changed), so every claim can be read on two grids per optic.
"""
from __future__ import annotations

import copy
import importlib.util
import os

import numpy as np

# VERIFY-WP-B7c's fixture module has the same basename as this one, so it is
# loaded by PATH under a distinct module name rather than by ``sys.path``.
_VPATH = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'probe_verify_b7c', 'fixtures.py')
_spec = importlib.util.spec_from_file_location('_verify_b7c_fixtures', _VPATH)
VFX = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(VFX)

gauss = VFX.gauss
_singlet = VFX._singlet
_alt = VFX._alt


def _airspaced(r1, t1, g1, gap, r2, r3, t2, g2, r4, semi, wl, aperture):
    """Two elements with a real AIR GAP -- four surfaces, two of them
    glass->air."""
    return {'wavelength': wl, 'aperture_diameter': aperture, 'surfaces': [
        {'radius': r1, 'thickness': t1, 'glass_before': 'air',
         'glass_after': g1, 'semi_diameter': semi},
        {'radius': r2, 'thickness': gap, 'glass_before': g1,
         'glass_after': 'air', 'semi_diameter': semi},
        {'radius': r3, 'thickness': t2, 'glass_before': 'air',
         'glass_after': g2, 'semi_diameter': semi},
        {'radius': r4, 'thickness': 0.0, 'glass_before': g2,
         'glass_after': 'air', 'semi_diameter': semi}],
        'thicknesses': [t1, gap, t2], 'stop_index': 0}


def _conic_biconvex(r1, r2, k1, thickness, semi, glass, wl, aperture):
    return {'wavelength': wl, 'aperture_diameter': aperture, 'surfaces': [
        {'radius': r1, 'conic': k1, 'thickness': thickness,
         'glass_before': 'air', 'glass_after': glass, 'semi_diameter': semi},
        {'radius': r2, 'conic': 0.0, 'thickness': 0.0,
         'glass_before': glass, 'glass_after': 'air', 'semi_diameter': semi}],
        'thicknesses': [thickness], 'stop_index': 0}


FIX_A = dict(
    name='A',
    note='AIR-SPACED doublet N-BK7 +3.2/-3.2 (t=0.80) | gap 0.35 | N-SF11 '
         '-12.0/+30.0 (t=0.55), 1.20 mm aperture, 1.064 um',
    prescription=_airspaced(3.2e-3, 0.80e-3, 'N-BK7', 0.35e-3,
                            -3.2e-3, -12.0e-3, 0.55e-3, 'N-SF11', 30.0e-3,
                            0.60e-3, 1.064e-6, 1.20e-3),
    wavelength=1.064e-6, N=512, dx=3.00e-6, w0=420e-6,
)

FIX_K = dict(
    name='K',
    note='N-LAK22 biconvex R=+/-3.0 mm with CONIC k1=-0.55 on surface 1, '
         't=0.70 mm, 1.00 mm aperture, 1.31 um',
    prescription=_conic_biconvex(3.0e-3, -3.0e-3, -0.55, 0.70e-3, 0.50e-3,
                                 'N-LAK22', 1.31e-6, 1.00e-3),
    wavelength=1.31e-6, N=512, dx=2.00e-6, w0=360e-6,
)

FIX_G = dict(
    name='G',
    note='fast N-SF11 biconvex R=+/-2.0 mm, t=0.85 mm, stopped to 0.85 mm '
         '(NA ~ 0.33), 850 nm',
    prescription=_singlet(2.0e-3, -2.0e-3, 0.85e-3, 0.55e-3,
                          'N-SF11', 850e-9, 0.85e-3),
    wavelength=850e-9, N=768, dx=1.20e-6, w0=250e-6,
)

# VERIFY-B7b's own fold fixture, which WP-B7c reproduced the blow-up on and
# which the shipped decision tests already run: carried here so the round-2
# population CONTAINS the fixture the round-1 bar was derived against, and so
# the tests can find an R-5 plane (refused by the arbiter while INSIDE the
# power-ratio bar) on a 512-grid optic that is already in the suite.
FIX_V = dict(
    name='V',
    note="VERIFY-B7b's N-BAF10 biconvex R=+/-2.6 mm, t=0.70 mm, 0.90 mm "
         'aperture, 1.064 um',
    prescription=_singlet(2.6e-3, -2.6e-3, 0.70e-3, 0.45e-3,
                          'N-BAF10', 1.064e-6, 0.90e-3),
    wavelength=1.064e-6, N=512, dx=2.20e-6, w0=330e-6,
)

FIX_A_ALT = _alt(FIX_A, 640, 2.40e-6)
FIX_K_ALT = _alt(FIX_K, 400, 2.60e-6)
FIX_G_ALT = _alt(FIX_G, 640, 1.45e-6)

FIXTURES = dict(VFX.FIXTURES)
for _f in (FIX_A, FIX_K, FIX_G, FIX_V, FIX_A_ALT, FIX_K_ALT,
           FIX_G_ALT, _alt(FIX_V, 400, 2.80e-6)):
    FIXTURES[_f['name']] = _f

#: The optics the band is derived on.  ``P`` is excluded for the reason
#: VERIFY-WP-B7c gives (the shared oracle's Debye ``J0`` ceiling); ``G`` is
#: above NA 0.3 and is scored with the EXACT azimuthal quadrature instead.
DERIVATION_OPTICS = ('S', 'M', 'Q', 'F_alt', 'A', 'K', 'G', 'V')


def input_field(fx, dtype=np.complex128):
    return gauss(fx['N'], fx['dx'], fx['w0'], dtype=dtype)


def na_of(fx, efl_m):
    """Geometric NA at the clear aperture, for the oracle-envelope table."""
    return float(0.5 * fx['prescription']['aperture_diameter'] / efl_m)


__all__ = ['FIXTURES', 'DERIVATION_OPTICS', 'gauss', 'input_field',
           'FIX_A', 'FIX_K', 'FIX_G', 'FIX_V', 'FIX_A_ALT', 'FIX_K_ALT',
           'FIX_G_ALT',
           'na_of', 'copy']
