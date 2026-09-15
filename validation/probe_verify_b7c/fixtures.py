"""Fixtures for the VERIFY-WP-B7c re-derivation -- FOUR optics the WP-B7c
study did not use, plus the alternate grids the band is re-derived on.

WP-B7c derived its refusal bar on V (N-BAF10 biconvex / 1.064 um), C (N-BK7
plano-convex / 780 nm), D (N-SF11 biconvex / 1.55 um) and WP-B7b's two
N-LAK22 singlets at 850 nm / 1.55 um.  None of the prescriptions below shares
a glass, a wavelength, a grid or a surface count with any of those, and two of
them are shapes that study never had at all (a CEMENTED DOUBLET and a
MENISCUS).

* ``P`` -- fast N-LASF9 biconvex at 633 nm.  The fastest optic in either
  study (f/1.8 at the clear aperture), so its marginal focus is the most
  aberrated and its fold the widest.
* ``Q`` -- a CEMENTED DOUBLET, N-BK7 / N-SF6, at 1.31 um.  Three surfaces and
  a buried cemented interface: the ray map is the sum of two elements'
  aberration, not one's.
* ``M`` -- a positive MENISCUS in N-SF6 at 2.0 um.  Both centres of curvature
  on the same side, so the spherical aberration is strongly undercorrected
  and the fold is far from the paraxial focus.
* ``S`` -- N-SK16 plano-convex, CONVEX side first (the low-aberration
  orientation), at 532 nm -- the shortest wavelength anywhere in the study,
  and the orientation opposite to WP-B7c's own plano-convex.

``*_ALT`` are the same optics on a DIFFERENT grid (N and dx both changed), so
the band can be read on two grids per Task A.
"""
from __future__ import annotations

import copy

import numpy as np


def gauss(N, dx, w0, dtype=np.complex128):
    x = (np.arange(N) - N / 2.0) * dx
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(dtype)


def _singlet(r1, r2, thickness, semi, glass, wl, aperture):
    return {'wavelength': wl, 'aperture_diameter': aperture, 'surfaces': [
        {'radius': r1, 'thickness': thickness, 'glass_before': 'air',
         'glass_after': glass, 'semi_diameter': semi},
        {'radius': r2, 'thickness': 0.0, 'glass_before': glass,
         'glass_after': 'air', 'semi_diameter': semi}],
        'thicknesses': [thickness], 'stop_index': 0}


def _doublet(r1, t1, g1, r2, t2, g2, r3, semi, wl, aperture):
    return {'wavelength': wl, 'aperture_diameter': aperture, 'surfaces': [
        {'radius': r1, 'thickness': t1, 'glass_before': 'air',
         'glass_after': g1, 'semi_diameter': semi},
        {'radius': r2, 'thickness': t2, 'glass_before': g1,
         'glass_after': g2, 'semi_diameter': semi},
        {'radius': r3, 'thickness': 0.0, 'glass_before': g2,
         'glass_after': 'air', 'semi_diameter': semi}],
        'thicknesses': [t1, t2], 'stop_index': 0}


FIX_P = dict(
    name='P',
    note='fast N-LASF9 biconvex R=+/-2.2 mm, t=0.80 mm, 1.00 mm aperture, '
         '633 nm',
    prescription=_singlet(2.2e-3, -2.2e-3, 0.80e-3, 0.50e-3,
                          'N-LASF9', 633e-9, 1.00e-3),
    wavelength=633e-9, N=512, dx=1.80e-6, w0=340e-6,
)

FIX_Q = dict(
    name='Q',
    note='cemented doublet N-BK7 R=+3.0/-2.0 + N-SF6 -2.0/-6.0, 1.40 mm '
         'aperture, 1.31 um',
    prescription=_doublet(3.0e-3, 0.90e-3, 'N-BK7',
                          -2.0e-3, 0.60e-3, 'N-SF6',
                          -6.0e-3, 0.70e-3, 1.31e-6, 1.40e-3),
    wavelength=1.31e-6, N=512, dx=2.60e-6, w0=480e-6,
)

FIX_M = dict(
    name='M',
    note='positive MENISCUS N-SF6 R=+1.6/+4.2 mm, t=0.75 mm, 1.10 mm '
         'aperture, 2.0 um',
    prescription=_singlet(1.6e-3, 4.2e-3, 0.75e-3, 0.55e-3,
                          'N-SF6', 2.0e-6, 1.10e-3),
    wavelength=2.0e-6, N=512, dx=3.60e-6, w0=390e-6,
)

FIX_S = dict(
    name='S',
    note='N-SK16 plano-convex R=+2.4 mm CONVEX side first, t=0.85 mm, '
         '1.00 mm aperture, 532 nm',
    prescription={'wavelength': 532e-9, 'aperture_diameter': 1.00e-3,
                  'surfaces': [
                      {'radius': 2.4e-3, 'thickness': 0.85e-3,
                       'glass_before': 'air', 'glass_after': 'N-SK16',
                       'semi_diameter': 0.50e-3},
                      {'radius': float('inf'), 'thickness': 0.0,
                       'glass_before': 'N-SK16', 'glass_after': 'air',
                       'semi_diameter': 0.50e-3}],
                  'thicknesses': [0.85e-3], 'stop_index': 0},
    wavelength=532e-9, N=512, dx=1.50e-6, w0=330e-6,
)


def _alt(fx, N, dx, w0=None, tag='alt'):
    """The same optic on a different grid -- N and dx both changed."""
    out = copy.deepcopy(fx)
    out['name'] = fx['name'] + '_' + tag
    out['N'] = N
    out['dx'] = dx
    if w0 is not None:
        out['w0'] = w0
    out['note'] = fx['note'] + f' [grid {N} x {dx * 1e6:.2f} um]'
    return out


FIX_P_ALT = _alt(FIX_P, 640, 1.40e-6)
FIX_Q_ALT = _alt(FIX_Q, 384, 3.40e-6)
FIX_M_ALT = _alt(FIX_M, 400, 4.50e-6)
FIX_S_ALT = _alt(FIX_S, 640, 1.15e-6)

FIXTURES = {f['name']: f for f in (FIX_P, FIX_Q, FIX_M, FIX_S,
                                   FIX_P_ALT, FIX_Q_ALT, FIX_M_ALT,
                                   FIX_S_ALT)}


# ``F`` -- the same fast N-LASF9 biconvex stopped down to a 0.60 mm clear
# aperture (f/2.0, NA 0.26).  ``P`` at its full 1.00 mm aperture is f/1.2 /
# NA 0.45, where the shared oracle's Debye ``J0`` ring form and the library's
# own scalar members disagree with EACH OTHER (measured: fidelity 0.737
# between ``caustic='uniform'`` and ``caustic='wave'`` at z = 888 um, stable
# under a 16x refinement of the oracle), so ``P`` cannot separate an accepted
# completed field from a broken one.  ``F`` is inside the envelope every optic
# in WP-B7b / VERIFY-B7b / WP-B7c sits in (NA 0.12 .. 0.29) and is the fastest
# optic that is.
FIX_F = dict(
    name='F',
    note='fast N-LASF9 biconvex R=+/-2.2 mm stopped to 0.60 mm (f/2.0), '
         '633 nm',
    prescription=_singlet(2.2e-3, -2.2e-3, 0.80e-3, 0.50e-3,
                          'N-LASF9', 633e-9, 0.60e-3),
    wavelength=633e-9, N=512, dx=1.80e-6, w0=220e-6,
)
FIX_F_ALT = _alt(FIX_F, 640, 1.40e-6)
FIXTURES['F'] = FIX_F
FIXTURES['F_alt'] = FIX_F_ALT


def input_field(fx):
    return gauss(fx['N'], fx['dx'], fx['w0'])
