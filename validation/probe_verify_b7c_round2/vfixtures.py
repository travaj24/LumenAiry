"""VERIFY-WP-B7c round 2 -- the verifier's OWN optics.

None of these is a builder fixture.  The brief asks for at least one inside
the 0.12-0.29 NA envelope every published fixture sits in, one above 0.33,
one conic-surfaced, and one doublet; a fifth is a plano-convex run PLANO-FIRST
(the shared-oracle fixtures are all convex-first or biconvex), and a sixth is
a deliberately slow optic used as the CONVERGED control (the reading must be
1 there).

Two builder fixtures are imported at the bottom for the claims that are
stated ON them (claim 2's ``V`` at z = 1761 um, claim 4's ``F_alt``): those
are the builder's own numbers and must be re-measured on the builder's own
geometry, not on a lookalike.
"""
from __future__ import annotations

import numpy as np


def gauss(N, dx, w0, dtype=np.complex128):
    x = (np.arange(N) - N / 2.0) * dx
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X * X + Y * Y) / (w0 * w0)).astype(dtype)


def _singlet(r1, r2, t, semi, glass, wl, aperture, k1=0.0, k2=0.0):
    return {'wavelength': wl, 'aperture_diameter': aperture, 'surfaces': [
        {'radius': r1, 'conic': k1, 'thickness': t, 'glass_before': 'air',
         'glass_after': glass, 'semi_diameter': semi},
        {'radius': r2, 'conic': k2, 'thickness': 0.0, 'glass_before': glass,
         'glass_after': 'air', 'semi_diameter': semi}],
        'thicknesses': [t], 'stop_index': 0}


def _cemented(r1, r2, r3, t1, t2, g1, g2, semi, wl, aperture):
    return {'wavelength': wl, 'aperture_diameter': aperture, 'surfaces': [
        {'radius': r1, 'thickness': t1, 'glass_before': 'air',
         'glass_after': g1, 'semi_diameter': semi},
        {'radius': r2, 'thickness': t2, 'glass_before': g1,
         'glass_after': g2, 'semi_diameter': semi},
        {'radius': r3, 'thickness': 0.0, 'glass_before': g2,
         'glass_after': 'air', 'semi_diameter': semi}],
        'thicknesses': [t1, t2], 'stop_index': 0}


# --- W: plano-convex run PLANO-FIRST, inside the 0.12-0.29 envelope --------
FIX_W = dict(
    name='W',
    note='N-BAK4 plano-convex PLANO-FIRST (R1 = inf, R2 = -2.9 mm), '
         't = 0.80 mm, 1.10 mm aperture, 780 nm',
    prescription=_singlet(np.inf, -2.9e-3, 0.80e-3, 0.55e-3, 'N-BAK4',
                          780e-9, 1.10e-3),
    wavelength=780e-9, N=512, dx=2.40e-6, w0=400e-6,
)

# --- X: fast biconvex, ABOVE NA 0.33 --------------------------------------
FIX_X = dict(
    name='X',
    note='N-SF10 biconvex R = +/-2.0 mm, t = 0.95 mm, stopped to 1.15 mm '
         '(NA 0.3865, y_max/z ~ 0.5 -- above the 0.33 the brief asks for and '
         'above every published fixture), 1.064 um',
    prescription=_singlet(2.0e-3, -2.0e-3, 0.95e-3, 0.63e-3, 'N-SF10',
                          1.064e-6, 1.15e-3),
    wavelength=1.064e-6, N=640, dx=1.40e-6, w0=330e-6,
)

# --- Y: conic on the SECOND surface, OBLATE (k > 0) ------------------------
FIX_Y = dict(
    name='Y',
    note='N-BK7 biconvex R = +/-2.4 mm with an OBLATE conic k2 = +0.90 on '
         'surface 2, t = 0.75 mm, 0.95 mm aperture, 633 nm',
    prescription=_singlet(2.4e-3, -2.4e-3, 0.75e-3, 0.50e-3, 'N-BK7',
                          633e-9, 0.95e-3, k1=0.0, k2=0.90),
    wavelength=633e-9, N=512, dx=1.80e-6, w0=340e-6,
)

# --- Z: CEMENTED doublet, glasses the campaign has not used ---------------
FIX_Z = dict(
    name='Z',
    note='CEMENTED doublet N-BAK4 (+2.8 / -2.2, t = 0.75) / N-F2 '
         '(-2.2 / -9.0, t = 0.45), 1.00 mm aperture, 532 nm',
    prescription=_cemented(2.8e-3, -2.2e-3, -9.0e-3, 0.75e-3, 0.45e-3,
                           'N-BAK4', 'N-F2', 0.55e-3, 532e-9, 1.00e-3),
    wavelength=532e-9, N=512, dx=1.60e-6, w0=340e-6,
)

# --- C: the CONVERGED control -- a slow optic far from any caustic --------
FIX_C = dict(
    name='C',
    note='N-BK7 plano-convex R1 = +12.0 mm, t = 1.0 mm, 1.60 mm aperture '
         '(NA ~ 0.034), 1.064 um -- the reading must be 1 here',
    prescription=_singlet(12.0e-3, np.inf, 1.00e-3, 0.85e-3, 'N-BK7',
                          1.064e-6, 1.60e-3),
    wavelength=1.064e-6, N=512, dx=4.00e-6, w0=520e-6,
)

FIXTURES = {f['name']: f for f in (FIX_W, FIX_X, FIX_Y, FIX_Z, FIX_C)}


def alt(fx, N, dx):
    g = dict(fx)
    g['name'] = fx['name'] + '_alt'
    g['N'] = N
    g['dx'] = dx
    return g


FIXTURES['W_alt'] = alt(FIX_W, 640, 1.90e-6)
FIXTURES['X_alt'] = alt(FIX_X, 512, 1.75e-6)
FIXTURES['Y_alt'] = alt(FIX_Y, 640, 1.45e-6)
FIXTURES['Z_alt'] = alt(FIX_Z, 400, 2.05e-6)
FIXTURES['C_alt'] = alt(FIX_C, 400, 5.20e-6)


def _builder_fixtures():
    """``V`` and ``F_alt`` from the builder's archive -- only for the claims
    that are stated on THOSE geometries (claim 2, claim 4)."""
    import importlib.util
    import os
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
        __file__))), 'probe_wp_b7c_round2', 'fixtures.py')
    spec = importlib.util.spec_from_file_location('_b7c2_fixtures', p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def builder(name):
    return _builder_fixtures().FIXTURES[name]


def input_field(fx, dtype=np.complex128):
    return gauss(fx['N'], fx['dx'], fx['w0'], dtype=dtype)
