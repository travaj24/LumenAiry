"""VERIFY-WP-B7c round 3 -- the verifier's OWN optics.

The brief asks for at least six optics, at least two of which no round of this
campaign has ever used, one ASPHERIC with a non-monotone focal locus and one
at NA >= 0.4.

FOUR are new here and appear in no fixture module of WP-B7b, VERIFY-B7b,
WP-B7c, VERIFY-WP-B7c, WP-B7c round 2, VERIFY round 2 or WP-B7c round 3:

``VA``  an even-ASPHERE on the EXIT surface of an N-BK7 biconvex.  Round 3's
        own ``AS`` deforms the FIRST (entrance) surface of a plano-convex;
        this one deforms the SECOND, so it is the EXIT-side ray angles --
        what the fold completion's ``zeta`` fit reads -- that are changed,
        and the base optic is a biconvex rather than a plano-convex.  The
        focal locus turns over at h = 386 um of a 650 um clear semi-aperture
        (f falls 2773.9 -> 2728.3 um and then rises to 2907.4), which
        ``v3geom.py`` ASSERTS on the running build as
        ``turns_in_focal_locus >= 1``; every spherical, conic and cemented
        optic in this population reads zero turns.
``VX``  an N-SF6 biconvex at **NA >= 0.4**, a glass/speed combination the
        campaign has not run (round 3's high-NA optic is N-LASF9
        plano-convex, the round-2 verifier's is N-SF10 biconvex).
``VC``  a HYPERBOLIC conic (``k < -1``) on the first surface of an N-LAK22
        plano-convex.  Every conic in the campaign is a PROLATE ELLIPSOID
        (round 2's ``K``, ``k = -0.55``) or an OBLATE one (the verifier's
        ``Y``, ``k = +0.90``); a hyperboloid over-corrects, so the marginal
        focus moves to the far side of the paraxial one and the fold opens
        from the other end.
``VD``  an AIR-SPACED negative-then-positive pair (a Barlow-like doublet run
        with the DIVERGING element first).  Round 2's ``A`` is air-spaced
        positive-first; putting the negative element first makes the beam
        expand inside the gap, so the second element works at a larger zone
        than the stop.

THREE are re-typed from the published notes of the optics the claims are
stated ON, because a claim about a specific optic has to be re-measured on
that optic's own geometry and not on a lookalike:

``V``     VERIFY-B7b's N-BAF10 biconvex -- the fixture the bar is argued on;
``HN``    round 3's NA-0.41 plano-convex -- claim 2 is about this optic;
``W_alt`` the plano-first N-BAK4 on its second grid -- claim 2's best REFUSED
          plane is here;
``Q``     the cemented doublet of round 2's own oracle-floor row (claim 9).

They are TYPED HERE from the prescriptions in the published fixture modules
rather than imported, so this module has no import dependency on any builder
probe.  ``v3geom.py`` prints the paraxial/marginal foci it measures for each,
which is the cross-check that the re-typing is right.
"""
from __future__ import annotations

import numpy as np


def gauss(N, dx, w0, dtype=np.complex128):
    x = (np.arange(int(N)) - int(N) / 2.0) * float(dx)
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X * X + Y * Y) / (float(w0) ** 2)).astype(dtype)


def _surf(radius, thickness, glass_before, glass_after, semi, conic=0.0,
          aspheric=None):
    s = {'radius': radius, 'conic': conic, 'thickness': thickness,
         'glass_before': glass_before, 'glass_after': glass_after,
         'semi_diameter': semi}
    if aspheric:
        s['aspheric_coeffs'] = dict(aspheric)
    return s


def _singlet(r1, r2, t, semi, glass, wl, aperture, k1=0.0, k2=0.0,
             a1=None, a2=None):
    return {'wavelength': wl, 'aperture_diameter': aperture, 'surfaces': [
        _surf(r1, t, 'air', glass, semi, k1, a1),
        _surf(r2, 0.0, glass, 'air', semi, k2, a2)],
        'thicknesses': [t], 'stop_index': 0}


def _cemented(r1, r2, r3, t1, t2, g1, g2, semi, wl, aperture):
    return {'wavelength': wl, 'aperture_diameter': aperture, 'surfaces': [
        _surf(r1, t1, 'air', g1, semi),
        _surf(r2, t2, g1, g2, semi),
        _surf(r3, 0.0, g2, 'air', semi)],
        'thicknesses': [t1, t2], 'stop_index': 0}


def _airspaced(r1, r2, t1, g1, gap, r3, r4, t2, g2, semi, wl, aperture):
    return {'wavelength': wl, 'aperture_diameter': aperture, 'surfaces': [
        _surf(r1, t1, 'air', g1, semi),
        _surf(r2, gap, g1, 'air', semi),
        _surf(r3, t2, 'air', g2, semi),
        _surf(r4, 0.0, g2, 'air', semi)],
        'thicknesses': [t1, gap, t2], 'stop_index': 0}


# --- VA: the even ASPHERE, departure on the EXIT surface -------------------
#: 4th- and 6th-order departure on surface 2 [m^-3, m^-5].  The departure is
#: on the EXIT surface (round 3's ``AS`` deforms the ENTRANCE surface), so it
#: is the exit-side ray angles -- what the fold completion's ``zeta`` fit
#: reads -- that are being deformed.  At the 0.65 mm clear semi-aperture the
#: two terms contribute -1.79 um and +9.05 um of sag, a net +7.26 um = 11.5
#: waves at 633 nm, so this is an asphere and not a perturbed sphere.  That
#: the focal locus f(h) acquires an interior stationary point is ASSERTED by
#: ``v3geom.py`` on the running build (``turns_in_focal_locus >= 1``), not
#: assumed from the coefficients.
_VA_A4, _VA_A6 = -1.00e7, 1.20e14
FIX_VA = dict(
    name='VA',
    note='N-BK7 biconvex R = +3.0 / -3.0 mm with an EVEN-ASPHERIC departure '
         'a4 = -1.00e7, a6 = +1.20e14 on the EXIT surface, t = 0.80 mm, '
         '1.30 mm aperture, 633 nm -- non-monotone focal locus',
    prescription=_singlet(3.0e-3, -3.0e-3, 0.80e-3, 0.68e-3, 'N-BK7',
                          633e-9, 1.30e-3,
                          a2={4: _VA_A4, 6: _VA_A6}),
    wavelength=633e-9, N=512, dx=2.30e-6, w0=460e-6,
)

# --- VX: NA >= 0.4 ---------------------------------------------------------
FIX_VX = dict(
    name='VX',
    note='N-SF6 biconvex R = +/-1.85 mm, t = 0.95 mm, 1.10 mm aperture '
         '(NA 0.437), 1.064 um -- the high-NA member',
    prescription=_singlet(1.85e-3, -1.85e-3, 0.95e-3, 0.57e-3, 'N-SF6',
                          1.064e-6, 1.10e-3),
    wavelength=1.064e-6, N=640, dx=1.60e-6, w0=400e-6,
)

# --- VC: HYPERBOLIC conic --------------------------------------------------
FIX_VC = dict(
    name='VC',
    note='N-LAK22 plano-convex CONVEX-first R1 = +2.10 mm with a HYPERBOLIC '
         'conic k1 = -2.20, t = 0.80 mm, 1.20 mm aperture, 850 nm',
    prescription=_singlet(2.10e-3, np.inf, 0.80e-3, 0.62e-3, 'N-LAK22',
                          850e-9, 1.20e-3, k1=-2.20),
    wavelength=850e-9, N=512, dx=2.00e-6, w0=430e-6,
)

# --- VD: AIR-SPACED, NEGATIVE element first --------------------------------
FIX_VD = dict(
    name='VD',
    note='AIR-SPACED pair, DIVERGING element first: N-SF10 -10.0 / +10.0 '
         '(t = 0.45) | gap 0.30 | N-BK7 +1.8 / -4.5 (t = 0.85), 1.20 mm '
         'aperture, 532 nm',
    prescription=_airspaced(-10.0e-3, 10.0e-3, 0.45e-3, 'N-SF10', 0.30e-3,
                            1.8e-3, -4.5e-3, 0.85e-3, 'N-BK7',
                            0.62e-3, 532e-9, 1.20e-3),
    wavelength=532e-9, N=512, dx=2.10e-6, w0=430e-6,
)

# --- re-typed reference optics --------------------------------------------
FIX_V = dict(
    name='V',
    note="VERIFY-B7b's N-BAF10 biconvex R = +/-2.6 mm, t = 0.70 mm, 0.90 mm "
         'aperture, 1.064 um (re-typed here from the published prescription)',
    prescription=_singlet(2.6e-3, -2.6e-3, 0.70e-3, 0.45e-3, 'N-BAF10',
                          1.064e-6, 0.90e-3),
    wavelength=1.064e-6, N=512, dx=2.20e-6, w0=330e-6,
)

FIX_HN = dict(
    name='HN',
    note="WP-B7c round 3's N-LASF9 plano-convex CONVEX-first R1 = +2.40 mm, "
         't = 1.00 mm, 2.30 mm aperture, 1.064 um -- claim 2 is about this '
         'optic (re-typed from the published prescription)',
    prescription=_singlet(2.40e-3, np.inf, 1.00e-3, 1.20e-3, 'N-LASF9',
                          1.064e-6, 2.30e-3),
    wavelength=1.064e-6, N=640, dx=2.00e-6, w0=640e-6,
)

FIX_W = dict(
    name='W',
    note='N-BAK4 plano-convex PLANO-FIRST R1 = inf, R2 = -2.9 mm, '
         't = 0.80 mm, 1.10 mm aperture, 780 nm (re-typed)',
    prescription=_singlet(np.inf, -2.9e-3, 0.80e-3, 0.55e-3, 'N-BAK4',
                          780e-9, 1.10e-3),
    wavelength=780e-9, N=512, dx=2.40e-6, w0=400e-6,
)

FIX_Q = dict(
    name='Q',
    note='cemented doublet N-BK7 +3.0 / -2.0 + N-SF6 -2.0 / -6.0, 1.40 mm '
         "aperture, 1.31 um -- round 2's own oracle-floor row (re-typed)",
    prescription=_cemented(3.0e-3, -2.0e-3, -6.0e-3, 0.90e-3, 0.60e-3,
                           'N-BK7', 'N-SF6', 0.70e-3, 1.31e-6, 1.40e-3),
    wavelength=1.31e-6, N=512, dx=2.60e-6, w0=480e-6,
)


# --- VS: the SLOW control, for the regime E7's own case lives in ----------
FIX_VS = dict(
    name='VS',
    note='N-BK7 plano-convex R1 = +14.0 mm, t = 1.20 mm, 1.50 mm aperture '
         '(NA ~ 0.028), 1.064 um -- a slow optic scanned THROUGH its own '
         "focus, which is the regime E7's own case (a returned field at "
         'oracle fidelity 0.46 with every reading nominal) lives in',
    prescription=_singlet(14.0e-3, np.inf, 1.20e-3, 0.80e-3, 'N-BK7',
                          1.064e-6, 1.50e-3),
    wavelength=1.064e-6, N=512, dx=4.00e-6, w0=500e-6,
    #: 26.8 mm of propagation puts the angular spectrum's band limit, not the
    #: source, in charge of the fine window: at the default width the limit
    #: is 0.054/um against a largest source-to-pixel angle of 0.077/um, so
    #: the far tail would be truncated.  Widened until it clears.
    or_window_mult=3.0,
)


def alt(fx, N, dx, tag='alt'):
    """The same prescription on another grid.

    ``tag`` is part of the NAME, so two regrids of one prescription cannot
    end up sharing a label -- which would silently merge two grids in every
    per-optic table downstream.
    """
    g = dict(fx)
    g['name'] = f"{fx['name']}_{tag}"
    g['N'] = int(N)
    g['dx'] = float(dx)
    return g


FIXTURES = {f['name']: f for f in (FIX_VA, FIX_VX, FIX_VC, FIX_VD,
                                   FIX_V, FIX_HN, FIX_W, FIX_Q, FIX_VS)}
FIXTURES['VA_alt'] = alt(FIX_VA, 640, 1.85e-6)
FIXTURES['VX_alt'] = alt(FIX_VX, 512, 2.00e-6)
FIXTURES['VC_alt'] = alt(FIX_VC, 640, 1.60e-6)
FIXTURES['W_alt'] = alt(FIX_W, 640, 1.90e-6)
FIXTURES['V_alt'] = alt(FIX_V, 400, 2.80e-6)

#: DELIBERATELY COARSE grids.  The fallback route's population is dominated
#: by planes at which the completion DECLINES, and the reason it declines most
#: often near a caustic is that the grid cannot resolve the fold's Airy scale.
#: A population sampled only on well-resolved grids therefore under-samples
#: exactly the regime claim 8 is about, so three of the optics are also run on
#: a grid that is too coarse for their own fold.
FIXTURES['VA_c'] = alt(FIX_VA, 256, 4.60e-6, tag='c')
FIXTURES['HN_c'] = alt(FIX_HN, 320, 4.00e-6, tag='c')
FIXTURES['W_c'] = alt(FIX_W, 256, 4.80e-6, tag='c')

#: the optics this verification derives its own numbers on
OPTICS = ('VA', 'VA_alt', 'VX', 'VX_alt', 'VC', 'VC_alt', 'VD',
          'V', 'V_alt', 'HN', 'W', 'W_alt')

#: the FALLBACK-route enrichment: the slow control and the three coarse grids
FALLBACK_ENRICHMENT = ('VS', 'VA_c', 'HN_c', 'W_c')

#: new to the campaign -- in no fixture module of any earlier round
NEW_TO_THE_CAMPAIGN = ('VA', 'VX', 'VC', 'VD')

PROVENANCE = {
    'VS': 'VERIFY round 3 (new, slow control)',
    'VA_c': 'VERIFY round 3 (new, deliberately coarse grid)',
    'HN_c': 'VERIFY round 3 (new, deliberately coarse grid)',
    'W_c': 'VERIFY round 3 (new, deliberately coarse grid)',
    'VA': 'VERIFY round 3 (new)', 'VA_alt': 'VERIFY round 3 (new)',
    'VX': 'VERIFY round 3 (new)', 'VX_alt': 'VERIFY round 3 (new)',
    'VC': 'VERIFY round 3 (new)', 'VC_alt': 'VERIFY round 3 (new)',
    'VD': 'VERIFY round 3 (new)',
    'V': 'VERIFY-B7b (the fixture the bar is argued on), re-typed',
    'V_alt': 'VERIFY-B7b, second grid, re-typed',
    'HN': 'WP-B7c round 3 (claim 2 is about this optic), re-typed',
    'W': 'VERIFY round 2, re-typed', 'W_alt': 'VERIFY round 2, re-typed',
    'Q': "VERIFY-WP-B7c / round 2's oracle-floor row, re-typed",
}


def input_field(fx, dtype=np.complex128):
    return gauss(fx['N'], fx['dx'], fx['w0'], dtype=dtype)
