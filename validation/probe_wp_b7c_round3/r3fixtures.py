"""WP-B7c round 3 -- the population the bar is re-derived on.

The brief asks for at least TWELVE optics, of which at least three are new to
the campaign: an ASPHERIC singlet, a MENISCUS and a high-NA optic at NA >=
0.35.  What is here is

* the eight the BUILDER derived the 1.06 bar on (``V``, ``S``, ``M``, ``Q``,
  ``F_alt``, ``A``, ``K``, ``G``) -- so the round-3 population CONTAINS the
  population the bar was derived on;
* ``P``, the f/1.2 optic the builder EXCLUDED because the shared oracle's
  Debye ``J0`` form does not hold at ``y_max/z = 0.452``.  Round 2's verifier
  showed the angular spectrum has no such ceiling (it scored ``X`` at
  ``y_max/z = 0.461``), so ``P`` is scored here rather than dropped -- it is
  the far tail of the NA axis and the only optic in the campaign that has
  never been scored at all;
* the five the round-2 VERIFIER added (``W`` plano-first, ``X`` at NA 0.3865,
  ``Y`` oblate-conic, ``Z`` cemented, ``C`` the slow converged control);
* THREE this round adds, below.

That is 16 distinct prescriptions, 21 (prescription, grid) pairs counting the
``*_alt`` grids.

The three new ones:

``AS``  a genuinely ASPHERIC singlet -- an even-aspheric departure
        (``aspheric_coeffs={4: ..., 6: ...}``) on the convex surface of an
        N-BK7 plano-convex.  Neither round used ``aspheric_coeffs`` at all:
        round 2's ``K`` and the verifier's ``Y`` are CONIC, which is a
        two-parameter deformation of a sphere that leaves the ray map
        analytic in ``h^2`` to all orders.  A 4th/6th-order departure changes
        the SIGN of the spherical-aberration gradient across the pupil, so
        the fold ring is not monotone in launch height -- the one shape a
        fold-completion built on a single interior turning point is least
        likely to be right on.
``MC``  a CONCAVE-FIRST positive meniscus.  Round 2's ``M`` is a positive
        meniscus run CONVEX-first (R = +1.6 / +4.2 mm); bending it the other
        way (R = -4.5 / -1.5 mm) puts the steep surface LAST, which moves the
        marginal focus the opposite way and gives an overcorrected fold.
``HN``  a plano-convex N-LASF9 run CONVEX-FIRST at **NA 0.40** -- above the
        verifier's ``X`` (0.3865) and the builder's ``G`` (0.33), and above
        the 0.35 the brief asks for.  A plano-convex at this speed has the
        largest spherical aberration in the campaign, so its fold ring is
        both wide and deep in ``z``.

Grids follow the campaign's convention: ``N`` x ``dx`` covers ~2x the clear
aperture, with an ``*_alt`` grid per optic on which ``N`` and ``dx`` both
change.
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_VAL = os.path.dirname(_HERE)


def _load(relpath, name):
    p = os.path.join(_VAL, *relpath)
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#: the round-2 BUILDER's fixture module (which itself imports VERIFY-WP-B7c's)
BFX = _load(('probe_wp_b7c_round2', 'fixtures.py'), '_b7c3_builder_fixtures')
#: the round-2 VERIFIER's fixture module
VFX = _load(('probe_verify_b7c_round2', 'vfixtures.py'), '_b7c3_verifier_fixtures')

gauss = VFX.gauss
_singlet = VFX._singlet


def _plano_asphere(r1, t, semi, glass, wl, aperture, coeffs):
    """Plano-convex, CONVEX first, with an even-aspheric departure on the
    convex surface (the plano second surface is untouched)."""
    return {'wavelength': wl, 'aperture_diameter': aperture, 'surfaces': [
        {'radius': r1, 'conic': 0.0, 'aspheric_coeffs': dict(coeffs),
         'thickness': t, 'glass_before': 'air', 'glass_after': glass,
         'semi_diameter': semi},
        {'radius': np.inf, 'conic': 0.0, 'thickness': 0.0,
         'glass_before': glass, 'glass_after': 'air',
         'semi_diameter': semi}],
        'thicknesses': [t], 'stop_index': 0}


# --- AS: the ASPHERIC singlet ---------------------------------------------
#: 4th- and 6th-order departure [m^-3, m^-5].  Chosen on THIS round's own
#: geometry probe (``r3geom.py``, ``geom_win.json``), not copied: at the
#: 0.882 mm clear semi-aperture the two terms contribute -6.05 um and
#: +9.41 um of sag against a 2.60 mm base radius (155 um of base sag), a net
#: +3.36 um -- 4.3 waves at 780 nm, so an asphere and not a perturbed sphere.
#: Their opposite signs are the point: the focal locus ``f(h)`` acquires an
#: INTERIOR STATIONARY POINT (``r3geom`` reads one turn in ``f(h)`` where
#: every spherical and conic optic in the campaign reads zero), so the
#: landing map has a second interior caustic and the fold ring is not a
#: single monotone branch in launch height.  That is the geometry a
#: completion built on ONE interior turning point is least likely to be right
#: on, which is why it is in the population.
_AS_A4, _AS_A6 = -1.00e7, 2.00e13
FIX_AS = dict(
    name='AS',
    note='N-BK7 plano-convex CONVEX-first R1 = +2.60 mm with an EVEN-ASPHERIC '
         'departure a4 = -1.00e7, a6 = +2.00e13 on surface 1, t = 0.85 mm, '
         '1.80 mm aperture (NA 0.198), 780 nm -- non-monotone focal locus',
    prescription=_plano_asphere(2.60e-3, 0.85e-3, 0.92e-3, 'N-BK7', 780e-9,
                                1.80e-3, {4: _AS_A4, 6: _AS_A6}),
    wavelength=780e-9, N=512, dx=2.60e-6, w0=620e-6,
)

# --- MC: CONCAVE-FIRST positive meniscus ----------------------------------
FIX_MC = dict(
    name='MC',
    note='N-SK16 CONCAVE-FIRST positive meniscus R = -4.5 / -1.5 mm, '
         't = 0.70 mm, 1.00 mm aperture, 633 nm',
    prescription=_singlet(-4.5e-3, -1.5e-3, 0.70e-3, 0.52e-3, 'N-SK16',
                          633e-9, 1.00e-3),
    wavelength=633e-9, N=512, dx=1.90e-6, w0=350e-6,
)

# --- HN: NA 0.40, above everything either round ran ------------------------
FIX_HN = dict(
    name='HN',
    note='N-LASF9 plano-convex CONVEX-first R1 = +2.40 mm, t = 1.00 mm, '
         '2.30 mm aperture (NA ~ 0.40 -- above the verifier 0.3865 and the '
         'builder 0.33), 1.064 um',
    prescription=_singlet(2.40e-3, np.inf, 1.00e-3, 1.20e-3, 'N-LASF9',
                          1.064e-6, 2.30e-3),
    wavelength=1.064e-6, N=640, dx=2.00e-6, w0=640e-6,
)


def alt(fx, N, dx):
    g = dict(fx)
    g['name'] = fx['name'] + '_alt'
    g['N'] = N
    g['dx'] = dx
    return g


NEW = (FIX_AS, FIX_MC, FIX_HN)

#: every fixture either round shipped, plus the three above.
FIXTURES = {}
for _src in (BFX.FIXTURES, VFX.FIXTURES):
    for _k, _v in _src.items():
        FIXTURES[_k] = _v
for _f in NEW:
    FIXTURES[_f['name']] = _f
FIXTURES['AS_alt'] = alt(FIX_AS, 640, 2.15e-6)
FIXTURES['MC_alt'] = alt(FIX_MC, 400, 2.45e-6)
FIXTURES['HN_alt'] = alt(FIX_HN, 512, 2.55e-6)

#: the 16 DISTINCT prescriptions, in the order the report tables them.
#: ``F`` is ``F_alt``'s prescription on another grid and ``P_alt`` is ``P``'s,
#: so they are grid rows rather than optics.
OPTICS = ('V', 'S', 'M', 'Q', 'F_alt', 'A', 'K', 'G', 'P',
          'W', 'X', 'Y', 'Z', 'C', 'AS', 'MC', 'HN')

#: optics no earlier round has EVER scored against an oracle
NEW_TO_THE_CAMPAIGN = ('AS', 'MC', 'HN', 'P')

#: who added each optic, for the population table
PROVENANCE = {
    'V': 'VERIFY-B7b (builder round 2)', 'S': 'VERIFY-WP-B7c',
    'M': 'VERIFY-WP-B7c', 'Q': 'VERIFY-WP-B7c', 'F_alt': 'VERIFY-WP-B7c',
    'F': 'VERIFY-WP-B7c', 'A': 'WP-B7c round 2', 'K': 'WP-B7c round 2',
    'G': 'WP-B7c round 2',
    'P': 'WP-B7c round 2 (EXCLUDED there -- oracle ceiling)',
    'W': 'VERIFY round 2', 'X': 'VERIFY round 2', 'Y': 'VERIFY round 2',
    'Z': 'VERIFY round 2', 'C': 'VERIFY round 2',
    'AS': 'ROUND 3 (new)', 'MC': 'ROUND 3 (new)', 'HN': 'ROUND 3 (new)',
}


def input_field(fx, dtype=np.complex128):
    return gauss(fx['N'], fx['dx'], fx['w0'], dtype=dtype)
