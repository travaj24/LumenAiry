"""W4: immersed-conjugate pupil coordinates (``compute_pupils``).

(``W4`` here is the niche-audit campaign wave that follows W3-1..W3-4 in
``test_niche_audit_w3_oracles.py`` -- unrelated to the v5.17.1
"wave-4 audit fixes" in ``test_audit_w4_*.py``.)

W3-T2 (commit 1523d8e) restored the SIGN of the exit pupil's terminal
index and recorded -- measured, but explicitly NOT fixed -- that both
pupil positions still dropped its MAGNITUDE whenever the conjugate is
IMMERSED (``xp_z`` -24.161034 mm reported vs -36.647419 mm exact for a
BK7 image space, "ratio exactly 1/n", with the mirror-image defect on
``ep_z`` for a glass object space).  This file is that flagged
finding's own oracle, verdict and pin.

THE CONVENTION (the whole finding, in one sentence)
---------------------------------------------------
A pupil position is a GEOMETRIC z coordinate -- the plane, in real
space and in the real medium, where the chief ray crosses the axis --
and is therefore index-INDEPENDENT as a definition.  The Welford ABCD
algebra ``compute_pupils`` uses works in REDUCED coordinates
``(y, nu = n u)``, where a transfer of geometric length ``t`` through
index ``n`` enters the matrix as the reduced length ``t / n``.  Solving
the imaging condition there therefore returns ``t / n``, and the
geometric coordinate is recovered only by multiplying back by the index
of that terminal medium: ``surfaces[0].glass_before`` for the EP, the
medium the last surface refracts into for the XP.  Pre-fix
``compute_pupils`` returned the REDUCED distances under the name of
signed coordinates -- invisible for air conjugates (``n = 1``), which
is exactly why R-1 and W3-T2 both walked past it.

WHY GEOMETRIC IS THE REQUIRED CONTRACT (consumer audit, done BEFORE
the edit -- the R-1 lesson).  Every consumer of ``ep_z`` / ``xp_z``
advances REAL rays through REAL space by these numbers, and NONE of
them carries a compensating index factor:

* ``analysis/field.py`` -- ``_chief_y_offset`` shifts a launched bundle
  by ``-sin(fa) * ep_z / cos(fa)``, and ``relative_illumination``
  back-propagates each ray by ``t = ep_z / cos(fa)`` along its own
  direction cosines so it lands on ``(px*ep_r, py*ep_r, ep_z)``.  Both
  are ray-geometry in the object medium.
* ``raytrace/ray_fan.py`` -- ``_ep_offset`` = ``-ep_z * tan(fa)``,
  used by the transverse-aberration, OPD and spot entry points.
* ``analysis/image_plane_wfe.py`` -- the reference-sphere radius is
  ``(img_d_m - fod.xp_z) / N_chief`` with ``img_d_m`` a geometric
  image distance, so a reduced ``xp_z`` is not even dimensionally
  admissible; ``best_rms`` inverts the same expression.

Those four modules (plus ``first_order_data``, a pass-through) are the
complete set of ``ep_z`` / ``xp_z`` readers in the library.  The argument
is not merely dimensional: replaying ``ray_fan``'s own aiming recipe
through the real surfaces (``TestW4ConsumerAimingLandsOnTheStop``) puts
the chief on the stop centre to within 0.0023% of the stop radius with
the GEOMETRIC ``ep_z``, and misses by 0.90%-6.8% of it with the REDUCED
one at only 0.2-1.0 deg of field.
``seidel_coefficients`` corroborates the EP side from the inside: it
already launches its chief ray at ``y_0 = -B_pre * n_first * u_0 /
A_pre``, i.e. crossing the axis at ``+B_pre * n_first / A_pre`` -- the
geometric value.  Pre-fix it and ``compute_pupils`` therefore
DISAGREED by exactly ``n_first`` on every immersed-object design
(R-1's Pin 3 only ever exercised air).

VERDICT: wrong, both sides, exactly the reduced/geometric confusion.
Measured against the exact real-ray oracle below (it root-finds real
rays through the real surfaces with ``raytrace.trace`` and reads axis
crossings, sharing no code with the paraxial ABCD path it judges):

    design                    quantity  pre-fix        exact        ratio
    glass_image               xp_z      -23.876267 mm  -36.215523   1/n_img
    glass_object              ep_z       +8.439248 mm  +12.800652   1/n_obj
    both_immersed             ep_z       +6.601947 mm  +10.013833   1/n_obj
    both_immersed             xp_z      -23.752251 mm  -39.136342   1/n_img
    stop_in_glass_immersed    ep_z       +6.981790 mm  +10.589979   1/n_obj
    stop_in_glass_immersed    xp_z      -13.574616 mm  -20.589979   1/n_img
    glass_image_fold          xp_z      -15.211832 mm  -23.073307   1/n_img
    glass_object_fold         ep_z       -5.191646 mm   -7.874689   1/n_obj

with the ratios equal to ``1/n`` to 12 digits (0.659282685426 for
N-BK7, 0.606910363484 for N-SF2), i.e. -34.1% / -39.3% errors.
``both_immersed`` uses N-BK7 object space and N-SF2 image space, so the
two factors are proven INDEPENDENT; ``stop_in_glass_immersed`` puts the
stop in a THIRD medium (N-SF2 between N-BK7 ends), so "use the index at
the stop" is rejected too; the two ``*_fold`` designs put an odd mirror
count on the defective side, proving the magnitude composes
multiplicatively with W3-T2's ``sign_out`` rather than replacing it.

Both RADII are provably index-independent (``ep_radius = |r_stop / A|``
and ``xp_radius = |det(M) r_stop / D|`` are height ratios; no reduced
length survives) and were measured bit-unchanged by the fix, as was
``f/#``.

FIX: ``ep_z = B_pre * n_obj / A_pre`` and ``xp_z = -B_post * n_out /
D_post`` with ``n_out = sign_out * _image_space_index(...)``.  For air
conjugates both indices are exactly ``1.0``, so the multiplication is
an IEEE no-op -- asserted bitwise below on all seven R-1 /
W3-T2 air designs.

Tolerance: ``1e-9`` relative.  The oracle's own floor on these designs
was measured at ``1.3e-12`` relative (worst case over all four
quantities and all eight designs, dominated by the finite-difference
conjugate-height probe), so 1e-9 is ~3 orders above the floor and ~8
orders below every defect above.
"""
from __future__ import annotations

import numpy as np
import pytest

from lumenairy.glass import get_glass_index
from lumenairy.raytrace import (
    RayBundle,
    Surface,
    compute_pupils,
    first_order_data,
    seidel_coefficients,
    system_abcd,
    trace,
)

# NOTE: ``seidel._image_space_index`` is imported lazily inside the two
# tests that need it, NOT at module scope, so that the whole file still
# COLLECTS against a pre-fix build (where the helper does not exist) and
# the value pins below can be seen to fail on their own numbers rather
# than being masked by a collection error.

_WL = 587.5618e-9         # He d-line, the campaign's wavelength
_TOL = 1e-9                # relative tolerance vs the oracle (floor 1.3e-12)
_N_BK7 = get_glass_index('N-BK7', _WL)      # 1.5168000345005885
_N_SF2 = get_glass_index('N-SF2', _WL)      # 1.64768977457945


# ======================================================================
# Exact real-ray oracle -- no paraxial / ABCD code involved
#
# Identical construction to R-1's (tests/unit/test_niche_audit_r1_
# compute_pupils.py), which is the point: it reads the GEOMETRIC axis
# crossing of a real traced ray, in the real medium, so it is the
# definition of the convention under test.  ``trace`` refracts surface 0
# with ``n1 = surfaces[0].glass_before``, so a launched direction
# ``(0, sin th, cos th)`` is a real direction in an immersed object
# space and ``-y0 / u0`` is that space's real z coordinate.
# ======================================================================
def _ray(y0, u0, wl):
    """One meridional ray launched at (0, y0, 0) with slope u0 = tan(theta)."""
    y0 = np.atleast_1d(np.asarray(y0, float))
    th = np.arctan(float(u0))
    z = np.zeros_like(y0)
    return RayBundle(x=z.copy(), y=y0.copy(), z=z.copy(), L=z.copy(),
                     M=np.full_like(y0, np.sin(th)),
                     N=np.full_like(y0, np.cos(th)),
                     wavelength=wl, alive=np.ones_like(y0, bool),
                     opd=z.copy())


def _state_at(surfaces, wl, idx, y0, u0):
    """(height, real slope) of the real ray at surface ``idx``.

    Every design here keeps the stop and the last surface FLAT, so the
    recorded height is the height at that vertex plane, and the recorded
    slope is the real slope in the medium following that surface.
    """
    b = trace(_ray(y0, u0, wl), surfaces, wl).ray_history[idx]
    assert bool(b.alive[0]), f"oracle ray died at surface {idx}"
    return float(b.y[0]), float(b.M[0] / b.N[0])


def _aim_at_stop(surfaces, wl, stop, u0, target):
    """Launch height whose real ray reaches ``y = target`` at the stop."""
    fa = _state_at(surfaces, wl, stop, 0.0, u0)[0] - target
    fb = _state_at(surfaces, wl, stop, 1e-6, u0)[0] - target
    slope = (fb - fa) / 1e-6
    y0 = -fa / slope
    for _ in range(3):     # Newton polish on the real trace (map is affine)
        y0 -= (_state_at(surfaces, wl, stop, y0, u0)[0] - target) / slope
    return y0


def _oracle(surfaces, wl, stop, u0=1e-6, eps=1e-9):
    """Exact (ep_z, ep_radius, xp_z, xp_radius), all GEOMETRIC.

    ``ep_z`` is the axis crossing of the real ray through the stop CENTRE
    extrapolated upstream of surface 0 -- a signed z coordinate in the
    OBJECT medium by construction.  ``xp_z`` is the same ray's axis
    crossing downstream of the last vertex, in the IMAGE medium.  The
    radii are the conjugate heights of a tiny stop-height pencil at
    those planes.
    """
    r_stop = float(surfaces[stop].semi_diameter)
    last = len(surfaces) - 1
    y0c = _aim_at_stop(surfaces, wl, stop, u0, 0.0)
    ep_z = -y0c / u0
    yL, uL = _state_at(surfaces, wl, last, y0c, u0)
    xp_z = -yL / uL
    y0e = _aim_at_stop(surfaces, wl, stop, u0, eps)
    ep_radius = abs(r_stop * (y0e + u0 * ep_z) / eps)
    yLe, uLe = _state_at(surfaces, wl, last, y0e, u0)
    xp_radius = abs(r_stop * (yLe + uLe * xp_z) / eps)
    return dict(ep_z=ep_z, ep_radius=ep_radius, xp_z=xp_z,
                xp_radius=xp_radius)


def _rel(a, b):
    return abs(a - b) / abs(b) if b != 0.0 else abs(a - b)


# ======================================================================
# Immersed designs
# ======================================================================
def _d_glass_image():
    """Air object space, stop in the air gap, LAST surface refracts into
    N-BK7 -- an immersed / cemented-to-detector image space.  XP side."""
    return [
        Surface(radius=45e-3, glass_before='air', glass_after='N-BK7',
                thickness=6e-3, semi_diameter=np.inf),
        Surface(radius=-60e-3, glass_before='N-BK7', glass_after='air',
                thickness=8e-3, semi_diameter=np.inf),
        Surface(radius=np.inf, glass_before='air', glass_after='air',
                thickness=10e-3, semi_diameter=1.5e-3, is_stop=True),
        Surface(radius=80e-3, glass_before='air', glass_after='N-BK7',
                thickness=20e-3, semi_diameter=np.inf),
        Surface(radius=np.inf, glass_before='N-BK7', glass_after='N-BK7',
                thickness=0.0, semi_diameter=np.inf),
    ], 2


def _d_glass_object():
    """N-BK7 OBJECT space (surface 0 is the exit face of a block), air
    image space.  EP side -- the stop's object side is glass."""
    return [
        Surface(radius=70e-3, glass_before='N-BK7', glass_after='air',
                thickness=9e-3, semi_diameter=np.inf),
        Surface(radius=np.inf, glass_before='air', glass_after='air',
                thickness=12e-3, semi_diameter=1.2e-3, is_stop=True),
        Surface(radius=55e-3, glass_before='air', glass_after='N-SF2',
                thickness=4e-3, semi_diameter=np.inf),
        Surface(radius=-90e-3, glass_before='N-SF2', glass_after='air',
                thickness=25e-3, semi_diameter=np.inf),
        Surface(radius=np.inf, glass_before='air', glass_after='air',
                thickness=0.0, semi_diameter=np.inf),
    ], 1


def _d_both_immersed():
    """BOTH conjugates immersed, in DIFFERENT glasses (N-BK7 object,
    N-SF2 image), so the EP and XP factors are independently pinned."""
    return [
        Surface(radius=60e-3, glass_before='N-BK7', glass_after='air',
                thickness=7e-3, semi_diameter=np.inf),
        Surface(radius=np.inf, glass_before='air', glass_after='air',
                thickness=11e-3, semi_diameter=1.4e-3, is_stop=True),
        Surface(radius=50e-3, glass_before='air', glass_after='N-SF2',
                thickness=18e-3, semi_diameter=np.inf),
        Surface(radius=np.inf, glass_before='N-SF2', glass_after='N-SF2',
                thickness=0.0, semi_diameter=np.inf),
    ], 1


def _d_stop_in_glass_immersed():
    """Stop INSIDE glass with both conjugates immersed, and the stop's own
    medium (N-SF2) DIFFERENT from both terminal media (N-BK7): every leg
    adjacent to the stop is in glass, and "use the index at the stop" is
    a distinguishable wrong answer.  Symmetric about the stop (R = +-40
    at equal 6 mm distances, flat index-matched plates outside), so it
    also carries the oracle-free mirror-image check."""
    return [
        Surface(radius=np.inf, glass_before='N-BK7', glass_after='N-BK7',
                thickness=5e-3, semi_diameter=np.inf),
        Surface(radius=40e-3, glass_before='N-BK7', glass_after='N-SF2',
                thickness=6e-3, semi_diameter=np.inf),
        Surface(radius=np.inf, glass_before='N-SF2', glass_after='N-SF2',
                thickness=6e-3, semi_diameter=1.0e-3, is_stop=True),
        Surface(radius=-40e-3, glass_before='N-SF2', glass_after='N-BK7',
                thickness=15e-3, semi_diameter=np.inf),
        Surface(radius=np.inf, glass_before='N-BK7', glass_after='N-BK7',
                thickness=0.0, semi_diameter=np.inf),
    ], 2


def _d_glass_image_fold():
    """Glass image space AND an odd mirror count AFTER the stop: the W4
    magnitude must compose with the W3-T2 sign, not replace it."""
    return [
        Surface(radius=45e-3, glass_before='air', glass_after='N-BK7',
                thickness=6e-3, semi_diameter=np.inf),
        Surface(radius=-60e-3, glass_before='N-BK7', glass_after='air',
                thickness=8e-3, semi_diameter=np.inf),
        Surface(radius=np.inf, glass_before='air', glass_after='air',
                thickness=10e-3, semi_diameter=1.5e-3, is_stop=True),
        Surface(radius=np.inf, glass_before='air', glass_after='air',
                thickness=12e-3, semi_diameter=np.inf, is_mirror=True),
        Surface(radius=80e-3, glass_before='air', glass_after='N-BK7',
                thickness=20e-3, semi_diameter=np.inf),
        Surface(radius=np.inf, glass_before='N-BK7', glass_after='N-BK7',
                thickness=0.0, semi_diameter=np.inf),
    ], 2


def _d_glass_object_fold():
    """Glass object space AND an odd mirror count BEFORE the stop (which
    is the leg W3-T2 signed): the EP magnitude is independent of it."""
    return [
        Surface(radius=70e-3, glass_before='N-BK7', glass_after='air',
                thickness=9e-3, semi_diameter=np.inf),
        Surface(radius=np.inf, glass_before='air', glass_after='air',
                thickness=14e-3, semi_diameter=np.inf, is_mirror=True),
        Surface(radius=np.inf, glass_before='air', glass_after='air',
                thickness=12e-3, semi_diameter=1.2e-3, is_stop=True),
        Surface(radius=55e-3, glass_before='air', glass_after='N-SF2',
                thickness=4e-3, semi_diameter=np.inf),
        Surface(radius=-90e-3, glass_before='N-SF2', glass_after='air',
                thickness=25e-3, semi_diameter=np.inf),
        Surface(radius=np.inf, glass_before='air', glass_after='air',
                thickness=0.0, semi_diameter=np.inf),
    ], 2


def _d_powerless_immersed_pre(t=10e-3):
    """Oracle-free EP discriminator: nothing but ``t`` of HOMOGENEOUS
    N-BK7 (a flat index-matched face, then the stop) ahead of the stop,
    so the stop images to ITSELF and ``ep_z`` must be ``+t`` exactly.
    The reduced convention gives ``+t / n``.  This is the immersed twin
    of R-1b's air discriminator."""
    return [
        Surface(radius=np.inf, glass_before='N-BK7', glass_after='N-BK7',
                thickness=t, semi_diameter=np.inf),
        Surface(radius=np.inf, glass_before='N-BK7', glass_after='N-BK7',
                thickness=15e-3, semi_diameter=2e-3, is_stop=True),
        Surface(radius=50e-3, glass_before='N-BK7', glass_after='air',
                thickness=4e-3, semi_diameter=np.inf),
        Surface(radius=np.inf, glass_before='air', glass_after='air',
                thickness=0.0, semi_diameter=np.inf),
    ], 1


def _d_powerless_immersed_post(t=12e-3):
    """Oracle-free XP discriminator: nothing but ``t`` of homogeneous
    N-BK7 after the stop, so ``xp_z`` must be ``-t`` exactly."""
    return [
        Surface(radius=50e-3, glass_before='air', glass_after='N-BK7',
                thickness=4e-3, semi_diameter=np.inf),
        Surface(radius=np.inf, glass_before='N-BK7', glass_after='N-BK7',
                thickness=t, semi_diameter=1.5e-3, is_stop=True),
        Surface(radius=np.inf, glass_before='N-BK7', glass_after='N-BK7',
                thickness=0.0, semi_diameter=np.inf),
    ], 1


_DESIGNS = {
    'glass_image': _d_glass_image,
    'glass_object': _d_glass_object,
    'both_immersed': _d_both_immersed,
    'stop_in_glass_immersed': _d_stop_in_glass_immersed,
    'glass_image_fold': _d_glass_image_fold,
    'glass_object_fold': _d_glass_object_fold,
}

# Exact real-ray oracle values [m], 12 significant digits.  ``fnum`` is
# ``first_order_data``'s, pinned because the fix must NOT move it (it is
# |efl| / (2 ep_radius) and neither factor carries a reduced length);
# measured identical on the pre-fix and post-fix code.
_EXACT = {
    'glass_image': dict(
        ep_z=+1.466036076913e-02, ep_radius=+1.882248654308e-03,
        xp_z=-3.621552321461e-02, xp_radius=+1.603592053577e-03,
        fnum=+1.126472995783e+01),
    'glass_object': dict(
        ep_z=+1.280065180146e-02, ep_radius=+1.125233079315e-03,
        xp_z=-4.359779276964e-02, xp_radius=+1.584523577793e-03,
        fnum=+3.080912207960e+01),
    'both_immersed': dict(
        ep_z=+1.001383283985e-02, ep_radius=+1.320389321211e-03,
        xp_z=-3.913634187077e-02, xp_radius=+1.632637354412e-03,
        fnum=+5.964514682166e+01),
    'stop_in_glass_immersed': dict(
        ep_z=+1.058997875636e-02, ep_radius=+1.012059448342e-03,
        xp_z=-2.058997875636e-02, xp_radius=+1.012059448342e-03,
        fnum=+7.640018227536e+01),
    'glass_image_fold': dict(
        ep_z=+1.466036076913e-02, ep_radius=+1.882248654308e-03,
        xp_z=-2.307330720069e-02, xp_radius=+1.519633668305e-03,
        fnum=+1.857129710025e+01),
    'glass_object_fold': dict(
        ep_z=-7.874688700517e-03, ep_radius=+1.245995019209e-03,
        xp_z=-4.359779276964e-02, xp_radius=+1.584523577793e-03,
        fnum=+1.695660684908e+01),
}

# Pre-fix ``compute_pupils`` on the same designs, measured on 865e922
# (the reduced distances).  ``_DEFECTIVE`` names which side was wrong on
# each design and which index was dropped there; the other side was
# already exact (air conjugate) and its pre-fix value is the exact one.
_PREFIX = {
    'glass_image': dict(ep_z=+1.466036076912e-02, xp_z=-2.387626739903e-02),
    'glass_object': dict(ep_z=+8.439248094867e-03, xp_z=-4.359779276962e-02),
    'both_immersed': dict(ep_z=+6.601946606059e-03, xp_z=-2.375225147024e-02),
    'stop_in_glass_immersed': dict(ep_z=+6.981789633100e-03,
                                   xp_z=-1.357461648736e-02),
    'glass_image_fold': dict(ep_z=+1.466036076912e-02,
                             xp_z=-1.521183193293e-02),
    'glass_object_fold': dict(ep_z=-5.191645913367e-03,
                              xp_z=-4.359779276962e-02),
}

_DEFECTIVE = {
    'glass_image': {'xp_z': _N_BK7},
    'glass_object': {'ep_z': _N_BK7},
    'both_immersed': {'ep_z': _N_BK7, 'xp_z': _N_SF2},
    'stop_in_glass_immersed': {'ep_z': _N_BK7, 'xp_z': _N_BK7},
    'glass_image_fold': {'xp_z': _N_BK7},
    'glass_object_fold': {'ep_z': _N_BK7},
}

# Radii, measured on the PRE-fix code: the fix must leave them alone.
_PREFIX_RADII = {
    'glass_image': (+1.882248654307e-03, +1.603592053577e-03),
    'glass_object': (+1.125233079316e-03, +1.584523577791e-03),
    'both_immersed': (+1.320389321212e-03, +1.632637354411e-03),
    'stop_in_glass_immersed': (+1.012059448342e-03, +1.012059448342e-03),
    'glass_image_fold': (+1.882248654307e-03, +1.519633668305e-03),
    'glass_object_fold': (+1.245995019208e-03, +1.584523577791e-03),
}

_ALL = list(_DESIGNS)


# ======================================================================
# Oracle-free discriminators: the stop imaged through homogeneous glass
# ======================================================================
class TestW4PowerlessImmersedLegs:
    """No oracle, no ABCD, no ambiguity: with nothing but a homogeneous
    glass block between the stop and the reference vertex, the stop
    images to ITSELF, so the pupil coordinate IS the geometric gap."""

    @pytest.mark.parametrize('t', [4e-3, 10e-3, 25e-3])
    def test_ep_is_the_stop_at_plus_t(self, t):
        surfaces, stop = _d_powerless_immersed_pre(t)
        p = compute_pupils(surfaces, _WL, stop_index=stop)
        assert abs(p.ep_z - t) < 1e-15, (
            f"ep_z = {p.ep_z!r} for a powerless leg of {t} m of N-BK7 ahead "
            f"of the stop; the EP *is* the stop, so ep_z must be +{t} "
            f"(the reduced convention gives {t / _N_BK7!r}).")
        assert abs(p.ep_radius - 2e-3) < 1e-15
        # ... and the discriminator really does discriminate
        assert abs(t / _N_BK7 - t) > 0.3 * t

    @pytest.mark.parametrize('t', [4e-3, 12e-3, 25e-3])
    def test_xp_is_the_stop_at_minus_t(self, t):
        surfaces, stop = _d_powerless_immersed_post(t)
        p = compute_pupils(surfaces, _WL, stop_index=stop)
        assert abs(p.xp_z + t) < 1e-15, (
            f"xp_z = {p.xp_z!r} for a powerless leg of {t} m of N-BK7 behind "
            f"the stop; the XP *is* the stop, so xp_z must be -{t} "
            f"(the reduced convention gives {-t / _N_BK7!r}).")
        assert abs(p.xp_radius - 1.5e-3) < 1e-15

    def test_the_oracle_reproduces_both_closed_forms(self):
        """Oracle self-check: it must land on the same two exact numbers,
        which is what licenses it as ground truth on the designs below."""
        surfaces, stop = _d_powerless_immersed_pre(10e-3)
        assert _rel(_oracle(surfaces, _WL, stop)['ep_z'], 10e-3) < 1e-12
        surfaces, stop = _d_powerless_immersed_post(12e-3)
        assert _rel(_oracle(surfaces, _WL, stop)['xp_z'], -12e-3) < 1e-12


# ======================================================================
# The immersed designs vs the exact real-ray oracle
# ======================================================================
class TestW4ImmersedConjugatesVsExactRealRay:

    @pytest.mark.parametrize('name', _ALL)
    def test_all_four_quantities_match_the_live_oracle(self, name):
        surfaces, stop = _DESIGNS[name]()
        o = _oracle(surfaces, _WL, stop)
        p = compute_pupils(surfaces, _WL, stop_index=stop)
        for key, got in (('ep_z', p.ep_z), ('ep_radius', p.ep_radius),
                         ('xp_z', p.xp_z), ('xp_radius', p.xp_radius)):
            assert _rel(got, o[key]) < _TOL, (
                f"{name}.{key}: compute_pupils gave {got!r}, exact real-ray "
                f"oracle {o[key]!r} (rel {_rel(got, o[key]):.3e}); the "
                f"pre-fix reduced value was "
                f"{_PREFIX[name].get(key, 'n/a')!r}.")

    @pytest.mark.parametrize('name', _ALL)
    def test_pinned_against_hardcoded_oracle_values(self, name):
        """Same numbers hard-coded, so the oracle and the library must
        BOTH change for this to move."""
        surfaces, stop = _DESIGNS[name]()
        p = compute_pupils(surfaces, _WL, stop_index=stop)
        fod = first_order_data(surfaces, _WL, stop_index=stop)
        got = dict(ep_z=p.ep_z, ep_radius=p.ep_radius, xp_z=p.xp_z,
                   xp_radius=p.xp_radius, fnum=fod.fnum)
        for key, want in _EXACT[name].items():
            assert _rel(got[key], want) < _TOL, (
                f"{name}.{key} = {got[key]!r}, pinned value {want!r} "
                f"(rel {_rel(got[key], want):.3e})")

    @pytest.mark.parametrize('name', _ALL)
    def test_prefix_reduced_values_are_rejected(self, name):
        """The 1/n values must not come back."""
        surfaces, stop = _DESIGNS[name]()
        p = compute_pupils(surfaces, _WL, stop_index=stop)
        got = dict(ep_z=p.ep_z, xp_z=p.xp_z)
        for key, n_dropped in _DEFECTIVE[name].items():
            bad = _PREFIX[name][key]
            assert _rel(got[key], bad) > 0.2, (
                f"{name}.{key} = {got[key]!r} is the pre-fix REDUCED value "
                f"{bad!r} (short by n = {n_dropped!r}).")

    @pytest.mark.parametrize('name', _ALL)
    def test_the_defect_ratio_was_exactly_one_over_n(self, name):
        """Mechanism discriminator, not just a magnitude check: the pinned
        pre-fix value times the dropped index must reproduce the exact
        value to roundoff.  Nothing but a missing reduced-to-geometric
        conversion can do that on six designs and three indices."""
        for key, n_dropped in _DEFECTIVE[name].items():
            # pure arithmetic on hard-coded literals (no trace, no matrix
            # product), so 1e-11 is platform-independent here
            recovered = _PREFIX[name][key] * n_dropped
            assert _rel(recovered, _EXACT[name][key]) < 1e-11, (
                f"{name}.{key}: pre-fix {_PREFIX[name][key]!r} * n "
                f"{n_dropped!r} = {recovered!r}, exact "
                f"{_EXACT[name][key]!r} -- the 1/n signature does not hold, "
                f"so the mechanism is not the reduced/geometric confusion.")

    @pytest.mark.parametrize('name', _ALL)
    def test_the_side_with_an_air_conjugate_never_moved(self, name):
        """Four of these designs are immersed on ONE side only, so they
        pin that the two factors are applied PER SIDE: the air side must
        still be bit-identical to the pre-fix expression.  Bit-identity is
        checked against an in-process recomputation from the same
        sub-system matrices (the ``_PREFIX`` literals are 12-digit
        truncations, so they can only be compared at 1e-12)."""
        from lumenairy.raytrace.seidel import _post_stop_abcd, _pre_stop_abcd
        surfaces, stop = _DESIGNS[name]()
        p = compute_pupils(surfaces, _WL, stop_index=stop)
        if 'ep_z' not in _DEFECTIVE[name]:
            M = _pre_stop_abcd(surfaces, _WL, stop)
            pre_fix_ep = float(M[0, 1]) / float(M[0, 0])
            assert pre_fix_ep == p.ep_z, (
                f"{name}.ep_z = {p.ep_z!r} moved off the pre-fix expression "
                f"{pre_fix_ep!r}, but that conjugate is in AIR.")
            assert _rel(p.ep_z, _PREFIX[name]['ep_z']) < 1e-11
        if 'xp_z' not in _DEFECTIVE[name]:
            M = _post_stop_abcd(surfaces, _WL, stop)
            m_post = sum(1 for s in surfaces[stop + 1:]
                         if s.is_mirror and not s.is_coordbrk)
            sign = -1.0 if (m_post % 2) else 1.0
            pre_fix_xp = -float(M[0, 1]) * sign / float(M[1, 1])
            assert pre_fix_xp == p.xp_z, (
                f"{name}.xp_z = {p.xp_z!r} moved off the pre-fix expression "
                f"{pre_fix_xp!r}, but that conjugate is in AIR.")
            assert _rel(p.xp_z, _PREFIX[name]['xp_z']) < 1e-11


# ======================================================================
# What must NOT change: radii, f/#, and every air-immersed system
# ======================================================================
class TestW4RadiiAndFNumberAreIndexIndependent:
    """``ep_radius = |r_stop / A|`` and ``xp_radius = |det(M) r_stop / D|``
    are height RATIOS -- no reduced length survives -- so the fix cannot
    touch them, and ``f/#`` inherits that."""

    @pytest.mark.parametrize('name', _ALL)
    def test_radii_match_the_prefix_code_values(self, name):
        surfaces, stop = _DESIGNS[name]()
        p = compute_pupils(surfaces, _WL, stop_index=stop)
        ep_r, xp_r = _PREFIX_RADII[name]
        # 12-digit literals compared at the campaign's ``_TOL``: these are
        # ABCD / Sellmeier outputs, so a last-bit cross-platform drift is
        # allowed for, while the defect these guard against would be a
        # multiplicative index factor (>= 34%).
        assert _rel(p.ep_radius, ep_r) < _TOL
        assert _rel(p.xp_radius, xp_r) < _TOL

    @pytest.mark.parametrize('name', _ALL)
    def test_fnumber_matches_the_pinned_value(self, name):
        surfaces, stop = _DESIGNS[name]()
        fod = first_order_data(surfaces, _WL, stop_index=stop)
        assert _rel(fod.fnum, _EXACT[name]['fnum']) < _TOL


class TestW4AirConjugatesAreBitIdentical:
    """Constraint (1): no air-immersed system may move by one bit.

    Two independent statements.  First, structural: for an air
    conjugate both W4 factors are exactly ``1.0``, and IEEE-754
    multiplication by exactly 1.0 is the identity, so ``B * n / A`` is
    bit-for-bit ``B / A``.  Second, empirical: recompute the pre-fix
    expression in-process from the module's own sub-system builders and
    require exact float equality on all seven R-1 / W3-T2 air
    designs.
    """

    @staticmethod
    def _air_designs():
        from tests.unit.test_niche_audit_r1_compute_pupils import (
            _d_both_sides,
            _d_front_stop,
            _d_gap,
            _d_powerless_pre_stop,
            _d_stop_in_glass,
        )
        return [
            ('gap0', _d_gap(0.0)),
            ('gap10', _d_gap(10e-3)),
            ('gap40', _d_gap(40e-3)),
            ('both_sides', _d_both_sides()),
            ('stop_in_glass_air_ends', _d_stop_in_glass()),
            ('powerless_pre_stop', _d_powerless_pre_stop()),
            ('front_stop', _d_front_stop()),
        ]

    def test_air_terminal_indices_are_exactly_one(self):
        from lumenairy.raytrace.seidel import _image_space_index
        for name, (surfaces, _stop) in self._air_designs():
            n_obj = abs(float(get_glass_index(surfaces[0].glass_before, _WL)))
            n_img = _image_space_index(surfaces, _WL)
            assert n_obj == 1.0, f"{name}: n_obj = {n_obj!r} is not exactly 1"
            assert n_img == 1.0, f"{name}: n_img = {n_img!r} is not exactly 1"

    def test_prefix_expression_reproduced_exactly(self):
        """In-process bit-identity (immune to cross-platform drift: both
        sides are evaluated here, on the same machine, from the same
        matrices)."""
        from lumenairy.raytrace.seidel import _post_stop_abcd, _pre_stop_abcd
        for name, (surfaces, stop) in self._air_designs():
            p = compute_pupils(surfaces, _WL, stop_index=stop)
            if stop != 0:
                M = _pre_stop_abcd(surfaces, _WL, stop)
                pre_fix_ep = float(M[0, 1]) / float(M[0, 0])
                assert pre_fix_ep == p.ep_z, (
                    f"{name}: ep_z = {p.ep_z!r} but the pre-fix expression "
                    f"gives {pre_fix_ep!r}; air systems must be bit-identical.")
            if stop != len(surfaces) - 1:
                M = _post_stop_abcd(surfaces, _WL, stop)
                m_post = sum(1 for s in surfaces[stop + 1:]
                             if s.is_mirror and not s.is_coordbrk)
                sign = -1.0 if (m_post % 2) else 1.0
                pre_fix_xp = -float(M[0, 1]) * sign / float(M[1, 1])
                assert pre_fix_xp == p.xp_z, (
                    f"{name}: xp_z = {p.xp_z!r} but the pre-fix expression "
                    f"gives {pre_fix_xp!r}; air systems must be bit-identical.")


# ======================================================================
# Which index -- the wrong-index candidates must all be rejected
# ======================================================================
class TestW4WhichIndexIsRestored:

    def test_ep_uses_the_object_space_index_not_the_image_one(self):
        """``both_immersed`` has n_obj = N-BK7 and n_img = N-SF2, so
        swapping them is a distinguishable 8.6% error."""
        surfaces, stop = _d_both_immersed()
        p = compute_pupils(surfaces, _WL, stop_index=stop)
        exact = _EXACT['both_immersed']
        swapped_ep = _PREFIX['both_immersed']['ep_z'] * _N_SF2
        swapped_xp = _PREFIX['both_immersed']['xp_z'] * _N_BK7
        assert _rel(p.ep_z, exact['ep_z']) < _TOL
        assert _rel(p.ep_z, swapped_ep) > 0.05, (
            f"ep_z = {p.ep_z!r} looks like the IMAGE-space index was used "
            f"({swapped_ep!r}); it must be the object-space index.")
        assert _rel(p.xp_z, swapped_xp) > 0.05, (
            f"xp_z = {p.xp_z!r} looks like the OBJECT-space index was used "
            f"({swapped_xp!r}).")

    def test_neither_side_uses_the_index_at_the_stop(self):
        """``stop_in_glass_immersed`` puts the stop in N-SF2 between N-BK7
        conjugates, so "the local index at the stop" is a third,
        distinguishable answer (8.7% off)."""
        surfaces, stop = _d_stop_in_glass_immersed()
        p = compute_pupils(surfaces, _WL, stop_index=stop)
        pre = _PREFIX['stop_in_glass_immersed']
        assert _rel(p.ep_z, pre['ep_z'] * _N_SF2) > 0.05
        assert _rel(p.xp_z, pre['xp_z'] * _N_SF2) > 0.05
        assert _rel(p.ep_z, _EXACT['stop_in_glass_immersed']['ep_z']) < _TOL
        assert _rel(p.xp_z, _EXACT['stop_in_glass_immersed']['xp_z']) < _TOL

    def test_magnitude_composes_with_the_w3_t2_parity_sign(self):
        """The two ``*_fold`` designs put an odd mirror count on the
        defective side.  Pre-fix the SIGN was already right (W3-T2) and
        only the magnitude was missing, so the ratio must still be a
        positive 1/n -- i.e. W4 multiplies ``sign_out``, it does not
        replace it, and a sign regression is caught here."""
        for name in ('glass_image_fold', 'glass_object_fold'):
            surfaces, stop = _DESIGNS[name]()
            p = compute_pupils(surfaces, _WL, stop_index=stop)
            for key, got in (('ep_z', p.ep_z), ('xp_z', p.xp_z)):
                if key not in _DEFECTIVE[name]:
                    continue
                ratio = _PREFIX[name][key] / _EXACT[name][key]
                assert ratio > 0, f"{name}.{key}: pre-fix ratio {ratio} < 0"
                assert abs(ratio - 1.0 / _DEFECTIVE[name][key]) < 1e-11
                assert np.sign(got) == np.sign(_EXACT[name][key])

    def test_image_space_index_follows_system_abcd_bookkeeping(self):
        """``_image_space_index`` unit test.  ``system_abcd`` gives a
        mirror Welford's ``n2 = -n1``, so a mirror-terminated list's
        output medium is ``glass_before`` (magnitude only -- the sign is
        the caller's mirror-parity term).  Refracting and coord-break
        last surfaces use ``glass_after``."""
        from lumenairy.raytrace.seidel import _image_space_index
        refract = [Surface(radius=50e-3, glass_before='air',
                           glass_after='N-SF2')]
        assert _image_space_index(refract, _WL) == abs(_N_SF2)
        mirror = [Surface(radius=50e-3, glass_before='N-BK7',
                          glass_after='air', is_mirror=True)]
        assert _image_space_index(mirror, _WL) == abs(_N_BK7)
        cb = [Surface(glass_before='N-BK7', glass_after='N-BK7',
                      is_coordbrk=True)]
        assert _image_space_index(cb, _WL) == abs(_N_BK7)


# ======================================================================
# The consumer contract, measured end to end
# ======================================================================
class TestW4ConsumerAimingLandsOnTheStop:
    """The convention argument, settled by measurement rather than by
    dimensional analysis.

    ``ray_fan._ep_offset`` launches its chief at ``y0 = -ep_z tan(fa)``
    with slope ``tan(fa)`` (``analysis/field._chief_y_offset`` is the same
    recipe with the ``cos`` factored differently).  By DEFINITION a ray
    through the entrance-pupil centre passes through the STOP CENTRE, so
    this recipe is a direct test of which number the consumer needs.
    Traced through the real surfaces on three immersed designs: the
    GEOMETRIC ``ep_z`` lands the chief within 0.0023% of the stop radius
    (pure residual aberration, no first-order error), while the REDUCED
    value misses by 0.90% to 6.8% of the stop radius at only 0.2-1.0 deg
    of field -- i.e. every off-axis chief ray, every ray fan referenced to
    it, and every ``relative_illumination`` vignetting count was sampling
    the wrong pupil zone on immersed-object systems.
    """

    _DESIGNS_HERE = ('glass_object', 'both_immersed',
                     'stop_in_glass_immersed')

    @pytest.mark.parametrize('name', _DESIGNS_HERE)
    @pytest.mark.parametrize('fa_deg', [0.2, 0.5, 1.0])
    def test_geometric_ep_z_puts_the_chief_on_the_stop_centre(self, name,
                                                              fa_deg):
        surfaces, stop = _DESIGNS[name]()
        p = compute_pupils(surfaces, _WL, stop_index=stop)
        r_stop = float(surfaces[stop].semi_diameter)
        n_obj = abs(float(get_glass_index(surfaces[0].glass_before, _WL)))
        assert n_obj > 1.1, "design is not immersed on the object side"
        u = float(np.tan(np.radians(fa_deg)))
        # ray_fan's own recipe, with the fixed (geometric) coordinate
        y_stop, _ = _state_at(surfaces, _WL, stop, -p.ep_z * u, u)
        assert abs(y_stop) < 1e-3 * r_stop, (
            f"{name} at {fa_deg} deg: the chief aimed through ep_z = "
            f"{p.ep_z!r} misses the stop centre by {y_stop!r} m "
            f"({abs(y_stop) / r_stop:.2%} of the stop radius).")
        # ...and with the pre-fix reduced coordinate it demonstrably misses
        y_stop_reduced, _ = _state_at(surfaces, _WL, stop,
                                      -(p.ep_z / n_obj) * u, u)
        assert abs(y_stop_reduced) > 5e-3 * r_stop, (
            f"{name} at {fa_deg} deg: the REDUCED ep_z "
            f"{p.ep_z / n_obj!r} should visibly miss the stop centre, but "
            f"it lands at {y_stop_reduced!r} m -- this design no longer "
            f"discriminates the two conventions.")


# ======================================================================
# Independent cross-checks (no oracle)
# ======================================================================
class TestW4IndependentCrossChecks:

    def test_symmetric_immersed_design_gives_mirror_image_pupils(self):
        """No oracle and no ABCD: ``stop_in_glass_immersed`` is symmetric
        about the stop (R = +-40 mm at equal 6 mm distances; the outer
        N-BK7 plates are flat and index-matched, hence powerless), so the
        EP and XP must be mirror images ABOUT THE STOP PLANE, with equal
        radii.  Scaling both coordinates by 1/n from their two DIFFERENT
        outer vertices is not a symmetry operation, which is why the
        pre-fix values failed this by 3.4 mm."""
        surfaces, stop = _d_stop_in_glass_immersed()
        p = compute_pupils(surfaces, _WL, stop_index=stop)
        z_stop = sum(float(s.thickness) for s in surfaces[:stop])   # 11 mm
        z_last = sum(float(s.thickness) for s in surfaces[:-1])     # 32 mm
        ep_from_stop = p.ep_z - z_stop
        xp_from_stop = (z_last + p.xp_z) - z_stop
        assert abs(p.ep_radius - p.xp_radius) < 1e-15
        assert abs(ep_from_stop + xp_from_stop) < 1e-12, (
            f"EP sits {ep_from_stop!r} m from the stop but XP sits "
            f"{xp_from_stop!r} m from it; a design symmetric about the stop "
            f"requires equal and opposite offsets.")
        assert abs(ep_from_stop) > 1e-4, "degenerate: both pupils at the stop"
        # the pre-fix pair was asymmetric by 3.4 mm
        pre = _PREFIX['stop_in_glass_immersed']
        assert abs((pre['ep_z'] - z_stop)
                   + (z_last + pre['xp_z'] - z_stop)) > 3e-3

    @pytest.mark.parametrize('name', _ALL)
    def test_seidel_implied_entrance_pupil_agrees(self, name):
        """R-1's Pin 3, extended to immersed conjugates -- and this one
        is a genuine second opinion: ``seidel_coefficients`` builds its
        chief ray from ``nu_0 = n_first * u_0`` and so ALREADY implied the
        geometric EP, meaning the two functions disagreed by exactly
        ``n_first`` on every immersed-object design before this fix."""
        surfaces, stop = _DESIGNS[name]()
        sigma = 1e-3
        sd, _ = seidel_coefficients(surfaces, _WL, stop_index=stop,
                                    field_angle=sigma)
        ep_z_seidel = -float(sd['y_chief'][0]) / sigma
        ep_r_seidel = abs(float(sd['y_marginal'][0]))
        p = compute_pupils(surfaces, _WL, stop_index=stop)
        assert abs(ep_z_seidel - p.ep_z) <= 1e-12 * max(1.0, abs(p.ep_z)), (
            f"{name}: seidel_coefficients implies ep_z = {ep_z_seidel!r} but "
            f"compute_pupils reports {p.ep_z!r}.")
        assert abs(ep_r_seidel - p.ep_radius) <= 1e-15 + 1e-12 * p.ep_radius

    @pytest.mark.parametrize('name', _ALL)
    def test_every_reported_pupil_is_finite_and_physical(self, name):
        """Cheap sanity net: the fix multiplies by an index, so a lookup
        failure would surface as NaN / inf / a sign flip rather than as a
        merely wrong number."""
        surfaces, stop = _DESIGNS[name]()
        p = compute_pupils(surfaces, _WL, stop_index=stop)
        for key, got in (('ep_z', p.ep_z), ('ep_radius', p.ep_radius),
                         ('xp_z', p.xp_z), ('xp_radius', p.xp_radius)):
            assert np.isfinite(got), f"{name}.{key} = {got!r}"
        assert p.ep_radius > 0 and p.xp_radius > 0
        # both radii stay within an order of magnitude of the stop radius
        r_stop = float(surfaces[stop].semi_diameter)
        assert 0.1 * r_stop < p.ep_radius < 10.0 * r_stop
        assert 0.1 * r_stop < p.xp_radius < 10.0 * r_stop


# ======================================================================
# ======================================================================
# W4b: the FOCAL-DISTANCE sibling -- system_abcd's bfl / ffl and
#      FirstOrderData's principal planes were reduced too
# ======================================================================
# ======================================================================
_W4B_ORACLE = """W4b: reduced-vs-geometric focal distances.

(Same file as W4 on purpose: same mechanism, same designs, same oracle
machinery, one layer up.  W4 fixed ``compute_pupils``; the flag it left
behind -- ``system_abcd``'s ``efl``/``bfl``/``ffl`` come straight out of
the REDUCED matrix -- is closed here.)

VERDICT
-------
``bfl`` (``-A/C``) and ``ffl`` (``-D/C``) ARE reduced distances and were
wrong by the terminal index for immersed conjugates.  FIXED:
``bfl = n_img * (-A/C)``, ``ffl = n_obj * (-D/C)``.  Measured [mm]:

    design                  quantity  pre-fix     exact       ratio
    glass_image             bfl       +12.251795  +18.583524  1/n_BK7
    glass_object            ffl       +40.798207  +61.882721  1/n_BK7
    both_immersed           bfl      +171.005538 +281.764076  1/n_SF2
    both_immersed           ffl      +120.783374 +183.204226  1/n_BK7
    stop_in_glass_immersed  bfl      +141.068436 +213.972609  1/n_BK7
    stop_in_glass_immersed  ffl      +147.661263 +223.972609  1/n_BK7
    glass_image_fold        bfl       +71.654969 +108.686259  1/n_BK7
    glass_object_fold       ffl       -28.036251  -42.525386  1/n_BK7

i.e. -34.1% / -39.3%, ratio equal to 1/n to 12 digits.
``FirstOrderData.pp_object_z``/``pp_image_z`` inherited it exactly (they
are ``f - FFL`` and ``BFL - f'``) and are fixed with them.  Ground truth
is the exact real-ray oracle below AND, for ``bfl``, the OPERATIONAL
definition every consumer uses -- ``bfl`` is appended to the surface
list as a thickness, so "the thickness at which the real parallel ray
lands on the axis" is the contract (pinned separately).

``efl`` is NOT converted.  It stays ``-1/C = 1/Phi``, the REDUCED
(equivalent / air-referred) focal length.  That is a decision taken by
consumer audit; three legs:

1. The NA consumers need ``1/Phi``.  ``NA' = r_ep / |1/Phi|`` exactly
   (verified against the real marginal ray on all six immersed designs
   to <= 6.1e-12), so ``ui/model.py``'s ``na = epd / (2 |efl|)``,
   ``ui/spot_field_dock.py``'s Airy radius and ``first_order_data``'s
   ``fnum = |efl| / (2 ep_radius)`` are correct ONLY with the reduced
   value.  ``fnum`` is therefore ``1 / (2 NA')`` -- the standard
   immersed generalisation of f/# -- and is UNCHANGED by this fix:
   11.264729957829 vs 11.264729957791 on ``glass_image``, where the
   air-only ``f'/D_ep`` would read 17.086342789.  W4's first draft of
   the flag called ``fnum`` defective on exactly that basis; the
   measurement retracts it, and the retraction is pinned below.
2. ``analysis/image_plane_wfe.py`` solves ``1/v = 1/efl - 1/u``, the
   Gauss equation in REDUCED distances.
3. THE ALGEBRA TWIN settles it structurally (the S11 lockstep lesson).
   ``algebra/from_prescription`` emits ``FreeSpace(t / n)`` +
   ``ThinLens(1/phi)``, i.e. it builds the SAME reduced matrix, and
   ``Operator.efl`` is documented as ``-1/C`` "matching system_abcd".  A
   ``CompositeOperator`` has folded the media into its reduced lengths
   and keeps NO terminal-index information, so it structurally CANNOT
   produce ``n'/Phi`` or a geometric ``bfl``.  Redefining ``efl`` would
   break that documented lockstep with no way for the twin to follow;
   ``bfl``/``ffl`` are not exposed by the twin at all, so making THEM
   geometric breaks no lockstep.  Both halves are pinned below, on an
   IMMERSED prescription.

FRAMES (the trap -- and a retracted error of my own)
----------------------------------------------------
``bfl``/``ffl`` are reported in the UNFOLDED (along-the-ray) frame:
S11-1's oracle (``test_niche_s11_sibling_deferred._exact_bfl_unfolded``)
maps the traced global-z axis crossing into it with an explicit
``(-1) ** n_mirrors``, and its consumers un-map it the same way when
they use ``bfl`` as a THICKNESS.  ``xp_z``/``ep_z`` (W3-T2, W4) are
global-z coordinates and DO carry the mirror-parity sign.  So the
conversion here needs the index MAGNITUDE only: the global-z distance is
``sign * n_img * (-A/C)``, and mapping it back into the unfolded frame
multiplies by ``sign`` again.  A first draft of this fix applied the
sign here too and broke seven S11-1 pins (measured: a single concave
mirror R = -100 mm has its focus at global z -100.000000 mm and unfolded
``bfl`` +100.000000 mm).  Category error, retracted;
``test_w4b_frame_convention_is_the_unfolded_one`` pins it so it cannot
come back.

Consequence: air-to-air systems are bit-identical at ANY mirror count
(both factors are exactly 1.0) -- asserted bitwise below on the seven
R-1/W3-T2 air designs plus four air MIRROR controls (0-2 mirrors, both
post-mirror thickness sign conventions).

Oracle floor: 5.3e-11 relative (worst over 6 designs x 4 quantities; the
``ffl`` root-find is the noisy one), so ``_TOL = 1e-9`` is ~19x above the
floor and ~7 orders below every defect above.
"""

# Independent local re-implementations of the three conversion factors.
# The ORACLE must not import the code it judges (the campaign rule), and
# keeping them local also lets every value pin below fail on its own
# NUMBERS against a pre-fix build rather than on an ImportError.  The
# library helpers are imported ONLY by the tests that unit-test them.
def _loc_parity_sign(surfaces):
    m = sum(1 for s in surfaces if s.is_mirror and not s.is_coordbrk)
    return -1.0 if (m % 2) else 1.0


def _loc_n_obj(surfaces, wl=None):
    return abs(float(get_glass_index(surfaces[0].glass_before, wl or _WL)))


def _loc_n_img(surfaces, wl=None):
    last = surfaces[-1]
    glass = (last.glass_before if (last.is_mirror and not last.is_coordbrk)
             else last.glass_after)
    return abs(float(get_glass_index(glass, wl or _WL)))


_HP = 1e-7          # parallel-ray launch height for the focal oracle
_U0F = 1e-6         # slope for the exit-parallel root-find


def _oracle_focal(surfaces, wl):
    """Exact real-ray focal distances, in ``system_abcd``'s own frame.

    * ``bfl`` -- parallel ray in (u = 0, height ``_HP``); its axis
      crossing after the last surface, mapped into the unfolded frame.
    * ``f'``  -- ``-h / u_out`` for that ray, same mapping.
    * ``H'``  -- where the OUTGOING ray reaches the incoming height.
    * ``ffl`` -- root-find the input height whose ray EXITS parallel; the
      front focal point is that input ray's axis crossing.  Object-side,
      so NO frame mapping (the input frame has parity 0).
    * ``H``   -- where the INCOMING ray reaches the outgoing height.
    """
    last = len(surfaces) - 1
    s = _loc_parity_sign(surfaces)
    y_L, u_L = _state_at(surfaces, wl, last, _HP, 0.0)
    bfl = s * (-y_L / u_L)
    f_prime = s * (-_HP / u_L)
    pp_image_z = s * ((_HP - y_L) / u_L)
    _, ua = _state_at(surfaces, wl, last, 0.0, _U0F)
    _, ub = _state_at(surfaces, wl, last, 1e-7, _U0F)
    slope = (ub - ua) / 1e-7
    y0 = -ua / slope
    for _ in range(3):
        y0 -= _state_at(surfaces, wl, last, y0, _U0F)[1] / slope
    yL2, uL2 = _state_at(surfaces, wl, last, y0, _U0F)
    assert abs(uL2) < 1e-13, f"oracle: exit not parallel (u={uL2!r})"
    ffl = y0 / _U0F
    pp_object_z = (yL2 - y0) / _U0F
    return dict(bfl=bfl, ffl=ffl, f_prime=f_prime, pp_image_z=pp_image_z,
                pp_object_z=pp_object_z, u_L=u_L)


def _append_flat(surfaces, T):
    """``surfaces`` + a flat index-matched plane ``T`` further on.

    Exactly what ``analysis``'s ``_append_image_plane`` and
    ``ui``'s ``surfs[-1].thickness = bfl`` do, which is why it is the
    operational contract for ``bfl``.
    """
    last = surfaces[-1]
    glass = (last.glass_before if (last.is_mirror and not last.is_coordbrk)
             else last.glass_after)
    out = [Surface(radius=s.radius, conic=s.conic,
                   semi_diameter=s.semi_diameter,
                   glass_before=s.glass_before, glass_after=s.glass_after,
                   is_mirror=s.is_mirror, is_stop=s.is_stop,
                   thickness=(float(T) if i == len(surfaces) - 1
                              else s.thickness),
                   is_coordbrk=s.is_coordbrk)
           for i, s in enumerate(surfaces)]
    out.append(Surface(radius=np.inf, glass_before=glass, glass_after=glass,
                       thickness=0.0, semi_diameter=np.inf))
    return out


def _bfl_operational(surfaces, wl, scale):
    """The thickness at which the real parallel ray lands on the axis.

    ``y_image(T)`` is affine in ``T``, so two probes plus a Newton polish
    is exact.  Returned in GLOBAL z (thickness semantics), i.e. it must be
    compared with ``(-1) ** n_mirrors * bfl``.
    """
    s = abs(scale) if abs(scale) > 1e-6 else 1e-3
    n = len(surfaces)
    y0 = _state_at(_append_flat(surfaces, 0.0), wl, n, _HP, 0.0)[0]
    y1 = _state_at(_append_flat(surfaces, s), wl, n, _HP, 0.0)[0]
    slope = (y1 - y0) / s
    T = -y0 / slope
    for _ in range(3):
        T -= _state_at(_append_flat(surfaces, T), wl, n, _HP, 0.0)[0] / slope
    return T


# Post-fix values == the exact oracle to <= 5.3e-11, 12 significant
# digits.  ``f_prime``/``f_object`` are ``n_img * efl`` / ``n_obj * efl``.
_EXACT_FOCAL = {
    'glass_image': dict(
        bfl=+1.858352382661e-02, ffl=+3.511459421589e-02,
        pp_image_z=-4.573796761541e-02, pp_object_z=+7.291451392618e-03,
        efl=+4.240604560851e-02, f_prime=+6.432149144202e-02,
        f_object=+4.240604560851e-02, fnum=+1.126472995783e+01),
    'glass_object': dict(
        bfl=+5.403776792272e-02, ffl=+6.188272127217e-02,
        pp_image_z=-1.529711869456e-02, pp_object_z=+4.328443714100e-02,
        efl=+6.933488661727e-02, f_prime=+6.933488661727e-02,
        f_object=+1.051671584132e-01, fnum=+3.080912207960e+01),
    'both_immersed': dict(
        bfl=+2.817640762774e-01, ffl=+1.832042262387e-01,
        pp_image_z=+2.223706977438e-02, pp_object_z=+5.570638575325e-02,
        efl=+1.575096298509e-01, f_prime=+2.595270065031e-01,
        f_object=+2.389106119920e-01, fnum=+5.964514682166e+01),
    'stop_in_glass_immersed': dict(
        bfl=+2.139726088441e-01, ffl=+2.239726088441e-01,
        pp_image_z=-2.058997875636e-02, pp_object_z=+1.058997875636e-02,
        efl=+1.546430526537e-01, f_prime=+2.345625876004e-01,
        f_object=+2.345625876004e-01, fnum=+7.640018227536e+01),
    'glass_image_fold': dict(
        bfl=+1.086862588571e-01, ffl=+7.193354268866e-02,
        pp_image_z=+2.644344672449e-03, pp_object_z=-2.021944737293e-03,
        efl=+6.991159795136e-02, f_prime=+1.060419141846e-01,
        f_object=+6.991159795136e-02, fnum=+1.857129710025e+01),
    'glass_object_fold': dict(
        bfl=-1.013849393378e-02, ffl=-4.252538633442e-02,
        pp_image_z=+3.211720141947e-02, pp_object_z=-2.156805383522e-02,
        efl=-4.225569535324e-02, f_prime=-4.225569535324e-02,
        f_object=-6.409344016965e-02, fnum=+1.695660684908e+01),
}

# Pre-fix (reduced) values, measured on 8fdcccc.
_PREFIX_FOCAL = {
    'glass_image': dict(
        bfl=+1.225179549309e-02, ffl=+3.511459421589e-02,
        pp_image_z=-3.015425011542e-02, pp_object_z=+7.291451392618e-03),
    'glass_object': dict(
        bfl=+5.403776792272e-02, ffl=+4.079820666180e-02,
        pp_image_z=-1.529711869456e-02, pp_object_z=+2.853667995548e-02),
    'both_immersed': dict(
        bfl=+1.710055379505e-01, ffl=+1.207833742561e-01,
        pp_image_z=+1.349590809961e-02, pp_object_z=+3.672625559479e-02),
    'stop_in_glass_immersed': dict(
        bfl=+1.410684361664e-01, ffl=+1.476612630206e-01,
        pp_image_z=-1.357461648736e-02, pp_object_z=+6.981789633100e-03),
    'glass_image_fold': dict(
        bfl=+7.165496860821e-02, ffl=+7.193354268866e-02,
        pp_image_z=+1.743370656845e-03, pp_object_z=-2.021944737293e-03),
    'glass_object_fold': dict(
        bfl=-1.013849393378e-02, ffl=-2.803625090134e-02,
        pp_image_z=+3.211720141947e-02, pp_object_z=-1.421944445190e-02),
}

# Which quantity was reduced on which design, and by which index.
# ``bfl``/``pp_image_z`` take the IMAGE index, ``ffl``/``pp_object_z``
# the OBJECT one; a design whose conjugate on that side is AIR is
# unaffected there, which is what pins the per-side application.
_DEFECTIVE_FOCAL = {
    'glass_image': {'bfl': _N_BK7, 'pp_image_z': _N_BK7},
    'glass_object': {'ffl': _N_BK7, 'pp_object_z': _N_BK7},
    'both_immersed': {'bfl': _N_SF2, 'pp_image_z': _N_SF2,
                      'ffl': _N_BK7, 'pp_object_z': _N_BK7},
    'stop_in_glass_immersed': {'bfl': _N_BK7, 'pp_image_z': _N_BK7,
                               'ffl': _N_BK7, 'pp_object_z': _N_BK7},
    'glass_image_fold': {'bfl': _N_BK7, 'pp_image_z': _N_BK7},
    'glass_object_fold': {'ffl': _N_BK7, 'pp_object_z': _N_BK7},
}

_FOCAL_KEYS = ('bfl', 'ffl', 'pp_image_z', 'pp_object_z')


def _focal_report(surfaces, stop):
    """``{bfl, ffl, pp_image_z, pp_object_z, efl, fnum}`` as reported."""
    _M, efl, bfl, ffl = system_abcd(surfaces, _WL)
    fod = first_order_data(surfaces, _WL, stop_index=stop)
    return dict(bfl=bfl, ffl=ffl, pp_image_z=fod.pp_image_z,
                pp_object_z=fod.pp_object_z, efl=efl, fnum=fod.fnum)


# ----------------------------------------------------------------------
# air mirror controls (0-2 mirrors, both post-mirror thickness signs)
# ----------------------------------------------------------------------
def _d_mirror_concave(t=50e-3, R=-100e-3):
    """Single concave mirror in air -- ``system_abcd``'s own docstring
    case (``efl = +50 mm`` is "the conventional answer")."""
    return [
        Surface(radius=R, glass_before='air', glass_after='air',
                thickness=t, semi_diameter=10e-3, is_mirror=True,
                is_stop=True),
        Surface(radius=np.inf, glass_before='air', glass_after='air',
                thickness=0.0, semi_diameter=np.inf),
    ], 0


def _d_two_mirrors():
    """Flat fold + concave mirror: EVEN parity (no frame flip)."""
    return [
        Surface(radius=np.inf, glass_before='air', glass_after='air',
                thickness=20e-3, semi_diameter=10e-3, is_mirror=True,
                is_stop=True),
        Surface(radius=-100e-3, glass_before='air', glass_after='air',
                thickness=50e-3, semi_diameter=np.inf, is_mirror=True),
        Surface(radius=np.inf, glass_before='air', glass_after='air',
                thickness=0.0, semi_diameter=np.inf),
    ], 0


def _d_singlet_then_fold(t_sign=+1.0):
    """Air singlet then a flat fold: ODD parity, no immersion.  ``t_sign``
    exercises both post-mirror thickness conventions."""
    return [
        Surface(radius=50e-3, glass_before='air', glass_after='N-BK7',
                thickness=5e-3, semi_diameter=2e-3, is_stop=True),
        Surface(radius=-50e-3, glass_before='N-BK7', glass_after='air',
                thickness=10e-3, semi_diameter=np.inf),
        Surface(radius=np.inf, glass_before='air', glass_after='air',
                thickness=t_sign * 15e-3, semi_diameter=np.inf,
                is_mirror=True),
        Surface(radius=np.inf, glass_before='air', glass_after='air',
                thickness=0.0, semi_diameter=np.inf),
    ], 0


_AIR_MIRROR_DESIGNS = {
    'mirror_concave': _d_mirror_concave,
    'two_mirrors_even': _d_two_mirrors,
    'singlet_then_fold': _d_singlet_then_fold,
    'singlet_then_fold_negt': lambda: _d_singlet_then_fold(-1.0),
}


# ======================================================================
# W4b vs the exact real-ray oracle
# ======================================================================
class TestW4bFocalDistancesVsExactRealRay:

    @pytest.mark.parametrize('name', _ALL)
    def test_all_four_quantities_match_the_live_oracle(self, name):
        surfaces, stop = _DESIGNS[name]()
        o = _oracle_focal(surfaces, _WL)
        got = _focal_report(surfaces, stop)
        for key in _FOCAL_KEYS:
            assert _rel(got[key], o[key]) < _TOL, (
                f"{name}.{key}: reported {got[key]!r}, exact real-ray "
                f"oracle {o[key]!r} (rel {_rel(got[key], o[key]):.3e}); the "
                f"pre-fix reduced value was {_PREFIX_FOCAL[name][key]!r}.")

    @pytest.mark.parametrize('name', _ALL)
    def test_pinned_against_hardcoded_oracle_values(self, name):
        surfaces, stop = _DESIGNS[name]()
        got = _focal_report(surfaces, stop)
        for key, want in _EXACT_FOCAL[name].items():
            if key in ('f_prime', 'f_object'):
                continue          # covered by the focal-length test below
            assert _rel(got[key], want) < _TOL, (
                f"{name}.{key} = {got[key]!r}, pinned {want!r} "
                f"(rel {_rel(got[key], want):.3e})")

    @pytest.mark.parametrize('name', _ALL)
    def test_prefix_reduced_values_are_rejected(self, name):
        surfaces, stop = _DESIGNS[name]()
        got = _focal_report(surfaces, stop)
        for key, n_dropped in _DEFECTIVE_FOCAL[name].items():
            bad = _PREFIX_FOCAL[name][key]
            assert _rel(got[key], bad) > 0.2, (
                f"{name}.{key} = {got[key]!r} is the pre-fix REDUCED value "
                f"{bad!r} (short by n = {n_dropped!r}).")

    @pytest.mark.parametrize('name', _ALL)
    def test_the_defect_ratio_was_exactly_one_over_n(self, name):
        """Mechanism, not just magnitude: pinned pre-fix value times the
        dropped index reproduces the exact value to roundoff.  Pure
        arithmetic on hard-coded literals, so platform-independent."""
        for key, n_dropped in _DEFECTIVE_FOCAL[name].items():
            recovered = _PREFIX_FOCAL[name][key] * n_dropped
            assert _rel(recovered, _EXACT_FOCAL[name][key]) < 1e-11, (
                f"{name}.{key}: pre-fix {_PREFIX_FOCAL[name][key]!r} * n "
                f"{n_dropped!r} = {recovered!r} vs exact "
                f"{_EXACT_FOCAL[name][key]!r} -- the 1/n signature does not "
                f"hold, so the mechanism is not reduced-vs-geometric.")

    @pytest.mark.parametrize('name', _ALL)
    def test_the_side_with_an_air_conjugate_never_moved(self, name):
        """Per-side application: ``bfl``/``pp_image_z`` take the image
        index and ``ffl``/``pp_object_z`` the object one, so a design
        immersed on ONE side only must be bit-identical on the other."""
        surfaces, stop = _DESIGNS[name]()
        M, efl, bfl, ffl = system_abcd(surfaces, _WL)
        A, C, D = float(M[0, 0]), float(M[1, 0]), float(M[1, 1])
        got = _focal_report(surfaces, stop)
        if 'bfl' not in _DEFECTIVE_FOCAL[name]:
            assert bfl == -A / C, (
                f"{name}: bfl {bfl!r} != the pre-fix -A/C {-A / C!r} even "
                f"though image space is AIR.")
            assert got['pp_image_z'] == bfl - efl
        if 'ffl' not in _DEFECTIVE_FOCAL[name]:
            assert ffl == -D / C, (
                f"{name}: ffl {ffl!r} != the pre-fix -D/C {-D / C!r} even "
                f"though object space is AIR.")
            assert got['pp_object_z'] == efl - ffl

    @pytest.mark.parametrize('name', _ALL)
    def test_focal_lengths_are_recoverable_from_the_reduced_efl(self, name):
        """``f' = n_img * efl`` and ``f = n_obj * efl`` -- the pieces a
        caller needs once ``efl`` is documented as reduced.  ``f'`` is
        checked against the real parallel ray's ``-h / u_out``."""
        surfaces, _stop = _DESIGNS[name]()
        _M, efl, _b, _f = system_abcd(surfaces, _WL)
        n_img = _loc_n_img(surfaces)
        n_obj = _loc_n_obj(surfaces)
        o = _oracle_focal(surfaces, _WL)
        assert _rel(n_img * efl, o['f_prime']) < _TOL, (
            f"{name}: n_img*efl = {n_img * efl!r} but the real parallel "
            f"ray gives f' = {o['f_prime']!r}")
        assert _rel(n_img * efl, _EXACT_FOCAL[name]['f_prime']) < _TOL
        assert _rel(n_obj * efl, _EXACT_FOCAL[name]['f_object']) < _TOL
        # and the two-sided relation f'/f == n'/n
        if abs(n_obj * efl) > 1e-12:
            assert _rel((n_img * efl) / (n_obj * efl), n_img / n_obj) < 1e-11


# ======================================================================
# The operational contract: bfl is consumed as a THICKNESS
# ======================================================================
class TestW4bOperationalBflContract:
    """``bfl`` is fed straight back into the surface list as a thickness
    (``analysis``'s ``_append_image_plane(surfaces, bfl)``, ``ui``'s
    ``surfs[-1].thickness = bfl``), so the convention-free ground truth
    is "the thickness at which the real parallel ray lands on the axis".
    Thickness is a GLOBAL-z step in ``trace`` while ``bfl`` is reported in
    the unfolded frame, so the comparison carries S11-1's
    ``(-1) ** n_mirrors`` -- exactly as
    ``test_niche_s11_sibling_deferred`` does when it writes
    ``surf[-1].thickness = (-1.0) ** n_mir * fod.bfl``.
    """

    @pytest.mark.parametrize('name', _ALL)
    def test_immersed_designs(self, name):
        surfaces, _stop = _DESIGNS[name]()
        _M, _efl, bfl, _ffl = system_abcd(surfaces, _WL)
        s = _loc_parity_sign(surfaces)
        T = _bfl_operational(surfaces, _WL, bfl)
        assert _rel(s * bfl, T) < _TOL, (
            f"{name}: appending (-1)**m * bfl = {s * bfl!r} as a thickness "
            f"does not land the parallel ray on the axis; the operational "
            f"thickness is {T!r} (rel {_rel(s * bfl, T):.3e}).  Pre-fix "
            f"bfl was {_PREFIX_FOCAL[name]['bfl']!r}.")
        # the pre-fix value demonstrably did NOT satisfy this contract
        if 'bfl' in _DEFECTIVE_FOCAL[name]:
            assert _rel(s * _PREFIX_FOCAL[name]['bfl'], T) > 0.2

    @pytest.mark.parametrize('name', sorted(_AIR_MIRROR_DESIGNS))
    def test_air_mirror_controls(self, name):
        surfaces, _stop = _AIR_MIRROR_DESIGNS[name]()
        _M, _efl, bfl, _ffl = system_abcd(surfaces, _WL)
        s = _loc_parity_sign(surfaces)
        T = _bfl_operational(surfaces, _WL, bfl)
        assert abs(s * bfl - T) < 1e-9 * max(1e-3, abs(T)), (
            f"{name}: (-1)**m * bfl = {s * bfl!r} vs operational {T!r}")


class TestW4bFrameConvention:
    """Pins the frame so the retracted category error cannot return."""

    @pytest.mark.parametrize('name', ['mirror_concave', 'singlet_then_fold',
                                      'singlet_then_fold_negt'])
    def test_w4b_frame_convention_is_the_unfolded_one(self, name):
        """For ODD mirror parity, ``bfl`` must have the OPPOSITE sign to
        the global-z axis crossing -- that is what "unfolded frame" means
        and what S11-1's oracle asserts.  A first draft of W4b applied
        ``_mirror_parity_sign`` to ``bfl`` and broke seven S11-1 pins."""
        surfaces, _stop = _AIR_MIRROR_DESIGNS[name]()
        _M, _efl, bfl, _ffl = system_abcd(surfaces, _WL)
        T = _bfl_operational(surfaces, _WL, bfl)          # global z
        assert bfl * T < 0, (
            f"{name}: bfl {bfl!r} and the global-z focus {T!r} have the "
            f"SAME sign; the unfolded-frame convention has been broken.")
        assert abs(abs(bfl) - abs(T)) < 1e-9 * abs(T)

    def test_even_parity_needs_no_mapping(self):
        surfaces, _stop = _AIR_MIRROR_DESIGNS['two_mirrors_even']()
        _M, _efl, bfl, _ffl = system_abcd(surfaces, _WL)
        T = _bfl_operational(surfaces, _WL, bfl)
        assert bfl * T > 0 and abs(bfl - T) < 1e-9 * abs(T)


# ======================================================================
# What must NOT change
# ======================================================================
class TestW4bAirIsBitIdentical:
    """Air-to-air systems must not move by one bit -- at ANY mirror count,
    since the conversion is magnitude-only.

    Structural: both factors are exactly ``1.0`` and IEEE multiplication
    by 1.0 is the identity.  Empirical: recompute the pre-fix expressions
    in-process from the returned matrix and require exact equality.
    """

    @staticmethod
    def _air_designs():
        return (TestW4AirConjugatesAreBitIdentical._air_designs()
                + [(k, v()) for k, v in sorted(_AIR_MIRROR_DESIGNS.items())])

    def test_both_terminal_indices_are_exactly_one(self):
        from lumenairy.raytrace.seidel import _image_space_index, _object_space_index
        for name, (surfaces, _stop) in self._air_designs():
            assert _object_space_index(surfaces, _WL) == 1.0, name
            assert _image_space_index(surfaces, _WL) == 1.0, name

    def test_prefix_expressions_reproduced_exactly(self):
        for name, (surfaces, stop) in self._air_designs():
            M, efl, bfl, ffl = system_abcd(surfaces, _WL)
            A, C, D = float(M[0, 0]), float(M[1, 0]), float(M[1, 1])
            if abs(C) <= 1e-30:
                continue
            assert efl == -1.0 / C, f"{name}: efl moved"
            assert bfl == -A / C, (
                f"{name}: bfl = {bfl!r} but the pre-fix expression gives "
                f"{-A / C!r}; air systems must be bit-identical.")
            assert ffl == -D / C, (
                f"{name}: ffl = {ffl!r} but the pre-fix expression gives "
                f"{-D / C!r}.")
            fod = first_order_data(surfaces, _WL, stop_index=stop)
            assert fod.pp_object_z == efl - ffl, f"{name}: pp_object_z moved"
            assert fod.pp_image_z == bfl - efl, f"{name}: pp_image_z moved"

    def test_find_paraxial_focus_tracks_bfl(self):
        """It is a one-liner over ``system_abcd``; pin that it still is,
        so the geometric fix reaches every ``find_paraxial_focus`` caller
        (the ui image-plane placement path) too."""
        from lumenairy.raytrace import find_paraxial_focus
        for name in _ALL:
            surfaces, _stop = _DESIGNS[name]()
            _M, _efl, bfl, _ffl = system_abcd(surfaces, _WL)
            assert find_paraxial_focus(surfaces, _WL) == bfl, name


class TestW4bEflConventionAndFNumber:
    """``efl`` stays reduced; ``fnum`` stays ``1/(2 NA')``."""

    @pytest.mark.parametrize('name', _ALL)
    def test_efl_is_untouched_and_equals_minus_one_over_C(self, name):
        surfaces, _stop = _DESIGNS[name]()
        M, efl, _b, _f = system_abcd(surfaces, _WL)
        assert efl == -1.0 / float(M[1, 0])
        assert _rel(efl, _EXACT_FOCAL[name]['efl']) < _TOL

    @pytest.mark.parametrize('name', _ALL)
    def test_fnum_is_one_over_twice_the_image_space_NA(self, name):
        """The measurement that retracts W4's "fnum is wrong" flag: the
        reported f/# equals ``1/(2 NA')`` computed from the REAL marginal
        ray, on every immersed design."""
        surfaces, stop = _DESIGNS[name]()
        fod = first_order_data(surfaces, _WL, stop_index=stop)
        p = compute_pupils(surfaces, _WL, stop_index=stop)
        n_img = _loc_n_img(surfaces)
        o = _oracle_focal(surfaces, _WL)
        na = abs(n_img * o['u_L']) * (p.ep_radius / _HP)
        assert _rel(fod.fnum, 1.0 / (2.0 * na)) < _TOL, (
            f"{name}: fnum {fod.fnum!r} != 1/(2 NA') "
            f"{1.0 / (2.0 * na)!r} (NA' = {na!r})")
        assert _rel(fod.fnum, _EXACT_FOCAL[name]['fnum']) < _TOL

    def test_the_air_only_f_over_D_formula_is_rejected(self):
        """``f'/D_ep`` is the AIR-ONLY f/#; on a BK7 image space it reads
        17.086 where ``1/(2 NA')`` reads 11.265.  Pinning the difference
        keeps the retraction honest -- if someone "fixes" fnum to
        ``f'/D_ep`` this fails."""
        surfaces, stop = _DESIGNS['glass_image']()
        fod = first_order_data(surfaces, _WL, stop_index=stop)
        p = compute_pupils(surfaces, _WL, stop_index=stop)
        n_img = _loc_n_img(surfaces)
        f_over_d = abs(n_img * fod.efl) / (2.0 * p.ep_radius)
        assert _rel(f_over_d, 1.708634278862e+01) < _TOL
        assert _rel(fod.fnum, 1.126472995783e+01) < _TOL
        assert _rel(fod.fnum, f_over_d) > 0.3


# ======================================================================
# The algebra twin (S11 lockstep)
# ======================================================================
def _immersed_prescription():
    """``glass_object``-like design as a prescription dict: N-BK7 object
    space, air image space, so the twin sees an immersed conjugate."""
    return {
        'name': 'W4bImmersedObject',
        'aperture_diameter': 4e-3,
        'surfaces': [
            {'radius': 70e-3, 'conic': 0.0, 'aspheric_coeffs': None,
             'glass_before': 'N-BK7', 'glass_after': 'air'},
            {'radius': 55e-3, 'conic': 0.0, 'aspheric_coeffs': None,
             'glass_before': 'air', 'glass_after': 'N-SF2'},
            {'radius': -90e-3, 'conic': 0.0, 'aspheric_coeffs': None,
             'glass_before': 'N-SF2', 'glass_after': 'air'},
        ],
        'thicknesses': [21e-3, 4e-3, 0.0],
    }


def _immersed_image_prescription():
    """Air object space, N-BK7 IMAGE space (the ``bfl`` side)."""
    return {
        'name': 'W4bImmersedImage',
        'aperture_diameter': 4e-3,
        'surfaces': [
            {'radius': 45e-3, 'conic': 0.0, 'aspheric_coeffs': None,
             'glass_before': 'air', 'glass_after': 'N-BK7'},
            {'radius': -60e-3, 'conic': 0.0, 'aspheric_coeffs': None,
             'glass_before': 'N-BK7', 'glass_after': 'air'},
            {'radius': 80e-3, 'conic': 0.0, 'aspheric_coeffs': None,
             'glass_before': 'air', 'glass_after': 'N-BK7'},
        ],
        'thicknesses': [6e-3, 18e-3, 0.0],
    }


class TestW4bAlgebraTwinLockstep:
    """S11 lesson: check the lockstep twin for the same defect.

    ``algebra/from_prescription`` builds the SAME reduced matrix out of
    ``FreeSpace(t / n)`` + ``ThinLens(1/phi)``, so:

    * its ``.abcd`` and its ``.efl`` must still match ``system_abcd``
      EXACTLY on an immersed prescription -- which they do, because
      ``efl`` was deliberately left reduced.  Had ``efl`` been redefined
      as ``n'/Phi``, this test would fail and the twin could not follow:
      a ``CompositeOperator`` has no terminal-index information at all.
    * it exposes no ``bfl``/``ffl``, so the geometric change here cannot
      desynchronise the two layers.  Pinned structurally below.
    """

    @pytest.mark.parametrize('builder', [_immersed_prescription,
                                         _immersed_image_prescription])
    def test_twin_abcd_and_efl_still_match_on_an_immersed_prescription(
            self, builder):
        import lumenairy as la
        from lumenairy.raytrace import surfaces_from_prescription
        rx = builder()
        wl = _WL
        surfaces = surfaces_from_prescription(rx)
        M_ref, efl_ref, _bfl, _ffl = system_abcd(surfaces, wl)
        op = la.Operator.from_prescription(rx, wl)
        assert np.allclose(op.abcd, M_ref, rtol=1e-9, atol=1e-12), (
            f"{rx['name']}: twin ABCD\n{op.abcd}\ndiffers from "
            f"system_abcd\n{M_ref}")
        assert _rel(op.efl, efl_ref) < 1e-9, (
            f"{rx['name']}: twin efl {op.efl!r} != system_abcd efl "
            f"{efl_ref!r} -- the reduced-EFL lockstep is broken.")
        # the twin's efl is the REDUCED one, so it is NOT the geometric
        # image-space focal length whenever the image space is glass
        n_img = _loc_n_img(surfaces, wl)
        if n_img != 1.0:
            assert _rel(op.efl, n_img * efl_ref) > 0.2

    def test_the_twin_exposes_no_focal_distances(self):
        """Structural half: nothing to keep in lockstep on the ``bfl`` /
        ``ffl`` side, because the algebra layer never exposed them (and
        could not compute them -- a CompositeOperator has folded the
        media into its reduced lengths)."""
        import lumenairy as la
        op = la.Operator.from_prescription(_immersed_prescription(), _WL)
        for attr in ('bfl', 'ffl', 'pp_image_z', 'pp_object_z'):
            assert not hasattr(op, attr), (
                f"Operator now exposes {attr!r}; it must be kept in "
                f"lockstep with system_abcd's geometric convention (see "
                f"the W4b note in seidel.system_abcd).")
        assert hasattr(op, 'efl')
