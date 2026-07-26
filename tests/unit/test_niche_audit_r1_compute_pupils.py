"""AUDIT_ADVERSARIAL_CODEBASE_2026_07_25 R-1: ``compute_pupils`` stop-split.

Three defects in one function, all in the construction of the sub-systems
the stop is imaged through, all measured against an EXACT REAL-RAY oracle
(embedded below -- it root-finds real rays through the actual surfaces and
reads axis crossings, so it shares no code with the paraxial ABCD path it
judges):

R-1a  the pre-stop sub-system dropped its FINAL leg.  ``system_abcd`` walks
      transfers only BETWEEN surfaces, so ``system_abcd(surfaces[:stop])``
      ends at surface ``stop-1``'s vertex, NOT at the stop.  ``ep_z`` was
      therefore frozen at its gap=0 value and ``ep_radius`` under-reported
      on every non-front-stop system:

          gap [mm]     0      2      5     10     20     40
          d ep_radius  0%  -4.2% -10.5% -21.0% -42.1% -84.1%
          f/#       11.88  11.88  11.88  11.88  11.88  11.88   (frozen)
          f/# exact 11.88  11.38  10.63   9.38   6.88   1.88

      ``seidel_coefficients`` built the SAME sub-system correctly (it
      prepended the missing leg explicitly), so the two disagreed; both
      now call the single shared ``_pre_stop_abcd``.

R-1b  ``ep_z`` had the wrong SIGN.  ``PupilInfo`` documents (and
      ``analysis/field.py`` / ``analysis/image_plane_wfe.py`` geometrically
      require) a SIGNED coordinate measured from surface 0, but the code
      returned the object DISTANCE ``-B/A`` -- the mirror image of the true
      pupil plane.  Discriminator: with a powerless pre-stop leg (flat
      dummy, 10 mm gap, then the stop) the EP *is* the stop at ``+10 mm``;
      pre-fix ``compute_pupils`` returned ``-0.0`` (R-1a masked it by
      leaving ``B = 0``), and R-1a's fix alone would have returned
      ``-10 mm``.

R-1c  the post-stop sub-system's LEADING leg was always evaluated in AIR
      (it was supplied as a dummy air-to-air ``Surface``).  When the stop's
      image-side medium is glass the leg was short by ``n``: measured
      ``xp_z`` +3.8% / ``xp_radius`` +1.9% for a stop declared on a BK7
      lens surface, +7.9% / +2.5% for a stop inside BK7.

Hard-coded ``_PREFIX_*`` values below were measured on the pre-fix code
(reconstructed byte-faithfully) and are asserted to be REJECTED, so a
regression cannot pass silently.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from lumenairy.raytrace import (
    RayBundle,
    Surface,
    compute_pupils,
    first_order_data,
    seidel_coefficients,
    trace,
)

_WL = 587.5618e-9          # He d-line, the audit's wavelength
_TOL = 1e-6                # relative tolerance vs the oracle (it is itself
                           # only good to ~1e-10 rel; 1e-6 is generous but
                           # still ~5 orders tighter than every defect above)


# ======================================================================
# Exact real-ray oracle -- no paraxial/ABCD code involved
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
    """(height, slope) of the real ray at surface ``idx``.

    Both designs used here keep flat surfaces at the stop and at the image
    plane, so the recorded height IS the height at that vertex plane.
    """
    b = trace(_ray(y0, u0, wl), surfaces, wl).ray_history[idx]
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
    """Exact (ep_z, ep_radius, xp_z, xp_radius, efl, fnum).

    ``ep_z`` is the axis crossing of the real ray through the stop CENTRE,
    extrapolated upstream of surface 0 -- a SIGNED z coordinate by
    construction.  ``xp_z`` is the same ray's axis crossing downstream,
    measured from the LAST surface vertex.  The radii are the conjugate
    heights of a tiny stop-height pencil at those planes.
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
    h = 1e-7               # EFL = -1/C from the real trace
    yo, uo = _state_at(surfaces, wl, last, h, 0.0)
    efl = -h / uo
    return dict(ep_z=ep_z, ep_radius=ep_radius, xp_z=xp_z,
                xp_radius=xp_radius, efl=efl,
                fnum=abs(efl) / (2.0 * ep_radius))


# ======================================================================
# Designs
# ======================================================================
def _d_gap(gap):
    """Biconvex N-BK7 singlet | air gap | flat stop | image.  (audit design)"""
    return [
        Surface(radius=50e-3, glass_before='air', glass_after='N-BK7',
                thickness=5e-3, semi_diameter=np.inf),
        Surface(radius=-50e-3, glass_before='N-BK7', glass_after='air',
                thickness=gap, semi_diameter=np.inf),
        Surface(radius=np.inf, glass_before='air', glass_after='air',
                thickness=20e-3, semi_diameter=2e-3, is_stop=True),
        Surface(radius=np.inf, glass_before='air', glass_after='air',
                thickness=0.0, semi_diameter=np.inf),
    ], 2


def _d_both_sides():
    """Power on BOTH sides of the stop."""
    return [
        Surface(radius=40e-3, glass_before='air', glass_after='N-BK7',
                thickness=4e-3, semi_diameter=np.inf),
        Surface(radius=-120e-3, glass_before='N-BK7', glass_after='air',
                thickness=8e-3, semi_diameter=np.inf),
        Surface(radius=np.inf, glass_before='air', glass_after='air',
                thickness=12e-3, semi_diameter=1.5e-3, is_stop=True),
        Surface(radius=60e-3, glass_before='air', glass_after='N-SF2',
                thickness=3e-3, semi_diameter=np.inf),
        Surface(radius=-80e-3, glass_before='N-SF2', glass_after='air',
                thickness=30e-3, semi_diameter=np.inf),
        Surface(radius=np.inf, glass_before='air', glass_after='air',
                thickness=0.0, semi_diameter=np.inf),
    ], 2


def _d_stop_in_glass():
    """Stop INSIDE the glass: both stop-adjacent legs live in n=1.5168."""
    return [
        Surface(radius=45e-3, glass_before='air', glass_after='N-BK7',
                thickness=6e-3, semi_diameter=np.inf),
        Surface(radius=np.inf, glass_before='N-BK7', glass_after='N-BK7',
                thickness=6e-3, semi_diameter=1.2e-3, is_stop=True),
        Surface(radius=-45e-3, glass_before='N-BK7', glass_after='air',
                thickness=25e-3, semi_diameter=np.inf),
        Surface(radius=np.inf, glass_before='air', glass_after='air',
                thickness=0.0, semi_diameter=np.inf),
    ], 1


def _d_powerless_pre_stop(t=10e-3):
    """R-1b sign discriminator: nothing but ``t`` of air before the stop,
    so the EP *is* the stop, at z = +t."""
    return [
        Surface(radius=np.inf, glass_before='air', glass_after='air',
                thickness=t, semi_diameter=np.inf),
        Surface(radius=np.inf, glass_before='air', glass_after='air',
                thickness=15e-3, semi_diameter=2e-3, is_stop=True),
        Surface(radius=50e-3, glass_before='air', glass_after='N-BK7',
                thickness=4e-3, semi_diameter=np.inf),
        Surface(radius=-50e-3, glass_before='N-BK7', glass_after='air',
                thickness=40e-3, semi_diameter=np.inf),
        Surface(radius=np.inf, glass_before='air', glass_after='air',
                thickness=0.0, semi_diameter=np.inf),
    ], 1


def _d_front_stop():
    """Stop AT surface 0 (a BK7 lens surface) -- the EP-invariance case,
    and the R-1c case (the XP leg leaves the stop inside glass)."""
    return [
        Surface(radius=50e-3, glass_before='air', glass_after='N-BK7',
                thickness=5e-3, semi_diameter=2e-3, is_stop=True),
        Surface(radius=-50e-3, glass_before='N-BK7', glass_after='air',
                thickness=45e-3, semi_diameter=np.inf),
        Surface(radius=np.inf, glass_before='air', glass_after='air',
                thickness=0.0, semi_diameter=np.inf),
    ], 0


def _rel(a, b):
    return abs(a - b) / abs(b) if b != 0.0 else abs(a - b)


# ======================================================================
# Pin 1 -- non-front-stop design vs the exact oracle
# ======================================================================
# Oracle values, 12 significant digits (metres / dimensionless):
_EXACT_GAP10 = dict(ep_z=+1.698595526057e-02, ep_radius=+2.622160185451e-03,
                    xp_z=-2.000000000000e-02, xp_radius=+2.000000000000e-03,
                    fnum=+9.384055946383e+00)
# Pre-fix compute_pupils on the same design (byte-faithful reconstruction):
_PREFIX_GAP10 = dict(ep_z=-3.412689673224e-03, ep_radius=+2.070547125634e-03,
                     fnum=+1.188405594645e+01)


class TestR1EntrancePupilVsExactTrace:
    """R-1a/R-1b: EP position, radius and f/# on a 10 mm-gap singlet."""

    def test_ep_z_ep_radius_fnum_match_exact_trace(self):
        surfaces, stop = _d_gap(10e-3)
        o = _oracle(surfaces, _WL, stop)
        p = compute_pupils(surfaces, _WL, stop_index=stop)
        fod = first_order_data(surfaces, _WL, stop_index=stop)
        for key, got in (('ep_z', p.ep_z), ('ep_radius', p.ep_radius),
                         ('xp_z', p.xp_z), ('xp_radius', p.xp_radius),
                         ('fnum', fod.fnum)):
            assert _rel(got, o[key]) < _TOL, (
                f"{key}: compute_pupils gave {got!r}, exact real-ray oracle "
                f"{o[key]!r} (rel {_rel(got, o[key]):.3e}); the pre-fix code "
                f"gave {_PREFIX_GAP10.get(key, 'n/a')}.")

    def test_pinned_against_hardcoded_exact_values(self):
        """Same numbers, hard-coded, so the oracle and the library must
        BOTH have to change for this to move."""
        surfaces, stop = _d_gap(10e-3)
        p = compute_pupils(surfaces, _WL, stop_index=stop)
        fod = first_order_data(surfaces, _WL, stop_index=stop)
        got = dict(ep_z=p.ep_z, ep_radius=p.ep_radius, xp_z=p.xp_z,
                   xp_radius=p.xp_radius, fnum=fod.fnum)
        for key, want in _EXACT_GAP10.items():
            assert _rel(got[key], want) < _TOL, (
                f"{key} = {got[key]!r}, pinned exact value {want!r}")

    def test_prefix_values_are_rejected(self):
        """The R-1a/R-1b numbers must not come back."""
        surfaces, stop = _d_gap(10e-3)
        p = compute_pupils(surfaces, _WL, stop_index=stop)
        fod = first_order_data(surfaces, _WL, stop_index=stop)
        assert _rel(p.ep_z, _PREFIX_GAP10['ep_z']) > 0.5, (
            f"ep_z = {p.ep_z!r} is the pre-fix (wrong-sign, gap-frozen) value")
        assert _rel(p.ep_radius, _PREFIX_GAP10['ep_radius']) > 0.1, (
            f"ep_radius = {p.ep_radius!r} is the pre-fix value "
            f"({_PREFIX_GAP10['ep_radius']!r}, 21% low)")
        assert _rel(fod.fnum, _PREFIX_GAP10['fnum']) > 0.1, (
            f"f/# = {fod.fnum!r} is the pre-fix value "
            f"({_PREFIX_GAP10['fnum']!r})")

    def test_power_on_both_sides_of_the_stop(self):
        surfaces, stop = _d_both_sides()
        o = _oracle(surfaces, _WL, stop)
        p = compute_pupils(surfaces, _WL, stop_index=stop)
        fod = first_order_data(surfaces, _WL, stop_index=stop)
        for key, got in (('ep_z', p.ep_z), ('ep_radius', p.ep_radius),
                         ('xp_z', p.xp_z), ('xp_radius', p.xp_radius),
                         ('fnum', fod.fnum)):
            assert _rel(got, o[key]) < _TOL, (
                f"{key}: got {got!r}, exact {o[key]!r}")
        # pre-fix: ep_z = -2.730151738579e-03, ep_radius 14.1% low,
        #          f/# = 11.288 vs 9.691 exact
        assert _rel(p.ep_radius, 1.552910344226e-03) > 0.1
        assert _rel(fod.fnum, 1.128829932468e+01) > 0.1


class TestR1EntrancePupilSignConvention:
    """R-1b: ``ep_z`` is a SIGNED coordinate, not a distance."""

    def test_powerless_pre_stop_leg_puts_the_ep_at_the_stop(self):
        t = 10e-3
        surfaces, stop = _d_powerless_pre_stop(t)
        o = _oracle(surfaces, _WL, stop)
        p = compute_pupils(surfaces, _WL, stop_index=stop)
        # With nothing but air ahead of it the stop images to itself.
        assert _rel(o['ep_z'], +t) < 1e-9, "oracle self-check"
        assert abs(p.ep_z - t) < 1e-12, (
            f"ep_z = {p.ep_z!r} for a powerless pre-stop leg of {t} m; the "
            f"EP *is* the stop, so ep_z must be +{t} (pre-fix: -0.0; "
            f"pre-fix + the R-1a transfer alone: {-t}).")
        assert abs(p.ep_radius - 2e-3) < 1e-12

    def test_ep_is_downstream_for_a_stop_behind_a_positive_lens(self):
        """A stop closer than one focal length behind a positive singlet has
        a VIRTUAL entrance pupil, i.e. ep_z > 0 (downstream of surface 0).
        The pre-fix code reported it upstream."""
        for gap in (0.0, 2e-3, 10e-3, 40e-3):
            surfaces, stop = _d_gap(gap)
            p = compute_pupils(surfaces, _WL, stop_index=stop)
            assert p.ep_z > 0.0, (
                f"gap={gap}: ep_z = {p.ep_z!r} <= 0; the exact ray through "
                f"the stop centre crosses the axis DOWNSTREAM of surface 0.")


class TestR1FrontStopInvariance:
    """Pin 2: front-stop / gap=0 systems must not move (except the two
    values the audit proves were wrong: ``ep_z``'s sign, and the R-1c
    ``xp_*`` pair when the stop's image-side medium is glass)."""

    # Pre-fix values, measured on the pre-fix code:
    _PREFIX_GAP0 = dict(ep_z=-3.412689673224e-03,
                        ep_radius=+2.070547125634e-03,
                        xp_z=-2.000000000000e-02,
                        xp_radius=+2.000000000000e-03,
                        fnum=+1.188405594645e+01)

    def test_gap0_values_unchanged_except_ep_z_sign(self):
        surfaces, stop = _d_gap(0.0)
        p = compute_pupils(surfaces, _WL, stop_index=stop)
        fod = first_order_data(surfaces, _WL, stop_index=stop)
        pre = self._PREFIX_GAP0
        # magnitudes: bit-identical to the pre-fix code
        assert abs(abs(p.ep_z) - abs(pre['ep_z'])) < 1e-15
        assert abs(p.ep_radius - pre['ep_radius']) < 1e-15
        assert abs(p.xp_z - pre['xp_z']) < 1e-15
        assert abs(p.xp_radius - pre['xp_radius']) < 1e-15
        assert abs(fod.fnum - pre['fnum']) < 1e-12
        # ...and the sign of ep_z is the one the exact ray gives
        o = _oracle(surfaces, _WL, stop)
        assert p.ep_z > 0 and o['ep_z'] > 0
        assert _rel(p.ep_z, o['ep_z']) < _TOL

    def test_gap_zero_error_is_exactly_zero_the_mechanism_discriminator(self):
        """R-1a's signature: the dropped leg has length ``gap``, so the
        radius error must vanish EXACTLY at gap=0 and nowhere else."""
        surfaces, stop = _d_gap(0.0)
        o = _oracle(surfaces, _WL, stop)
        p = compute_pupils(surfaces, _WL, stop_index=stop)
        assert _rel(p.ep_radius, o['ep_radius']) < 1e-9
        # pre-fix ep_radius was exactly this value at gap=0 too
        assert abs(p.ep_radius - self._PREFIX_GAP0['ep_radius']) < 1e-15

    def test_front_stop_ep_side_unchanged(self):
        surfaces, stop = _d_front_stop()
        p = compute_pupils(surfaces, _WL, stop_index=stop)
        fod = first_order_data(surfaces, _WL, stop_index=stop)
        assert p.ep_z == 0.0
        assert abs(p.ep_radius - 2e-3) < 1e-15
        assert abs(fod.fnum - 1.230324894040e+01) < 1e-9   # pre-fix value


class TestR1ExitPupilTransferMedium:
    """R-1c: the stop -> first-post-surface leg must use the STOP's own
    image-side glass, not air."""

    def test_front_stop_on_a_glass_surface(self):
        surfaces, stop = _d_front_stop()
        o = _oracle(surfaces, _WL, stop)
        p = compute_pupils(surfaces, _WL, stop_index=stop)
        assert _rel(p.xp_z, o['xp_z']) < _TOL, (
            f"xp_z = {p.xp_z!r}, exact {o['xp_z']!r}")
        assert _rel(p.xp_radius, o['xp_radius']) < _TOL
        # pre-fix (leg walked in air): -5.027248188184e-02 / 2.108992752738e-03
        assert _rel(p.xp_z, -5.027248188184e-02) > 0.01
        assert _rel(p.xp_radius, 2.108992752738e-03) > 0.01

    def test_stop_inside_glass(self):
        surfaces, stop = _d_stop_in_glass()
        o = _oracle(surfaces, _WL, stop)
        p = compute_pupils(surfaces, _WL, stop_index=stop)
        fod = first_order_data(surfaces, _WL, stop_index=stop)
        for key, got in (('ep_z', p.ep_z), ('ep_radius', p.ep_radius),
                         ('xp_z', p.xp_z), ('xp_radius', p.xp_radius),
                         ('fnum', fod.fnum)):
            assert _rel(got, o[key]) < _TOL, (
                f"{key}: got {got!r}, exact {o[key]!r}")
        # pre-fix: xp_z = -3.144403714949e-02 (+7.9%),
        #          xp_radius = 1.288807429898e-03 (+2.5%),
        #          ep_z = 0 (!), ep_radius = 1.2e-3 (-4.5%)
        assert _rel(p.xp_z, -3.144403714949e-02) > 0.01
        assert _rel(p.xp_radius, 1.288807429898e-03) > 0.01
        assert _rel(p.ep_radius, 1.2e-3) > 0.01

    def test_symmetric_design_gives_mirror_image_pupils(self):
        """Independent geometric check with no oracle at all: the
        stop-in-glass design is symmetric about the stop, so the EP and XP
        must be mirror images -- equal radii, and equal distances from
        their respective outer surfaces."""
        surfaces, stop = _d_stop_in_glass()
        p = compute_pupils(surfaces, _WL, stop_index=stop)
        assert abs(p.ep_radius - p.xp_radius) < 1e-12
        # last surface sits 25 mm past surface 2; XP measured from it
        xp_from_surf2 = 25e-3 + p.xp_z
        assert abs(xp_from_surf2 + p.ep_z) < 1e-12, (
            f"EP at +{p.ep_z!r} after surface 0 but XP at {xp_from_surf2!r} "
            f"relative to surface 2; a symmetric system requires "
            f"xp = -ep.")


class TestR1ConsistencyWithSeidelCoefficients:
    """Pin 3: the two functions build the same pre-stop sub-system.

    ``seidel_coefficients`` exposes its own initial conditions: for an
    object at infinity the marginal ray enters parallel to the axis at
    ``y_marginal[0] = r_stop / A_pre`` -- i.e. AT the entrance-pupil edge --
    and the chief ray enters at ``y_chief[0] = -B_pre * sigma / A_pre``, so
    it crosses the axis at ``z = -y_chief[0] / sigma = +B_pre / A_pre``, the
    EP.  Both must agree with ``compute_pupils`` to roundoff.
    """

    @pytest.mark.parametrize('design', [
        _d_gap(0.0), _d_gap(10e-3), _d_both_sides(), _d_stop_in_glass(),
        _d_powerless_pre_stop(), _d_front_stop()])
    def test_seidel_implied_entrance_pupil_agrees(self, design):
        surfaces, stop = design
        sigma = 1e-3
        sd, _ = seidel_coefficients(surfaces, _WL, stop_index=stop,
                                    field_angle=sigma)
        ep_z_seidel = -float(sd['y_chief'][0]) / sigma
        ep_r_seidel = abs(float(sd['y_marginal'][0]))
        p = compute_pupils(surfaces, _WL, stop_index=stop)
        assert abs(ep_z_seidel - p.ep_z) <= 1e-12 * max(1.0, abs(p.ep_z)), (
            f"seidel_coefficients implies ep_z = {ep_z_seidel!r} but "
            f"compute_pupils reports {p.ep_z!r}: the two stop-splits have "
            f"diverged again.")
        assert abs(ep_r_seidel - p.ep_radius) <= 1e-15 + 1e-12 * p.ep_radius, (
            f"seidel_coefficients implies ep_radius = {ep_r_seidel!r} but "
            f"compute_pupils reports {p.ep_radius!r}.")


class TestR1GapSweep:
    """Pin 4: the whole sweep, not just one gap."""

    # (gap, ep_z, ep_radius, fnum) from the exact real-ray oracle
    _SWEEP = [
        (0.0,   +3.412689673223e-03, +2.070547125635e-03, +1.188405594638e+01),
        (2e-3,  +5.650420834186e-03, +2.161487785773e-03, +1.138405594638e+01),
        (5e-3,  +9.401575018995e-03, +2.313933461016e-03, +1.063405594638e+01),
        (10e-3, +1.698595526057e-02, +2.622160185451e-03, +9.384055946383e+00),
        (20e-3, +4.041770085815e-02, +3.574418638135e-03, +6.884055946366e+00),
        (40e-3, +2.738341364032e-01, +1.306038599816e-02, +1.884055944757e+00),
    ]

    @pytest.mark.parametrize('gap,ep_z,ep_radius,fnum', _SWEEP)
    def test_sweep_matches_oracle(self, gap, ep_z, ep_radius, fnum):
        surfaces, stop = _d_gap(gap)
        p = compute_pupils(surfaces, _WL, stop_index=stop)
        fod = first_order_data(surfaces, _WL, stop_index=stop)
        o = _oracle(surfaces, _WL, stop)          # live oracle too
        assert _rel(o['ep_radius'], ep_radius) < 1e-8, "pinned oracle drifted"
        assert _rel(p.ep_z, ep_z) < _TOL
        assert _rel(p.ep_radius, ep_radius) < _TOL
        assert _rel(fod.fnum, fnum) < _TOL

    def test_pupil_grows_and_fnumber_falls_monotonically_with_gap(self):
        """R-1a froze both; the physics is monotone (moving the stop away
        from the lens makes its object-space image bigger)."""
        radii, fnums, zs = [], [], []
        for gap, *_ in self._SWEEP:
            surfaces, stop = _d_gap(gap)
            radii.append(compute_pupils(surfaces, _WL, stop_index=stop
                                        ).ep_radius)
            fnums.append(first_order_data(surfaces, _WL,
                                          stop_index=stop).fnum)
            zs.append(compute_pupils(surfaces, _WL, stop_index=stop).ep_z)
        assert np.all(np.diff(radii) > 0), f"ep_radius not monotone: {radii}"
        assert np.all(np.diff(fnums) < 0), f"f/# not monotone: {fnums}"
        assert np.all(np.diff(zs) > 0), f"ep_z not monotone: {zs}"
        # and the sweep must actually SPAN something (frozen values would
        # have passed a laxer version of this test)
        assert radii[-1] / radii[0] > 5.0


def test_no_warnings_on_the_fixed_path():
    """The fix removed the dummy-``Surface`` construction; make sure nothing
    started warning (e.g. an infinite-semi_diameter path)."""
    surfaces, stop = _d_gap(10e-3)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        compute_pupils(surfaces, _WL, stop_index=stop)
        first_order_data(surfaces, _WL, stop_index=stop)
