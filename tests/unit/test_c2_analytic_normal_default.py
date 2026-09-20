"""WP-C2 -- ``sphere_normal='analytic'`` and ``renormalize='exit'`` as the
ray tracer's defaults (5.49.0).

The maintainer's decision is section 1.3 of
``docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/MAINTAINER_DECISIONS_2026_09.md``;
the measurements behind it are WP-B9 sections 2, 5 and 6, VERIFY-WP-B9
sections 3 and 6, and this work package's own re-measurement in
``validation/probe_c2_analytic_normal/`` (both builds, every probe).

What this file pins, and why each claim is a DECISION rather than a
reading:

* **the defaults themselves**, asked both from the signature and from a
  call -- a default is a contract, and a silent revert has to be caught
  by a named test rather than by a fixture drifting;  each flip lands in
  its own commit, and its arms land with it;
* **the way back** -- ``sphere_normal='generic'`` /
  ``renormalize='surface'`` reproduce the pre-5.49.0 arithmetic exactly,
  which is asserted here against the generic route forced at the normal
  dispatch, and archive-to-archive against 49ddf4bd in
  ``validation/probe_c2_analytic_normal/byte_identity.py``;
* **the accuracy decision**, against a 60-digit ``decimal`` oracle whose
  own error is ~44 digits below float64, with the bar in ULP and the
  generic route measured alongside so the comparison is two-sided;
* **the vignetting decision** -- the two domain gates straddle over a
  band about 1 ULP of ``h`` wide at ``0.99995 |R|``, which is the ONE
  observable the flip moves discontinuously, and no bundle that is not
  aimed at it lands in it;
* **the clamp stays** (the ledger's recommendation: VERIFY-B9 3.2
  measured that the closed form is NOT well-conditioned at the rim, so
  dropping the clamp is a separate vignetting decision nobody has
  taken), pinned by locating the threshold on the running build.

The mutation matrix at the bottom names which test catches which
regression.
"""
from __future__ import annotations

import decimal
import inspect
import math
import unittest.mock as _mock

import numpy as np
import pytest

import lumenairy as la
from lumenairy.raytrace import intersection as _isect
from lumenairy.raytrace import surface as _surf_mod
from lumenairy.raytrace.core import Surface
from lumenairy.raytrace.surface import (
    _is_pure_spherical,
    _sphere_normal,
    _surface_normal,
)
from lumenairy.raytrace.trace import _make_bundle, trace
from lumenairy.raytrace.world_trace import trace_world

WL = 587.6e-9
ULP = 2.0 ** -52

# 5.49.0's defaults, written out once so a revert changes exactly one
# place in this file and every arm below moves with it.
C2_SPHERE_NORMAL = 'analytic'
C2_RENORMALIZE = 'exit'
PRE_C2_SPHERE_NORMAL = 'generic'
PRE_C2_RENORMALIZE = 'surface'


# ===========================================================================
# Fixtures -- built here, so every number below states the geometry it
# came from.
# ===========================================================================

def _spherical7():
    return [
        Surface(radius=0.0515, thickness=0.008, glass_before='air',
                glass_after='N-BK7', semi_diameter=0.0127),
        Surface(radius=-0.0345, thickness=0.003, glass_before='N-BK7',
                glass_after='N-SF5', semi_diameter=0.0127),
        Surface(radius=-0.120, thickness=0.010, glass_before='N-SF5',
                glass_after='air', semi_diameter=0.0127),
        Surface(radius=0.080, thickness=0.006, glass_before='air',
                glass_after='N-BK7', semi_diameter=0.0127),
        Surface(radius=-0.080, thickness=0.090, glass_before='N-BK7',
                glass_after='air', semi_diameter=0.0127),
        Surface(radius=np.inf, thickness=0.010, glass_before='air',
                glass_after='air', semi_diameter=0.02),
        Surface(radius=np.inf, thickness=0.0, glass_before='air',
                glass_after='air', semi_diameter=0.02),
    ]


def _conic3():
    """No pure sphere anywhere: the control.  Neither switch can move a
    bit here, which is what makes it a control."""
    return [
        Surface(radius=0.0515, conic=-0.6, thickness=0.008,
                glass_before='air', glass_after='N-BK7',
                semi_diameter=0.0127),
        Surface(radius=-0.0345, conic=-1.2, thickness=0.100,
                glass_before='N-BK7', glass_after='air',
                semi_diameter=0.0127),
        Surface(radius=np.inf, thickness=0.0, glass_before='air',
                glass_after='air', semi_diameter=0.030),
    ]


def _cassegrain():
    """Two SPHERICAL mirrors -- the closed form has to serve reflection
    as well as refraction, and a sign error shows up here first."""
    return [
        Surface(radius=-0.400, thickness=-0.150, glass_before='air',
                glass_after='air', is_mirror=True, semi_diameter=0.050),
        Surface(radius=-0.120, thickness=0.250, glass_before='air',
                glass_after='air', is_mirror=True, semi_diameter=0.015),
        Surface(radius=np.inf, thickness=0.0, glass_before='air',
                glass_after='air', semi_diameter=0.030),
    ]


def _bundle(n, semi=0.0126, tilt=0.0, seed=20260920):
    rng = np.random.default_rng(seed)
    r = semi * np.sqrt(rng.random(n))
    th = 2 * np.pi * rng.random(n)
    L = np.full(n, math.sin(math.radians(tilt)))
    return _make_bundle(r * np.cos(th), r * np.sin(th), L,
                        np.zeros(n), WL)


def _decimal_sphere_normal(x, y, R, prec=60):
    """60-digit oracle for the outward unit normal of the sphere
    ``x^2 + y^2 + (z - R)^2 = R^2`` above ``(x, y)``.

    Independent of the library and of WP-B9's copy of this helper: it
    evaluates the textbook ``n = (-dz/dx, -dz/dy, 1) / |.|`` with
    ``dz/dh = h / (R sqrt(1 - h^2/R^2))`` in ``decimal`` at 60
    significant digits -- about 44 digits beyond float64, so the
    oracle's own error is four decades below the last bit of anything it
    is compared with and is not measurable here.
    """
    ctx = decimal.Context(prec=prec)
    dec = ctx.create_decimal
    X, Y, RR = dec(repr(float(x))), dec(repr(float(y))), dec(repr(float(R)))
    h_sq = ctx.add(ctx.multiply(X, X), ctx.multiply(Y, Y))
    h = h_sq.sqrt(ctx)
    one = dec(1)
    inner = ctx.subtract(one, ctx.divide(h_sq, ctx.multiply(RR, RR)))
    root = inner.sqrt(ctx)
    dz_dh = ctx.divide(h, ctx.multiply(RR, root))
    if h == 0:
        dz_dx = dz_dy = dec(0)
    else:
        dz_dx = ctx.divide(ctx.multiply(dz_dh, X), h)
        dz_dy = ctx.divide(ctx.multiply(dz_dh, Y), h)
    mag = ctx.add(ctx.add(ctx.multiply(dz_dx, dz_dx),
                          ctx.multiply(dz_dy, dz_dy)), one).sqrt(ctx)
    return np.array([float(ctx.divide(-dz_dx, mag)),
                     float(ctx.divide(-dz_dy, mag)),
                     float(ctx.divide(one, mag))])


def _generic_normal(x, y, surf):
    """The pre-5.49.0 route, composed here exactly as
    ``_surface_normal`` composes it, so the oracle comparison is
    two-sided rather than a floor bar on one route."""
    dz_dx, dz_dy = _surf_mod._surface_sag_derivatives_xy(x, y, surf)
    mag = np.sqrt(dz_dx ** 2 + dz_dy ** 2 + 1.0)
    return -dz_dx / mag, -dz_dy / mag, 1.0 / mag


# ===========================================================================
# 1 -- the defaults, from the signature AND from a call
# ===========================================================================

def test_c2_the_defaults_are_what_the_ledger_decided_from_the_signature():
    """MUTATION ARM 1a: a silent revert of either default fails here.

    5.49.0 moved ``trace`` and ``trace_world`` to
    ``sphere_normal='analytic'`` (this commit) and, in its own commit,
    to ``renormalize='exit'`` -- whose arms live beside these.  The
    PRIVATE helpers deliberately did not move -- ``analysis.ghost`` and
    the finite-difference differential path call ``_refract`` /
    ``_reflect`` directly with no trace loop around them to run a single
    exit-plane rescale, so ``renormalize=True`` there is the only
    setting under which they own a unit direction at all.  That
    separation is asserted here so neither half can drift into the
    other.
    """
    for fn in (trace, trace_world):
        params = inspect.signature(fn).parameters
        assert params['sphere_normal'].default == C2_SPHERE_NORMAL, fn
    for fn in (_isect._refract, _isect._reflect):
        params = inspect.signature(fn).parameters
        assert params['sphere_normal'].default == PRE_C2_SPHERE_NORMAL, fn
        assert params['renormalize'].default is True, fn
    assert (inspect.signature(_surface_normal)
            .parameters['analytic_sphere'].default is False)


def test_c2_the_defaults_are_what_a_call_actually_takes():
    """MUTATION ARM 1b: the signature is not the contract -- what the
    loop PASSES is.  A revert that keeps the signature and hard-codes
    the old route inside ``trace`` fails here and not above.

    Asked at every surface, for every combination, on a stack that has
    both spheres and flats.
    """
    S = _spherical7()
    rays = _bundle(64)
    seen = []
    real = _surf_mod._surface_normal

    def watching(x, y, surface, *, analytic_sphere=False):
        seen.append(analytic_sphere)
        return real(x, y, surface, analytic_sphere=analytic_sphere)

    with _mock.patch.object(_isect, '_surface_normal', watching):
        trace(rays, S, WL, output_filter='last')
        assert seen and all(seen), seen
        n = len(seen)
        seen.clear()
        trace(rays, S, WL, output_filter='last',
              sphere_normal=PRE_C2_SPHERE_NORMAL)
        assert seen and not any(seen), seen
        assert len(seen) == n



# ===========================================================================
# 2 -- the way back
# ===========================================================================

def test_c2_the_way_back_is_the_pre_5_49_0_arithmetic_exactly():
    """MUTATION ARM 2: ``sphere_normal='generic'`` must BE the old
    route, not merely a slower one.

    Forced comparison: a ``_surface_normal`` that IGNORES
    ``analytic_sphere`` and always takes the sag-derivative route is
    installed, and the default trace under it is compared with an
    unpatched ``sphere_normal='generic'`` trace.  Byte-identical means
    the keyword reaches every normal evaluation and changes nothing
    else.  The archive-to-archive half of this claim -- that the result
    also equals what 49ddf4bd produced -- is
    ``validation/probe_c2_analytic_normal/byte_identity.py``, which
    measured 938 of 1008 arrays (1 630 399 values) identical with both
    old keywords passed, the 70 exceptions being exactly the entry
    points that trace INTERNALLY and expose no keyword to pass.
    """
    real = _surf_mod._surface_normal

    def forced_generic(x, y, surface, *, analytic_sphere=False):
        return real(x, y, surface, analytic_sphere=False)

    for S in (_spherical7(), _cassegrain(), _conic3()):
        rays = _bundle(1500, semi=0.0126)
        want = trace(rays, S, WL, output_filter='last',
                     sphere_normal=PRE_C2_SPHERE_NORMAL,
                     renormalize=PRE_C2_RENORMALIZE).image_rays
        with _mock.patch.object(_isect, '_surface_normal',
                                forced_generic):
            got = trace(rays, S, WL, output_filter='last',
                        renormalize=PRE_C2_RENORMALIZE).image_rays
        for f in ('x', 'y', 'z', 'L', 'M', 'N', 'opd', 'alive',
                  'error_code'):
            assert np.array_equal(getattr(want, f), getattr(got, f),
                                  equal_nan=True), (f, S[0].radius)


def test_c2_the_control_without_a_sphere_is_byte_identical_either_way():
    """The switch is confined to what the predicate selects.

    A prescription with no pure sphere must be byte-identical under
    BOTH spellings, which is what makes any move on a spherical
    prescription attributable to the normal route and not to something
    else the flip disturbed.
    """
    S = _conic3()
    rays = _bundle(2000)
    a = trace(rays, S, WL, output_filter='last',
              sphere_normal=C2_SPHERE_NORMAL).image_rays
    g = trace(rays, S, WL, output_filter='last',
              sphere_normal=PRE_C2_SPHERE_NORMAL).image_rays
    for f in ('x', 'y', 'z', 'L', 'M', 'N', 'opd', 'alive',
              'error_code'):
        assert np.array_equal(getattr(a, f), getattr(g, f),
                              equal_nan=True), f
    assert not any(_is_pure_spherical(s) for s in S)


# ===========================================================================
# 3 -- the accuracy decision, against the 60-digit oracle
# ===========================================================================

@pytest.mark.parametrize('R', [0.002, -0.0345, 0.0515, 0.5, -1.0])
def test_c2_the_closed_form_is_the_better_route_over_the_aperture(R):
    """THE DECISION the flip rests on, re-derived here rather than
    quoted: over the WORKING aperture the closed form is within a few
    ULP of the truth and never materially worse than the route it
    replaces.

    BAR, derived: 4 ULP of a unit vector (``4 * 2**-52``), which is the
    accumulation of the handful of roundings either closed form can
    make, with the generic route's own error measured alongside so this
    is a comparison and not a floor.  Over 1056 points (eight radii of
    both signs from 2 mm to 1 m, eleven heights, six azimuths,
    refracting and mirror alike) WP-C2 measured the closed form at
    <= 1.75 ULP out to ``h = 0.95 |R|`` against the generic route's 2.0
    (Windows py3.14 / numpy 2.4.4) and 2.25 (WSL py3.12 / numpy 2.4.6),
    never worse by more than 1 ULP at any of 672 points there, closer at
    56 % of all points against 11 % the other way.
    """
    surf = Surface(radius=R, thickness=0.0, glass_before='air',
                   glass_after='N-BK7', semi_diameter=abs(R))
    fracs = np.array([0.0, 0.01, 0.05, 0.2, 0.5, 0.8, 0.95])
    e_fast, e_slow = [], []
    for frac in fracs:
        for az in (0.0, 0.3, math.pi / 4, 2.7):
            h = abs(R) * frac
            x = np.array([h * math.cos(az)])
            y = np.array([h * math.sin(az)])
            ref = _decimal_sphere_normal(float(x[0]), float(y[0]), R)
            fast = np.array([float(np.ravel(c)[0])
                             for c in _sphere_normal(x, y, R)])
            slow = np.array([float(np.ravel(c)[0])
                             for c in _generic_normal(x, y, surf)])
            e_fast.append(float(np.max(np.abs(fast - ref))) / ULP)
            e_slow.append(float(np.max(np.abs(slow - ref))) / ULP)
    e_fast = np.array(e_fast)
    e_slow = np.array(e_slow)
    assert np.all(e_fast <= 4.0), (R, e_fast.max(), e_slow.max())
    assert np.all(e_fast <= e_slow + 1.0), (R, e_fast, e_slow)
    assert np.any(e_fast < e_slow), (R, e_fast, e_slow)


def test_c2_the_closed_form_is_a_unit_vector_by_construction():
    """``|n|^2 = h^2/R^2 + (1 - h^2/R^2) = 1`` identically, so the only
    departure is rounding.  BAR: 4 ULP -- measured 1.0 (Windows) and
    1.5 (WSL) over 1056 points.  The generic route has to divide by a
    computed magnitude and carries no such identity, which is why this
    arm exists at all."""
    for R in (0.002, -0.0345, 0.0515, 0.5, -1.0):
        h = abs(R) * np.linspace(0.0, 0.95, 60)
        nx, ny, nz = _sphere_normal(h / np.sqrt(2.0), h / np.sqrt(2.0), R)
        defect = np.abs(nx ** 2 + ny ** 2 + nz ** 2 - 1.0)
        assert float(np.max(defect)) <= 4.0 * ULP, (R, defect.max() / ULP)


def test_c2_above_0p95_R_neither_route_dominates_and_the_test_says_so():
    """The honest other half of the accuracy claim, asserted so nobody
    reads the arm above as a bound everywhere.

    ``nz = sqrt(1 - u)`` cancels as ``u -> 1``: the relative error is
    bounded below by ``eps/2 * u/(1 - u)`` for ANY float64 evaluation,
    generic or closed form, because the information is not in the
    inputs (VERIFY-WP-B9 3.2).  So BOTH routes must leave the 4 ULP
    band before the clamp, and the closed form must NOT be claimed as
    uniformly better there.  Measured at ``0.99994 |R|``: 57 ULP
    (closed form) against 76 (generic) in the worst case, with 22 of
    1056 points where the closed form happens to round worse, by up to
    35 ULP.
    """
    R = 0.0515
    surf = Surface(radius=R, thickness=0.0, glass_before='air',
                   glass_after='N-BK7', semi_diameter=abs(R))
    worst_fast = worst_slow = 0.0
    for frac in (0.999, 0.9999, 0.99994):
        for az in (0.0, 0.3, math.pi / 4, 2.7):
            h = abs(R) * frac
            x = np.array([h * math.cos(az)])
            y = np.array([h * math.sin(az)])
            ref = _decimal_sphere_normal(float(x[0]), float(y[0]), R)
            fast = np.array([float(np.ravel(c)[0])
                             for c in _sphere_normal(x, y, R)])
            slow = np.array([float(np.ravel(c)[0])
                             for c in _generic_normal(x, y, surf)])
            worst_fast = max(worst_fast,
                             float(np.max(np.abs(fast - ref))) / ULP)
            worst_slow = max(worst_slow,
                             float(np.max(np.abs(slow - ref))) / ULP)
    assert worst_fast > 4.0 and worst_slow > 4.0, (worst_fast, worst_slow)
    assert worst_fast < 1e3 and worst_slow < 1e3, (worst_fast, worst_slow)


# ===========================================================================
# 4 -- the clamp stays, and where it is
# ===========================================================================

def test_c2_the_domain_clamp_stays_where_the_ledger_left_it():
    """MUTATION ARM 3: moving or removing the clamp is caught here.

    The ledger's decision (section 1.3) is that the clamp STAYS: WP-B9
    proposed dropping it on the grounds that the closed form is
    well-conditioned at the rim, and VERIFY-B9 3.2 measured that this is
    false -- the relative error of ``sqrt(1 - u)`` is bounded below by
    ``eps/2 * u/(1 - u)``, about 1.1e-12 at ``u = 0.9999``, for BOTH
    routes.  Dropping it is a vignetting decision nobody has taken, so
    it is pinned rather than left to drift.

    The threshold is LOCATED on the running build by bisection rather
    than asserted from a constant in the source, so a change to the
    expression -- not only to the literal -- is caught.
    """
    R = 0.0515
    lo, hi = 0.99, 1.2                 # in u = h^2/R^2
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        h = abs(R) * math.sqrt(mid)
        nz = float(np.ravel(_sphere_normal(np.array([h]),
                                           np.array([0.0]), R)[2])[0])
        if math.isfinite(nz):
            lo = mid
        else:
            hi = mid
    assert abs(lo - 0.9999) < 1e-9, (
        f'the closed form\'s domain clamp moved: NaN begins at '
        f'u = h^2/R^2 = {hi:.12f} (last finite {lo:.12f}), expected '
        f'0.9999 -- the ledger section 1.3 decision is that the clamp '
        f'STAYS until a vignetting decision is taken.')
    # and it still kills the ray, rather than returning a grazing normal
    from lumenairy.raytrace import RAY_NAN
    h = abs(R) * math.sqrt(0.99995)
    S = [Surface(radius=R, thickness=0.001, glass_before='air',
                 glass_after='N-BK7', semi_diameter=abs(R) * 2),
         Surface(radius=np.inf, thickness=0.0, glass_before='N-BK7',
                 glass_after='air', semi_diameter=1.0)]
    b = _make_bundle(np.array([h]), np.array([0.0]), np.array([0.0]),
                     np.array([0.0]), WL)
    img = trace(b, S, WL, output_filter='last',
                sphere_normal=PRE_C2_SPHERE_NORMAL).image_rays
    assert not img.alive[0] and img.error_code[0] == RAY_NAN


# ===========================================================================
# 5 -- the vignetting decision: the rim band, and nothing else
# ===========================================================================

def test_c2_the_rim_band_is_the_one_discontinuous_difference():
    """THE VIGNETTING DECISION (VERIFY-WP-B9 3.3).

    The two routes gate the domain from DIFFERENT expressions --
    ``(x*x + y*y)/(R*R)`` here against
    ``(1 + conic) * sqrt(x*x + y*y)**2 / R**2`` there -- which differ by
    up to 1 ULP, so a band about 1 ULP of ``h`` wide at ``0.99995 |R|``
    exists in which one route refuses a ray the other serves.  This arm
    makes the claim CONCRETE by constructing such a point with a
    directed ``nextafter`` walk (never by hoping a bundle finds one) and
    showing it reaches ``_refract``'s ``alive`` flag.

    Note which side is right: the closed form's gate has one fewer
    rounding, so it is the more accurate predicate; matching it to the
    generic route would make it worse.
    """
    found = None
    for R in (0.0515, -0.0345, -0.120, 0.5, -1.0):
        for az in (0.0, 0.3, math.pi / 4, 1.1, 2.7, 4.9):
            h = abs(R) * math.sqrt(0.9999)
            for _ in range(40):
                h = math.nextafter(h, 0.0)
            for _ in range(80):
                x, y = h * math.cos(az), h * math.sin(az)
                fast = (x * x + y * y) / (R * R) < 0.9999
                slow = math.sqrt(x * x + y * y) ** 2 / (R * R) < 0.9999
                if fast != slow:
                    found = (R, x, y, fast, slow)
                    break
                h = math.nextafter(h, math.inf)
            if found:
                break
        if found:
            break
    assert found is not None, (
        'no straddle point found: either the two gates now evaluate the '
        'same expression -- in which case this arm should become an '
        'equality pin -- or the walk no longer covers the band.  '
        'Measured 2026-09-20: R = -0.12 m, h/|R| = 0.9999499987499374.')
    R, x, y, fast_valid, slow_valid = found
    assert fast_valid != slow_valid
    surf = Surface(radius=R, thickness=0.0, glass_before='air',
                   glass_after='N-BK7', semi_diameter=abs(R) * 2)
    states = {}
    for spelling in (C2_SPHERE_NORMAL, PRE_C2_SPHERE_NORMAL):
        rays = _make_bundle(np.array([x]), np.array([y]),
                            np.array([0.0]), np.array([0.0]), WL)
        _isect._refract(rays, surf, 1.0, 1.5168, sphere_normal=spelling)
        states[spelling] = (bool(rays.alive[0]), int(rays.error_code[0]))
    assert states[C2_SPHERE_NORMAL] != states[PRE_C2_SPHERE_NORMAL], states


def test_c2_no_bundle_that_is_not_aimed_at_the_rim_band_lands_in_it():
    """The other side of the same decision, and the reason the Migration
    note can say the flip moves no shipped vignetting count.

    The band is about 1 ULP of ``h`` wide out of an aperture of order
    millimetres, so the chance a sampled ray lands in it is ~1e-16 per
    ray.  WP-C2 traced 360 000 rays over twelve prescription and
    field-angle combinations -- including a sphere whose clear aperture
    is opened to ``0.99999 |R|``, three field angles on a seven-surface
    stack, a two-mirror Cassegrain, and three shipped prescription
    builders -- and moved ZERO ``alive`` flags and ZERO error codes.
    A reduced version of that sweep runs here.
    """
    total_alive_moved = total_code_moved = 0
    for S, semi, tilts in ((_spherical7(), 0.0127, (0.0, 3.0, 8.0)),
                           (_cassegrain(), 0.050, (0.0,)),
                           ([Surface(radius=0.0515, thickness=0.050,
                                     glass_before='air',
                                     glass_after='N-BK7',
                                     semi_diameter=0.0515 * 0.99999),
                             Surface(radius=np.inf, thickness=0.0,
                                     glass_before='N-BK7',
                                     glass_after='air',
                                     semi_diameter=0.103)],
                            0.0515, (0.0,))):
        for tilt in tilts:
            rays = _bundle(8000, semi=semi, tilt=tilt)
            g = trace(rays, S, WL, output_filter='last',
                      sphere_normal=PRE_C2_SPHERE_NORMAL).image_rays
            a = trace(rays, S, WL, output_filter='last',
                      sphere_normal=C2_SPHERE_NORMAL).image_rays
            total_alive_moved += int(np.sum(np.asarray(g.alive)
                                            != np.asarray(a.alive)))
            total_code_moved += int(np.sum(np.asarray(g.error_code)
                                           != np.asarray(a.error_code)))
    assert total_alive_moved == 0 and total_code_moved == 0, (
        f'{total_alive_moved} alive flags and {total_code_moved} error '
        f'codes moved between the two normal routes on bundles that are '
        f'not aimed at the rim band.  The flip is entitled to move the '
        f'band and nothing else; this is a real vignetting change and '
        f'the Migration note has to name it.')


# ===========================================================================
# 6 -- the mirror sign
# ===========================================================================

def test_c2_a_spherical_mirror_reflects_about_the_outward_normal():
    """MUTATION ARM 4: a sign flip in ``_sphere_normal`` is caught here.

    Reflection is the one place a normal's SIGN survives into the
    answer: refraction through the shared ``refract_snell`` core
    orients the normal against the ray first, so ``n`` and ``-n`` give
    the same refracted direction, while a mirror's ``d_r = d_i + 2
    cos_i n`` does not care what orientation was chosen but DOES care
    that the normal is the radius vector of the surface actually hit.

    ORACLE: the law of reflection about the normal taken from the
    60-digit ``decimal`` oracle, evaluated independently of the library
    at the intersection point the library returned.  BAR: derived --
    the reflected direction is a linear combination of ``d`` and ``n``
    with coefficients of order 1, so its error is a small multiple of
    the normal's, i.e. tens of ULP; 1e-13 is four decades above that
    and eleven below a sign flip, which moves ``d_r`` by O(1).
    """
    S = _cassegrain()
    n = 512
    rng = np.random.default_rng(20260920)
    r = 0.045 * np.sqrt(rng.random(n))
    th = 2 * np.pi * rng.random(n)
    rays = _make_bundle(r * np.cos(th), r * np.sin(th),
                        np.zeros(n), np.zeros(n), WL)
    res = trace(rays, S, WL, output_filter='all')
    hit = res.rays_at(0)
    alive = np.asarray(hit.alive)
    assert alive.sum() > n // 2
    R = S[0].radius
    worst = 0.0
    idx = np.flatnonzero(alive)[::37][:12]
    for i in idx:
        nrm = _decimal_sphere_normal(float(hit.x[i]), float(hit.y[i]), R)
        d = np.array([0.0, 0.0, 1.0])          # the launch direction
        cos_i = -float(np.dot(d, nrm))
        if cos_i < 0:
            nrm, cos_i = -nrm, -cos_i
        want = d + 2.0 * cos_i * nrm
        got = np.array([float(hit.L[i]), float(hit.M[i]),
                        float(hit.N[i])])
        worst = max(worst, float(np.max(np.abs(got - want))))
    assert worst < 1e-13, (
        f'a spherical mirror no longer reflects about the outward '
        f'normal: worst |d_r - oracle| = {worst:.3e} against a 1e-13 '
        f'bar.  A sign flip in the closed-form normal moves this by '
        f'O(1); measured 2026-09-20 at the ULP level.')


def test_c2_the_predicate_is_what_selects_the_closed_form():
    """The closed form must be reached for exactly the surfaces the
    SHARED predicate accepts -- the failure mode of the v4.12.0 attempt
    was a normal that did not match the intersection's surface."""
    kinds = [
        (Surface(radius=0.05, thickness=0.0, glass_before='air',
                 glass_after='air', semi_diameter=0.02), True),
        (Surface(radius=0.05, thickness=0.0, glass_before='air',
                 glass_after='air', is_mirror=True,
                 semi_diameter=0.02), True),
        (Surface(radius=0.05, conic=-0.6, thickness=0.0,
                 glass_before='air', glass_after='air',
                 semi_diameter=0.02), False),
        (Surface(radius=0.05, aspheric_coeffs={4: 1.0e3},
                 thickness=0.0, glass_before='air', glass_after='air',
                 semi_diameter=0.02), False),
        (Surface(radius=0.05, radius_y=0.06, thickness=0.0,
                 glass_before='air', glass_after='air',
                 semi_diameter=0.02), False),
        (Surface(radius=np.inf, thickness=0.0, glass_before='air',
                 glass_after='air', semi_diameter=0.02), False),
    ]
    x = np.linspace(-0.01, 0.01, 21)
    y = 0.4 * x
    for surf, is_sphere in kinds:
        assert _is_pure_spherical(surf) is is_sphere, surf
        got = _surface_normal(x, y, surf, analytic_sphere=True)
        ref = _generic_normal(x, y, surf)
        same = all(np.array_equal(g, r) for g, r in zip(got, ref))
        # a NON-sphere must be byte-identical either way; a sphere must
        # take the other route (and so is allowed to differ)
        assert same == (not is_sphere), surf


# ===========================================================================
# 7 -- the mutation matrix, stated
# ===========================================================================

def test_c2_mutation_matrix_is_stated_and_each_arm_is_named():
    """Which test catches which regression -- written down so a future
    reader does not have to infer it, and asserted so the names cannot
    rot silently.

    | mutation                                   | caught by |
    |--------------------------------------------|-----------|
    | either default silently reverted (signature) | ``test_c2_the_defaults_are_what_the_ledger_decided_from_the_signature`` |
    | either default reverted inside the loop only | ``test_c2_the_defaults_are_what_a_call_actually_takes`` |
    | ``'generic'`` stops being the old arithmetic | ``test_c2_the_way_back_is_the_pre_5_49_0_arithmetic_exactly`` |
    | the closed form loses accuracy               | ``test_c2_the_closed_form_is_the_better_route_over_the_aperture`` |
    | it stops being a unit vector                 | ``test_c2_the_closed_form_is_a_unit_vector_by_construction`` |
    | the 0.9999 domain clamp moves or goes        | ``test_c2_the_domain_clamp_stays_where_the_ledger_left_it`` |
    | the rim band stops being the only difference | ``test_c2_no_bundle_that_is_not_aimed_at_the_rim_band_lands_in_it`` |
    | the normal's SIGN flips (mirror)             | ``test_c2_a_spherical_mirror_reflects_about_the_outward_normal`` |
    | the selection predicate widens or narrows    | ``test_c2_the_predicate_is_what_selects_the_closed_form`` |
    """
    import sys
    mod = sys.modules[__name__]
    for name in (
            'test_c2_the_defaults_are_what_the_ledger_decided_from_the_signature',
            'test_c2_the_defaults_are_what_a_call_actually_takes',
            'test_c2_the_way_back_is_the_pre_5_49_0_arithmetic_exactly',
            'test_c2_the_closed_form_is_the_better_route_over_the_aperture',
            'test_c2_the_closed_form_is_a_unit_vector_by_construction',
            'test_c2_the_domain_clamp_stays_where_the_ledger_left_it',
            'test_c2_no_bundle_that_is_not_aimed_at_the_rim_band_lands_in_it',
            'test_c2_a_spherical_mirror_reflects_about_the_outward_normal',
            'test_c2_the_predicate_is_what_selects_the_closed_form'):
        assert callable(getattr(mod, name, None)), (
            f'{name} named in the mutation matrix no longer exists; '
            f'either restore it or update the table above.')
    assert la is not None
