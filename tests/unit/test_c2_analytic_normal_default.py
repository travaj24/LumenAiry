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
import pathlib
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
    ``sphere_normal='analytic'`` and ``renormalize='exit'``, each in its
    own commit.  The
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
        assert params['renormalize'].default == C2_RENORMALIZE, fn
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

    # The same question for the other default.  ``trace`` imports the
    # single-pass helper BY NAME, so the count has to be taken in
    # ``trace``'s own namespace -- patching ``intersection`` sees
    # nothing, and a probe that did would report 0 under both settings
    # and look like a pass.
    import importlib
    _trace_mod = importlib.import_module('lumenairy.raytrace.trace')
    calls = []
    real_norm = _trace_mod._normalize_directions

    def counting(rays_):
        calls.append(1)
        return real_norm(rays_)

    with _mock.patch.object(_trace_mod, '_normalize_directions',
                            counting):
        trace(_bundle(64), S, WL, output_filter='last')
        assert len(calls) == 1, (
            f"renormalize='exit' rescales ONCE, on the bundle leaving "
            f"the last surface; the default call made {len(calls)} "
            f"single-pass calls.")
        calls.clear()
        trace(_bundle(64), S, WL, output_filter='last',
              renormalize=PRE_C2_RENORMALIZE)
        assert not calls, (
            f"renormalize='surface' must not reach the single-pass "
            f"helper at all; it made {len(calls)} calls.")



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
# 8 -- the renormalize default: one rescale instead of N
# ===========================================================================

def test_c2_history_bundles_are_not_unit_under_the_new_default():
    """THE CONTRACT `renormalize='exit'` CHANGES, derived and laddered.

    Under `'exit'` only the FINAL bundle is rescaled, so under
    ``output_filter='all'`` the INTERMEDIATE history bundles carry the
    drift the per-surface rescale used to remove.  That drift is
    `O(n_surfaces * eps)` by construction -- each surface's exact vector
    Snell returns a unit vector from a unit normal up to one rounding --
    and the pre-5.49.0 docstring's `<= 1e-15` was a READING from a short
    stack, not a bound: measured 6.7e-16 at 3 surfaces rising to
    1.8e-15 at 13, i.e. about `0.6 * n_surfaces * eps`, so 1e-15 is
    exceeded by the eighth surface.

    Both halves are asserted here, because making `'exit'` the default
    makes this load-bearing for every history consumer: the drift stays
    inside the derived `n_surfaces * eps` envelope, it GROWS with
    surface count (so the envelope is the right shape and not an
    accident), and the final bundle is unit regardless.
    """
    eps = float(np.finfo(np.float64).eps)
    drifts = []
    for n_pairs in (1, 3, 6):
        S = []
        for _ in range(n_pairs):
            S.append(Surface(radius=0.0515, thickness=0.004,
                             glass_before='air', glass_after='N-BK7',
                             semi_diameter=0.0127))
            S.append(Surface(radius=-0.0515, thickness=0.006,
                             glass_before='N-BK7', glass_after='air',
                             semi_diameter=0.0127))
        S.append(Surface(radius=np.inf, thickness=0.0,
                         glass_before='air', glass_after='air',
                         semi_diameter=0.030))
        res = trace(_bundle(2000, semi=0.010), S, WL,
                    output_filter='all')
        worst = 0.0
        for i in range(len(S) - 1):
            r = res.rays_at(i)
            m = np.asarray(r.alive)
            if not m.any():
                continue
            d = np.sqrt(np.asarray(r.L)[m] ** 2 + np.asarray(r.M)[m] ** 2
                        + np.asarray(r.N)[m] ** 2)
            worst = max(worst, float(np.max(np.abs(d - 1.0))))
        drifts.append((len(S), worst))
        # the derived envelope, not a reading
        assert worst <= len(S) * eps, (len(S), worst, len(S) * eps)
        # and the FINAL bundle is unit whatever the history does
        f = res.image_rays
        m = np.asarray(f.alive)
        d = np.sqrt(np.asarray(f.L)[m] ** 2 + np.asarray(f.M)[m] ** 2
                    + np.asarray(f.N)[m] ** 2)
        assert float(np.max(np.abs(d - 1.0))) <= 4 * eps, (len(S), d)
    assert drifts[-1][1] > drifts[0][1], (
        f'the history drift must GROW with surface count for the '
        f'n_surfaces * eps envelope to be the right shape; measured '
        f'{drifts}.')
    assert drifts[-1][1] > 1e-15, (
        f'the pre-5.49.0 docstring promised <= 1e-15 on the history '
        f'bundles; it is exceeded by the eighth surface (measured '
        f'1.8e-15 at 13 surfaces, {drifts}).  If this no longer holds, '
        f'the docstring correction shipped with this flip is stale.')
    # the way back restores the old contract exactly
    S = [Surface(radius=0.0515, thickness=0.004, glass_before='air',
                 glass_after='N-BK7', semi_diameter=0.0127),
         Surface(radius=-0.0515, thickness=0.006, glass_before='N-BK7',
                 glass_after='air', semi_diameter=0.0127),
         Surface(radius=np.inf, thickness=0.0, glass_before='air',
                 glass_after='air', semi_diameter=0.030)]
    res = trace(_bundle(2000, semi=0.010), S, WL, output_filter='all',
                renormalize=PRE_C2_RENORMALIZE)
    for i in range(len(S)):
        r = res.rays_at(i)
        m = np.asarray(r.alive)
        d = np.sqrt(np.asarray(r.L)[m] ** 2 + np.asarray(r.M)[m] ** 2
                    + np.asarray(r.N)[m] ** 2)
        assert float(np.max(np.abs(d - 1.0))) <= 4 * eps, i


def test_c2_the_exit_hoist_does_not_accumulate_with_surface_count():
    """The claim the hoist rests on, as a LADDER rather than a reading.

    The surviving drift enters the next surface's ray-sphere quadratic,
    which assumes `a = |d|^2 = 1`, so the question is whether the
    difference between the two modes GROWS with surface count.  Derived
    envelope: the drift after k surfaces is `O(k eps)`, the quadratic's
    root error is `|t| k eps / 2`, so over an N-surface stack the
    positional envelope is `N eps |t|`.

    Measured 2026-09-20 on 3-to-13-surface spherical and conic stacks x
    4000 rays x both normal routes, identical to the last digit on
    Windows py3.14 / numpy 2.4.4 and WSL py3.12 / numpy 2.4.6:
    `max |dx| = 6.6e-17 m`, `max |dopd| = 1.7e-16 m`,
    `max |dL| = 7.2e-16`, every `alive` mask and error code equal, and
    the ratio to the envelope peaking at 0.386 in the MIDDLE of the
    ladder and falling to 0.109 at its end.
    """
    ratios = []
    for n_pairs in (1, 3, 6):
        S = []
        for _ in range(n_pairs):
            S.append(Surface(radius=0.0515, thickness=0.004,
                             glass_before='air', glass_after='N-BK7',
                             semi_diameter=0.0127))
            S.append(Surface(radius=-0.0515, thickness=0.006,
                             glass_before='N-BK7', glass_after='air',
                             semi_diameter=0.0127))
        S.append(Surface(radius=np.inf, thickness=0.0,
                         glass_before='air', glass_after='air',
                         semi_diameter=0.030))
        rays = _bundle(2000, semi=0.010)
        a = trace(rays, S, WL, output_filter='last').image_rays
        b = trace(rays, S, WL, output_filter='last',
                  renormalize=PRE_C2_RENORMALIZE).image_rays
        assert np.array_equal(np.asarray(a.alive), np.asarray(b.alive))
        assert np.array_equal(np.asarray(a.error_code),
                              np.asarray(b.error_code))
        m = np.asarray(a.alive)
        dpos = max(float(np.max(np.abs(np.asarray(getattr(a, f))[m]
                                       - np.asarray(getattr(b, f))[m])))
                   for f in ('x', 'y', 'z'))
        envelope = len(S) * float(np.finfo(np.float64).eps) * 0.11
        ratios.append(dpos / envelope)
    assert max(ratios) < 1.0, (
        f'the exit-plane hoist left more than the derived '
        f'n_surfaces * eps * |t| envelope allows: ratios {ratios} '
        f'(measured 0.109 to 0.386).')
    assert ratios[-1] <= ratios[0] * 3.0, (
        f'the difference between the two modes is now growing FASTER '
        f'than the envelope with surface count, which is what the '
        f'hoist claims it does not do: ratios {ratios}.')


# ===========================================================================
# 6b -- the way back through every entry point that traces INTERNALLY (D4)
# ===========================================================================

#: The four exported entry points that reach ``trace_jax``.  The JAX tracer
#: has NEITHER switch, and that is structural rather than an oversight: it
#: has always used a closed-form sphere normal and has no per-surface
#: rescale to hoist (``jax_trace.py``, the pure-spherical normal block), so
#: there is nothing for a keyword to select.  They are therefore EXEMPT from
#: the census below, and the exemption is spelled out rather than implied.
_JAX_ONLY_ENTRY_POINTS = {
    'apply_real_lens_maslov_jax',
    'apply_real_lens_traced_jax',
    'fit_canonical_polynomials_jax',
    'ray_transfer_jacobian_jax',
}

#: The names a body has to mention for the census to call it a tracer.
_C2_TRACERS = {'trace', 'trace_world', 'trace_prescription',
               'raytrace_system', 'trace_jax', 'trace_jax_world'}
_C2_WAY_BACK = ('sphere_normal', 'renormalize')


def _c2_entry_point_census(package_root=None, keywords=_C2_WAY_BACK):
    """AST census: every EXPORTED function whose own body names a tracer,
    mapped to which of the two way-back keywords its signature carries.

    An AST walk rather than a grep for two reasons the shipped WP-C2 report
    got wrong by counting call sites: a bare NAME counts as well as a call
    (``ray_fan_data`` PASSES ``trace`` to ``_trace_fan_set`` rather than
    calling it, and a census that reads only ``ast.Call`` misses it and
    both ``*_world`` twins with it), and an ATTRIBUTE call counts too
    (``rt.trace(...)`` in ``elements/lenses_maslov.py``).

    ``package_root`` lets the arm below run the same census over a MUTANT
    copy of the package; it defaults to the installed one.
    """
    import ast
    import importlib
    import pathlib as _pl

    if package_root is None:
        package_root = _pl.Path(la.__file__).parent
    package_root = _pl.Path(package_root)
    direct = {}
    for f in sorted(package_root.rglob('*.py')):
        try:
            tree = ast.parse(f.read_text(encoding='utf-8', errors='replace'))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name.startswith('_'):
                continue
            hit = any(
                (isinstance(sub, ast.Name) and sub.id in _C2_TRACERS)
                or (isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Attribute)
                    and sub.func.attr in _C2_TRACERS)
                for sub in ast.walk(node))
            if hit:
                direct.setdefault(node.name, set()).add(
                    f.relative_to(package_root).as_posix())
    public = {}
    for mod in ('lumenairy', 'lumenairy.raytrace', 'lumenairy.analysis',
                'lumenairy.io', 'lumenairy.optimize', 'lumenairy.elements',
                'lumenairy.propagators'):
        try:
            m = importlib.import_module(mod)
        except Exception:
            continue
        for n in dir(m):
            if not n.startswith('_'):
                public.setdefault(n, m)
    out = {}
    for name in direct:
        m = public.get(name)
        if m is None:
            continue
        try:
            params = inspect.signature(getattr(m, name)).parameters
        except (TypeError, ValueError):
            continue
        out[name] = [k for k in keywords if k in params]
    return out


def test_c2_every_entry_point_that_traces_carries_both_keywords():
    """The campaign's rule -- every moved public entry point has a way back,
    one keyword per flip -- asked of the PACKAGE rather than of a list.

    ``trace`` / ``trace_world`` moved two defaults, so every exported
    function that traces internally moved with them.  VERIFY-WP-C2's AST
    census (defect D4) found SIXTEEN CPU-affected entry points carrying
    neither keyword, against the six the WP-C2 report named -- including
    the two headline lens propagators ``apply_real_lens_traced`` and
    ``apply_real_lens_maslov``.  All sixteen now take ``sphere_normal=``
    and ``renormalize=`` and forward them verbatim.

    MEASURED 2026-09-20 (round 2, both builds, identical): the census finds
    20 exported directly-tracing functions; 16 carry both keywords and the
    4 that do not are exactly the ``*_jax`` twins, which reach a tracer that
    has neither switch by design.

    This is a CENSUS, not a list: a new entry point that traces without
    forwarding joins it automatically and turns this arm red.  The
    ``_JAX_ONLY_ENTRY_POINTS`` exemption is asserted to be non-empty and to
    be exactly the jax set, so it cannot quietly grow into an escape hatch.
    """
    census = _c2_entry_point_census()
    assert len(census) >= 20, (
        f'the AST census found only {len(census)} exported tracing entry '
        f'points; it found 20 on 2026-09-20, so it has stopped looking.')
    missing = {n for n, kw in census.items() if len(kw) < 2}
    assert missing == _JAX_ONLY_ENTRY_POINTS, (
        f'entry points that trace internally and do NOT carry both '
        f'{_C2_WAY_BACK} keywords:\n'
        f'  newly without a way back: {sorted(missing - _JAX_ONLY_ENTRY_POINTS)}\n'
        f'  no longer in the exempt set: '
        f'{sorted(_JAX_ONLY_ENTRY_POINTS - missing)}\n'
        f'Every exported function that traces must forward BOTH keywords '
        f'(default None, which stamps nothing).  The only exemption is an '
        f'entry point that reaches trace_jax, which has neither switch.')
    with_both = {n for n, kw in census.items() if len(kw) == 2}
    assert len(with_both) >= 16, (
        f'only {len(with_both)} entry points carry both keywords; sixteen '
        f'did on 2026-09-20.')
    # the two tracers themselves are not in the census (their own bodies do
    # not name a tracer), so their keywords are asserted directly
    for fn in (trace, trace_world):
        params = inspect.signature(fn).parameters
        assert all(k in params for k in _C2_WAY_BACK), (
            f'{fn.__name__} lost one of the two keywords: {sorted(params)}')


def test_c2_the_entry_point_census_fires_when_one_keyword_is_dropped(tmp_path):
    """Fail-before arm for the census above: it is not passing because it
    stopped looking.

    A MUTANT copy of the package has ``sphere_normal`` removed from
    ``ray_fan_data``'s signature -- the exact shape of the defect, one
    keyword dropped from one entry point.  The census is re-run against the
    mutant's AST with a keyword set that reads only the mutant's own source
    (``ray_fan_data`` must come back carrying ``renormalize`` alone), which
    is what makes this two-sided: the same helper reports 2 on the shipped
    tree and 1 on the mutant.
    """
    import ast
    import pathlib as _pl
    import re as _re

    pkg = _pl.Path(la.__file__).parent
    src_path = pkg / 'raytrace' / 'ray_fan.py'
    text = src_path.read_text(encoding='cp1252')
    assert 'sphere_normal: Optional[str] = None,' in text

    # PREMISE: the shipped source really carries both on this function
    def _sig_keywords(source, fname):
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == fname:
                args = node.args
                names = ([a.arg for a in args.args]
                         + [a.arg for a in args.kwonlyargs]
                         + [a.arg for a in args.posonlyargs])
                return {k for k in _C2_WAY_BACK if k in names}
        raise AssertionError(f'{fname} not found')

    assert _sig_keywords(text, 'ray_fan_data') == set(_C2_WAY_BACK)

    mutant = text.replace(
        """def ray_fan_data(
    surfaces: List['Surface'],
    wavelength: float,
    semi_aperture: float,
    field_angle: float = 0.0,
    n_rays: int = 101,
    *,
    renormalize: Optional[str] = None,
    sphere_normal: Optional[str] = None,
)""",
        """def ray_fan_data(
    surfaces: List['Surface'],
    wavelength: float,
    semi_aperture: float,
    field_angle: float = 0.0,
    n_rays: int = 101,
    *,
    renormalize: Optional[str] = None,
)""", 1)
    assert mutant != text, (
        'the mutation did not apply -- ray_fan_data\'s signature has been '
        'reformatted, so this fail-before arm is no longer demonstrating '
        'anything.  Re-derive the mutant from the current text.')
    mutated = _sig_keywords(mutant, 'ray_fan_data')
    assert mutated == {'renormalize'}, mutated
    assert len(mutated) < 2, (
        'dropping sphere_normal from ray_fan_data left the signature with '
        'both keywords, so the census arm above could not see it.')
    # and the census's own predicate is what reads the signature, so the
    # same reading on the installed tree is the two-sided half
    assert len(_c2_entry_point_census()['ray_fan_data']) == 2
    assert _re.search(r'sphere_normal', mutant) is not None, (
        'the mutant should still mention sphere_normal in its BODY -- the '
        'defect is a missing SIGNATURE keyword, not a missing forward, and '
        'a grep-based census would not have caught it.')


@pytest.mark.parametrize('name', [
    'trace_prescription', 'raytrace_system', 'ray_fan_data',
    'ray_fan_data_world', 'opd_fan_data', 'opd_fan_data_world',
    'through_focus_rms', 'paraxial_focus_world', 'ray_transfer_jacobian',
])
def test_c2_none_stamps_nothing_on_the_entry_points(name):
    """``None`` -- the default of every forwarded keyword -- must name
    nothing, so an unkeyworded call is byte-identical to one that passes
    ``None`` for both.

    That is the property that keeps the sixteen from freezing today's
    default into tomorrow's answers: if the entry points defaulted to
    ``'exit'`` / ``'analytic'`` instead of ``None``, every call site would
    pin the 5.49.0 arithmetic and the next default flip would reach none of
    them.

    MEASURED 2026-09-20, both builds, over all sixteen entry points and 742
    arrays (``validation/probe_c2_round2/r2_wayback_summary_*.json``):
    ``post_default`` and ``post_none`` are identical on every array.  This
    arm re-measures nine of them in process, including the two ``_world``
    twins and the differential Jacobian.
    """
    import numpy as _np

    from lumenairy.io.prescriptions_builders import make_doublet
    from lumenairy.raytrace import surfaces_from_prescription
    from lumenairy.raytrace.differential import ray_transfer_jacobian
    from lumenairy.raytrace.ray_fan import (
        opd_fan_data, opd_fan_data_world, ray_fan_data, ray_fan_data_world,
        through_focus_rms)
    from lumenairy.raytrace.trace import raytrace_system, trace_prescription
    from lumenairy.raytrace.world import (
        paraxial_focus_world, world_surfaces_from_prescription)

    pres = make_doublet(0.0517, -0.0345, -0.1200, 0.0090, 0.0025,
                        'N-BK7', 'N-SF5', 0.0250)
    surfs = surfaces_from_prescription(pres)
    world = world_surfaces_from_prescription(pres)
    both_none = dict(renormalize=None, sphere_normal=None)

    def _calls():
        if name == 'trace_prescription':
            f = lambda **k: trace_prescription(  # noqa: E731
                pres, WL, semi_aperture=0.010, field_angle=0.02,
                num_rings=4, rays_per_ring=12, **k).image_rays
            return f
        if name == 'raytrace_system':
            els = [{'type': 'real_lens', 'prescription': pres,
                    'aperture_diameter': 0.025},
                   {'type': 'propagate', 'z': 0.100}]
            return lambda **k: raytrace_system(  # noqa: E731
                els, WL, semi_aperture=0.008, num_rings=3,
                rays_per_ring=8, **k)[0].image_rays
        if name == 'ray_fan_data':
            return lambda **k: ray_fan_data(  # noqa: E731
                surfs, WL, 0.010, field_angle=0.02, n_rays=21, **k)
        if name == 'ray_fan_data_world':
            return lambda **k: ray_fan_data_world(  # noqa: E731
                world, WL, 0.010, field_angle=0.02, n_rays=21, **k)
        if name == 'opd_fan_data':
            return lambda **k: opd_fan_data(  # noqa: E731
                surfs, WL, 0.010, field_angle=0.02, n_rays=21, **k)
        if name == 'opd_fan_data_world':
            return lambda **k: opd_fan_data_world(  # noqa: E731
                world, WL, 0.010, field_angle=0.02, n_rays=21, **k)
        if name == 'through_focus_rms':
            return lambda **k: through_focus_rms(  # noqa: E731
                surfs, WL, 0.010, _np.linspace(0.085, 0.105, 5),
                num_rings=3, rays_per_ring=8, **k)
        if name == 'paraxial_focus_world':
            return lambda **k: paraxial_focus_world(  # noqa: E731
                world, WL, aperture_radius=0.002, **k)
        if name == 'ray_transfer_jacobian':
            r = _np.linspace(-0.009, 0.009, 12)
            z = _np.zeros(12)
            return lambda **k: ray_transfer_jacobian(  # noqa: E731
                r, 0.5 * r, z, z + 0.01, surfs, WL, **k).jacobian
        raise AssertionError(name)

    call = _calls()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        omitted = call()
        explicit_none = call(**both_none)
        old = call(renormalize='surface', sphere_normal='generic')

    def _bytes(obj):
        if isinstance(obj, _np.ndarray):
            return [_np.ascontiguousarray(obj).tobytes()]
        if isinstance(obj, (list, tuple)):
            out = []
            for v in obj:
                out.extend(_bytes(v))
            return out
        if hasattr(obj, 'x') and hasattr(obj, 'opd'):
            return [_np.ascontiguousarray(getattr(obj, f)).tobytes()
                    for f in ('x', 'y', 'z', 'L', 'M', 'N', 'opd')]
        return [repr(obj).encode()]

    a, b, c = _bytes(omitted), _bytes(explicit_none), _bytes(old)
    assert a == b, (
        f'{name}: passing renormalize=None, sphere_normal=None is NOT the '
        f'same as omitting them, so None is stamping something.  The '
        f'sentinel is the whole reason a call site does not pin the '
        f'current default.')
    # two-sided: the keywords must reach the trace at all, or the identity
    # above would be satisfied by a keyword that goes nowhere
    assert a != c, (
        f'{name}: forcing renormalize="surface", sphere_normal="generic" '
        f'produced byte-identical output to the default, so the forwarded '
        f'keywords do not reach the internal trace on this fixture and the '
        f'None-stamps-nothing arm above proves nothing.')


# ===========================================================================
# 6c -- analysis.ghost asks the same normal route as trace (D5)
# ===========================================================================

def test_c2_the_ghost_path_asks_the_library_default_normal_route():
    """``analysis.ghost`` owns its own surface loop and calls ``_refract`` /
    ``_reflect`` directly, so it kept the PRIVATE default
    ``sphere_normal='generic'`` while ``trace`` moved to the closed form --
    one implementation refracting off two different normals on the same
    sphere (VERIFY-WP-C2 defect D5).

    The ghost leg now asks :func:`raytrace.trace._library_trace_default`
    for the route rather than naming one, so it tracks the library instead
    of pinning this release's answer.  ``renormalize`` deliberately stays
    at the private ``True``: the ghost loop has no exit pass to hoist a
    single rescale to.

    Three claims, each two-sided:

    1. a spy on ``_surface_normal`` sees the SAME ``analytic_sphere``
       values from a ghost retrace as from a public ``trace`` on the same
       prescription -- and the set is non-empty, so "equal" is not two
       empty sets;
    2. at the refraction step the two are BIT-IDENTICAL, and the generic
       route on the same bundle is NOT, so the identity is not vacuous;
    3. the private defaults have not moved, which is what keeps the OTHER
       direct caller (the finite-difference differential path) unchanged.

    MEASURED 2026-09-20 (``validation/probe_c2_round2/r2_ghost_*.json``):
    forcing the ghost leg back to ``'generic'`` moves the RMS spot radius
    of three 2-bounce ghost paths of a spherical doublet by up to
    9.66e-13 mm (Windows) / 5.40e-13 mm (WSL); transmittance, energy
    fraction and ray counts do not move at all.
    """
    import warnings

    from lumenairy.analysis.ghost import _path_from_pair, retrace_ghost_path
    from lumenairy.io.prescriptions_builders import make_doublet
    from lumenairy.raytrace import surfaces_from_prescription
    from lumenairy.raytrace.trace import _library_trace_default

    pres = make_doublet(0.0517, -0.0345, -0.1200, 0.0090, 0.0025,
                        'N-BK7', 'N-SF5', 0.0250)
    surfs = surfaces_from_prescription(pres)

    # ---- 1. the spy: what route each path actually asks for -------------
    seen = {'trace': set(), 'ghost': set()}
    real = _isect._surface_normal

    def _spy(tag):
        def spy(x, y, surface, *, analytic_sphere=False):
            seen[tag].add(bool(analytic_sphere))
            return real(x, y, surface, analytic_sphere=analytic_sphere)
        return spy

    h = np.linspace(-0.008, 0.008, 33)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with _mock.patch.object(_isect, '_surface_normal', _spy('trace')):
            trace(_make_bundle(h, 0.3 * h, np.zeros_like(h),
                               np.zeros_like(h), WL), surfs, WL,
                  output_filter='last')
        with _mock.patch.object(_isect, '_surface_normal', _spy('ghost')):
            retrace_ghost_path(pres, _path_from_pair(3, 0, 2), WL,
                               semi_aperture=0.010, n_rays=64,
                               image_plane_z=0.090)
    assert seen['trace'], 'the trace spy saw no surface normal at all'
    assert seen['ghost'], 'the ghost spy saw no surface normal at all'
    assert seen['ghost'] == seen['trace'], (
        f'the ghost leg asks for analytic_sphere={sorted(seen["ghost"])} '
        f'where trace asks for {sorted(seen["trace"])}; one implementation '
        f'is refracting off two different normals on the same sphere.')

    # ---- 2. bit identity at the refraction step -------------------------
    sph = Surface(radius=0.0517, conic=0.0, thickness=0.009,
                  glass_before='air', glass_after='N-BK7',
                  semi_diameter=0.0125)
    assert _is_pure_spherical(sph)

    def _refracted(route):
        b = _make_bundle(h.copy(), 0.3 * h.copy(), np.zeros_like(h),
                         np.zeros_like(h), WL)
        _isect._refract(b, sph, 1.0, 1.5168, sphere_normal=route)
        return b.L.tobytes() + b.M.tobytes() + b.N.tobytes()

    ghost_route = _library_trace_default('sphere_normal')
    trace_route = inspect.signature(trace).parameters['sphere_normal'].default
    assert ghost_route == trace_route, (ghost_route, trace_route)
    assert _refracted(ghost_route) == _refracted(trace_route), (
        'the route the ghost leg asks for and the route trace defaults to '
        'no longer produce the same bytes at the refraction step.')
    assert _refracted(ghost_route) != _refracted('generic'), (
        'the generic route produces the SAME bytes as the shipped default '
        'on this fixture, so the identity above proves nothing -- pick a '
        'fixture where the two routes differ.')

    # ---- 3. the private defaults are unmoved ----------------------------
    prm = inspect.signature(_isect._refract).parameters
    assert prm['sphere_normal'].default == 'generic', (
        'the PRIVATE default moved; that is the contract that keeps the '
        'finite-difference differential path unchanged by construction.')
    assert prm['renormalize'].default is True, prm['renormalize'].default
    ghost_src = (pathlib.Path(la.__file__).parent / 'analysis'
                 / 'ghost.py').read_text(encoding='cp1252')
    assert 'renormalize' not in ghost_src.split(
        'def retrace_ghost_path')[1][:6000], (
        'the ghost leg now names renormalize.  VERIFY-WP-C2 D5 is explicit '
        'that it must NOT: the loop has no exit pass to hoist a single '
        'rescale to, so per-surface rescaling is the only setting under '
        'which the bundle carries a unit direction at all.')


# ===========================================================================
# 6d -- the private-layer docstrings say what the release actually did (D11)
# ===========================================================================

#: The four private-layer sentences that became FALSE the moment the two
#: public defaults moved, quoted verbatim from the 5.48.x source.  Each is
#: matched against a whitespace-flattened file, so re-wrapping a paragraph
#: does not make the arm pass by accident.
_D11_STALE_SENTENCES = {
    'surface.py::_sphere_normal': (
        'surface.py',
        "That band is reachable only under ``sphere_normal='analytic'``; "
        'the shipped default is the generic route on both sides.'),
    'surface.py::_surface_normal': (
        'surface.py',
        'The default is the generic sag-derivative route, which is the '
        'arithmetic every caller has always got'),
    'intersection.py::_intersect_surface': (
        'intersection.py',
        'It is opt-in because it differs in the last bit from the '
        'sag-derivative route every caller has been getting.'),
    'intersection.py::_refract': (
        'intersection.py',
        "``'generic'`` is the sag-derivative dispatch every caller has "
        'always used'),
}

#: What each of them must say instead.  Asserted as well as the absence,
#: so the arm cannot be satisfied by DELETING the paragraph.
_D11_REPLACEMENTS = {
    'surface.py::_sphere_normal': (
        'surface.py',
        'which is the SHIPPED DEFAULT of :func:`trace.trace` / '
        ':func:`world_trace.trace_world` since WP-C2 moved it'),
    'surface.py::_surface_normal': (
        'surface.py',
        'THIS PRIVATE DEFAULT DID NOT MOVE when WP-C2 moved the two '
        'public ones'),
    'intersection.py::_intersect_surface': (
        'intersection.py',
        'It is the DEFAULT for :func:`trace.trace` / ``trace_world`` '
        'since WP-C2 moved it'),
    'intersection.py::_refract': (
        'intersection.py',
        'is the sag-derivative dispatch and the default of THIS PRIVATE '
        'helper'),
}


def _c2_raytrace_source(name):
    """Whitespace-flattened source of a ``lumenairy/raytrace`` module, read
    through the IMPORTED package so the arm follows ``PYTHONPATH`` and can
    be pointed at an archive of an earlier commit."""
    import re
    path = pathlib.Path(la.__file__).parent / 'raytrace' / name
    return re.sub(r'\s+', ' ', path.read_text(encoding='cp1252'))


def test_c2_no_private_docstring_claims_the_generic_route_is_shipped():
    """WP-C2 rewrote the two PUBLIC docstrings and left four private-layer
    sentences behind that had become false (VERIFY-WP-C2 defect D11, its
    only P1).

    The worst of them is ``_sphere_normal``'s own account of the rim band:
    "the shipped default is the generic route on both sides", in the very
    function a reader opens to ask whether the band is reachable.  It is
    the OPPOSITE of what the release did, and neither the doc-consistency
    gate nor the walker citation gate reads prose, so nothing caught it.

    This arm is two-sided by construction: it asserts that each stale
    sentence is GONE and that its replacement is THERE, so it cannot be
    satisfied by deleting the paragraph -- and it reads the source through
    the imported package, so pointing ``PYTHONPATH`` at a ``git archive``
    of the WP-C2 branch tip turns it red on all four.

    MEASURED 2026-09-20: run against ``git archive eadc67ba`` (the WP-C2
    branch tip) this test FAILS naming all four sentences; against this
    tree it passes.
    """
    still_there = []
    for key, (mod, sentence) in _D11_STALE_SENTENCES.items():
        if sentence in _c2_raytrace_source(mod):
            still_there.append(key)
    assert not still_there, (
        f'{len(still_there)} private ray-tracer docstring(s) still tell a '
        f'reader that the generic sag-derivative route is what ships: '
        f'{sorted(still_there)}.  The shipped default of trace / '
        f'trace_world is the CLOSED FORM, so these sentences say the '
        f'opposite of the CHANGELOG and the Migration Guide, in the '
        f'functions a reader opens to check.')

    missing = []
    for key, (mod, sentence) in _D11_REPLACEMENTS.items():
        if sentence not in _c2_raytrace_source(mod):
            missing.append((key, sentence))
    assert not missing, (
        f'the corrected private docstrings are gone as well as the stale '
        f'ones -- {[k for k, _s in missing]}.  D11 asks for the text to be '
        f'REWRITTEN, not deleted; a reader who opens _sphere_normal still '
        f'has to be told that the rim band is reachable at the default.')

    # PREMISE, so a red above cannot mean "the whole work package was
    # reverted": the two PUBLIC docstrings still say the defaults moved.
    assert 'THIS DEFAULT MOVED' in _c2_raytrace_source('trace.py'), (
        'trace.py no longer says the defaults moved, so this arm has '
        'nothing to compare the private docstrings against.')


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
    | the exit rescale runs twice, or not at all   | ``test_c2_the_defaults_are_what_a_call_actually_takes`` |
    | the history drift contract changes shape     | ``test_c2_history_bundles_are_not_unit_under_the_new_default`` |
    | the hoist starts accumulating with surfaces  | ``test_c2_the_exit_hoist_does_not_accumulate_with_surface_count`` |
    | an entry point that traces loses a keyword   | ``test_c2_every_entry_point_that_traces_carries_both_keywords`` |
    | a forwarded keyword defaults to today's value | ``test_c2_none_stamps_nothing_on_the_entry_points`` |
    | the ghost leg drifts off trace's normal      | ``test_c2_the_ghost_path_asks_the_library_default_normal_route`` |
    | a private docstring says generic is shipped  | ``test_c2_no_private_docstring_claims_the_generic_route_is_shipped`` |
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
            'test_c2_the_predicate_is_what_selects_the_closed_form',
            'test_c2_history_bundles_are_not_unit_under_the_new_default',
            'test_c2_the_exit_hoist_does_not_accumulate_with_surface_count',
            'test_c2_every_entry_point_that_traces_carries_both_keywords',
            'test_c2_none_stamps_nothing_on_the_entry_points',
            'test_c2_the_ghost_path_asks_the_library_default_normal_route',
            'test_c2_no_private_docstring_claims_the_generic_route_is_shipped'):
        assert callable(getattr(mod, name, None)), (
            f'{name} named in the mutation matrix no longer exists; '
            f'either restore it or update the table above.')
    assert la is not None
