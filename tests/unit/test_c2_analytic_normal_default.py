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
    1.8e-15 at 13.

    THE COEFFICIENT IS NOT CONSTANT (VERIFY-WP-C2 defect D3).  The first
    version of this correction said "about `0.6 * n_surfaces * eps`",
    which is a reading at the LONG end.  Re-measured on this fixture's
    own ladder, identical on both builds:

        surfaces      3      5      7      9     11     13
        drift    6.66e-16 8.88e-16 1.22e-15 1.67e-15 1.67e-15 1.78e-15
        / n*eps    1.000  0.800  0.786  0.833  0.682  0.615

    ROUND 3 (VERIFY-WP-C2 round 2, defect VR2-D2): the envelope is
    `2 * n_surfaces * eps`, not `n_surfaces * eps`.  `n eps` holds on
    THIS ladder -- one element repeated -- and is exceeded on ordinary
    stacks of different radii and glasses.  Re-measured this round over
    90 combinations of surface count (3, 5, 7, 9, 13), glass (N-BK7,
    N-SF5, N-SF11), radius pair and field angle (0, 2, 5 deg),
    identical to the last digit on BOTH builds
    (`validation/probe_c2_round3/r3_history_drift_{win,wsl}.json`):

        exceeding n eps                 21 of 90
        worst ratio to n eps            1.6667 (3 surfaces, N-SF11, 5 deg:
                                        1.1102e-15 against 6.6613e-16)
        exceeding 2 n eps                0 of 90
        worst ratio to 2 n eps          0.8333 -- 1.2x of headroom
        ratio range, 3 -> 13 surfaces   1.00-1.67 -> 0.50-0.96
        1e-15 first exceeded at         3, 5 or 7 surfaces by stack
        final bundle drift              2.2e-16 on every one of the 90

    so 0.6 is 40 % under on a triplet -- the commonest case -- and so is
    1.0.  On THIS ladder `1e-15` is first exceeded at the SEVENTH
    surface (1.22e-15 against 8.88e-16 at five), which is why the band
    below is 5..9 and survives the restatement.

    Four claims are asserted here, because making `'exit'` the default
    makes this load-bearing for every history consumer: the drift stays
    inside the derived `2 * n_surfaces * eps` envelope, it GROWS with
    surface count (so the envelope is the right shape and not an
    accident), the coefficient FALLS with surface count (so the bound
    cannot be restated as a constant times `n * eps`), and the final
    bundle is unit regardless.
    """
    eps = float(np.finfo(np.float64).eps)
    drifts = []
    for n_pairs in (1, 2, 3, 6):
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
        # the derived envelope, not a reading.  VERIFY-WP-C2 round 2
        # (2026-09-20): the envelope is 2 n eps.  n eps holds on THIS
        # ladder and is exceeded by up to 1.6667x on an ordinary stack
        # of different radii and glasses -- 21 of 90 (surface count,
        # glass, radius pair, field angle) combinations exceed it, worst
        # 1.1102e-15 against 6.6613e-16 on a 3-surface N-SF11 stack at
        # 5 deg; 2 n eps holds on all 90, worst reading 0.8333 of it,
        # both builds identical to the last digit.
        assert worst <= 2 * len(S) * eps, (len(S), worst, 2 * len(S) * eps)
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
        f'bundles; measured 1.8e-15 at 13 surfaces ({drifts}).  If this '
        f'no longer holds, the docstring correction shipped with this '
        f'flip is stale.')
    # D3: the coefficient in front of n*eps is NOT constant, so the bound
    # cannot be restated as "about 0.6 n eps".  Measured 1.000 at three
    # surfaces against 0.615 at thirteen on BOTH builds -- a 1.63x fall --
    # and the bar is a RATIO, so nothing here pins a reading.
    ratios = [(n, d / (n * eps)) for n, d in drifts]
    assert all(r <= 2.0 + 4 * eps for _n, r in ratios), (
        f'the drift left the 2 * n_surfaces * eps envelope: {ratios}')
    first, last = ratios[0][1], ratios[-1][1]
    assert first > 1.25 * last, (
        f'the ratio to n_surfaces * eps no longer FALLS with surface '
        f'count ({ratios}), so "the coefficient is not constant" -- the '
        f'whole point of the D3 restatement -- has stopped being true '
        f'and the docstring should say a constant after all.  Measured '
        f'1.000 at 3 surfaces against 0.615 at 13 (1.63x) on both '
        f'builds, against a 1.25x bar.')
    # and 1e-15 is first exceeded at SEVEN surfaces, not eight.  Asserted
    # as a BAND rather than an equality: the quantity is deterministic and
    # identical on both development builds, but it is a sum of last-bit
    # roundings and a numpy release is entitled to move it by a rung.
    exceeding = [n for n, d in drifts if d > 1e-15]
    assert exceeding, (
        f'1e-15 is no longer exceeded anywhere on the ladder ({drifts}); '
        f'the retired bound has stopped being wrong, so the docstring '
        f'correction is stale.')
    assert 5 <= exceeding[0] <= 9, (
        f'1e-15 is first exceeded at {exceeding[0]} surfaces ({drifts}); '
        f'measured SEVEN on both development builds (1.22e-15 against '
        f'8.88e-16 at five).  Re-measure and restate the docstring.')
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

#: Exported functions that reach a tracer ONLY through a PRIVATE helper of
#: their own module, and carry neither way-back keyword.  WP-C2 round 3
#: (2026-09-20) gave the census that second hop -- which is what finally
#: named ``apply_real_lens``, the seventeenth entry point (VR2-D1) -- and
#: the same hop made this twelve visible for the first time.
#:
#: They are RECORDED rather than closed.  Closing them is twelve signatures
#: and six private helpers (``analysis/field.py::_trace`` alone serves six
#: of them), each needing its own archive-to-archive byte-identity proof and
#: its own forwarding pin, which is a work package of the size of WP-C2
#: itself and not something to land unverified in a round that is closing
#: seven filed defects.  Measured on both builds on 2026-09-20; the set is
#: allowed to SHRINK (give one the pair and it leaves) and never to grow,
#: so a NEW helper-routed entry point still turns the census red.
_C2_TRACES_VIA_A_PRIVATE_HELPER = {
    'apply_prescription_persurface_to_beamlets',
    'apply_real_lens_traced_multibranch',
    'apply_real_lens_traced_uniform',
    'distortion_grid',
    'distortion_vs_field',
    'field_aberration_sweep',
    'footprint_per_surface',
    'propagate_hfpi_through_prescription',
    'propagate_traced_carrier_chain',
    'propagate_traced_carrier_chain_multi',
    'relative_illumination',
    'spot_diagram_vs_field',
}

#: The names a body has to mention for the census to call it a tracer.
_C2_TRACERS = {'trace', 'trace_world', 'trace_prescription',
               'raytrace_system', 'trace_jax', 'trace_jax_world'}
_C2_WAY_BACK = ('sphere_normal', 'renormalize')


def _c2_entry_point_census(package_root=None, keywords=_C2_WAY_BACK,
                           hop=True):
    """AST census: every EXPORTED function whose own body names a tracer,
    mapped to which of the two way-back keywords its signature carries.

    An AST walk rather than a grep for two reasons the shipped WP-C2 report
    got wrong by counting call sites: a bare NAME counts as well as a call
    (``ray_fan_data`` PASSES ``trace`` to ``_trace_fan_set`` rather than
    calling it, and a census that reads only ``ast.Call`` misses it and
    both ``*_world`` twins with it), and an ATTRIBUTE call counts too
    (``rt.trace(...)`` in ``elements/lenses_maslov.py``).

    ALIAS-AWARE since round 3 (VERIFY-WP-C2 round 2, defect VR2-D1).  A
    body that imports the tracer under another name --
    ``from ..raytrace import trace as _rt_trace`` inside
    ``_apply_real_lens_impl`` -- names ``_rt_trace``, not ``trace``, so a
    census keyed on the tracer NAMES misses it.  That is how
    ``apply_real_lens`` hid from both this census and VERIFY-WP-C2's own
    for two rounds while tracing internally and MOVING archive to archive
    (2.8389e-13 Windows / 2.8387e-13 WSL).  The module's ``Import`` /
    ``ImportFrom`` aliases are collected first -- including the ones inside
    function bodies, which is where the lens modules put theirs to break an
    import cycle -- and a bare ``Name`` bound to one of them counts as a
    tracer mention.

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
        # ``{local name: tracer it is bound to}`` for the WHOLE module, so a
        # function-body import is seen by the walk of any function in it.
        aliases = {a.asname: a.name
                   for node in ast.walk(tree)
                   if isinstance(node, (ast.Import, ast.ImportFrom))
                   for a in node.names
                   if a.asname and a.name in _C2_TRACERS}
        traces = {}
        defs = {}
        names_used = {}
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            defs[node.name] = node
            used = set()
            hit = False
            for sub in ast.walk(node):
                if isinstance(sub, ast.Name):
                    used.add(sub.id)
                    if sub.id in _C2_TRACERS or sub.id in aliases:
                        hit = True
                elif (isinstance(sub, ast.Call)
                        and isinstance(sub.func, ast.Attribute)):
                    if sub.func.attr in _C2_TRACERS:
                        hit = True
            names_used[node.name] = used
            if hit:
                traces.setdefault(node.name, set()).add(
                    f.relative_to(package_root).as_posix())
        # ONE MORE HOP, through PRIVATE helpers of the SAME module.  An
        # exported entry point whose whole body is
        # ``return _apply_real_lens_impl(...)`` traces exactly as much as
        # the impl does, and a census that stops at the public body calls
        # it route-less -- which is how ``apply_real_lens`` (VR2-D1) hid
        # behind BOTH the private split-out AND the import alias.  Only
        # underscore-prefixed callees defined IN THIS FILE are followed, so
        # this stays a census of entry points and does not become the
        # 46-function transitive-caller population.
        changed = bool(hop)
        while changed:
            changed = False
            for name, used in names_used.items():
                if name in traces:
                    continue
                for callee in used:
                    if (callee.startswith('_') and callee in defs
                            and callee in traces):
                        traces.setdefault(name, set()).add(
                            f.relative_to(package_root).as_posix())
                        changed = True
                        break
        for name in traces:
            if name.startswith('_'):
                continue
            direct.setdefault(name, set()).update(traces[name])
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

    ROUND 3 (VERIFY-WP-C2 round 2, defect VR2-D1).  Both the shipped
    census and VERIFY-WP-C2's own decided "this body traces" from the
    tracer NAMES, and ``elements/_lens_real.py`` imports the tracer inside
    a function body as ``trace as _rt_trace`` and calls it from the
    PRIVATE split-out ``_apply_real_lens_impl``.  Two blind spots in
    series hid a SEVENTEENTH entry point --
    ``apply_real_lens(seidel_correction=True)``, the plain-lens propagator
    most users reach for first -- which traced with both keywords omitted
    and MOVED archive to archive (max abs delta 2.8389e-13 Windows /
    2.8387e-13 WSL on a 256 x 256 field).  The census now resolves import
    aliases and follows one hop into a private same-module helper, and
    ``apply_real_lens`` carries the pair.

    MEASURED 2026-09-20 (round 3, both builds, identical): 20 exported
    functions trace in their OWN bodies and 33 once the private-helper hop
    is followed; 17 carry both keywords.  Of the 16 that do not, 4 are
    exactly the ``*_jax`` twins, which reach a tracer that has neither
    switch by design, and 12 reach a tracer only through a private helper
    of their own module -- ``_C2_TRACES_VIA_A_PRIVATE_HELPER``, which
    round 3 made visible for the first time and recorded rather than
    closed.

    This is a CENSUS, not a list: a new entry point that traces without
    forwarding joins it automatically and turns this arm red.  Both
    exemptions are asserted to be non-empty, disjoint and exactly what
    they say, so neither can quietly grow into an escape hatch.
    """
    # the library names the pair itself; this census must be checking the
    # same two keywords the tracer documents as its way back, or it is
    # policing something else.
    from lumenairy.raytrace.trace import _WAY_BACK_KEYWORDS
    assert set(_WAY_BACK_KEYWORDS) == set(_C2_WAY_BACK), (
        f'raytrace.trace._WAY_BACK_KEYWORDS names {_WAY_BACK_KEYWORDS} and '
        f'this census checks {_C2_WAY_BACK}; they have to be the same pair.')

    # neither exemption may go empty or overlap -- an empty set is an
    # escape hatch with nothing written in it
    assert _C2_TRACES_VIA_A_PRIVATE_HELPER and _JAX_ONLY_ENTRY_POINTS, (
        'an exemption set went empty; delete it rather than leaving an '
        'unexplained escape hatch in the census.')
    assert not (_C2_TRACES_VIA_A_PRIVATE_HELPER & _JAX_ONLY_ENTRY_POINTS), (
        'the two exemptions overlap, so one of them is not saying what it '
        'says.')

    census = _c2_entry_point_census()
    assert len(census) >= 33, (
        f'the AST census found only {len(census)} exported tracing entry '
        f'points; it found 33 on 2026-09-20 (20 of them tracing in their '
        f'own bodies), so it has stopped looking.')
    exempt = _JAX_ONLY_ENTRY_POINTS | _C2_TRACES_VIA_A_PRIVATE_HELPER
    missing = {n for n, kw in census.items() if len(kw) < 2}
    assert missing == exempt, (
        f'entry points that trace internally and do NOT carry both '
        f'{_C2_WAY_BACK} keywords:\n'
        f'  newly without a way back: {sorted(missing - exempt)}\n'
        f'  no longer in the exempt set: {sorted(exempt - missing)}\n'
        f'Every exported function that traces in its OWN body must forward '
        f'BOTH keywords (default None, which stamps nothing).  The two '
        f'exemptions are an entry point that reaches trace_jax, which has '
        f'neither switch by design, and one that reaches a tracer only '
        f'through a private helper of its own module, which round 3 '
        f'recorded rather than closed.')
    with_both = {n for n, kw in census.items() if len(kw) == 2}
    assert len(with_both) >= 17, (
        f'only {len(with_both)} entry points carry both keywords; '
        f'seventeen did on 2026-09-20, after VR2-D1 was closed.')
    # VR2-D1: the seventeenth, named.  It is the one entry point the census
    # reaches through BOTH an import alias and a private split-out, so if
    # either resolution is lost this line says which.
    assert census.get('apply_real_lens') == list(_C2_WAY_BACK), (
        f'apply_real_lens -- the seventeenth entry point, which traces '
        f'through the aliased import in _apply_real_lens_impl whenever '
        f'seidel_correction=True -- reads '
        f'{census.get("apply_real_lens")!r} in the census.  Either it lost '
        f'the way back or the census lost the alias / private-helper '
        f'resolution that finds it (VR2-D1).')
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


def test_c2_the_census_names_a_route_less_caller_hidden_by_an_alias(tmp_path):
    """Fail-before arm for the two resolutions VR2-D1 added.

    The defect was not "a keyword was forgotten"; it was that the census
    COULD NOT SEE the entry point.  ``elements/_lens_real.py`` imports the
    tracer inside a function body as ``trace as _rt_trace`` and calls it
    from the PRIVATE split-out ``_apply_real_lens_impl``, so a census that
    keys on the tracer NAMES and stops at the public body reads
    ``apply_real_lens`` as not tracing at all -- which both the shipped
    census and VERIFY-WP-C2's own did, for two rounds.

    MEASURED 2026-09-20 on this verifier's own ``git archive 49ddf4bd``
    tree and on the tip, both builds, identical
    (``validation/probe_c2_round3/r3_census_{pre,post}_{win,wsl}.json``):

        reading        PRE entry points   PRE apply_real_lens   tip
        name only            20            not in census        20 / 16
        + import alias       20            not in census        20 / 16
        + private hop        33            IN, zero keywords    33 / 17

    so the alias resolution alone is NOT enough -- it takes the private
    hop as well -- and with both, the seventeenth is named on the pre-fix
    tree carrying neither keyword.

    This arm reproduces that shape in process, on a SCRATCH tree, using a
    name that does not trace anywhere in the real package
    (``make_doublet``, a prescription builder) so that naming it can only
    have come from the scratch body.  Two-sided three ways: with the alias
    AND the hop it is named, with the hop switched off it is not, and a
    second scratch caller that names the tracer DIRECTLY is named either
    way (so the arm is not merely asserting that the census sees
    everything).
    """
    root = tmp_path / 'scratchpkg'
    root.mkdir()
    (root / '__init__.py').write_text('', encoding='utf-8')
    (root / 'aliased.py').write_text(
        "from lumenairy.raytrace import trace as _rt_alias\n"
        "\n"
        "\n"
        "def _scratch_impl(rays, surfaces, wavelength):\n"
        "    return _rt_alias(rays, surfaces, wavelength)\n"
        "\n"
        "\n"
        "def make_doublet(rays, surfaces, wavelength):\n"
        "    return _scratch_impl(rays, surfaces, wavelength)\n",
        encoding='utf-8')
    (root / 'plain.py').write_text(
        "from lumenairy.raytrace import trace\n"
        "\n"
        "\n"
        "def make_singlet(rays, surfaces, wavelength):\n"
        "    return trace(rays, surfaces, wavelength)\n",
        encoding='utf-8')

    # PREMISE: neither name traces in the REAL package, so any hit below
    # comes from the scratch bodies and not from the installed ones.
    real = _c2_entry_point_census()
    assert 'make_doublet' not in real and 'make_singlet' not in real, (
        'make_doublet / make_singlet trace in the real package now, so '
        'this scratch arm can no longer attribute a hit to its own '
        'source.  Pick two other exported non-tracing names.')

    both = _c2_entry_point_census(package_root=root)
    assert 'make_doublet' in both, (
        'the census did not name a route-less exported caller that reaches '
        'the tracer through an IMPORT ALIAS and a PRIVATE helper -- the '
        'exact shape that hid apply_real_lens for two rounds (VR2-D1).')
    assert 'make_singlet' in both, (
        'the census stopped naming a caller that names the tracer '
        'outright, so it is no longer a census of anything.')

    # and the hop is load-bearing: without it the aliased-through-a-private
    # -helper caller disappears while the direct one stays
    no_hop = _c2_entry_point_census(package_root=root, hop=False)
    assert 'make_doublet' not in no_hop, (
        'the private-helper hop is no longer what finds the caller, so '
        'this arm is not measuring the resolution it says it measures.')
    assert 'make_singlet' in no_hop, no_hop

    # the keywords the census reports are read from the LIVE signature, so
    # both scratch names come back route-less -- which is what the PRE-tree
    # probe reads for apply_real_lens
    assert both['make_doublet'] == [], both['make_doublet']


#: The nine entry points whose whole answer is cheap enough to compare BYTE
#: for byte across three calls, which is the strongest form of the claim:
#: omitted == None, and both != the forced old routes.
_C2_BYTE_COMPARED = (
    'trace_prescription', 'raytrace_system', 'ray_fan_data',
    'ray_fan_data_world', 'opd_fan_data', 'opd_fan_data_world',
    'through_focus_rms', 'paraxial_focus_world', 'ray_transfer_jacobian',
)

#: The other EIGHT, measured by SPYING on the tracer instead of comparing
#: answers: they build a whole field or a figure, so three full calls each
#: would cost minutes, while one spied call each costs seconds and pins the
#: same property one level closer to the wire (the keyword ARRIVES at every
#: internal trace call, and the default call arrives with neither).
#:
#: VERIFY-WP-C2 round 2, defect VR2-D4: before these ids existed, dropping
#: the forward in ``elements/_lens_traced.py``,
#: ``propagators/asymptotic_canonical_fit.py``,
#: ``analysis/image_plane_wfe.py`` or ``analysis/aberration.py`` left the
#: WHOLE C2 suite green -- 58 passed, 0 failed on each of the four.  The
#: keyword stayed in the signature, so the census stayed green; it simply
#: went nowhere.
_C2_SPIED = (
    'caustic_diagnostic', 'eval_image_plane_wfe', 'plot_lens_layout',
    'fit_canonical_polynomials', 'fit_hf_polynomials',
    'apply_real_lens_traced', 'apply_real_lens_maslov',
    # VR2-D1: the seventeenth.  It traces only under seidel_correction=True.
    'apply_real_lens',
)


def _c2_spy_prescription():
    """A spherical doublet -- three PURE SPHERES, so ``sphere_normal`` has
    something to select on at every surface."""
    return {
        'name': 'c2 round-3 spy doublet',
        'aperture_diameter': 0.0280,
        'surfaces': [
            {'radius': 0.0731, 'conic': 0.0,
             'glass_before': 'air', 'glass_after': 'N-BK7'},
            {'radius': -0.0437, 'conic': 0.0,
             'glass_before': 'N-BK7', 'glass_after': 'N-SF5'},
            {'radius': -0.1013, 'conic': 0.0,
             'glass_before': 'N-SF5', 'glass_after': 'air'},
        ],
        'thicknesses': [0.0082, 0.0031],
    }


def _c2_spied_call(name, P):
    """``fn(**way_back_kwargs) -> None`` for one of the eight spied entry
    points.  Sized small deliberately: the arm measures WHICH keywords
    reach the tracer, not the answer, so every grid here is the smallest
    one that still traces."""
    import numpy as _np

    def call(**kw):
        if name == 'caustic_diagnostic':
            from lumenairy.analysis.aberration import caustic_diagnostic
            return caustic_diagnostic(P, WL, fan_radius=2.5e-3,
                                      n_z_per_gap=6,
                                      z_after_last_surface=0.0800, **kw)
        if name == 'eval_image_plane_wfe':
            from lumenairy.analysis.image_plane_wfe import eval_image_plane_wfe
            return eval_image_plane_wfe(
                dict(P, object_distance=float('inf')), WL, field=(0.0, 0.4),
                n_pupil=9, field_max_rad=0.022, **kw)
        if name == 'plot_lens_layout':
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            from lumenairy.analysis.plotting import plot_lens_layout
            fig, _ax = plot_lens_layout(P, wavelength=WL, show_rays=True,
                                        n_field_angles=2, max_field_deg=1.5,
                                        rays_per_fan=5, **kw)
            plt.close(fig)
            return None
        if name == 'fit_canonical_polynomials':
            from lumenairy.propagators.asymptotic_canonical_fit import (
                fit_canonical_polynomials)
            return fit_canonical_polynomials(P, WL, n_field=3, n_pupil=5,
                                             poly_order=4, **kw)
        if name == 'fit_hf_polynomials':
            from lumenairy.propagators.asymptotic_canonical_fit import (
                fit_hf_polynomials)
            return fit_hf_polynomials(P, WL, n_field=3, n_pupil=5,
                                      poly_order=4, **kw)
        n, dx = 48, 25e-6
        a = (_np.arange(n) - n // 2) * dx
        X, Y = _np.meshgrid(a, a, indexing='xy')
        E = _np.exp(-(X ** 2 + Y ** 2) / (0.35e-3) ** 2).astype(
            _np.complex128)
        if name == 'apply_real_lens_traced':
            from lumenairy.elements import apply_real_lens_traced
            return apply_real_lens_traced(
                E, prescription=P, wavelength=WL, dx=dx, ray_subsample=8,
                bandlimit=False, on_undersample='silent',
                on_noncollimated='silent', on_aperture_beam='silent',
                on_fit_domain_basis='silent', on_pool_memory='silent',
                n_workers=1, **kw)
        if name == 'apply_real_lens':
            from lumenairy.elements import apply_real_lens
            # the Seidel-residual fan is the ONLY trace this entry point
            # makes, so the correction has to be ON or the arm is vacuous
            return apply_real_lens(
                E, prescription=P, wavelength=WL, dx=dx,
                seidel_correction=True, seidel_poly_order=6, **kw)
        from lumenairy.elements import apply_real_lens_maslov
        return apply_real_lens_maslov(
            E, prescription=P, wavelength=WL, dx=dx, ray_field_samples=5,
            ray_pupil_samples=5, poly_order=4, output_subsample=2, **kw)

    return call


def _c2_spy_on_the_tracers(call):
    """Run ``call`` twice -- once with both keywords forced to the old
    routes, once at the default -- with every module attribute bound to
    ``trace`` / ``trace_world`` replaced by a recording wrapper.

    Returns ``(forced, default)``, each a list of
    ``(sphere_normal, renormalize)`` as the tracer actually received them.

    Patching every BINDING rather than the defining module is deliberate:
    the entry points bind ``trace`` at import time, so patching
    ``raytrace.trace.trace`` alone would reach only the late-binding
    callers.
    """
    import sys as _sys
    import warnings

    seen = []
    tmod = _sys.modules['lumenairy.raytrace.trace']
    wmod = _sys.modules['lumenairy.raytrace.world_trace']
    reals = [tmod.trace, wmod.trace_world]

    def _mk(real):
        def spied(*a, **k):
            seen.append((k.get('sphere_normal', '<omitted>'),
                         k.get('renormalize', '<omitted>')))
            return real(*a, **k)
        return spied

    # warm the imports BEFORE the patch, so every module the entry point
    # binds from is already in sys.modules when the spies go in
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        call()

    spy_of = {id(r): _mk(r) for r in reals}
    patched = []
    for mn, m in list(_sys.modules.items()):
        if not mn.startswith('lumenairy') or m is None:
            continue
        if not hasattr(m, '__dict__'):
            continue
        for an in list(vars(m)):
            try:
                v = getattr(m, an)
            except Exception:                     # pragma: no cover
                continue
            if id(v) in spy_of:
                setattr(m, an, spy_of[id(v)])
                patched.append((m, an, v))
    assert patched, 'no module attribute was bound to the tracer to patch'
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            seen.clear()
            call(sphere_normal='generic', renormalize='surface')
            forced = list(seen)
            seen.clear()
            call()
            default = list(seen)
    finally:
        for m, an, v in patched:
            setattr(m, an, v)
    return forced, default


@pytest.mark.parametrize('name', _C2_BYTE_COMPARED + _C2_SPIED)
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
    ``post_default`` and ``post_none`` are identical on every array.

    ROUND 3 (VERIFY-WP-C2 round 2, defect VR2-D4).  The ``a != c`` half of
    this arm is the only IN-PROCESS pin that a forwarded keyword actually
    REACHES the internal trace, and it used to parametrize NINE of the
    sixteen.  The other seven were covered only by a committed probe JSON,
    which is a recording rather than a gate: dropping the forward
    (``**_way_back_kwargs()`` with no arguments) in
    ``elements/_lens_traced.py``, ``propagators/asymptotic_canonical_fit.py``,
    ``analysis/image_plane_wfe.py`` or ``analysis/aberration.py`` left the
    whole C2 suite GREEN -- 58 passed, 0 failed on every one of the four.

    All SEVENTEEN are parametrized now.  The nine in ``_C2_BYTE_COMPARED``
    keep the byte comparison; the eight in ``_C2_SPIED`` -- which build a
    whole field or a figure -- are measured by spying on the tracer
    instead, which pins the same property one level closer to the wire and
    costs one call each instead of three.
    """
    if name in _C2_SPIED:
        _c2_assert_the_forward_reaches_the_tracer(name)
        return

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


def _c2_assert_the_forward_reaches_the_tracer(name):
    """The ``_C2_SPIED`` half of the arm above: the forwarded keywords must
    arrive at EVERY internal trace call, and the default call must arrive
    with NEITHER.

    Two-sided by construction.  The first half is VR2-D4 -- a keyword that
    is accepted and then dropped before the trace leaves the signature
    census green and the way back broken.  The second is the sentinel: if
    ``None`` stamped ``'analytic'`` / ``'exit'``, this release's default
    would be frozen into the call site and the next flip would reach none
    of the seventeen.

    MEASURED 2026-09-20 on Windows py3.14 / numpy 2.4.4 and WSL py3.12 /
    numpy 2.4.6: every one of the eight makes between one and four
    internal trace calls and every call carries the forced pair; the
    default call carries neither.  Cost 4.4 s for the seven that existed
    in round 2 (VERIFY-WP-C2 round 2 section 1, VR2-D4).
    """
    call = _c2_spied_call(name, _c2_spy_prescription())
    forced, default = _c2_spy_on_the_tracers(call)
    assert forced, (
        f'{name}: the tracer was never called, so this arm measured '
        f'nothing.  Measured at least one call per entry point on '
        f'2026-09-20; if the entry point stopped tracing, move it out of '
        f'_C2_SPIED rather than leaving a vacuous id behind.')
    assert set(forced) == {('generic', 'surface')}, (
        f'{name}: the forwarded keywords did NOT reach every internal '
        f'trace call -- the tracer saw {sorted(set(forced))} over '
        f'{len(forced)} call(s).  A keyword that is accepted and then '
        f'dropped before the trace is the VR2-D4 shape: it leaves the '
        f'signature census green and the way back broken.')
    assert set(default) == {('<omitted>', '<omitted>')}, (
        f'{name}: the DEFAULT call stamped {sorted(set(default))} on the '
        f'tracer.  None must name nothing, or this release\'s default is '
        f'frozen into the call site and the next flip reaches none of the '
        f'seventeen.')


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

    ROUND 3 (VR2-D3): the helper is now pinned for EVERY keyword the
    library names, not just for ``sphere_normal``.

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
    from lumenairy.raytrace.trace import (
        _library_trace_default, _WAY_BACK_KEYWORDS)

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
    # VR2-D3 (VERIFY-WP-C2 round 2, 2026-09-20): the helper offers itself
    # for BOTH switches and was pinned for ``sphere_normal`` alone.  A
    # mutant in which it answers ``'surface'`` for ``renormalize`` while
    # ``trace`` defaults to ``'exit'`` passed the whole C2 suite -- 58
    # passed, 0 failed.  Nothing consumes that answer today, which is why
    # it was P3; the day a second direct caller of ``_refract`` /
    # ``_reflect`` takes the helper up on its offer, a wrong answer is a
    # silent divergence between that caller and ``trace``.  The loop is
    # over the pair the LIBRARY names, so a third switch is covered the
    # day it is added.
    _params = inspect.signature(trace).parameters
    assert set(_WAY_BACK_KEYWORDS) == set(_C2_WAY_BACK), _WAY_BACK_KEYWORDS
    for _key in _WAY_BACK_KEYWORDS:
        assert _library_trace_default(_key) == _params[_key].default, (
            f'_library_trace_default({_key!r}) answers '
            f'{_library_trace_default(_key)!r} while trace defaults it to '
            f'{_params[_key].default!r}.  The helper exists so a direct '
            f'caller of _refract / _reflect can ASK the library rather '
            f'than write a route down; an answer that disagrees with the '
            f'tracer is worse than the hard-coded route it replaced '
            f'(VR2-D3).')
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
# 6e -- the release text's byte-identity counts ARE the probe's counts (D12)
# ===========================================================================

def _c2_repo_root():
    return pathlib.Path(la.__file__).resolve().parents[1]


def _c2_byte_identity_json(build):
    import json
    path = (_c2_repo_root() / 'validation' / 'probe_c2_analytic_normal'
            / f'byte_identity_old_kw_{build}.json')
    with path.open(encoding='utf-8') as fh:
        return json.load(fh)


def test_c2_the_release_text_byte_identity_counts_match_the_probe_json():
    """The CHANGELOG and the Migration Guide both carried a byte-identity
    count the probe's OWN committed output contradicts: "938 of 1008 ... and
    the 70 that are not", against `n_moved = 74` (Windows) and 73 (WSL) in
    `byte_identity_old_kw_*.json`, i.e. 934 and 935 identical
    (VERIFY-WP-C2 defect D12).

    Two user-facing documents were wrong, and they are the ones a reader
    quotes.  The fix is not to retype the numbers but to make the JSON the
    source of truth for them: this arm derives every count from the
    committed JSONs and requires the release text to contain exactly those,
    so the next time the probe is re-run and the text is not, the gate
    fires instead of the reader.

    The PREMISES are asserted first -- the two JSONs exist, disagree with
    each other by one array, and carry the family breakdown -- so a pass
    cannot come from a missing file or an empty document.
    """
    win = _c2_byte_identity_json('win')
    wsl = _c2_byte_identity_json('wsl')

    # --- premises
    assert win['n_common'] == wsl['n_common'] == 1008, (
        win['n_common'], wsl['n_common'])
    assert win['n_values'] == wsl['n_values'] == 1630399, (
        win['n_values'], wsl['n_values'])
    assert win['n_moved'] != wsl['n_moved'], (
        'the two builds now report the same moved count; this arm exists '
        'because they differed by one array (74 vs 73), so re-derive the '
        'claim rather than dropping the per-build split.')
    assert win['moved_families'], 'no family breakdown in the Windows JSON'

    n_total = win['n_common']
    win_moved, wsl_moved = win['n_moved'], wsl['n_moved']
    win_same, wsl_same = n_total - win_moved, n_total - wsl_moved

    root = _c2_repo_root()
    changelog = (root / 'CHANGELOG.md').read_text(encoding='utf-8')
    guide = (root / 'Migration-Guide.md').read_text(encoding='utf-8')
    entry = changelog.split('## [Unreleased]')[1].split('## [5.48.1]')[0]
    assert 'WP-C2' in entry, (
        'the Unreleased block no longer contains the WP-C2 entry, so this '
        'arm has nothing to check.')
    guide_section = guide.split(
        "### the ray tracer's `sphere_normal` default")[1]

    import re as _re

    def _flat(t):
        return _re.sub(r'\s+', ' ', t)

    entry, guide_section = _flat(entry), _flat(guide_section)
    for name, text in (('CHANGELOG [Unreleased] C2 entry', entry),
                       ('Migration-Guide 5.49.0 C2 section', guide_section)):
        assert f'{win_same} of {n_total}' in text, (
            f'{name} does not state "{win_same} of {n_total}" -- the '
            f'Windows byte-identity count the committed probe JSON '
            f'reports ({n_total} arrays, {win_moved} moved).')
        assert str(wsl_same) in text, (
            f'{name} does not state the WSL count {wsl_same} '
            f'({wsl_moved} moved).')
        assert str(win_moved) in text and str(wsl_moved) in text, (
            f'{name} does not state both moved counts '
            f'({win_moved} Windows, {wsl_moved} WSL).')
        # the stale pair must be gone, not merely joined by the right one
        assert '938 of 1008' not in text, (
            f'{name} still carries the superseded "938 of 1008"; the '
            f'probe JSON says {win_same}.')
        assert 'the 70 that are not' not in text, (
            f'{name} still carries the superseded "the 70 that are not"; '
            f'the probe JSON says {win_moved}.')

    # the family breakdown the text quotes is the JSON's, item by item
    for family, count in sorted(win['moved_families'].items()):
        assert f'`{family}` {count}' in entry, (
            f'the CHANGELOG entry does not carry the measured breakdown '
            f'"{family} {count}" from byte_identity_old_kw_win.json '
            f'({win["moved_families"]}).')
    assert sum(win['moved_families'].values()) == win_moved, (
        win['moved_families'], win_moved)


# ===========================================================================
# 6f -- the EDITED_IN_PLACE override refuses a claim that became false (D7)
# ===========================================================================

def _c2_load_reanchor():
    import importlib.util
    path = _c2_repo_root() / 'scripts' / 'reanchor_citations.py'
    spec = importlib.util.spec_from_file_location('_c2_reanchor', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_C2_REANCHOR_TARGET = 'lumenairy/raytrace/trace.py'
_C2_REANCHOR_BASE = 'SYNTHETIC-BASE'


#: The synthetic file both EDITED_IN_PLACE arms run against: 60 lines of
#: padding with a module-level ``def trace(`` at line 55, so line 61 has the
#: same ENCLOSING DEFINITION as the real ``lumenairy/raytrace/trace.py:61``.
#: VERIFY-WP-C2 round 2 defect VR2-D7: the content digest alone accepted the
#: expected line under a different ``def``, so the owner is now part of the
#: fixture rather than absent from it.
def _c2_synthetic_lines(current, owner='def trace('):
    lines = ['# pad'] * 60
    lines[54] = owner
    lines += ["    sphere_normal: str = 'generic',", '    ) -> None:']
    base = list(lines)
    lines[60] = current
    return lines, base


def _c2_try_override(ra, current_lines, base_lines):
    """Run ``_edited_in_place`` against a synthetic base and current file.

    Both sides are supplied, so the arm is a pure unit test of the guard and
    needs neither a git checkout nor the base commit to be present.
    """
    real_lines = ra.lines

    def patched(path, rev=None):
        if path != _C2_REANCHOR_TARGET:
            return real_lines(path, rev)
        return base_lines if rev == _C2_REANCHOR_BASE else current_lines

    ra.lines = patched
    ra.EDITED_IN_PLACE_REFUSALS.clear()
    try:
        num, how = ra._edited_in_place(_C2_REANCHOR_TARGET, 61,
                                       _C2_REANCHOR_BASE)
        return num, how, list(ra.EDITED_IN_PLACE_REFUSALS)
    finally:
        ra.lines = real_lines
        ra.EDITED_IN_PLACE_REFUSALS.clear()


@pytest.mark.parametrize('abuse,current,fires,owner', [
    ('the shipped state', "    sphere_normal: str = 'analytic',", True,
     'def trace('),
    ('the default silently REVERTED', "    sphere_normal: str = 'generic',",
     False, 'def trace('),
    ('a nonsense value', "    sphere_normal: str = 'not-a-route',", False,
     'def trace('),
    ('a stale copy of the old declaration',
     "    sphere_normal: str = 'generic',  # moved to line 1300", False,
     'def trace('),
    ('an unrelated line', '    renormalize: str = "exit",', False,
     'def trace('),
    # VERIFY-WP-C2 round 2, defect VR2-D7: the EXACT expected content under
    # a DIFFERENT enclosing definition.  The content digest cannot see it --
    # the text is right -- and the citation re-anchors to the right line in
    # the wrong function.
    ('the expected content under a DIFFERENT def',
     "    sphere_normal: str = 'analytic',", False,
     'def trace_a_different_thing('),
])
def test_c2_the_edited_in_place_override_pins_the_content(abuse, current,
                                                          fires, owner):
    """``scripts/reanchor_citations.py``'s ``EDITED_IN_PLACE`` map answers a
    citation whose CONTENT changed -- a default flip is exactly that case --
    and its guard used to compare only the text BEFORE the first ``=``.

    VERIFY-WP-C2 defect D7 abused that on both builds: the override fired on
    the default silently REVERTED to the value the entry says the release
    moved AWAY from, on a nonsense value, and on a STALE COPY left at the
    mapped line while the declaration moved elsewhere.  All three
    re-anchored clean and reported a 5.49.0 reason for a claim that had
    become false.  It refused an unrelated line and an out-of-range
    coordinate, which is only half a guard.

    Each entry now pins a SHA-256 of the exact content the release was
    supposed to produce, so the value after the ``=`` is inside the check,
    and a refusal is recorded with BOTH lines rather than folded into a
    silent fall-through.

    This is parametrized over all five cases so the fix cannot be a guard
    that refuses everything: the shipped state must still FIRE.
    """
    ra = _c2_load_reanchor()
    assert (_C2_REANCHOR_TARGET, 61) in ra.EDITED_IN_PLACE, (
        'the EDITED_IN_PLACE map no longer carries trace.py:61; if it has '
        'been retired, delete this arm.')
    entry = ra.EDITED_IN_PLACE[(_C2_REANCHOR_TARGET, 61)]
    assert len(entry) == 5, (
        f'an EDITED_IN_PLACE entry must carry (new_num, reason, digest, '
        f'recorded_for, enclosing_def); this one carries {len(entry)} '
        f'fields: {entry}')
    assert entry[4] == 'def trace(', (
        f'the entry for trace.py:61 records {entry[4]!r} as its enclosing '
        f'definition; the line is a parameter of ``def trace(``.')

    current_lines, base_lines = _c2_synthetic_lines(current, owner=owner)

    num, how, refusals = _c2_try_override(ra, current_lines, base_lines)
    if fires:
        assert num == 61, (
            f'the override no longer fires on {abuse!r}, which is the '
            f'change it was written for; how={how!r} refusals={refusals}')
        assert not refusals, refusals
    else:
        assert num is None, (
            f'the override still fires on {abuse!r}.  D7: the map must not '
            f're-anchor a citation whose CLAIM has become false.')
        if abuse != 'an unrelated line':
            # the unrelated line is caught by the older leading-token guard,
            # which returns before the digest is consulted; the others must
            # be caught by the DIGEST or by the enclosing-definition guard,
            # and must say so with both lines
            assert len(refusals) == 1, (
                f'{abuse!r} was refused silently; D7 asks for the refusal '
                f'to name both lines.  refusals={refusals}')
            r = refusals[0]
            assert r['found_line'] == current.strip(), r
            assert r['base_line'] == "sphere_normal: str = 'generic',", r
            if owner == 'def trace(':
                assert r['expected_digest'] != r['found_digest'], r
            else:
                # VR2-D7: the content is RIGHT -- that is the whole point --
                # so the digest matches and the OWNER is what refused it
                assert r['expected_digest'] == r['found_digest'], r
                assert r['expected_owner'] == 'def trace(', r
                assert 'belongs to' in r['why'], r


def test_c2_the_edited_in_place_override_refuses_an_out_of_range_coordinate():
    """The second arm the guard already had, kept: a file too short to hold
    the mapped coordinate is refused rather than indexed."""
    ra = _c2_load_reanchor()
    _cur, base_lines = _c2_synthetic_lines(
        "    sphere_normal: str = 'analytic',")
    num, _how, refusals = _c2_try_override(ra, base_lines[:30], base_lines)
    assert num is None, (
        'the override stopped refusing an out-of-range new coordinate.')
    assert not refusals, (
        'an out-of-range coordinate should be refused BEFORE the content '
        'check, so it is not a content refusal: %r' % (refusals,))


def test_c2_the_edited_in_place_map_is_version_pinned():
    """The release number used to live only in the human-readable reason
    string, so nothing compared it to anything and the map kept firing for
    every release after the one it records (D7).

    Each entry now records the release it was made for, and the override
    refuses once the package has gone PAST it -- by then the base commit a
    re-anchor should use is already past that release and there is no
    in-place edit left to answer.

    Two-sided: the entry fires at the version it records (and at every
    version before it, which is where the tree sits while the release is
    unreleased), and refuses at the next one.
    """
    ra = _c2_load_reanchor()
    entry = ra.EDITED_IN_PLACE[(_C2_REANCHOR_TARGET, 61)]
    recorded_for = entry[3]
    assert ra._version_tuple(recorded_for) >= ra._version_tuple(
        ra._source_version()), (
        f'the map records {recorded_for} and the package is already at '
        f'{ra._source_version()}; the entries are stale and should be '
        f'retired rather than re-pointed.')

    shipped, base_lines = _c2_synthetic_lines(
        "    sphere_normal: str = 'analytic',")

    real_version = ra._source_version
    try:
        ra._source_version = lambda: recorded_for
        num, _how, refusals = _c2_try_override(ra, shipped, base_lines)
        assert num == 61 and not refusals, (num, refusals)

        bumped = list(ra._version_tuple(recorded_for))
        bumped[-1] += 1
        ra._source_version = lambda: '.'.join(str(v) for v in bumped)
        num, _how, refusals = _c2_try_override(ra, shipped, base_lines)
        assert num is None, (
            f'the override still fires with the package at '
            f'{ra._source_version()}, one patch past the {recorded_for} it '
            f'records.  D7 asks for the map to be version-pinned.')
        assert len(refusals) == 1 and 'recorded for' in refusals[0]['why'], (
            refusals)
    finally:
        ra._source_version = real_version


# ===========================================================================
# 6g -- the two backends gate the sphere's domain differently (D6)
# ===========================================================================

def test_c2_the_two_backends_gate_the_sphere_domain_differently():
    """A KNOWN, DECIDED cross-backend difference, pinned two-sidedly.

    `surface._sphere_normal` is NaN outside `h**2/R**2 < 0.9999` and
    `_surface_sag_derivative` applies the same gate, so the NumPy tracer
    kills every ray past `h = 0.99995 |R|` on a pure sphere with
    `RAY_NAN` -- on BOTH normal routes, before and after WP-C2.
    `jax_trace._refract_jax` builds the pure-spherical normal from the
    intersection's own `z` and applies NO gate, so the two backends
    disagree about a whole outer ANNULUS rather than about one ULP.

    That is PRE-EXISTING; WP-C2 neither caused it nor changed it, and it
    is identical under all four `(renormalize, sphere_normal)` CPU
    settings.  The maintainer's decision -- section 1.10 of
    ``MAINTAINER_DECISIONS_2026_09.md`` -- is to LEAVE it rather than
    clamp JAX, because clamping would move every JAX answer on a
    prescription whose aperture reaches `|R|` and would put a
    non-differentiable step inside a gradient path.

    So this arm exists to make a one-sided future change LOUD.  It
    asserts BOTH halves: the CPU kills EVERY ray past the clamp, and JAX
    keeps EVERY one.  Adding a clamp to `jax_trace`, or removing the
    NumPy one, turns it red and brings whoever does it to the ledger.

    MEASURED 2026-09-20
    (``validation/probe_c2_round2/r2_jax_clamp_{win,wsl}.json``): ball
    lens R = 12.5 mm with its clear semi-diameter AT 12.5 mm -- the
    catalogue part that makes the rim reachable at all -- 40 000 rays
    from `0.9990 |R|` to `0.999999 |R|`, of which 1962 are past the
    clamp.  CPU keeps 0 of them on all four settings; JAX keeps all
    1962.  Identical on Windows py3.14 / jax 0.11.0 and WSL py3.12 /
    jax 0.10.2.  The counts here are DERIVED from the ray set rather
    than recorded, so the arm moves only if a backend's gate moves.
    """
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)

    from lumenairy.raytrace import surfaces_from_prescription
    from lumenairy.raytrace.jax_trace import make_jax_ray_state, trace_jax
    from lumenairy.raytrace.surface import RAY_NAN

    R = 0.0125
    pres = dict(name='ball', aperture_diameter=2 * R,
                surfaces=[dict(radius=R, conic=0.0, aspheric_coeffs=None,
                               radius_y=None, conic_y=None,
                               aspheric_coeffs_y=None, glass_before='air',
                               glass_after='N-BK7'),
                          dict(radius=-R, conic=0.0, aspheric_coeffs=None,
                               radius_y=None, conic_y=None,
                               aspheric_coeffs_y=None,
                               glass_before='N-BK7', glass_after='air')],
                thicknesses=[2 * R])
    surfs = surfaces_from_prescription(pres)

    n = 4000
    r = R * np.linspace(0.9990, 0.999999, n)
    z = np.zeros(n)
    past = (r / R) ** 2 >= 0.9999
    n_past = int(past.sum())
    assert n_past > 100, (
        f'premise: the ray set must actually reach past the clamp for '
        f'this comparison to have content; only {n_past} of {n} do.')
    assert n_past < n, (
        'premise: some rays must be INSIDE the clamp too, or "past the '
        'clamp" is the whole bundle and the arm is not a comparison.')

    # --- the CPU half, on every combination of the two flipped defaults
    for rn in (PRE_C2_RENORMALIZE, 'exit'):
        for sn in ('generic', 'analytic'):
            b = _make_bundle(r.copy(), z.copy(), z.copy(), z.copy(), WL)
            out = trace(b, surfs, WL, output_filter='last',
                        renormalize=rn, sphere_normal=sn).image_rays
            alive = np.asarray(out.alive, dtype=bool)
            code = np.asarray(out.error_code)
            assert int((alive & past).sum()) == 0, (
                f'the NumPy tracer ({rn}/{sn}) keeps '
                f'{int((alive & past).sum())} of the {n_past} rays past '
                f'its own domain clamp; the clamp kills every one.')
            assert int(((code == RAY_NAN) & past).sum()) == n_past, (
                f'the NumPy tracer ({rn}/{sn}) no longer reports every '
                f'ray past the clamp as RAY_NAN.')
            # and the other side: rays INSIDE the clamp are not all dead
            assert int((alive & ~past).sum()) > 0, (
                f'the NumPy tracer ({rn}/{sn}) killed every ray inside '
                f'the clamp too, so "past the clamp" is not what the '
                f'kill is attributable to.')

    # --- the JAX half
    st = make_jax_ray_state(x=r, y=z.copy(), z=z.copy(), L=z.copy(),
                            M=z.copy(), N=np.ones(n))
    ja = np.asarray(trace_jax(st, pres, WL).alive, dtype=bool)
    assert int((ja & past).sum()) == n_past, (
        f'the JAX tracer now kills {n_past - int((ja & past).sum())} of '
        f'the {n_past} rays past the NumPy domain clamp.  That is a '
        f'cross-backend BEHAVIOUR CHANGE: section 1.10 of '
        f'MAINTAINER_DECISIONS_2026_09.md decided to leave the two '
        f'backends different here and to document it.  If the decision '
        f'has been revisited, update the ledger, the CHANGELOG entry and '
        f'this arm together.')


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
    | the release text's counts drift off the JSON | ``test_c2_the_release_text_byte_identity_counts_match_the_probe_json`` |
    | a re-anchored citation's claim becomes false | ``test_c2_the_edited_in_place_override_pins_the_content`` |
    | the EDITED_IN_PLACE map outlives its release | ``test_c2_the_edited_in_place_map_is_version_pinned`` |
    | a backend's sphere-domain gate moves alone  | ``test_c2_the_two_backends_gate_the_sphere_domain_differently`` |
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
            'test_c2_no_private_docstring_claims_the_generic_route_is_shipped',
            'test_c2_the_release_text_byte_identity_counts_match_the_probe_json',
            'test_c2_the_edited_in_place_override_pins_the_content',
            'test_c2_the_edited_in_place_map_is_version_pinned',
            'test_c2_the_two_backends_gate_the_sphere_domain_differently'):
        assert callable(getattr(mod, name, None)), (
            f'{name} named in the mutation matrix no longer exists; '
            f'either restore it or update the table above.')
    assert la is not None
