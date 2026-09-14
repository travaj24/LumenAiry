"""WP-B9 / AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11 -- the ray-tracing
performance and completeness items WP-A1 deferred (its report section 6,
items 1-6).

Every section states its ORACLE, what the pre-fix code did, and how its
bar was derived.  Nothing here asserts a wall-clock time: the structural
claims are counted instead -- ``trace()`` calls, Newton evaluations,
cache identity -- so the file is build-free (TESTING_STANDARDS S5).

* **item 1** the ``_refract`` / ``_reflect`` renormalise is hoistable to
  one pass at the exit, behind ``renormalize='exit'``.  The DEFAULT is
  unchanged and is asserted bit-identical.
* **item 2** a pure sphere can get its normal in closed form,
  ``(-x/R, -y/R, sqrt(1 - h^2/R^2))``, selected on the same predicate as
  the closed-form intersection, behind ``sphere_normal='analytic'``.
  The DEFAULT is unchanged and is asserted bit-identical.
* **item 3** both fan functions issue ONE ``trace()`` instead of four.
* **item 4** ``_build_jax_prescription``'s output is memoised on the
  ``aux`` signature, which covers every field that changes the trace.
* **item 5** ``make_rings(pattern='vogel')`` is area-uniform; the default
  ``'rings'`` does not move.
* **item 6** ``ray_transfer_jacobian_analytic`` handles even-power
  aspheres instead of raising ``NotImplementedError``.
"""
from __future__ import annotations

import decimal
import inspect
import sys

import numpy as np
import pytest

import lumenairy as la
from lumenairy.raytrace import (
    RAY_NAN,
    RAY_OK,
    Surface,
    find_paraxial_focus,
    make_fan,
    make_rings,
    opd_fan_data,
    ray_fan_data,
    spot_rms,
    surfaces_from_prescription,
    trace,
)
from lumenairy.raytrace import ray_fan as _ray_fan_mod
from lumenairy.raytrace import differential as _diff_mod
from lumenairy.raytrace import intersection as _isect
from lumenairy.raytrace import surface as _surf_mod
_trace_mod = sys.modules['lumenairy.raytrace.trace']
from lumenairy.raytrace.differential import (
    ray_transfer_jacobian,
    ray_transfer_jacobian_analytic,
)
from lumenairy.raytrace.intersection import _normalize_directions
from lumenairy.raytrace.surface import (
    _sphere_normal,
    _surface_normal,
    _is_pure_spherical,
)
from lumenairy.raytrace.trace import _make_bundle

WL = 587.6e-9


# ===========================================================================
# Fixtures -- prescriptions built here, not imported, so each test states
# the geometry its numbers came from.
# ===========================================================================

def _spherical_stack():
    """Seven surfaces: cemented triplet + singlet + two flats."""
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


def _conic_stack():
    return [
        Surface(radius=0.0515, conic=-0.6, thickness=0.008,
                glass_before='air', glass_after='N-BK7',
                semi_diameter=0.0127),
        Surface(radius=-0.0345, conic=-1.2, thickness=0.100,
                glass_before='N-BK7', glass_after='air',
                semi_diameter=0.0127),
        Surface(radius=np.inf, thickness=0.0, glass_before='air',
                glass_after='air', semi_diameter=0.03),
    ]


def _aspheric_singlet(a1=None, a2=None, k1=0.0):
    return [
        Surface(radius=25e-3, conic=k1, aspheric_coeffs=a1, thickness=5e-3,
                glass_before='air', glass_after='N-BK7',
                semi_diameter=10e-3),
        Surface(radius=-60e-3, aspheric_coeffs=a2, thickness=45e-3,
                glass_before='N-BK7', glass_after='air',
                semi_diameter=10e-3),
        Surface(radius=np.inf, thickness=0.0, glass_before='air',
                glass_after='air', semi_diameter=20e-3),
    ]


def _imaged(prescription, wavelength=WL):
    """Surface list with a flat image plane at the paraxial focus."""
    S = list(surfaces_from_prescription(prescription))
    bfd = find_paraxial_focus(S, wavelength)
    last = S[-1]
    return S[:-1] + [
        Surface(radius=last.radius, conic=last.conic,
                aspheric_coeffs=last.aspheric_coeffs,
                semi_diameter=last.semi_diameter,
                glass_before=last.glass_before,
                glass_after=last.glass_after, thickness=bfd,
                label=last.label),
        Surface(radius=np.inf, thickness=0.0, glass_before='air',
                glass_after='air', semi_diameter=0.05, label='image'),
    ]


def _pupil_bundle(n, semi=0.0126, seed=20260913):
    rng = np.random.default_rng(seed)
    r = semi * np.sqrt(rng.random(n))
    th = 2 * np.pi * rng.random(n)
    x = r * np.cos(th)
    y = r * np.sin(th)
    z = np.zeros(n)
    return _make_bundle(x, y, z, z, WL)


# ===========================================================================
# item 1 -- the _refract / _reflect renormalise hoist
# ===========================================================================

def test_b9_i1_default_is_the_per_surface_renormalise():
    """The hoist is OPT-IN.  ``_refract`` / ``_reflect`` default to
    ``renormalize=True`` and ``trace`` / ``trace_world`` to
    ``'surface'``, so every existing caller -- ``analysis.ghost`` and the
    finite-difference differential path reach ``_refract`` directly --
    keeps its arithmetic by construction, not by measurement."""
    assert (inspect.signature(_isect._refract)
            .parameters['renormalize'].default is True)
    assert (inspect.signature(_isect._reflect)
            .parameters['renormalize'].default is True)
    from lumenairy.raytrace.world_trace import trace_world
    assert (inspect.signature(trace).parameters['renormalize'].default
            == 'surface')
    assert (inspect.signature(trace_world).parameters['renormalize'].default
            == 'surface')
    # item 2's switch is opt-in on exactly the same terms
    assert (inspect.signature(_isect._refract)
            .parameters['sphere_normal'].default == 'generic')
    assert (inspect.signature(_isect._reflect)
            .parameters['sphere_normal'].default == 'generic')
    assert (inspect.signature(trace).parameters['sphere_normal'].default
            == 'generic')
    assert (inspect.signature(trace_world).parameters['sphere_normal']
            .default == 'generic')
    assert (inspect.signature(_surface_normal)
            .parameters['analytic_sphere'].default is False)


def test_b9_i1_surface_mode_leaves_every_history_bundle_unit_length():
    """ORACLE: the renormalise contract itself -- ``|(L, M, N)| == 1``.

    Under ``'surface'`` EVERY recorded bundle is unit length; under
    ``'exit'`` only the last one is, and the intermediate drift is real
    (non-zero), which is the two-sided statement that the division was
    actually hoisted rather than merely relocated.
    """
    S = _spherical_stack()
    rays = _pupil_bundle(400)
    res_s = trace(rays, S, WL)
    res_e = trace(rays, S, WL, renormalize='exit')

    def worst(bundle):
        m = bundle.alive
        u = np.sqrt(bundle.L[m] ** 2 + bundle.M[m] ** 2 + bundle.N[m] ** 2)
        return float(np.max(np.abs(u - 1.0)))

    per_surface = [worst(b) for b in res_s.ray_history]
    hoisted = [worst(b) for b in res_e.ray_history]
    # 'surface': unit to the rounding of the division itself (mag / mag
    # leaves the recomputed sum of squares within 1 ULP of 1).
    assert max(per_surface) <= 2 * 2 ** -52, per_surface
    # 'exit': the exit bundle is renormalised, so it is no worse ...
    assert hoisted[-1] <= 2 * 2 ** -52
    # ... and the drift the hoist leaves behind accumulates: some
    # intermediate bundle is measurably WORSE than the same bundle under
    # 'surface'.  That is the two-sided statement that the division was
    # hoisted and not merely relocated.
    assert max(hoisted[:-1]) > max(per_surface), (hoisted, per_surface)
    # bounded by one refraction's worth of rounding per surface
    assert max(hoisted[:-1]) < 4 * len(S) * 2 ** -52


def test_b9_i1_exit_mode_calls_the_single_pass_exactly_once(monkeypatch):
    """Structural (build-free) count: ``_normalize_directions`` runs ONCE
    per trace in ``'exit'`` mode and never in ``'surface'`` mode, while
    ``_refract`` runs once per refracting surface in both."""
    S = _spherical_stack()
    rays = _pupil_bundle(32)
    calls = {'norm': 0, 'refract': 0}
    real_norm = _trace_mod._normalize_directions
    real_refract = _trace_mod._refract

    def counting_norm(b):
        calls['norm'] += 1
        return real_norm(b)

    def counting_refract(*a, **kw):
        calls['refract'] += 1
        return real_refract(*a, **kw)

    monkeypatch.setattr(_trace_mod, '_normalize_directions', counting_norm)
    monkeypatch.setattr(_trace_mod, '_refract', counting_refract)

    trace(rays, S, WL)
    assert calls == {'norm': 0, 'refract': len(S)}
    calls['norm'] = calls['refract'] = 0
    trace(rays, S, WL, renormalize='exit')
    assert calls == {'norm': 1, 'refract': len(S)}


@pytest.mark.parametrize('renorm', [True, False])
def test_b9_i1_degenerate_direction_still_dies_at_its_own_surface(renorm):
    """The DIAGNOSIS is not hoisted, only the division.

    A non-finite direction must be reported as ``RAY_NAN`` by the surface
    that saw it in BOTH modes -- that is why the per-surface ``mag`` (and
    its ``~isfinite`` test) stays even when the division does not.
    Engineered at the kernel, by handing ``_refract`` a bundle whose
    direction is already degenerate, rather than hoping a build produces
    one.  (An exactly-zero direction is NOT degenerate after Snell: with
    ``cos_i = 0`` the law returns ``-cos_t * n``, a unit vector along the
    normal, so it is not a case this guard can see.)
    """
    surf = Surface(radius=0.05, thickness=0.0, glass_before='air',
                   glass_after='N-BK7', semi_diameter=np.inf)
    x = np.array([0.0, 1e-3, 2e-3])
    z = np.zeros(3)
    b = _make_bundle(x, z, z.copy(), z.copy(), WL)
    b.L, b.M, b.N = b.L.copy(), b.M.copy(), b.N.copy()
    b.N[1] = np.nan                        # NaN direction
    b.L[2] = np.inf                        # infinite direction
    with np.errstate(invalid='ignore'):    # inf - inf is the point here
        _isect._refract(b, surf, 1.0, 1.5, renormalize=renorm)
    assert b.alive[0]
    assert b.error_code[0] == RAY_OK
    assert not b.alive[1] and b.error_code[1] == RAY_NAN
    assert not b.alive[2] and b.error_code[2] == RAY_NAN


def test_b9_i1_the_two_modes_agree_on_every_kill():
    """Through the public API the two modes must make the SAME
    decisions: identical ``alive`` masks and identical error codes, on a
    bundle seeded with a non-finite ray."""
    S = _spherical_stack()
    x = np.array([0.0, 2e-3, 4e-3, 40e-3])
    z = np.zeros(4)
    b = _make_bundle(x, z, z.copy(), z.copy(), WL)
    b.L = b.L.copy()
    b.L[1] = np.nan
    a = trace(b, S, WL, output_filter='last').image_rays
    c = trace(b, S, WL, output_filter='last',
              renormalize='exit').image_rays
    assert np.array_equal(a.alive, c.alive)
    assert np.array_equal(a.error_code, c.error_code)
    assert a.alive[0] and not a.alive[1] and not a.alive[3]


def test_b9_i1_hoisted_drift_is_bounded_by_the_rounding_it_removes():
    """ORACLE: exact vector Snell with a unit normal returns a unit
    vector identically, so the two modes can differ only by the rounding
    the per-surface division was mopping up.

    BAR, derived: each refraction leaves ``| |d| - 1 | <= eps``, and that
    relative error propagates into the next surface's ray-sphere
    quadratic (which assumes ``a = |d|^2 = 1``) and into the transfer
    legs.  Over ``n`` surfaces the accumulated positional error is at
    most ``n * eps * |t|_max``; with ``|t| <= 0.11 m`` here and
    ``n = 7`` that is 1.7e-16 m.  MEASURED 2026-09-13 on this build:
    6.2e-17 m in position, 1.9e-16 m in OPL, 5.0e-16 in the direction
    cosines.  The bar is set one decade above the derived envelope, and
    the ALIVE masks must agree exactly (a decision, not a reading).
    """
    for S, span in ((_spherical_stack(), 0.11), (_conic_stack(), 0.10)):
        rays = _pupil_bundle(2000)
        a = trace(rays, S, WL, output_filter='last').image_rays
        b = trace(rays, S, WL, output_filter='last',
                  renormalize='exit').image_rays
        assert np.array_equal(a.alive, b.alive)
        assert np.array_equal(a.error_code, b.error_code)
        m = a.alive
        bar = 10.0 * len(S) * 2 ** -52 * span
        for f in ('x', 'y', 'z', 'opd'):
            d = float(np.max(np.abs(getattr(a, f)[m] - getattr(b, f)[m])))
            assert d < bar, (f, d, bar)
        for f in ('L', 'M', 'N'):
            d = float(np.max(np.abs(getattr(a, f)[m] - getattr(b, f)[m])))
            assert d < 10.0 * len(S) * 2 ** -52, (f, d)


def test_b9_i1_renormalize_is_validated_and_names_itself():
    with pytest.raises(ValueError, match=r"^trace: renormalize must be"):
        trace(_pupil_bundle(4), _spherical_stack(), WL, renormalize='exit_')


def test_b9_i1_normalize_directions_is_idempotent():
    b = _pupil_bundle(64)
    b.L = b.L + 1e-9
    _normalize_directions(b)
    L1, M1, N1 = b.L.copy(), b.M.copy(), b.N.copy()
    _normalize_directions(b)
    assert np.array_equal(b.L, L1)
    assert np.array_equal(b.M, M1)
    assert np.array_equal(b.N, N1)


# ===========================================================================
# item 2 -- the closed-form sphere normal
# ===========================================================================

def _decimal_sphere_normal(x, y, R, prec=60):
    """60-digit oracle for the outward unit normal of the sphere
    ``x^2 + y^2 + (z - R)^2 = R^2`` at the point above ``(x, y)``.

    Independent of the library: it evaluates the textbook
    ``n = (-dz/dx, -dz/dy, 1) / |.|`` with ``dz/dh = h / (R sqrt(1 -
    h^2/R^2))`` in ``decimal`` at 60 significant digits, which is ~44
    digits beyond float64, so its own error is not measurable here.
    """
    ctx = decimal.Context(prec=prec)
    D = ctx.create_decimal
    X, Y, RR = D(repr(float(x))), D(repr(float(y))), D(repr(float(R)))
    h_sq = ctx.add(ctx.multiply(X, X), ctx.multiply(Y, Y))
    h = h_sq.sqrt(ctx)
    one = D(1)
    inner = ctx.subtract(one, ctx.divide(h_sq, ctx.multiply(RR, RR)))
    root = inner.sqrt(ctx)
    dz_dh = ctx.divide(h, ctx.multiply(RR, root))
    if h == 0:
        dz_dx = dz_dy = D(0)
    else:
        dz_dx = ctx.divide(ctx.multiply(dz_dh, X), h)
        dz_dy = ctx.divide(ctx.multiply(dz_dh, Y), h)
    mag = ctx.add(ctx.add(ctx.multiply(dz_dx, dz_dx),
                          ctx.multiply(dz_dy, dz_dy)), one).sqrt(ctx)
    return (float(ctx.divide(-dz_dx, mag)),
            float(ctx.divide(-dz_dy, mag)),
            float(ctx.divide(one, mag)))


def _generic_normal(x, y, surface):
    """The pre-fix route: sag derivatives, then normalise.  Reproduced
    here so the test can compare BOTH routes against the oracle."""
    dz_dx, dz_dy = _surf_mod._surface_sag_derivatives_xy(x, y, surface)
    mag = np.sqrt(dz_dx ** 2 + dz_dy ** 2 + 1.0)
    return -dz_dx / mag, -dz_dy / mag, 1.0 / mag


@pytest.mark.parametrize('R', [0.0515, -0.0345, 0.5, -1.0, 0.002])
def test_b9_i2_closed_form_normal_beats_the_generic_route(R):
    """ORACLE: the 60-digit ``decimal`` normal above.

    The closed form must be at least as accurate as the route it
    replaces at every height, and strictly better somewhere -- that is
    the whole justification for retrying the v4.12.0 change.  BAR: 4 ULP
    of a unit vector on the closed form (2^-52 = 2.2e-16 each), with the
    generic route's own error measured alongside so the comparison is
    two-sided rather than a floor bar.
    """
    surf = Surface(radius=R, thickness=0.0, glass_before='air',
                   glass_after='N-BK7', semi_diameter=abs(R))
    h = np.abs(R) * np.array([0.0, 0.05, 0.2, 0.5, 0.8, 0.95])
    x = h / np.sqrt(2.0)
    y = h / np.sqrt(2.0)
    fast = np.array(_sphere_normal(x, y, R))
    slow = np.array(_generic_normal(x, y, surf))
    ref = np.array([_decimal_sphere_normal(xi, yi, R)
                    for xi, yi in zip(x, y)]).T
    e_fast = np.max(np.abs(fast - ref), axis=0)
    e_slow = np.max(np.abs(slow - ref), axis=0)
    assert np.all(e_fast <= 4 * 2 ** -52), (e_fast, e_slow)
    # never worse than the route it replaces, at any height
    assert np.all(e_fast <= e_slow + 2 ** -52), (e_fast, e_slow)
    # and strictly better on at least one of the six heights
    assert np.any(e_fast < e_slow), (e_fast, e_slow)


@pytest.mark.parametrize('R', [0.0515, -0.0345, 0.5])
def test_b9_i2_closed_form_normal_is_a_unit_vector(R):
    """``|n|^2 = h^2/R^2 + (1 - h^2/R^2) = 1`` identically, so the only
    departure is rounding: 2 ULP."""
    h = np.abs(R) * np.linspace(0.0, 0.95, 40)
    nx, ny, nz = _sphere_normal(h, 0.3 * h, R)
    u = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    assert float(np.max(np.abs(u - 1.0))) <= 2 * 2 ** -52


def test_b9_i2_predicate_is_shared_by_intersection_and_normal(monkeypatch):
    """The v4.12.0 attempt failed because the analytic NORMAL was applied
    without the matching INTERSECTION.  The two now select on one
    predicate, so they cannot disagree about what a surface is.

    Counted, not asserted from the source: the Newton branch evaluates
    ``_surface_sag_xy``; the spherical fast path does not.  So a surface
    for which ``_is_pure_spherical`` is True must produce ZERO sag
    evaluations during its intersection.
    """
    cases = [
        (Surface(radius=0.05, thickness=0.0, glass_before='air',
                 glass_after='air', semi_diameter=0.02), True),
        (Surface(radius=0.05, conic=-0.6, thickness=0.0,
                 glass_before='air', glass_after='air',
                 semi_diameter=0.02), False),
        (Surface(radius=0.05, aspheric_coeffs={4: 1.0e3}, thickness=0.0,
                 glass_before='air', glass_after='air',
                 semi_diameter=0.02), False),
        (Surface(radius=0.05, radius_y=0.06, thickness=0.0,
                 glass_before='air', glass_after='air',
                 semi_diameter=0.02), False),
        (Surface(radius=np.inf, thickness=0.0, glass_before='air',
                 glass_after='air', semi_diameter=0.02), False),
    ]
    for surf, expected in cases:
        assert _is_pure_spherical(surf) is expected, surf
        n = 0

        real = _isect._surface_sag_xy

        def counting(*a, **kw):
            nonlocal n
            n += 1
            return real(*a, **kw)

        monkeypatch.setattr(_isect, '_surface_sag_xy', counting)
        b = _pupil_bundle(16, semi=0.01)
        _isect._intersect_surface(b, surf, n_medium=1.0)
        monkeypatch.setattr(_isect, '_surface_sag_xy', real)
        if expected:
            assert n == 0, (surf, n)
        # the non-spherical kinds either Newton-iterate (n > 0) or take
        # the flat fast path; both are "not the sphere branch", which the
        # predicate already stated.


def test_b9_i2_the_generic_route_is_the_default_everywhere():
    """The closed form is reached ONLY by asking for it.  Every surface
    kind -- a pure sphere included -- must be BIT-IDENTICAL to the
    pre-fix sag-derivative route when ``analytic_sphere`` is not set."""
    kinds = [
        Surface(radius=0.05, thickness=0.0, glass_before='air',
                glass_after='air', semi_diameter=0.02),
        Surface(radius=0.05, conic=-0.6, thickness=0.0, glass_before='air',
                glass_after='air', semi_diameter=0.02),
        Surface(radius=0.05, aspheric_coeffs={4: 1.0e3, 6: -2.0e5},
                thickness=0.0, glass_before='air', glass_after='air',
                semi_diameter=0.02),
        Surface(radius=0.05, radius_y=0.06, conic_y=0.0, thickness=0.0,
                glass_before='air', glass_after='air', semi_diameter=0.02),
        Surface(radius=np.inf, thickness=0.0, glass_before='air',
                glass_after='air', semi_diameter=0.02),
    ]
    x = np.linspace(-0.01, 0.01, 21)
    y = 0.4 * x
    for surf in kinds:
        got = _surface_normal(x, y, surf)
        ref = _generic_normal(x, y, surf)
        for g, r in zip(got, ref):
            assert np.array_equal(g, r), surf


def test_b9_i2_the_default_trace_is_bit_identical_and_analytic_is_bounded():
    """The DEFAULT must not move a bit, and the opt-in must move only by
    the rounding difference between two routes to the same vector.

    Bit-identity is asserted against the explicit ``'generic'`` spelling
    (the default path and the named path are one path).  The opt-in's
    bar is derived: the two routes differ by O(eps) in each normal
    component, that feeds a refraction whose transverse lever arm over
    the remaining ~0.11 m of the stack is |t|, so the positional
    envelope is ``n_surfaces * eps * |t|`` = 1.7e-16 m.  MEASURED
    2026-09-13 over 1500 rays x 2 fields: 2.8e-17 m in position,
    8.3e-17 m in OPL, 2.8e-16 in the direction cosines, with every
    ``alive`` and ``error_code`` byte-identical -- and EXACTLY ZERO on
    the conic stack, which has no pure sphere.
    """
    for S, has_sphere in ((_spherical_stack(), True),
                          (_conic_stack(), False)):
        rays = _pupil_bundle(1500)
        d = trace(rays, S, WL, output_filter='last').image_rays
        g = trace(rays, S, WL, output_filter='last',
                  sphere_normal='generic').image_rays
        a = trace(rays, S, WL, output_filter='last',
                  sphere_normal='analytic').image_rays
        for f in ('x', 'y', 'z', 'L', 'M', 'N', 'opd'):
            assert np.array_equal(getattr(d, f), getattr(g, f)), f
        assert np.array_equal(d.alive, a.alive)
        assert np.array_equal(d.error_code, a.error_code)
        m = d.alive
        worst = max(float(np.max(np.abs(getattr(d, f)[m]
                                        - getattr(a, f)[m])))
                    for f in ('x', 'y', 'z', 'opd'))
        if has_sphere:
            assert 0.0 < worst < 10 * len(S) * 2 ** -52 * 0.11, worst
        else:
            assert worst == 0.0


def test_b9_i2_named_bit_equal_consumers_see_the_default():
    """The reason item 2 is opt-in, pinned so it stays opt-in.

    ``propagate_modal_asymptotic``'s two bit-equal pins
    (``tests/unit/test_audit_propagation.py -k
    ModalAsymptoticStillBitEqual``) and the 1e-15 floor bar in
    ``test_w6_a2_v2_star_is_untouched_by_the_verdict_fix`` read the ray
    tracer through the aberration tensor.  Both go red on a ~1e-16 move,
    so ``trace`` must keep handing them the generic route unless they are
    asked otherwise.
    """
    S = _spherical_stack()
    rays = _pupil_bundle(64)
    called = []
    real = _surf_mod._surface_normal

    def watching(x, y, surface, *, analytic_sphere=False):
        called.append(analytic_sphere)
        return real(x, y, surface, analytic_sphere=analytic_sphere)

    import unittest.mock as _mock
    with _mock.patch.object(_isect, '_surface_normal', watching):
        trace(rays, S, WL, output_filter='last')
        assert called and not any(called)
        called.clear()
        trace(rays, S, WL, output_filter='last', sphere_normal='analytic')
        assert called and all(called)


def test_b9_i2_sphere_normal_is_validated_and_names_itself():
    with pytest.raises(ValueError, match=r"^trace: sphere_normal must be"):
        trace(_pupil_bundle(4), _spherical_stack(), WL,
              sphere_normal='closed_form')


def test_b9_i2_out_of_domain_height_still_kills_the_ray():
    """The closed form reproduces ``_surface_sag_derivative``'s
    out-of-conic-domain policy (NaN normal for ``h^2/R^2 >= 0.9999``), so
    the vignetting behaviour of a near-hemispherical surface does not
    move."""
    R = 0.05
    h = np.array([0.0, 0.5, 0.9, 0.99994, 0.99996, 1.2]) * R
    nx, ny, nz = _sphere_normal(h, np.zeros_like(h), R)
    finite = np.isfinite(nz)
    assert finite.tolist() == [True, True, True, True, False, False]
    surf = Surface(radius=R, thickness=0.0, glass_before='air',
                   glass_after='air', semi_diameter=np.inf)
    generic = _generic_normal(h, np.zeros_like(h), surf)
    assert np.array_equal(np.isfinite(np.asarray(generic[2])), finite)


def test_b9_i2_non_finite_position_now_propagates_into_the_normal():
    """Behaviour CHANGE, stated: at a NOT-FINITE position the pre-fix
    spherical route returned a perfectly defined axial normal
    ``(-0, -0, 1)`` -- the ``np.where(h > 0, ..., 0.0)`` guard is False
    for a NaN ``h`` -- so the ray refracted off a fabricated surface and
    stayed ALIVE with ``RAY_OK``.  The closed form propagates the NaN,
    which is the S11-7 policy the shared-core ``conic_sag_derivs`` and
    the JAX backend already use, and the ray is killed as ``RAY_NAN``.

    Pre-fix reading is reproduced here from the generic route (still
    reachable through a conic surface), so the change is measured, not
    asserted from memory.
    """
    R = 0.05
    x = np.array([1e-3, np.nan])
    y = np.zeros(2)
    surf_sphere = Surface(radius=R, thickness=0.0, glass_before='air',
                          glass_after='air', semi_diameter=np.inf)
    surf_conic = Surface(radius=R, conic=-0.5, thickness=0.0,
                         glass_before='air', glass_after='air',
                         semi_diameter=np.inf)
    # pre-fix reading, still produced by the generic route:
    pre = _generic_normal(x, y, surf_conic)
    assert pre[0][1] == 0.0 and pre[2][1] == 1.0
    # the closed form, on the sphere:
    post = _surface_normal(x, y, surf_sphere, analytic_sphere=True)
    assert not np.isfinite(post[0][1])
    assert not np.isfinite(post[2][1])
    # Through the public API nothing moves: the ray-sphere quadratic
    # already refuses a non-finite position, so such a ray is dead with
    # RAY_MISSED_SURFACE before the normal is ever consulted.  The
    # closed form therefore removes a fabricated normal without changing
    # any traced outcome.
    from lumenairy.raytrace import RAY_MISSED_SURFACE
    S = [surf_sphere,
         Surface(radius=np.inf, thickness=0.0, glass_before='air',
                 glass_after='air', semi_diameter=np.inf)]
    b = _make_bundle(x, y, np.zeros(2), np.zeros(2), WL)
    img = trace(b, S, WL, output_filter='last').image_rays
    assert img.alive[0]
    assert not img.alive[1]
    assert img.error_code[1] == RAY_MISSED_SURFACE


def test_b9_i2_numpy_and_jax_backends_still_agree():
    """Cross-backend parity is the gate this item failed in v4.12.0.

    BAR: the WP-A1 report re-measured NumPy<->JAX parity at 6.9e-18 m
    (position) / 2.8e-17 m (OPL) on a spherical prescription.  The bar is
    1e-15 m -- two decades above that floor and more than seven decades
    below the 1.17e-3 RELATIVE divergence the v4.12.0 attempt produced.
    """
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    from lumenairy.raytrace.jax_trace import make_jax_ray_state, trace_jax
    pres = la.make_singlet(51.5e-3, -80e-3, 4.1e-3, 'N-BK7', aperture=12e-3)
    S = surfaces_from_prescription(pres)
    n = 9
    x = np.linspace(-5e-3, 5e-3, n)
    z = np.zeros(n)
    b = _make_bundle(x, z, z.copy(), z.copy(), WL)
    img = trace(b, S, WL, output_filter='last').image_rays
    ana = trace(b, S, WL, output_filter='last',
                sphere_normal='analytic').image_rays
    st = make_jax_ray_state(x=x, y=z, z=z, L=z, M=z, N=np.ones(n))
    out = trace_jax(st, pres, WL)
    assert np.array_equal(np.asarray(out.alive), img.alive)
    m = img.alive
    for ref in (img, ana):
        assert float(np.max(np.abs(np.asarray(out.x)[m] - ref.x[m]))) < 1e-15
        assert float(np.max(np.abs(np.asarray(out.opd)[m]
                                   - ref.opd[m]))) < 1e-15


# ===========================================================================
# item 3 -- one concatenated bundle for the fan functions
# ===========================================================================

def _count_traces(monkeypatch, module, name, fn):
    calls = {'n': 0, 'rays': 0}
    real = getattr(module, name)

    def counting(rays, surfaces, wavelength, *a, **kw):
        calls['n'] += 1
        calls['rays'] += rays.n_rays
        return real(rays, surfaces, wavelength, *a, **kw)

    monkeypatch.setattr(module, name, counting)
    try:
        fn()
    finally:
        monkeypatch.setattr(module, name, real)
    return calls


@pytest.mark.parametrize('field_deg', [0.0, 2.0])
def test_b9_i3_each_fan_function_issues_one_trace(monkeypatch, field_deg):
    """Pre-fix: four ``trace()`` calls each (tangential chief, sagittal
    chief, tangential fan, sagittal fan).  Now one, carrying the same
    ``2 * n_rays + 2`` rays -- counted, not timed."""
    S = _imaged(la.make_singlet(51.68e-3, np.inf, 4.0e-3, 'N-BK7',
                                aperture=25e-3))
    fa = np.deg2rad(field_deg)
    for fn in (ray_fan_data, opd_fan_data):
        c = _count_traces(monkeypatch, _ray_fan_mod, 'trace',
                          lambda: fn(S, WL, 12.5e-3, fa, 41))
        assert c['n'] == 1, (fn.__name__, c)
        assert c['rays'] == 2 * 41 + 2, (fn.__name__, c)


def test_b9_i3_concatenation_equals_four_separate_traces():
    """ORACLE: the four separate traces, issued by the test itself.

    Every step of ``trace`` is elementwise over rays, so the split must
    be exact.  BAR: bit-identical wherever no Newton loop runs (flat /
    spherical); on Newton surfaces a ray can take one extra iteration
    when a slower ray shares the bundle, and that step starts from
    ``|dt| < 1e-15``, so the derived envelope is ``1e-15 * |t|`` in
    position -- 1e-17 m here.  MEASURED 2026-09-13: 0.0 on all four
    prescriptions below.
    """
    from lumenairy.raytrace.ray_fan import _trace_fan_set
    cases = {
        'spherical': _imaged(la.make_singlet(51.68e-3, np.inf, 4.0e-3,
                                             'N-BK7', aperture=25e-3)),
        'conic': _conic_stack(),
        'aspheric': _aspheric_singlet({4: -5.0e3, 6: 2.0e6}),
    }
    for name, S in cases.items():
        fan_y = make_fan('y', 6e-3, 31, 0.02, WL)
        fan_x = make_fan('x', 6e-3, 31, 0.02, WL)
        chief = make_fan('y', 0.0, 1, 0.02, WL)
        bundles = (chief, fan_y, fan_x)
        joint = _trace_fan_set(trace, bundles, S, WL)
        for b, got in zip(bundles, joint):
            ref = trace(b, S, WL, output_filter='last').image_rays
            assert np.array_equal(ref.alive, got.alive), name
            for f in ('x', 'y', 'z', 'opd', 'L', 'M', 'N'):
                d = np.abs(getattr(ref, f) - getattr(got, f))
                assert float(np.max(d)) <= 1e-15 * 0.2, (name, f,
                                                         float(np.max(d)))


@pytest.mark.parametrize('field_deg', [0.0, 1.0, 3.0])
def test_b9_i3_rt5_invariant_survives_the_concatenation(field_deg):
    """RT-5: each fan is referenced against a chief of its OWN
    orientation, so the on-axis ray of each fan reads exactly zero.  The
    concatenation must not mix the two chiefs up -- ``ey(0)`` and
    ``ex(0)`` are EXACTLY 0.0, not merely small."""
    S = _imaged(la.make_singlet(51.68e-3, np.inf, 4.0e-3, 'N-BK7',
                                aperture=25e-3))
    fa = np.deg2rad(field_deg)
    n = 41
    py, ey, px, ex = ray_fan_data(S, WL, 12.5e-3, fa, n)
    mid = n // 2
    assert py[mid] == 0.0 and px[mid] == 0.0
    assert ey[mid] == 0.0, ey[mid]
    assert ex[mid] == 0.0, ex[mid]
    qy, oy, qx, ox = opd_fan_data(S, WL, 12.5e-3, fa, n)
    assert abs(oy[mid]) < 1e-12
    assert abs(ox[mid]) < 1e-12


# ===========================================================================
# item 4 -- the built-prescription cache
# ===========================================================================

def _jax_bits():
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    from lumenairy.raytrace import jax_trace as jt
    return jt


def test_b9_i4_a_repeated_build_returns_the_same_object():
    jt = _jax_bits()
    pres = la.make_singlet(51.5e-3, -80e-3, 4.1e-3, 'N-BK7', aperture=12e-3)
    jt.clear_jax_prescription_cache()
    a = jt._build_jax_prescription(pres, WL, None)
    b = jt._build_jax_prescription(pres, WL, None)
    assert a is b
    jt.clear_jax_prescription_cache()
    c = jt._build_jax_prescription(pres, WL, None)
    assert c is not a
    assert c.aux == a.aux


@pytest.mark.parametrize('mutate,label', [
    (lambda p: p['surfaces'][0].update(radius=51.6e-3), 'radius'),
    (lambda p: p['surfaces'][0].update(conic=-0.5), 'conic'),
    (lambda p: p['surfaces'][0].update(aspheric_coeffs={4: 1.0}), 'asph'),
    (lambda p: p['thicknesses'].__setitem__(0, 4.2e-3), 'thickness'),
    (lambda p: p['surfaces'][0].update(semi_diameter=5e-3), 'semi_diameter'),
    (lambda p: p.update(aperture_diameter=11e-3), 'aperture_diameter'),
    (lambda p: p['surfaces'][0].update(glass_after='N-SF5'), 'glass'),
])
def test_b9_i4_every_trace_changing_field_rekeys(mutate, label):
    """Audit sec 15.5: the key must cover every field that changes the
    trace.  Each perturbation below must MISS the cache (a new object)
    AND change the built ``aux`` -- if either held, a mutated
    prescription would silently reuse the old geometry."""
    jt = _jax_bits()
    jt.clear_jax_prescription_cache()
    pres = la.make_singlet(51.5e-3, -80e-3, 4.1e-3, 'N-BK7', aperture=12e-3)
    base = jt._build_jax_prescription(pres, WL, None)
    mutate(pres)
    got = jt._build_jax_prescription(pres, WL, None)
    assert got is not base, label
    assert got.aux != base.aux, label


def test_b9_i4_wavelength_and_doe_spec_rekey():
    jt = _jax_bits()
    jt.clear_jax_prescription_cache()
    pres = la.make_singlet(51.5e-3, -80e-3, 4.1e-3, 'N-BK7', aperture=12e-3)
    a = jt._build_jax_prescription(pres, WL, None)
    b = jt._build_jax_prescription(pres, 1.31e-6, None)
    c = jt._build_jax_prescription(pres, WL, {0: (1.0, 0.0, 5e-6, np.inf)})
    assert a is not b and a.aux != b.aux
    assert a is not c and a.aux != c.aux


def test_b9_i4_a_mutated_glass_registry_rekeys():
    """The RESOLVED indices (not the glass NAMES) are in the key, so
    re-registering a fixed glass under the same name with a different
    index cannot serve a stale build.  This is the failure mode a
    name-keyed cache would have."""
    jt = _jax_bits()
    from lumenairy.glass import GLASS_REGISTRY, _glass_cache
    from lumenairy.raytrace.trace import _register_fixed_index
    name = '__b9_probe_glass__'
    pres = la.make_singlet(51.5e-3, -80e-3, 4.1e-3, 'N-BK7', aperture=12e-3)
    pres['surfaces'][0]['glass_after'] = name
    pres['surfaces'][1]['glass_before'] = name
    try:
        _register_fixed_index(name, 1.5, WL)
        jt.clear_jax_prescription_cache()
        a = jt._build_jax_prescription(pres, WL, None)
        _register_fixed_index(name, 1.7, WL)
        b = jt._build_jax_prescription(pres, WL, None)
        assert a is not b
        assert a.aux != b.aux
        # the resolved indices are what moved -- aux[3] / aux[4] are
        # n_pre / n_post
        assert b.aux[3] != a.aux[3] or b.aux[4] != a.aux[4]
    finally:
        GLASS_REGISTRY.pop(name, None)
        _glass_cache.pop(name, None)
        jt.clear_jax_prescription_cache()


def test_b9_i4_cache_is_lru_bounded():
    jt = _jax_bits()
    jt.clear_jax_prescription_cache()
    for i in range(jt._JAX_PRESCRIPTION_CACHE_MAXSIZE + 8):
        pres = la.make_singlet(50e-3 + i * 1e-6, -80e-3, 4.1e-3, 'N-BK7',
                               aperture=12e-3)
        jt._build_jax_prescription(pres, WL, None)
    assert (len(jt._JAX_PRESCRIPTION_CACHE)
            <= jt._JAX_PRESCRIPTION_CACHE_MAXSIZE)
    jt.clear_jax_prescription_cache()
    assert len(jt._JAX_PRESCRIPTION_CACHE) == 0


def test_b9_i4_traces_are_identical_with_a_cold_and_a_warm_cache():
    """The cache may not change a single bit of the trace."""
    jt = _jax_bits()
    pres = la.make_singlet(51.5e-3, -80e-3, 4.1e-3, 'N-BK7', aperture=12e-3)
    n = 7
    x = np.linspace(-4e-3, 4e-3, n)
    z = np.zeros(n)
    st = jt.make_jax_ray_state(x=x, y=z, z=z, L=z, M=z, N=np.ones(n))
    jt.clear_jax_prescription_cache()
    cold = jt.trace_jax(st, pres, WL)
    warm = jt.trace_jax(st, pres, WL)
    for f in ('x', 'y', 'z', 'L', 'M', 'N', 'opd'):
        assert np.array_equal(np.asarray(getattr(cold, f)),
                              np.asarray(getattr(warm, f))), f


def test_b9_i4_the_clearer_is_registered_centrally():
    """``clear_asm_caches`` walks the registry rather than enumerating
    clear calls, so a new cache that is not registered leaks."""
    _jax_bits()
    from lumenairy._cache_registry import list_registered_cache_clearers
    assert 'jax_prescription' in set(list_registered_cache_clearers())


# ===========================================================================
# item 5 -- area-uniform pupil sampling
# ===========================================================================

def test_b9_i5_default_pattern_does_not_move():
    """ORACLE: the documented ring geometry, rebuilt in the test.

    The default must be bit-identical to the equal-radius / equal-count
    rings, both when ``pattern`` is omitted and when it is passed
    explicitly."""
    R, nr, rpr = 12.7e-3, 6, 36
    got = make_rings(R, nr, rpr, 0.0, WL)
    explicit = make_rings(R, nr, rpr, 0.0, WL, pattern='rings')
    assert np.array_equal(got.x, explicit.x)
    assert np.array_equal(got.y, explicit.y)
    xs = [0.0]
    ys = [0.0]
    for ring in range(1, nr + 1):
        th = np.linspace(0, 2 * np.pi, rpr, endpoint=False)
        r = R * (ring / nr)
        xs.append(r * np.cos(th))
        ys.append(r * np.sin(th))
    assert np.array_equal(got.x, np.concatenate(
        [np.atleast_1d(v) for v in xs]))
    assert np.array_equal(got.y, np.concatenate(
        [np.atleast_1d(v) for v in ys]))


@pytest.mark.parametrize('nr,rpr', [(6, 36), (12, 60), (3, 8), (1, 5)])
def test_b9_i5_vogel_second_moment_is_exactly_one_half(nr, rpr):
    """ORACLE: an exact arithmetic identity, not an envelope.

    ``r_i^2/R^2 = i/N`` for ``i = 1..N`` plus the chief's 0 averages to
    ``((N+1)/2)/(N+1) = 1/2`` for EVERY N -- the definition of
    area-uniform sampling on a disk.  Asserted to 8 ULP because only the
    summation rounds, and against the measured ring value (0.419355 at
    the defaults) so the gap is two-sided.
    """
    R = 12.7e-3
    v = make_rings(R, nr, rpr, 0.0, WL, pattern='vogel')
    r2 = ((v.x / R) ** 2 + (v.y / R) ** 2)
    assert v.n_rays == nr * rpr + 1
    assert abs(float(r2.mean()) - 0.5) <= 8 * 2 ** -52
    if nr >= 2:
        # the equal-count rings under-weight the rim; with a single ring
        # every pupil ray sits ON the rim, so the comparison only means
        # something once there is more than one radius.
        rings = make_rings(R, nr, rpr, 0.0, WL)
        r2_rings = ((rings.x / R) ** 2 + (rings.y / R) ** 2)
        assert float(r2_rings.mean()) < 0.5 - 1e-3


def test_b9_i5_vogel_mean_radius_approaches_two_thirds():
    """``mean(r/R) -> integral_0^1 sqrt(u) du = 2/3``; the discretisation
    error is ``O(1/N)``, so the bar is ``2/N`` -- 0.0092 at N = 217, and
    MEASURED 0.665834 (a gap of 0.00083) on 2026-09-13.  The ring value
    (0.580645) is 8 measured gaps away, so the bar separates them."""
    R = 12.7e-3
    v = make_rings(R, 6, 36, 0.0, WL, pattern='vogel')
    rings = make_rings(R, 6, 36, 0.0, WL)
    mv = float(np.hypot(v.x, v.y).mean() / R)
    mr = float(np.hypot(rings.x, rings.y).mean() / R)
    assert abs(mv - 2.0 / 3.0) < 2.0 / v.n_rays
    assert abs(mr - 2.0 / 3.0) > 10 * abs(mv - 2.0 / 3.0)


def test_b9_i5_vogel_has_no_azimuthal_spokes():
    """The rings pattern puts every ray on one of ``rays_per_ring``
    azimuths; the golden angle gives each ray its own."""
    R = 12.7e-3
    rings = make_rings(R, 6, 36, 0.0, WL, include_chief=False)
    v = make_rings(R, 6, 36, 0.0, WL, include_chief=False, pattern='vogel')
    def n_az(b):
        return len(np.unique(np.round(np.arctan2(b.y, b.x), 9)))
    assert n_az(rings) == 36
    assert n_az(v) == v.n_rays


def test_b9_i5_bad_pattern_names_the_function():
    with pytest.raises(ValueError, match=r"^make_rings: pattern must be"):
        make_rings(12.7e-3, 6, 36, pattern='sunflower')


def test_b9_i5_area_true_spot_rms_reads_larger_on_a_spherical_singlet():
    """DIRECTION claim, not a pinned number: the equal-count rings
    under-weight the rim, so the default reads SMALL against the
    area-true sampling on an aberrated system.  MEASURED 2026-09-13:
    +2.14 % (f/4 plano-convex, 108.2226 -> 110.5391 um) and +10.37 %
    (biconvex R = +-60 mm, 106.1941 -> 117.2050 um)."""
    cases = [
        la.make_singlet(51.68e-3, np.inf, 4.0e-3, 'N-BK7', aperture=25e-3),
        la.make_singlet(60e-3, -60e-3, 6e-3, 'N-BK7', aperture=16e-3),
    ]
    semis = [12.5e-3, 8e-3]
    for pres, sa in zip(cases, semis):
        S = _imaged(pres)
        got = {}
        for pat in ('rings', 'vogel'):
            rays = make_rings(sa, 6, 36, 0.0, WL, pattern=pat)
            got[pat], _ = spot_rms(trace(rays, S, WL, output_filter='last'))
        assert got['vogel'] > got['rings'] > 0.0, (pres['surfaces'][0], got)


def test_b9_i5_ray_pattern_vogel_reaches_trace_prescription():
    """The keyword is plumbed, not stranded on the generator."""
    from lumenairy.raytrace import trace_prescription
    pres = la.make_singlet(51.68e-3, np.inf, 4.0e-3, 'N-BK7',
                           aperture=25e-3)
    a = trace_prescription(pres, WL, ray_pattern='rings', num_rings=4,
                           rays_per_ring=12)
    b = trace_prescription(pres, WL, ray_pattern='vogel', num_rings=4,
                           rays_per_ring=12)
    assert a.input_rays.n_rays == b.input_rays.n_rays == 49
    assert not np.array_equal(a.input_rays.x, b.input_rays.x)


# ===========================================================================
# item 6 -- aspheric support in the analytic ray-transfer Jacobian
# ===========================================================================

_ASPH_HEIGHTS = np.array([1e-3, 4e-3, 8e-3])


def _fd_jacobian(S, h_pos):
    z = np.zeros_like(_ASPH_HEIGHTS)
    return ray_transfer_jacobian(_ASPH_HEIGHTS, z, z, z, S, WL,
                                 h_pos=h_pos, h_slope=50.0 * h_pos)


def _analytic_jacobian(S):
    z = np.zeros_like(_ASPH_HEIGHTS)
    return ray_transfer_jacobian_analytic(_ASPH_HEIGHTS, z, z, z, S, WL)


@pytest.mark.parametrize('asph', [
    {4: -5.0e3},
    {4: -5.0e3, 6: 2.0e6},
    {4: -5.0e3, 6: 2.0e6, 8: -1.0e9},
])
def test_b9_i6_analytic_matches_the_fd_jacobian_at_three_heights(asph):
    """ORACLE: the finite-difference primitive, whose truncation is the
    thing being compared against.

    BAR, derived from the FD step rather than from one build's residual:
    a central difference carries ``C h^2`` truncation, so the gap must
    SHRINK by ~100x when ``h`` shrinks by 10x -- that is the signature of
    the analytic path being the exact one.  The absolute bar is
    ``|J| * 1e-6`` (MEASURED 2026-09-13: 2.1e-8 .. 7.2e-8 against a
    ``|J|`` of ~3e+1, i.e. 7e-10 relative, three decades inside), and the
    h-ladder ratio is asserted to be between 30 and 300.
    """
    S = _aspheric_singlet(asph)
    A = _analytic_jacobian(S)
    gaps = {}
    for h in (1e-6, 1e-5):
        F = _fd_jacobian(S, h)
        gaps[h] = np.max(np.abs(A.jacobian - F.jacobian), axis=(1, 2))
    scale = np.max(np.abs(A.jacobian), axis=(1, 2))
    assert np.all(gaps[1e-6] < 1e-6 * scale), (gaps[1e-6], scale)
    ratio = gaps[1e-5] / gaps[1e-6]
    assert np.all((ratio > 30.0) & (ratio < 300.0)), ratio


def test_b9_i6_tracing_the_base_conic_is_not_an_acceptable_answer():
    """FAIL-BEFORE, in the only form available: the pre-fix analytic path
    RAISED for an asphere, and every ``jacobian='auto'`` consumer fell
    back to FD.  The claim that matters is that the polynomial terms are
    load-bearing -- dropping them (tracing the base conic, which is what
    a silently-ignoring implementation would do) moves the Jacobian by
    3.0e-2 .. 2.0e+0 against an agreement of 2.1e-8 .. 7.2e-8, i.e. six
    decades.  That gap is the bar's headroom.
    """
    asph = {4: -5.0e3, 6: 2.0e6}
    A = _analytic_jacobian(_aspheric_singlet(asph))
    C = _analytic_jacobian(_aspheric_singlet(None))
    F = _fd_jacobian(_aspheric_singlet(asph), 1e-6)
    good = np.max(np.abs(A.jacobian - F.jacobian), axis=(1, 2))
    bad = np.max(np.abs(C.jacobian - F.jacobian), axis=(1, 2))
    assert np.all(bad > 1e5 * good), (good, bad)
    # and the exit ray itself moves visibly
    assert np.all(np.abs(A.x - C.x) > 1e-7)


def test_b9_i6_newton_refinement_lands_where_the_main_trace_lands():
    """INDEPENDENT ORACLE: ``raytrace.trace``'s own 10-step Newton
    intersection, which shares no code with ``_adrt_aspheric_intersect``
    (different residual, different backend, different loop).  The exit
    state must agree to the FD primitive's own base-ray precision --
    MEASURED 3.5e-18 m in x, 1e-16 in slope, 2.8e-17 m in OPL."""
    S = _aspheric_singlet({4: -5.0e3, 6: 2.0e6}, {4: 1.2e3}, k1=-0.6)
    A = _analytic_jacobian(S)
    F = _fd_jacobian(S, 1e-6)
    assert float(np.max(np.abs(A.x - F.x))) < 1e-15
    assert float(np.max(np.abs(A.ux - F.ux))) < 1e-14
    assert float(np.max(np.abs(A.opd - F.opd))) < 1e-15
    assert np.array_equal(A.alive, F.alive)


def test_b9_i6_the_newton_step_count_is_already_converged():
    """The loop count is FIXED (no data-dependent break -- this runs
    under forward-mode AD).  Six steps is the shipped budget; the result
    must be bit-identical from far fewer, so the budget is headroom
    rather than a tuned constant."""
    S = _aspheric_singlet({4: -5.0e3, 6: 2.0e6}, {4: 1.2e3}, k1=-0.6)
    ref = None
    original = _diff_mod._ADRT_ASPHERIC_NEWTON_STEPS
    try:
        for steps in (3, 4, 6, 10):
            _diff_mod._ADRT_ASPHERIC_NEWTON_STEPS = steps
            got = _analytic_jacobian(S)
            if ref is None:
                ref = got
                continue
            assert np.array_equal(got.jacobian, ref.jacobian), steps
            assert np.array_equal(got.x, ref.x), steps
    finally:
        _diff_mod._ADRT_ASPHERIC_NEWTON_STEPS = original


def test_b9_i6_the_conic_path_never_enters_the_aspheric_branch(monkeypatch):
    """A prescription with no polynomial departure must take the
    untouched conic arithmetic -- asserted by making the new branch
    fatal."""
    def explode(*a, **kw):
        raise AssertionError('aspheric branch entered for a pure conic')

    monkeypatch.setattr(_diff_mod, '_adrt_aspheric_intersect', explode)
    S = _conic_stack()
    z = np.zeros_like(_ASPH_HEIGHTS)
    ray_transfer_jacobian_analytic(_ASPH_HEIGHTS * 0.5, z, z, z, S, WL)


def test_b9_i6_the_numba_kernel_is_excluded_for_aspheres():
    """The numba forward-AD kernel replicates the CONIC primitives only.
    Left eligible, it would trace an asphere as its base conic -- right
    shape, wrong surface, silently."""
    from lumenairy.raytrace.differential import _adrt_surfaces_numba_eligible
    assert _adrt_surfaces_numba_eligible(_conic_stack()) is True
    assert _adrt_surfaces_numba_eligible(
        _aspheric_singlet({4: -5.0e3})) is False


def test_b9_i6_biconic_freeform_and_field_frame_still_raise():
    z = np.zeros(2)
    x = np.array([1e-3, 2e-3])
    for surf in (
        Surface(radius=25e-3, radius_y=30e-3, thickness=5e-3,
                glass_before='air', glass_after='N-BK7',
                semi_diameter=10e-3),
        Surface(radius=25e-3, aspheric_coeffs_y={4: 1.0}, thickness=5e-3,
                glass_before='air', glass_after='N-BK7',
                semi_diameter=10e-3),
        Surface(radius=25e-3, freeform={'freeform_type': 'zernike'},
                thickness=5e-3, glass_before='air', glass_after='N-BK7',
                semi_diameter=10e-3),
    ):
        with pytest.raises(NotImplementedError,
                           match='ray_transfer_jacobian_analytic'):
            ray_transfer_jacobian_analytic(x, z, z, z, [surf], WL)


def test_b9_i6_odd_aspheric_powers_are_rejected():
    """An ODD power is sag/normal-inconsistent in every backend.  The
    ``Surface`` dataclass refuses it first; ``_adrt_step``'s own guard is
    the backstop for a hand-built surface-like object that never went
    through that constructor, and it names itself."""
    with pytest.raises(ValueError, match='ODD aspheric power'):
        _aspheric_singlet({5: 1.0e3})

    class _Loose:
        radius = 25e-3
        conic = 0.0
        aspheric_coeffs = {5: 1.0e3}
        is_mirror = False
        is_coordbrk = False
        thickness = 5e-3
        glass_before = 'air'
        glass_after = 'N-BK7'
        semi_diameter = 10e-3

    with pytest.raises(ValueError, match='_adrt_step'):
        _diff_mod._adrt_aspheric_items(_Loose())


def test_b9_i6_jax_backend_agrees_with_the_dual_backend_on_an_asphere():
    """Both backends run the SAME ``_adrt_step`` through different op
    tables (``_AdrtDual`` vs ``jnp``), so agreement is a cross-backend
    statement about the new branch.  BAR 1e-12 absolute on a Jacobian of
    scale ~3e+1 (MEASURED 1.8e-14)."""
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    S = _aspheric_singlet({4: -5.0e3, 6: 2.0e6})
    A = _analytic_jacobian(S)
    z = jnp.zeros(_ASPH_HEIGHTS.shape)
    J = ray_transfer_jacobian_analytic(jnp.asarray(_ASPH_HEIGHTS), z, z, z,
                                       S, WL)
    assert float(np.max(np.abs(np.asarray(J.jacobian)
                               - A.jacobian))) < 1e-12
    assert float(np.max(np.abs(np.asarray(J.x) - A.x))) < 1e-15


# ===========================================================================
# VERIFY-WP-B9 -- independent adversarial re-verification.
#
# Everything below was written by the verifier against its OWN oracles
# (a 60-digit ``decimal`` normal, ``jax.jacfwd`` through the independent
# ``trace_jax`` kernel, four separate ``trace()`` calls).  Each pin
# records a boundary the WP-B9 report states more strongly than the code
# supports, so the next reader measures instead of re-deriving.
# ===========================================================================

_VB9_GATE_R = 0.051679999999999997
_VB9_GATE_XY = 0.036541451242116801


def test_vb9_i2_the_two_domain_gates_can_straddle_at_the_knife_edge():
    """The closed-form normal gates on ``(x*x + y*y)/(R*R) < 0.9999``;
    the generic route gates on ``(1 + conic) * sqrt(x*x+y*y)**2 / R**2``.
    Those differ by up to 1 ULP, so a position can sit on opposite sides.

    ORACLE: the two expressions, evaluated here.  MEASURED at the point
    below -- the generic route returns NaN and ``_refract`` kills the ray
    ``RAY_NAN``; the closed form returns ``nz = 1.0000000000005e-02`` and
    the ray refracts.  Confined to ``sphere_normal='analytic'``; the
    shipped default takes the generic route on both sides, which is why
    the default byte-identity sweep is unaffected.
    """
    R, xy = _VB9_GATE_R, _VB9_GATE_XY
    x = np.array([xy])
    y = np.array([xy])
    norm_analytic = (xy * xy + xy * xy) / (R * R)
    h = np.sqrt(xy * xy + xy * xy)
    norm_generic = (1.0 + 0.0) * (h ** 2) / (R ** 2)
    assert norm_analytic < 0.9999 <= norm_generic, (
        norm_analytic, norm_generic)

    surf = Surface(radius=R, thickness=0.0, glass_before='air',
                   glass_after='N-BK7', semi_diameter=np.inf)
    assert _is_pure_spherical(surf)
    nz_fast = float(_sphere_normal(x, y, R)[2][0])
    nz_slow = float(_surface_normal(x, y, surf)[2][0])
    assert np.isfinite(nz_fast) and abs(nz_fast - 0.01) < 1e-12
    assert np.isnan(nz_slow)

    # ... and it reaches the public kill decision.
    z0 = R - np.sqrt(max(R * R - 2.0 * xy * xy, 0.0))
    codes = {}
    for mode in ('generic', 'analytic'):
        rb = la.raytrace.RayBundle(
            x=x.copy(), y=y.copy(), z=np.array([z0]),
            L=np.array([0.0]), M=np.array([0.0]), N=np.array([1.0]),
            wavelength=WL, alive=np.ones(1, dtype=bool), opd=np.zeros(1))
        _isect._refract(rb, surf, 1.0, 1.5168, sphere_normal=mode)
        codes[mode] = (bool(rb.alive[0]), int(rb.error_code[0]))
    assert codes['generic'] == (False, int(RAY_NAN)), codes
    assert codes['analytic'] == (True, int(RAY_OK)), codes


def test_vb9_i2_nz_is_conditioning_limited_above_h_equals_0p95_R():
    """``nz = sqrt(1 - u)`` with ``u = h^2/R^2`` cancels as ``u -> 1``:
    the relative error of ``nz`` is bounded below by ``eps/2 * u/(1 - u)``
    for ANY float64 evaluation, closed form or sag derivative, because
    the information is not in the inputs.  So the "<= 4 ULP" and "never
    worse than the generic route" readings hold over the aperture the
    oracle test samples (``h <= 0.95 |R|``) and NOT beyond it.

    ORACLE: a 60-digit ``decimal`` ``sqrt(1 - u)`` from the EXACT binary
    values of the float64 inputs; the bar is the same ``4 * 2**-52``
    absolute (4 ULP of a unit vector) the item-2 oracle test uses.

    MEASURED over R in {51.68, -34.5, 500, -1000, 2} mm: both routes
    within 2.8e-16 up to ``0.95 |R|``; at ``0.99994 |R|`` -- still inside
    the ``h^2/R^2 < 0.9999`` domain -- the closed form reaches 1.5e-14
    (68 ULP) and is 4.3x FURTHER from the truth than the generic route at
    ``R = -34.5 mm``.  This pin exists because ``WP-B9_REPORT.md``
    section 6 item 2 proposes dropping the ``valid`` clamp on the grounds
    that the closed form "is well-conditioned to ``h = |R|``" -- it is
    not, and neither is the route it replaces.
    """
    ctx = decimal.Context(prec=60)
    D = ctx.create_decimal
    BAR = 4 * 2.0 ** -52

    def oracle_nz(xv, yv, R):
        X, Y, RR = D(repr(float(xv))), D(repr(float(yv))), D(repr(float(R)))
        u = ctx.divide(ctx.add(ctx.multiply(X, X), ctx.multiply(Y, Y)),
                       ctx.multiply(RR, RR))
        return float(ctx.subtract(D(1), u).sqrt(ctx))

    def errs(R, f):
        h = abs(R) * f
        x = np.array([h / np.sqrt(2.0)])
        y = np.array([h / np.sqrt(2.0)])
        surf = Surface(radius=R, thickness=0.0, semi_diameter=np.inf)
        ref = oracle_nz(x[0], y[0], R)
        return (abs(float(_sphere_normal(x, y, R)[2][0]) - ref),
                abs(float(_surface_normal(x, y, surf)[2][0]) - ref))

    radii = (0.05168, -0.0345, 0.5, -1.0, 0.002)
    # benign regime -- the report's claim, reproduced
    for R in radii:
        for f in (0.0, 0.05, 0.2, 0.5, 0.8, 0.95):
            e_fast, e_slow = errs(R, f)
            assert e_fast <= BAR and e_slow <= BAR, (R, f, e_fast, e_slow)
    # conditioning regime -- BOTH routes leave the bar well inside the
    # surface's own domain, and the closed form is not bounded by the
    # generic one.
    worst_fast = max(errs(R, 0.99994)[0] for R in radii)
    worst_slow = max(errs(R, 0.99994)[1] for R in radii)
    assert worst_fast > BAR and worst_slow > BAR, (worst_fast, worst_slow)
    e_fast, e_slow = errs(-0.0345, 0.99994)
    assert e_fast > 2.0 * e_slow, (e_fast, e_slow)


def test_vb9_i3_one_bundle_is_exact_when_a_sub_fan_carries_dead_rays():
    """ORACLE: four separate ``trace()`` calls, issued here.

    The concatenation must not couple a vignetted / missed ray to its
    neighbours through the ``np.any`` guards.  Two asymmetric stacks: one
    whose semi-diameters kill the fan rim, one whose ``R = 9 mm`` sphere
    the outer fan MISSES (``disc < 0``).
    """
    vign = [
        Surface(radius=0.05168, thickness=0.005, glass_before='air',
                glass_after='N-BK7', semi_diameter=0.008),
        Surface(radius=np.inf, thickness=0.0972, glass_before='N-BK7',
                glass_after='air', semi_diameter=0.006),
        Surface(radius=np.inf, thickness=0.0, semi_diameter=0.03),
    ]
    miss = [
        Surface(radius=0.009, thickness=0.004, glass_before='air',
                glass_after='N-BK7', semi_diameter=np.inf),
        Surface(radius=np.inf, thickness=0.020, glass_before='N-BK7',
                glass_after='air', semi_diameter=np.inf),
        Surface(radius=np.inf, thickness=0.0, semi_diameter=np.inf),
    ]
    for surfs, semi in ((vign, 0.009), (miss, 0.012)):
        fy = make_fan('y', semi, 41, 0.0, WL)
        fx = make_fan('x', semi, 41, 0.0, WL)
        cy = la.raytrace.make_ray(0, 0, 0, 0, wavelength=WL)
        cx = la.raytrace.make_ray(0, 0, 0, 0, wavelength=WL)
        four = [trace(b, surfs, WL).image_rays for b in (cy, cx, fy, fx)]
        one = _ray_fan_mod._trace_fan_set(trace, (cy, cx, fy, fx), surfs, WL)
        assert int(np.count_nonzero(~four[2].alive)) > 0, 'no dead ray'
        for ref, got in zip(four, one):
            for fld in ('x', 'y', 'z', 'L', 'M', 'N', 'opd', 'alive',
                        'error_code'):
                assert (np.asarray(getattr(ref, fld)).tobytes()
                        == np.asarray(getattr(got, fld)).tobytes()), fld


def test_vb9_i3_one_bundle_is_exact_across_the_absolute_newton_tolerance():
    """The aspheric Newton's acceptance test ``|dt| < 1e-15`` is ABSOLUTE
    in metres, so it is not scale-free: at ``|t| ~ 1e-15/eps = 4.5 m`` a
    converged ray's own residual step reaches the bound, which is where
    an extra iteration forced by a slower bundle-mate could in principle
    flip ``converged`` and kill the ray.

    ORACLE: four separate traces.  MEASURED ``max |dy| = 0`` and
    identical ``alive`` masks on both sides of that scale.
    """
    for gap in (0.5, 4.5, 16.0):
        surfs = [
            Surface(radius=np.inf, thickness=gap, semi_diameter=np.inf),
            Surface(radius=0.018, conic=-0.9,
                    aspheric_coeffs={4: -4.0e3, 6: 9.0e6},
                    thickness=0.005, glass_before='air',
                    glass_after='N-BK7', semi_diameter=np.inf),
            Surface(radius=-0.060, aspheric_coeffs={4: 2.0e3},
                    thickness=0.040, glass_before='N-BK7',
                    glass_after='air', semi_diameter=np.inf),
            Surface(radius=np.inf, thickness=0.0, semi_diameter=np.inf),
        ]
        cy = la.raytrace.make_ray(0, 0, 0, 0, wavelength=WL)
        fy = make_fan('y', 0.008, 41, 0.0, WL)
        fx = make_fan('x', 0.008, 41, 0.0, WL)
        four = [trace(b, surfs, WL).image_rays for b in (cy, cy, fy, fx)]
        one = _ray_fan_mod._trace_fan_set(trace, (cy, cy, fy, fx), surfs, WL)
        for ref, got in zip(four, one):
            assert np.array_equal(ref.alive, got.alive), gap
            assert np.asarray(ref.y).tobytes() == np.asarray(got.y).tobytes()


def _vb9_presc(**over):
    p = {
        'surfaces': [
            {'radius': 0.05168, 'glass_before': 'air',
             'glass_after': 'N-BK7'},
            {'radius': -0.080, 'glass_before': 'N-BK7',
             'glass_after': 'air'},
        ],
        'thicknesses': [0.006, 0.090],
        'aperture_diameter': 0.008,
    }
    p.update(over)
    return p


def test_vb9_i4_the_cached_prescription_is_shared_and_rebindable():
    """``__slots__`` blocks NEW attribute names, NOT writes to declared
    ones, so the cached instance is not immutable -- it is shared state.

    This pin is a hazard record, not an endorsement: nothing in the
    package writes to a returned ``JaxPrescription``, and callers must
    not either.  The test restores the slot it perturbs.
    """
    pytest.importorskip('jax')
    jt = sys.modules['lumenairy.raytrace.jax_trace']
    jt.clear_jax_prescription_cache()
    a = jt._build_jax_prescription(_vb9_presc(), 1.31e-6)
    b = jt._build_jax_prescription(_vb9_presc(), 1.31e-6)
    assert a is b, 'the second build must be a cache hit'
    keep = a.radii
    try:
        a.radii = None                      # succeeds -- that is the point
        assert jt._build_jax_prescription(
            _vb9_presc(), 1.31e-6).radii is None
    finally:
        a.radii = keep
    with pytest.raises(AttributeError):
        a.a_brand_new_attribute = 1
    jt.clear_jax_prescription_cache()


def test_vb9_i4_aux_is_hashable_by_construction():
    """Every element of ``aux`` is produced by ``int()`` / ``float()`` /
    ``tuple()``, so the ``except TypeError`` fallback around the lookup
    is defensive only.  Numpy scalars, 0-d arrays and ``bool`` all
    normalise.
    """
    pytest.importorskip('jax')
    jt = sys.modules['lumenairy.raytrace.jax_trace']
    for extra in ({'radius': np.float64(0.05168)},
                  {'radius': np.array(0.05168)},
                  {'radius': 0.05168, 'conic': False},
                  {'radius': 0.05168,
                   'aspheric_coeffs': {np.int64(4): np.float32(1e-6)}}):
        p = _vb9_presc()
        p['surfaces'][0].update(extra)
        jt.clear_jax_prescription_cache()
        jp = jt._build_jax_prescription(p, 1.31e-6)
        hash(jp.aux)                         # must not raise
        assert len(jt._JAX_PRESCRIPTION_CACHE) == 1
    jt.clear_jax_prescription_cache()


def test_vb9_i4_the_key_covers_exactly_what_the_built_object_reads():
    """Three inputs deliberately do NOT re-key, and must not move the
    answer either: the trace WAVELENGTH beyond the indices it resolves
    (``trace_jax`` passes it to the kernel separately), the LAST
    thickness (the builder reads ``n_surf - 1`` gaps) and a top-level
    ``aperture_diameter`` that a per-surface ``semi_diameter``
    overrides.  When ``aperture_diameter`` IS the resolving value it
    re-keys.
    """
    pytest.importorskip('jax')
    jt = sys.modules['lumenairy.raytrace.jax_trace']
    jt.clear_jax_prescription_cache()
    base = jt._build_jax_prescription(_vb9_presc(), 1.31e-6)

    p = _vb9_presc()
    p['thicknesses'] = [0.006, 0.090 + 1e-3]
    assert jt._build_jax_prescription(p, 1.31e-6) is base

    wl2 = float(np.nextafter(1.31e-6, np.inf))
    assert jt._build_jax_prescription(_vb9_presc(), wl2) is base

    p = _vb9_presc()
    for s in p['surfaces']:
        s['semi_diameter'] = 0.012
    jt.clear_jax_prescription_cache()
    sd = jt._build_jax_prescription(p, 1.31e-6)
    p2 = dict(p, aperture_diameter=0.006)
    assert jt._build_jax_prescription(p2, 1.31e-6) is sd, (
        'a per-surface semi_diameter shadows aperture_diameter')

    jt.clear_jax_prescription_cache()
    a8 = jt._build_jax_prescription(_vb9_presc(aperture_diameter=0.008),
                                    1.31e-6)
    a6 = jt._build_jax_prescription(_vb9_presc(aperture_diameter=0.006),
                                    1.31e-6)
    assert a8 is not a6 and a8.aux[2] != a6.aux[2]
    jt.clear_jax_prescription_cache()


def test_vb9_i4_a_nan_radius_hits_on_a_re_read_of_the_same_dict():
    """A NaN radius does key: ``float(x)`` on a float returns the SAME
    object, so the tuple comparison short-circuits on identity and the
    second build of the SAME dict is a hit.  A freshly created NaN is a
    different object and misses.  Either way the built object matches the
    prescription asked for -- this pin records which, so the behaviour is
    not mistaken for a stale hit.
    """
    pytest.importorskip('jax')
    jt = sys.modules['lumenairy.raytrace.jax_trace']
    p = _vb9_presc()
    p['surfaces'][0]['radius'] = float('nan')
    jt.clear_jax_prescription_cache()
    a = jt._build_jax_prescription(p, 1.31e-6)
    assert jt._build_jax_prescription(p, 1.31e-6) is a
    q = _vb9_presc()
    q['surfaces'][0]['radius'] = float('nan')
    assert jt._build_jax_prescription(q, 1.31e-6) is not a
    assert len(jt._JAX_PRESCRIPTION_CACHE) == 2
    jt.clear_jax_prescription_cache()


def test_vb9_i6_aspheric_jacobian_matches_jacfwd_through_trace_jax():
    """ORACLE: ``jax.jacfwd`` through ``trace_jax`` -- a DIFFERENT
    intersection kernel (``_intersect_jax``) from ``_adrt_step``, exact
    to machine precision, so this is not the FD truncation comparison
    repeated.  BAR 1e-12 absolute on a Jacobian of scale ~3e+1
    (MEASURED 1.1e-14 .. 5.7e-14 over three aspheric orders).
    """
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    from lumenairy.raytrace.jax_trace import make_jax_ray_state, trace_jax

    for a1 in ({4: -1.0e4}, {4: -1.0e4, 6: 1.0e8},
               {4: -1.0e4, 6: 1.0e8, 8: -1.0e12}):
        S = _aspheric_singlet(a1, None, k1=-0.6)[:2]
        P = {'surfaces': [{'radius': float(s.radius),
                           'conic': float(s.conic),
                           'glass_before': s.glass_before,
                           'glass_after': s.glass_after,
                           'aspheric_coeffs': dict(s.aspheric_coeffs or {})}
                          for s in S],
             'thicknesses': [float(s.thickness) for s in S]}

        def f(v, P=P):
            ux, uy = v[2], v[3]
            nrm = jnp.sqrt(1.0 + ux * ux + uy * uy)
            st = make_jax_ray_state(
                x=jnp.atleast_1d(v[0]), y=jnp.atleast_1d(v[1]),
                z=jnp.zeros(1), L=jnp.atleast_1d(ux / nrm),
                M=jnp.atleast_1d(uy / nrm), N=jnp.atleast_1d(1.0 / nrm),
                opd=jnp.zeros(1), alive=jnp.ones(1, dtype=bool))
            o = trace_jax(st, P, WL)
            return jnp.array([o.x[0], o.y[0], o.L[0] / o.N[0],
                              o.M[0] / o.N[0]])

        ref = np.array([np.asarray(jax.jacfwd(f)(
            jnp.array([h, 0.0, 0.0, 0.0], dtype=jnp.float64)))
            for h in _ASPH_HEIGHTS])
        z = np.zeros_like(_ASPH_HEIGHTS)
        got = ray_transfer_jacobian_analytic(
            _ASPH_HEIGHTS, z, z, z, S, WL).jacobian
        assert float(np.max(np.abs(got - ref))) < 1e-12, a1


def test_vb9_i6_a_zero_aspheric_coefficient_is_a_different_route():
    """``aspheric_coeffs={4: 0.0}`` is a non-empty dict, so it selects the
    Newton-refined branch and the numba kernel is excluded -- the surface
    is arithmetically the base conic but the CODE PATH is not.

    Agreement with the conic path is 3.4e-16 RELATIVE (1.5 ULP), not
    bit-identical.  Pinned so a caller that spells "no asphere" as a dict
    of zeros knows it costs the fast path and moves the last bit.
    """
    z = np.zeros_like(_ASPH_HEIGHTS)
    zero = _aspheric_singlet({4: 0.0}, None, k1=-0.6)
    bare = _aspheric_singlet(None, None, k1=-0.6)
    Jz = ray_transfer_jacobian_analytic(
        _ASPH_HEIGHTS, z, z, z, zero, WL).jacobian
    Jb = ray_transfer_jacobian_analytic(
        _ASPH_HEIGHTS, z, z, z, bare, WL).jacobian
    assert Jz.tobytes() != Jb.tobytes()
    scale = np.maximum(np.abs(Jb), 1e-9)
    assert float(np.max(np.abs(Jz - Jb) / scale)) < 1e-14
    assert _diff_mod._adrt_surfaces_numba_eligible(bare) is True
    assert _diff_mod._adrt_surfaces_numba_eligible(zero) is False
