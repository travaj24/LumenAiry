"""The REMAP rung -- ``surface_model='tangent_facet_remap'``.

Route 3 (``'tangent_facet'``) REFERENCES the transverse walk away and truncates
the referencing series; this model REPRESENTS the walk by resampling the field
to the walked positions, so the element is a screen PLUS a coordinate remap.

Every bar in this file is either an EXACT structural claim (``np.array_equal``,
a refusal, a zero by construction, a conserved quantity) or a comparison against
an INDEPENDENT oracle built in this file from exact ray algebra -- closed-form
plane-facet eikonals, or a per-ray Newton intersection of a true sphere with
exact vector Snell.  Bars are derived from the oracle's own floor and the
measured numbers are dated in the comment.  Nothing pins a build's residual, a
count, or a ratio whose pass/fail boundary sits inside a cross-build spread
(docs/TESTING_STANDARDS.md S1-S5).

The three claims the file exists to defend:

* the model is EXACT for a plane facet, with no oracle and no tolerance;
* the amplitude factor is the DERIVED energy Jacobian ``1/sqrt(det A)``, checked
  against a coordinate map whose answer is known in closed form;
* the fold guard REFUSES rather than degrading, proved on an engineered folding
  prescription whose un-guarded arm is shown to be wrong.
"""
import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements import _lens_real as LR
from lumenairy.elements import apply_real_lens

LAM = 1.31e-6


@pytest.fixture(autouse=True)
def _deterministic_fft():
    """The byte-identity claims are about arithmetic, not about the FFT
    planner's dtype promotion; pin it the way the sibling suites do."""
    la.set_fft_auto_promote(False)
    yield
    la.set_fft_auto_promote(False)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def _plate(t=3.0e-3, glass='N-BK7'):
    return {'surfaces': [
        {'radius': np.inf, 'conic': 0.0, 'glass_before': 'AIR',
         'glass_after': glass},
        {'radius': np.inf, 'conic': 0.0, 'glass_before': glass,
         'glass_after': 'AIR'}],
        'thicknesses': [t]}


def _singlet(R=19.6e-3, glass='N-SSK2', t=4.0e-3, ap=3.0e-3):
    return {'surfaces': [
        {'radius': R, 'conic': 0.0, 'aspheric_coeffs': None,
         'glass_before': 'AIR', 'glass_after': glass},
        {'radius': np.inf, 'conic': 0.0, 'aspheric_coeffs': None,
         'glass_before': glass, 'glass_after': 'AIR'}],
        'thicknesses': [t], 'aperture_diameter': ap, 'name': 'singlet'}


def _biconvex(R=12.6e-3, glass='N-SSK2', t=4.0e-3, ap=3.0e-3):
    return {'surfaces': [
        {'radius': R, 'conic': 0.0, 'aspheric_coeffs': None,
         'glass_before': 'AIR', 'glass_after': glass},
        {'radius': -R, 'conic': 0.0, 'aspheric_coeffs': None,
         'glass_before': glass, 'glass_after': 'AIR'}],
        'thicknesses': [t], 'aperture_diameter': ap, 'name': 'biconvex'}


def _field(N, dx, w=None, tilt=0.0):
    a = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(a, a)   # axis 0 is y, axis 1 is x
    w = w if w is not None else 0.25 * N * dx
    E = np.exp(-(X ** 2 + Y ** 2) / w ** 2)
    if tilt:
        E = E * np.exp(1j * 2 * np.pi / LAM * tilt * X)
    return E.astype(np.complex128)


def _rm(E, presc, dx, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return apply_real_lens(E, prescription=presc, wavelength=LAM, dx=dx,
                               surface_model='tangent_facet_remap', **kw)


def _tf(E, presc, dx, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return apply_real_lens(E, prescription=presc, wavelength=LAM, dx=dx,
                               surface_model='tangent_facet', **kw)


def _thin(E, presc, dx, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return apply_real_lens(E, prescription=presc, wavelength=LAM, dx=dx,
                               **kw)


# ---------------------------------------------------------------------------
# independent oracles, built here from exact ray algebra
# ---------------------------------------------------------------------------
def _sphere_derivs(x, y, R):
    """A true sphere's sag and its first TWO analytic derivative sets.

    ``sag = R - sqrt(R^2 - r^2)`` -- verified against the library's own
    ``_surface_sag_general(r^2, R, 0, None)`` to the last bit in
    :func:`test_the_sphere_oracle_matches_the_librarys_own_sag`."""
    r2 = x * x + y * y
    q = np.sqrt(R * R - r2)
    sag = R - q
    gx = x / q
    gy = y / q
    q3 = q ** 3
    return sag, gx, gy, 1.0 / q + x * x / q3, x * y / q3, x * y / q3, \
        1.0 / q + y * y / q3


def _exact_sphere_facet(x, y, R, px, py, n1, n2, iters=80):
    """EXACT ray algebra through one spherical facet, per pixel.

    Newton the ray onto the true sphere, refract with exact vector Snell at the
    true normal, and read off the exit eikonal at the ray's own re-crossing of
    the vertex plane.  Both leg eikonals are exact in closed form because both
    legs are straight: rising to ``(x + s q, s)`` costs ``s n1^2/pz1`` and
    descending to ``(x + W, 0)`` costs ``-s n2^2/pz2``.  So this oracle has no
    quadrature and no tolerance of its own -- only the Newton residual, which is
    driven to machine precision.

    Returns ``(opd, Wx, Wy, p_out_x, p_out_y, s_hit)``."""
    pz1 = np.sqrt(n1 * n1 - px * px - py * py)
    qx, qy = px / pz1, py / pz1
    s = np.zeros_like(x)
    for _ in range(iters):
        xs, ys = x + s * qx, y + s * qy
        f = np.sqrt(R * R - xs * xs - ys * ys)
        sag = R - f
        dsag = (xs * qx + ys * qy) / f            # d(sag)/ds along the ray
        s = s - (s - sag) / (1.0 - dsag)
    xs, ys = x + s * qx, y + s * qy
    f = np.sqrt(R * R - xs * xs - ys * ys)
    gx, gy = xs / f, ys / f
    inv = 1.0 / np.sqrt(1.0 + gx * gx + gy * gy)
    a = (-gx * px - gy * py + pz1) * inv
    b = np.sqrt(n2 * n2 - n1 * n1 + a * a)
    dz = (b - a) * inv
    pox, poy = px - dz * gx, py - dz * gy
    pz2 = pz1 + dz
    return (s * (n2 * n2 / pz2 - n1 * n1 / pz1),
            s * (qx - pox / pz2), s * (qy - poy / pz2), pox, poy, s)


# ---------------------------------------------------------------------------
# 1.  THE OFF PATHS DID NOT MOVE
# ---------------------------------------------------------------------------
def test_the_default_thin_path_is_byte_identical():
    """``surface_model`` already defaulted to ``'thin'``, so the remap rung is
    reachable only through a value that did not exist in 5.36.0.  The null is
    structural -- measured anyway, on the option combinations that share the
    surface loop with it."""
    N, dx = 192, 25e-6
    E = _field(N, dx, w=1.2e-3)
    for presc in (_singlet(), _biconvex(), _plate()):
        base = apply_real_lens(E, prescription=presc, wavelength=LAM, dx=dx)
        for kw in ({}, {'surface_model': 'thin'}, {'remap_order': 3},
                   {'sag_chunk_rows': 32}, {'sag_chunk_rows': 7},
                   {'bandlimit': True}):
            got = apply_real_lens(E, prescription=presc, wavelength=LAM,
                                  dx=dx, **kw)
            assert np.array_equal(base.view(np.uint8), got.view(np.uint8)), \
                (presc.get('name'), kw)


def test_the_shipped_tangent_facet_path_cannot_reach_the_remap_machinery():
    """STRUCTURAL byte-null for ``surface_model='tangent_facet'``.

    Byte-identity against a version this tree no longer contains cannot be
    asserted from inside the tree, so the claim is made one level up and made
    UNCONDITIONAL: every remap entry point is replaced by a detonator, and route
    3 still produces its answer.  A future edit that lets the shipped model into
    the remap machinery therefore fails here instead of changing bits quietly.
    (The bit-level comparison against the v5.36.0 tree is a two-tree
    measurement and lives in the build note.)"""
    N, dx = 192, 25e-6
    E = _field(N, dx, w=1.2e-3)

    def _boom(*a, **k):
        raise AssertionError('the shipped tangent_facet path reached the '
                             'remap machinery')

    saved = (LR._tangent_facet_remap_screen, LR._tangent_facet_remap_apply,
             LR._tf_remap_quadratic_eikonal, LR._tf_remap_phi)
    LR._tangent_facet_remap_screen = _boom
    LR._tangent_facet_remap_apply = _boom
    LR._tf_remap_quadratic_eikonal = _boom
    LR._tf_remap_phi = _boom
    try:
        for presc in (_singlet(), _biconvex(), _plate()):
            out = _tf(E, presc, dx)
            assert np.all(np.isfinite(out))
    finally:
        (LR._tangent_facet_remap_screen, LR._tangent_facet_remap_apply,
         LR._tf_remap_quadratic_eikonal, LR._tf_remap_phi) = saved


def test_a_plane_plate_is_byte_identical_at_every_tilt():
    """``sag == 0`` makes the walk identically zero AND makes one reduction skip
    the whole block, so a plate is not approximately unchanged -- it is the same
    bits as the thin screen.  Exact claim, no tolerance."""
    N, dx = 192, 25e-6
    presc = _plate()
    for tilt in (0.0, 0.010, 0.020, 0.055, 0.100, 0.200):
        E = _field(N, dx, w=1.2e-3, tilt=tilt)
        assert np.array_equal(_thin(E, presc, dx).view(np.uint8),
                              _rm(E, presc, dx).view(np.uint8)), tilt


# ---------------------------------------------------------------------------
# 2.  THE IDENTITY -- exact, and checked without an oracle
# ---------------------------------------------------------------------------
def test_the_remap_screen_is_exact_for_a_plane_facet():
    """NO ORACLE, NO TOLERANCE TO CHOOSE.

    For a tilted PLANE facet under a plane wave both eikonals are closed form,
    so the (OPD, walk, exit momentum) triple a wave model must apply is known
    exactly.  (R4)'s hit-point fixed point is EXACT for a plane (its curvature
    term vanishes) and (R5)'s normal correction vanishes too, so this is an
    identity and not an expansion.

    BAR.  Measured 2026-08-16 over these 27 cells: worst 5.95e-14 relative,
    median ~2e-16, growing exactly where the ``b - a`` cancellation says it
    should (slope 0.24).  The bar is 1e-11 -- 2.2 decades above the measurement
    (which bounds the cross-build spread of a float64 cancellation) and about 9
    decades below the smallest defect that could hide here: dropping the
    hit-point fixed point alone moves a slope-0.24 / 150 mrad cell by ~4 % (the
    ``slope * tilt`` term it exists to carry)."""
    n, s0 = 129, 2.3e-4
    xs = np.linspace(-1.5e-3, 1.5e-3, n)
    X, _Y = np.meshgrid(xs, xs)
    sl = (slice(2, -2), slice(2, -2))
    zero = np.zeros_like(X)
    worst = 0.0
    for slope in (0.05, 0.12, 0.24):
        for (n1, n2) in ((1.0, 1.8047), (1.8047, 1.0), (1.5917, 1.8047)):
            for tilt in (0.0, 0.055, 0.15):
                p = n1 * tilt
                got, gwx, gwy, gox, goy, ok = LR._tangent_facet_remap_screen(
                    s0 + slope * X, np.full_like(X, slope), zero,
                    zero, zero, zero, zero,
                    np.full_like(X, p), zero, n1, n2, np)
                assert bool(np.all(ok))
                # the exact plane-facet algebra
                pz1 = np.sqrt(n1 ** 2 - p ** 2)
                q = p / pz1
                sh = (s0 + slope * X) / (1.0 - slope * q)
                inv = 1.0 / np.sqrt(1.0 + slope ** 2)
                a_ = (-slope * p + pz1) * inv
                b_ = np.sqrt(n2 ** 2 - n1 ** 2 + a_ ** 2)
                dz = (b_ - a_) * inv
                po = p - dz * slope
                pz2 = pz1 + dz
                w_want = sh * (q - po / pz2)
                x1 = X + w_want
                c = p * X + sh * (n1 ** 2 / pz1 - n2 ** 2 / pz2) - po * x1
                opd_want = p * X - (po * x1 + c)
                worst = max(
                    worst,
                    float((np.abs(got[sl] - opd_want[sl])
                           / np.abs(opd_want[sl])).max()),
                    float((np.abs(gwx[sl] - w_want[sl])
                           / np.maximum(np.abs(w_want[sl]), 1e-30)).max()),
                    float(np.abs(gox[sl] - po).max() / max(abs(po), 1e-30)))
                # a facet tilted only in x walks only in x
                assert np.array_equal(gwy, np.zeros_like(gwy))
                assert np.array_equal(goy, np.zeros_like(goy))
    assert worst < 1e-11, worst


def test_the_sphere_oracle_matches_the_librarys_own_sag():
    """Guard on the oracle itself -- if this file's closed-form sphere were not
    the library's surface, every comparison below would be scoring the wrong
    geometry."""
    R = 19.6e-3
    r = np.linspace(0.0, 3.0e-3, 51)
    sag, gx, _gy, _a, _b, _c, _d = _sphere_derivs(r, np.zeros_like(r), R)
    # The two forms are algebraically identical, not bit-identical: this file's
    # ``R - sqrt(R^2 - r^2)`` is a cancelling subtraction of two numbers of size
    # R, so it rounds at ``eps * R`` in ABSOLUTE terms, while the library's
    # ``h^2/R / (1 + sqrt(...))`` is the rearrangement that does not cancel.  The
    # bar is therefore absolute and derived: measured 2026-08-16, worst absolute
    # deviation 6.8e-19 m against ``eps * R`` = 4.4e-18 m, so ``16 eps R`` sits
    # ~2 decades above the measurement and ~13 decades below the ~1e-5 m sag
    # difference a wrong conic or a sign error would produce.  (Stated as a
    # relative number instead it would read 7.4e-12 -- and the worst point is
    # r = 0.06 mm where the sag is 9e-8 m, which is exactly why relative is the
    # wrong frame here.)
    assert np.allclose(sag, LR._surface_sag_general(r ** 2, R, 0.0, None),
                       rtol=0.0, atol=16.0 * np.finfo(np.float64).eps * R)
    # and the analytic first derivative against a central difference
    h = 1e-9
    fd = (_sphere_derivs(r + h, np.zeros_like(r), R)[0]
          - _sphere_derivs(r - h, np.zeros_like(r), R)[0]) / (2 * h)
    assert np.max(np.abs(gx - fd)) < 1e-6


def test_against_exact_sphere_rays_the_remap_beats_the_referenced_screen():
    """THE MODEL CLAIM, against a per-ray Newton intersection of a true sphere.

    The remap's residual is the (R4) hit-point truncation and nothing else, and
    it is a SECOND-order truncation in the walk against route 3's referencing
    series.  Fed with ANALYTIC derivatives so the comparison isolates the model
    from the grid.

    Two-sided and build-free: (a) the remap's own residual must fall at least
    16x when the pupil halves -- a >= 4th-order rate, which is a DECISION about
    the truncation order rather than a reading of either number; (b) it must sit
    at least 100x below the walk term route 3 can only reference away.  Measured
    2026-08-16 on R = 12.6 mm N-SF57 at 100 mrad, 3 mm semi-pupil: remap OPD
    residual 1.6e-13 m against a 2.1e-08 m walk term, and the residual falls
    ~64x per halving."""
    R, n1, n2 = 12.6e-3, 1.0, 1.8467
    prev = None
    for rad in (3.0e-3, 1.5e-3, 0.75e-3):
        a = np.linspace(-rad, rad, 41)
        X, Y = np.meshgrid(a, a)   # axis 0 is y, axis 1 is x
        m = (X ** 2 + Y ** 2) <= rad ** 2
        px = np.full_like(X, n1 * 0.1)
        py = np.zeros_like(X)
        sag, gx, gy, hxx, hxy, hyx, hyy = _sphere_derivs(X, Y, R)
        got, wx, wy, pox, poy, ok = LR._tangent_facet_remap_screen(
            sag, gx, gy, hxx, hxy, hyx, hyy, px, py, n1, n2, np)
        assert bool(np.all(ok))
        w_opd, w_wx, w_wy, w_ox, w_oy, _s = _exact_sphere_facet(
            X, Y, R, px, py, n1, n2)
        res = float(np.max(np.abs(got - w_opd)[m]))
        walk_term = float(np.max(np.abs(w_ox * w_wx + w_oy * w_wy)[m]))
        # (b) the residual is far below the term a vertex-plane screen can only
        # reference away -- i.e. the remap has actually removed that series.
        assert res * 100.0 < walk_term, (rad, res, walk_term)
        assert float(np.max(np.abs(wx - w_wx)[m])) * 100.0 < walk_term
        assert float(np.max(np.abs(wy - w_wy)[m])) * 100.0 < walk_term
        if prev is not None:
            # (a) the convergence RATE identifies the truncation, not its size
            assert prev > 16.0 * res, (rad, prev, res)
        prev = res


def test_the_hit_point_fixed_point_and_the_crossing_normal_are_both_load_bearing():
    """A PAIR, like route 3's (T2).  Half of it is a defect, not a saving.

    Sampling the facet at the vertex plane, or taking its normal at the pixel
    instead of at the crossing, each costs orders of magnitude.  Measured
    2026-08-16 on design 121 group 5 at a 3 mm pupil (bundle arm, waves rms):
    hit-at-vertex 1.14e-01, linear fixed point only 5.67e-04, normal at the
    pixel 1.66e-04, and the shipped pair 2.56e-08.  Here the same statement is
    made locally against the sphere oracle, as an ORDERING with a 10x margin."""
    R, n1, n2, rad = 12.6e-3, 1.0, 1.8467, 3.0e-3
    a = np.linspace(-rad, rad, 41)
    X, Y = np.meshgrid(a, a)   # axis 0 is y, axis 1 is x
    m = (X ** 2 + Y ** 2) <= rad ** 2
    px = np.full_like(X, n1 * 0.1)
    py = np.zeros_like(X)
    sag, gx, gy, hxx, hxy, hyx, hyy = _sphere_derivs(X, Y, R)
    zero = np.zeros_like(X)
    w_opd = _exact_sphere_facet(X, Y, R, px, py, n1, n2)[0]

    def _res(*args):
        return float(np.max(np.abs(LR._tangent_facet_remap_screen(
            *args, px, py, n1, n2, np)[0] - w_opd)[m]))

    full = _res(sag, gx, gy, hxx, hxy, hyx, hyy)
    no_hessian = _res(sag, gx, gy, zero, zero, zero, zero)
    assert full * 10.0 < no_hessian, (full, no_hessian)


def _r3_gap(sag, gx, gy, hxx, hxy, hyx, hyy, px, py, n1, n2, h, core):
    """``|A^T p_out - (p - grad OPD)|`` relative to ``|p|`` -- the two sides of
    (R3), computed INDEPENDENTLY: the left from the closed-form Snell momentum
    and the walk's Jacobian, the right by differentiating the imprinted OPD on
    the grid, exactly as the field experiences it."""
    opd, wx, wy, pox, poy, _ok = LR._tangent_facet_remap_screen(
        sag, gx, gy, hxx, hxy, hyx, hyy, px, py, n1, n2, np)
    wx_y, wx_x = np.gradient(wx, h, h)      # axis 0 is y, axis 1 is x
    wy_y, wy_x = np.gradient(wy, h, h)
    o_y, o_x = np.gradient(opd, h, h)
    lhs_x = (1.0 + wx_x) * pox + wy_x * poy
    lhs_y = wx_y * pox + (1.0 + wy_y) * poy
    return max(float(np.max(np.abs(lhs_x - (px - o_x))[core])),
               float(np.max(np.abs(lhs_y - (py - o_y))[core]))) \
        / float(np.max(np.abs(px)))


def test_the_composite_kick_is_the_exact_refracted_momentum_for_a_plane_facet():
    """(R3) -- THE PROPERTY NO SCREEN COULD HAVE, as an IDENTITY.

    BUILD_TANGENT_FACET S0.1 measured the obstruction: a screen's kick is the
    gradient of its own value, so the exact facet kick is unreachable and the
    tangent-facet RAY arm is not a Lagrangian model.  A remap escapes it,
    because the kick is the gradient of the COMPOSITE: with ``A = I + dW/dx``,
    ``grad(S_in - OPD) = A^T p_out``.

    On a PLANE facet -- where (R4) and (R5) are exact, so the model has no
    truncation left -- that is an identity, and it is asserted as one.  Both
    sides are linear in x there, so ``np.gradient``'s central difference is
    exact too and the residual is pure float64 rounding: measured 2026-08-16 at
    1.5e-15 to 4.6e-14 relative across slopes 0.05/0.24 and n = 41/81/161.  Bar
    1e-11, ~2.3 decades above and ~9 decades below the 7.2e-02 gap the same
    measurement finds for the screen-vs-exact-kick pair (below)."""
    n1, n2, rad = 1.0, 1.8467, 3.0e-3
    for n in (41, 81, 161):
        a = np.linspace(-rad, rad, n)
        h = float(a[1] - a[0])
        X, _Y = np.meshgrid(a, a)          # axis 0 is y, axis 1 is x
        core = (slice(3, -3), slice(3, -3))
        z = np.zeros_like(X)
        for slope in (0.05, 0.24):
            gap = _r3_gap(2.3e-4 + slope * X, np.full_like(X, slope), z,
                          z, z, z, z, np.full_like(X, n1 * 0.1), z,
                          n1, n2, h, core)
            assert gap < 1e-11, (n, slope, gap)


def test_on_a_curved_facet_the_remap_is_far_more_lagrangian_than_the_screen():
    """The same statement where it is not free.

    On a CURVED facet the remap's (R3) gap is no longer zero -- it is exactly
    the (R4)/(R5) truncation, and it converges to that floor rather than to
    zero, which is the honest shape of the claim.  What matters is the
    comparison: route 3's screen kick (the gradient of its own OPD) against the
    EXACT facet kick its ray arm uses is the inconsistency
    BUILD_TANGENT_FACET S0.1 priced at 25x, and it is orders larger.

    Measured 2026-08-16, R = 12.6 mm N-SF57 at 100 mrad over a 3 mm semi-pupil,
    relative to |p|: remap 1.77e-04 / 1.33e-04 / 1.29e-04 at h = 75 / 37.5 /
    18.75 um (converging to its truncation floor), route 3 6.47e-02 / 7.20e-02 /
    7.59e-02 -- separations of 365x / 540x / 591x.  The bar is a 50x ORDERING,
    a decision about which model is Lagrangian rather than a reading of either
    number, and it sits a decade inside the smallest measured separation."""
    R, n1, n2, rad = 12.6e-3, 1.0, 1.8467, 3.0e-3
    for n in (81, 161):
        a = np.linspace(-rad, rad, n)
        h = float(a[1] - a[0])
        X, Y = np.meshgrid(a, a)
        core = (slice(3, -3), slice(3, -3))
        px = np.full_like(X, n1 * 0.1)
        py = np.zeros_like(X)
        sag, gx, gy, hxx, hxy, hyx, hyy = _sphere_derivs(X, Y, R)
        remap = _r3_gap(sag, gx, gy, hxx, hxy, hyx, hyy, px, py, n1, n2, h,
                        core)
        # route 3's own screen-vs-exact-kick gap on the identical fixture
        opd_t, _ok = LR._tangent_facet_screen(sag, gx, gy, px, py, n1, n2,
                                              h, h, np)
        _t_y, t_x = np.gradient(opd_t, h, h)
        inv = 1.0 / np.sqrt(1.0 + gx * gx + gy * gy)
        pz1 = np.sqrt(n1 * n1 - px * px - py * py)
        a_ = (-gx * px - gy * py + pz1) * inv
        dz = (np.sqrt(n2 * n2 - n1 * n1 + a_ * a_) - a_) * inv
        screen = float(np.max(np.abs((px - dz * gx) - (px - t_x))[core])) \
            / float(np.max(np.abs(px)))
        assert remap * 50.0 < screen, (n, remap, screen)


# ---------------------------------------------------------------------------
# 3.  THE AMPLITUDE JACOBIAN -- derived, and checked in closed form
# ---------------------------------------------------------------------------
def test_a_linear_walk_is_remapped_exactly_with_the_derived_jacobian():
    """THE JACOBIAN, AGAINST A MAP WHOSE ANSWER IS KNOWN.

    For ``W = c x`` the map is the uniform dilation ``u = (1 + c) x``, so
    ``det A = (1 + c)^2`` EXACTLY and energy conservation forces
    ``E_out(u) = E_in(u/(1+c)) / (1 + c)``.  That closed form is the oracle, and
    it fixes the Jacobian's exponent, its sign and its evaluation point at once:
    a ``sqrt`` on the wrong side, or the determinant sampled at the target
    instead of the source, all fail it.

    BARS, AND WHY THIS ONE SEPARATES THE TWO ERROR SOURCES.  The only error
    left after the Jacobian is the spline through a smooth Gaussian, and that
    error depends on the ORDER while a wrong Jacobian exponent does not -- a
    missing or inverted ``sqrt`` shows as ``O(|c|)`` ~ 0.2 at every order.  So
    the test asserts a ladder as well as a level.  Measured 2026-08-16 on a
    ``w0 = 24 dx`` Gaussian, relative to the peak: order 1 / 3 / 5 read
    8.3e-04 / 1.8e-07 / 7.6e-11 (and 2.9e-03 / 2.3e-06 / 3.8e-09 at
    ``w0 = 12 dx``, i.e. the fourth-order rate a cubic spline must have).  Bars:
    order 3 below 1e-05 -- ~1.8 decades above the measurement and ~4.3 decades
    below the 0.2 an exponent error would produce -- and each order at least
    100x better than the one below it, which is a decision about what the
    residual IS rather than a reading of its size."""
    N, dx = 256, 4.0e-6
    a = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(a, a)               # axis 0 is y, axis 1 is x
    w0 = 24.0 * dx
    E = np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(np.complex128)
    m = (X ** 2 + Y ** 2) <= (0.25 * N * dx) ** 2
    for c in (-0.20, 0.15, 0.35):
        wx = c * X
        wy = c * Y
        zero = np.zeros_like(X)
        rel = {}
        for order in (1, 3, 5):
            got, _px, _py = LR._tangent_facet_remap_apply(
                E, wx, wy, zero, zero, dx, dx, 2 * np.pi / LAM, order, np, 0)
            want = np.exp(-((X / (1.0 + c)) ** 2 + (Y / (1.0 + c)) ** 2)
                          / w0 ** 2) / (1.0 + c)
            rel[order] = (float(np.max(np.abs(got - want)[m]))
                          / float(np.max(np.abs(want))))
            if order == 3:
                # ... and the energy statement the Jacobian was DERIVED from.
                # Measured 5.0e-08; bar 1e-05, and an exponent error would put
                # this at |(1+c)^2 - 1| >= 0.28.
                assert abs(float(np.sum(np.abs(got) ** 2))
                           / float(np.sum(np.abs(E) ** 2)) - 1.0) < 1e-5, c
        assert rel[3] < 1e-5, (c, rel)
        assert rel[3] * 100.0 < rel[1], (c, rel)
        assert rel[5] * 100.0 < rel[3], (c, rel)


def test_the_element_conserves_energy_as_well_as_the_screen_it_replaces():
    """The remap MOVES energy, so it has to move all of it.  Two-sided: the
    remapped power must track the pure-screen (unitary by construction) power,
    and the deficit must be far below the aperture's own.

    Measured 2026-08-16 on the R = 12.6 mm biconvex at Nyquist-resolved 4 um
    sampling: thin and tangent_facet both 0.999897294 of the input, remap
    0.999897039 -- a 2.6e-07 relative deficit against a 1e-04 bar (2.6 decades
    of headroom, and 3 decades below the 1.0e-4 the aperture itself removes)."""
    N, dx = 1536, 4.0e-6
    E = _field(N, dx, w=0.7e-3)
    presc = _biconvex()
    p_in = float(np.sum(np.abs(E) ** 2))
    p_screen = float(np.sum(np.abs(_tf(E, presc, dx)) ** 2)) / p_in
    p_remap = float(np.sum(np.abs(_rm(E, presc, dx)) ** 2)) / p_in
    assert abs(p_remap - p_screen) < 1.0e-4, (p_screen, p_remap)


def test_the_demodulation_is_what_makes_the_resampling_faithful():
    """A lens interior runs at a few pixels per fringe; a spline through that
    loses amplitude.  The demodulation by the analytic quadratic eikonal is a
    SIMILARITY transform (multiply, resample, divide), so it cannot change the
    physics -- only the interpolation error -- and it must strictly reduce it.

    Two-sided: with the demodulation the power deficit must be at least 10x
    smaller.  Measured 2026-08-16 on the biconvex at 4 um: 3.4e-03 of the input
    lost without it, 2.6e-07 with it -- a 1.3e4 separation, so a 10x bar sits 3
    decades inside the measurement."""
    N, dx = 1536, 4.0e-6
    E = _field(N, dx, w=0.7e-3)
    presc = _biconvex()
    p_in = float(np.sum(np.abs(E) ** 2))
    p_ref = float(np.sum(np.abs(_tf(E, presc, dx)) ** 2)) / p_in
    on = float(np.sum(np.abs(_rm(E, presc, dx)) ** 2)) / p_in
    saved = LR._tf_remap_quadratic_eikonal
    LR._tf_remap_quadratic_eikonal = lambda *a, **k: None
    try:
        off = float(np.sum(np.abs(_rm(E, presc, dx)) ** 2)) / p_in
    finally:
        LR._tf_remap_quadratic_eikonal = saved
    assert abs(on - p_ref) * 10.0 < abs(off - p_ref), (p_ref, on, off)


def test_the_eikonal_fit_is_exact_for_a_linear_momentum_field():
    """The demodulating quadratic is fitted to the field's own momentum with the
    cross term SHARED between components -- the curl-free constraint that makes
    it a phase rather than a pair of ramps.  For a linear momentum field it is
    therefore an identity, asserted as an equality against its own definition
    rather than as a tolerance on a residual."""
    N, dx = 96, 5e-6
    x_ax = (np.arange(N) - N // 2) * dx
    y_ax = (np.arange(N) - N // 2) * dx
    X = x_ax[None, :] + np.zeros((N, 1))
    Y = y_ax[:, None] + np.zeros((1, N))
    c0, c1, c2, d0, d2 = 0.031, 7.7, -3.3, -0.019, 5.1
    px = c0 + c1 * X + c2 * Y
    py = d0 + c2 * X + d2 * Y
    w = np.exp(-(X ** 2 + Y ** 2) / (0.3 * N * dx) ** 2)
    got = LR._tf_remap_quadratic_eikonal(px, py, w, x_ax, y_ax, np)
    assert got is not None
    assert np.allclose(got, (c0, c1, c2, d0, d2), rtol=1e-9, atol=1e-12), got
    # and the gradient of the fitted Phi reproduces the momentum it was fitted
    # to, which is the property the demodulation actually uses
    phi = LR._tf_remap_phi(got, X, Y)
    gy, gx = np.gradient(phi, dx, dx)
    core = (slice(2, -2), slice(2, -2))
    assert np.max(np.abs(gx - px)[core]) < 1e-9
    assert np.max(np.abs(gy - py)[core]) < 1e-9


def test_a_degenerate_weight_drops_the_demodulation_instead_of_failing():
    """The fit is an accelerator, not a dependency: an empty or degenerate
    weight returns None and the caller resamples undemodulated.  Refuse-nothing
    here is correct -- losing the demodulation costs interpolation accuracy and
    changes no physics."""
    N, dx = 32, 5e-6
    x_ax = (np.arange(N) - N // 2) * dx
    zero = np.zeros((N, N))
    assert LR._tf_remap_quadratic_eikonal(zero, zero, zero, x_ax, x_ax,
                                          np) is None
    nan_w = np.full((N, N), np.nan)
    assert LR._tf_remap_quadratic_eikonal(zero, zero, nan_w, x_ax, x_ax,
                                          np) is None


# ---------------------------------------------------------------------------
# 4.  CAUSTIC SAFETY -- the fold guard, proved on an engineered fold
# ---------------------------------------------------------------------------
_FOLD_PERIOD = 40e-6


def _corrugated(amp, X):
    """A prescription whose transverse walk FOLDS.

    Engineered, not hoped for: a real lens interior is caustic-free by design
    contract, which is the whole reason this model is allowed to be a ray map.
    With ``sag = A cos(k x)`` the dominant walk term is
    ``-s p_out/pz2 ~ (A^2 k dz / 2 pz2) sin(2 k x)``, so ``dW/dx`` reaches -1 at
    a slope amplitude ``A k ~ sqrt(pz2/dz)`` = 1.48 for N-SF57 -- of order one,
    i.e. a corrugation far steeper than any lens.  The departure is injected as
    a ``form_error`` map because that is the hook this model's whole-grid sag
    pipeline actually reads."""
    return {'surfaces': [
        {'radius': np.inf, 'conic': 0.0,
         'form_error': amp * np.cos(2 * np.pi * X / _FOLD_PERIOD),
         'glass_before': 'AIR', 'glass_after': 'N-SF57'},
        {'radius': np.inf, 'conic': 0.0,
         'glass_before': 'N-SF57', 'glass_after': 'AIR'}],
        'thicknesses': [2.0e-4]}


def test_a_folding_walk_is_refused_and_a_gentle_one_is_not():
    """FAIL-BEFORE, two-sided, with the state ENGINEERED through the public API
    and the ladder scanned rather than a magic amplitude assumed.

    Below the fold the model must RUN (otherwise the guard is just a
    prohibition); above it the model must REFUSE with the fold named.  Measured
    2026-08-16, min det(I + dW/dx) against the corrugation slope A*k: 0.998 at
    0.063, 0.639 at 0.942, 0.014 at 1.571, then -0.186 at 1.728 and -0.886 at
    2.199 -- so the refusal boundary is a sign change of a smooth quantity, not
    a threshold on a noisy one."""
    N, dx = 256, 2.0e-6
    a = (np.arange(N) - N // 2) * dx
    X, _Y = np.meshgrid(a, a)   # axis 0 is y, axis 1 is x
    E = _field(N, dx, w=1.0e-4)
    # (a) gentle: it runs, and the answer is finite
    for amp_um in (0.4, 1.6, 6.0):
        out = _rm(E, _corrugated(amp_um * 1e-6, X), dx)
        assert np.all(np.isfinite(out)), amp_um
    # (b) steep: the ladder must reach a refusal that NAMES the fold
    fold_amp = None
    for amp_um in (8.0, 10.0, 11.0, 12.0, 14.0, 18.0, 24.0):
        try:
            _rm(E, _corrugated(amp_um * 1e-6, X), dx)
        except ValueError as exc:
            if 'folds' in str(exc):
                fold_amp = amp_um * 1e-6
                assert 'REFUSES' in str(exc)
                assert 'single-valued' in str(exc)
                break
    assert fold_amp is not None, 'ladder exhausted: no fold engineered'


def test_the_unguarded_arm_is_wrong_where_the_guard_refuses():
    """THE REFUSAL EARNS ITS KEEP.

    A guard that declines a case the un-guarded code would have handled fine is
    a cost, not a safety property.  With both bars removed the same folding
    prescription returns a NON-FINITE field (``1/sqrt(det)`` at ``det < 0``),
    i.e. the field would be silently poisoned -- and independently, the map is
    genuinely multi-valued there: ``x + W(x) = u`` has 4 roots at a sampled
    ``u``, so no pull-back of any interpolation order could be right.  Both are
    exact structural claims, not tolerances."""
    N, dx = 256, 2.0e-6
    a = (np.arange(N) - N // 2) * dx
    X, _Y = np.meshgrid(a, a)   # axis 0 is y, axis 1 is x
    E = _field(N, dx, w=1.0e-4)
    presc = _corrugated(12.0e-6, X)
    with pytest.raises(ValueError, match='folds'):
        _rm(E, presc, dx)
    saved = (LR._TF_REMAP_MIN_DET, LR._TF_REMAP_PULLBACK_TOL_PX)
    LR._TF_REMAP_MIN_DET = -1e30
    LR._TF_REMAP_PULLBACK_TOL_PX = 1.0e30
    try:
        bad = _rm(E, presc, dx)
    finally:
        (LR._TF_REMAP_MIN_DET, LR._TF_REMAP_PULLBACK_TOL_PX) = saved
    assert not np.all(np.isfinite(bad)), 'the un-guarded arm was harmless'
    # the map really is multi-valued there
    n1, n2, amp = 1.0, 1.847, 12.0e-6
    xs = np.linspace(-_FOLD_PERIOD, _FOLD_PERIOD, 200001)
    s = amp * np.cos(2 * np.pi * xs / _FOLD_PERIOD)
    g = -amp * (2 * np.pi / _FOLD_PERIOD) * np.sin(2 * np.pi * xs / _FOLD_PERIOD)
    inv = 1.0 / np.sqrt(1.0 + g * g)
    a_ = n1 * inv
    dz = (np.sqrt(n2 ** 2 - n1 ** 2 + a_ ** 2) - a_) * inv
    u = xs + s * (dz * g) / (n1 + dz)
    target = float(u[len(u) // 2 + 700])
    assert int(np.sum(np.diff(np.sign(u - target)) != 0)) > 1


def test_a_near_singular_but_unfolded_map_is_refused_by_the_inversion():
    """The two bars cover different failures and BOTH refuse.

    ``det > 0`` says the map is single-valued; the pull-back's convergence says
    it is invertible ON THIS GRID.  The fixed point contracts at exactly the
    rate the determinant approaches zero, so a map that is unfolded but nearly
    singular is declined by the iteration rather than truncated into an answer.
    Measured 2026-08-16: at A*k = 1.571 the determinant is +0.014 (unfolded) and
    the inversion is refused."""
    N, dx = 256, 2.0e-6
    a = (np.arange(N) - N // 2) * dx
    X, _Y = np.meshgrid(a, a)   # axis 0 is y, axis 1 is x
    E = _field(N, dx, w=1.0e-4)
    with pytest.raises(ValueError, match='did not converge'):
        _rm(E, _corrugated(10.0e-6, X), dx)


# ---------------------------------------------------------------------------
# 5.  INTERPOLATION ORDER
# ---------------------------------------------------------------------------
def test_raising_the_interpolation_order_converges():
    """The resampling order is a knob with a documented cost, so its effect has
    to be measured rather than assumed.  Bar is an ORDERING: successive orders
    must agree with each other better than the lowest pair does, which is a
    decision about convergence and not a reading.  Measured 2026-08-16 on the
    biconvex at 4 um, N = 1536, relative to the peak: |o3 - o1| 3.69e-04 and
    |o5 - o3| 1.37e-04 -- only 2.7x, because 4 um is close to the exit
    wavefront's Nyquist on that fixture and the resampling converges with the
    SAMPLING as well as with the order.  The bar is the ordering ALONE for
    exactly that reason: a ratio bar here would be pinning the fixture's
    sampling rather than the resampler."""
    N, dx = 1536, 4.0e-6
    E = _field(N, dx, w=0.7e-3)
    presc = _biconvex()
    o = {k: _rm(E, presc, dx, remap_order=k) for k in (1, 3, 5)}
    peak = float(np.max(np.abs(o[3])))
    d31 = float(np.max(np.abs(o[3] - o[1]))) / peak
    d53 = float(np.max(np.abs(o[5] - o[3]))) / peak
    assert d53 < d31, (d31, d53)
    for k in (1, 3, 5):
        assert np.all(np.isfinite(o[k])), k


# ---------------------------------------------------------------------------
# 6.  REFUSALS -- the supported envelope
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('kw,frag', [
    ({'slant_correction': True}, 'slant_correction'),
    ({'surface_frame': True}, 'surface_frame'),
    ({'use_gpu': True}, 'use_gpu'),
    ({'wave_propagator': 'fresnel'}, 'angular-spectrum'),
    ({'conjugate': 0.5}, 'conjugate'),
    ({'displaced_mode': 'split'}, 'displaced_mode'),
    ({'displaced_obliquity': 'pointwise'}, 'displaced_obliquity'),
])
def test_the_unsupported_envelope_refuses_rather_than_running(kw, frag):
    E = _field(96, 25e-6, w=0.6e-3)
    with pytest.raises((ValueError, NotImplementedError), match=frag):
        _rm(E, _singlet(), 25e-6, **kw)


def test_screen_obliquity_is_refused_because_it_would_double_count():
    """The remap rung supersedes equations (4) and (7) exactly as route 3 does
    -- it imprints the tangent-facet OPD at the field's own ray angle, so the
    angular DIFFERENCE those add is already inside it.  ``carrier=`` is still
    honoured (it seeds the momentum accumulator), so this refuses the
    correction and not the carrier."""
    E = _field(96, 25e-6, w=0.6e-3)
    car = la.TiltedCarrier(float('inf'), 0.05, 0.0)
    with pytest.raises(ValueError, match='screen_obliquity'):
        _rm(E, _singlet(), 25e-6, carrier=car, screen_obliquity=True)
    out = _rm(E, _singlet(), 25e-6, carrier=car)
    assert np.all(np.isfinite(out))


@pytest.mark.parametrize('order', [0, 2, 4, 'cubic', None])
def test_an_unmeasured_interpolation_order_is_refused(order):
    E = _field(96, 25e-6, w=0.6e-3)
    with pytest.raises(ValueError, match='remap_order'):
        _rm(E, _singlet(), 25e-6, remap_order=order)


def test_remap_order_is_refused_on_every_model_that_resamples_nothing():
    E = _field(96, 25e-6, w=0.6e-3)
    for model in ('thin', 'tangent_facet'):
        with pytest.raises(ValueError, match='remap_order'):
            apply_real_lens(E, prescription=_singlet(), wavelength=LAM,
                            dx=25e-6, surface_model=model, remap_order=5)


def test_the_model_name_is_in_the_valid_set_and_a_typo_is_refused():
    assert 'tangent_facet_remap' in LR._VALID_SURFACE_MODELS
    assert set(LR._TANGENT_FACET_MODELS) <= set(LR._VALID_SURFACE_MODELS)
    E = _field(96, 25e-6, w=0.6e-3)
    with pytest.raises(ValueError, match='unknown surface_model'):
        apply_real_lens(E, prescription=_singlet(), wavelength=LAM, dx=25e-6,
                        surface_model='tangent_facet_remaps')


def test_sag_chunk_rows_is_inert_for_this_model():
    """Whole-grid only, for route 3's reason and one more: the model
    differentiates a gradient twice AND then resamples the whole field, so an
    exact band would need a halo on the sag, on the accumulator and on the walk.
    Priced and refused rather than approximated -- and pinned INERT with
    ``np.array_equal`` so a future change that lets the model into the band loop
    has to arrive with its own byte-identity argument instead of taking effect
    silently."""
    N, dx = 192, 25e-6
    E = _field(N, dx, w=1.2e-3)
    presc = _biconvex()
    base = _rm(E, presc, dx)
    for cr in (0, 1, 7, 64, 4096):
        got = _rm(E, presc, dx, sag_chunk_rows=cr)
        assert np.array_equal(base.view(np.uint8), got.view(np.uint8)), cr


# ---------------------------------------------------------------------------
# 7.  THE MODEL IS REACHED, AND IT MOVES THE ANSWER
# ---------------------------------------------------------------------------
def test_the_remap_actually_moves_the_field_relative_to_the_screen():
    """A null test in reverse: if the remap block were skipped the answer would
    equal route 3's, so this pins that the walk is APPLIED.  The separation is
    a structural inequality, and its size tracks the walk rather than a
    tolerance."""
    N, dx = 512, 4.0e-6
    E = _field(N, dx, w=0.5e-3)
    for presc in (_singlet(), _biconvex()):
        a = _tf(E, presc, dx)
        b = _rm(E, presc, dx)
        assert not np.array_equal(a.view(np.uint8), b.view(np.uint8))
        assert np.max(np.abs(a - b)) > 1e-6 * np.max(np.abs(a))


def test_a_carrier_seeds_the_accumulator_and_a_zero_tilt_carrier_is_the_null():
    """The model reads the FIELD's momentum, so a zero-angle carrier must be
    indistinguishable from no carrier at all -- a structural zero, asserted as
    bits."""
    N, dx = 256, 10e-6
    E = _field(N, dx, w=0.8e-3)
    presc = _singlet()
    zero_car = la.TiltedCarrier(float('inf'), 0.0, 0.0)
    assert np.array_equal(_rm(E, presc, dx).view(np.uint8),
                          _rm(E, presc, dx, carrier=zero_car).view(np.uint8))
    tilted = _rm(E, presc, dx, carrier=la.TiltedCarrier(float('inf'), 0.05,
                                                        0.0))
    assert not np.array_equal(_rm(E, presc, dx).view(np.uint8),
                              tilted.view(np.uint8))
