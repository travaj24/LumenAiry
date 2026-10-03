"""CURVED-CELL MAP, Phase B, for the PURE staggered 2-D PMM: the Gordon-Hall
TRANSFINITE map (edge curves: line, circular arc, ellipse arc, sinusoid), the
corner (Duffy) quadrature at the singular vertices, and the circle / fillet /
ellipse / sinusoid gates B1-B10 of
``docs/audits/PLAN_PMM2D_CURVED_CELLS_2026_09_26.md`` section 4.2, as built
and measured in ``docs/audits/BUILD_PMM2D_CURVED_B_2026_10_02.md``.

Words.  A transfinite map bends the solver's rectangular ``(u, v)`` wall
grid so that chosen grid EDGES become given curves (four 90-degree arcs make
the middle cell of a 3 x 3 grid a disk); inside each cell it blends the
cell's four edge curves.  A closed smooth curve on a tensor grid has four
SINGULAR VERTICES (``det J = 0`` at a cell corner), where the effective
tensors grow like 1 / distance; the cells that own one are integrated by a
Duffy-collapsed rule.  A FAIL-BEFORE arm is a deliberately broken variant
that must fail the bar, proving the test can see the defect it guards.

Fixture (the planning probes' P3 / P4 fixture): lambda = 1, square period
1.2, height 0.5, air over n = 1.45, eps 4 (n = 2) features, normal incidence
unless stated; the circle has r = 0.36 = 0.3 p.  Row 0 of R / T is the input
E along x ('tm'), row 1 E along y ('te').

EVERY BAR is derived from a measurement made by this build on 2026-10-02
(Windows 11, CPython 3.14.6, numpy 2.4.4, scipy 1.17.1, OMP/OPENBLAS/MKL = 1;
probe JSON under ``validation/probe_pmm2d_curved/build_b/``), stated in the
assertion's comment with its gap on both sides.  Build-to-build spread: the
curved solve's R / T carry a ROUND-OFF floor of ~1e-9 at M = 6 (a 1e-15
random perturbation of the half-space weights moves them 6.8e-10; 1.4e-11 at
M = 8 -- build finding F-B4); no bar below sits within two decades of it.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import functools  # noqa: E402
import json  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lumenairy.elements.pmm import PMM2DStackPure  # noqa: E402
from lumenairy.elements.pmm import _curvemap as CM  # noqa: E402
from lumenairy.elements.pmm import stack2d_pure as SP  # noqa: E402
from lumenairy.elements.pmm import twod_staggered as TS  # noqa: E402

_P = 1.2
_WL = 1.0
_DEPTH = 0.5
_NSUP, _NSUB = 1.0, 1.45
_K0 = 2 * np.pi / _WL
_R = 0.36
_ORD9 = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1),
         (-1, -1)]
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROBE = os.path.join(_HERE, "..", "..", "validation", "probe_pmm2d_curved")


def _circle3():
    cm, _w = CM._circle_map_3x3(_P, _R)
    eps = np.ones((3, 3), complex)
    eps[1, 1] = 4.0
    return cm, eps


def _circle5():
    cm, _w = CM._circle_map_5x5(_P, _R)
    eps = np.ones((5, 5), complex)
    eps[1:4, 1:4] = 4.0
    return cm, eps


def _solve(cmap, eps_cell, M, theta=0.0, phi=0.0, n_orders=3):
    st = PMM2DStackPure(_P, _P, n_superstrate=_NSUP, n_substrate=_NSUB,
                        n_modes=M, n_orders=n_orders, cmap=cmap)
    st.add_layer(_DEPTH, eps_cell=eps_cell)
    st.set_source(_WL, theta=theta, phi=phi)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # fail-before arms trip closure
        o, R, T, J = st.solve()
    return np.asarray(o), np.asarray(R), np.asarray(T), st


def _solve_walls(xw, yw, eps_cell, M):
    """The SHIPPED unmapped solver on explicit walls (one patterned layer
    between the half-spaces: the per-layer stack is conforming)."""
    st = PMM2DStackPure(_P, _P, n_superstrate=_NSUP, n_substrate=_NSUB,
                        n_modes=M, n_orders=3, layer_grids="per-layer")
    st.add_layer(_DEPTH, eps_cell=eps_cell, x_walls=xw, y_walls=yw,
                 max_pencil_dof=20000)
    st.set_source(_WL)
    o, R, T = st.solve(jones=False)
    return np.asarray(o), np.asarray(R), np.asarray(T)


def _idx(o, orders=_ORD9):
    return [int(np.nonzero((o[:, 0] == m) & (o[:, 1] == n))[0][0])
            for m, n in orders]


def _vec(o, R, T):
    i = _idx(o)
    return np.concatenate([R[:, i].ravel(), T[:, i].ravel()])


def _rel(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


class _NoCofactorView:
    """FAIL-BEFORE view of a map for the far projector only (Phase A's
    engineered defect): positions kept, cofactor replaced by the identity."""

    def __init__(self, cmap):
        self._m = cmap

    def geom(self, sx, sy, U, V):
        X, Y, xu, xv, yu, yv = self._m.geom(sx, sy, U, V)
        return (X, Y, np.ones_like(xu), np.zeros_like(xv), np.zeros_like(yu),
                np.ones_like(yv))


def _patch_no_cofactor(monkeypatch):
    orig = SP._far_projector_2d

    def patched(bx, by, ox, oy, a0x=0.0, a0y=0.0, cmap=None):
        if cmap is not None:
            cmap = _NoCofactorView(cmap)
        return orig(bx, by, ox, oy, a0x, a0y, cmap=cmap)
    monkeypatch.setattr(SP, "_far_projector_2d", patched)


def _airy_rows(theta, phi, n2=2.0):
    """Exact reflectance of the uniform n2 film for incident lab E_x / E_y
    (s / p split of the incident transverse field)."""
    k0 = _K0
    ns = (_NSUP, n2, _NSUB)
    st = _NSUP * np.sin(theta)
    kz = [np.sqrt(complex(n * n - st * st)) for n in ns]

    def slab(r12, r23):
        ph = np.exp(2j * kz[1] * k0 * _DEPTH)
        return abs((r12 + r23 * ph) / (1 + r12 * r23 * ph)) ** 2
    rs = slab((kz[0] - kz[1]) / (kz[0] + kz[1]),
              (kz[1] - kz[2]) / (kz[1] + kz[2]))
    e = [n * n for n in ns]
    rp = slab((e[1] * kz[0] - e[0] * kz[1]) / (e[1] * kz[0] + e[0] * kz[1]),
              (e[2] * kz[1] - e[1] * kz[2]) / (e[2] * kz[1] + e[1] * kz[2]))
    out = []
    for et in ((1.0, 0.0), (0.0, 1.0)):
        a = -np.sin(phi) * et[0] + np.cos(phi) * et[1]
        b = (np.cos(phi) * et[0] + np.sin(phi) * et[1]) / np.cos(theta)
        out.append((a * a * rs + b * b * rp) / (a * a + b * b))
    return np.array(out)


def _film_err(cmap, M, theta=0.0, phi=0.0):
    Nx, Ny = cmap.shape
    o, R, T, _st = _solve(cmap, np.full((Nx, Ny), 4.0 + 0j), M, theta, phi)
    Rx = _airy_rows(theta, phi)
    i0 = _idx(o, [(0, 0)])[0]
    R = R.copy()
    T = T.copy()
    R[:, i0] -= Rx
    T[:, i0] -= 1.0 - Rx
    return float(max(np.abs(R).max(), np.abs(T).max()))


# =========================================================================== #
# B1 -- no map = today's bytes; the transfinite identity is the identity map
# =========================================================================== #
def test_b1_no_map_never_reaches_the_phase_b_code(monkeypatch):
    """``cmap=None`` dispatches to the shipped Kronecker code: every function
    Phase B added (the corner rule, its weights, the transfinite map's
    pointwise evaluation) is booby-trapped and an unmapped solve must not
    touch one.  The byte identity itself is a build-doc measurement against
    ``git archive 539ce4a3`` (``b2_compare.json``: 109 / 109 SHA-256 of
    operators, modes, R / T / Jones and absorption identical); this is its
    build-free restatement."""
    def boom(*a, **k):
        raise AssertionError("Phase B code reached on the no-map path")
    for name in ("_stag_duffy_points", "_stag_map_singular_corners",
                 "_stag_map_eff", "_stag_map_geom5", "_stag_quad_weighted",
                 "_stag_map_nodes", "_far_projector_mapped"):
        monkeypatch.setattr(TS, name, boom)
    monkeypatch.setattr(TS, "_StagMapQuad", boom)
    eps = np.ones((3, 3), complex)
    eps[1, 1] = 4.0
    st = PMM2DStackPure(_P, _P, n_superstrate=_NSUP, n_substrate=_NSUB,
                        n_modes=4, n_orders=2)
    st.add_layer(0.2, eps=2.1)
    st.add_layer(0.3, eps_cell=eps)
    st.set_source(_WL, theta=0.2, phi=0.3)
    st.solve(retain_internal=True)
    st.layer_absorption()


def test_b1_transfinite_identity_is_the_identity_map():
    """A ``TransfiniteMap`` with no curve and identity vertices IS the
    identity: same node count as ``IdentityMap`` (2M + 8, polynomial
    weights), no corner-rule cell, and the operators / R / T of Phase A's
    identity arm.  Measured 2026-10-02 (``b2_identity.json``, Phase A's A2
    fixture: 3 x 3 pillar on non-uniform walls): operators vs the kron
    assembly <= 9.9e-15 (M = 5) / 4.8e-14 (M = 7), vs ``IdentityMap`` <=
    5.0e-15 / 3.3e-14; R / T vs kron 6.4e-15 / 4.1e-14.  Bar 1e-11 (Phase A's
    A2 bar; 2.3 decades above).  UPPER GAP: moving ONE interior vertex image
    by 1e-6 p (a straight-edged, bilinear map) must move L by more than 1e-9
    relative (5 decades above round-off, 3 under the 1e-6 it is)."""
    M = 5
    xw = np.array([0.0, 0.25, 0.85, _P])
    yw = np.array([0.0, 0.30, 0.90, _P])
    eps = np.ones((3, 3), complex)
    eps[1, 1] = 4.0
    s0 = TS.Granet2DTransverseE(_P, _P, xw, yw, M, eps, k0=_K0)
    tf = CM.TransfiniteMap(xw, yw)
    s2 = TS.Granet2DTransverseE(_P, _P, xw, yw, M, eps, k0=_K0, cmap=tf)
    assert tf.singular_vertices == []
    assert s2._qrule.n == 2 * M + 8 and s2._qrule.points == {}
    for name in ("Rmat", "Lmat", "Stt", "Schur"):
        assert _rel(getattr(s2, name), getattr(s0, name)) <= 1e-11, name
    V = np.empty((4, 4, 2))
    V[..., 0] = xw[:, None]
    V[..., 1] = yw[None, :]
    V[1, 1, 0] += 1e-6 * _P
    s3 = TS.Granet2DTransverseE(_P, _P, xw, yw, M, eps, k0=_K0,
                                cmap=CM.TransfiniteMap(xw, yw, V))
    assert _rel(s3.Lmat, s2.Lmat) > 1e-9


def test_b1_transfinite_geometry_and_construction():
    """Geometry exactness, C0, refusals and the fingerprint.

    * Areas: the mapped disk (3 x 3 and 5 x 5) equals pi r^2, the ellipse
      pi a b, the fillet pillar side^2 - (4 - pi) r^2, the whole cell p^2 --
      measured <= 5e-16 relative; bar 1e-12 (plan B1).
    * C0: the two cells sharing an edge evaluate the SAME physical curve
      and the same tangent there (one EdgeCurve object serves both; the
      blend's vertex terms cancel to round-off -- measured <= 4.4e-16 over
      the four edges of the disk cell; bar 1e-14).
    * A vertex image moved by 1e-9 p no longer meets its curve's end: the
      constructor RAISES (plan B2 fail-before); the unmoved one is legal.
    * A circle too large for its cell folds the outer cells (det J < 0
      inside): refused.
    * The four singular vertices of the circle are found (corners of the
      disk cell); the sinusoidal stripe has none.
    * The analytic Jacobian matches a central difference to 1e-8
      (measured <= 2e-10 at h = 1e-6).
    * The fingerprint changes with every curve parameter (radius by
      1e-9) and is equal for equal content."""
    from numpy.polynomial.legendre import leggauss
    xg, wg = leggauss(48)

    def area(cm, cells):
        A = 0.0
        for sx, sy in cells:
            J1 = 0.5 * (cm.u_bounds[sx + 1] - cm.u_bounds[sx])
            J2 = 0.5 * (cm.v_bounds[sy + 1] - cm.v_bounds[sy])
            U = 0.5 * (cm.u_bounds[sx] + cm.u_bounds[sx + 1]) + J1 * xg
            V = 0.5 * (cm.v_bounds[sy] + cm.v_bounds[sy + 1]) + J2 * xg
            _X, _Y, xu, xv, yu, yv = cm.geom(sx, sy, U, V)
            A += float(np.sum(np.outer(wg, wg) * (xu * yv - xv * yu))) \
                * J1 * J2
        return A
    c3, _ = CM._circle_map_3x3(_P, _R)
    c5, _ = CM._circle_map_5x5(_P, _R)
    e3, _ = CM._ellipse_map_3x3(_P, (0.40, 0.28))
    f5, _ = CM._fillet_map_5x5(_P, 0.3, 0.12)
    inner = [(i, j) for i in (1, 2, 3) for j in (1, 2, 3)]
    for cm, cells, exact in (
            (c3, [(1, 1)], np.pi * _R ** 2), (c5, inner, np.pi * _R ** 2),
            (e3, [(1, 1)], np.pi * 0.40 * 0.28),
            (f5, inner, 0.6 ** 2 - (4 - np.pi) * 0.12 ** 2)):
        assert abs(area(cm, cells) / exact - 1) <= 1e-12
        allc = [(i, j) for i in range(cm.shape[0]) for j in range(cm.shape[1])]
        assert abs(area(cm, allc) / _P ** 2 - 1) <= 1e-12
    # C0: the disk's bottom arc seen from the disk cell (1, 1) at t = 0 and
    # from the cell below (1, 0) at t = 1
    s = np.linspace(0.05, 0.95, 7)
    U = c3.u_bounds[1] + s * (c3.u_bounds[2] - c3.u_bounds[1])
    a = c3.geom(1, 1, U, np.array([c3.v_bounds[1]]))
    b = c3.geom(1, 0, U, np.array([c3.v_bounds[1]]))
    for k in (0, 1, 2, 4):       # positions and the TANGENT d Phi / du
        assert float(np.max(np.abs(a[k] - b[k]))) <= 1e-14
    # the endpoint check (fail-before: a perturbed vertex image)
    w = c3.u_bounds
    V = np.empty((4, 4, 2))
    V[..., 0] = w[:, None]
    V[..., 1] = w[None, :]
    CM.TransfiniteMap(w, w, V, dict(c3.curved_edges))
    V[1, 1, 0] += 1e-9 * _P
    with pytest.raises(ValueError, match="would not meet"):
        CM.TransfiniteMap(w, w, V, dict(c3.curved_edges))
    with pytest.raises(ValueError, match="det J must be > 0"):
        CM._circle_map_3x3(_P, 0.8)
    assert sorted(c3.singular_vertices) == [(1, 1, 0, 0), (1, 1, 0, 1),
                                            (1, 1, 1, 0), (1, 1, 1, 1)]
    s3, _ = CM._sine_stripe_map_3x3(_P, 0.3, 0.9, 0.12)
    assert s3.singular_vertices == []
    # analytic Jacobian vs a central difference
    h = 1e-6
    for cm in (c3, s3, e3):
        for sx, sy in ((1, 1), (0, 1), (2, 2)):
            u = np.array([0.5 * (cm.u_bounds[sx] + cm.u_bounds[sx + 1])
                          + 0.013])
            v = np.array([0.5 * (cm.v_bounds[sy] + cm.v_bounds[sy + 1])
                          - 0.021])
            g = cm.geom(sx, sy, u, v)
            gp, gm = cm.geom(sx, sy, u + h, v), cm.geom(sx, sy, u - h, v)
            hp, hm = cm.geom(sx, sy, u, v + h), cm.geom(sx, sy, u, v - h)
            fd = ((gp[0] - gm[0]) / (2 * h), (hp[0] - hm[0]) / (2 * h),
                  (gp[1] - gm[1]) / (2 * h), (hp[1] - hm[1]) / (2 * h))
            for an, num in zip(g[2:], fd):
                assert abs(float(an[0, 0] - num[0, 0])) <= 1e-8
    # fingerprint
    assert c3.fingerprint == CM._circle_map_3x3(_P, _R)[0].fingerprint
    assert c3.fingerprint != CM._circle_map_3x3(_P, _R + 1e-9)[0].fingerprint


# =========================================================================== #
# B8 -- the singular vertices: the corner (Duffy) rule
# =========================================================================== #
def test_b8_corner_rule_is_the_tensor_kernel_on_smooth_weights():
    """ONE kernel: the corner (Duffy) rule runs through the SAME axis-factor
    function as the tensor rule.  On the identity map (polynomial weights,
    which both rules integrate exactly) with corner cells FORCED onto the
    middle cell (all four corners) and a corner cell (one), all 18 weighted
    blocks agree with the tensor-rule blocks to round-off (measured
    <= 3e-15 relative); bar 1e-12."""
    M = 5
    xw = np.array([0.0, 0.25, 0.85, _P])
    bx = TS.Basis1D(_P, xw, M)
    eps = np.full((3, 3, 1, 1), 2.5 + 0.1j)
    rule_t = TS._stag_map_quad_rule(M)
    nq = rule_t[0].size
    W_t = np.broadcast_to(eps, (3, 3, nq, nq))
    corners = {(1, 1): [(-1, -1), (-1, 1), (1, -1), (1, 1)],
               (0, 2): [(1, -1)]}
    quad = TS._StagMapQuad(M, nq, corners)
    W_p = TS._StagNodeWeight(
        W_t, {c: np.full(quad.points[c][2].shape, 2.5 + 0.1j)
              for c in corners})
    B, T_ = "B", "Btilde"
    for xs, ys in (((B, "m", B), (T_, "m", T_)), ((B, "m", T_), (T_, "m", B)),
                   ((B, "d", T_), (T_, "m", T_)),
                   ((T_, "dL", T_), (T_, "m", B)), ((B, "m", B), (B, "m", B))):
        a = TS._stag_quad_weighted(bx, bx, xs, ys, W_t, rule_t)
        b = TS._stag_quad_weighted(bx, bx, xs, ys, W_p, quad)
        assert _rel(b, a) <= 1e-12, (xs, ys, _rel(b, a))


def test_b8_corner_rule_decision_on_the_circle():
    """THE QUADRATURE DECISION (build doc section 2,
    ``b1_quadrature_*.json``).  On the 3 x 3 circle at M = 6:

    * WITH the corner rule the adaptive moment criterion is met at the
      first check, nq = 2M + 8 = 20, and the corner rule is used exactly in
      the disk cell (all four corners);
    * the corner-rule OPERATORS at n = 20 and n = 40 agree to <= 1e-11
      relative (measured 8.9e-13 at n = 20 vs the n = 48 top rung);
    * FAIL-BEFORE: without the corner rule (the Phase-A tensor rule in every
      cell) the criterion cannot be met -- it WARNS at its cap (moments fall
      only like n^-2: 5.6e-3 at n = 20 ... 1.4e-6 at n = 1280, measured) --
      and the operators at n = 20 are 6.3e-2 relative from the corner-rule
      ones (bar 1e-3: 1.8 decades under the defect, 9 above the agreement)."""
    M = 6
    cm, eps = _circle3()
    bx = TS.Basis1D(_P, cm.u_walls, M)
    by = TS.Basis1D(_P, cm.v_walls, M)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        n = TS._stag_map_nodes(bx, by, cm, M)
    assert n == 2 * M + 8
    corners = TS._stag_map_singular_corners(cm)
    assert list(corners) == [(1, 1)] and len(corners[(1, 1)]) == 4
    orig = TS._stag_map_nodes

    def forced(nn):
        return lambda *a, **k: nn
    try:
        TS._stag_map_nodes = forced(20)
        s20 = TS.Granet2DTransverseE(_P, _P, cm.u_walls, cm.v_walls, M, eps,
                                     k0=_K0, cmap=cm)
        TS._stag_map_nodes = forced(40)
        s40 = TS.Granet2DTransverseE(_P, _P, cm.u_walls, cm.v_walls, M, eps,
                                     k0=_K0, cmap=cm)
        TS._stag_map_nodes = forced(20)
        orig_c = TS._stag_map_singular_corners
        TS._stag_map_singular_corners = lambda c: {}
        try:
            sp = TS.Granet2DTransverseE(_P, _P, cm.u_walls, cm.v_walls, M,
                                        eps, k0=_K0, cmap=cm)
        finally:
            TS._stag_map_singular_corners = orig_c
    finally:
        TS._stag_map_nodes = orig
    sc = max(np.abs(s40.Lmat).max(), np.abs(s40.Rmat).max())
    d_ok = max(np.abs(s20.Lmat - s40.Lmat).max(),
               np.abs(s20.Rmat - s40.Rmat).max()) / sc
    d_plain = max(np.abs(sp.Lmat - s40.Lmat).max(),
                  np.abs(sp.Rmat - s40.Rmat).max()) / sc
    assert d_ok <= 1e-11
    assert d_plain > 1e-3
    orig_c = TS._stag_map_singular_corners
    try:
        TS._stag_map_singular_corners = lambda c: {}
        with pytest.warns(UserWarning, match="not resolved"):
            TS._stag_map_nodes(bx, by, cm, M)
    finally:
        TS._stag_map_singular_corners = orig_c


# =========================================================================== #
# B2 -- a uniform film under the circle map is exact (through the vertices)
# =========================================================================== #
def test_b2_film_under_the_circle_map_is_exact_and_spectral(monkeypatch):
    """A uniform eps-4 film passed as a PATTERNED cell under the 3 x 3 circle
    map: the only thing that can be wrong is the map (its four singular
    vertices included).  Measured 2026-10-02 (``b3_film_normal.json``),
    max over every order and both inputs against the Airy slab: 8.1e-07,
    2.8e-08, 4.3e-11, 3.3e-13, 5.4e-14 at M = 4 .. 8 (spectral); the
    Phase-A tensor rule everywhere reads the same to two digits
    (``b3_film_nocof_plain.json``: the film does not see the corner
    quadrature).  Bars: M = 6 <= 1e-9 (1.4 decades above 4.3e-11, 2.3 above
    the round-off floor), the M = 4 -> 6 drop >= 3 decades (measured 4.3).
    FAIL-BEFORE: the no-cofactor far projector, measured 0.148 at M = 6 --
    asserted > 1e-2 at M = 4 (measured below).

    RE-DERIVED 2026-10-03 (Phase C, an intentional algorithm change: under a
    map the incident wave now enters through its exact modal decomposition,
    which no longer passes through the far projector, so the engineered
    defect enters once -- the outgoing projection and the order-0
    renormalisation -- instead of twice).  Measured
    (``build_c/c_b2_nocof_rederive.json``): the correct arm 2.0e-6 / 1.1e-8 /
    8.4e-12 at M = 4 / 5 / 6, the no-cofactor arm FLAT at 4.77e-3 (a wrong
    bilinear form, not a discretisation error).  The fail-before bar becomes
    > 1e-3 at M = 4: 0.68 decades under the defect, 2.7 above the correct
    reading."""
    cm, _ = _circle3()
    e4 = _film_err(cm, 4)
    e6 = _film_err(cm, 6)
    assert e6 <= 1e-9, e6
    assert e4 / e6 >= 1e3, (e4, e6)
    _patch_no_cofactor(monkeypatch)
    assert _film_err(cm, 4) > 1e-3


# =========================================================================== #
# B3 / B4 -- the circular pillar against the saved FEM oracle; staircases
# =========================================================================== #
_ORD_FEM = {"0,0": [(0, 0)], "1,0": [(1, 0), (-1, 0)],
            "0,1": [(0, 1), (0, -1)],
            "1,1": [(1, 1), (-1, 1), (1, -1), (-1, -1)]}


@functools.lru_cache(maxsize=None)
def _fem_oracle():
    """The planner's SAVED NGSolve oracle (``fem/summary.json``; plan
    3.3.1), provenance asserted: the three independent meshes it names, every
    propagating order present, R + T - 1 below 1e-6.  Its own error bar is
    the largest per-order max-deviation over the three meshes (8.3e-6).
    E along y ('te', row 1 here).  Returns {(side, (m, n)): value}."""
    with open(os.path.join(_PROBE, "fem", "summary.json")) as f:
        d = json.load(f)
    assert d["best_from"] == ["h1.0_e20 p4", "h0.8_e30 p4", "h1.0 p6"]
    assert abs(d["RplusT"]["value"] - 1.0) < 1e-6
    spread = max(rec["maxdev"] for side in ("R", "T")
                 for rec in d[side].values())
    assert 5e-6 < spread < 1e-5            # 8.34e-6, the oracle's own bar
    ref = {}
    for side in ("R", "T"):
        for key, rec in d[side].items():
            assert key in _ORD_FEM
            for mn in _ORD_FEM[key]:
                ref[(side, mn)] = rec["value"]
    assert len(ref) == 18
    return ref


def _dist_fem(o, R, T):
    ref = _fem_oracle()
    i = dict(zip(_ORD9, _idx(o)))
    return max(abs((R if side == "R" else T)[1, i[mn]] - v)
               for (side, mn), v in ref.items())


@functools.lru_cache(maxsize=None)
def _circle3_m7():
    cm, eps = _circle3()
    o, R, T, _st = _solve(cm, eps, 7)
    return o, R, T


def _stair(k, M):
    """The planning probe's 4k-step staircase of the circle on the SHIPPED
    solver: walls at c +- r i / k, a cell filled when its centre is inside."""
    c = _P / 2
    inner = sorted([c - _R * i / k for i in range(1, k + 1)]
                   + [c + _R * i / k for i in range(1, k + 1)])
    w = np.array([0.0] + inner + [_P])
    mid = 0.5 * (w[:-1] + w[1:])
    n = mid.size
    eps = np.ones((n, n), complex)
    for i in range(n):
        for j in range(n):
            if (mid[i] - c) ** 2 + (mid[j] - c) ** 2 < _R ** 2:
                eps[i, j] = 4.0
    return _solve_walls(w[1:-1], w[1:-1], eps, M)


@functools.lru_cache(maxsize=None)
def _stair_k1_m7():
    return _stair(1, 7)


def test_b3_circle_lands_on_the_saved_fem_oracle():
    """The circular pillar on the 3 x 3 transfinite map against the
    planner's 3-D FEM (independent engine, saved; provenance asserted).
    Build-doc ladder (``b4_circle.json``), largest per-order distance to the
    FEM: 6.2e-3, 8.2e-4, 4.6e-4, 2.8e-5, 1.7e-5 at M = 6 .. 10, ... inside
    the FEM's own 8.3e-6 band at M = 11 / 12 (the STOP condition, 2.5e-5 at
    M = 11, is met there).  Unit-test size M = 7: measured 8.18e-04; bar
    2.5e-3 (0.5 decades above).  FAIL-BEFORE: the 4-step staircase of the
    same circle on the shipped solver at the same M, measured 7.07e-2 --
    1.45 decades above the bar (the curved cell is what moves the answer
    onto the FEM)."""
    assert _dist_fem(*_circle3_m7()) <= 2.5e-3
    assert _dist_fem(*_stair_k1_m7()) > 2.5e-3 * 10


def test_b4_staircases_converge_toward_the_curved_circle():
    """The shipped solver's staircases of the circle (4 steps: k = 1 on
    3 x 3; 8 steps: k = 2 on 5 x 5) approach the CURVED answer: their
    distances to it fall with the step count and their step points at it.
    Measured 2026-10-02 (``b4_circle.json``; planner's saved staircase
    JSON for the k = 4, 16-step rung): distance to the curved M = 7 answer
    7.12e-2 (k = 1, M = 7) and 5.25e-2 (k = 2, M = 5); the cosine between
    the step (k2 - k1) and (curved - k1) is 0.933 (0.991 for k = 2 -> 4,
    0.997 for k = 1 -> 4 against the FEM).  Two-sided: the curved answer is
    within 2.5e-3 of the FEM (test B3) while each staircase stays >= 1e-2
    from the curved answer (measured >= 5.2e-2: a curved solve that secretly
    staircased would collapse onto them); ordering d(k1) > d(k2); direction
    cosine >= 0.8 (measured 0.93)."""
    o, R, T = _circle3_m7()
    cur = _vec(o, R, T)
    s1 = _vec(*_stair_k1_m7())
    s2 = _vec(*_stair(2, 5))
    d1 = float(np.max(np.abs(s1 - cur)))
    d2 = float(np.max(np.abs(s2 - cur)))
    assert d1 > d2 >= 1e-2, (d1, d2)
    step, aim = s2 - s1, cur - s1
    cos = float(step @ aim / np.linalg.norm(step) / np.linalg.norm(aim))
    assert cos >= 0.8, cos


# =========================================================================== #
# B5 -- in-plane Bloch modes: spectral for the circle, stalled for a square
# =========================================================================== #
def _top_modes(cmap, w, eps, M, K=4):
    """The K leading eigenvalues n_eff^2 of the region pencil (L, -R) --
    -R is Hermitian positive definite, so Cholesky whitening + a standard
    eig (the eigenvalues do not depend on the eigensolver)."""
    import scipy.linalg as sla
    sol = TS.Granet2DTransverseE(_P, _P, w, w, M, eps, k0=_K0, cmap=cmap)
    B = -sol.Rmat
    Lc = sla.cholesky(0.5 * (B + B.conj().T), lower=True)
    A = sla.solve_triangular(Lc, sol.Lmat, lower=True)
    A = sla.solve_triangular(Lc, A.conj().T, lower=True).conj().T
    g2 = sla.eigvals(A, overwrite_a=True, check_finite=False)
    return g2[np.argsort(-g2.real)][:K]


def _rungs(cmap, w, eps, Ms):
    g = [_top_modes(cmap, w, eps, M) for M in Ms]
    return [float(np.max(np.abs(a - b))) for a, b in zip(g, g[1:])]


def test_b5_circle_modes_converge_spectrally():
    """The in-plane Bloch modes (the leading four n_eff^2 of the layer's own
    pencil; no rim, so only the cross-section is seen) of the circle on the
    5 x 5 map.  Build-doc ladder (``b5_modes_circle5.json``): rung changes
    2.2e-3, 2.1e-5, 1.5e-6, 4.7e-8, 2.5e-9 for M = 4 -> 9 (and 5.8e-4,
    4.0e-5, 6.5e-6, 3.3e-7 on the 3 x 3 map, M = 8 -> 12) -- the planner's
    P3/P4 ladders reproduced to two digits.  DECISION on the RATE at
    M = 5, 6, 7: the change falls >= 5x per rung (measured 13.5x) and is
    <= 1e-5 by 6 -> 7 (measured 1.5e-6).  The SAME decision fails for a
    square pillar (next test): that is the fail-before."""
    cm, eps = _circle5()
    d56, d67 = _rungs(cm, cm.u_bounds, eps, (5, 6, 7))
    assert d56 / d67 >= 5.0 and d67 <= 1e-5, (d56, d67)


def test_b5_square_pillar_modes_stall():
    """The square pillar (side 0.6, 3 x 3, no map) at M = 8, 9, 10: the four
    90-degree corners cap the in-plane rate (build doc,
    ``b5_modes_rect3.json``: 5.0e-5, 5.9e-5, 1.5e-5, 3.6e-5, 5.9e-6, 1.6e-5
    for M = 6 -> 12, no trend).  The circle's decision (>= 5x per rung) must
    FAIL here: measured ratio d(8->9) / d(9->10) = 0.42 (the change GROWS),
    1.1 decades under the bar."""
    w = np.array([0.0, 0.3, 0.9, _P])
    eps = np.ones((3, 3), complex)
    eps[1, 1] = 4.0
    d89, d910 = _rungs(None, w, eps, (8, 9, 10))
    assert d89 / d910 < 5.0, (d89, d910)


# =========================================================================== #
# B10 -- oblique and conical incidence under the circle map
# =========================================================================== #
def _reverse_deg(th_deg, ph_deg, m, n):
    """(theta', phi') of the incidence along -k_mn: the reversed channel of
    (incident -> reflected order (m, n)); that reversed incidence sends its
    order (m, n) back along -k_in."""
    st = np.sin(np.deg2rad(th_deg))
    kx = -(st * np.cos(np.deg2rad(ph_deg)) + m * _WL / _P)
    ky = -(st * np.sin(np.deg2rad(ph_deg)) + n * _WL / _P)
    return (float(np.rad2deg(np.arcsin(np.hypot(kx, ky)))),
            float(np.rad2deg(np.arctan2(ky, kx))))


def _jones_sv(st, m, n):
    """Singular values of the POWER-NORMALIZED 2 x 2 reflection Jones block
    of order (m, n): N = W_out^1/2 A G_in^-1/2, A the lab E_t -> order E_t
    amplitudes of the two inputs, G_in / W_out the power metrics of the
    incident / outgoing plane waves (orthonormal polarization bases on both
    sides, so the singular values are basis-free)."""
    md = st._modal
    o = np.asarray(md["orders"])
    k = int(np.nonzero((o[:, 0] == m) & (o[:, 1] == n))[0][0])
    A = np.array([[md["rx"][c][k] for c in (0, 1)],
                  [md["ry"][c][k] for c in (0, 1)]])
    kx0, ky0, kzi = md["kx0"], md["ky0"], md["kz_inc"]
    kxo, kyo = md["kx"][k], md["ky"][k]
    kzo = float(np.real(md["kz_ref"][k]))
    Gin = np.eye(2) + np.outer([kx0, ky0], [kx0, ky0]) / kzi ** 2
    Wo = (kzo / kzi) * (np.eye(2) + np.outer([kxo, kyo], [kxo, kyo])
                        / kzo ** 2)

    def msqrt(S, p):
        w, V = np.linalg.eigh(S)
        return (V * w ** p) @ V.conj().T
    return np.linalg.svd(msqrt(Wo, 0.5) @ A @ msqrt(Gin, -0.5),
                         compute_uv=False)


@functools.lru_cache(maxsize=None)
def _oblique(M, th, ph):
    cm, eps = _circle3()
    return _solve(cm, eps, M, np.deg2rad(th), np.deg2rad(ph))


def test_b10_oblique_circle_is_reciprocal_and_mirror_symmetric():
    """theta = 25 deg, phi = 0 on the circle (3 x 3 map, M = 6) -- NOT
    probed by the planner.  Build-doc ladders (``b10_oblique.json``): the
    rung-to-rung change falls 1.7e-2, 6.2e-3, 8.1e-4, 2.5e-4 (M = 6 -> 10),
    closure 1.9e-3 .. 5.3e-7; the 5 x 5 map 9.8e-4, 5.7e-5, 2.8e-5.
    Decisions at M = 6, both independent of convergence:

    * RECIPROCITY (an independent physical identity, not built into the
      solver): the singular values of the power-normalized 2 x 2 Jones
      block of the reflected order (-1, 0) equal those of the REVERSED
      channel (incidence along -k_(-1,0), theta' = 24.2498 deg, whose order
      (-1, 0) leaves along -k_in).  Measured 3.1e-6 (M = 6), 8.3e-7, 4.1e-8,
      9.7e-9 (M = 9) -- three decades under the convergence level.  Bar
      3e-5 (1 decade above); WRONG PAIRING (the reversed run's specular
      channel) measured 0.11, 3.6 decades above the bar.
    * y-MIRROR: the structure and the incidence are symmetric under
      y -> -y, so R(m, n) = R(m, -n) for both lab inputs.  Measured 5.2e-14
      (M = 6) .. 1.2e-8 (M = 10, the round-off floor F-B4); bar 1e-6."""
    fwd = _oblique(6, 25.0, 0.0)
    rev = _oblique(6, *_reverse_deg(25.0, 0.0, -1, 0))
    sf = _jones_sv(fwd[3], -1, 0)
    sr = _jones_sv(rev[3], -1, 0)
    sw = _jones_sv(rev[3], 0, 0)
    assert float(np.max(np.abs(sf - sr))) <= 3e-5, (sf, sr)
    assert float(np.max(np.abs(sf - sw))) >= 1e-2, (sf, sw)
    o, R, T, _st = fwd
    mir = 0.0
    for m, n in _ORD9:
        i, j = _idx(o, [(m, n), (m, -n)])
        mir = max(mir, float(np.max(np.abs(R[:, i] - R[:, j]))),
                  float(np.max(np.abs(T[:, i] - T[:, j]))))
    assert mir <= 1e-6, mir


def test_b10_conical_circle_is_reciprocal():
    """Conical incidence theta = 25 deg, phi = 40 deg (3 x 3 map, M = 6):
    no mirror symmetry left, s and p mix in every order.  RECIPROCITY of the
    channel (incident -> reflected order (-1, 0)) against its reversal
    (theta' = 35.2731 deg, phi' = -28.0614 deg): singular values of the
    power-normalized Jones blocks agree to 3.9e-5 (M = 6) and 9.4e-6
    (M = 7) (``b10_oblique.json``) while the rung-to-rung change is 3.0e-2
    and 5.7e-3.  Bar 3e-4 (0.9 decades above); the wrong pairing (the
    reversed run's specular channel) measured 9.9e-2, 2.5 decades above."""
    fwd = _oblique(6, 25.0, 40.0)
    rev = _oblique(6, *_reverse_deg(25.0, 40.0, -1, 0))
    sf = _jones_sv(fwd[3], -1, 0)
    sr = _jones_sv(rev[3], -1, 0)
    sw = _jones_sv(rev[3], 0, 0)
    assert float(np.max(np.abs(sf - sr))) <= 3e-4, (sf, sr)
    assert float(np.max(np.abs(sf - sw))) >= 1e-2, (sf, sw)


# =========================================================================== #
# B6 / B7 -- fillets: geometry fidelity, and a same-area square is no surrogate
# =========================================================================== #
_SIDE = 0.6


def _fillet(ratio):
    cm, _w = CM._fillet_map_5x5(_P, _SIDE / 2, ratio * _SIDE)
    eps = np.ones((5, 5), complex)
    eps[1:4, 1:4] = 4.0
    return cm, eps


def _square_cell(side):
    w = np.array([(_P - side) / 2, (_P + side) / 2])
    eps = np.ones((3, 3), complex)
    eps[1, 1] = 4.0
    return w, eps


@functools.lru_cache(maxsize=None)
def _r00_t00(kind, ratio, M):
    """(R00, T00) of input 'te' for the sharp square ('square'), its
    SAME-AREA square ('eqarea', side^2 - (4 - pi) r^2) -- both the shipped
    solver on 3 x 3 walls -- or the fillet map ('fillet', 5 x 5)."""
    if kind == "fillet":
        cm, eps = _fillet(ratio)
        o, R, T, _st = _solve(cm, eps, M)
    else:
        rf = ratio * _SIDE
        side = np.sqrt(_SIDE ** 2 - (4 - np.pi) * rf ** 2)
        w, eps = _square_cell(side)
        o, R, T = _solve_walls(w, w, eps, M)
    i0 = _idx(o, [(0, 0)])[0]
    return float(R[1, i0]), float(T[1, i0])


def test_b6_fillet_modes_approach_the_square_as_a_power_of_r():
    """The in-plane Bloch modes (no rim) see the fillet cleanly: the leading
    n_eff^2 FALLS monotonically with the fillet radius and its distance to the
    sharp square vanishes as r -> 0 with a power BETWEEN the corner exponent
    2 lambda = 1.61 (lambda = 0.806 for a 90-degree eps-4 corner: rounding a
    corner whose field is rho^(lambda - 1) removes energy ~ r^(2 lambda)) and
    2 (the area removed, (4 - pi) r^2) -- restated 2026-10-03 after the Phase
    B verifier's D-3.  Measured 2026-10-02 (``b5_modes_*.json``): sharp
    square (3 x 3, M = 8) 3.0079446; r / side = 0.05 (M = 6) 3.0070140; 0.2
    (M = 6) 2.9901856 -- differences to the square 9.3e-4 and 1.78e-2, ratio
    19 (r^2 predicts 16, r^1.61 9.3).  Each value is converged to <= 1.2e-5
    (rung changes to M = 9 / 12), two decades under the smaller difference.
    Bars: monotone with steps >= 1e-4; the ratio in [8, 32], which ACCEPTS
    both powers (9.3 and 16) and REJECTS a map whose r -> 0 limit missed the
    square by an OFFSET (~1) or a linear approach (4); the measured 19 sits
    0.4 decades inside each end."""
    w = np.array([0.0, 0.3, 0.9, _P])
    eps = np.ones((3, 3), complex)
    eps[1, 1] = 4.0
    sq = _top_modes(None, w, eps, 8, K=1)[0].real
    c05, e05 = _fillet(0.05)
    c20, e20 = _fillet(0.2)
    f05 = _top_modes(c05, c05.u_bounds, e05, 6, K=1)[0].real
    f20 = _top_modes(c20, c20.u_bounds, e20, 6, K=1)[0].real
    assert sq - f05 >= 1e-4 and f05 - f20 >= 1e-4, (sq, f05, f20)
    ratio = (sq - f20) / (sq - f05)
    assert 8.0 <= ratio <= 32.0, ratio


def test_b6_fillet_moves_the_efficiencies_far_above_convergence():
    """A fillet is GEOMETRY FIDELITY: at r / side = 0.2 it moves the
    zeroth orders far above the convergence level.  Build-doc ladder
    (``b6_fillet.json``): R00 -1.7e-3, T00 +7.4e-3 against the sharp square
    at the top rungs (the planner's P4 numbers reproduced), while the
    efficiency ladders change ~1e-5 per rung.  Unit-test size (fillet 5 x 5
    at M = 5, square 3 x 3 at M = 6), measured: R00 -1.80e-3, T00 +7.65e-3;
    bars R00 <= -5e-4 and T00 >= +2e-3 (0.55 / 0.58 decades inside), while
    the M = 5 / M = 6 discretisation errors are <= 1.4e-4 (R00) and
    4.4e-4 (T00) against the top rungs."""
    r_sq, t_sq = _r00_t00("square", 0.0, 6)
    r_f, t_f = _r00_t00("fillet", 0.2, 5)
    assert r_f - r_sq <= -5e-4, (r_f, r_sq)
    assert t_f - t_sq >= 2e-3, (t_f, t_sq)


def test_b7_same_area_square_is_not_a_fillet():
    """The obvious cheap surrogate -- shrink the square until its AREA
    matches the filleted pillar -- moves R00 the WRONG WAY.  Measured
    2026-10-02 at r / side = 0.2 (unit-test sizes, shipped solver for both
    squares at M = 6, fillet at M = 5): same-area square R00 +2.4e-4 against
    the sharp square (build doc: +3.2e-4 at M = 7 and M = 11), fillet
    -1.80e-3.  Two-sided: the same-area shift must be POSITIVE (> +5e-5,
    0.7 decades inside) while the fillet's is negative (test above), and the
    two answers differ by >= 1e-3 in R00 (measured 2.04e-3) -- 1.5 decades
    above the build-doc convergence level of either."""
    r_sq, _t = _r00_t00("square", 0.0, 6)
    r_eq, _t = _r00_t00("eqarea", 0.2, 6)
    r_f, _t = _r00_t00("fillet", 0.2, 5)
    assert r_eq - r_sq > 5e-5, (r_eq, r_sq)
    assert r_eq - r_f >= 1e-3, (r_eq, r_f)


# =========================================================================== #
# B9 -- two new shapes: an ellipse and a sinusoidal wall
# =========================================================================== #
_ELL = (0.40, 0.28)
_SX1, _SX2, _SA = 0.3, 0.9, 0.12


def _sym_te_tm(o, R, T):
    """max |te(m, n) - tm(n, m)| over the nine orders: zero for a four-fold
    symmetric cell (the circle), large for an ellipse."""
    out = 0.0
    for m, n in _ORD9:
        i, j = _idx(o, [(m, n), (n, m)])
        out = max(out, abs(R[1, i] - R[0, j]), abs(T[1, i] - T[0, j]))
    return float(out)


@functools.lru_cache(maxsize=None)
def _ellipse_m7():
    cm, _w = CM._ellipse_map_3x3(_P, _ELL)
    eps = np.ones((3, 3), complex)
    eps[1, 1] = 4.0
    o, R, T, _st = _solve(cm, eps, 7)
    return o, R, T


def _rcwa_vec(shapes, n, monkeypatch=None):
    from lumenairy.elements.rcwa import twod
    rows = {}
    for pol in ("te", "tm"):
        o, R, T = twod.rcwa_efficiency_2d_shapes(
            _P, _P, 1.0, shapes, _NSUB, _NSUP, _DEPTH, _WL,
            polarization=pol, n_orders_x=n, n_orders_y=n)
        o = np.asarray(o)
        i = _idx(o)
        rows[pol] = (np.asarray(R)[i], np.asarray(T)[i])
    # the library's row order: 0 = tm (E along x), 1 = te (E along y)
    return np.concatenate([rows["tm"][0], rows["te"][0], rows["tm"][1],
                           rows["te"][1]])


def test_b10_film_under_the_circle_map_at_conical_incidence():
    """The uniform film under the 3 x 3 circle map at CONICAL incidence
    (theta = 25 deg, phi = 40 deg) against the s / p Airy slab: measured
    3.3e-04, 2.7e-05, 1.4e-06, 6.3e-08, 3.4e-09 at M = 4 .. 8
    (``b3_film_oblique.json``; 25 deg in-plane: 7.2e-4 .. 1.0e-8; the 5 x 5
    map 2.4e-7 at M = 4) -- spectral; the Bloch glue and the alpha0 kernel
    need nothing new under a periodic curved map.  Bars: M = 6 <= 1e-5
    (0.85 decades above 1.4e-6) and the M = 4 -> 6 drop >= 1.5 decades
    (measured 2.4)."""
    cm, _ = _circle3()
    th, ph = np.deg2rad(25.0), np.deg2rad(40.0)
    e4 = _film_err(cm, 4, th, ph)
    e6 = _film_err(cm, 6, th, ph)
    assert e6 <= 1e-5, e6
    assert e4 / e6 >= 10 ** 1.5, (e4, e6)


def test_b9_ellipse_against_the_exact_form_factor_rcwa():
    """An axis-aligned ELLIPSE (semi-axes 0.40 x 0.28, the 3 x 3 map; four
    singular vertices) -- a new shape, no planner number.  Reference: the
    shipped 2-D RCWA with the EXACT ellipse form factor (Laurent), which
    converges algebraically (~1 / N) toward the curved answer: build-doc
    ladder (``b9_shapes.json``) 1.29e-2, 9.1e-3, 7.0e-3, 5.7e-3, 4.8e-3,
    4.1e-3 from the curved top rung (M = 11) at 9 .. 29 orders per axis, the
    1 / N Richardson pair 1.9e-3 .. 3.0e-4 (falling); the curved ladder
    changes 7.5e-3 .. 2.3e-5 per rung (M = 6 -> 11), closure 6.3e-9 at
    M = 11.  Unit-test size (curved M = 7, RCWA 9 and 13 orders): distances
    1.35e-2 and 9.8e-3 (the Fourier engine approaches), Richardson pair
    2.5e-3.  Bars: Richardson <= 6e-3 (0.4 decades), raw 13-order RCWA
    >= 6e-3 (0.2 decades: the reference is not yet there by itself).
    FAIL-BEFORE: the four-fold symmetry te(m, n) = tm(n, m), exact for the
    circle (B3: <= 8.9e-10), is broken by the ellipse (measured 0.11; bar
    >= 1e-2) -- the solver sees the shape, not a circle."""
    o, R, T = _ellipse_m7()
    cur = _vec(o, R, T)
    shp = [{"shape": "ellipse", "eps": 4.0, "semi_axes": _ELL,
            "center": (_P / 2, _P / 2)}]
    r4 = _rcwa_vec(shp, 4)
    r6 = _rcwa_vec(shp, 6)
    rich = (13 * r6 - 9 * r4) / 4
    d4 = float(np.max(np.abs(r4 - cur)))
    d6 = float(np.max(np.abs(r6 - cur)))
    assert d4 > d6 >= 6e-3, (d4, d6)
    assert float(np.max(np.abs(rich - cur))) <= 6e-3
    assert _sym_te_tm(o, R, T) >= 1e-2


def _sine_ff(gxv, gyv, px, py):
    """Exact Fourier form factor of the ridge x1 + A sin(2 pi y / p) < x <
    x2 + A sin(2 pi y / p) (Jacobi-Anger); checked against a 4096^2 pixel
    FFT to 4.6e-6 in the build (``b9_sine_rcwa_n*.json``)."""
    from scipy.special import jv
    n = np.rint(gyv / (2 * np.pi / py)).astype(int)
    zero = np.abs(gxv) < 1e-12
    g = np.where(zero, 1.0, gxv)
    X = (np.exp(-1j * g * _SX2) - np.exp(-1j * g * _SX1)) / (-1j * g)
    out = np.where(zero, 0.0, jv(-n, g * _SA) * X / px).astype(complex)
    out[zero & (n == 0)] = (_SX2 - _SX1) / px
    return out


def test_b9_sinusoidal_wall_mirror_and_exact_form_factor(monkeypatch):
    """A constant-width ridge bounded by two in-phase sinusoids, x = 0.3 /
    0.9 + 0.12 sin(2 pi y / p) (the 3 x 3 map whose interior vertical grid
    lines ARE the sinusoids; no singular vertex).  Build-doc ladder
    (``b9_shapes.json``): rung changes 1.6e-3, 9.9e-5, 3.2e-5 (M = 6 -> 9),
    closure 7.5e-9 at M = 9; the exact-form-factor RCWA (probe-only shape,
    Jacobi-Anger) approaches it like 1 / N (2.1e-2 .. 8.8e-3 at 9 .. 21
    orders, Richardson 1.2e-3 .. 4.8e-4).  Decisions at M = 6:

    * MIRROR: A -> -A is the mirror image y -> -y of the device (the v walls
      are mirror-symmetric), so R(m, n; A) = R(m, -n; -A) for both inputs --
      exact in the discretisation (measured at the round-off floor); bar
      1e-7;
    * the RCWA at 9 / 13 orders approaches the curved answer (2.13e-2 >
      1.45e-2 >= 6e-3) and its 1 / N Richardson pair lands within 6e-3
      (measured 2.7e-3)."""
    sols = {}
    for A in (_SA, -_SA):
        cm, _w = CM._sine_stripe_map_3x3(_P, _SX1, _SX2, A)
        eps = np.ones((3, 3), complex)
        eps[1, :] = 4.0
        sols[A] = _solve(cm, eps, 6)[:3]
    (o, R, T), (o2, R2, T2) = sols[_SA], sols[-_SA]
    mir = 0.0
    for m, n in _ORD9:
        i = _idx(o, [(m, n)])[0]
        j = _idx(o2, [(m, -n)])[0]
        mir = max(mir, float(np.max(np.abs(R[:, i] - R2[:, j]))),
                  float(np.max(np.abs(T[:, i] - T2[:, j]))))
    assert mir <= 1e-7, mir
    from lumenairy.elements.rcwa import twod
    orig_ff = twod._shape_form_factor

    def ff(shape, gxv, gyv, px, py):
        if shape["shape"] == "sinstripe":
            return _sine_ff(gxv, gyv, px, py)
        return orig_ff(shape, gxv, gyv, px, py)
    monkeypatch.setattr(twod, "_shape_form_factor", ff)
    monkeypatch.setattr(twod, "_validate_shapes", lambda *a, **k: None)
    shp = [{"shape": "sinstripe", "eps": 4.0}]
    cur = _vec(o, R, T)
    r4 = _rcwa_vec(shp, 4)
    r6 = _rcwa_vec(shp, 6)
    d4 = float(np.max(np.abs(r4 - cur)))
    d6 = float(np.max(np.abs(r6 - cur)))
    assert d4 > d6 >= 6e-3, (d4, d6)
    assert float(np.max(np.abs((13 * r6 - 9 * r4) / 4 - cur))) <= 6e-3
