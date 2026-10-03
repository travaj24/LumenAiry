"""INDEPENDENT VERIFICATION of Phase B of the curved-cell build (pure
staggered 2-D PMM: transfinite maps, the corner / Duffy quadrature) -- the
decision tests for the gaps this verification found next to
``tests/unit/test_pmm2d_staggered_curved_b.py`` (the circle's symmetry, the
seam condition F-B1, the corner rule on regular cells and on every shape, a
second FEM radius, the curve-derivative trust gap D-1).  The mutation matrix
of the build tests left ONE survivor, which is an equivalent mutant (see
the fingerprint test).  Report: ``docs/audits/VERIFY_PMM2D_CURVED_B_2026_10_02.md``;
probes and JSON: ``validation/probe_pmm2d_curved/verify_b/``.

Every map here is built from the PUBLIC primitives (``TransfiniteMap``,
``Arc.through``, ``EllipseArc``, ``Sinusoid``), not from the builder's
private ``_circle_map_3x3`` & co. (one exception: the quadrature test also
runs on the builder's ``_fillet_map_5x5``, which the verifier's own fillet
construction reproduces to 1e-14, ``v1_geometry_win.json``).  Fixture: lambda 1, square period 1.2,
depth 0.5, air over n = 1.45, eps-4 features.  Every bar quotes the
measurement it rests on (Windows 11, CPython 3.14.6, numpy 2.4.4, scipy
1.17.1, BLAS pinned to one thread; 2026-10-02) and its gap on both sides.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from numpy.polynomial.legendre import leggauss, legvander  # noqa: E402

from lumenairy.elements.pmm import PMM2DStackPure  # noqa: E402
from lumenairy.elements.pmm import _curvemap as CM  # noqa: E402
from lumenairy.elements.pmm import twod_staggered as TS  # noqa: E402

_P = 1.2
_WL = 1.0
_K0 = 2 * np.pi / _WL
_DEPTH = 0.5
_ORD9 = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1),
         (-1, -1)]
_DEG = np.pi / 180.0
_HERE = os.path.dirname(os.path.abspath(__file__))
_VB = os.path.join(_HERE, "..", "..", "validation", "probe_pmm2d_curved",
                   "verify_b")


def _verts(uw, vw):
    V = np.empty((len(uw), len(vw), 2))
    V[..., 0] = np.asarray(uw)[:, None]
    V[..., 1] = np.asarray(vw)[None, :]
    return V


def _circle3(r, c=(_P / 2, _P / 2)):
    c = np.asarray(c, float)
    h = r / np.sqrt(2.0)
    w = np.array([0.0, c[0] - h, c[0] + h, _P])
    V = _verts(w, w)
    ed = {("h", 1, 1): CM.Arc.through(V[1, 1], V[2, 1], c),
          ("h", 1, 2): CM.Arc.through(V[1, 2], V[2, 2], c),
          ("v", 1, 1): CM.Arc.through(V[1, 1], V[1, 2], c),
          ("v", 2, 1): CM.Arc.through(V[2, 1], V[2, 2], c)}
    eps = np.ones((3, 3), complex)
    eps[1, 1] = 4.0
    return CM.TransfiniteMap(w, w, V, ed), eps


def _solve(cmap, eps, M, theta=0.0, phi=0.0):
    st = PMM2DStackPure(_P, _P, n_superstrate=1.0, n_substrate=1.45,
                        n_modes=M, n_orders=3, cmap=cmap)
    st.add_layer(_DEPTH, eps_cell=eps)
    st.set_source(_WL, theta=theta, phi=phi)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o, R, T, _J = st.solve()
    return np.asarray(o), np.asarray(R), np.asarray(T)


def _solve_walls(xw, yw, eps, M, theta=0.0, phi=0.0):
    st = PMM2DStackPure(_P, _P, n_superstrate=1.0, n_substrate=1.45,
                        n_modes=M, n_orders=3, layer_grids="per-layer")
    st.add_layer(_DEPTH, eps_cell=eps, x_walls=xw, y_walls=yw)
    st.set_source(_WL, theta=theta, phi=phi)
    o, R, T = st.solve(jones=False)
    return np.asarray(o), np.asarray(R), np.asarray(T)


def _idx(o, orders=_ORD9):
    return [int(np.nonzero((o[:, 0] == m) & (o[:, 1] == n))[0][0])
            for m, n in orders]


def _vec(o, R, T):
    i = _idx(o)
    return np.concatenate([R[:, i].ravel(), T[:, i].ravel()])


def _sym(o, R, T):
    s = 0.0
    for m, n in _ORD9:
        a, b = _idx(o, [(m, n), (n, m)])
        s = max(s, abs(R[1, a] - R[0, b]), abs(T[1, a] - T[0, b]))
    return float(s)


# --------------------------------------------------------------------------
# the fingerprint sees a curve that changes while every wall and vertex stays
# --------------------------------------------------------------------------
def test_verify_b_fingerprint_sees_every_arc_parameter():
    """The fingerprint keys caches ("equal exactly when the geometry is
    equal").  The build's test changes the radius of a WHOLE circle map,
    which also moves the walls and vertex images.  Here two maps share every
    wall and every vertex image and differ ONLY in one edge curve (two arcs
    through the same two vertices about different centres, so different
    radii and bulges): their fingerprints must differ, and equal content
    must give equal fingerprints.  (Verifier mutant m4 -- the radius dropped
    from ``Arc.key`` -- survives this and every test, and is EQUIVALENT: the
    constructor pins both arc ends to the vertex images to 1e-12, so centre +
    angles + vertex images determine the radius; no collision is possible.)"""
    w = np.array([0.0, 0.4, 0.8, _P])
    V = _verts(w, w)
    P0, P1 = V[1, 1], V[2, 1]
    mid = 0.5 * (P0 + P1)
    maps = []
    for d in (0.6, 0.9):                       # centre below the edge
        c = mid - np.array([0.0, d])
        maps.append(CM.TransfiniteMap(w, w, V, {("h", 1, 1):
                                                CM.Arc.through(P0, P1, c)}))
    a, b = maps
    assert a.curved_edges[("h", 1, 1)].radius != b.curved_edges[
        ("h", 1, 1)].radius
    assert a.fingerprint != b.fingerprint
    c = mid - np.array([0.0, 0.6])
    a2 = CM.TransfiniteMap(w, w, V, {("h", 1, 1): CM.Arc.through(P0, P1, c)})
    assert a2.fingerprint == a.fingerprint


# --------------------------------------------------------------------------
# the circle's four-fold symmetry is never asserted by the build tests
# --------------------------------------------------------------------------
def test_verify_b_circle_fourfold_symmetry_is_exact():
    """A circle centred in a square cell is four-fold symmetric, so te(m, n)
    = tm(n, m) for every order -- independent of convergence (the map is
    symmetric).  Measured 2026-10-02 at M = 6 (``v11_bars_win.json``):
    2.2e-9 (the round-off floor of the incident projection, F-B4); a circle
    whose x half-axis is off by 1e-6 relative (an ellipse) reads 3.0e-7, by
    1e-4 reads 3.0e-5 (linear, 0.30 per unit aspect).  Bar 1e-7: 1.65
    decades above the circle, it sees an aspect defect of 3e-7.  The build
    tests never assert that the circle's SOLVE is symmetric (only that the
    ellipse's is not); the verifier's mutant m5 (the builder's circle map
    pushed out by 1e-4 r) was caught there by the AREA check of the map, so
    a symmetry defect in the solver itself (quadrature, assembly) would have
    gone unseen."""
    o, R, T = _solve(*_circle3(0.36), 6)
    assert _sym(o, R, T) <= 1e-7
    # the same check sees a 1e-5 aspect defect (measured 3.0e-6 expected)
    a = 0.36 * (1 + 1e-5)
    h = np.array([a, 0.36]) / np.sqrt(2.0)
    c = np.array([_P / 2, _P / 2])
    uw = np.array([0.0, c[0] - h[0], c[0] + h[0], _P])
    vw = np.array([0.0, c[1] - h[1], c[1] + h[1], _P])
    ax = (a, 0.36)
    ed = {("h", 1, 1): CM.EllipseArc(c, ax, 225 * _DEG, 315 * _DEG),
          ("h", 1, 2): CM.EllipseArc(c, ax, 135 * _DEG, 45 * _DEG),
          ("v", 1, 1): CM.EllipseArc(c, ax, 225 * _DEG, 135 * _DEG),
          ("v", 2, 1): CM.EllipseArc(c, ax, -45 * _DEG, 45 * _DEG)}
    eps = np.ones((3, 3), complex)
    eps[1, 1] = 4.0
    o, R, T = _solve(CM.TransfiniteMap(uw, vw, None, ed), eps, 6)
    assert _sym(o, R, T) > 1e-6


# --------------------------------------------------------------------------
# F-B1: what the seam needs -- positions (hence the tangent), not the normal
# derivative
# --------------------------------------------------------------------------
def test_verify_b_normal_kink_is_harmless_and_position_shift_is_not(
        monkeypatch):
    """F-B1 derived (report section 5): across a u = const line the (u, v)
    problem's interface conditions are those of E'_v = E . Phi_v, E'_z and
    the flux D'^u = (y_v, -x_v) . D, all built from the TANGENT Phi_v; the
    normal derivative Phi_u enters only the broken unknown E'_u.  So a map
    whose normal derivative JUMPS at the periodic seam is exact, while a
    seam whose POSITIONS do not match is wrong.  Measured
    (``v5_kink_win.json``):

    * a piecewise-affine map, x_u = 1.67 left of the seam and 1.0 right of
      it, vs the unmapped solver on the physical walls (the same polynomial
      spaces): 1.6e-14 at M = 5, normal incidence; bar 1e-11;
    * the c3 circle re-parametrised with KINKED outer cells (u walls 0.20 /
      0.709 instead of 0.345 / 0.855; seam x_u 1.38 vs 0.56): 1.7e-9 from the
      plain c3 circle at M = 6 (1.5e-12 at M = 9), against a rung change of
      5.8e-3; bar 1e-7;
    * FAIL-BEFORE: the u = p side's vertices shifted by 0.05 in y against
      u = 0 is REFUSED by validate(); forced past it, the answer is 4.1e-3
      wrong at M = 5 and does not converge away (4.0e-3 at M = 9); bar
      >= 1e-3."""
    uw = np.array([0.0, 0.3, 0.9, _P])
    xi = np.array([0.0, 0.5, 0.9, _P])
    V = _verts(uw, uw)
    V[:, :, 0] = xi[:, None]
    km = CM.TransfiniteMap(uw, uw, V)
    eps = np.ones((3, 3), complex)
    eps[1, 1] = 4.0
    a = _vec(*_solve(km, eps, 5))
    b = _vec(*_solve_walls(xi[1:-1], uw[1:-1], eps, 5))
    assert float(np.max(np.abs(a - b))) <= 1e-11
    c3, e3 = _circle3(0.36)
    h = 0.36 / np.sqrt(2)
    uk = np.array([0.0, 0.20, 0.20 + 2 * h, _P])
    ck = CM.TransfiniteMap(uk, uk, c3.vertex_images.copy(),
                           dict(c3.curved_edges))
    d = float(np.max(np.abs(_vec(*_solve(c3, e3, 6))
                            - _vec(*_solve(ck, e3, 6)))))
    assert d <= 1e-7
    Vb = _verts(uw, uw)
    Vb[3, 1, 1] += 0.05
    Vb[3, 2, 1] += 0.05
    with pytest.raises(ValueError, match="lattice-periodic"):
        CM.TransfiniteMap(uw, uw, Vb)
    monkeypatch.setattr(CM.CellMap, "validate", lambda self, n=7: self)
    bad = CM.TransfiniteMap(uw, uw, Vb)
    monkeypatch.undo()
    w = np.array([0.3, 0.9])
    d = float(np.max(np.abs(_vec(*_solve(bad, eps, 5))
                            - _vec(*_solve_walls(w, w, eps, 5)))))
    assert d >= 1e-3


# --------------------------------------------------------------------------
# B8: the corner rule fed a REGULAR cell; the Duffy rule on every shape
# --------------------------------------------------------------------------
def test_verify_b_corner_rule_on_a_regular_cell_is_the_tensor_rule():
    """The corner rule given a cell with NO singular vertex (corners = [])
    lays the tensor Gauss nodes through the point path; given a FORCED
    corner on a smooth (identity-map) cell it Duffy-integrates polynomial
    weights exactly.  Both must equal the tensor-rule operators to round-off
    -- NOT bit-for-bit (the point path sums in another order).  Measured
    (``v5_corner0_win.json``): Lmat 5.0e-14 / 1.4e-13 relative, never
    bitwise.  Bar 1e-12."""
    w = np.array([0.0, 0.3, 0.9, _P])
    eps = np.ones((3, 3), complex)
    eps[1, 1] = 4.0
    tf = CM.TransfiniteMap(w, w)
    base = TS.Granet2DTransverseE(_P, _P, w, w, 6, eps, k0=_K0, cmap=tf)
    orig = TS._stag_map_singular_corners
    for fake in ({(1, 1): []}, {(1, 1): [(-1, -1)]},
                 {(1, 1): [(-1, -1), (-1, 1), (1, -1), (1, 1)]}):
        TS._stag_map_singular_corners = lambda c, f=fake: f
        try:
            s = TS.Granet2DTransverseE(_P, _P, w, w, 6, eps, k0=_K0, cmap=tf)
        finally:
            TS._stag_map_singular_corners = orig
        for a in ("Lmat", "Rmat", "Stt", "Schur"):
            x, y = getattr(s, a), getattr(base, a)
            assert float(np.max(np.abs(x - y)) / np.max(np.abs(y))) <= 1e-12


def test_verify_b_duffy_is_spectral_on_every_shape_and_plain_gauss_is_n2():
    """The quadrature decision re-derived: det J has a SIMPLE zero at every
    singular vertex (``v1_geometry_win.json``: exponent 1.0000 on every ray,
    circle / ellipse / fillet), so the geometric weights are homogeneous of
    degree -1; their 1-D marginal is log(1/s), on which Gauss-Legendre
    converges as n^-2 -- the moment error halves twice per doubling
    (measured ratios 3.9-4.0); the Duffy collapse cancels the 1 / rho
    EXACTLY (the transformed integrand is analytic), so at the solver's
    n = 2 M + 8 the moments are at round-off on EVERY shape (measured
    <= 8e-15, fillet r / side 0.02 4.4e-14; ``v2_moments_win.json``).
    Bars: Duffy <= 1e-12 at n = 2 M + 8, M = 6 and 8, on the circle,
    ellipse and fillet corner cells; plain-Gauss ratio err(20) / err(40) in
    [3, 5] (n^-2 = 4; an n^-1 family reads 2, a spectral one >> 10)."""
    def maps():
        c3, _ = _circle3(0.36)
        c = np.array([_P / 2, _P / 2])
        a, b = 0.42, 0.24
        uw = np.array([0.0, c[0] - a / np.sqrt(2), c[0] + a / np.sqrt(2), _P])
        vw = np.array([0.0, c[1] - b / np.sqrt(2), c[1] + b / np.sqrt(2), _P])
        ed = {("h", 1, 1): CM.EllipseArc(c, (a, b), 225 * _DEG, 315 * _DEG),
              ("h", 1, 2): CM.EllipseArc(c, (a, b), 135 * _DEG, 45 * _DEG),
              ("v", 1, 1): CM.EllipseArc(c, (a, b), 225 * _DEG, 135 * _DEG),
              ("v", 2, 1): CM.EllipseArc(c, (a, b), -45 * _DEG, 45 * _DEG)}
        e3 = CM.TransfiniteMap(uw, vw, None, ed)
        f5, _ = CM._fillet_map_5x5(_P, 0.3, 0.012)
        return (c3, e3, f5)

    def geom5(cm, sx, sy, s, t):
        du = 0.5 * (cm.u_bounds[sx + 1] - cm.u_bounds[sx])
        dv = 0.5 * (cm.v_bounds[sy + 1] - cm.v_bounds[sy])
        U = 0.5 * (cm.u_bounds[sx + 1] + cm.u_bounds[sx]) + du * s
        V = 0.5 * (cm.v_bounds[sy + 1] + cm.v_bounds[sy]) + dv * t
        _X, _Y, xu, xv, yu, yv = cm.geom_points(sx, sy, U, V)
        sg = xu * yv - xv * yu
        return [sg, 1 / sg, (xu * xu + yu * yu) / sg,
                (xu * xv + yu * yv) / sg, (xv * xv + yv * yv) / sg]

    def mom(cm, cell, s, t, w, deg):
        f = geom5(cm, cell[0], cell[1], s, t)
        Ps = legvander(s, deg) * w[:, None]
        Pt = legvander(t, deg)
        return [Ps.T @ (fk[:, None] * Pt) for fk in f]

    def gl(n):
        x, w = leggauss(n)
        return np.repeat(x, n), np.tile(x, n), np.outer(w, w).ravel()

    def err(a, b):
        return max(float(np.max(np.abs(x - y)) / np.max(np.abs(y)))
                   for x, y in zip(a, b))

    for cm in maps():
        cells = TS._stag_map_singular_corners(cm)
        assert cells
        for M in (6, 8):
            deg = 2 * M - 2
            for cell, cs in cells.items():
                ref = mom(cm, cell, *TS._stag_duffy_points(cs, 64), deg)
                got = mom(cm, cell, *TS._stag_duffy_points(cs, 2 * M + 8),
                          deg)
                assert err(got, ref) <= 1e-12
        cell = sorted(cells)[0]
        ref = mom(cm, cell, *TS._stag_duffy_points(cells[cell], 64), 10)
        e20 = err(mom(cm, cell, *gl(20), 10), ref)
        e40 = err(mom(cm, cell, *gl(40), 10), ref)
        assert 3.0 <= e20 / e40 <= 5.0


# --------------------------------------------------------------------------
# B3 at a SECOND radius: the verifier's own FEM (r = 0.24 = 0.2 p)
# --------------------------------------------------------------------------
def _fem240():
    with open(os.path.join(_VB, "fem_r240_results.jsonl")) as f:
        recs = [json.loads(line) for line in f]
    sel = []
    for tag, p in (("h1.0_e20", 4), ("h0.8_e30", 4), ("h1.0", 6)):
        m = [x for x in recs if x["tag"] == tag and x["p"] == p]
        assert m and abs(m[-1]["rad"] - 240.0) < 1e-9
        assert abs(m[-1]["RplusT"] - 1.0) < 1e-6
        sel.append(m[-1])
    keys = {"0,0": [(0, 0)], "1,0": [(1, 0), (-1, 0)],
            "0,1": [(0, 1), (0, -1)],
            "1,1": [(1, 1), (-1, 1), (1, -1), (-1, -1)]}
    ref, spread = {}, 0.0
    for side in ("R", "T"):
        for k, mns in keys.items():
            v = np.array([x[side][k]["eff"] for x in sel])
            spread = max(spread, float(np.max(np.abs(v - v.mean()))))
            for mn in mns:
                ref[(side, mn)] = float(v.mean())
    assert 5e-6 < spread < 3e-5               # 1.47e-5, the oracle's own bar
    return ref


def test_verify_b_circle_lands_on_a_second_fem_radius():
    """The circle agreement is not a single-fixture fact: a pillar of
    r = 0.24 (0.2 p) against this verifier's own NGSolve run of the
    planner's runner (``fem_circle_r.py``; three independent meshes, spread
    1.47e-5, R + T - 1 <= 1.7e-7; the runner's provenance re-checked by
    reproducing the saved r = 0.36 'h1.0 p4' record to 1.4e-14).  Ladder
    (``v3_summary.json``): 6.0e-4, 1.35e-4, 4.9e-5, 2.8e-5, 1.1e-5, 5.2e-6,
    2.2e-6 at M = 6 .. 12 (r = 0.48: 1.6e-5 at M = 12).  Unit size M = 6:
    measured 5.96e-4; bar 2e-3 (0.5 decades).  FAIL-BEFORE: the 4-step
    staircase of the same circle on the shipped solver, measured 8.0e-2 at
    M = 7 (``v11_bars_win.json``) and asserted > 1e-2 at M = 6."""
    ref = _fem240()
    c = _P / 2

    def dist(o, R, T):
        i = dict(zip(_ORD9, _idx(o)))
        return max(abs((R if s == "R" else T)[1, i[mn]] - v)
                   for (s, mn), v in ref.items())
    assert dist(*_solve(*_circle3(0.24), 6)) <= 2e-3
    eps = np.ones((3, 3), complex)
    eps[1, 1] = 4.0
    w = np.array([c - 0.24, c + 0.24])
    assert dist(*_solve_walls(w, w, eps, 6)) > 1e-2


# --------------------------------------------------------------------------
# defect D-1: a user EdgeCurve whose derivative is not d/ds of its value
# --------------------------------------------------------------------------
class _BadSinusoid(CM.Sinusoid):
    def __call__(self, s):
        v, d = super().__call__(s)
        d = d.copy()
        d[:, 0] *= 1.1                        # a 10 % derivative bug
        return v, d


def test_verify_b_inconsistent_curve_derivative_is_refused():
    """The map's POSITIONS come from each curve's value and its JACOBIAN
    from the curve's analytic derivative; nothing checks that the two agree.
    A Sinusoid subclass with a 10 % derivative bug is accepted and moves
    R / T by 3.2e-2 (M = 5) / 4.6e-3 (M = 7) (``v5_curveder_win.json``) --
    silently wrong.  ``EdgeCurve`` is public (``__all__``) and invites user
    curves.  Was a strict xfail until the constructor refused it (fixed in
    Phase D, the report's section-12 edit applied verbatim)."""
    vw = np.array([0.0, 0.36, 0.9, _P])
    uw = np.array([0.0, 0.3, 0.9, _P])
    V = _verts(uw, vw)
    k = 2 * np.pi / _P
    for i in (1, 2):
        V[i, :, 0] += 0.1 * np.sin(k * vw)
    ed = {("v", i, j): _BadSinusoid(uw[i], 0.1, _P, vw[j], vw[j + 1])
          for i in (1, 2) for j in range(3)}
    with pytest.raises(ValueError):
        CM.TransfiniteMap(uw, vw, V, ed)
