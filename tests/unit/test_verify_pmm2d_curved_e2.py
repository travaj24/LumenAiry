"""INDEPENDENT VERIFIER decision tests -- curved-cell Phase E2 of the pure
staggered 2-D PMM: per-layer maps joined by the CURVED (non-separable)
mortar (``docs/audits/VERIFY_PMM2D_CURVED_E2_2026_10_03.md``).

What they close.  The verifier's mutation matrix
(``validation/probe_pmm2d_curved/verify_e2/v8_*``) found that the twelve
Phase E2 gates survive two kernel mutants -- the tangency search disabled
(the cross-mass then moves by 2e-2 on a pair whose pulled-back wall is
tangent to the inner direction; 4e-6 in R / T on a crossing device, closure
blind) and the square-root substitution at tangency ends disabled (the
adaptive rule then runs to its 96-node cap) -- because no gate compares the
cross-mass against an oracle that does not share the kernel's cut logic.
The oracle here is a BRUTE-FORCE cross-mass in PHYSICAL coordinates (an
iterated x / y Gauss rule cut at every crossing with every wall of both
maps, both maps inverted by its own Newton, events at every vertex image,
y-extremum (square-root substitution) and wall-wall intersection).  It is
restricted to pairs WITHOUT singular vertices (no closed curve), where it
converges to round-off in seconds; the probe version with geometric grading
toward singular vertices is ``verify_e2/v2_brute.py``.

It integrates the H row DIRECTLY -- b's covariant H pulled into a's
coordinates (``S^T``, ``S = J_b^-1 J_a``) and projected in a's PLAIN
``du_a dv_a`` -- so the kernel's ``CrossH = [[X22^H, -X12^H], [-X21^H,
X11^H]]`` (the cofactor of the E row's ``T = J_a^-1 J_b``) is tested, not
assumed.

A sixth test pins the two measured defaults (q-matching, riding) on both
sides without a solve.  Two further tests pin open defects
(``xfail(strict=True)``): V-E2-D1, a grazing cut whose sliver falls between
two of the 65 samples of the pulled-back wall is MISSED (the cross-mass
silently 5.6e-9 off at a 1e-6 graze, the adaptive rule reporting 8e-15);
V-E2-D2, two CROSSING circles are refused in 46 of 56 sampled layouts (a
cell pair touches a 45-degree point of both maps).

Fixture: square period 1.2, M = 4 (3 for the stack-size kernel), the 2 x 2
sinusoid wall maps of ``SinusoidalWall``.  Every bar is a measurement of
2026-10-03 (Windows 11, CPython 3.14.6, numpy 2.4.4, scipy 1.17.1, BLAS 1
thread), stated with its gap on both sides.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import functools  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from numpy.polynomial import legendre as _leg  # noqa: E402

from lumenairy.elements.pmm import (  # noqa: E402
    Circle,
    SinusoidalWall,
    compile_shapes,
)
from lumenairy.elements.pmm import _curvemortar as CMM  # noqa: E402
from lumenairy.elements.pmm import twod_staggered as TS  # noqa: E402

_P = 1.2


# --------------------------------------------------------------- fixtures
@functools.lru_cache(maxsize=None)
def _map(kind, *args):
    if kind == "sinx":
        shp = [SinusoidalWall("x", args[0], args[1], eps=2.25)]
    elif kind == "siny":
        shp = [SinusoidalWall("y", args[0], args[1], phase=args[2],
                              eps=2.25)]
    else:
        shp = [Circle(args[0], args[1], args[2], 3.2)]
    return compile_shapes(_P, _P, shp, 1.0)[3]


def _grid(cm, M, walls=None):
    if cm is None:
        return TS.StagGridOps(_P, _P, np.asarray(walls[0], float),
                              np.asarray(walls[1], float), M, 1.0, 1.0)
    return TS.StagGridOps(_P, _P, cm.u_walls, cm.v_walls, M, 1.0, 1.0,
                          cmap=cm)


def _kernel(ga, gb):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return CMM.StagCrossOpsMapped(ga, gb)


# ------------------------------------------------- the brute-force oracle
def _lf(M, ref):
    L = [_leg.legval(ref, [0.0] * k + [1.0]) for k in range(M + 1)]
    out = np.empty((M, ref.size))
    out[0], out[1] = 0.5 * (L[0] - L[1]), 0.5 * (L[0] + L[1])
    for a in range(2, M):
        out[a] = L[a] - L[a - 2]
    return out


def _vals(g, comp, U, V, sx, sy):
    """every global function of component ``comp`` (0: B(u) x Bt(v), 1:
    Bt(u) x B(v)) at points of cell (sx, sy): (qq, n), index iv qx + iu."""
    def one(b, which, c, s):
        S = np.asarray(b.B if which == "B" else b.Btilde)
        xl, xr = b.xb[s], b.xb[s + 1]
        return S[:, s, :] @ _lf(b.M, (2.0 * c - (xl + xr)) / (xr - xl))
    wu, wv = ("B", "Btilde") if comp == 0 else ("Btilde", "B")
    Fu, Fv = one(g.bx, wu, U, sx), one(g.by, wv, V, sy)
    return (Fv[:, None, :] * Fu[None, :, :]).reshape(-1, U.size)


class _Own:
    """forward map + own Newton inversion / location (identity if None)."""

    def __init__(self, cm, ub, vb):
        self.cm, self.ub, self.vb = cm, np.asarray(ub), np.asarray(vb)
        self.Nx, self.Ny = self.ub.size - 1, self.vb.size - 1
        if cm is not None:
            assert not cm.singular_vertices, "oracle: no singular vertices"

    def fwd(self, sx, sy, U, V):
        if self.cm is None:
            o, z = np.ones_like(U), np.zeros_like(U)
            return U.copy(), V.copy(), o, z, z.copy(), o.copy()
        return self.cm.geom_points(int(sx), int(sy), U, V)

    def inv(self, sx, sy, X, Y):
        if self.cm is None:
            return X.copy(), Y.copy(), np.zeros(X.size)
        U = np.full(X.size, 0.5 * (self.ub[sx] + self.ub[sx + 1]))
        V = np.full(X.size, 0.5 * (self.vb[sy] + self.vb[sy + 1]))
        for _ in range(60):
            g = self.fwd(sx, sy, U, V)
            fx, fy = g[0] - X, g[1] - Y
            det = g[2] * g[5] - g[3] * g[4]
            U = U - (g[5] * fx - g[3] * fy) / det
            V = V - (-g[4] * fx + g[2] * fy) / det
        g = self.fwd(sx, sy, U, V)
        return U, V, np.hypot(g[0] - X, g[1] - Y)

    def locate(self, X, Y):
        SX = np.full(X.size, -1)
        SY = np.full(X.size, -1)
        for sx in range(self.Nx):
            for sy in range(self.Ny):
                U, V, F = self.inv(sx, sy, X, Y)
                ok = ((F < 1e-12) & (U > self.ub[sx]) & (U < self.ub[sx + 1])
                      & (V > self.vb[sy]) & (V < self.vb[sy + 1]))
                SX[ok], SY[ok] = sx, sy
        assert np.all(SX >= 0), "oracle: a point was not located"
        return SX, SY

    def edges(self):
        return ([("u", k, j) for k in range(1, self.Nx)
                 for j in range(self.Ny)]
                + [("v", k, j) for j in range(1, self.Ny)
                   for k in range(self.Nx)])

    def edge(self, e, t):
        kind, k, j = e
        t = np.atleast_1d(np.asarray(t, float))
        if kind == "u":
            V = self.vb[j] + t * (self.vb[j + 1] - self.vb[j])
            g = self.fwd(min(k, self.Nx - 1), j, np.full(t.shape,
                                                         self.ub[k]), V)
            s = self.vb[j + 1] - self.vb[j]
            return g[0], g[1], g[3] * s, g[5] * s
        U = self.ub[k] + t * (self.ub[k + 1] - self.ub[k])
        g = self.fwd(k, min(j, self.Ny - 1), U, np.full(t.shape,
                                                        self.vb[j]))
        s = self.ub[k + 1] - self.ub[k]
        return g[0], g[1], g[2] * s, g[4] * s


_TT = np.linspace(0.0, 1.0, 801)


def _bisect(f, lo, hi):
    flo = f(lo)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        left = np.sign(fm) == np.sign(flo)
        lo, flo, hi = (np.where(left, mid, lo), np.where(left, fm, flo),
                       np.where(left, hi, mid))
    return 0.5 * (lo + hi)


def _pieces(m, e):
    """monotone-in-y pieces of an edge (split at y-extrema) + extrema y."""
    dY = m.edge(e, _TT)[3]
    cuts, ext = [0.0], []
    for i in np.nonzero(np.sign(dY[:-1]) * np.sign(dY[1:]) < 0)[0]:
        t = float(_bisect(lambda t: m.edge(e, t)[3], np.array([_TT[i]]),
                          np.array([_TT[i + 1]]))[0])
        cuts.append(t)
        ext.append(float(m.edge(e, t)[1][0]))
    cuts.append(1.0)
    return list(zip(cuts[:-1], cuts[1:])), ext


def _intersections(A, ea, B, eb):
    """y of every crossing of two edges: an exact segment-segment test on
    the 801-point polylines (so two crossings closer than any coarse
    stride are both seen), each refined by a 2-D Newton."""
    Xa, Ya = A.edge(ea, _TT)[:2]
    Xb, Yb = B.edge(eb, _TT)[:2]
    P0 = np.c_[Xa[:-1], Ya[:-1]][:, None, :]
    P1 = np.c_[Xa[1:], Ya[1:]][:, None, :]
    Q0 = np.c_[Xb[:-1], Yb[:-1]][None, :, :]
    Q1 = np.c_[Xb[1:], Yb[1:]][None, :, :]

    def orient(a, b, c):
        return ((b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1])
                - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0]))
    hit = np.argwhere((orient(P0, P1, Q0) * orient(P0, P1, Q1) <= 0)
                      & (orient(Q0, Q1, P0) * orient(Q0, Q1, P1) <= 0))
    out = []
    for i, j in hit:
        ta, tb = _TT[i] + 0.5 / 800, _TT[j] + 0.5 / 800
        for _ in range(50):
            xa, ya, dxa, dya = (v[0] for v in A.edge(ea, ta))
            xb, yb, dxb, dyb = (v[0] for v in B.edge(eb, tb))
            det = -dxa * dyb + dxb * dya
            if abs(det) < 1e-300:
                break
            ta += ((xa - xb) * dyb - dxb * (ya - yb)) / det
            tb -= (dxa * (ya - yb) - dya * (xa - xb)) / det
        xa, ya = (v[0] for v in A.edge(ea, ta)[:2])
        xb, yb = (v[0] for v in B.edge(eb, tb)[:2])
        if (np.hypot(xa - xb, ya - yb) < 1e-13 and -1e-12 <= ta <= 1
                + 1e-12 and -1e-12 <= tb <= 1 + 1e-12):
            out.append(float(ya))
    return out


def _brute(ga, gb, n=24):
    """(X, CH): the E row (2 qq_b, 2 qq_a) with weight T / det J_b and the
    H row (2 qq_a, 2 qq_b) with weight S^T / det J_a, in PHYSICAL x, y."""
    A = _Own(ga.cmap, ga.bx.xb, ga.by.xb)
    B = _Own(gb.cmap, gb.bx.xb, gb.by.xb)
    allE = [(A, e) for e in A.edges()] + [(B, e) for e in B.edges()]
    ev = {0.0: False, _P: False}
    for m in (A, B):
        for i in range(m.Nx + 1):
            for j in range(m.Ny + 1):
                y = float(m.fwd(min(i, m.Nx - 1), min(j, m.Ny - 1),
                                np.array([m.ub[i]]),
                                np.array([m.vb[j]]))[1][0])
                ev.setdefault(y, False)
    pieces = {}
    for m, e in allE:
        Y = m.edge(e, _TT)[1]
        if np.ptp(Y) < 1e-14:
            ev.setdefault(float(Y[0]), False)
            pieces[(id(m), e)] = []
            continue
        pcs, ext = _pieces(m, e)
        pieces[(id(m), e)] = pcs
        for y in ext:
            ev[y] = True                          # a y-extremum: sqrt end
    for ea in A.edges():
        for eb in B.edges():
            for y in _intersections(A, ea, B, eb):
                ev.setdefault(y, False)
    ys = sorted(y for y in ev if 0.0 <= y <= _P)
    mrg = []
    for y in ys:
        if mrg and y - mrg[-1][0] < 1e-13:
            mrg[-1] = (mrg[-1][0], mrg[-1][1] or ev[y])
        else:
            mrg.append((y, ev[y]))
    x, w = _leg.leggauss(n)
    x, w = 0.5 * (x + 1.0), 0.5 * w
    oy, ow = [], []
    for (y0, s0), (y1, s1) in zip(mrg[:-1], mrg[1:]):
        parts = ([(y0, 0.5 * (y0 + y1), s0, False),
                  (0.5 * (y0 + y1), y1, False, s1)] if s0 and s1
                 else [(y0, y1, s0, s1)])
        for a, b, sa, sb in parts:
            L = b - a
            if sa:
                oy.append(a + L * x * x)
                ow.append(2.0 * L * x * w)
            elif sb:
                oy.append(b - L * x * x)
                ow.append(2.0 * L * x * w)
            else:
                oy.append(a + L * x)
                ow.append(L * w)
    oy, ow = np.concatenate(oy), np.concatenate(ow)
    cr = [[0.0, _P] for _ in oy]
    for m, e in allE:
        for t0, t1 in pieces[(id(m), e)]:
            y0 = float(m.edge(e, t0)[1][0])
            y1 = float(m.edge(e, t1)[1][0])
            hit = np.nonzero((oy > min(y0, y1)) & (oy < max(y0, y1)))[0]
            if hit.size:
                tg = oy[hit]
                t = _bisect(lambda t, tg=tg: m.edge(e, t)[1] - tg,
                            np.full(hit.size, t0), np.full(hit.size, t1))
                for k, xx in zip(hit, m.edge(e, t)[0]):
                    cr[k].append(float(xx))
    SY, SW, SA, SB = [], [], [], []
    for k in range(oy.size):
        c = np.unique(np.array(cr[k]))
        for a, b in zip(c[:-1], c[1:]):
            if b - a > 1e-14:
                SY.append(oy[k])
                SW.append(ow[k])
                SA.append(a)
                SB.append(b)
    SY, SW, SA, SB = map(np.asarray, (SY, SW, SA, SB))
    mid = 0.5 * (SA + SB)
    asx, asy = A.locate(mid, SY)
    bsx, bsy = B.locate(mid, SY)
    NX = (SA[:, None] + (SB - SA)[:, None] * x[None, :]).ravel()
    NY = np.repeat(SY, n)
    NW = (SW * (SB - SA))[:, None] * w[None, :]
    NW = NW.ravel()
    cells = np.stack([np.repeat(v, n) for v in (asx, asy, bsx, bsy)], 1)
    qa, qb = ga.qq, gb.qq
    X = np.zeros((2 * qb, 2 * qa), complex)
    CH = np.zeros((2 * qa, 2 * qb), complex)
    for key in {tuple(r) for r in cells.tolist()}:
        s = np.all(cells == key, axis=1)
        xx, yy, ww = NX[s], NY[s], NW[s]
        Ua, Va, Fa = A.inv(key[0], key[1], xx, yy)
        Ub, Vb, Fb = B.inv(key[2], key[3], xx, yy)
        assert max(Fa.max(), Fb.max()) < 1e-13
        g1 = A.fwd(key[0], key[1], Ua, Va)
        g2 = B.fwd(key[2], key[3], Ub, Vb)
        Ja = np.array([[g1[2], g1[3]], [g1[4], g1[5]]])
        Jb = np.array([[g2[2], g2[3]], [g2[4], g2[5]]])
        da = Ja[0, 0] * Ja[1, 1] - Ja[0, 1] * Ja[1, 0]
        db = Jb[0, 0] * Jb[1, 1] - Jb[0, 1] * Jb[1, 0]
        Jai = np.array([[Ja[1, 1], -Ja[0, 1]], [-Ja[1, 0], Ja[0, 0]]]) / da
        Jbi = np.array([[Jb[1, 1], -Jb[0, 1]], [-Jb[1, 0], Jb[0, 0]]]) / db
        T = np.einsum("ijn,jkn->ikn", Jai, Jb)
        S = np.einsum("ijn,jkn->ikn", Jbi, Ja)
        fa = [_vals(ga, c, Ua, Va, key[0], key[1]) for c in (0, 1)]
        fb = [_vals(gb, c, Ub, Vb, key[2], key[3]) for c in (0, 1)]
        for be in (0, 1):
            for al in (0, 1):
                X[be * qb:(be + 1) * qb, al * qa:(al + 1) * qa] += (
                    (np.conj(fb[be]) * (ww * T[al, be] / db)) @ fa[al].T)
        for k in (0, 1):                 # H_k lives in V_(1-k)
            for m in (0, 1):
                CH[k * qa:(k + 1) * qa, m * qb:(m + 1) * qb] += (
                    (np.conj(fa[1 - k]) * (ww * S[m, k] / da))
                    @ fb[1 - m].T)
    return X, CH


def _rel(A_, B_):
    return float(np.max(np.abs(A_ - B_)) / np.max(np.abs(B_)))


@functools.lru_cache(maxsize=None)
def _pair(name):
    if name == "sinx_siny":
        return (_grid(_map("sinx", 0.55, 0.12), 4),
                _grid(_map("siny", 0.62, 0.10, 0.4), 4))
    off = {"sx_tan": 0.0, "sx_graze6": 1e-6}[name]
    return (_grid(_map("sinx", 0.55, 0.12), 4),
            _grid(None, 4, ([0.0, 0.67 - off, _P], [0.0, 0.5, _P])))


@functools.lru_cache(maxsize=None)
def _oracle(name):
    ga, gb = _pair(name)
    return _brute(ga, gb)


# ================================================================ tests
def test_ve2_1_cross_mass_equals_the_physical_brute_force_both_rows(
        monkeypatch):
    """Two CROSSING sinusoid walls (x-wall over y-wall; the y-wall's crests
    are tangent to the x-wall map's inner direction).  Measured: E row
    2.0e-15, H row (its own physical formula) 2.0e-15 -- bar 1e-12, 2.7
    decades over.  Fail-before: the tangency search off moves the kernel
    by 1.9e-2 (bar 1e-4); the H row without its off-diagonal cofactor
    blocks misses the oracle by 8.8e-2 (bar 1e-2)."""
    ga, gb = _pair("sinx_siny")
    Xb, Hb = _oracle("sinx_siny")
    cr = _kernel(ga, gb)
    assert _rel(cr.EH, Xb) <= 1e-12
    assert _rel(cr.H, Hb) <= 1e-12
    qa, qb = ga.qq, gb.qq
    Hn = np.zeros_like(cr.H)
    Hn[:qa, :qb] = cr.EH[qb:, qa:].conj().T
    Hn[qa:, qb:] = cr.EH[:qb, :qa].conj().T
    assert _rel(Hn, Hb) >= 1e-2
    monkeypatch.setattr(CMM, "_tangencies", lambda *a, **k: [])
    assert _rel(CMM.curved_cross_mass(ga, gb, 27), Xb) >= 1e-4


def test_ve2_2_wall_tangent_to_an_outline_is_exact():
    """An unmapped x-wall TOUCHING the sinusoid's crest (x = 0.67 at
    y = 0.3).  Measured 3.3e-15 (E) / 3.4e-15 (H) against the oracle; bar
    1e-12."""
    ga, gb = _pair("sx_tan")
    Xb, Hb = _oracle("sx_tan")
    cr = _kernel(ga, gb)
    assert _rel(cr.EH, Xb) <= 1e-12
    assert _rel(cr.H, Hb) <= 1e-12


@pytest.mark.xfail(strict=True, reason=(
    "V-E2-D1: _cell_pieces samples the pulled-back wall at 65 points and "
    "drops a piece that lies between two samples -- a grazing cut of "
    "1e-6 (sliver 1.6e-3 long) is missed: 5.6e-9 off, adaptive change "
    "8e-15 (verify_e2/v2d_graze_sweep_win.json)"))
def test_ve2_3_grazing_cut_is_not_lost_between_samples():
    """The x-wall 1e-6 inside the crest: measured 5.56e-9 (sliver
    missed; v2_brute's graded oracle agrees with this one to 1e-15 on the
    pair family); with 8x the wall samples 2.4e-15.  Bar 1e-12."""
    ga, gb = _pair("sx_graze6")
    Xb, _Hb = _oracle("sx_graze6")
    assert _rel(_kernel(ga, gb).EH, Xb) <= 1e-12


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason=(
    "V-E2-D2: two CROSSING circles are refused when a cell pair touches a "
    "45-degree point of both maps -- 46 of 56 sampled crossing layouts "
    "(verify_e2/v2c_circle_pairs_win.json), not only 'nearly coincident "
    "45-degree points' as BUILD_E2 3.5 states"))
def test_ve2_4_two_crossing_circles_have_a_cross_mass():
    """A circle r 0.30 at the centre over r 0.26 offset (0.12, 0.07): their
    nearest 45-degree points are 0.10 apart; refused today."""
    ga = _grid(_map("circ", 0.6, 0.6, 0.30), 3)
    gb = _grid(_map("circ", 0.72, 0.67, 0.26), 3)
    cr = _kernel(ga, gb)
    assert np.all(np.isfinite(cr.EH))


def test_ve2_5_square_root_substitution_keeps_the_rule_under_its_cap(
        monkeypatch):
    """A circle (r 0.33 at (0.55, 0.62)) over a crossing sinusoid wall
    (x0 0.66, A 0.1, phase 0.7) at M = 3: its pulled-back wall has tangency
    ends.  Measured: n = 59, last change 6.1e-14, no warning.  Fail-before:
    the substitution off runs to the 96-node cap (1.4e-12, warns).  Bar
    n <= 80 and change <= 1e-12."""
    ga = _grid(_map("circ", 0.55, 0.62, 0.33), 3)
    gb = _grid(compile_shapes(_P, _P, [SinusoidalWall(
        "x", 0.66, 0.1, phase=0.7, eps=1.9)], 1.0)[3], 3)

    def run():
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter("always")
            _X, n, chg = CMM.curved_cross_mass_adaptive(ga, gb)
        return n, chg, len(wl)
    n, chg, nw = run()
    assert n <= 80 and chg <= 1e-12 and nw == 0
    o = CMM._outer_rule
    monkeypatch.setattr(CMM, "_outer_rule",
                        lambda a, b, s0, s1, k: o(a, b, False, False, k))
    n2, _c2, _w2 = run()
    assert n2 > 80


def _defaults_stack(layers, M=4):
    from lumenairy.elements.pmm import PMM2DStackPure
    st = PMM2DStackPure(_P, _P, n_superstrate=1.0, n_substrate=1.45,
                        n_modes=M, n_orders=3, layer_grids="per-layer")
    for t, shp, kw in layers:
        if shp is None:
            st.add_layer(t, **kw)
        else:
            st.add_layer(t, shapes=shp, background_eps=1.0, **kw)
    return st


def test_ve2_6_q_matching_and_riding_are_two_sided():
    """The two measured defaults, both sides, without a solve (verifier
    item 5, ``verify_e2/v5_checks_win.json``).  q-MATCHING: a sinusoid
    shape layer (2 x 2 map) under a circle (3 x 3) at M = 4 takes M 6 (q 10
    against the circle's 9 -> ceil(9 / 2) + 1); naming ``n_modes`` exempts
    it ([4, 4]); a vacuum-painted (homogeneous) shape layer is not matched.
    RIDING: a uniform layer between the circle and the sinusoid rides the
    layer ABOVE (its map, walls and M); naming ``n_modes`` keeps it on its
    own unmapped grid.  The fast path is taken for a mergeable stack and
    left when one layer names ``n_modes``.  Mutants that survive all 12
    E2 ids -- q-matching over-applied to a named layer, riding over-applied
    to a named uniform layer, riding forced below -- fail here."""
    circ = [Circle(0.6, 0.6, 0.36, 4.0)]

    def sinw(x0=0.6, A=0.12, eps=2.25):
        return [SinusoidalWall("x", x0, A, eps=eps)]
    st = _defaults_stack([(0.3, circ, {}), (0.25, sinw(), {})])
    assert st._perlayer_modal_counts() == [4, 6]
    st = _defaults_stack([(0.3, circ, {}), (0.25, sinw(), {"n_modes": 4})])
    assert st._perlayer_modal_counts() == [4, 4]
    st = _defaults_stack([(0.25, sinw(eps=1.0), {}), (0.3, circ, {})])
    Ms = st._perlayer_modal_counts()
    assert Ms == [4, 4]
    geo = st._perlayer_geometry(Ms)
    assert geo[0][3] is st._layers[1]["cmap"]
    # a uniform layer between two maps rides the one ABOVE
    st = _defaults_stack([(0.3, circ, {}), (0.1, None, {"eps": 1.7}),
                          (0.25, sinw(), {})])
    geo = st._perlayer_geometry(st._perlayer_modal_counts())
    assert geo[1][3] is st._layers[0]["cmap"] and geo[1][2] == 4
    st = _defaults_stack([(0.3, circ, {}), (0.1, None, {"eps": 1.7,
                                                        "n_modes": 4}),
                          (0.25, sinw(), {})])
    geo = st._perlayer_geometry(st._perlayer_modal_counts())
    assert geo[1][3] is None
    # the fast path: taken for a mergeable pair, left when n_modes is named
    st = _defaults_stack([(0.3, circ, {}), (0.25, sinw(0.12, 0.05), {})])
    assert st._perlayer_fast_ok()
    st = _defaults_stack([(0.3, circ, {"n_modes": 4}),
                          (0.25, sinw(0.12, 0.05), {})])
    assert not st._perlayer_fast_ok()
