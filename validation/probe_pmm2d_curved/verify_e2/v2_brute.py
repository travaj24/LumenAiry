"""V2 -- an INDEPENDENT brute-force cross-mass of the curved mortar, in
PHYSICAL coordinates (Phase E2 verifier, item 2).

What the mortar must integrate (re-derived here, not read from the kernel):
the unknowns are COVARIANT components ``E' = J^T E``, ``H' = J^T H`` (1-forms,
``J = d(x, y)/d(u, v)``).  Between two layers on maps ``Phi_a`` (upper) and
``Phi_b`` (lower):

* E row: a's tangential E, pulled into b's coordinates, ``E'_b = T^T E'_a``
  with ``T = J_a^-1 J_b``, L2-projected onto b's E space in b's PLAIN
  ``du_b dv_b`` (each side's own Gram is its plain Gram, so the pairing is the
  metric-free flux ``INT (E x h*) . z dx dy``):

      X_{beta alpha}[i, j] = INT conj(f^b_{beta,i}) T_{alpha beta} f^a_{alpha,j}
                             du_b dv_b
                           = INT conj(f^b_{beta,i}) T_{alpha beta} f^a_{alpha,j}
                             / det J_b  dx dy                 (PHYSICAL form)

* H row: b's tangential H, pulled into a's coordinates, ``H'_a = S^T H'_b``
  with ``S = J_b^-1 J_a = T^-1``, L2-projected onto a's H space (H1 in V2,
  H2 in V1) in a's PLAIN ``du_a dv_a``:

      CrossH[k, l] = INT conj(g^a_k) S_{lk} g^b_l / det J_a  dx dy

  -- a DIFFERENT Jacobian factor per block from the E row (S^T / det J_a
  against T / det J_b); written in b's coordinates it is the cofactor
  ``det(T) T^-T`` of T, which is what the kernel's
  ``CrossH = [[X22^H, -X12^H], [-X21^H, X11^H]]`` encodes.  This probe
  integrates the H row DIRECTLY (its own formula), so it tests that identity
  instead of assuming it.

The brute force: an iterated integral, OUTER over physical ``y``, INNER over
physical ``x``; the inner line is cut at every crossing with every wall of
BOTH maps (own root-finding on the forward maps), each sub-interval is
located in both maps once (own damped Newton from a sample table) and every
node is inverted by its own Newton; the outer variable is broken at every
y-event (grid-vertex images, y-extrema of every wall curve -- with the
``y = y0 + L xi^2`` substitution there --, wall-wall intersections of the two
maps, singular vertices -- with geometric grading there, and the inner
sub-intervals near a singular vertex graded toward it).  The only library
pieces used are the FORWARD maps (``geom_points``), the basis STENCILS
(``Basis1D.B`` / ``.Btilde``: the definition of the basis), and the kernel
under test for the comparison.

Usage: python v2_brute.py <case> [M] [n_gauss]
  case: circ_sin | sin_circ | tangent | graze3 | graze6 | circ_sin_tau |
        tan_x | circ_circ
"""
import sys
import time

import numpy as np
from _ve import dump
from numpy.polynomial import legendre as npleg
from scipy.spatial import cKDTree

from lumenairy.elements.pmm import _curvemortar as CMOR, twod_staggered as TS
from lumenairy.elements.pmm.shapes2d import Circle, SinusoidalWall, _merge

P = 1.2
import os as _os

IFL = float(_os.environ.get('V2_IFL', '0.05'))
LEV = int(_os.environ.get('V2_LEV', '18'))


# --------------------------------------------------------------- own basis
def local_funcs(M, ref):
    """modified-Legendre local functions (own evaluation, numpy legval)."""
    ref = np.asarray(ref, dtype=float)
    L = [npleg.legval(ref, [0.0] * k + [1.0]) for k in range(M + 1)]
    out = np.empty((M, ref.size))
    out[0] = 0.5 * (L[0] - L[1])
    out[1] = 0.5 * (L[0] + L[1])
    for a in range(2, M):
        out[a] = L[a] - L[a - 2]
    return out


def basis_vals(basis, which, coord, seg):
    """values of every global function of set ``which`` of ``basis`` at
    ``coord`` (all in segment ``seg``): (dim, n)."""
    S = np.asarray(basis.B if which == "B" else basis.Btilde)   # (dim,N,M)
    xl, xr = basis.xb[seg], basis.xb[seg + 1]
    ref = (2.0 * coord - (xl + xr)) / (xr - xl)
    F = local_funcs(basis.M, ref)                              # (M, n)
    return np.einsum("im,mn->in", S[:, seg, :], F)


def comp_vals(g, comp, U, V, sx, sy):
    """2-D global functions of component ``comp`` (0: V1 = B(u) x Bt(v),
    1: V2 = Bt(u) x B(v)) at points in cell (sx, sy): (qq, n), index
    iv * qx + iu."""
    wu, wv = ("B", "Btilde") if comp == 0 else ("Btilde", "B")
    Fu = basis_vals(g.bx, wu, U, sx)
    Fv = basis_vals(g.by, wv, V, sy)
    return (Fv[:, None, :] * Fu[None, :, :]).reshape(-1, U.size)


# ------------------------------------------------------------- own maps
class OwnMap:
    """Forward map + own inversion/location.  cmap None = identity."""

    def __init__(self, cmap, ub, vb):
        self.cmap = cmap
        self.ub = np.asarray(ub, float)
        self.vb = np.asarray(vb, float)
        self.Nx, self.Ny = self.ub.size - 1, self.vb.size - 1
        self.sing = []
        if cmap is None:
            return
        for sx, sy, cu, cv in (getattr(cmap, "singular_vertices", None)
                               or ()):
            u, v = self.ub[sx + cu], self.vb[sy + cv]
            X, Y = self.fwd(sx, sy, np.array([u]), np.array([v]))[:2]
            self.sing.append((float(X[0]), float(Y[0])))
        # sample tables (dense, interior + boundary) for Newton guesses
        # INTERIOR samples (a corner may be singular: det J = 0 there), dense
        # plus geometrically graded toward both ends (near-corner guesses)
        gq = 10.0 ** -np.arange(1.0, 11.0)
        t = np.unique(np.r_[np.linspace(0.0, 1.0, 61)[1:-1], gq, 1.0 - gq])
        self.tab = {}
        self.bbox = {}
        for sx in range(self.Nx):
            for sy in range(self.Ny):
                U = self.ub[sx] + t * (self.ub[sx + 1] - self.ub[sx])
                V = self.vb[sy] + t * (self.vb[sy + 1] - self.vb[sy])
                UU, VV = (a.ravel() for a in np.meshgrid(U, V,
                                                        indexing="ij"))
                X, Y = self.fwd(sx, sy, UU, VV)[:2]
                self.tab[(sx, sy)] = (UU, VV, cKDTree(np.c_[X, Y]))
                self.bbox[(sx, sy)] = (X.min(), X.max(), Y.min(), Y.max())

    def fwd(self, sx, sy, U, V):
        U = np.asarray(U, float)
        V = np.asarray(V, float)
        if self.cmap is None:
            o, z = np.ones_like(U), np.zeros_like(U)
            return U.copy(), V.copy(), o, z, z.copy(), o.copy()
        return self.cmap.geom_points(int(sx), int(sy), U, V)

    def newton(self, sx, sy, X, Y, U, V, it=200):
        U = U.copy()
        V = V.copy()
        g = self.fwd(sx, sy, U, V)
        F = np.hypot(g[0] - X, g[1] - Y)
        for _ in range(it):
            act = F > 2e-16 * P
            if not np.any(act):
                break
            a = np.nonzero(act)[0]
            fx, fy = g[0][a] - X[a], g[1][a] - Y[a]
            xu, xv, yu, yv = g[2][a], g[3][a], g[4][a], g[5][a]
            det = xu * yv - xv * yu
            with np.errstate(all="ignore"):
                du = (yv * fx - xv * fy) / det
                dv = (-yu * fx + xu * fy) / det
            du = np.nan_to_num(du)
            dv = np.nan_to_num(dv)
            lam = np.ones(a.size)
            Un, Vn = U[a] - du, V[a] - dv
            for _h in range(30):
                gn = self.fwd(sx, sy, Un, Vn)
                Fn = np.hypot(gn[0] - X[a], gn[1] - Y[a])
                bad = ~(Fn < F[a])
                if not np.any(bad):
                    break
                lam[bad] *= 0.5
                Un = np.where(bad, U[a] - lam * du, Un)
                Vn = np.where(bad, V[a] - lam * dv, Vn)
            gn = self.fwd(sx, sy, Un, Vn)
            Fn = np.hypot(gn[0] - X[a], gn[1] - Y[a])
            imp = Fn < F[a]
            if not np.any(imp):
                break
            ai = a[imp]
            U[ai], V[ai], F[ai] = Un[imp], Vn[imp], Fn[imp]
            g = self.fwd(sx, sy, U, V)
        return U, V, F

    def invert_in(self, sx, sy, X, Y):
        if self.cmap is None:
            return X.copy(), Y.copy(), np.zeros(X.size)
        UU, VV, tree = self.tab[(sx, sy)]
        _d, k = tree.query(np.c_[X, Y])
        U, V, F = self.newton(sx, sy, X, Y, UU[k], VV[k])
        bad = np.nonzero(F > 1e-13 * P)[0]
        if bad.size:
            # retry from the 24 nearest samples, keep the best converged
            _d, kk = tree.query(np.c_[X[bad], Y[bad]], k=6)
            for j in range(kk.shape[1]):
                still = F[bad] > 1e-13 * P
                if not np.any(still):
                    break
                b = bad[still]
                Ut, Vt, Ft = self.newton(sx, sy, X[b], Y[b], UU[kk[still, j]],
                                         VV[kk[still, j]])
                hu = self.ub[sx + 1] - self.ub[sx]
                hv = self.vb[sy + 1] - self.vb[sy]
                ins = ((Ut > self.ub[sx] - 1e-9 * hu)
                       & (Ut < self.ub[sx + 1] + 1e-9 * hu)
                       & (Vt > self.vb[sy] - 1e-9 * hv)
                       & (Vt < self.vb[sy + 1] + 1e-9 * hv))
                imp = (Ft < F[b]) & ins
                U[b[imp]], V[b[imp]], F[b[imp]] = Ut[imp], Vt[imp], Ft[imp]
        return U, V, F

    def locate(self, X, Y):
        """(sx, sy, U, V) per point (points strictly inside a cell)."""
        X = np.asarray(X, float)
        Y = np.asarray(Y, float)
        n = X.size
        SX = np.full(n, -1)
        SY = np.full(n, -1)
        U = np.zeros(n)
        V = np.zeros(n)
        if self.cmap is None:
            SX = np.clip(np.searchsorted(self.ub, X, "right") - 1, 0,
                         self.Nx - 1)
            SY = np.clip(np.searchsorted(self.vb, Y, "right") - 1, 0,
                         self.Ny - 1)
            return SX, SY, X.copy(), Y.copy()
        best = np.full(n, np.inf)
        for (sx, sy), (x0, x1, y0, y1) in self.bbox.items():
            m = 1e-3
            c = np.nonzero((X >= x0 - m) & (X <= x1 + m) & (Y >= y0 - m)
                           & (Y <= y1 + m))[0]
            if c.size == 0:
                continue
            Uc, Vc, F = self.invert_in(sx, sy, X[c], Y[c])
            hu = self.ub[sx + 1] - self.ub[sx]
            hv = self.vb[sy + 1] - self.vb[sy]
            # distance outside the cell in reference units (0 inside)
            out = np.maximum.reduce([
                np.zeros(c.size), (self.ub[sx] - Uc) / hu,
                (Uc - self.ub[sx + 1]) / hu, (self.vb[sy] - Vc) / hv,
                (Vc - self.vb[sy + 1]) / hv])
            score = out + (F > 1e-12 * P) * 1e3
            better = score < best[c]
            ci = c[better]
            best[ci] = score[better]
            SX[ci], SY[ci] = sx, sy
            U[ci], V[ci] = Uc[better], Vc[better]
        if np.any(best > 1e-9):
            raise RuntimeError(f"own locate failed: {np.max(best)}")
        return SX, SY, U, V

    # walls: interior edges as callables tau -> (X, Y, dX, dY)
    def edges(self):
        out = []
        for k in range(1, self.Nx):
            for j in range(self.Ny):
                out.append(("u", k, j))
        for j in range(1, self.Ny):
            for k in range(self.Nx):
                out.append(("v", k, j))
        return out

    def edge_eval(self, e, tau):
        kind, k, j = e
        tau = np.asarray(tau, float)
        if kind == "u":
            cell = (min(k, self.Nx - 1), j)
            U = np.full(tau.shape, self.ub[k])
            V = self.vb[j] + tau * (self.vb[j + 1] - self.vb[j])
            g = self.fwd(*cell, U, V)
            s = self.vb[j + 1] - self.vb[j]
            return g[0], g[1], g[3] * s, g[5] * s
        cell = (k, min(j, self.Ny - 1))
        U = self.ub[k] + tau * (self.ub[k + 1] - self.ub[k])
        V = np.full(tau.shape, self.vb[j])
        g = self.fwd(*cell, U, V)
        s = self.ub[k + 1] - self.ub[k]
        return g[0], g[1], g[2] * s, g[4] * s

    def vertices(self):
        pts = []
        for i in range(self.Nx + 1):
            for j in range(self.Ny + 1):
                sx, sy = min(i, self.Nx - 1), min(j, self.Ny - 1)
                X, Y = self.fwd(sx, sy, np.array([self.ub[i]]),
                                np.array([self.vb[j]]))[:2]
                pts.append((float(X[0]), float(Y[0])))
        return pts


# --------------------------------------------------------- edge geometry
NS = 2001
TS_ = np.linspace(0.0, 1.0, NS)


def edge_samples(m, e):
    X, Y, dX, dY = m.edge_eval(e, TS_)
    return X, Y, dX, dY


def refine_root(fun, lo, hi, it=200):
    """bisection + secant hybrid on a bracket [lo, hi] (fun(lo)*fun(hi)<=0),
    vectorised over arrays lo, hi."""
    lo = lo.copy()
    hi = hi.copy()
    flo = fun(lo)
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        fm = fun(mid)
        left = np.sign(fm) == np.sign(flo)
        lo = np.where(left, mid, lo)
        flo = np.where(left, fm, flo)
        hi = np.where(left, hi, mid)
        if np.max(hi - lo) < 1e-17:
            break
    return 0.5 * (lo + hi)


def edge_y_extrema(m, e):
    X, Y, dX, dY = edge_samples(m, e)
    s = np.sign(dY)
    out = []
    flat = np.max(Y) - np.min(Y) < 1e-14
    if flat:
        return [float(np.mean(Y))], True
    for i in range(NS - 1):
        if s[i] == 0 and 0 < i:
            out.append(float(Y[i]))
        elif s[i] * s[i + 1] < 0:
            t = refine_root(lambda t: m.edge_eval(e, t)[3],
                            np.array([TS_[i]]), np.array([TS_[i + 1]]))
            out.append(float(m.edge_eval(e, t)[1][0]))
    return out, False


def edge_pieces(m, e):
    """monotone-in-y pieces of edge e: list of (t0, t1) split at every
    y-extremum (refined to round-off)."""
    X, Y, dX, dY = edge_samples(m, e)
    s = np.sign(dY)
    cuts = [0.0]
    for i in range(NS - 1):
        if s[i] == 0 and i > 0:
            cuts.append(float(TS_[i]))
        elif s[i] * s[i + 1] < 0:
            t = refine_root(lambda t: m.edge_eval(e, t)[3],
                            np.array([TS_[i]]), np.array([TS_[i + 1]]))
            cuts.append(float(t[0]))
    cuts.append(1.0)
    return [(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if b > a]


_PIECES = {}


def crossings_at(m, e, ys):
    """x of every crossing of edge e with the lines y = ys (list per y), by
    bisection on each MONOTONE piece (so two crossings near an extremum are
    never lost inside one sample interval)."""
    X, Y, dX, dY = edge_samples(m, e)
    res = [[] for _ in ys]
    if np.max(Y) - np.min(Y) < 1e-14:
        return res                     # horizontal edge: an event, no cut
    key = (id(m), e)
    if key not in _PIECES:
        _PIECES[key] = edge_pieces(m, e)
    ys = np.asarray(ys)
    for t0, t1 in _PIECES[key]:
        y0 = float(m.edge_eval(e, np.array([t0]))[1][0])
        y1 = float(m.edge_eval(e, np.array([t1]))[1][0])
        lo, hi = min(y0, y1), max(y0, y1)
        hit = np.nonzero((ys > lo) & (ys < hi))[0]
        if hit.size == 0:
            continue
        tgt = ys[hit]
        t = refine_root(lambda t, tgt=tgt: m.edge_eval(e, t)[1] - tgt,
                        np.full(hit.size, t0), np.full(hit.size, t1))
        xs = m.edge_eval(e, t)[0]
        for k, x in zip(hit, xs):
            res[k].append(float(x))
    return res


def edge_edge_intersections(ma, ea, mb, eb):
    Xa, Ya = edge_samples(ma, ea)[:2]
    Xb, Yb = edge_samples(mb, eb)[:2]
    # coarse polyline intersection (subsample)
    st = 4
    A0 = np.c_[Xa[:-st:st], Ya[:-st:st]]
    A1 = np.c_[Xa[st::st], Ya[st::st]]
    B0 = np.c_[Xb[:-st:st], Yb[:-st:st]]
    B1 = np.c_[Xb[st::st], Yb[st::st]]
    # bounding-box prefilter
    axl = np.minimum(A0[:, 0], A1[:, 0])[:, None]
    axh = np.maximum(A0[:, 0], A1[:, 0])[:, None]
    ayl = np.minimum(A0[:, 1], A1[:, 1])[:, None]
    ayh = np.maximum(A0[:, 1], A1[:, 1])[:, None]
    bxl = np.minimum(B0[:, 0], B1[:, 0])[None, :]
    bxh = np.maximum(B0[:, 0], B1[:, 0])[None, :]
    byl = np.minimum(B0[:, 1], B1[:, 1])[None, :]
    byh = np.maximum(B0[:, 1], B1[:, 1])[None, :]
    mg = 1e-3
    cand = np.argwhere((axl <= bxh + mg) & (bxl <= axh + mg)
                       & (ayl <= byh + mg) & (byl <= ayh + mg))
    out = []
    for i, j in cand:
        ta = TS_[i * st] + 0.5 * st / (NS - 1)
        tb = TS_[j * st] + 0.5 * st / (NS - 1)
        ok = False
        for _ in range(60):
            xa, ya, dxa, dya = (v[0] for v in ma.edge_eval(ea,
                                                          np.array([ta])))
            xb, yb, dxb, dyb = (v[0] for v in mb.edge_eval(eb,
                                                          np.array([tb])))
            fx, fy = xa - xb, ya - yb
            det = -dxa * dyb + dxb * dya
            if abs(det) < 1e-300:
                break
            dta = (-fx * (-dyb) + dxb * (-fy)) / det
            dtb = (dxa * (-fy) - dya * (-fx)) / det
            ta, tb = ta + dta, tb + dtb
            if abs(dta) + abs(dtb) < 1e-15:
                ok = True
                break
        if not ok:
            xa_, ya_ = (v[0] for v in ma.edge_eval(ea, np.array([ta]))[:2])
            xb_, yb_ = (v[0] for v in mb.edge_eval(eb, np.array([tb]))[:2])
            ok = bool(np.hypot(xa_ - xb_, ya_ - yb_) < 1e-14)
        if ok and -1e-12 <= ta <= 1 + 1e-12 and -1e-12 <= tb <= 1 + 1e-12:
            out.append(float(ma.edge_eval(ea, np.array([ta]))[1][0]))
    return out


# -------------------------------------------------------------- the rule
def graded(a, b, toward_a, toward_b, n, levels, sig=0.15):
    """Gauss nodes/weights on [a, b], geometrically graded toward the ends
    flagged."""
    x, w = npleg.leggauss(n)
    x = 0.5 * (x + 1.0)
    w = 0.5 * w
    L = b - a
    if L <= 0:
        return np.zeros(0), np.zeros(0)
    brk = [0.0, 1.0]
    if toward_a and toward_b:
        brk = [0.0, 0.5, 1.0]
    pts = set(brk)
    if toward_a:
        hi = 0.5 if toward_b else 1.0
        s = hi
        for _ in range(levels):
            s *= sig
            pts.add(s)
    if toward_b:
        lo = 0.5 if toward_a else 0.0
        s = 1.0 - lo
        for _ in range(levels):
            s *= sig
            pts.add(1.0 - s)
    pts = np.array(sorted(pts))
    nodes, wts = [], []
    for p0, p1 in zip(pts[:-1], pts[1:]):
        nodes.append(a + L * (p0 + (p1 - p0) * x))
        wts.append(L * (p1 - p0) * w)
    return np.concatenate(nodes), np.concatenate(wts)


def sqrt_rule(a, b, sq_a, sq_b, n):
    """Gauss on [a, b] with the xi^2 substitution at the flagged ends."""
    x, w = npleg.leggauss(n)
    x = 0.5 * (x + 1.0)
    w = 0.5 * w
    if sq_a and sq_b:
        m = 0.5 * (a + b)
        n1, w1 = sqrt_rule(a, m, True, False, n)
        n2, w2 = sqrt_rule(m, b, False, True, n)
        return np.r_[n1, n2], np.r_[w1, w2]
    L = b - a
    if sq_a:
        return a + L * x * x, 2.0 * L * x * w
    if sq_b:
        return b - L * x * x, 2.0 * L * x * w
    return a + L * x, L * w


def brute_cross(ga, gb, n=20, lev=18, verbose=True):
    """X (E row, (2qq_b, 2qq_a)) and CH (H row, (2qq_a, 2qq_b)) by the
    physical iterated rule."""
    A = OwnMap(ga.cmap, ga.bx.xb, ga.by.xb)
    B = OwnMap(gb.cmap, gb.bx.xb, gb.by.xb)
    maps = [(A, e) for e in A.edges()] + [(B, e) for e in B.edges()]
    sing = A.sing + B.sing
    # ---- y events
    ev = {0.0: "end", P: "end"}
    for m in (A, B):
        for x, y in m.vertices():
            if 0 < y < P:
                ev.setdefault(y, "vertex")
    ext = []
    for m, e in maps:
        ys, flat = edge_y_extrema(m, e)
        for y in ys:
            if 0 < y < P:
                if flat:
                    ev.setdefault(y, "flat")
                else:
                    ext.append(y)
                    ev[y] = "ext"
    for ea in A.edges():
        for eb in B.edges():
            for y in edge_edge_intersections(A, ea, B, eb):
                if 0 < y < P:
                    ev.setdefault(y, "xing")
    for x, y in sing:
        ev[y] = "sing"
    ys = sorted(ev)
    # merge near-duplicates (keep the strongest flag)
    rank = {"end": 0, "vertex": 1, "flat": 1, "xing": 1, "ext": 2,
            "sing": 3}
    merged = []
    for y in ys:
        if merged and y - merged[-1][0] < 1e-13:
            if rank[ev[y]] > rank[merged[-1][1]]:
                merged[-1] = (merged[-1][0], ev[y])
            continue
        merged.append((y, ev[y]))
    # ---- outer nodes
    oy, ow = [], []
    for (y0, f0), (y1, f1) in zip(merged[:-1], merged[1:]):
        g0, g1 = f0 == "sing", f1 == "sing"
        s0, s1 = f0 == "ext", f1 == "ext"
        if g0 or g1:
            # graded toward singular ends; an ext end gets sqrt rule inside
            # the coarse piece -- simplest: grade toward both flagged ends
            yy, ww = graded(y0, y1, g0 or s0, g1 or s1, n, lev)
        else:
            yy, ww = sqrt_rule(y0, y1, s0, s1, n)
            if s0 or s1:
                # and a few grading levels for safety
                yy, ww = _sqrt_graded(y0, y1, s0, s1, n)
        oy.append(yy)
        ow.append(ww)
    oy = np.concatenate(oy)
    ow = np.concatenate(ow)
    if verbose:
        print(f"  events {len(merged)}, outer nodes {oy.size}")
    # ---- crossings per outer node
    cr = [[0.0, P] for _ in oy]
    for m, e in maps:
        res = crossings_at(m, e, oy)
        for k, xs in enumerate(res):
            cr[k].extend(xs)
    qa, qb = ga.qq, gb.qq
    X = np.zeros((2 * qb, 2 * qa), dtype=complex)
    CH = np.zeros((2 * qa, 2 * qb), dtype=complex)
    # ---- inner sub-intervals, nodes
    segs_y, segs_w, segs_a, segs_b = [], [], [], []
    for k in range(oy.size):
        c = np.unique(np.round(np.array(cr[k]), 15))
        for a, b in zip(c[:-1], c[1:]):
            if b - a > 1e-14:
                segs_y.append(oy[k])
                segs_w.append(ow[k])
                segs_a.append(a)
                segs_b.append(b)
    sy_, sw_, sa_, sb_ = (np.array(v) for v in (segs_y, segs_w, segs_a,
                                                 segs_b))
    mid = 0.5 * (sa_ + sb_)
    asx, asy, _au, _av = A.locate(mid, sy_)
    bsx, bsy, _bu, _bv = B.locate(mid, sy_)
    xg, wg = npleg.leggauss(n)
    xg = 0.5 * (xg + 1.0)
    wg = 0.5 * wg
    NY, NW, NSA, NSB, NSAy, NSBy = [], [], [], [], [], []
    for i in range(sy_.size):
        a, b, y = sa_[i], sb_[i], sy_[i]
        ta = tb = False
        dmin = np.inf
        for xs, ysng in sing:
            d = abs(y - ysng)
            if d < 0.2 * P:
                # grade toward the end nearest the singular x
                if a - 0.05 <= xs <= b + 0.05:
                    if abs(xs - a) <= abs(xs - b):
                        ta = True
                    else:
                        tb = True
                    dmin = min(dmin, d)
        if ta or tb:
            Lr = b - a
            lv = int(np.clip(np.ceil(np.log(max(IFL * dmin, 1e-15 * P)
                                            / Lr) / np.log(0.15)), 1, lev))
            xx, ww = graded(a, b, ta, tb, n, lv)
        else:
            xx = a + (b - a) * xg
            ww = (b - a) * wg
        NY.append(np.full(xx.size, y))
        NW.append(ww * sw_[i])
        NSA.append(np.full(xx.size, asx[i]))
        NSAy.append(np.full(xx.size, asy[i]))
        NSB.append(np.full(xx.size, bsx[i]))
        NSBy.append(np.full(xx.size, bsy[i]))
        if i == 0:
            NX = [xx]
        else:
            NX.append(xx)
    NX = np.concatenate(NX)
    NY = np.concatenate(NY)
    NW = np.concatenate(NW)
    NSA = np.concatenate(NSA)
    NSAy = np.concatenate(NSAy)
    NSB = np.concatenate(NSB)
    NSBy = np.concatenate(NSBy)
    if verbose:
        print(f"  inner segments {sy_.size}, nodes {NX.size}")
    maxres = 0.0
    BAD = []
    for (ax, ay) in sorted(set(zip(NSA.tolist(), NSAy.tolist()))):
        for (bx, by) in sorted(set(zip(NSB.tolist(), NSBy.tolist()))):
            sel = np.nonzero((NSA == ax) & (NSAy == ay) & (NSB == bx)
                             & (NSBy == by))[0]
            if sel.size == 0:
                continue
            for c0 in range(0, sel.size, 40000):
                s = sel[c0:c0 + 40000]
                x, y, w = NX[s], NY[s], NW[s]
                Ua, Va, Fa = A.invert_in(ax, ay, x, y)
                Ub, Vb, Fb = B.invert_in(bx, by, x, y)
                maxres = max(maxres, float(np.max(Fa, initial=0)),
                             float(np.max(Fb, initial=0)))
                badn = (Fa > 1e-12 * P) | (Fb > 1e-12 * P)
                if np.any(badn):
                    dsg = min([np.hypot(x[badn] - xs_, y[badn] - ys_).min()
                               for xs_, ys_ in sing] or [np.inf])
                    BAD.append((int(badn.sum()),
                                float(np.abs(w[badn]).sum()), float(dsg),
                                float(max(Fa.max(), Fb.max()))))
                ga_ = A.fwd(ax, ay, Ua, Va)
                gb_ = B.fwd(bx, by, Ub, Vb)
                Ja = np.array([[ga_[2], ga_[3]], [ga_[4], ga_[5]]])
                Jb = np.array([[gb_[2], gb_[3]], [gb_[4], gb_[5]]])
                dJa = Ja[0, 0] * Ja[1, 1] - Ja[0, 1] * Ja[1, 0]
                dJb = Jb[0, 0] * Jb[1, 1] - Jb[0, 1] * Jb[1, 0]
                Jai = np.array([[Ja[1, 1], -Ja[0, 1]],
                                [-Ja[1, 0], Ja[0, 0]]]) / dJa
                Jbi = np.array([[Jb[1, 1], -Jb[0, 1]],
                                [-Jb[1, 0], Jb[0, 0]]]) / dJb
                T = np.einsum("ijn,jkn->ikn", Jai, Jb)
                S = np.einsum("ijn,jkn->ikn", Jbi, Ja)
                fa = [comp_vals(ga, c, Ua, Va, ax, ay) for c in (0, 1)]
                fb = [comp_vals(gb, c, Ub, Vb, bx, by) for c in (0, 1)]
                for be in (0, 1):
                    for al in (0, 1):
                        wt = w * T[al, be] / dJb
                        X[be * qb:(be + 1) * qb, al * qa:(al + 1) * qa] += (
                            (np.conj(fb[be]) * wt) @ fa[al].T)
                # H row: rows a's [H1 (V2); H2 (V1)], cols b's [H1; H2]
                hsp = {0: 1, 1: 0}        # H_k lives in V_(other)
                for k in (0, 1):
                    for l_ in (0, 1):
                        wt = w * S[l_, k] / dJa
                        CH[k * qa:(k + 1) * qa, l_ * qb:(l_ + 1) * qb] += (
                            (np.conj(fa[hsp[k]]) * wt) @ fb[hsp[l_]].T)
    if verbose:
        print(f"  max inversion residual {maxres:.2e}; failed groups {BAD[:8]}")
    return X, CH, dict(nodes=int(NX.size), outer=int(oy.size),
                       events=len(merged), maxres=maxres, bad=BAD)


def _sqrt_graded(y0, y1, s0, s1, n):
    """xi^2 substitution at an extremum end, after splitting a short piece
    off it (so the substituted piece is small)."""
    if s0 and s1:
        m = 0.5 * (y0 + y1)
        a = _sqrt_graded(y0, m, True, False, n)
        b = _sqrt_graded(m, y1, False, True, n)
        return np.r_[a[0], b[0]], np.r_[a[1], b[1]]
    return sqrt_rule(y0, y1, s0, s1, n)


# ----------------------------------------------------------------- cases
def circle_map(r=0.36, c=(0.6, 0.6)):
    U, V, cm, cells, ident, mus = _merge(
        P, P, [("a", [Circle(c[0], c[1], r, 4.0)], 1.0, None)])
    return cm


def sin_map(x0=0.6, A=0.12, axis="x", phase=0.0):
    U, V, cm, cells, ident, mus = _merge(
        P, P, [("b", [SinusoidalWall(axis, x0, A, phase=phase, eps=2.25)],
                1.0, None)])
    return cm


def grid(cm, M, tau=(1.0, 1.0), walls=None):
    if cm is None:
        wx, wy = walls
        return TS.StagGridOps(P, P, np.asarray(wx), np.asarray(wy), M,
                              tau[0], tau[1])
    return TS.StagGridOps(P, P, cm.u_walls, cm.v_walls, M, tau[0], tau[1],
                          cmap=cm)


def blocks_err(Xk, Xb, qa, qb):
    sc = float(np.max(np.abs(Xb)))
    out = {}
    for nm, (r0, r1, c0, c1) in {
            "11": (0, qb, 0, qa), "12": (0, qb, qa, 2 * qa),
            "21": (qb, 2 * qb, 0, qa), "22": (qb, 2 * qb, qa, 2 * qa)}.items():
        out[nm] = float(np.max(np.abs(Xk[r0:r1, c0:c1] - Xb[r0:r1, c0:c1]))
                        / sc)
    out["all"] = float(np.max(np.abs(Xk - Xb)) / sc)
    out["scale"] = sc
    return out


def hblocks_err(Hk, Hb, qa, qb):
    sc = float(np.max(np.abs(Hb)))
    out = {}
    for nm, (r0, r1, c0, c1) in {
            "H11": (0, qa, 0, qb), "H12": (0, qa, qb, 2 * qb),
            "H21": (qa, 2 * qa, 0, qb), "H22": (qa, 2 * qa, qb, 2 * qb)}.items():
        out[nm] = float(np.max(np.abs(Hk[r0:r1, c0:c1] - Hb[r0:r1, c0:c1]))
                        / sc)
    out["all"] = float(np.max(np.abs(Hk - Hb)) / sc)
    return out


if __name__ == "__main__":
    case = sys.argv[1]
    M = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    ng = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    tau = (1.0, 1.0)
    if case == "circ_sin":
        ga, gb = grid(circle_map(), M), grid(sin_map(), M)
    elif case == "sin_circ":
        ga, gb = grid(sin_map(), M), grid(circle_map(), M)
    elif case == "circ_sin_tau":
        tau = (np.exp(0.7j), np.exp(-0.4j))
        ga, gb = grid(circle_map(), M, tau), grid(sin_map(), M, tau)
    elif case == "siny_siny":
        cm = sin_map(0.62, 0.10, axis="y", phase=0.4)
        ga, gb = grid(cm, M), grid(cm, M)
    elif case == "sinx_sinx":
        cm = sin_map(0.55, 0.12)
        ga, gb = grid(cm, M), grid(cm, M)
    elif case.startswith("sx_"):
        # NO singular vertices: the sinusoid x-wall map (x = 0.55 + 0.12
        # sin(2 pi y / P), crest x = 0.67 at y = 0.3) over an UNMAPPED grid
        # whose x-wall TOUCHES the crest (tangent) or grazes it
        off = {"sx_tan": 0.0, "sx_graze3": 1e-3, "sx_graze6": 1e-6,
               "sx_graze9": 1e-9}[case]
        ga = grid(sin_map(0.55, 0.12), M)
        gb = grid(None, M, walls=([0.0, 0.67 - off, 1.2], [0.0, 0.5, 1.2]))
    elif case == "sinx_siny":
        # two sinusoid maps (no singular vertices): x-wall over a y-wall
        ga = grid(sin_map(0.55, 0.12), M)
        gb = grid(sin_map(0.62, 0.10, axis="y", phase=0.4), M)
    elif case == "circ_circ":
        # a circle over an OFF-CENTRE circle that crosses it
        ga = grid(circle_map(0.30, (0.5, 0.55)), M)
        gb = grid(circle_map(0.26, (0.74, 0.66)), M)
    elif case.startswith("tangent") or case.startswith("graze"):
        # the circle (r 0.36 at 0.6) over an UNMAPPED grid whose y-wall
        # touches (tangent) or grazes the circle's bottom (y = 0.24)
        off = {"tangent": 0.0, "graze3": 1e-3, "graze6": 1e-6}[case]
        yw = 0.24 + off
        ga = grid(circle_map(), M)
        gb = grid(None, M, walls=([0.0, 0.45, 1.2], [0.0, yw, 1.2]))
    elif case == "tan_x":
        # the circle over a SINUSOID wall layer whose straight v-wall
        # (y = 0.6) is tangent... and an unmapped x-wall at x = 0.96 (the
        # circle's rightmost point, tangent to the arc at its 0-deg point)
        ga = grid(circle_map(), M)
        gb = grid(None, M, walls=([0.0, 0.96, 1.2], [0.0, 0.6, 1.2]))
    else:
        raise SystemExit(case)
    t0 = time.perf_counter()
    cr = CMOR.StagCrossOpsMapped(ga, gb)
    t_k = time.perf_counter() - t0
    Xk, Hk = cr.EH, cr.H
    print(f"{case} M={M}: kernel n={cr.n} change={cr.change:.1e} "
          f"{t_k:.1f}s")
    t0 = time.perf_counter()
    Xb, Hb, info = brute_cross(ga, gb, n=ng, lev=LEV)
    t_b = time.perf_counter() - t0
    eX = blocks_err(Xk, Xb, ga.qq, gb.qq)
    eH = hblocks_err(Hk, Hb, ga.qq, gb.qq)
    # the H identity applied to the BRUTE X (tests the adjugate identity
    # independently of the kernel's quadrature)
    eHid = hblocks_err(CMOR.cross_h_from_x(Xb, ga.qq, gb.qq), Hb, ga.qq,
                       gb.qq)
    # fail-before: the H row built WITHOUT the cofactor (= X^H block-swapped
    # as if T = I in the H row: blkdiag(X22^H, X11^H) only, off-diagonals 0)
    qa, qb = ga.qq, gb.qq
    Hn = np.zeros_like(Hk)
    Hn[:qa, :qb] = Xk[qb:, qa:].conj().T
    Hn[qa:, qb:] = Xk[:qb, :qa].conj().T
    eHfb = hblocks_err(Hn, Hb, qa, qb)
    print(f"  brute {t_b:.1f}s  E-row {eX}")
    print(f"  H-row kernel vs brute {eH}")
    print(f"  H identity on brute X vs brute H {eHid['all']:.2e}")
    print(f"  fail-before (H without off-diagonal cofactor) {eHfb['all']:.2e}")
    dump(f"v2_brute_{case}_M{M}_n{ng}_L{LEV}_f{IFL:g}",
         dict(case=case, M=M, n_gauss=ng, kernel_n=cr.n,
              kernel_change=cr.change, kernel_wall=t_k, brute_wall=t_b,
              brute_info=info, E_err=eX, H_err=eH, H_identity_on_brute=eHid,
              failbefore_H_no_offdiag=eHfb,
              tau=[complex(t) for t in tau], lev=LEV, ifl=IFL,
              Xb_re=Xb.real, Xb_im=Xb.imag))
