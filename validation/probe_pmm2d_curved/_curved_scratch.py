"""SCRATCH mapped (curvilinear) staggered 2-D PMM -- PROBE CODE ONLY.

Planning probe for docs/audits/PLAN_PMM2D_CURVED_CELLS_2026_09_26.md.  Nothing
here is library code; it COPIES the operator structure of
``lumenairy.elements.pmm.twod_staggered.Granet2DTransverseE._assemble`` (the
MAGNETIC + BLOCK-FORM-TENSOR route) and replaces every piecewise-constant
``eps[sx, sy] * kron(Gy[sy], Gx[sx])`` by a 2-D Gauss-Legendre quadrature
assembly with a weight that varies inside the segment.

Formulation (z-independent in-plane map (x, y) = Phi(u, v), w = z):
  J = d(x, y)/d(u, v),  g = J^T J,  sqrt(g) = det J,
  eps'_t = eps sqrt(g) g^-1,  eps'_33 = eps sqrt(g)
  mu'_t  =     sqrt(g) g^-1,  mu'_33  =     sqrt(g)
  -> chi_t = [mu'_t]^-1 = g / sqrt(g),  chi33 = 1 / sqrt(g)
Unknowns are the COVARIANT components E' = J^T E, H' = J^T H.  The map is the
same in EVERY region (half-spaces included), so the interfaces stay square
modal matches; the only map-aware far-field piece is the Rayleigh extraction:
  det(J) [Ex; Ey] = [[ y_v, -y_u], [-x_v, x_u]] [E'_u; E'_v]
  a_m = (1/A) INT INT det(J) E(Phi(u,v)) exp(+i k_m . Phi(u,v)) du dv
"""
from __future__ import annotations

import os
import time

import numpy as np
import scipy.linalg as sla
from numpy.polynomial.legendre import leggauss

import lumenairy

_ROOT = os.path.normcase(os.path.abspath(r"C:\tmp\lum_curved"))
assert os.path.normcase(os.path.abspath(lumenairy.__file__)).startswith(_ROOT), (
    f"lumenairy imported from {lumenairy.__file__}, not the worktree {_ROOT}")

from lumenairy.elements.pmm._core import (  # noqa: E402
    _forward_branch_flip,
    _guarded_lstsq,
    _interface_smatrix,
    _propagation_smatrix,
    _redheffer_star,
)
from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    Basis1D,
    _inv_lam,
    _modleg_value_deriv,
    _pmm2d_order_kz,
)
from lumenairy.elements.rcwa._core import _project_efficiency  # noqa: E402

_C = np.complex128
HERE = os.path.dirname(os.path.abspath(__file__))

#: Probe-speed switch: solve the Hermitian-PD pencil (L, -R) by Cholesky
#: whitening + a STANDARD eig (zgeev, ~2.5x cheaper than the QZ the shipped
#: in-plane path pays).  Recorded in every JSON via env_record().
WHITEN = os.environ.get("CURVED_PROBE_WHITEN", "1") == "1"


def eig_pencil(L, B):
    """Generalized eig L x = g B x with B Hermitian positive definite."""
    if not WHITEN:
        return sla.eig(L, B)
    B = 0.5 * (B + B.conj().T)
    C = np.linalg.cholesky(B)                            # B = C C^H
    A = sla.solve_triangular(C, L, lower=True)           # C^-1 L
    A = sla.solve_triangular(C, A.conj().T, lower=True).conj().T   # C^-1 L C^-H
    g, Y = sla.eig(A, overwrite_a=True, check_finite=False)
    X = sla.solve_triangular(C.conj().T, Y, lower=False)
    return g, X


# =========================================================================== #
# Maps.  geom(sx, sy, U, V) -> dict of (nqx, nqy) arrays X, Y, xu, xv, yu, yv
# at the tensor grid of PHYSICAL-u points U (nqx,) x V (nqy,) inside cell
# (sx, sy) of the (u, v) wall grid.
# =========================================================================== #
class IdentityMap:
    name = "identity"

    def geom(self, sx, sy, U, V):
        X = np.broadcast_to(U[:, None], (U.size, V.size)).astype(float)
        Y = np.broadcast_to(V[None, :], (U.size, V.size)).astype(float)
        one = np.ones_like(X)
        zero = np.zeros_like(X)
        return dict(X=X, Y=Y, xu=one, xv=zero, yu=zero, yv=one)


class SineStretchX:
    """x = u + a sin(2 pi u / px), y = v  (periodic; monotone iff a < px/(2 pi))."""
    name = "sine_x"

    def __init__(self, a, px):
        self.a = float(a)
        self.px = float(px)
        if not (abs(self.a) * 2 * np.pi / self.px < 1.0):
            raise ValueError("non-monotone stretch")

    def x_of_u(self, u):
        return u + self.a * np.sin(2 * np.pi * u / self.px)

    def u_of_x(self, x):
        x = np.asarray(x, dtype=float)
        u = np.array(x, dtype=float, copy=True)
        for _ in range(200):
            f = self.x_of_u(u) - x
            fp = 1 + self.a * 2 * np.pi / self.px * np.cos(2 * np.pi * u / self.px)
            du = f / fp
            u = u - du
            if np.max(np.abs(du)) < 1e-15:
                break
        return u

    def geom(self, sx, sy, U, V):
        k = 2 * np.pi / self.px
        X = (U + self.a * np.sin(k * U))[:, None] + 0 * V[None, :]
        xu = (1 + self.a * k * np.cos(k * U))[:, None] + 0 * V[None, :]
        Y = 0 * U[:, None] + V[None, :]
        zero = np.zeros_like(X)
        return dict(X=X, Y=Y, xu=xu, xv=zero, yu=zero, yv=np.ones_like(X))


# ---- edge curves on s in [0, 1]: value (n, 2) and derivative d/ds (n, 2) ----
class Line:
    def __init__(self, P, Q):
        self.P = np.asarray(P, float)
        self.Q = np.asarray(Q, float)

    def __call__(self, s):
        s = np.atleast_1d(np.asarray(s, float))
        return (self.P[None, :] + s[:, None] * (self.Q - self.P)[None, :],
                np.broadcast_to((self.Q - self.P)[None, :], (s.size, 2)))


class Arc:
    def __init__(self, c, r, th0, th1):
        self.c = np.asarray(c, float)
        self.r = float(r)
        self.th0 = float(th0)
        self.th1 = float(th1)

    def __call__(self, s):
        s = np.atleast_1d(np.asarray(s, float))
        th = self.th0 + s * (self.th1 - self.th0)
        val = self.c[None, :] + self.r * np.stack([np.cos(th), np.sin(th)], 1)
        der = self.r * (self.th1 - self.th0) * np.stack([-np.sin(th), np.cos(th)], 1)
        return val, der


class TransfiniteMap:
    """Gordon-Hall (bilinearly blended) transfinite map per rectangular (u, v)
    cell of a tensor wall grid.  ``verts[i][j]`` = physical image of grid vertex
    (u_i, v_j); ``curved`` maps ('h', i, j) (edge (i,j)->(i+1,j)) or
    ('v', i, j) (edge (i,j)->(i,j+1)) to a curve whose s=0 end is the
    lower-index vertex.  Every other edge is the straight segment between its
    vertex images.  Shared edges => C0 across cells by construction; the outer
    boundary is identity (periodic)."""
    name = "transfinite"

    def __init__(self, uw, vw, verts, curved):
        self.uw = np.asarray(uw, float)
        self.vw = np.asarray(vw, float)
        self.verts = verts
        self.curved = dict(curved)
        for key, crv in self.curved.items():
            kind, i, j = key
            ends = crv(np.array([0.0, 1.0]))[0]
            P = np.asarray(verts[i][j], float)
            Q = np.asarray(verts[i + 1][j] if kind == "h" else verts[i][j + 1], float)
            if np.max(np.abs(ends[0] - P)) > 1e-12 or np.max(np.abs(ends[1] - Q)) > 1e-12:
                raise ValueError(f"curve {key} endpoints {ends} != vertices {P}, {Q}")

    def edge(self, kind, i, j):
        c = self.curved.get((kind, i, j))
        if c is not None:
            return c
        P = self.verts[i][j]
        Q = self.verts[i + 1][j] if kind == "h" else self.verts[i][j + 1]
        return Line(P, Q)

    def geom(self, sx, sy, U, V):
        u0, u1 = self.uw[sx], self.uw[sx + 1]
        v0, v1 = self.vw[sy], self.vw[sy + 1]
        s = (np.asarray(U, float) - u0) / (u1 - u0)
        t = (np.asarray(V, float) - v0) / (v1 - v0)
        Bv, Bd = self.edge("h", sx, sy)(s)          # bottom (t = 0)
        Tv, Td = self.edge("h", sx, sy + 1)(s)      # top    (t = 1)
        Lv, Ld = self.edge("v", sx, sy)(t)          # left   (s = 0)
        Rv, Rd = self.edge("v", sx + 1, sy)(t)      # right  (s = 1)
        P00 = np.asarray(self.verts[sx][sy], float)
        P10 = np.asarray(self.verts[sx + 1][sy], float)
        P01 = np.asarray(self.verts[sx][sy + 1], float)
        P11 = np.asarray(self.verts[sx + 1][sy + 1], float)
        S = s[:, None, None]
        Tt = t[None, :, None]
        Phi = ((1 - Tt) * Bv[:, None, :] + Tt * Tv[:, None, :]
               + (1 - S) * Lv[None, :, :] + S * Rv[None, :, :]
               - ((1 - S) * (1 - Tt) * P00 + S * (1 - Tt) * P10
                  + (1 - S) * Tt * P01 + S * Tt * P11))
        Ps = ((1 - Tt) * Bd[:, None, :] + Tt * Td[:, None, :]
              - Lv[None, :, :] + Rv[None, :, :]
              - (-(1 - Tt) * P00 + (1 - Tt) * P10 - Tt * P01 + Tt * P11))
        Pt = (-Bv[:, None, :] + Tv[:, None, :]
              + (1 - S) * Ld[None, :, :] + S * Rd[None, :, :]
              - (-(1 - S) * P00 - S * P10 + (1 - S) * P01 + S * P11))
        du = u1 - u0
        dv = v1 - v0
        return dict(X=Phi[..., 0], Y=Phi[..., 1],
                    xu=Ps[..., 0] / du, yu=Ps[..., 1] / du,
                    xv=Pt[..., 0] / dv, yv=Pt[..., 1] / dv)


def circle_map_3x3(P, r):
    """Square cell [0,P]^2, circle radius r at the centre.  3x3 wall grid
    u, v in {0, a, b, P}, a = P/2 - r/sqrt2, b = P/2 + r/sqrt2: every grid
    vertex is IDENTITY-mapped (the four middle-cell corners are the 45-degree
    points of the circle) and only the middle cell's four edges are arcs.
    det J = 0 at those four corners (a 90-degree parameter corner opened onto a
    smooth curve) -- unavoidable for any closed smooth curve built from
    coordinate lines of a TENSOR grid."""
    c = (P / 2, P / 2)
    a = P / 2 - r / np.sqrt(2)
    b = P / 2 + r / np.sqrt(2)
    w = np.array([0.0, a, b, P])
    verts = [[(w[i], w[j]) for j in range(4)] for i in range(4)]
    d = np.pi / 180
    curved = {
        ("h", 1, 1): Arc(c, r, 225 * d, 315 * d),   # bottom of middle cell
        ("h", 1, 2): Arc(c, r, 135 * d, 45 * d),    # top
        ("v", 1, 1): Arc(c, r, 225 * d, 135 * d),   # left
        ("v", 2, 1): Arc(c, r, -45 * d, 45 * d),    # right
    }
    return TransfiniteMap(w, w, verts, curved), w


def circle_map_5x5(P, r, inner=0.5):
    """Circle radius r in the square cell; 5x5 wall grid.  The DISK is the inner
    3x3 block, its centre cell a straight-edged square of half-size
    ``inner * r / sqrt2``; the loop vertices sit at the 45-degree points
    (identity-mapped), the loop-side vertices at 45 +- 30 degrees... chosen so
    each loop edge is a PURE arc.  Same unavoidable det J = 0 at the four loop
    corners; the extra cells only improve the shape of the disk interior."""
    c = np.array([P / 2, P / 2])
    a1 = P / 2 - r / np.sqrt(2)
    h = inner * r / np.sqrt(2)
    a2 = P / 2 - h
    w = np.array([0.0, a1, a2, P - a2, P - a1, P])
    d = np.pi / 180
    # loop-side vertex angles: the parameter vertex at u = a2 on the bottom side
    # maps to the circle point with x = a2 (vertical projection), i.e.
    # theta = 270 - asin(h / r) ... keep x-coordinate identity where possible.
    phi = np.arcsin(h / r)          # angle from the axis

    def circ(th):
        return (c[0] + r * np.cos(th), c[1] + r * np.sin(th))

    def img(i, j):
        x, y = w[i], w[j]
        if i in (1, 4) and j in (1, 4):
            return (x, y)                         # 45-degree points (identity)
        if j in (1, 4) and i in (2, 3):           # bottom / top loop side
            sgn = -1 if i == 2 else 1
            base = 270 * d if j == 1 else 90 * d
            th = base + (sgn * phi if j == 1 else -sgn * phi)
            return circ(th)
        if i in (1, 4) and j in (2, 3):           # left / right loop side
            sgn = -1 if j == 2 else 1
            base = 180 * d if i == 1 else 0.0
            th = base - sgn * phi if i == 1 else base + sgn * phi
            return circ(th)
        return (x, y)
    verts = [[img(i, j) for j in range(6)] for i in range(6)]

    def ang(pt):
        return np.arctan2(pt[1] - c[1], pt[0] - c[0])

    curved = {}

    def arc_between(key, p, q):
        t0, t1 = ang(p), ang(q)
        # shortest signed sweep
        dt = (t1 - t0 + np.pi) % (2 * np.pi) - np.pi
        curved[key] = Arc(c, r, t0, t0 + dt)
    for i in (1, 2, 3):
        arc_between(("h", i, 1), verts[i][1], verts[i + 1][1])
        arc_between(("h", i, 4), verts[i][4], verts[i + 1][4])
    for j in (1, 2, 3):
        arc_between(("v", 1, j), verts[1][j], verts[1][j + 1])
        arc_between(("v", 4, j), verts[4][j], verts[4][j + 1])
    return TransfiniteMap(w, w, verts, curved), w


def fillet_map_5x5(P, half, rf):
    """Square pillar [P/2-half, P/2+half]^2 with fillet radius rf > 0 at the
    four corners.  5x5 wall grid u, v in {0, a, b, P-b, P-a, P} with
    a = P/2 - half + rf (1 - 1/sqrt2) (the 45-degree points of the fillet arcs,
    identity-mapped) and b = P/2 - half + rf (the fillet centres).  The pillar
    is the inner 3x3 block; every cell edge is a PURE arc or a straight line
    (the arc/line tangency points are grid vertices), so the map is analytic
    inside every cell."""
    lo = P / 2 - half
    hi = P / 2 + half
    a = lo + rf * (1 - 1 / np.sqrt(2))
    b = lo + rf
    w = np.array([0.0, a, b, P - b, P - a, P])

    def img(i, j):
        x, y = w[i], w[j]
        if i in (1, 4) and j in (1, 4):
            return (x, y)                          # 45-degree fillet points
        if j in (1, 4) and i in (2, 3):
            return (x, lo if j == 1 else hi)       # bottom/top tangency points
        if i in (1, 4) and j in (2, 3):
            return (lo if i == 1 else hi, y)       # left/right tangency points
        return (x, y)
    verts = [[img(i, j) for j in range(6)] for i in range(6)]
    d = np.pi / 180
    cBL, cBR, cTL, cTR = (b, b), (P - b, b), (b, P - b), (P - b, P - b)
    curved = {
        ("h", 1, 1): Arc(cBL, rf, 225 * d, 270 * d),
        ("h", 3, 1): Arc(cBR, rf, 270 * d, 315 * d),
        ("h", 1, 4): Arc(cTL, rf, 135 * d, 90 * d),
        ("h", 3, 4): Arc(cTR, rf, 90 * d, 45 * d),
        ("v", 1, 1): Arc(cBL, rf, 225 * d, 180 * d),
        ("v", 1, 3): Arc(cTL, rf, 180 * d, 135 * d),
        ("v", 4, 1): Arc(cBR, rf, 315 * d, 360 * d),
        ("v", 4, 3): Arc(cTR, rf, 0 * d, 45 * d),
    }
    return TransfiniteMap(w, w, verts, curved), w


def detJ_range(cmap, uw, vw, n=41):
    """(min, max) of det J over a dense per-cell sample INCLUDING cell corners."""
    lo, hi = np.inf, -np.inf
    for sx in range(len(uw) - 1):
        for sy in range(len(vw) - 1):
            U = np.linspace(uw[sx], uw[sx + 1], n)
            V = np.linspace(vw[sy], vw[sy + 1], n)
            g = cmap.geom(sx, sy, U, V)
            dj = g["xu"] * g["yv"] - g["xv"] * g["yu"]
            lo = min(lo, float(dj.min()))
            hi = max(hi, float(dj.max()))
    return lo, hi


def mapped_area(cmap, uw, vw, cells, nq=40):
    """Physical area of the listed (sx, sy) cells = INT det J du dv (Gauss)."""
    xg, wg = leggauss(nq)
    A = 0.0
    for sx, sy in cells:
        J1 = 0.5 * (uw[sx + 1] - uw[sx])
        J2 = 0.5 * (vw[sy + 1] - vw[sy])
        U = 0.5 * (uw[sx] + uw[sx + 1]) + J1 * xg
        V = 0.5 * (vw[sy] + vw[sy + 1]) + J2 * xg
        g = cmap.geom(sx, sy, U, V)
        dj = g["xu"] * g["yv"] - g["xv"] * g["yu"]
        A += float(np.sum(wg[:, None] * wg[None, :] * dj) * J1 * J2)
    return A


# =========================================================================== #
# Variable-coefficient staggered assembly
# =========================================================================== #
def _axis_factor(basis, s, lset, op, rset, V, Vp, wg):
    """Per-quadrature-point 1-D factor on segment s, restricted to support:
    F[p, i, j] = scale * w_p * (conj(L_i) f)(p) * (R_j g)(p)."""
    SL = np.asarray(getattr(basis, lset))[:, s, :]
    SR = np.asarray(getattr(basis, rset))[:, s, :]
    supL = np.nonzero(np.any(SL != 0, axis=1))[0]
    supR = np.nonzero(np.any(SR != 0, axis=1))[0]
    J = basis.Jn[s]
    if op == "m":
        fa, gc, scale = V, V, J
    elif op == "d":          # derivative on the TRIAL (right) function
        fa, gc, scale = V, Vp, 1.0
    elif op == "dL":         # derivative on the TEST (left) function
        fa, gc, scale = Vp, V, 1.0
    else:
        raise ValueError(op)
    Lp = np.conj(SL[supL]) @ fa          # (nL, nq)
    Rp = SR[supR] @ gc                   # (nR, nq)
    F = scale * wg[:, None, None] * Lp.T[:, :, None] * Rp.T[:, None, :]
    return supL, supR, F


class CurvedGranet:
    """Mapped staggered transverse-E region solver (scalar eps per (u,v) cell).

    ``cmap=None`` is the IDENTITY map evaluated through the SAME quadrature
    path (the P1 comparison arm)."""

    def __init__(self, px, py, uw, vw, M, eps_cell, cmap=None, alpha0x=0.0,
                 alpha0y=0.0, k0=2 * np.pi, nq=None):
        self.k0 = float(k0)
        self.M = int(M)
        taux = np.exp(-1j * alpha0x * px)
        tauy = np.exp(-1j * alpha0y * py)
        self.bx = Basis1D(px, uw, M, taux)
        self.by = Basis1D(py, vw, M, tauy)
        assert self.bx.dim == self.by.dim
        self.q = self.bx.dim
        self.qq = self.q * self.q
        self.eps_cell = np.asarray(eps_cell, dtype=_C)
        assert self.eps_cell.shape == (self.bx.N, self.by.N)
        self.cmap = IdentityMap() if cmap is None else cmap
        self.nq = int(nq) if nq is not None else 2 * self.M + 8
        t0 = time.perf_counter()
        self._weights()
        self._assemble()
        self.t_assemble = time.perf_counter() - t0

    # ---- weights at the 2-D quadrature points of every cell ----
    def _weights(self):
        nq = self.nq
        xg, wg = leggauss(nq)
        self.xg, self.wg = xg, wg
        self.V, self.Vp = _modleg_value_deriv(self.M, xg)
        W = {k: {} for k in ("one", "e11", "e12", "e21", "e22", "e33",
                             "c11", "c12", "c21", "c22", "c33")}
        detmin = np.inf
        for sx in range(self.bx.N):
            U = 0.5 * (self.bx.xb[sx] + self.bx.xb[sx + 1]) + self.bx.Jn[sx] * xg
            for sy in range(self.by.N):
                Vv = 0.5 * (self.by.xb[sy] + self.by.xb[sy + 1]) + self.by.Jn[sy] * xg
                g = self.cmap.geom(sx, sy, U, Vv)
                xu, xv, yu, yv = g["xu"], g["xv"], g["yu"], g["yv"]
                sg = xu * yv - xv * yu                       # sqrt(g) = det J
                detmin = min(detmin, float(sg.min()))
                g11 = xu * xu + yu * yu
                g12 = xu * xv + yu * yv
                g22 = xv * xv + yv * yv
                e = self.eps_cell[sx, sy]
                # eps'_t = eps sqrt(g) g^-1 = eps [[g22, -g12], [-g12, g11]] / sqrt(g)
                W["one"][sx, sy] = np.ones_like(sg)
                W["e11"][sx, sy] = e * g22 / sg
                W["e12"][sx, sy] = -e * g12 / sg
                W["e21"][sx, sy] = -e * g12 / sg
                W["e22"][sx, sy] = e * g11 / sg
                W["e33"][sx, sy] = e * sg
                # chi_t = [mu'_t]^-1 = g / sqrt(g) ;  chi33 = 1 / sqrt(g)
                W["c11"][sx, sy] = g11 / sg
                W["c12"][sx, sy] = g12 / sg
                W["c21"][sx, sy] = g12 / sg
                W["c22"][sx, sy] = g22 / sg
                W["c33"][sx, sy] = 1.0 / sg
        self.W = W
        self.detmin = detmin

    def blk(self, xspec, yspec, wname):
        """Variable-coefficient generalisation of ``_eps_weighted`` /
        ``_eps_dir``: sum over cells of INT INT w(u,v) (x-factor)(y-factor).
        xspec = (lset, op, rset) on x; yspec on y.  Index I = ix + q*iy."""
        q = self.q
        out = np.zeros((q, q, q, q), dtype=_C)     # [iy, ix, jy, jx]
        fx = [_axis_factor(self.bx, sx, xspec[0], xspec[1], xspec[2],
                           self.V, self.Vp, self.wg) for sx in range(self.bx.N)]
        fy = [_axis_factor(self.by, sy, yspec[0], yspec[1], yspec[2],
                           self.V, self.Vp, self.wg) for sy in range(self.by.N)]
        Wd = self.W[wname]
        for sx in range(self.bx.N):
            sLx, sRx, Fx = fx[sx]
            for sy in range(self.by.N):
                sLy, sRy, Fy = fy[sy]
                Wc = Wd[sx, sy]                              # (nq_x, nq_y)
                T = np.einsum("pr,rab->pab", Wc, Fy, optimize=True)
                loc = np.einsum("pij,pab->aibj", Fx, T, optimize=True)
                out[np.ix_(sLy, sLx, sRy, sRx)] += loc
        return out.reshape(q * q, q * q)

    def _assemble(self):
        k0 = self.k0
        qq = self.qq
        B, T = "B", "Btilde"
        # R = C[chi_t]C  (Granet Eq. 24 / A39), magnetic route
        R = np.zeros((2 * qq, 2 * qq), dtype=_C)
        R[:qq, :qq] = -self.blk((B, "m", B), (T, "m", T), "c22")
        R[:qq, qq:] = self.blk((B, "m", T), (T, "m", B), "c21")
        R[qq:, :qq] = self.blk((T, "m", B), (B, "m", T), "c12")
        R[qq:, qq:] = -self.blk((T, "m", T), (B, "m", B), "c11")
        # plain block Gram for the Eq.-25 H recovery (NO chi_t)
        G1 = self.blk((B, "m", B), (T, "m", T), "one")
        G2 = self.blk((T, "m", T), (B, "m", B), "one")
        # [eps_t] (Eq. 40, all four blocks)
        E11 = self.blk((B, "m", B), (T, "m", T), "e11")
        E22 = self.blk((T, "m", T), (B, "m", B), "e22")
        E12 = self.blk((B, "m", T), (T, "m", B), "e12")
        E21 = self.blk((T, "m", B), (B, "m", T), "e21")
        # S_tt = -Curl^H Gw^-1 Gw_chi33 Gw^-1 Curl (metric-free Curl)
        bx, by = self.bx, self.by
        Mbb_x = bx.mass(bx.B, bx.B)
        Mbb_y = by.mass(by.B, by.B)
        dbt_x = bx.mixed(bx.B, bx.Btilde) / k0
        dbt_y = by.mixed(by.B, by.Btilde) / k0
        Gw = np.kron(Mbb_y, Mbb_x)
        Curl = np.concatenate([np.kron(dbt_y, Mbb_x), -np.kron(Mbb_y, dbt_x)],
                              axis=1)
        Gw_inv = np.linalg.inv(Gw)
        Gw_chi = self.blk((B, "m", B), (B, "m", B), "c33")
        Stt = -Curl.conj().T @ (Gw_inv @ Gw_chi @ Gw_inv) @ Curl
        # K_tz = C[chi_t][d2; -d1]  (Eq. 21 / A43)
        Ktz = np.concatenate([
            (-self.blk((B, "d", T), (T, "m", T), "c22")
             + self.blk((B, "m", T), (T, "d", T), "c21")) / k0,
            (-self.blk((T, "m", T), (B, "d", T), "c11")
             + self.blk((T, "d", T), (B, "m", T), "c12")) / k0,
        ], axis=0)
        Meps33 = self.blk((T, "m", T), (T, "m", T), "e33")
        # K_zt = div(eps_t E_t) (Eq. 44), derivative on the V3 TEST function
        Kzt_E1 = (-self.blk((T, "dL", B), (T, "m", T), "e11")
                  - self.blk((T, "m", B), (T, "dL", T), "e21")) / k0
        Kzt_E2 = (-self.blk((T, "m", T), (T, "dL", B), "e22")
                  - self.blk((T, "dL", T), (T, "m", B), "e12")) / k0
        Kzt = np.concatenate([Kzt_E1, Kzt_E2], axis=1)
        Schur = Ktz @ np.linalg.solve(Meps33, Kzt)
        L = np.zeros((2 * qq, 2 * qq), dtype=_C)
        L[:qq, :qq] = E11
        L[qq:, qq:] = E22
        L[:qq, qq:] = E12
        L[qq:, :qq] = E21
        L += Stt
        L -= Schur
        self.Rmat, self.Lmat = R, L
        self.Gram = (G1, G2)
        self.Et = (E11, E12, E21, E22)
        self.Stt, self.Schur = Stt, Schur
        self.Meps33 = Meps33


def curved_region_modes(sol, eig_timer=None):
    """Eq.-25 modes of a mapped region: eig(L, -R); H partner through the PLAIN
    Gram (the chi_t != I trap of the magnetic route)."""
    t0 = time.perf_counter()
    g2, W = eig_pencil(sol.Lmat, -sol.Rmat)
    t_eig = time.perf_counter() - t0
    q = _forward_branch_flip(np.sqrt(np.asarray(g2, dtype=_C)))
    lam = -1j * q
    qq = sol.qq
    E11, E12, E21, E22 = sol.Et
    Lhh = np.block([[E11, E12], [E21, E22]]) + sol.Stt
    LW = Lhh @ W
    G1, G2 = sol.Gram
    Dual = np.concatenate([np.linalg.solve(G1, LW[:qq]),
                           np.linalg.solve(G2, LW[qq:])], axis=0)
    rot = np.concatenate([-Dual[qq:], Dual[:qq]], axis=0)
    V = rot * _inv_lam(q)[None, :]
    if eig_timer is not None:
        eig_timer.append(t_eig)
    return W, V, lam, g2


def curved_geom_eig(sol_h, eps_h):
    """Shared eps-free geometric eig of a MAPPED homogeneous region.  With an
    isotropic eps the mapped operators satisfy -R == [eps'_t]/eps EXACTLY
    (-R = adj(chi_t)^T = chi_t^-1 because det chi_t = 1, and eps'_t = eps
    chi_t^-1), and the Schur term is eps-free, so L(eps) = eps (-R) + L0 and ONE
    eig serves every uniform isotropic layer.  Returns (modes(eps), rel) where
    rel = max|[eps'_t]/eps - (-R)| / max|R| (the claim, measured)."""
    E11, E12, E21, E22 = sol_h.Et
    Et = np.block([[E11, E12], [E21, E22]])
    mR = -sol_h.Rmat
    L0 = sol_h.Lmat - eps_h * mR
    g2g, W0 = eig_pencil(L0, mR)
    qq = sol_h.qq
    G1, G2 = sol_h.Gram
    mRW0 = mR @ W0
    SttW0 = sol_h.Stt @ W0

    def modes(eps):
        g2 = g2g + eps
        q = _forward_branch_flip(np.sqrt(np.asarray(g2, dtype=_C)))
        LW = eps * mRW0 + SttW0            # Lhh W0 with [eps'_t] = eps (-R)
        Dual = np.concatenate([np.linalg.solve(G1, LW[:qq]),
                               np.linalg.solve(G2, LW[qq:])], axis=0)
        rot = np.concatenate([-Dual[qq:], Dual[:qq]], axis=0)
        return W0, rot * _inv_lam(q)[None, :], -1j * q

    rel = float(np.max(np.abs(Et / eps_h - mR)) / np.max(np.abs(mR)))
    return modes, rel


def curved_far_projector(sol, ox, oy, alpha0x=0.0, alpha0y=0.0, nq=None):
    """Pulled-back Rayleigh projector: (2 Nfo, 2 qq) operator mapping a region's
    [E'_u ; E'_v] coefficient vector onto physical [Ex_orders ; Ey_orders].
    Kernel exp(+i k_m . Phi) -- the shipped ``_stag_fourier_projection`` sign."""
    bx, by, cmap = sol.bx, sol.by, sol.cmap
    px, py = bx.d, by.d
    A = px * py
    nq = int(nq) if nq is not None else 2 * sol.M + 16
    xg, wg = leggauss(nq)
    Vref, _ = _modleg_value_deriv(sol.M, xg)
    ox = np.asarray(ox)
    oy = np.asarray(oy)
    kxv = (np.tile(ox, len(oy)) * 2 * np.pi / px + alpha0x)
    kyv = (np.repeat(oy, len(ox)) * 2 * np.pi / py + alpha0y)
    Nfo = kxv.size
    q = sol.q
    P = {k: np.zeros((Nfo, q, q), dtype=_C) for k in ("xu", "xv", "yu", "yv")}
    SBx, STx = np.asarray(bx.B), np.asarray(bx.Btilde)
    SBy, STy = np.asarray(by.B), np.asarray(by.Btilde)
    for sx in range(bx.N):
        U = 0.5 * (bx.xb[sx] + bx.xb[sx + 1]) + bx.Jn[sx] * xg
        Bx_v = SBx[:, sx, :] @ Vref           # (q, nq)  B set on x
        Tx_v = STx[:, sx, :] @ Vref
        for sy in range(by.N):
            Vv = 0.5 * (by.xb[sy] + by.xb[sy + 1]) + by.Jn[sy] * xg
            By_v = SBy[:, sy, :] @ Vref
            Ty_v = STy[:, sy, :] @ Vref
            g = cmap.geom(sx, sy, U, Vv)
            w2 = (wg[:, None] * wg[None, :]) * (bx.Jn[sx] * by.Jn[sy] / A)
            ph = np.exp(1j * (kxv[:, None, None] * g["X"][None]
                              + kyv[:, None, None] * g["Y"][None]))   # (Nfo,p,r)
            # det(J) Ex = yv E'u - yu E'v ; det(J) Ey = -xv E'u + xu E'v
            for key, coef, Xv, Yv in (("xu", g["yv"], Bx_v, Ty_v),
                                      ("xv", -g["yu"], Tx_v, By_v),
                                      ("yu", -g["xv"], Bx_v, Ty_v),
                                      ("yv", g["xu"], Tx_v, By_v)):
                if not np.any(coef):
                    continue
                K = ph * (coef * w2)[None]
                P[key] += np.einsum("mpr,ip,jr->mji", K, Xv, Yv, optimize=True)
    Pm = {k: v.reshape(Nfo, q * q) for k, v in P.items()}
    return np.block([[Pm["xu"], Pm["xv"]], [Pm["yu"], Pm["yv"]]])


def solve_curved(px, py, uw, vw, M, eps_cell, n_sup, n_sub, depth, wl,
                 cmap=None, n_orders=3, nq=None, nq_far=None, geom_eig=True,
                 pols=("te", "tm")):
    """Single patterned layer between two half-spaces, SAME map everywhere.
    Normal incidence only.  Returns dict with orders, R[pol], T[pol], timings,
    diagnostics."""
    k0 = 2 * np.pi / wl
    eps_sup = _C(n_sup) ** 2
    eps_sub = _C(n_sub) ** 2
    teig = []
    t0 = time.perf_counter()
    sol = CurvedGranet(px, py, uw, vw, M, eps_cell, cmap, k0=k0, nq=nq)
    Wl, Vl, lam_l, _ = curved_region_modes(sol, teig)
    t_asm = sol.t_assemble
    Nx, Ny = sol.bx.N, sol.by.N
    sol_h = CurvedGranet(px, py, uw, vw, M, np.full((Nx, Ny), eps_sup), cmap,
                         k0=k0, nq=nq)
    t_asm += sol_h.t_assemble
    diag = {"detJ_min_quad": sol.detmin}
    if geom_eig:
        t1 = time.perf_counter()
        modes, rel = curved_geom_eig(sol_h, eps_sup)
        teig.append(time.perf_counter() - t1)
        diag["geom_split_rel"] = rel
        Wsup, Vsup, _ = modes(eps_sup)
        Wsub, Vsub, _ = modes(eps_sub)
    else:
        Wsup, Vsup, _, _ = curved_region_modes(sol_h, teig)
        sol_b = CurvedGranet(px, py, uw, vw, M, np.full((Nx, Ny), eps_sub),
                             cmap, k0=k0, nq=nq)
        t_asm += sol_b.t_assemble
        Wsub, Vsub, _, _ = curved_region_modes(sol_b, teig)
    S = _interface_smatrix(Wsup, Vsup, Wl, Vl)
    S = _redheffer_star(S, _propagation_smatrix(lam_l, k0 * depth))
    S = _redheffer_star(S, _interface_smatrix(Wl, Vl, Wsub, Vsub))
    S11, _S12, S21, _S22 = S
    ox = np.arange(-n_orders, n_orders + 1)
    oy = ox
    order_x = np.tile(ox, len(oy))
    order_y = np.repeat(oy, len(ox))
    Nfo = order_x.size
    t2 = time.perf_counter()
    Pf = curved_far_projector(sol, ox, oy, nq=nq_far)
    t_far = time.perf_counter() - t2
    Hsup = Pf @ Wsup
    Hsub = Pf @ Wsub
    kxv = order_x * (wl / px)
    kyv = order_y * (wl / py)
    kz_ref, kz_trn, kz_inc, safe_r, safe_t = _pmm2d_order_kz(
        eps_sup, eps_sub, kxv, kyv, 0.0, 0.0)
    delta = ((order_x == 0) & (order_y == 0)).astype(_C)
    out = {"orders": np.stack([order_x, order_y], 1), "R": {}, "T": {}}
    for pol in pols:
        ex0, ey0 = (0.0, 1.0) if pol == "te" else (1.0, 0.0)
        rhs = np.concatenate([ex0 * delta, ey0 * delta])
        cinc = _guarded_lstsq(Hsup, rhs, "curved probe far field")
        r_ord = Hsup @ (S11 @ cinc)
        t_ord = Hsub @ (S21 @ cinc)
        rx, ry = r_ord[:Nfo], r_ord[Nfo:]
        tx, ty = t_ord[:Nfo], t_ord[Nfo:]
        rz = -(kxv * rx + kyv * ry) / safe_r
        tz = -(kxv * tx + kyv * ty) / safe_t
        R, T = _project_efficiency(np, kz_ref, kz_trn, kz_inc,
                                   rx, ry, rz, tx, ty, tz, 1.0)
        out["R"][pol] = np.real(np.asarray(R))
        out["T"][pol] = np.real(np.asarray(T))
    out["t_total"] = time.perf_counter() - t0
    out["t_assemble"] = t_asm
    out["t_eig"] = float(np.sum(teig))
    out["t_far"] = t_far
    out["dof"] = 2 * sol.qq
    out["diag"] = diag
    return out


ORD9 = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]


def vec(out, pol, orders=ORD9):
    """[R(orders); T(orders)] as one flat vector."""
    o = out["orders"]
    idx = [int(np.nonzero((o[:, 0] == m) & (o[:, 1] == n))[0][0]) for m, n in orders]
    return np.concatenate([out["R"][pol][idx], out["T"][pol][idx]])


def table(out, pol, orders=ORD9):
    o = out["orders"]
    res = {}
    for (m, n) in orders:
        i = int(np.nonzero((o[:, 0] == m) & (o[:, 1] == n))[0][0])
        res[f"{m},{n}"] = [float(out["R"][pol][i]), float(out["T"][pol][i])]
    res["sumR"] = float(np.sum(out["R"][pol]))
    res["sumT"] = float(np.sum(out["T"][pol]))
    return res


def peak_rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().peak_wset / 2**20
    except Exception:
        return float("nan")


def env_record():
    import platform
    import sys

    import scipy
    return {"python": sys.version.split()[0], "numpy": np.__version__,
            "scipy": scipy.__version__, "lumenairy": lumenairy.__file__,
            "machine": platform.node(), "whiten_eig": WHITEN,
            "threads": {k: os.environ.get(k) for k in
                        ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                         "MKL_NUM_THREADS")}}
