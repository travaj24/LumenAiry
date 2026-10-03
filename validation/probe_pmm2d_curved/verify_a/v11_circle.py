"""V11 -- what Phase B inherits: the planning probe's 3 x 3 CIRCLE map (a
transfinite / Gordon-Hall blend whose middle cell's edges are arcs; det J = 0
at the four 45-degree corners, weights ~ 1/r there) run through the SHIPPED
Phase-A solver via the map protocol.

  nodes   the adaptive node count, whether the cap fires, the warning text,
          and its cost, M = 4..8;
  film    a uniform film under the circle map vs the Airy slab (normal),
          M = 4..7, with the shipped adaptive rule AND with the planning rule
          2M + 8 forced (the planning probe measured 4.0e-6 .. 3.9e-14 at
          M = 4..8 with 2M + 8).

The Line / Arc / TransfiniteMap / circle_map_3x3 classes are COPIED from the
planning scratch (validation/probe_pmm2d_curved/_curved_scratch.py lines
124-235), which cannot be imported here (it asserts a different worktree).
"""
import sys
import time
import warnings

import _vcommon as C
import numpy as np

from lumenairy.elements.pmm import twod_staggered as TS
from lumenairy.elements.pmm._curvemap import CellMap

ARM = sys.argv[1]


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
        self.r, self.th0, self.th1 = float(r), float(th0), float(th1)

    def __call__(self, s):
        s = np.atleast_1d(np.asarray(s, float))
        th = self.th0 + s * (self.th1 - self.th0)
        val = self.c[None, :] + self.r * np.stack([np.cos(th), np.sin(th)], 1)
        der = self.r * (self.th1 - self.th0) * np.stack(
            [-np.sin(th), np.cos(th)], 1)
        return val, der


class CircleMap(CellMap):
    """Planning circle map, wrapped in the shipped protocol."""

    def __init__(self, P, r):
        c = (P / 2, P / 2)
        a = P / 2 - r / np.sqrt(2)
        b = P / 2 + r / np.sqrt(2)
        w = np.array([0.0, a, b, P])
        self.verts = [[(w[i], w[j]) for j in range(4)] for i in range(4)]
        d = np.pi / 180
        self.curved = {("h", 1, 1): Arc(c, r, 225 * d, 315 * d),
                       ("h", 1, 2): Arc(c, r, 135 * d, 45 * d),
                       ("v", 1, 1): Arc(c, r, 225 * d, 135 * d),
                       ("v", 2, 1): Arc(c, r, -45 * d, 45 * d)}
        self.r = r
        self._init_walls(w, w, P, P)
        try:
            self.validate()
            self.shipped_validate = "pass"
        except ValueError as e:          # the shipped check refuses it
            self.shipped_validate = str(e)
        self.seam_check()

    def seam_check(self, n=9):
        """The C0 conditions the solver actually relies on: POSITION
        periodicity across both seams (which makes the TANGENTIAL Jacobian
        column periodic too) -- the NORMAL column may jump, as it may across
        every interior grid line.  Returns the mismatches."""
        from numpy.polynomial.legendre import leggauss
        xg = leggauss(n)[0]
        out = {}
        Nx, Ny = self.shape
        P = self.period_x
        for sy in range(Ny):
            V = 0.5 * (self.v_bounds[sy] + self.v_bounds[sy + 1]) + 0.5 * (
                self.v_bounds[sy + 1] - self.v_bounds[sy]) * xg
            a = self.geom(0, sy, np.array([0.0]), V)
            b = self.geom(Nx - 1, sy, np.array([P]), V)
            out[f"u_seam_row{sy}"] = dict(
                pos=float(max(np.abs(b[0] - a[0] - P).max(),
                              np.abs(b[1] - a[1]).max())),
                tangential=float(max(np.abs(b[3] - a[3]).max(),
                                     np.abs(b[5] - a[5]).max())),
                normal=float(max(np.abs(b[2] - a[2]).max(),
                                 np.abs(b[4] - a[4]).max())))
        self.seam = out
        return out

    def edge(self, kind, i, j):
        cv = self.curved.get((kind, i, j))
        if cv is not None:
            return cv
        P0 = self.verts[i][j]
        Q = self.verts[i + 1][j] if kind == "h" else self.verts[i][j + 1]
        return Line(P0, Q)

    def geom(self, sx, sy, U, V):
        uw, vw = self.u_bounds, self.v_bounds
        u0, u1 = uw[sx], uw[sx + 1]
        v0, v1 = vw[sy], vw[sy + 1]
        s = (np.asarray(U, float) - u0) / (u1 - u0)
        t = (np.asarray(V, float) - v0) / (v1 - v0)
        Bv, Bd = self.edge("h", sx, sy)(s)
        Tv, Td = self.edge("h", sx, sy + 1)(s)
        Lv, Ld = self.edge("v", sx, sy)(t)
        Rv, Rd = self.edge("v", sx + 1, sy)(t)
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
        du, dv = u1 - u0, v1 - v0
        return (Phi[..., 0], Phi[..., 1], Ps[..., 0] / du, Pt[..., 0] / dv,
                Ps[..., 1] / du, Pt[..., 1] / dv)

    def _key(self):
        return ("CircleMap", self.r)


def run_nodes():
    cm = CircleMap(C.P, 0.3 * C.P)
    rows = []
    for M in range(4, 9):
        bx = TS.Basis1D(C.P, cm.u_walls, M)
        by = TS.Basis1D(C.P, cm.v_walls, M)
        t = time.perf_counter()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            n = TS._stag_map_nodes(bx, by, cm, M)
        rows.append(dict(M=M, nq=n, seconds=time.perf_counter() - t,
                         warning=[str(x.message) for x in w]))
        print(rows[-1], flush=True)
    C.dump("v11_circle_nodes", {"rows": rows,
                                "shipped_validate": cm.shipped_validate,
                                "seam": cm.seam})


def run_film():
    cm = CircleMap(C.P, 0.3 * C.P)
    film = np.full((3, 3), C.EPS_F)
    ex = C.airy()
    orig = TS._stag_map_nodes
    rows = []
    for M in range(4, 8):
        row = dict(M=M)
        for arm in ("adaptive", "2M+8"):
            if arm == "2M+8":
                TS._stag_map_nodes = (lambda *a, _n=2 * M + 8, **k: _n)
            t = time.perf_counter()
            try:
                o, R, T, J, _ = C.stack_solve(cm, [film], M)
            finally:
                TS._stag_map_nodes = orig
            i0 = C.i00(o)
            Rr, Tr = R.copy(), T.copy()
            Rr[:, i0] -= ex["s"][0]
            Tr[:, i0] -= ex["s"][1]
            row[arm] = float(max(np.abs(Rr).max(), np.abs(Tr).max()))
            row[arm + "_s"] = time.perf_counter() - t
        rows.append(row)
        print(row, flush=True)
    C.dump("v11_circle_film", {"rows": rows})


{"nodes": run_nodes, "film": run_film}[ARM]()
