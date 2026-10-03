"""The CURVED (non-separable) mortar of the pure staggered 2-D PMM: the
cross-mass between two layers whose staggered bases live on DIFFERENT
coordinate maps (Phase E2 of the curved-cell plan).

What this module is for
-----------------------
:class:`~lumenairy.elements.pmm.PMM2DStackPure` with
``layer_grids='per-layer'`` gives every layer its own element grid and couples
neighbouring grids by an L2 MORTAR (``docs/audits/BUILD_PMM2D_STAGGERED_
MORTAR_2026_09_11.md``).  On a straight wall grid the mortar's cross-mass --
the overlap integral of one layer's basis functions against the other's -- is
a Kronecker product of 1-D cross-masses (both bases live on rectangles of the
same physical plane), and the shipped
:class:`~lumenairy.elements.pmm.twod_staggered.StagCrossOps` computes it that
way.  When the layers carry two different curved maps -- a circle in one
layer, a sinusoidal wall in the next -- the overlap is an integral over the
PHYSICAL cell of one layer's basis against the other's, each pulled back
through its own map, and it does not factor.  This module computes it.

The words used below
--------------------
* ``Phi_a``, ``Phi_b`` -- the two layers' maps ``(u, v) -> (x, y)``; ``J_a``,
  ``J_b`` their Jacobians.  ``None`` is the identity (an unmapped layer).
* The **transition map** ``Psi = Phi_a^-1 o Phi_b`` takes a point given in
  layer b's ``(u, v)`` to layer a's; its Jacobian is ``T = J_a^-1 J_b``.
* **Covariant components** ``E' = J^T E``: the solver's unknowns.  Physical
  continuity of the tangential field gives ``E'_b = T^T E'_a`` -- a 1-form
  pulled back through ``Psi``.
* A **piece** is the overlap of one cell of layer a with one cell of layer b.

The formula
-----------
The shipped mortar imposes continuity weakly in each layer's own plain
``du dv`` product, which is the metric-free flux pairing
``INT (E x h*) . z dx dy`` (a 2-D cross product of covariant components
carries ``det J``, which cancels the area element).  With two maps the same
pairing between a's field and b's test functions reads, in b's coordinates,

    X_{beta alpha}[i, j] = INT conj(f^b_{beta, i}(q)) T_{alpha beta}(q)
                                f^a_{alpha, j}(Psi(q)) du_b dv_b

(``beta``, ``alpha`` = the component, 1 = along u, 2 = along v), or, the
same number by the change of variables, in a's coordinates with the weight
``adj(J_b^-1 J_a)`` in place of ``T``.  ``X`` is the operator the shipped
mortar calls ``CrossE^H`` (rows: b's test functions, V1 then V2; columns:
a's functions); the H-row operator is NOT a second integral -- the same
bilinear form gives

    CrossH = [[ X22^H, -X12^H],
              [-X21^H,  X11^H]]

(rows a's ``[V2; V1]`` -- the H1 / H2 placement of the Eq.-25 dual --
columns b's ``[H1; H2]``), which reduces to the shipped
``blkdiag(C2, C1)`` when ``T = I`` (one map, or none).  Every other piece of
the mortar -- each side's own plain Gram, the V1/V2 swap of the H rows, the
S-matrix algebra, the rectangular Redheffer star -- is unchanged.

The quadrature
--------------
Inside one piece the integrand is analytic: both bases are polynomials on
their own cells, both maps are analytic inside a cell, and ``Psi`` is smooth
there.  ACROSS the other layer's cell walls it is not -- the staggered set
``B`` is discontinuous across walls and ``Btilde`` kinks there -- and in the
coordinates of one layer the other layer's walls are CURVES.  A tensor Gauss
rule on one layer's cells that ignores those curves converges only
algebraically (measured in the build doc: the "composite map" route of the
plan's design question).  The rule used here is an ITERATED Gauss rule cut
along the curves, on every cell of the PRIMARY layer (the coordinates the
integral runs in):

1. the other layer's interior wall edges are pulled back into the cell
   (each a smooth curve; two of them meet only at a grid vertex of the other
   layer, never cross, because the other layer's walls form a grid);
2. an INNER direction is chosen (u or v) that minimises the number of points
   where a curve is tangent to it;
3. the OUTER coordinate is broken at every curve end (a cell side or a
   vertex of the other layer), every tangency point and every curve that is
   exactly parallel to the inner direction; between breakpoints the inner
   integral is analytic in the outer coordinate.  An interval that ends at a
   tangency takes the substitution ``s = s0 + L xi^2`` (the crossing points
   move like ``sqrt(s - s0)`` there, which the substitution makes smooth);
4. along every outer node the inner coordinate is broken at every crossing
   of a curve (a 2-D Newton solve for the curve parameter and the crossing
   point at once, no nested inversion) and each sub-interval takes a Gauss
   rule; the cell of the other layer is located once per sub-interval and
   every node of it is inverted there (Newton on that cell's analytic map).

The singular vertices.  A transfinite map has ``det J = 0`` at the four
45-degree points of a closed curve, where ``Phi^-1`` behaves like a square
root.  In the coordinates of THAT map the integrand stays analytic (its own
basis is a polynomial and ``adj(J)`` is bounded), while in the other layer's
coordinates it carries the square-root singularity.  So every piece is
integrated in the coordinates of the layer whose singular vertex it touches;
a piece that touches singular vertices of BOTH maps is refused, naming them.
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss

from .twod_staggered import _C, _modleg_value_deriv

#: Newton tolerance of the map inversion, relative to the period: the step
#: (in ``u``, ``v``) at which the iteration stops.  The inversion converges
#: quadratically, so the iterate is far below this when it stops; 1e-14 sits
#: two decades above double round-off of a coordinate of order the period.
_CURVE_MORTAR_INV_TOL = 1.0e-14
#: A converged inversion must reproduce the physical point to this (relative
#: to the period) or the point is treated as outside the cell.
_CURVE_MORTAR_RES_TOL = 1.0e-11
#: A curve whose outer coordinate varies by less than this (relative to the
#: period) across a cell is treated as exactly parallel to the inner
#: direction (one outer breakpoint, no crossings).  The approximation error is
#: (the variation) x (the integrand's jump) -- three decades under the
#: 1e-10 mortar-exactness bar.
_CURVE_MORTAR_FLAT_TOL = 1.0e-13
#: Samples along a pulled-back curve (Chebyshev-Lobatto) for locating the
#: cell entry / exit points and the tangency points.
_CURVE_MORTAR_SAMPLES = 65


def _gl01(n):
    """Gauss-Legendre on [0, 1]: (nodes, weights)."""
    x, w = leggauss(int(n))
    return 0.5 * (x + 1.0), 0.5 * w


class _MapView:
    """Pointwise access to ONE layer's coordinate map on its ``(u, v)`` wall
    grid -- position, Jacobian, inversion and point location.  ``cmap=None``
    is the identity (an unmapped layer), handled in closed form."""

    _NS = 17                     # per-axis sample grid of a cell (guesses)

    def __init__(self, cmap, ub, vb, period_x, period_y):
        self.cmap = cmap
        self.ub = np.asarray(ub, dtype=float)
        self.vb = np.asarray(vb, dtype=float)
        self.px = float(period_x)
        self.py = float(period_y)
        self.scale = max(self.px, self.py)
        self.Nx = self.ub.size - 1
        self.Ny = self.vb.size - 1
        self.identity = cmap is None
        # (sx, sy) -> [physical singular points] / [(u, v, x, y)]
        self.sing: dict[tuple[int, int], list[tuple[float, float]]] = {}
        self.sing_uv: dict[tuple[int, int],
                           list[tuple[float, float, float, float]]] = {}
        if self.identity:
            return
        # INTERIOR sample points (Gauss nodes): a Newton guess must never sit
        # on a singular corner, where det J = 0
        t = _gl01(self._NS)[0]
        tl = np.linspace(0.0, 1.0, 65)
        ub_s = np.zeros_like(tl)
        self._samp = {}
        self.bbox = np.empty((self.Nx, self.Ny, 4))
        for sx in range(self.Nx):
            for sy in range(self.Ny):
                U = self.ub[sx] + t * (self.ub[sx + 1] - self.ub[sx])
                V = self.vb[sy] + t * (self.vb[sy + 1] - self.vb[sy])
                UU, VV = (a.ravel() for a in np.meshgrid(U, V, indexing="ij"))
                X, Y = self.geom(sx, sy, UU, VV)[:2]
                self._samp[(sx, sy)] = (UU, VV, X, Y)
                # the bounding box from the cell's OUTLINE (the image of the
                # boundary of a cell bounds it: the map has det J > 0 inside)
                Ub = np.concatenate([ub_s * 0 + self.ub[sx], ub_s * 0
                                     + self.ub[sx + 1], self.ub[sx] + tl
                                     * (self.ub[sx + 1] - self.ub[sx]),
                                     self.ub[sx] + tl * (self.ub[sx + 1]
                                                         - self.ub[sx])])
                Vb = np.concatenate([self.vb[sy] + tl * (self.vb[sy + 1]
                                                         - self.vb[sy]),
                                     self.vb[sy] + tl * (self.vb[sy + 1]
                                                         - self.vb[sy]),
                                     ub_s * 0 + self.vb[sy],
                                     ub_s * 0 + self.vb[sy + 1]])
                Xb, Yb = self.geom(sx, sy, Ub, Vb)[:2]
                self.bbox[sx, sy] = (Xb.min(), Xb.max(), Yb.min(), Yb.max())
        for sx, sy, cu, cv in getattr(cmap, "singular_vertices", None) or ():
            u = self.ub[sx + cu]
            v = self.vb[sy + cv]
            X, Y = self.geom(sx, sy, np.array([u]), np.array([v]))[:2]
            self.sing.setdefault((int(sx), int(sy)), []).append(
                (float(X[0]), float(Y[0])))
            self.sing_uv.setdefault((int(sx), int(sy)), []).append(
                (float(u), float(v), float(X[0]), float(Y[0])))

    # ------------------------------------------------------------- geometry
    def geom(self, sx, sy, U, V):
        """``(X, Y, x_u, x_v, y_u, y_v)`` at the POINTS ``(U[k], V[k])``,
        evaluated with cell ``(sx, sy)``'s analytic formula (which extends
        smoothly beyond the cell)."""
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        if self.identity:
            one = np.ones_like(U)
            zero = np.zeros_like(U)
            return U.copy(), V.copy(), one, zero, zero, one
        return self.cmap.geom_points(int(sx), int(sy), U, V)

    def cell_of(self, U, V):
        """The cell index of ``(u, v)`` points (interior points)."""
        sx = np.clip(np.searchsorted(self.ub, U, side="right") - 1, 0,
                     self.Nx - 1)
        sy = np.clip(np.searchsorted(self.vb, V, side="right") - 1, 0,
                     self.Ny - 1)
        return sx, sy

    def invert(self, sx, sy, X, Y, guess=None, tol=None):
        """Newton inversion of cell ``(sx, sy)``'s analytic formula:
        ``(U, V, ok)`` with ``Phi(U, V) = (X, Y)``.  ``ok`` is False where
        the iteration did not reach the residual tolerance (a point far
        outside the cell's image).  The returned ``(U, V)`` may lie outside
        the cell -- the caller decides what that means."""
        X = np.asarray(X, dtype=float).ravel()
        Y = np.asarray(Y, dtype=float).ravel()
        if self.identity:
            return X.copy(), Y.copy(), np.ones(X.shape, dtype=bool)
        tol = ((_CURVE_MORTAR_INV_TOL if tol is None else float(tol))
               * self.scale)
        # points outside the cell's physical bounding box have no preimage
        # in the cell: no Newton for them (they come back ok = False)
        x0, x1, y0, y1 = self.bbox[int(sx), int(sy)]
        mg = 1e-3 * self.scale
        far = (X < x0 - mg) | (X > x1 + mg) | (Y < y0 - mg) | (Y > y1 + mg)
        if np.any(far):
            U = np.full(X.shape, np.nan)
            V = np.full(X.shape, np.nan)
            ok = np.zeros(X.shape, dtype=bool)
            near = ~far
            if np.any(near):
                g = None if guess is None else (
                    np.asarray(guess[0], dtype=float).ravel()[near],
                    np.asarray(guess[1], dtype=float).ravel()[near])
                U[near], V[near], ok[near] = self.invert(
                    sx, sy, X[near], Y[near], guess=g, tol=tol / self.scale)
            return U, V, ok
        if guess is None:
            UU, VV, XS, YS = self._samp[(int(sx), int(sy))]
            d = ((X[:, None] - XS[None, :]) ** 2
                 + (Y[:, None] - YS[None, :]) ** 2)
            k = np.argmin(d, axis=1)
            U = UU[k].copy()
            V = VV[k].copy()
        else:
            U = np.array(guess[0], dtype=float).ravel()
            V = np.array(guess[1], dtype=float).ravel()
        U, V = self._newton(sx, sy, X, Y, U, V, tol)
        Xr, Yr = self.geom(sx, sy, U, V)[:2]
        bad = ~(np.hypot(Xr - X, Yr - Y) <= _CURVE_MORTAR_RES_TOL
                * self.scale)
        if guess is not None and np.any(bad):
            # a poor caller guess: restart those points from the nearest
            # sample of the cell
            UU, VV, XS, YS = self._samp[(int(sx), int(sy))]
            d = ((X[bad][:, None] - XS[None, :]) ** 2
                 + (Y[bad][:, None] - YS[None, :]) ** 2)
            k = np.argmin(d, axis=1)
            Ub, Vb = self._newton(sx, sy, X[bad], Y[bad], UU[k].copy(),
                                  VV[k].copy(), tol)
            U[bad], V[bad] = Ub, Vb
        Xr, Yr = self.geom(sx, sy, U, V)[:2]
        ok = (np.hypot(Xr - X, Yr - Y) <= _CURVE_MORTAR_RES_TOL * self.scale)
        # a point AT a singular vertex of this cell (det J = 0 there, where
        # Newton cannot converge) is that corner, exactly
        for u0, v0, x0, y0 in self.sing_uv.get((int(sx), int(sy)), ()):
            at = np.hypot(X - x0, Y - y0) <= 1e-12 * self.scale
            U[at], V[at] = u0, v0
            ok = ok | at
        return U, V, ok & np.isfinite(U) & np.isfinite(V)

    def _newton(self, sx, sy, X, Y, U, V, tol):
        """Damped Newton for ``Phi(U, V) = (X, Y)`` on cell ``(sx, sy)``'s
        formula: every step is halved (up to 12 times) until the residual
        decreases, and never exceeds half a cell -- near a singular vertex
        (where ``Phi`` behaves like a square root) the full step overshoots
        into the folded extension of the cell's formula."""
        hu = self.ub[sx + 1] - self.ub[sx]
        hv = self.vb[sy + 1] - self.vb[sy]
        act = np.ones(X.shape, dtype=bool)
        Xa, Ya, xu, xv, yu, yv = self.geom(sx, sy, U, V)
        F = np.hypot(Xa - X, Ya - Y)
        for _it in range(80):
            ia = np.nonzero(act)[0]
            if ia.size == 0:
                break
            fx = Xa[ia] - X[ia]
            fy = Ya[ia] - Y[ia]
            det = xu[ia] * yv[ia] - xv[ia] * yu[ia]
            with np.errstate(divide="ignore", invalid="ignore"):
                du = (yv[ia] * fx - xv[ia] * fy) / det
                dv = (-yu[ia] * fx + xu[ia] * fy) / det
            du = np.clip(np.where(np.isfinite(du), du, 0.0), -0.5 * hu,
                         0.5 * hu)
            dv = np.clip(np.where(np.isfinite(dv), dv, 0.0), -0.5 * hv,
                         0.5 * hv)
            lam = np.ones(ia.size)
            pend = np.arange(ia.size)
            Un = U[ia].copy()
            Vn = V[ia].copy()
            gn = [np.empty(ia.size) for _ in range(6)]
            for _h in range(9):
                Ut = U[ia][pend] - lam[pend] * du[pend]
                Vt = V[ia][pend] - lam[pend] * dv[pend]
                gt = self.geom(sx, sy, Ut, Vt)
                Ft = np.hypot(gt[0] - X[ia][pend], gt[1] - Y[ia][pend])
                acc = (Ft < F[ia][pend]) | (_h == 8) | ~np.isfinite(
                    F[ia][pend])
                sel = pend[acc]
                Un[sel], Vn[sel] = Ut[acc], Vt[acc]
                for g, v in zip(gn, gt):
                    g[sel] = v[acc]
                pend = pend[~acc]
                if pend.size == 0:
                    break
                lam[pend] *= 0.5
            step = np.maximum(np.abs(Un - U[ia]), np.abs(Vn - V[ia]))
            U[ia], V[ia] = Un, Vn
            Xa[ia], Ya[ia], xu[ia], xv[ia], yu[ia], yv[ia] = gn
            F[ia] = np.hypot(Xa[ia] - X[ia], Ya[ia] - Y[ia])
            # an iterate a full cell outside the cell cannot be a preimage in
            # it: stop (the point comes back ok = False)
            gone = ((Un < self.ub[sx] - hu) | (Un > self.ub[sx + 1] + hu)
                    | (Vn < self.vb[sy] - hv) | (Vn > self.vb[sy + 1] + hv))
            act[ia[(step <= tol) | (F[ia] <= 1e-3 * tol) | gone]] = False
        return U, V

    def inside(self, sx, sy, U, V, tol=1e-12):
        t = tol * self.scale
        return ((U >= self.ub[sx] - t) & (U <= self.ub[sx + 1] + t)
                & (V >= self.vb[sy] - t) & (V <= self.vb[sy + 1] + t))

    def locate(self, X, Y, closure=False):
        """``(sx, sy, U, V, found)`` of physical points: the cell whose image
        holds each point and its ``(u, v)``.  ``closure=True`` returns, per
        point, the LIST of every cell whose closed image holds it (boundary
        points belong to several)."""
        X = np.asarray(X, dtype=float).ravel()
        Y = np.asarray(Y, dtype=float).ravel()
        n = X.size
        if self.identity:
            if closure:
                t = 1e-12 * self.scale
                out = []
                for x, y in zip(X, Y):
                    cs = []
                    for sx in range(self.Nx):
                        if not (self.ub[sx] - t <= x <= self.ub[sx + 1] + t):
                            continue
                        for sy in range(self.Ny):
                            if self.vb[sy] - t <= y <= self.vb[sy + 1] + t:
                                cs.append((sx, sy))
                    out.append(cs)
                return out
            sx, sy = self.cell_of(X, Y)
            return sx, sy, X.copy(), Y.copy(), np.ones(n, dtype=bool)
        SX = np.full(n, -1)
        SY = np.full(n, -1)
        U = np.zeros(n)
        V = np.zeros(n)
        found = np.zeros(n, dtype=bool)
        lists: list[list[tuple[int, int]]] = [[] for _ in range(n)]
        m = 1e-3 * self.scale        # a prefilter: generous on purpose
        for sx in range(self.Nx):
            for sy in range(self.Ny):
                x0, x1, y0, y1 = self.bbox[sx, sy]
                cand = ((X >= x0 - m) & (X <= x1 + m) & (Y >= y0 - m)
                        & (Y <= y1 + m))
                if not closure:
                    cand &= ~found
                if not np.any(cand):
                    continue
                ci = np.nonzero(cand)[0]
                Uc, Vc, ok = self.invert(sx, sy, X[ci], Y[ci])
                hit = ok & self.inside(sx, sy, Uc, Vc)
                for k in ci[hit] if closure else ():
                    lists[k].append((sx, sy))
                if closure:
                    continue
                h = ci[hit]
                SX[h], SY[h], U[h], V[h] = sx, sy, Uc[hit], Vc[hit]
                found[h] = True
        if closure:
            return lists
        return SX, SY, U, V, found

    # ----------------------------------------------------------- the edges
    def interior_edges(self):
        """The interior wall edges of the grid as ``(kind, fixed, lo, hi,
        cell)``: ``kind`` 'u' (a ``u = fixed`` edge running ``v`` from ``lo``
        to ``hi``) or 'v'; ``cell`` the cell whose formula evaluates it.  The
        cell boundary (``u = 0, p`` and ``v = 0, p``) is not an interior wall:
        every map takes it onto the lattice boundary, which is the other
        layer's boundary too."""
        out = []
        for k in range(1, self.Nx):
            for j in range(self.Ny):
                out.append(("u", self.ub[k], self.vb[j], self.vb[j + 1],
                            (k, j)))
        for j in range(1, self.Ny):
            for k in range(self.Nx):
                out.append(("v", self.vb[j], self.ub[k], self.ub[k + 1],
                            (k, j)))
        return out

    def edge_eval(self, e, tau):
        """Physical position and d/dtau of edge ``e`` at ``tau`` in [0, 1]."""
        kind, fixed, lo, hi, (sx, sy) = e
        tau = np.asarray(tau, dtype=float)
        run = lo + tau * (hi - lo)
        fx = np.full(tau.shape, fixed)
        if kind == "u":
            X, Y, _xu, xv, _yu, yv = self.geom(sx, sy, fx, run)
            return X, Y, xv * (hi - lo), yv * (hi - lo)
        X, Y, xu, _xv, yu, _yv = self.geom(sx, sy, run, fx)
        return X, Y, xu * (hi - lo), yu * (hi - lo)


# ------------------------------------------------------------------ pieces
class _Piece:
    """One other-layer edge's portion inside a primary cell, pulled back
    into the primary cell's ``(u, v)``: a sample table ``(tau, pu, pv,
    dpu, dpv)`` on ``[t0, t1]``."""

    __slots__ = ("edge", "t0", "t1", "tau", "pu", "pv", "dpu", "dpv")
    edge: tuple[object, ...]
    t0: float
    t1: float
    tau: np.ndarray
    pu: np.ndarray
    pv: np.ndarray
    dpu: np.ndarray
    dpv: np.ndarray


def _pullback(Pm, sx, sy, Om, e, tau, guess=None):
    """``(pu, pv, dpu, dpv, ok)`` of edge ``e`` of map ``Om`` at ``tau``,
    expressed in cell ``(sx, sy)`` of map ``Pm``."""
    X, Y, dX, dY = Om.edge_eval(e, tau)
    U, V, ok = Pm.invert(sx, sy, X, Y, guess=guess)
    _X, _Y, xu, xv, yu, yv = Pm.geom(sx, sy, U, V)
    det = xu * yv - xv * yu
    with np.errstate(divide="ignore", invalid="ignore"):
        dpu = (yv * dX - xv * dY) / det
        dpv = (-yu * dX + xu * dY) / det
    # at a singular vertex of the primary map the derivative is undefined;
    # it is only read for its SIGN (tangency search), where 0 = no sign
    dpu = np.where(np.isfinite(dpu), dpu, 0.0)
    dpv = np.where(np.isfinite(dpv), dpv, 0.0)
    return U, V, dpu, dpv, ok


def _side_crossing(Pm, sx, sy, Om, e, tlo, thi, t_in, u_in, v_in):
    """The parameter in ``[tlo, thi]`` where edge ``e`` of ``Om`` crosses a
    SIDE of cell ``(sx, sy)`` of ``Pm`` -- a 2-D Newton on (curve parameter,
    position along the side) for all four sides at once, no map inversion;
    the accepted root nearest the inside sample ``t_in`` (``u_in``,
    ``v_in``) wins.  ``None`` when no side converges inside the bracket."""
    a0, a1 = Pm.ub[sx], Pm.ub[sx + 1]
    c0, c1 = Pm.vb[sy], Pm.vb[sy + 1]
    fixed = np.array([a0, a1, c0, c1])
    isu = np.array([True, True, False, False])     # side u = fixed
    lo_w = np.where(isu, c0, a0)
    hi_w = np.where(isu, c1, a1)
    w = np.where(isu, v_in, u_in) + 0.0 * fixed
    tau = np.full(4, 0.5 * (tlo + thi))
    tol = _CURVE_MORTAR_INV_TOL * Pm.scale
    for _it in range(40):
        Xe, Ye, dXe, dYe = Om.edge_eval(e, tau)
        U = np.where(isu, fixed, w)
        V = np.where(isu, w, fixed)
        X, Y, xu, xv, yu, yv = Pm.geom(sx, sy, U, V)
        ci = np.where(isu, xv, xu)
        di = np.where(isu, yv, yu)
        fx, fy = X - Xe, Y - Ye
        det = -ci * dYe + dXe * di
        with np.errstate(divide="ignore", invalid="ignore"):
            d_w = (fx * dYe - dXe * fy) / det
            d_t = (di * fx - ci * fy) / det
        d_w = np.where(np.isfinite(d_w), d_w, 0.0)
        d_t = np.where(np.isfinite(d_t), d_t, 0.0)
        w = w + np.clip(d_w, -0.5 * (hi_w - lo_w), 0.5 * (hi_w - lo_w))
        tau = tau + np.clip(d_t, -(thi - tlo), thi - tlo)
        if np.max(np.abs(d_w)) <= tol and np.max(np.abs(d_t)) <= 1e-15:
            break
    Xe, Ye = Om.edge_eval(e, tau)[:2]
    U = np.where(isu, fixed, w)
    V = np.where(isu, w, fixed)
    X, Y = Pm.geom(sx, sy, U, V)[:2]
    res = np.hypot(X - Xe, Y - Ye)
    tt = 1e-12 * Pm.scale
    ok = ((res <= _CURVE_MORTAR_RES_TOL * Pm.scale) & (tau >= tlo - 1e-14)
          & (tau <= thi + 1e-14) & (w >= lo_w - tt) & (w <= hi_w + tt)
          & np.isfinite(tau))
    if not np.any(ok):
        return None
    k = np.nonzero(ok)[0]
    return float(tau[k[np.argmin(np.abs(tau[k] - t_in))]])


def _bracket_scan(pred_vec, lo, hi, want_lo_true, rounds=12, m=33):
    """Vectorised bisection: ``pred_vec(t_array) -> bool array`` changes
    value once in ``[lo, hi]`` (``want_lo_true``: the predicate is True at
    ``lo``); each round evaluates ``m`` points and keeps the sub-bracket
    holding the change."""
    for _ in range(rounds):
        t = np.linspace(lo, hi, m)
        p = np.asarray(pred_vec(t), dtype=bool)
        if want_lo_true:
            k = int(np.argmin(p)) if not np.all(p) else m - 1
        else:
            k = int(np.argmax(p)) if np.any(p) else m - 1
        lo, hi = t[max(k - 1, 0)], t[k]
        if hi - lo <= 1e-16:
            break
    return lo, hi


def _cell_pieces(Pm, sx, sy, Om, edges):
    """Every portion of ``Om``'s interior edges inside cell ``(sx, sy)`` of
    ``Pm``, as :class:`_Piece` s (entry / exit points located to round-off:
    a 2-D Newton against the cell's sides, a vectorised bisection on the
    inside predicate as the fallback)."""
    a0, a1 = Pm.ub[sx], Pm.ub[sx + 1]
    c0, c1 = Pm.vb[sy], Pm.vb[sy + 1]
    if Pm.identity:
        bx0, bx1, by0, by1 = a0, a1, c0, c1
    else:
        bx0, bx1, by0, by1 = Pm.bbox[sx, sy]
    m = 1e-3 * Pm.scale            # a prefilter: generous on purpose
    K = _CURVE_MORTAR_SAMPLES
    tk = 0.5 * (1.0 - np.cos(np.pi * np.arange(K) / (K - 1)))
    out = []
    for e, ebox in edges:
        if (ebox[1] < bx0 - m or ebox[0] > bx1 + m or ebox[3] < by0 - m
                or ebox[2] > by1 + m):
            continue
        U, V, _du, _dv, ok = _pullback(Pm, sx, sy, Om, e, tk)
        ins = ok & Pm.inside(sx, sy, U, V)
        if not np.any(ins):
            continue

        def inside_vec(t, e=e):
            u, v, _a, _b, okk = _pullback(Pm, sx, sy, Om, e, t)
            return okk & Pm.inside(sx, sy, u, v)

        k = 0
        while k < K:
            if not ins[k]:
                k += 1
                continue
            k0 = k
            while k + 1 < K and ins[k + 1]:
                k += 1
            k1 = k
            k += 1
            t0, t1 = tk[k0], tk[k1]
            if k0 > 0:                               # refine the entry
                r = _side_crossing(Pm, sx, sy, Om, e, tk[k0 - 1], tk[k0],
                                   tk[k0], U[k0], V[k0])
                if r is None:
                    r = _bracket_scan(inside_vec, tk[k0 - 1], tk[k0],
                                      False)[1]
                t0 = r
            if k1 < K - 1:                           # refine the exit
                r = _side_crossing(Pm, sx, sy, Om, e, tk[k1], tk[k1 + 1],
                                   tk[k1], U[k1], V[k1])
                if r is None:
                    r = _bracket_scan(inside_vec, tk[k1], tk[k1 + 1],
                                      True)[0]
                t1 = r
            if t1 - t0 <= 1e-14:
                continue
            xe, ye = Om.edge_eval(e, np.array([t0, t1]))[:2]
            if float(np.hypot(xe[1] - xe[0], ye[1] - ye[0])) <= (
                    1e-9 * Pm.scale):
                # a touch at a vertex (the inside test admits a point within
                # the residual tolerance of a corner): no cut
                continue
            ts = t0 + (t1 - t0) * tk
            gu = np.interp(ts, tk, U)
            gv = np.interp(ts, tk, V)
            pu, pv, dpu, dpv, okp = _pullback(Pm, sx, sy, Om, e, ts)
            if not np.all(okp):
                pu2, pv2, dpu2, dpv2, okp2 = _pullback(
                    Pm, sx, sy, Om, e, ts, guess=(gu, gv))
                fix = ~okp & okp2
                pu[fix], pv[fix] = pu2[fix], pv2[fix]
                dpu[fix], dpv[fix] = dpu2[fix], dpv2[fix]
                okp = okp | okp2
            if not np.all(okp):
                raise RuntimeError(
                    "curved mortar: a pulled-back wall of the neighbouring "
                    "layer could not be inverted inside its cell (map "
                    "inversion failed) -- the two maps are too different "
                    "for the cut-cell quadrature.")
            pu = np.clip(pu, a0, a1)
            pv = np.clip(pv, c0, c1)
            tb = 1e-12 * Pm.scale
            if any(bool(np.all(np.abs(c - b) <= tb)) for c, b in
                   ((pu, a0), (pu, a1), (pv, c0), (pv, c1))):
                # the other layer's wall runs ALONG this cell's side (the two
                # maps share that wall): no cut inside the cell
                continue
            pc = _Piece()
            pc.edge, pc.t0, pc.t1 = e, t0, t1
            pc.tau, pc.pu, pc.pv, pc.dpu, pc.dpv = ts, pu, pv, dpu, dpv
            out.append(pc)
    return out


def _tangencies(Pm, sx, sy, Om, pc, comp):
    """Parameters inside ``pc`` where the pulled-back curve's ``comp``
    coordinate ('u' or 'v') is stationary (the curve is tangent to the other
    direction), located by a vectorised bisection on the sign of its
    derivative."""
    d = pc.dpu if comp == "u" else pc.dpv
    out = []
    s = np.sign(d)
    nz = np.nonzero(s)[0]
    # sign changes between consecutive NONZERO samples; a sample that is
    # exactly 0 between them (a symmetric curve's tangency often sits on a
    # sample) is the tangency itself
    for i, j in zip(nz[:-1], nz[1:]):
        if s[i] * s[j] > 0.0:
            continue
        if j - i > 1:
            out.append(float(pc.tau[(i + j) // 2]) if j - i == 2 else
                       0.5 * float(pc.tau[i + 1] + pc.tau[j - 1]))
            continue
        slo = s[i]

        def pred(t, slo=slo):
            r = _pullback(Pm, sx, sy, Om, pc.edge, t)
            dm = r[2] if comp == "u" else r[3]
            return np.sign(dm) == slo

        lo, hi = _bracket_scan(pred, pc.tau[i], pc.tau[j], True)
        out.append(0.5 * (lo + hi))
    return out


def _outer_rule(o0, o1, sq0, sq1, n):
    """Nodes and weights on ``[o0, o1]``: Gauss, or the substitution
    ``o = o0 + L xi^2`` from an end that is a tangency (``sq0`` / ``sq1``);
    both ends -> split at the midpoint."""
    if sq0 and sq1:
        m = 0.5 * (o0 + o1)
        a = _outer_rule(o0, m, True, False, n)
        b = _outer_rule(m, o1, False, True, n)
        return np.concatenate([a[0], b[0]]), np.concatenate([a[1], b[1]])
    x, w = _gl01(n)
    L = o1 - o0
    if sq0:
        return o0 + L * x * x, w * 2.0 * L * x
    if sq1:
        return o1 - L * x * x, w * 2.0 * L * x
    return o0 + L * x, w * L


def _crossings(Pm, sx, sy, Om, pc, ta, tb, inner, onodes):
    """Inner coordinates where the monotone sub-arc ``[ta, tb]`` of piece
    ``pc`` crosses the lines ``outer = onodes`` (all strictly inside the
    sub-arc's outer range): a 2-D Newton on (curve parameter, inner
    coordinate) -- ``Phi_P(point on the line) = edge(tau)`` -- with no
    nested map inversion."""
    sel = (pc.tau >= ta - 1e-15) & (pc.tau <= tb + 1e-15)
    tt = pc.tau[sel]
    po = (pc.pv if inner == "u" else pc.pu)[sel]
    pi = (pc.pu if inner == "u" else pc.pv)[sel]
    if po[-1] < po[0]:
        tt, po, pi = tt[::-1], po[::-1], pi[::-1]
    tau = np.interp(onodes, po, tt)
    ii = np.interp(onodes, po, pi)
    lo_t, hi_t = min(ta, tb), max(ta, tb)
    tol = _CURVE_MORTAR_INV_TOL * Pm.scale
    for _it in range(40):
        Xe, Ye, dXe, dYe = Om.edge_eval(pc.edge, tau)
        if inner == "u":
            X, Y, xu, _xv, yu, _yv = Pm.geom(sx, sy, ii, onodes)
            ci, di = xu, yu
        else:
            X, Y, _xu, xv, _yu, yv = Pm.geom(sx, sy, onodes, ii)
            ci, di = xv, yv
        fx = X - Xe
        fy = Y - Ye
        # [ci, -dXe; di, -dYe] [d_i; d_tau] = -[fx; fy]
        det = -ci * dYe + dXe * di
        with np.errstate(divide="ignore", invalid="ignore"):
            d_i = (fx * dYe - dXe * fy) / det
            d_t = (di * fx - ci * fy) / det
        d_i = np.where(np.isfinite(d_i), d_i, 0.0)
        d_t = np.where(np.isfinite(d_t), d_t, 0.0)
        ii = ii + d_i
        tau = np.clip(tau + d_t, lo_t, hi_t)
        if (np.max(np.abs(d_i)) <= tol and np.max(np.abs(d_t)) * max(
                1.0, float(np.max(np.hypot(dXe, dYe)))) <= tol):
            break
    return ii


def _cut_cell_nodes(Pm, sx, sy, Om, edges, n, cut=True):
    """The cut-cell quadrature of cell ``(sx, sy)`` of the primary map
    ``Pm`` against the other map ``Om``: ``(pu, pv, w, osx, osy, qu, qv)``
    -- the nodes in the primary's ``(u, v)``, their weights (the ``du dv``
    measure), and the other layer's cell and ``(u, v)`` at each node.
    ``cut=False`` is the MEASUREMENT ARM of the plan's "composite map"
    route: one tensor Gauss rule on the cell, ignoring the other layer's
    walls (each node is located on its own)."""
    a0, a1 = Pm.ub[sx], Pm.ub[sx + 1]
    c0, c1 = Pm.vb[sy], Pm.vb[sy + 1]
    tiny = 1e-14 * Pm.scale
    if not cut:
        x, w = _gl01(n)
        U = a0 + (a1 - a0) * x
        V = c0 + (c1 - c0) * x
        UU, VV = (a.ravel() for a in np.meshgrid(U, V, indexing="ij"))
        WW = np.outer(w * (a1 - a0), w * (c1 - c0)).ravel()
        X, Y = Pm.geom(sx, sy, UU, VV)[:2]
        osx, osy, qu, qv, found = Om.locate(X, Y)
        if not np.all(found):
            raise RuntimeError("curved mortar: a node could not be located "
                               "in the neighbouring layer's map.")
        return UU, VV, WW, osx, osy, qu, qv
    pieces = _cell_pieces(Pm, sx, sy, Om, edges)
    # ---- the inner direction: fewest tangencies (ties -> u) ------------
    # inner direction -> pieces parallel to it / tangency parameters
    flat: dict[str, list[int]] = {"u": [], "v": []}
    tang: dict[str, dict[int, list[float]]] = {"u": {}, "v": {}}
    ftol = _CURVE_MORTAR_FLAT_TOL * Pm.scale
    for k, pc in enumerate(pieces):
        for inner, oc in (("u", pc.pv), ("v", pc.pu)):
            if float(np.max(oc) - np.min(oc)) <= ftol:
                flat[inner].append(k)
                tang[inner][k] = []
            else:
                tang[inner][k] = _tangencies(Pm, sx, sy, Om, pc,
                                             "v" if inner == "u" else "u")
    cnt = {d: sum(len(v) for v in tang[d].values()) for d in ("u", "v")}
    inner = "u" if cnt["u"] <= cnt["v"] else "v"
    (i0, i1), (o0, o1) = (((a0, a1), (c0, c1)) if inner == "u"
                          else ((c0, c1), (a0, a1)))
    # ---- outer breakpoints ----------------------------------------------
    bps = [(o0, False), (o1, False)]
    arcs = []                              # (piece, ta, tb, olo, ohi)
    for k, pc in enumerate(pieces):
        oc = pc.pv if inner == "u" else pc.pu
        if k in flat[inner]:
            bps.append((float(np.mean(oc)), False))
            continue
        ts = [pc.t0] + list(tang[inner][k]) + [pc.t1]
        for t in tang[inner][k]:
            r = _pullback(Pm, sx, sy, Om, pc.edge, np.array([t]))
            ov = float(r[1][0] if inner == "u" else r[0][0])
            bps.append((ov, True))
        bps.append((float(oc[0]), False))
        bps.append((float(oc[-1]), False))
        for ta, tb in zip(ts[:-1], ts[1:]):
            ra = _pullback(Pm, sx, sy, Om, pc.edge, np.array([ta, tb]))
            ov = ra[1] if inner == "u" else ra[0]
            arcs.append((pc, ta, tb, float(min(ov)), float(max(ov))))
    bps.sort()
    merged: list[tuple[float, bool]] = []
    for o, sq in bps:
        o = min(max(o, o0), o1)
        if merged and o - merged[-1][0] <= 1e-13 * Pm.scale:
            merged[-1] = (merged[-1][0], merged[-1][1] or sq)
        else:
            merged.append((o, sq))
    on_p, ow_p = [], []
    for (oa, sa), (ob, sb) in zip(merged[:-1], merged[1:]):
        if ob - oa <= tiny:
            continue
        x, w = _outer_rule(oa, ob, sa, sb, n)
        on_p.append(x)
        ow_p.append(w)
    on = np.concatenate(on_p)
    ow = np.concatenate(ow_p)
    # ---- crossings per outer node ---------------------------------------
    cross = [[i0, i1] for _ in range(on.size)]
    for pc, ta, tb, olo, ohi in arcs:
        hit = np.nonzero((on > olo) & (on < ohi))[0]
        if hit.size == 0:
            continue
        ii = _crossings(Pm, sx, sy, Om, pc, ta, tb, inner, on[hit])
        for kh, v in zip(hit.tolist(), ii):
            cross[kh].append(float(min(max(v, i0), i1)))
    xg, wg = _gl01(n)
    segs = []
    for k in range(on.size):
        cs = sorted(cross[k])
        for a, b in zip(cs[:-1], cs[1:]):
            if b - a > tiny:
                segs.append((on[k], ow[k], a, b))
    seg_o, seg_w, seg_a, seg_b = (np.asarray(c, dtype=float)
                                  for c in zip(*segs))
    # locate every sub-interval's midpoint in the other map
    mid = 0.5 * (seg_a + seg_b)
    if inner == "u":
        Xm, Ym = Pm.geom(sx, sy, mid, seg_o)[:2]
    else:
        Xm, Ym = Pm.geom(sx, sy, seg_o, mid)[:2]
    msx, msy, mqu, mqv, found = Om.locate(Xm, Ym)
    if not np.all(found):
        raise RuntimeError("curved mortar: a quadrature sub-interval could "
                           "not be located in the neighbouring layer's map.")
    # the nodes
    L = (seg_b - seg_a)
    I = seg_a[:, None] + L[:, None] * xg[None, :]
    O = np.broadcast_to(seg_o[:, None], I.shape)
    W = (seg_w * L)[:, None] * wg[None, :]
    osx = np.broadcast_to(msx[:, None], I.shape).ravel()
    osy = np.broadcast_to(msy[:, None], I.shape).ravel()
    gu = np.broadcast_to(mqu[:, None], I.shape).ravel()
    gv = np.broadcast_to(mqv[:, None], I.shape).ravel()
    if inner == "u":
        pu, pv = I.ravel(), O.ravel()
    else:
        pu, pv = O.ravel(), I.ravel()
    W = W.ravel()
    X, Y = Pm.geom(sx, sy, pu, pv)[:2]
    qu = np.empty_like(pu)
    qv = np.empty_like(pv)
    for cx, cy in set(zip(msx.tolist(), msy.tolist())):
        s = (osx == cx) & (osy == cy)
        qu[s], qv[s], ok = Om.invert(cx, cy, X[s], Y[s],
                                     guess=(gu[s], gv[s]))
        if not np.all(ok):
            raise RuntimeError(
                "curved mortar: a quadrature node could not be inverted in "
                "the neighbouring layer's map.")
    return pu, pv, W, osx.copy(), osy.copy(), qu, qv


# ------------------------------------------------------------ assignment
def _same_cell(A, ac, B, bc):
    """True when cell ``ac`` of map ``A`` and cell ``bc`` of map ``B`` are
    the same ``(u, v)`` rectangle carrying the same map (positions and
    Jacobians equal on a probe grid to 1e-12 of the period)."""
    t = 1e-13 * A.scale
    if (abs(A.ub[ac[0]] - B.ub[bc[0]]) > t
            or abs(A.ub[ac[0] + 1] - B.ub[bc[0] + 1]) > t
            or abs(A.vb[ac[1]] - B.vb[bc[1]]) > t
            or abs(A.vb[ac[1] + 1] - B.vb[bc[1] + 1]) > t):
        return False
    x, _w = _gl01(7)
    U = A.ub[ac[0]] + x * (A.ub[ac[0] + 1] - A.ub[ac[0]])
    V = A.vb[ac[1]] + x * (A.vb[ac[1] + 1] - A.vb[ac[1]])
    UU, VV = (a.ravel() for a in np.meshgrid(U, V, indexing="ij"))
    ga = A.geom(ac[0], ac[1], UU, VV)
    gb = B.geom(bc[0], bc[1], UU, VV)
    return all(float(np.max(np.abs(p - q))) <= 1e-12 * A.scale
               for p, q in zip(ga, gb))


def _assign_pairs(A, B):
    """Which coordinates each overlapping cell pair (a-cell, b-cell) is
    integrated in: ``'a'`` where the pair touches a singular vertex of map a
    only, ``'b'`` where it touches one of map b only, and the DEFAULT
    otherwise (the curved side; ``a`` when both or neither are curved).
    Returns ``(default, {(acell, bcell): side})`` for the exceptions; a pair
    touching singular vertices of BOTH maps RAISES."""
    if A.sing and not B.sing:
        default = "a"
    elif B.sing and not A.sing:
        default = "b"
    elif A.identity and not B.identity:
        default = "b"
    else:
        default = "a"
    marks: dict[tuple[tuple[int, int], tuple[int, int]], set[str]] = {}
    for side, M_own, M_oth in (("a", A, B), ("b", B, A)):
        for cell, pts in M_own.sing.items():
            for x, y in pts:
                for oc in M_oth.locate(np.array([x]), np.array([y]),
                                       closure=True)[0]:
                    key = (cell, oc) if side == "a" else (oc, cell)
                    marks.setdefault(key, set()).add(side)
    out = {}
    for key, sides in marks.items():
        if len(sides) == 2 and _same_cell(A, key[0], B, key[1]):
            # the two layers carry the SAME map on the SAME cell (a shared
            # singular vertex, e.g. one circle in both layers): the
            # transition map is the identity there and the integrand is
            # regular in either coordinates
            sides = {"a"}
        if len(sides) == 2:
            raise NotImplementedError(
                f"curved mortar: the overlap of cell {key[0]} of the upper "
                f"layer's map and cell {key[1]} of the lower layer's touches "
                f"a SINGULAR vertex (det J = 0, a closed curve's 45-degree "
                f"point) of BOTH maps; the cross-mass integrand is singular "
                f"in either layer's coordinates there.  Move one outline so "
                f"the two maps' singular vertices do not share a cell "
                f"overlap, or use layer_grids='shared' (one merged map) "
                f"when the outlines do not cross.")
        out[key] = sides.pop()
    return default, out


# ------------------------------------------------------------- the kernel
def _component_sets(basis_u, basis_v, beta):
    """Global sets of component ``beta`` (0 = E1 in V1 = B(u) x Btilde(v),
    1 = E2 in V2 = Btilde(u) x B(v)) as ``(Su, Sv)`` stencil arrays."""
    if beta == 0:
        return np.asarray(basis_u.B), np.asarray(basis_v.Btilde)
    return np.asarray(basis_u.Btilde), np.asarray(basis_v.B)


def _local_vals(basis, s, coord):
    """The ``M`` modified-Legendre local functions of segment ``s`` of
    ``basis`` at the coordinates ``coord`` -> ``(M, n)``."""
    xl, xr = basis.xb[s], basis.xb[s + 1]
    ref = (2.0 * coord - (xl + xr)) / (xr - xl)
    return _modleg_value_deriv(basis.M, ref)[0]


def curved_cross_mass(ga, gb, n, cut=True, one_map=None, primary=None):
    """THE non-separable cross-mass kernel: ``X`` (the mortar's ``CrossE^H``,
    ``(2 qq_b, 2 qq_a)``, rows b's ``[V1; V2]`` test functions, columns a's
    ``[V1; V2]``) between grid ``ga`` (upper layer, map ``ga.cmap``) and
    grid ``gb`` (lower layer, map ``gb.cmap``), by the cut-cell rule with
    ``n`` Gauss nodes per sub-interval per direction (see the module
    docstring).

    ``cut=False`` and ``one_map`` are MEASUREMENT / FAIL-BEFORE arms, never
    used by the solver: ``cut=False`` ignores the other layer's walls (one
    tensor Gauss rule per cell -- the "composite map" route); ``one_map='a'``
    (or ``'b'``) evaluates BOTH bases on that one layer's map (the
    cross-mass "computed on one layer's map only"); ``primary='a'`` /
    ``'b'`` integrates EVERY piece in that layer's coordinates (the
    self-consistency arm: the same integral in the other coordinates)."""
    A = _MapView(ga.cmap, ga.bx.xb, ga.by.xb, ga.bx.d, ga.by.d)
    B = _MapView(gb.cmap, gb.bx.xb, gb.by.xb, gb.bx.d, gb.by.d)
    if one_map is not None:
        M_one = A if one_map == "a" else B
        A = B = M_one
    if primary is None:
        default, exc = _assign_pairs(A, B)
    else:
        default, exc = primary, {}
    sides = {default} | set(exc.values())
    qa, qb = ga.qq, gb.qq
    X = np.zeros((2 * qb, 2 * qa), dtype=_C)
    sets_a = [_component_sets(ga.bx, ga.by, c) for c in (0, 1)]
    sets_b = [_component_sets(gb.bx, gb.by, c) for c in (0, 1)]
    for side in sorted(sides):
        Pm, Om = (A, B) if side == "a" else (B, A)
        edges = []
        for e in Om.interior_edges():
            tt = np.linspace(0.0, 1.0, 65)
            Xe, Ye = Om.edge_eval(e, tt)[:2]
            edges.append((e, (Xe.min(), Xe.max(), Ye.min(), Ye.max())))
        for psx in range(Pm.Nx):
            for psy in range(Pm.Ny):
                pu, pv, w, osx, osy, qu, qv = _cut_cell_nodes(
                    Pm, psx, psy, Om, edges, n, cut=cut)
                for cx, cy in sorted(set(zip(osx.tolist(), osy.tolist()))):
                    pcell, ocell = (psx, psy), (cx, cy)
                    key = (pcell, ocell) if side == "a" else (ocell, pcell)
                    if exc.get(key, default) != side:
                        continue
                    s = (osx == cx) & (osy == cy)
                    if side == "a":
                        ac, bc = pcell, ocell
                        ua, va, ub_, vb_ = pu[s], pv[s], qu[s], qv[s]
                    else:
                        ac, bc = ocell, pcell
                        ua, va, ub_, vb_ = qu[s], qv[s], pu[s], pv[s]
                    _add_pair(X, ga, gb, A, B, ac, bc, ua, va, ub_, vb_,
                              w[s], side, sets_a, sets_b)
    return X


def _add_pair(X, ga, gb, A, B, ac, bc, ua, va, ub_, vb_, w, side, sets_a,
              sets_b):
    """Accumulate one cell pair's nodes into ``X``."""
    _Xa, _Ya, axu, axv, ayu, ayv = A.geom(ac[0], ac[1], ua, va)
    _Xb, _Yb, bxu, bxv, byu, byv = B.geom(bc[0], bc[1], ub_, vb_)
    if side == "b":
        # T = J_a^-1 J_b, measure du_b dv_b
        da = axu * ayv - axv * ayu
        i11, i12, i21, i22 = ayv / da, -axv / da, -ayu / da, axu / da
        Wt = ((i11 * bxu + i12 * byu, i11 * bxv + i12 * byv),
              (i21 * bxu + i22 * byu, i21 * bxv + i22 * byv))
    else:
        # adj(J_b^-1 J_a), measure du_a dv_a
        db = bxu * byv - bxv * byu
        j11, j12, j21, j22 = byv / db, -bxv / db, -byu / db, bxu / db
        s11 = j11 * axu + j12 * ayu
        s12 = j11 * axv + j12 * ayv
        s21 = j21 * axu + j22 * ayu
        s22 = j21 * axv + j22 * ayv
        Wt = ((s22, -s12), (-s21, s11))
    Fa_u = _local_vals(ga.bx, ac[0], ua)
    Fa_v = _local_vals(ga.by, ac[1], va)
    Fb_u = _local_vals(gb.bx, bc[0], ub_)
    Fb_v = _local_vals(gb.by, bc[1], vb_)
    Ma, Mb = ga.bx.M, gb.bx.M
    Ga = (Fa_v[:, None, :] * Fa_u[None, :, :]).reshape(Ma * Ma, -1)
    Gb = (Fb_v[:, None, :] * Fb_u[None, :, :]).reshape(Mb * Mb, -1)
    qxa, qxb = ga.bx.dim, gb.bx.dim
    for beta in (0, 1):
        Sbu, Sbv = sets_b[beta]
        SU = Sbu[:, bc[0], :]
        SV = Sbv[:, bc[1], :]
        su = np.nonzero(np.any(SU != 0, axis=1))[0]
        sv = np.nonzero(np.any(SV != 0, axis=1))[0]
        Kb = np.kron(SV[sv], SU[su])                      # (nb, Mb^2)
        Ib = (sv[:, None] * qxb + su[None, :]).ravel() + beta * gb.qq
        for alpha in (0, 1):
            wt = w * Wt[alpha][beta]
            if not np.any(wt):
                continue
            L = (Gb * wt[None, :]) @ Ga.T                 # (Mb^2, Ma^2)
            Sau, Sav = sets_a[alpha]
            AU = Sau[:, ac[0], :]
            AV = Sav[:, ac[1], :]
            au = np.nonzero(np.any(AU != 0, axis=1))[0]
            av = np.nonzero(np.any(AV != 0, axis=1))[0]
            Ka = np.kron(AV[av], AU[au])                  # (na, Ma^2)
            Ja = (av[:, None] * qxa + au[None, :]).ravel() + alpha * ga.qq
            X[np.ix_(Ib, Ja)] += np.conj(Kb) @ L @ Ka.T


def cross_h_from_x(X, qa, qb):
    """The H-row cross operator from ``X`` (see the module docstring):
    ``[[X22^H, -X12^H], [-X21^H, X11^H]]``, ``(2 qq_a, 2 qq_b)``."""
    X11 = X[:qb, :qa]
    X12 = X[:qb, qa:]
    X21 = X[qb:, :qa]
    X22 = X[qb:, qa:]
    return np.block([[X22.conj().T, -X12.conj().T],
                     [-X21.conj().T, X11.conj().T]])


#: Relative change of the cross-mass between two successive node counts at
#: which the adaptive rule of :func:`curved_cross_mass_adaptive` stops.  Two
#: decades under the 1e-10 mortar-exactness bar of gate E2-2; the measured
#: rule is spectral (``docs/audits/BUILD_PMM2D_CURVED_E2_2026_10_03.md``,
#: gate E2-5), so the stopping rung is far below it.
_CURVE_MORTAR_QUAD_TOL = 1.0e-12
#: Largest node count per sub-interval per direction the adaptive rule may
#: reach before it WARNS with the change it achieved.
_CURVE_MORTAR_QUAD_CAP = 96


def curved_cross_mass_adaptive(ga, gb, tol=None, cap=None):
    """:func:`curved_cross_mass` at an ADAPTIVE node count: starting from
    ``n = max(M_a, M_b) + 4`` it grows by 1.5x until two successive
    cross-masses agree to ``tol`` relative (max-norm); returns ``(X, n,
    change)`` with ``X`` the finer of the two.  Reaching ``cap`` WARNS."""
    import warnings
    tol = _CURVE_MORTAR_QUAD_TOL if tol is None else float(tol)
    cap = _CURVE_MORTAR_QUAD_CAP if cap is None else int(cap)
    n = min(max(ga.M, gb.M) + 4, cap)
    X0 = curved_cross_mass(ga, gb, n)
    while True:
        n1 = min(int(np.ceil(1.5 * n)), cap)
        X1 = curved_cross_mass(ga, gb, n1) if n1 > n else X0
        sc = float(np.max(np.abs(X1))) or 1.0
        chg = float(np.max(np.abs(X1 - X0))) / sc
        if chg <= tol or n1 >= cap:
            break
        n, X0 = n1, X1
    if chg > tol:
        warnings.warn(
            f"PMM2DStackPure: the curved mortar's cross-mass between two "
            f"differently mapped layers did not settle to {tol:.0e} within "
            f"{n1} Gauss nodes per sub-interval (last change {chg:.2e}); the "
            f"interface carries a quadrature error of that order.  The two "
            f"maps are too different inside a cell -- add walls (grid_hint) "
            f"where they are steep.", stacklevel=4)
    return X1, n1, chg


class StagCrossOpsMapped:
    """The cross-mass between two :class:`~lumenairy.elements.pmm.
    twod_staggered.StagGridOps` on DIFFERENT coordinate maps -- the
    non-separable twin of :class:`~lumenairy.elements.pmm.twod_staggered.
    StagCrossOps` (Phase E2 of the curved-cell plan).

    ``EH`` is the mortar's ``CrossE^H`` (``(2 qq_b, 2 qq_a)``, dense) and
    ``H`` its ``CrossH`` (``(2 qq_a, 2 qq_b)``), derived from ``EH`` by the
    bilinear-form identity of this module's docstring.  ``dense = True``
    tells the shared mortar algebra
    (:func:`~lumenairy.elements.pmm._core._interface_smatrix_mortar_2d`,
    :func:`~lumenairy.elements.pmm._core._interface_smatrix_general_mortar_2d`)
    to apply them as matrices instead of separable Kronecker pairs; every
    other piece of that algebra is unchanged.  ``C1`` / ``C2`` are ``None``
    (there is no separable factor).  ``n`` and ``change`` record the
    adaptive node count reached and the last relative change."""

    dense = True
    __slots__ = ("EH", "H", "C1", "C2", "n", "change")

    def __init__(self, ga, gb, tol=None):
        X, n, chg = curved_cross_mass_adaptive(ga, gb, tol=tol)
        self.EH = X
        self.H = cross_h_from_x(X, ga.qq, gb.qq)
        self.C1 = self.C2 = None
        self.n = n
        self.change = chg
