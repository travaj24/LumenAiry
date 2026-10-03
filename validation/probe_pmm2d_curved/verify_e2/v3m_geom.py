"""V3 (decisive physics check): the E2-4 crossing device on ONE conforming
merged map, built by hand.  Fixture = the builder's E2-4: lambda 1, P 1.2,
n_sup 1.0, n_sub 1.45; layer 1 (0.3) eps-4 disk c = (0.6, 0.6), r = 0.36;
layer 2 (0.25) wall x = 0.6 + 0.12 sin(2 pi y / 1.2), eps 2.25 right of it.

Grid (4 x 4 cells):
  u walls 0, c - h, 0.6 (the SINUSOID u-line), c + h, 1.2   (h = r / sqrt 2)
  v walls 0, 0.12 (a straight DUMMY line), c - h, c + h, 1.2
The circle's bottom / top arcs (v = c -+ h) are split at the two points
where the sinusoid crosses the circle (A ~ (0.716, 0.259), B ~ (0.484,
0.941), point-symmetric about c).  Disk = cells (1, 2) and (2, 2); the wall
layer's eps 2.25 = cells i >= 2.
"""
import numpy as np
from scipy.optimize import brentq

from lumenairy.elements.pmm._curvemap import Arc, Sinusoid, TransfiniteMap

P = 1.2
WL = 1.0
N_SUP, N_SUB = 1.0, 1.45
CX = CY = 0.6
R = 0.36
X0, AMP = 0.6, 0.12
EPS_D, EPS_W = 4.0, 2.25
D1, D2 = 0.3, 0.25
Y_DUMMY = 0.12
DEG = np.pi / 180.0


def xs(y):
    return X0 + AMP * np.sin(2.0 * np.pi * y / P)


def crossings():
    """y of the two sinusoid / circle crossings, to round-off (bracketed
    brentq on f(y) = (xs(y) - cx)^2 + (y - cy)^2 - r^2, then Newton)."""
    f = lambda y: (xs(y) - CX) ** 2 + (y - CY) ** 2 - R ** 2  # noqa: E731

    def df(y):
        return (2 * (xs(y) - CX) * AMP * 2 * np.pi / P
                * np.cos(2 * np.pi * y / P) + 2 * (y - CY))
    out = []
    for lo, hi in ((0.2, 0.4), (0.8, 1.0)):
        y = brentq(f, lo, hi, xtol=1e-16, rtol=1e-15, maxiter=200)
        for _ in range(3):
            y = y - f(y) / df(y)
        out.append(y)
    return out


def build(y_dummy=Y_DUMMY):
    h = R / np.sqrt(2.0)
    a1, a2 = CX - h, CX + h
    uw = np.array([0.0, a1, X0, a2, P])
    vw = np.array([0.0, y_dummy, a1, a2, P])
    yA, yB = crossings()
    A = np.array([xs(yA), yA])
    B = np.array([xs(yB), yB])
    thA = np.arctan2(A[1] - CY, A[0] - CX) + 2 * np.pi   # in (225, 315) deg
    thB = np.arctan2(B[1] - CY, B[0] - CX)               # in (45, 135) deg
    assert 225 * DEG < thA < 315 * DEG and 45 * DEG < thB < 135 * DEG
    yline = [0.0, y_dummy, yA, yB, P]                    # sinusoid ys
    V = np.empty((5, 5, 2))
    for i in range(5):
        for j in range(5):
            V[i, j] = (uw[i], vw[j])
    for j in range(5):
        V[2, j] = (xs(yline[j]), yline[j])
    c = (CX, CY)
    curved = {("h", 1, 2): Arc(c, R, 225 * DEG, thA),
              ("h", 2, 2): Arc(c, R, thA, 315 * DEG),
              ("h", 1, 3): Arc(c, R, 135 * DEG, thB),
              ("h", 2, 3): Arc(c, R, thB, 45 * DEG),
              ("v", 1, 2): Arc(c, R, 225 * DEG, 135 * DEG),
              ("v", 3, 2): Arc(c, R, -45 * DEG, 45 * DEG)}
    for j in range(4):
        curved[("v", 2, j)] = Sinusoid(X0, AMP, P, yline[j], yline[j + 1],
                                       along="y")
    tm = TransfiniteMap(uw, vw, V, curved)
    info = dict(u_walls=uw, v_walls=vw, A=A, B=B, thA_deg=thA / DEG,
                thB_deg=thB / DEG, yA=yA, yB=yB,
                f_resid=[float((xs(y) - CX) ** 2 + (y - CY) ** 2 - R ** 2)
                         for y in (yA, yB)])
    return tm, info


def cells():
    c1 = np.ones((4, 4), complex)
    c1[1, 2] = c1[2, 2] = EPS_D
    c2 = np.ones((4, 4), complex)
    c2[2:, :] = EPS_W
    return c1, c2


def cell_areas(tm, n=40):
    """Physical area of every (u, v) cell by Gauss-Legendre on det J."""
    g, w = np.polynomial.legendre.leggauss(n)
    Nx, Ny = tm.shape
    A = np.zeros((Nx, Ny))
    detmin = np.inf
    for sx in range(Nx):
        u0, u1 = tm.u_bounds[sx], tm.u_bounds[sx + 1]
        U = u0 + (g + 1) / 2 * (u1 - u0)
        wu = w / 2 * (u1 - u0)
        for sy in range(Ny):
            v0, v1 = tm.v_bounds[sy], tm.v_bounds[sy + 1]
            Vv = v0 + (g + 1) / 2 * (v1 - v0)
            wv = w / 2 * (v1 - v0)
            x, y, xu, xv, yu, yv = tm.geom(sx, sy, U, Vv)
            det = xu * yv - xv * yu
            detmin = min(detmin, float(det.min()))
            A[sx, sy] = float(wu @ det @ wv)
    return A, detmin


def stack(M, cmap, n_orders=3):
    from lumenairy.elements.pmm import PMM2DStackPure
    c1, c2 = cells()
    st = PMM2DStackPure(P, P, n_superstrate=N_SUP, n_substrate=N_SUB,
                        n_modes=M, n_orders=n_orders, cmap=cmap)
    st.add_layer(D1, eps_cell=c1)
    st.add_layer(D2, eps_cell=c2)
    return st


def perlayer_stack(M, n_orders=3):
    from lumenairy.elements.pmm import PMM2DStackPure
    from lumenairy.elements.pmm.shapes2d import Circle, SinusoidalWall
    st = PMM2DStackPure(P, P, n_superstrate=N_SUP, n_substrate=N_SUB,
                        n_modes=M, n_orders=n_orders,
                        layer_grids="per-layer")
    st.add_layer(D1, shapes=[Circle(CX, CY, R, EPS_D)], background_eps=1.0)
    st.add_layer(D2, shapes=[SinusoidalWall("x", X0, AMP, eps=EPS_W)],
                 background_eps=1.0)
    return st
