"""Shared fixtures of the INDEPENDENT Phase-B verification probes (curved
cells for the pure staggered 2-D PMM: transfinite maps, the corner rule).

Every map here is built by THIS file from the public primitives
(``TransfiniteMap``, ``Line``, ``Arc``, ``Arc.through``, ``EllipseArc``,
``Sinusoid``) -- the builder's private ``_circle_map_3x3`` / ``_fillet_map_5x5``
/ ... are NOT used except where a probe compares against them on purpose.

Run every probe with BLAS pinned on the command line and the worktree on
PYTHONPATH:

  cd /c/tmp/lum_vcurved_b && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 PYTHONPATH=C:/tmp/lum_vcurved_b \
    python validation/probe_pmm2d_curved/verify_b/<probe>.py ...

Fixture (the planning / build fixture, so the saved FEM applies): lambda 1,
square period 1.2, depth 0.5, air over n = 1.45, eps 4 features.  Row 0 of
R / T is the input E along x, row 1 E along y.
"""
import json
import os
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normcase(os.path.abspath(os.path.join(HERE, "..", "..", "..")))
import lumenairy  # noqa: E402

LUM_FILE = os.path.normcase(os.path.abspath(lumenairy.__file__))
assert LUM_FILE.startswith(ROOT), (
    f"lumenairy imported from {lumenairy.__file__}, not the tree {ROOT}")

from lumenairy.elements.pmm import (  # noqa: E402
    PMM2DStackPure,
    _curvemap as CM,
    stack2d_pure as SP,  # noqa: F401  (re-exported for the probes)
    twod_staggered as TS,  # noqa: F401  (re-exported for the probes)
)

P = 1.2
WL = 1.0
K0 = 2 * np.pi / WL
DEPTH = 0.5
N_SUP, N_SUB = 1.0, 1.45
EPS_P = 4.0
ORD9 = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1),
        (-1, -1)]
DEG = np.pi / 180.0


# --------------------------------------------------------------------------
# maps, built here from the primitives
# --------------------------------------------------------------------------
def grid_vertices(uw, vw):
    V = np.empty((len(uw), len(vw), 2))
    V[..., 0] = np.asarray(uw)[:, None]
    V[..., 1] = np.asarray(vw)[None, :]
    return V


def vcircle3(r, center=None, period=P):
    """Circle of radius r: 3 x 3 walls through the 45-degree points, the
    four edges of the middle cell = Arc.through(...) of the identity vertex
    images.  Returns (map, eps_cell)."""
    c = np.array([period / 2, period / 2] if center is None else center,
                 float)
    h = r / np.sqrt(2.0)
    uw = np.array([0.0, c[0] - h, c[0] + h, period])
    vw = np.array([0.0, c[1] - h, c[1] + h, period])
    V = grid_vertices(uw, vw)
    ed = {("h", 1, 1): Arc_through(V[1, 1], V[2, 1], c),
          ("h", 1, 2): Arc_through(V[1, 2], V[2, 2], c),
          ("v", 1, 1): Arc_through(V[1, 1], V[1, 2], c),
          ("v", 2, 1): Arc_through(V[2, 1], V[2, 2], c)}
    cm = CM.TransfiniteMap(uw, vw, V, ed)
    eps = np.ones((3, 3), complex)
    eps[1, 1] = EPS_P
    return cm, eps


def Arc_through(P0, P1, c):
    return CM.Arc.through(P0, P1, c)


def vcircle5(r, inner=0.6, period=P):
    """Circle on a 5 x 5 grid, built here: the inner straight square has
    half-size inner * r / sqrt 2; the loop's side vertices sit on the circle
    at the same u (or v) as the inner walls' ANGLE (a different construction
    from the builder's: the side vertex of inner wall u = c + h is the circle
    point at polar angle asin(h / r) measured from the axis)."""
    c = np.array([period / 2, period / 2])
    h45 = r / np.sqrt(2.0)
    hi = inner * h45
    w = np.array([0.0, c[0] - h45, c[0] - hi, c[0] + hi, c[0] + h45,
                  period])
    V = grid_vertices(w, w)
    ph = np.arcsin(hi / r)

    def on(th):
        return c + r * np.array([np.cos(th), np.sin(th)])
    # loop side vertices (bottom j=1, top j=4, left i=1, right i=4)
    V[2, 1], V[3, 1] = on(-np.pi / 2 - ph), on(-np.pi / 2 + ph)
    V[2, 4], V[3, 4] = on(np.pi / 2 + ph), on(np.pi / 2 - ph)
    V[1, 2], V[1, 3] = on(np.pi + ph), on(np.pi - ph)
    V[4, 2], V[4, 3] = on(-ph), on(ph)
    ed = {}
    for i in (1, 2, 3):
        for j in (1, 4):
            ed[("h", i, j)] = Arc_through(V[i, j], V[i + 1, j], c)
    for j in (1, 2, 3):
        for i in (1, 4):
            ed[("v", i, j)] = Arc_through(V[i, j], V[i, j + 1], c)
    cm = CM.TransfiniteMap(w, w, V, ed)
    eps = np.ones((5, 5), complex)
    eps[1:4, 1:4] = EPS_P
    return cm, eps


def vfillet(side, rf, split_mid=False, split_out=False, period=P,
            extra=()):
    """Square pillar of SIDE with its corners rounded to radius rf, built
    here from Arc.through on vertex images: walls through the 45-degree
    fillet points (a) and the tangency points (b); optional extra STRAIGHT
    walls at the centre (split_mid, the pillar's straight middle) and in the
    air gaps (split_out) -- an h-refinement of the same topology, which gives
    an independent ladder at fixed M; ``extra`` adds further straight walls
    (inside the pillar's straight middle or in the air gaps -- a GRADING
    toward the fillet).  Returns (map, eps_cell)."""
    c = period / 2
    lo, hi = c - side / 2, c + side / 2
    a = lo + rf * (1.0 - 1.0 / np.sqrt(2.0))
    b = lo + rf
    w = [0.0, a, b, period - b, period - a, period]
    if split_mid:
        w.insert(3, c)
    if split_out:
        w.insert(1, a / 2)
        w.insert(len(w) - 1, period - a / 2)
    w = np.array(sorted(set(w) | {float(e) for e in extra}))
    n = w.size
    ia, ib = int(np.argmin(abs(w - a))), int(np.argmin(abs(w - b)))
    ja, jb = int(np.argmin(abs(w - (period - a)))), int(
        np.argmin(abs(w - (period - b))))
    V = grid_vertices(w, w)
    # vertices on the pillar boundary: the u = b ... P - b walls at v = a map
    # onto y = lo (bottom side), etc.; the 45-degree points stay put
    for i in range(n):
        for j in range(n):
            x, y = w[i], w[j]
            onx = ib <= i <= jb       # between the tangency walls in u
            ony = ib <= j <= jb
            if onx and j == ia:
                V[i, j] = (x, lo)
            elif onx and j == ja:
                V[i, j] = (x, hi)
            elif ony and i == ia:
                V[i, j] = (lo, y)
            elif ony and i == ja:
                V[i, j] = (hi, y)
    cBL, cBR = np.array([b, b]), np.array([period - b, b])
    cTL, cTR = np.array([b, period - b]), np.array([period - b, period - b])
    ed = {("h", ia, ia): Arc_through(V[ia, ia], V[ib, ia], cBL),
          ("v", ia, ia): Arc_through(V[ia, ia], V[ia, ib], cBL),
          ("h", jb, ia): Arc_through(V[jb, ia], V[ja, ia], cBR),
          ("v", ja, ia): Arc_through(V[ja, ia], V[ja, ib], cBR),
          ("h", ia, ja): Arc_through(V[ia, ja], V[ib, ja], cTL),
          ("v", ia, jb): Arc_through(V[ia, jb], V[ia, ja], cTL),
          ("h", jb, ja): Arc_through(V[jb, ja], V[ja, ja], cTR),
          ("v", ja, jb): Arc_through(V[ja, jb], V[ja, ja], cTR)}
    cm = CM.TransfiniteMap(w, w, V, ed)
    eps = np.ones((n - 1, n - 1), complex)
    eps[ia:ja, ia:ja] = EPS_P
    return cm, eps


def vellipse3(a, b, period=P):
    """Axis-aligned ellipse (semi-axes a along x, b along y) on 3 x 3 walls
    through the parametric 45-degree points; built here."""
    c = np.array([period / 2, period / 2])
    uw = np.array([0.0, c[0] - a / np.sqrt(2), c[0] + a / np.sqrt(2), period])
    vw = np.array([0.0, c[1] - b / np.sqrt(2), c[1] + b / np.sqrt(2), period])
    V = grid_vertices(uw, vw)
    E = CM.EllipseArc
    ed = {("h", 1, 1): E(c, (a, b), 225 * DEG, 315 * DEG),
          ("h", 1, 2): E(c, (a, b), 135 * DEG, 45 * DEG),
          ("v", 1, 1): E(c, (a, b), 225 * DEG, 135 * DEG),
          ("v", 2, 1): E(c, (a, b), -45 * DEG, 45 * DEG)}
    cm = CM.TransfiniteMap(uw, vw, V, ed)
    eps = np.ones((3, 3), complex)
    eps[1, 1] = EPS_P
    return cm, eps


def vsine_ridge(x1, x2, A, vw=None, period=P):
    """Constant-width ridge between x = x1 + A sin(2 pi y / p) and x = x2 +
    A sin(2 pi y / p); 3 columns, v walls vw (default 3 rows at 0, 0.3,
    0.75 p: a different v grid from the builder's thirds; the solver needs
    Nx == Ny)."""
    vw = (np.array([0.0, 0.3, 0.75, 1.0]) * period if vw is None
          else np.asarray(vw, float))
    uw = np.array([0.0, x1, x2, period])
    k = 2 * np.pi / period
    V = grid_vertices(uw, vw)
    for i in (1, 2):
        V[i, :, 0] += A * np.sin(k * vw)
    ed = {}
    for i, base in ((1, x1), (2, x2)):
        for j in range(vw.size - 1):
            ed[("v", i, j)] = CM.Sinusoid(base, A, period, vw[j], vw[j + 1])
    cm = CM.TransfiniteMap(uw, vw, V, ed)
    eps = np.ones((3, vw.size - 1), complex)
    eps[1, :] = EPS_P
    return cm, eps


def vyuniform_curved(x1, x2, xs, A, vw=None, period=P):
    """A y-UNIFORM lamellar stripe eps 4 on x1 < x < x2 (straight walls),
    with one extra grid line x = xs + A sin(2 pi y / p) INSIDE the air
    (a fictitious curved wall: the same material on both sides).  The map is
    genuinely curved (non-separable), the physical structure depends on x
    only, so at ANY incidence the only k_y present is the incident one: all
    power in orders with n != 0 is numerical."""
    vw = (np.array([0.0, 0.2, 0.45, 0.7, 1.0]) * period if vw is None
          else np.asarray(vw, float))
    uw = np.array([0.0, xs, x1, x2, period])
    k = 2 * np.pi / period
    V = grid_vertices(uw, vw)
    V[1, :, 0] += A * np.sin(k * vw)
    ed = {("v", 1, j): CM.Sinusoid(xs, A, period, vw[j], vw[j + 1])
          for j in range(vw.size - 1)}
    cm = CM.TransfiniteMap(uw, vw, V, ed)
    eps = np.ones((4, vw.size - 1), complex)
    eps[2, :] = EPS_P
    return cm, eps


# --------------------------------------------------------------------------
# solves
# --------------------------------------------------------------------------
def solve_map(cmap, eps_cell, M, theta=0.0, phi=0.0, n_orders=3,
              depth=DEPTH):
    import warnings
    st = PMM2DStackPure(P, P, n_superstrate=N_SUP, n_substrate=N_SUB,
                        n_modes=M, n_orders=n_orders, cmap=cmap)
    st.add_layer(depth, eps_cell=eps_cell)
    st.set_source(WL, theta=theta, phi=phi)
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        o, R, T, J = st.solve()
    return np.asarray(o), np.asarray(R), np.asarray(T), st, [
        str(w.message)[:160] for w in wl]


def solve_walls(xw, yw, eps_cell, M, theta=0.0, phi=0.0, n_orders=3):
    st = PMM2DStackPure(P, P, n_superstrate=N_SUP, n_substrate=N_SUB,
                        n_modes=M, n_orders=n_orders, layer_grids="per-layer")
    st.add_layer(DEPTH, eps_cell=eps_cell, x_walls=xw, y_walls=yw,
                 max_pencil_dof=40000)
    st.set_source(WL, theta=theta, phi=phi)
    o, R, T = st.solve(jones=False)
    return np.asarray(o), np.asarray(R), np.asarray(T)


def idx(o, orders=ORD9):
    return [int(np.nonzero((o[:, 0] == m) & (o[:, 1] == n))[0][0])
            for m, n in orders]


def vec(o, R, T, orders=ORD9):
    i = idx(o, orders)
    return np.concatenate([R[:, i].ravel(), T[:, i].ravel()])


def env():
    import scipy
    return {"python": sys.version.split()[0], "numpy": np.__version__,
            "scipy": scipy.__version__, "lumenairy": lumenairy.__file__,
            "machine": platform.node(), "platform": platform.platform(),
            "threads": {k: os.environ.get(k) for k in
                        ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                         "MKL_NUM_THREADS")},
            "when": time.strftime("%Y-%m-%d %H:%M:%S")}


def build_tag():
    return "wsl" if platform.system() == "Linux" else "win"


def dump(name, res):
    def conv(x):
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, complex):
            return [x.real, x.imag]
        raise TypeError(type(x))
    res = dict(res)
    res.setdefault("env", env())
    with open(os.path.join(HERE, name), "w") as f:
        json.dump(res, f, indent=1, default=conv)


def load(name):
    with open(os.path.join(HERE, name)) as f:
        return json.load(f)
