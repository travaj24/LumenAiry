"""Shared fixtures and helpers of the Phase-B build probes (transfinite maps:
the circle, the fillet, the ellipse, the sinusoidal wall).

Every probe drives the LIBRARY (``PMM2DStackPure(cmap=)``) and asserts that
``lumenairy`` is imported from the worktree that holds this file.  Run every
probe with BLAS pinned on the command line:

  cd /c/tmp/lum_curved_b && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 PYTHONPATH=C:/tmp/lum_curved_b \
    python validation/probe_pmm2d_curved/build_b/<probe>.py

Fixture (the planning probes' P3 / P4 fixture): lambda = 1, square period
1.2, depth 0.5, air above, n = 1.45 below, eps 4 (n = 2) features, normal
incidence unless stated; the circle has r = 0.36 = 0.3 p; the fillet pillar
side 0.6.  Row 0 of R / T is the 'tm' input (E along x), row 1 'te'
(E along y) -- the library's single-layer convention at normal incidence.
"""
import json
import os
import platform
import sys
from contextlib import contextmanager

import numpy as np

import lumenairy

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normcase(os.path.abspath(os.path.join(HERE, "..", "..", "..")))
assert os.path.normcase(os.path.abspath(lumenairy.__file__)).startswith(ROOT), (
    f"lumenairy imported from {lumenairy.__file__}, not the worktree {ROOT}")

from lumenairy.elements.pmm import (
    PMM2DStackPure,  # noqa: E402
    _curvemap as CM,  # noqa: E402
    stack2d_pure as SP,  # noqa: E402
    twod_staggered as TS,  # noqa: E402
)

P = 1.2
WL = 1.0
DEPTH = 0.5
N_SUP, N_SUB = 1.0, 1.45
R_CIRC = 0.36
EPS_P = 4.0
ORD9 = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1),
        (-1, -1)]


def circle3():
    cm, w = CM._circle_map_3x3(P, R_CIRC)
    eps = np.ones((3, 3), complex)
    eps[1, 1] = EPS_P
    return cm, eps


def circle5():
    cm, w = CM._circle_map_5x5(P, R_CIRC)
    eps = np.ones((5, 5), complex)
    eps[1:4, 1:4] = EPS_P
    return cm, eps


def fillet5(ratio, side=0.6):
    cm, w = CM._fillet_map_5x5(P, side / 2, ratio * side)
    eps = np.ones((5, 5), complex)
    eps[1:4, 1:4] = EPS_P
    return cm, eps


def solve(cmap, eps_cell, M, n_orders=3, theta=0.0, phi=0.0, retain=False,
          depth=DEPTH, wl=WL, n_sup=N_SUP, n_sub=N_SUB):
    """Single layer through PMM2DStackPure under ``cmap`` (``None`` = the
    shipped shared-grid solver on an INTEGER grid); (orders, R, T, J)."""
    st = PMM2DStackPure(P, P, n_superstrate=n_sup, n_substrate=n_sub,
                        n_modes=M, n_orders=n_orders, cmap=cmap)
    st.add_layer(depth, eps_cell=eps_cell)
    st.set_source(wl, theta=theta, phi=phi)
    o, R, T, J = st.solve(retain_internal=retain)
    return np.asarray(o), np.asarray(R), np.asarray(T), J


def solve_walls(x_walls, y_walls, eps_cell, M, n_orders=3, theta=0.0,
                phi=0.0, max_pencil_dof=None):
    """The SHIPPED unmapped solver on explicit (non-uniform) walls: a
    one-layer ``layer_grids='per-layer'`` stack (one patterned layer between
    the half-spaces is conforming)."""
    st = PMM2DStackPure(P, P, n_superstrate=N_SUP, n_substrate=N_SUB,
                        n_modes=M, n_orders=n_orders, layer_grids="per-layer")
    kw = {} if max_pencil_dof is None else {"max_pencil_dof": max_pencil_dof}
    st.add_layer(DEPTH, eps_cell=eps_cell, x_walls=x_walls, y_walls=y_walls,
                 **kw)
    st.set_source(WL, theta=theta, phi=phi)
    o, R, T = st.solve(jones=False)
    return np.asarray(o), np.asarray(R), np.asarray(T)


def idx_of(o, orders=ORD9):
    return [int(np.nonzero((o[:, 0] == m) & (o[:, 1] == n))[0][0])
            for m, n in orders]


def vec(o, R, T, orders=ORD9):
    """[R(orders); T(orders)] for BOTH incident polarizations."""
    idx = idx_of(o, orders)
    return np.concatenate([R[:, idx].ravel(), T[:, idx].ravel()])


def table(o, R, T, row, orders=ORD9):
    idx = idx_of(o, orders)
    return {f"{m},{n}": [float(R[row, i]), float(T[row, i])]
            for (m, n), i in zip(orders, idx)}


@contextmanager
def fixed_nodes(nq):
    """Force the mapped assembly's per-axis node count to ``nq`` (the
    quadrature ladder's knob): patches the ONE node-count function."""
    orig = TS._stag_map_nodes
    TS._stag_map_nodes = lambda *a, **k: int(nq)
    try:
        yield
    finally:
        TS._stag_map_nodes = orig


def env_record():
    import scipy
    return {"python": sys.version.split()[0], "numpy": np.__version__,
            "scipy": scipy.__version__, "lumenairy": lumenairy.__file__,
            "machine": platform.node(),
            "threads": {k: os.environ.get(k) for k in
                        ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                         "MKL_NUM_THREADS")}}


def dump(name, res):
    with open(os.path.join(HERE, name), "w") as f:
        json.dump(res, f, indent=1)


def load(name, sub=None):
    base = HERE if sub is None else os.path.join(HERE, "..", sub)
    with open(os.path.join(base, name)) as f:
        return json.load(f)


class NoCofactorView:
    """FAIL-BEFORE view of a map for the FAR PROJECTOR ONLY: positions kept
    (the kernel exp(+i k . Phi) is right), the cofactor replaced by the
    identity -- the covariant coefficients read as if they were Cartesian
    (planning probe P2b arm ``trap_F_no_cofactor``; Phase A's engineered
    defect, ``build_a/_common.py``)."""

    def __init__(self, cmap):
        self._m = cmap

    def geom(self, sx, sy, U, V):
        X, Y, xu, xv, yu, yv = self._m.geom(sx, sy, U, V)
        return X, Y, np.ones_like(xu), np.zeros_like(xv), np.zeros_like(yu), \
            np.ones_like(yv)


@contextmanager
def no_cofactor():
    """Engineered defect through the REAL stack path: the stack's far
    projector call is handed the no-cofactor view of its map."""
    orig = SP._far_projector_2d

    def patched(bx, by, ox, oy, a0x=0.0, a0y=0.0, cmap=None):
        if cmap is not None:
            cmap = NoCofactorView(cmap)
        return orig(bx, by, ox, oy, a0x, a0y, cmap=cmap)
    SP._far_projector_2d = patched
    try:
        yield
    finally:
        SP._far_projector_2d = orig


@contextmanager
def plain_rule():
    """The Phase-A tensor rule in EVERY cell (no corner / Duffy rule): the
    alternative (a) of the quadrature decision, used as a comparison arm."""
    orig = TS._stag_map_singular_corners
    TS._stag_map_singular_corners = lambda cmap: {}
    try:
        yield
    finally:
        TS._stag_map_singular_corners = orig


def airy_sp(theta, n2=2.0, depth=DEPTH):
    """|r_s|^2, |r_p|^2 of a uniform n2 film between N_SUP and N_SUB."""
    k0 = 2 * np.pi / WL
    ns = (N_SUP, n2, N_SUB)
    st = N_SUP * np.sin(theta)
    kz = [np.sqrt(complex(n * n - st * st)) for n in ns]

    def slab(r12, r23):
        ph = np.exp(2j * kz[1] * k0 * depth)
        return abs((r12 + r23 * ph) / (1 + r12 * r23 * ph)) ** 2

    rs = slab((kz[0] - kz[1]) / (kz[0] + kz[1]),
              (kz[1] - kz[2]) / (kz[1] + kz[2]))
    e = [n * n for n in ns]
    rp = slab((e[1] * kz[0] - e[0] * kz[1]) / (e[1] * kz[0] + e[0] * kz[1]),
              (e[2] * kz[1] - e[1] * kz[2]) / (e[2] * kz[1] + e[1] * kz[2]))
    return rs, rp


def r_exact_rows(theta, phi, n2=2.0):
    """Exact reflectance for incident lab E_x (row 0) and E_y (row 1):
    E_t = (1, 0) / (0, 1) = a s_t + b cos(theta) k_t with s_t = (-sin phi,
    cos phi), k_t = (cos phi, sin phi) (Phase A's ``a3_film.py``)."""
    rs, rp = airy_sp(theta, n2)
    out = []
    for et in ((1.0, 0.0), (0.0, 1.0)):
        a = -np.sin(phi) * et[0] + np.cos(phi) * et[1]
        b = (np.cos(phi) * et[0] + np.sin(phi) * et[1]) / np.cos(theta)
        out.append((a * a * rs + b * b * rp) / (a * a + b * b))
    return np.array(out)


def film_err(o, R, T, theta=0.0, phi=0.0, n2=2.0):
    """max over both inputs and EVERY order of the distance to the exact
    lossless film: the specular order carries R_exact / 1 - R_exact, every
    other order zero."""
    Rx = r_exact_rows(theta, phi, n2)
    i0 = int(np.nonzero((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
    R = R.copy()
    T = T.copy()
    R[:, i0] -= Rx
    T[:, i0] -= 1.0 - Rx
    return float(max(np.abs(R).max(), np.abs(T).max()))
