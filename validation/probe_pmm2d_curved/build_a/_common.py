"""Shared fixtures and helpers of the Phase-A build probes (curved-cell map).

Every probe here drives the LIBRARY (``pmm_jones_2d_staggered(cmap=)`` /
``PMM2DStackPure(cmap=)``), not the planning scratch solver, and asserts that
``lumenairy`` is imported from the worktree it runs in.  Run every probe with
BLAS pinned on the command line:

  cd /c/tmp/lum_curved && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 PYTHONPATH=C:/tmp/lum_curved \
    python validation/probe_pmm2d_curved/build_a/<probe>.py

Fixture (the planning probes' P2 fixture): lambda = 1, square period 1.2,
depth 0.5, air above, n = 1.45 below, eps 4 features, normal incidence unless
stated.  ``stripe`` = y-uniform ridge x in [0.25, 0.85] (corner-free; exact
oracle ``pmm_efficiency_1d`` at degree 40), ``pillar`` = x in [0.25, 0.85],
y in [0.30, 0.90], ``film`` = uniform eps 4 (exact oracle: the Airy slab).
Map: ``x = u + a sin(2 pi u / p)``, ``y = v`` with the (u, v) walls at the
PREIMAGES of the physical walls, so the physical structure is unchanged.
"""
import os
import platform
import sys
from contextlib import contextmanager

import numpy as np

import lumenairy

ROOT = os.path.normcase(os.path.abspath(r"C:\tmp\lum_curved"))
assert os.path.normcase(os.path.abspath(lumenairy.__file__)).startswith(ROOT), (
    f"lumenairy imported from {lumenairy.__file__}, not the worktree {ROOT}")

from lumenairy.elements.pmm import (
    PMM2DStackPure,  # noqa: E402
    stack2d_pure as SP,  # noqa: E402
    twod_staggered as TS,  # noqa: E402
)
from lumenairy.elements.pmm._curvemap import (  # noqa: E402
    IdentityMap,
    SeparableStretch,
    SineStretch,
)

HERE = os.path.dirname(os.path.abspath(__file__))
P = 1.2
WL = 1.0
DEPTH = 0.5
N_SUP, N_SUB = 1.0, 1.45
XW = np.array([0.0, 0.25, 0.85, P])
YW = {"stripe": np.array([0.0, 0.4, 0.8, P]),
      "pillar": np.array([0.0, 0.30, 0.90, P]),
      "film": np.array([0.0, 0.4, 0.8, P])}
ORD9 = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1),
        (-1, -1)]


def cell(fixture, eps=4.0 + 0j):
    c = np.ones((3, 3), complex)
    if fixture == "stripe":
        c[1, :] = eps
    elif fixture == "pillar":
        c[1, 1] = eps
    else:
        c[:] = eps
    return c


def stretch_map(a_frac, fixture="stripe"):
    """The P2 map: walls at the preimages of the physical walls; a_frac = 0
    returns None (the unmapped solver on the same physical walls)."""
    if a_frac == 0:
        return None
    if fixture == "film":
        # a uniform film has no walls to keep: the (u, v) grid is the
        # UNIFORM 3 x 3 lattice (the planning probe's A3 fixture)
        return SeparableStretch(3, 3, fx=SineStretch(a_frac * P),
                                period_x=P, period_y=P)
    return SeparableStretch.from_physical_walls(
        XW, YW[fixture], fx=SineStretch(a_frac * P))


def solve(fixture, a_frac, M, n_orders=3, theta=0.0, phi=0.0,
          retain=False, cmap="stretch", eps=4.0 + 0j, depth=DEPTH):
    """Single layer through PMM2DStackPure; returns (orders, R, T, J, stack).

    ``a_frac = 0`` with ``cmap='stretch'`` is the UNMAPPED shipped solver on
    the same physical walls -- the shared path takes integer grids only, so
    it runs as a one-layer ``layer_grids='per-layer'`` stack with
    ``x_walls`` / ``y_walls`` (one patterned layer between the half-spaces is
    conforming, i.e. the shared cascade with non-uniform walls).
    ``cmap='identity'`` is the identity map through the quadrature path on
    the same walls."""
    if cmap == "stretch" and a_frac == 0 and fixture == "film":
        st = PMM2DStackPure(P, P, n_superstrate=N_SUP, n_substrate=N_SUB,
                            n_modes=M, n_orders=n_orders)
        st.add_layer(depth, eps_cell=cell(fixture, eps))
    elif cmap == "stretch" and a_frac == 0:
        st = PMM2DStackPure(P, P, n_superstrate=N_SUP, n_substrate=N_SUB,
                            n_modes=M, n_orders=n_orders,
                            layer_grids="per-layer")
        st.add_layer(depth, eps_cell=cell(fixture, eps), x_walls=XW,
                     y_walls=YW[fixture])
    else:
        if cmap == "stretch":
            cm = stretch_map(a_frac, fixture)
        elif cmap == "identity":
            cm = IdentityMap(XW, YW[fixture])
        else:
            cm = cmap
        st = PMM2DStackPure(P, P, n_superstrate=N_SUP, n_substrate=N_SUB,
                            n_modes=M, n_orders=n_orders, cmap=cm)
        st.add_layer(depth, eps_cell=cell(fixture, eps))
    st.set_source(WL, theta=theta, phi=phi)
    o, R, T, J = st.solve(retain_internal=retain)
    return o, R, T, J, st


def vec(o, R, T, orders=ORD9):
    """[R(orders); T(orders)] for BOTH incident polarizations (rows)."""
    idx = [int(np.nonzero((o[:, 0] == m) & (o[:, 1] == n))[0][0])
           for m, n in orders]
    return np.concatenate([R[:, idx].ravel(), T[:, idx].ravel()])


def airy(n2=2.0, depth=DEPTH):
    k0 = 2 * np.pi / WL
    r12 = (N_SUP - n2) / (N_SUP + n2)
    r23 = (n2 - N_SUB) / (n2 + N_SUB)
    t12 = 2 * N_SUP / (N_SUP + n2)
    t23 = 2 * n2 / (n2 + N_SUB)
    ph = np.exp(1j * n2 * k0 * depth)
    R = abs((r12 + r23 * ph ** 2) / (1 + r12 * r23 * ph ** 2)) ** 2
    T = abs(t12 * t23 * ph / (1 + r12 * r23 * ph ** 2)) ** 2 * N_SUB / N_SUP
    return float(R), float(T)


class NoCofactorView:
    """FAIL-BEFORE view of a map for the FAR PROJECTOR ONLY: positions kept
    (the kernel exp(+i k . Phi) is right), the cofactor replaced by the
    identity -- i.e. the covariant coefficients read as if they were
    Cartesian (planning probe P2b arm ``trap_F_no_cofactor``)."""

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
def mixed_hgram():
    """Engineered defect: the half-spaces' H partner through -R (the shipped
    unmapped ``_homog_geom_cache`` behaviour) while the patterned layer keeps
    the plain Gram -- the planning probe's ``trap_Hmix`` arm."""
    orig = SP._homog_geom_cache

    def patched(solver):
        W0, g2, GW0, SttW0, _Ginv, qq = orig(solver)
        return W0, g2, GW0, SttW0, np.linalg.inv(-solver.Rmat), qq
    SP._homog_geom_cache = patched
    try:
        yield
    finally:
        SP._homog_geom_cache = orig


def env_record():
    import scipy
    return {"python": sys.version.split()[0], "numpy": np.__version__,
            "scipy": scipy.__version__, "lumenairy": lumenairy.__file__,
            "machine": platform.node(),
            "threads": {k: os.environ.get(k) for k in
                        ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                         "MKL_NUM_THREADS")}}


__all__ = ["IdentityMap", "SeparableStretch", "SineStretch", "TS", "SP",
           "PMM2DStackPure"]
