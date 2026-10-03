"""D3 -- the corner (Duffy) rule with a TENSOR (and magnetic) weight.

Phase B measured the corner rule on SCALAR cells: operators to round-off by
n = 16.  A tensor weight adds the individual Jacobian products
adj(J)_ik adj(J)_jl / sg (and J_ki J_lj / sg for a material mu) -- the same
1/r singularity at the four 45-degree vertices, but not the five metric
combinations the adaptive node criterion measures.  This probe:

* forces the node count n (8 .. 48) and reads the OPERATORS (L, R) against
  the top rung, for the corner rule and for the plain tensor rule;
* reads the node count the adaptive criterion picks (n0) and the operator
  change from n0 to 2 n0 -- is n0 adequate for the tensor weights;
* the same n0 vs 2 n0 reading under the a = 0.15 p sine stretch (where the
  criterion doubles), with the LC tensor.

Cells: the 3 x 3 and 5 x 5 circle maps (P = 1.2, r = 0.36); the disk filled
with LC30 (director at 30 deg), host eps 1; arm 'epsmu' adds mu = diag(2, 2,
1) in the disk.

usage: python d3_quad.py <c3|c5> <M>      -> d3_quad_<map>_M<M>.json
       python d3_quad.py stretch <M>       -> d3_quad_stretch_M<M>.json
"""
import sys
from contextlib import contextmanager

import _dcommon as D
import numpy as np

TS = D.TS
P = 1.2
K0 = 2 * np.pi


@contextmanager
def nodes(n):
    orig = TS._stag_map_nodes
    TS._stag_map_nodes = lambda *a, **k: int(n)
    try:
        yield
    finally:
        TS._stag_map_nodes = orig


@contextmanager
def plain():
    orig = TS._stag_map_singular_corners
    TS._stag_map_singular_corners = lambda c: {}
    try:
        yield
    finally:
        TS._stag_map_singular_corners = orig


def cells(kind):
    cm = D.make_map(kind, P)
    N = cm.shape[0]
    eps = np.broadcast_to(np.eye(3, dtype=complex), (N, N, 3, 3)).copy()
    mu = np.broadcast_to(np.eye(3, dtype=complex), (N, N, 3, 3)).copy()
    disk = [(1, 1)] if N == 3 else [(i, j) for i in (1, 2, 3)
                                    for j in (1, 2, 3)]
    for c in disk:
        eps[c] = D.LC30
        mu[c] = np.diag([2.0, 2.0, 1.0])
    return cm, eps, mu


def ops(cm, eps, mu, M):
    s = TS.Granet2DTransverseE(P, P, cm.u_walls, cm.v_walls, M, eps, k0=K0,
                               mu_cell=mu, cmap=cm)
    return s.Lmat, s.Rmat


def rel(a, b):
    sc = max(np.abs(b[0]).max(), np.abs(b[1]).max())
    return float(max(np.abs(a[0] - b[0]).max(), np.abs(a[1] - b[1]).max())
                 / sc)


def run(kind, M, ns=(8, 12, 16, 20, 24, 32, 48)):
    cm, eps, mu = cells(kind)
    out = {"map": kind, "M": M, "ns": list(ns)}
    for arm, m in (("eps", None), ("epsmu", mu)):
        cor, pla = {}, {}
        for n in ns:
            with nodes(n):
                cor[n] = ops(cm, eps, m, M)
                with plain():
                    pla[n] = ops(cm, eps, m, M)
        top = cor[ns[-1]]
        out[arm] = {"corner_to_top": [rel(cor[n], top) for n in ns],
                    "plain_to_corner_top": [rel(pla[n], top) for n in ns]}
        n0 = TS._stag_map_nodes(
            TS.Basis1D(P, cm.u_walls, M), TS.Basis1D(P, cm.v_walls, M), cm, M)
        a = ops(cm, eps, m, M)
        with nodes(2 * n0):
            b = ops(cm, eps, m, M)
        out[arm]["adaptive_n0"] = int(n0)
        out[arm]["adaptive_n0_vs_2n0"] = rel(a, b)
        out[arm]["adaptive_to_corner_top"] = rel(a, top)
        print(kind, M, arm, out[arm], flush=True)
    D.dump(f"d3_quad_{kind}_M{M}.json", out)


def run_stretch(M):
    out = {"M": M}
    for a in (0.05, 0.15):
        cm = D.stretch_map(P, a)
        eps = np.broadcast_to(np.eye(3, dtype=complex), (3, 3, 3, 3)).copy()
        eps[1, :] = D.LC30
        mu = np.broadcast_to(np.eye(3, dtype=complex), (3, 3, 3, 3)).copy()
        mu[1, :] = np.diag([2.0, 2.0, 1.0])
        n0 = TS._stag_map_nodes(
            TS.Basis1D(P, cm.u_walls, M), TS.Basis1D(P, cm.v_walls, M), cm, M)
        row = {"n0": int(n0)}
        for arm, m in (("eps", None), ("epsmu", mu)):
            x = ops(cm, eps, m, M)
            with nodes(2 * n0):
                y = ops(cm, eps, m, M)
            with nodes(4 * n0):
                z = ops(cm, eps, m, M)
            row[arm] = {"n0_vs_4n0": rel(x, z), "2n0_vs_4n0": rel(y, z)}
        out[f"a{a}"] = row
        print(a, row, flush=True)
    D.dump(f"d3_quad_stretch_M{M}.json", out)


if __name__ == "__main__":
    if sys.argv[1] == "stretch":
        run_stretch(int(sys.argv[2]))
    else:
        run(sys.argv[1], int(sys.argv[2]))
