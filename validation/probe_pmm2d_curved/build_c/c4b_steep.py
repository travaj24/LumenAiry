"""C4b -- F5 for the shape layer: what a STEEP cell costs, and whether
subdividing it pays.  A uniform eps-4 film (exact answer: the Airy slab) laid
on the map of a sinusoidal ridge (x0 = 0.3, width 0.5, amplitude A, one wave
per cell): the only error is the map's.  The ridge's walls are grid lines, so
the cells on either side blend the sinusoid into the straight cell edge; at
large A that blend is steep (the local stretch x_u runs (0.3 -+ A) / 0.3).

  c4b_steep.py <A> <M>

Arms (same device, same outline, different subdivision):
  coarse  -- compile_shapes default (u walls 0.3, 0.8; v squared up)
  hint5   -- grid_hint=5 (halves the WIDEST segments: not the steep ones)
  steep   -- the steep transition columns halved in u and the rows in v
             (u walls + 0.15, 0.95 + 0.05...; v quartered), through
             RefinedMap on the same base map
Also the per-cell steepness (max / min over a 16 x 16 Gauss grid of the
largest singular value of J) of the coarse map.
Output: c4b_steep_A<A>_M<M>.json
"""
import sys

import _common as C
import numpy as np
from numpy.polynomial.legendre import leggauss

from lumenairy.elements.pmm import SinusoidalWall, compile_shapes  # noqa

CM = C.CM


def airy():
    k0 = 2 * np.pi / C.WL
    n = (C.N_SUP, 2.0, C.N_SUB)
    r01 = (n[0] - n[1]) / (n[0] + n[1])
    r12 = (n[1] - n[2]) / (n[1] + n[2])
    ph = np.exp(2j * n[1] * k0 * C.DEPTH)
    return abs((r01 + r12 * ph) / (1 + r01 * r12 * ph)) ** 2


def steepness(cm):
    xg, _ = leggauss(16)
    out = []
    for sx in range(cm.shape[0]):
        U = 0.5 * (cm.u_bounds[sx] + cm.u_bounds[sx + 1]) + 0.5 * (
            cm.u_bounds[sx + 1] - cm.u_bounds[sx]) * xg
        for sy in range(cm.shape[1]):
            V = 0.5 * (cm.v_bounds[sy] + cm.v_bounds[sy + 1]) + 0.5 * (
                cm.v_bounds[sy + 1] - cm.v_bounds[sy]) * xg
            _X, _Y, xu, xv, yu, yv = cm.geom(sx, sy, U, V)
            J = np.stack([np.stack([xu, xv], -1), np.stack([yu, yv], -1)],
                         -2)
            sv = np.linalg.svd(J, compute_uv=False)[..., 0]
            out.append(float(sv.max() / sv.min()))
    return out


def run(A, M):
    sh = SinusoidalWall("x", 0.3, A, eps=4.0, width=0.5)
    eps0, xw, yw, cm = compile_shapes(C.P, C.P, [sh], 1.0)
    _e6, _x6, _y6, cm6 = compile_shapes(C.P, C.P, [sh], 1.0, grid_hint=5)
    base = cm.base if isinstance(cm, CM.RefinedMap) else cm
    us = np.array([0.0, 0.15, 0.3, 0.8, 1.0, 1.2])
    vs = np.linspace(0.0, C.P, 6)
    cms = CM.RefinedMap(base, us, vs)
    Rx = airy()
    res = {"env": C.env_record(), "A": A, "M": M,
           "steep_coarse": steepness(cm), "steep_fine": steepness(cms)}
    for name, m in (("coarse", cm), ("hint5", cm6), ("steep", cms)):
        n = m.shape[0]
        o, R, T, _J = C.solve(m, np.full((n, n), 4.0 + 0j), M)
        i0 = C.idx(o, [(0, 0)])[0]
        R = R.copy()
        T = T.copy()
        R[:, i0] -= Rx
        T[:, i0] -= 1.0 - Rx
        res[name] = {"err": float(max(np.abs(R).max(), np.abs(T).max())),
                     "grid": list(m.shape)}
    print({k: v for k, v in res.items() if k not in ("env",)}, flush=True)
    C.dump(f"c4b_steep_A{A}_M{M}.json", res)


if __name__ == "__main__":
    run(float(sys.argv[1]), int(sys.argv[2]))
