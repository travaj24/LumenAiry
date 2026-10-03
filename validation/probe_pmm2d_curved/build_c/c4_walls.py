"""C4 -- the F5 trap is closed by construction: a primitive places every wall
at the PREIMAGE of its physical boundary (the circle's 45-degree points are
grid vertices at their own positions), so the map is as gentle as the
geometry allows.  The SAME exact disk can be laid on wrong walls by hand --
the arcs unchanged, the disk cell's corner vertices MOVED onto the
45-degree points from walls that are not there -- and the device is
identical (the disk is exact either way) while the map is not.

  c4_walls.py film <M>       uniform eps-4 film under each map vs Airy
  c4_walls.py pillar <M>     the circular pillar under each map
  c4_walls.py steep          per-cell steepness of each map (no solve)

Maps (all the 3 x 3 topology, r = 0.36 in p = 1.2, the disk = cell (1, 1)):
  prim     -- Circle(0.6, 0.6, 0.36) through compile_shapes (walls at
              c -+ r / sqrt 2 = 0.3454, 0.8546)
  lattice  -- the shipped uniform lattice walls 0.4, 0.8 (what an integer
              eps_cell grid gives), corners moved to the 45-degree points
  inner    -- walls at c -+ 0.8 r / sqrt 2, corners moved out to them
Output: c4_walls_<arm>[_M<M>].json
"""
import sys

import _common as C
import numpy as np
from numpy.polynomial.legendre import leggauss

from lumenairy.elements.pmm import Circle, compile_shapes  # noqa: E402

CM = C.CM
DEG = np.pi / 180


def hand_map(walls):
    """The 3 x 3 circle on the given interior walls (u = v), disk-cell
    corners moved to the circle's 45-degree points, the same four arcs."""
    c = (C.P / 2, C.P / 2)
    r = C.R_CIRC
    h = r / np.sqrt(2.0)
    w = np.array([0.0, walls[0], walls[1], C.P])
    V = np.empty((4, 4, 2))
    V[..., 0] = w[:, None]
    V[..., 1] = w[None, :]
    for i, x in ((1, c[0] - h), (2, c[0] + h)):
        for j, y in ((1, c[1] - h), (2, c[1] + h)):
            V[i, j] = (x, y)
    curved = {("h", 1, 1): CM.Arc(c, r, 225 * DEG, 315 * DEG),
              ("h", 1, 2): CM.Arc(c, r, 135 * DEG, 45 * DEG),
              ("v", 1, 1): CM.Arc(c, r, 225 * DEG, 135 * DEG),
              ("v", 2, 1): CM.Arc(c, r, -45 * DEG, 45 * DEG)}
    return CM.TransfiniteMap(w, w, V, curved)


def maps():
    _e, _x, _y, prim = compile_shapes(C.P, C.P, [Circle(0.6, 0.6, C.R_CIRC,
                                                        4.0)], 1.0)
    h = C.R_CIRC / np.sqrt(2.0)
    return {"prim": prim,
            "lattice": hand_map((0.4, 0.8)),
            "inner": hand_map((0.6 - 0.8 * h, 0.6 + 0.8 * h))}


def airy():
    k0 = 2 * np.pi / C.WL
    n = (C.N_SUP, 2.0, C.N_SUB)
    r01 = (n[0] - n[1]) / (n[0] + n[1])
    r12 = (n[1] - n[2]) / (n[1] + n[2])
    ph = np.exp(2j * n[1] * k0 * C.DEPTH)
    return abs((r01 + r12 * ph) / (1 + r01 * r12 * ph)) ** 2


def film(M):
    res = {"env": C.env_record(), "M": M}
    Rx = airy()
    for name, cm in maps().items():
        o, R, T, _J = C.solve(cm, np.full((3, 3), 4.0 + 0j), M)
        i0 = C.idx(o, [(0, 0)])[0]
        R = R.copy()
        T = T.copy()
        R[:, i0] -= Rx
        T[:, i0] -= 1.0 - Rx
        res[name] = float(max(np.abs(R).max(), np.abs(T).max()))
    print(res, flush=True)
    C.dump(f"c4_walls_film_M{M}.json", res)


def pillar(M):
    res = {"env": C.env_record(), "M": M}
    eps = np.ones((3, 3), complex)
    eps[1, 1] = C.EPS_P
    for name, cm in maps().items():
        o, R, T, _J = C.solve(cm, eps, M)
        res[name] = {"vec": C.vec(o, R, T).tolist(),
                     "closure": float(np.max(np.abs(R.sum(1) + T.sum(1)
                                                    - 1)))}
    print({k: v["closure"] for k, v in res.items() if isinstance(v, dict)
           and "closure" in v}, flush=True)
    C.dump(f"c4_walls_pillar_M{M}.json", res)


def steep():
    """Per cell: the spread of the local stretch -- max / min over a 16 x 16
    Gauss grid of the larger singular value of J, and of det J -- and
    whether the cell owns a singular vertex."""
    xg, _ = leggauss(16)
    out = {"env": C.env_record()}
    for name, cm in maps().items():
        sing = {(a, b) for a, b, _c, _d in cm.singular_vertices}
        rows = []
        for sx in range(cm.shape[0]):
            U = 0.5 * (cm.u_bounds[sx] + cm.u_bounds[sx + 1]) + 0.5 * (
                cm.u_bounds[sx + 1] - cm.u_bounds[sx]) * xg
            for sy in range(cm.shape[1]):
                V = 0.5 * (cm.v_bounds[sy] + cm.v_bounds[sy + 1]) + 0.5 * (
                    cm.v_bounds[sy + 1] - cm.v_bounds[sy]) * xg
                _X, _Y, xu, xv, yu, yv = cm.geom(sx, sy, U, V)
                J = np.stack([np.stack([xu, xv], -1),
                              np.stack([yu, yv], -1)], -2)
                sv = np.linalg.svd(J, compute_uv=False)
                det = xu * yv - xv * yu
                rows.append({"cell": [sx, sy], "singular": (sx, sy) in sing,
                             "smax_ratio": float(sv[..., 0].max()
                                                 / sv[..., 0].min()),
                             "det_ratio": float(det.max() / det.min())})
        out[name] = rows
    print(out, flush=True)
    C.dump("c4_walls_steep.json", out)


if __name__ == "__main__":
    if sys.argv[1] == "film":
        film(int(sys.argv[2]))
    elif sys.argv[1] == "pillar":
        pillar(int(sys.argv[2]))
    else:
        steep()
