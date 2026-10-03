"""C4c -- the 2e-9 film plateau of the coarse 3 x 3 sinusoidal-ridge map
(c4b): which ingredient holds it.

  c4c_plateau.py <M>

Arms, all the SAME geometry (ridge x0 = 0.3, width 0.5, A = 0.05) and the
uniform eps-4 film (exact answer: Airy):
  refined   -- compile_shapes' map: RefinedMap of a one-v-segment base
  direct    -- the same walls as ONE TransfiniteMap (Phase B's
               _sine_stripe_map_3x3 with v walls 0, 0.3, 0.6, 1.2)
  thirds    -- Phase B's own 3 x 3 (v walls at thirds)
  *_lstsq   -- the same with the shipped least-squares incident overlap
  *_nq      -- the same with the assembly node count forced to 4x
Output: c4c_plateau_M<M>.json
"""
import sys

import _common as C
import numpy as np

from lumenairy.elements.pmm import SinusoidalWall, compile_shapes  # noqa

TS, SP, CM = C.TS, C.SP, C.CM


def airy():
    k0 = 2 * np.pi / C.WL
    n = (C.N_SUP, 2.0, C.N_SUB)
    r01 = (n[0] - n[1]) / (n[0] + n[1])
    r12 = (n[1] - n[2]) / (n[1] + n[2])
    ph = np.exp(2j * n[1] * k0 * C.DEPTH)
    return abs((r01 + r12 * ph) / (1 + r01 * r12 * ph)) ** 2


def err(cm, M):
    n = cm.shape[0]
    o, R, T, _J = C.solve(cm, np.full((n, n), 4.0 + 0j), M)
    i0 = C.idx(o, [(0, 0)])[0]
    R = R.copy()
    T = T.copy()
    Rx = airy()
    R[:, i0] -= Rx
    T[:, i0] -= 1.0 - Rx
    return float(max(np.abs(R).max(), np.abs(T).max()))


def run(M):
    A = 0.05
    sh = SinusoidalWall("x", 0.3, A, eps=4.0, width=0.5)
    _e, _x, _y, refined = compile_shapes(C.P, C.P, [sh], 1.0)
    direct, _w = CM._sine_stripe_map_3x3(C.P, 0.3, 0.8, A,
                                         v_walls=refined.v_bounds)
    thirds, _w = CM._sine_stripe_map_3x3(C.P, 0.3, 0.8, A)
    maps = {"refined": refined, "direct": direct, "thirds": thirds}
    res = {"env": C.env_record(), "M": M,
           "v_walls": refined.v_bounds.tolist()}
    for k, m in maps.items():
        res[k] = err(m, M)
    orig = TS._stag_incident_coeffs_mapped
    SP._stag_incident_coeffs_mapped = lambda *a, **k: None
    try:
        for k in ("refined", "thirds"):
            res[k + "_lstsq"] = err(maps[k], M)
    finally:
        SP._stag_incident_coeffs_mapped = orig
    orig_n = TS._stag_map_nodes
    TS._stag_map_nodes = lambda bx, by, cm, MM, **kw: 4 * (2 * MM + 8)
    try:
        for k in ("refined", "thirds"):
            res[k + "_nq"] = err(maps[k], M)
    finally:
        TS._stag_map_nodes = orig_n
    print({k: v for k, v in res.items() if k != "env"}, flush=True)
    C.dump(f"c4c_plateau_M{M}.json", res)


if __name__ == "__main__":
    run(int(sys.argv[1]))
