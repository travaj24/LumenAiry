"""C9 -- the incident-field fix (Phase B finding F-B4, Phase A verifier D4),
measured BEFORE (the PRE tree, ``LUM_TREE=C:/tmp/curved_pre_c``: the
least-squares overlap ``cinc = lstsq(Hsup, delta_00)``) and AFTER (this tree:
the exact L2 modal decomposition ``cinc = W0^-1 G^-1 b``).

  c9_incident.py <M> <map>       map: 'c3' (centred 3 x 3 circle) or
                                 'c3off' (the circle at (0.55, 0.66), no
                                 mirror symmetry)

Arms, each on the single-layer circle pillar at normal incidence:
  floor    -- R / T change under a random 1e-15 relative perturbation of the
              HALF-SPACES' effective weights (Phase B's b11 arm, verbatim)
  orders   -- R / T at n_orders = 2, 3, 5 against n_orders = 8
  spacer   -- a uniform VACUUM layer of 0.3 on top (physically a no-op)
              against none, at n_orders = 3
Output: c9_incident_<tree>_<map>_M<M>.json
"""
import sys

import _common as C
import numpy as np

TS = C.TS


def run(M, kind):
    center = None if kind == "c3" else (0.55, 0.66)
    cm, eps = C.circle3(center)
    res = {"env": C.env_record(), "M": M, "map": kind}
    # floor
    orig = TS._stag_map_eff
    rng = np.random.default_rng(1)
    on = [False]

    def pert(eps_, sg, g11, g12, g22):
        out = orig(eps_, sg, g11, g12, g22)
        e = np.asarray(eps_)
        if on[0] and e.size > 1 and float(np.ptp(e.real)) == 0.0:
            out = {k: v * (1 + 1e-15 * rng.standard_normal(np.shape(v)))
                   for k, v in out.items()}
        return out
    TS._stag_map_eff = pert
    try:
        v0 = C.vec(*C.solve(cm, eps, M)[:3])
        on[0] = True
        v1 = C.vec(*C.solve(cm, eps, M)[:3])
    finally:
        TS._stag_map_eff = orig
    res["floor_dRT"] = float(np.max(np.abs(v1 - v0)))
    # orders
    vs = {n: C.vec(*C.solve(cm, eps, M, n_orders=n)[:3]) for n in (2, 3, 5, 8)}
    res["orders_dRT_vs_n8"] = {str(n): float(np.max(np.abs(vs[n] - vs[8])))
                               for n in (2, 3, 5)}
    # spacer
    o, R, T, J = C.solve(cm, eps, M, spacer=0.3)
    res["spacer_dRT"] = float(np.max(np.abs(C.vec(o, R, T) - vs[3])))
    o3, R3, T3, J3 = C.solve(cm, eps, M)
    res["spacer_dJ_abs"] = float(np.max(np.abs(np.abs(J) - np.abs(J3))))
    res["closure_n3"] = float(np.max(np.abs(R3.sum(1) + T3.sum(1) - 1.0)))
    print(res, flush=True)
    C.dump(f"c9_incident_{C.TREE}_{kind}_M{M}.json", res)


if __name__ == "__main__":
    run(int(sys.argv[1]), sys.argv[2])
