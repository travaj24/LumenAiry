"""F-B4 -- the ROUND-OFF floor of the curved solve, and where it comes from.

A random relative perturbation of 1e-15 of the HALF-SPACES' effective-tensor
weights (the homogeneous mapped solver; the patterned layer untouched) moves
the circle's R / T by far more than round-off.  Candidates: the eig of the
exactly degenerate plane-wave multiplets of the mapped half-space, and the
minimum-norm draw of the incident projection ``cinc = lstsq(Hsup, rhs)``
(``stack2d_pure.solve``), which is UNDERdetermined whenever the far-field
window (2 (2 n_orders + 1)^2 equations) is smaller than the half-space mode
count (2 q^2 unknowns).  Arms, on the 3 x 3 circle:

  pert M n_orders [which kind]
                   -- R / T change under the 1e-15 perturbation of the
                      half-space ('homog', default) or the layer ('layer')
                      weights, on the circle ('c3', default) or the identity
                      transfinite map on the same walls ('id')
  window M         -- R / T against n_orders = 2, 3, 5, 8, 10 (unperturbed):
                      what the under-determined projection costs in accuracy

  python validation/probe_pmm2d_curved/build_b/b11_floor.py pert 6 3
Output: b11_floor_pert_M<M>_n<n>.json, b11_floor_window_M<M>.json
"""
import sys

import _common as C
import numpy as np

TS = C.TS


def run_pert(M, n_orders, which="homog", kind="c3"):
    """which: 'homog' perturbs the half-spaces' weights only, 'layer' the
    patterned layer's only; kind: 'c3' the circle map, 'id' the identity
    TransfiniteMap on the same walls (the control)."""
    cm, eps = C.circle3()
    if kind == "id":
        cm = C.CM.TransfiniteMap(cm.u_walls, cm.v_walls)
    orig = TS._stag_map_eff
    rng = np.random.default_rng(1)
    on = [False]

    def pert(eps_, sg, g11, g12, g22):
        out = orig(eps_, sg, g11, g12, g22)
        e = np.asarray(eps_)
        homog = e.size > 1 and float(np.ptp(e.real)) == 0.0
        layer = e.size > 1 and float(np.ptp(e.real)) > 0.0
        hit = homog if which == "homog" else layer
        if on[0] and hit:
            out = {k: v * (1 + 1e-15 * rng.standard_normal(np.shape(v)))
                   for k, v in out.items()}
        return out
    TS._stag_map_eff = pert
    try:
        o, R, T, _J = C.solve(cm, eps, M, n_orders=n_orders)
        v0 = C.vec(o, R, T)
        on[0] = True
        o, R, T, _J = C.solve(cm, eps, M, n_orders=n_orders)
        v1 = C.vec(o, R, T)
    finally:
        TS._stag_map_eff = orig
    qq = (3 * (M - 1)) ** 2
    res = {"env": C.env_record(), "M": M, "n_orders": n_orders,
           "perturbed": which, "map": kind,
           "equations": 2 * (2 * n_orders + 1) ** 2, "unknowns": 2 * qq,
           "dRT": float(np.max(np.abs(v1 - v0)))}
    print(res, flush=True)
    sfx = "" if (which, kind) == ("homog", "c3") else f"_{which}_{kind}"
    C.dump(f"b11_floor_pert_M{M}_n{n_orders}{sfx}.json", res)


def run_window(M):
    cm, eps = C.circle3()
    res = {"env": C.env_record(), "M": M, "rows": []}
    vs = {}
    for n in (2, 3, 5, 8, 10):
        o, R, T, _J = C.solve(cm, eps, M, n_orders=n)
        vs[n] = C.vec(o, R, T)
    for n, v in vs.items():
        res["rows"].append({"n_orders": n,
                            "equations": 2 * (2 * n + 1) ** 2,
                            "unknowns": 2 * (3 * (M - 1)) ** 2,
                            "dRT_vs_n10": float(np.max(np.abs(v - vs[10])))})
    print(res, flush=True)
    C.dump(f"b11_floor_window_M{M}.json", res)


if __name__ == "__main__":
    if sys.argv[1] == "pert":
        run_pert(int(sys.argv[2]), int(sys.argv[3]), *sys.argv[4:6])
    else:
        run_window(int(sys.argv[2]))
