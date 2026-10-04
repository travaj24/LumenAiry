"""P2b -- the TM physical staircase converges at FIRST order in the slice
count on the P2 fixture; is that the staircase, the per-layer mortar or the
degree, and does it extrapolate to route B's limit?

Arms (moderate taper of P2, TM, normal incidence): the shipped
``PMMStack.add_tapered_grating`` on ``layer_grids='shared'`` and
``'per-layer'`` at degree 16 and 20, ns = 4 .. 64 (shared to 32: its union
grid grows with ns).  Each arm's per-component first-order (p = 1) and
second-order (p = 2) Richardson extrapolations of the top pair are scored
against route B's Richardson reference (P2's ``ref``, recomputed here).

Run:  PYTHONPATH=<worktree> OMP_NUM_THREADS=2 python p2b_tm_stair.py
Writes p2b_tm_stair.json.
"""
from __future__ import annotations

import os
import time
import warnings

import _zcommon as zc
import numpy as np
from p2_taper_ladder import EPS_G, EPS_R, EPS_SUB, WL, H, P, rich, route_b

HERE = os.path.dirname(os.path.abspath(__file__))


def stair(dt, db, pol, ns, degree, grids):
    from lumenairy.elements.pmm import PMMStack
    st = PMMStack(P, n_substrate=np.sqrt(EPS_SUB), n_superstrate=1.0,
                  degree=degree, far_field_orders=5, layer_grids=grids)
    st.add_tapered_grating(H, eps_ridge=EPS_R, eps_groove=EPS_G,
                           duty_top=dt, duty_bottom=db, n_slices=ns)
    st.set_source(WL, theta=0.0)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        orders, R, T, _J = st.solve()
    row = 1 if pol == "te" else 0
    o = [int(m) for m in np.asarray(orders)]
    sel = [o.index(m) for m in (-2, -1, 0, 1, 2)]
    v = np.array([float(R[row][i]) for i in sel]
                 + [float(T[row][i]) for i in sel])
    return v, time.perf_counter() - t0


def main():
    info = zc.assert_tree()
    dt, db = 0.40, 0.60
    out = dict(info=info)
    for pol in ("tm", "te"):
        ref = rich(route_b(dt, db, pol, 128), route_b(dt, db, pol, 256))
        rv = np.array(list(ref["R"]) + list(ref["T"]))
        arms = {}
        for grids, deg, nss in (("per-layer", 16, [4, 8, 16, 32, 64]),
                                ("per-layer", 20, [4, 8, 16, 32, 64]),
                                ("shared", 16, [4, 8, 16, 32])):
            if pol == "te" and deg == 20:
                continue
            key = f"{grids}_deg{deg}"
            vals, rows = {}, []
            for ns in nss:
                v, tm = stair(dt, db, pol, ns, deg, grids)
                vals[ns] = v
                rows.append(dict(ns=ns, eff=float(np.max(np.abs(v - rv))),
                                 time=tm))
                print(f"{pol} {key} ns {ns:3d}: {rows[-1]['eff']:.2e} "
                      f"({tm:.1f}s)")
            a, b = vals[nss[-2]], vals[nss[-1]]
            ex1 = 2 * b - a
            ex2 = (4 * b - a) / 3
            arms[key] = dict(ladder=rows,
                             rich_p1_vs_ref=float(np.max(np.abs(ex1 - rv))),
                             rich_p2_vs_ref=float(np.max(np.abs(ex2 - rv))))
            print(f"   {key}: Richardson p=1 {arms[key]['rich_p1_vs_ref']:.2e}"
                  f", p=2 {arms[key]['rich_p2_vs_ref']:.2e}")
        out[pol] = arms
    zc.dump(os.path.join(HERE, "p2b_tm_stair.json"), out)


if __name__ == "__main__":
    main()
