"""D9 -- the MUTATION MATRIX: every engineered defect of ``_dcommon.mutate``
against every gate, at fixed M, with the CORRECT arm as the reference row.

Gates (the quantity each reads):
  film_gyro_s05    -- gyrotropic film, sine stretch a = 0.05 p, normal, M 6:
                      Jones and R / T vs Berreman (two numbers)
  film_lc_shear    -- LC film, sheared bilinear map, normal, M 6: same
  film_lc_c3_con   -- LC film, 3 x 3 circle map, conical (25, 40), M 6
  film_mag_c3      -- magnetic film (eps 2, mu diag(1.8, 1.8, 1.3)), circle
                      map, normal, M 6: R / T vs the (eps, mu) Airy slab
  dual_c3          -- the magnetic pillar vs its eps dual, circle map, M 6
                      (duality residual)
  li_stretch       -- the Li 2003 gyrotropic grating under the D5 stretch,
                      M 7: max deviation from Li's FIRST (published) row
  pillar_lc_c3     -- the D4 tensor pillar, circle map, M 6: R / T against
                      the correct arm's own M 6 answer (sees ANY change)

usage: python d9_mutations.py <correct|transpose|side|no_sg_e33|mixed_sign|
                               mu_after|hgram_R>
       python d9_mutations.py summary
"""
import json
import os
import sys
import time
from contextlib import nullcontext

import _common as C
import _dcommon as D
import d5_li2003 as L
import d6_magnetic as G
import numpy as np

P = 1.2


def gates():
    out = {}
    P3 = D.G3["P"]
    dRT, dJ, _ = D.film_vs_berreman(D.GYRO, D.make_map("s05", P3), 6)
    out["film_gyro_s05"] = {"RT": dRT, "J": dJ}
    dRT, dJ, _ = D.film_vs_berreman(D.LC, D.make_map("shear", P3), 6)
    out["film_lc_shear"] = {"RT": dRT, "J": dJ}
    dRT, dJ, _ = D.film_vs_berreman(D.LC, D.make_map("c3", P3), 6,
                                    theta=np.deg2rad(25.0),
                                    phi=np.deg2rad(40.0))
    out["film_lc_c3_con"] = {"RT": dRT, "J": dJ}
    # magnetic film vs the (eps, mu) Airy slab
    cm = D.make_map("c3", P)
    st = D.PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45,
                          n_modes=6, n_orders=2, cmap=cm)
    st.add_layer(G.DEP, eps=G.EPS_F, mu=G.MU_F)
    st.set_source(G.WL)
    o, R, T, _J = st.solve(jones=True)
    o, R, T = np.asarray(o), np.asarray(R), np.asarray(T)
    Rx, Tx = G.airy(G.EPS_F, G.MU_F[0, 0], G.DEP)
    i0 = C.idx(o, [(0, 0)])[0]
    e = 0.0
    for row in (0, 1):
        Rr, Tr = R[row].copy(), T[row].copy()
        Rr[i0] -= Rx
        Tr[i0] -= Tx
        e = max(e, float(np.abs(Rr).max()), float(np.abs(Tr).max()))
    out["film_mag_c3"] = {"RT": e}
    # duality
    cm, mu_c, eps_c = G.pillar_cells("c3")
    one = np.broadcast_to(np.eye(3, dtype=complex), mu_c.shape).copy()
    om, Rm, Tm = G.solve_vac(cm, one, mu_c, 6)
    oe, Re, Te = G.solve_vac(cm, eps_c, None, 6)
    im, ie = C.idx(om), C.idx(oe)
    out["dual_c3"] = {"RT": float(max(
        np.abs(Rm[0, im] - Re[1, ie]).max(), np.abs(Tm[0, im] - Te[1, ie]).max(),
        np.abs(Rm[1, im] - Re[0, ie]).max(),
        np.abs(Tm[1, im] - Te[0, ie]).max()))}
    # Li 2003 under the stretch
    cmL = L.cmap_for("stretch")
    stL = D.PMM2DStackPure(L.PX, L.PY, n_superstrate=1.0, n_substrate=L.NSUB,
                           n_modes=7, n_orders=4, cmap=cmL)
    stL.add_layer(L.LAM, eps_cell=L.cell(L.LI_B, L.LI_A))
    stL.set_source(L.LAM)
    oL, RL, _TL, _JL = stL.solve(jones=True)
    oL, RL = np.asarray(oL), np.asarray(RL)
    R0 = RL[0, C.idx(oL, L.ORD)]
    out["li_stretch"] = {
        "row1": float(max(abs(R0[k] - v) for k, v in
                          enumerate(L.TABLE1.values()))),
        "row2": float(max(abs(R0[k] - v) for k, v in
                          enumerate(L.TABLE2.values())))}
    # the D4 tensor pillar
    import d4_pillar as Q
    cm, eps = Q.disk_cells("c3")
    st = D.PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45,
                          n_modes=6, n_orders=3, cmap=cm)
    st.add_layer(0.5, eps_cell=eps)
    st.set_source(1.0)
    o, R, T, _J = st.solve(jones=True)
    out["pillar_lc_c3"] = {"vec": C.vec(np.asarray(o), np.asarray(R),
                                        np.asarray(T)).tolist()}
    return out


def run(kind):
    t0 = time.perf_counter()
    ctx = nullcontext() if kind == "correct" else D.mutate(kind)
    with ctx:
        out = gates()
    out["t"] = time.perf_counter() - t0
    D.dump(f"d9_{kind}.json", out)
    print(kind, {k: v for k, v in out.items() if k != "pillar_lc_c3"},
          flush=True)


def summary():
    here = os.path.dirname(os.path.abspath(__file__))
    rs = {}
    for k in ("correct",) + D.MUTATIONS:
        fn = os.path.join(here, f"d9_{k}.json")
        if os.path.exists(fn):
            rs[k] = json.load(open(fn))
    ref = np.array(rs["correct"]["pillar_lc_c3"]["vec"])
    table = {}
    for k, r in rs.items():
        row = {g: v for g, v in r.items() if g not in ("pillar_lc_c3", "t",
                                                        "env")}
        row["pillar_lc_c3"] = {"RT_vs_correct": float(np.abs(
            np.array(r["pillar_lc_c3"]["vec"]) - ref).max())}
        table[k] = row
    D.dump("d9_summary.json", table)
    for k, row in table.items():
        print(k.ljust(11), json.dumps(row))


if __name__ == "__main__":
    if sys.argv[1] == "summary":
        summary()
    else:
        run(sys.argv[1])
