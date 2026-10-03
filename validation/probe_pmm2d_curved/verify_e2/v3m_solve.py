"""V3 decisive check -- solve the E2-4 crossing device one way at one rung.

  python v3m_solve.py merged   M [theta_deg phi_deg]   # hand-built 4x4 conforming map, SHARED path
  python v3m_solve.py perlayer M [theta_deg phi_deg]   # the E2 per-layer curved mortar (shapes=)
  python v3m_solve.py sanity   M                       # map/cell indexing: merged map with ONE layer's
                                                       # pattern vs the same layer through shapes= (Phase B/C maps)
Output: v3m_<mode>_M<M>_th<t>_ph<p>_<build>.json
"""
import sys
import time

import numpy as np
import v3m_geom as G
from _ve import closure, dump, rt_diff, solve

mode, M = sys.argv[1], int(sys.argv[2])
th_d = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
ph_d = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
th, ph = np.deg2rad(th_d), np.deg2rad(ph_d)
out = {"mode": mode, "M": M, "theta_deg": th_d, "phi_deg": ph_d}
t0 = time.perf_counter()
if mode == "merged":
    tm, info = G.build()
    out["geometry"] = info
    out["singular_vertices"] = tm.singular_vertices
    st = G.stack(M, tm)
    o, R, T, J = solve(st, G.WL, th, ph)
    out["pencil"] = 2 * (4 * (M - 1)) ** 2
elif mode == "perlayer":
    st = G.perlayer_stack(M)
    assert not st._perlayer_fast_ok()
    o, R, T, J = solve(st, G.WL, th, ph)
    try:
        out["modal_counts"] = [int(m) for m in st._perlayer_modal_counts()]
    except Exception as e:  # noqa: BLE001
        out["modal_counts"] = repr(e)
elif mode == "sanity":
    from lumenairy.elements.pmm import PMM2DStackPure
    from lumenairy.elements.pmm.shapes2d import Circle, SinusoidalWall
    tm, info = G.build()
    c1, c2 = G.cells()
    res = {}
    for name, cell, shp in (
            ("disk", c1, [Circle(G.CX, G.CY, G.R, G.EPS_D)]),
            ("wall", c2, [SinusoidalWall("x", G.X0, G.AMP, eps=G.EPS_W)])):
        st = PMM2DStackPure(G.P, G.P, n_superstrate=G.N_SUP,
                            n_substrate=G.N_SUB, n_modes=M, n_orders=3,
                            cmap=tm)
        st.add_layer(G.D1 if name == "disk" else G.D2, eps_cell=cell)
        a = solve(st, G.WL, th, ph)
        st2 = PMM2DStackPure(G.P, G.P, n_superstrate=G.N_SUP,
                             n_substrate=G.N_SUB, n_modes=M, n_orders=3)
        st2.add_layer(G.D1 if name == "disk" else G.D2, shapes=shp,
                      background_eps=1.0)
        b = solve(st2, G.WL, th, ph)
        res[name] = dict(diff=rt_diff(a, b), closure_merged4x4=closure(a),
                         closure_shapes=closure(b))
        print(name, res[name], flush=True)
    out["sanity"] = res
    o = R = T = None
if R is not None:
    out.update(orders=o, R=R, T=T, closure=closure((o, R, T, J)))
out["wall"] = time.perf_counter() - t0
print({k: v for k, v in out.items() if k not in ("R", "T", "orders",
                                                  "geometry")}, flush=True)
dump(f"v3m_{mode}_M{M}_th{th_d:g}_ph{ph_d:g}", out)
