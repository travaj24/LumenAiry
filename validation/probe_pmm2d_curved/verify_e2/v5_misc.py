"""V5 misc probes.

usage: v5_misc.py host <Ms> <Mc>   the q-matching claim: ALL-VACUUM two-layer
         stack (uniform eps 1 films on a sinusoid map / circle map, cmap= and
         explicit n_modes, sup = sub = 1: exact R00 = 0, T00 = 1)
       v5_misc.py sandwich <M>     vacuum / circle pillar / vacuum, sup = sub
         = 1: per-layer (fast path, forced, forced no-ride, vacuum-painted
         shape spacers) vs the shared stack with the same spacers and vs
         the pillar alone
       v5_misc.py fastkw <M>       the non-crossing pair with n_modes = M
         (the stack's own M) NAMED on one shape layer vs not named
       v5_misc.py qaxis <M>        a rectangle-only stripe (N_x = 3,
         N_y = 1) under the circle, merge disabled: q-matching reads the x
         walls only
"""
import sys
import time

import numpy as np
from _ve import dump, solve

from lumenairy.elements.pmm import PMM2DStackPure
from lumenairy.elements.pmm.shapes2d import Circle, Rect, SinusoidalWall, compile_shapes
from lumenairy.elements.pmm.stack2d_pure import _stag_walls_n as SP_n

P = 1.2
mode = sys.argv[1]


def d(a, b):
    return float(max(np.abs(a[1] - b[1]).max(), np.abs(a[2] - b[2]).max()))


C = [Circle(0.6, 0.6, 0.36, 4.0)]
out = {"mode": mode}
if mode == "host":
    Ms_, Mc = int(sys.argv[2]), int(sys.argv[3])
    _e, _x, _y, circ = compile_shapes(P, P, C, 1.0)
    _e, _x, _y, sinx = compile_shapes(
        P, P, [SinusoidalWall("x", 0.6, 0.12, eps=2.25)], 1.0)
    st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.0,
                        n_modes=max(Ms_, Mc), n_orders=3,
                        layer_grids="per-layer")
    st.add_layer(0.3, eps=1.0, cmap=sinx, n_modes=Ms_)
    st.add_layer(0.25, eps=1.0, cmap=circ, n_modes=Mc)
    t0 = time.perf_counter()
    o, R, T, J = solve(st)
    p0 = int(np.nonzero((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
    err = float(max(np.abs(R).max(), np.abs(T[:, p0] - 1).max(),
                    np.abs(np.delete(T, p0, axis=1)).max()))
    out.update(Ms=Ms_, Mc=Mc,
               sinx_N=(len(sinx.u_walls) - 1, len(sinx.v_walls) - 1),
               circ_N=(len(circ.u_walls) - 1, len(circ.v_walls) - 1),
               err_vs_exact=err, wall=time.perf_counter() - t0)
    print(out)
    dump(f"v5_misc_host_{Ms_}_{Mc}", out)
elif mode == "sandwich":
    M = int(sys.argv[2])
    kw = dict(n_superstrate=1.0, n_substrate=1.0, n_modes=M, n_orders=3)
    alone = PMM2DStackPure(P, P, **kw)
    alone.add_layer(0.3, shapes=C, background_eps=1.0)
    a = solve(alone)
    sh = PMM2DStackPure(P, P, **kw)
    sh.add_layer(0.15, eps=1.0)
    sh.add_layer(0.3, shapes=C, background_eps=1.0)
    sh.add_layer(0.2, eps=1.0)
    s = solve(sh)
    res = {}
    for arm in ("nat", "forced", "forced_noride", "vacshapes"):
        st = PMM2DStackPure(P, P, layer_grids="per-layer", **kw)
        if arm == "vacshapes":
            st.add_layer(0.15, shapes=[SinusoidalWall("x", 0.6, 0.12,
                                                      eps=1.0)],
                         background_eps=1.0)
        else:
            st.add_layer(0.15, eps=1.0)
        st.add_layer(0.3, shapes=C, background_eps=1.0)
        if arm == "vacshapes":
            st.add_layer(0.2, shapes=[SinusoidalWall("y", 0.5, 0.1,
                                                     eps=1.0)],
                         background_eps=1.0)
        else:
            st.add_layer(0.2, eps=1.0)
        if arm.startswith("forced"):
            st._e2_per_layer_maps = True
        if arm == "forced_noride":
            st._e2_no_ride = True
        fast = bool(st._perlayer_fast_ok())
        b = solve(st)
        res[arm] = dict(fast=fast,
                        bytes_eq_shared=bool(np.array_equal(b[1], s[1])
                                             and np.array_equal(b[2], s[2])),
                        vs_shared_spacers=d(b, s), vs_alone=d(b, a),
                        closure=np.abs(b[1].sum(1) + b[2].sum(1) - 1),
                        R=b[1], T=b[2])
        print(arm, {k: v for k, v in res[arm].items() if k not in "RT"})
    out.update(M=M, shared_spacers_vs_alone=d(s, a), arms=res,
               alone_R=a[1], alone_T=a[2], shared_R=s[1], shared_T=s[2],
               alone_closure=np.abs(a[1].sum(1) + a[2].sum(1) - 1))
    dump(f"v5_misc_sandwich_M{M}", out)
elif mode == "fastkw":
    M = int(sys.argv[2])
    res = {}
    for arm in ("plain", "named_same_M", "shared"):
        kw = dict(n_superstrate=1.0, n_substrate=1.45, n_modes=M,
                  n_orders=3)
        if arm != "shared":
            kw["layer_grids"] = "per-layer"
        st = PMM2DStackPure(P, P, **kw)
        st.add_layer(0.3, shapes=C, background_eps=1.0)
        extra = {"n_modes": M} if arm == "named_same_M" else {}
        st.add_layer(0.25, shapes=[SinusoidalWall("x", 0.12, 0.05,
                                                  eps=2.25)],
                     background_eps=1.0, **extra)
        fast = bool(st._perlayer_fast_ok()) if arm != "shared" else None
        Ms = st._perlayer_modal_counts() if arm != "shared" else None
        res[arm] = dict(fast=fast, Ms=Ms, sol=solve(st))
    out.update(M=M, fast={k: v["fast"] for k, v in res.items()},
               Ms={k: v["Ms"] for k, v in res.items()},
               named_vs_plain=d(res["named_same_M"]["sol"],
                                res["plain"]["sol"]),
               plain_bytes_eq_shared=bool(
                   np.array_equal(res["plain"]["sol"][1],
                                  res["shared"]["sol"][1])))
    print(out)
    dump(f"v5_misc_fastkw_M{M}", out)
elif mode == "qaxis":
    M = int(sys.argv[2])
    stripe = [Rect(0.10, 0.6, 0.16, P, 2.25)]
    res = {}
    for arm in ("default", "ymatch", "ref"):
        Mr = M + 3 if arm == "ref" else M
        kw = dict(n_superstrate=1.0, n_substrate=1.45, n_modes=Mr,
                  n_orders=3)
        if arm != "ref":
            kw["layer_grids"] = "per-layer"
        st = PMM2DStackPure(P, P, **kw)
        st.add_layer(0.3, shapes=C, background_eps=1.0)
        extra = {"n_modes": 3 * (M - 1) + 1} if arm == "ymatch" else {}
        st.add_layer(0.25, shapes=stripe, background_eps=1.0, **extra)
        info = {}
        if arm != "ref":
            st._e2_per_layer_maps = True
            info = dict(Ms=st._perlayer_modal_counts(),
                        Nxy=[(SP_n(L["wx"]), SP_n(L["wy"]))
                             for L in st._layers])
        t0 = time.perf_counter()
        res[arm] = dict(sol=solve(st), wall=time.perf_counter() - t0, **info)
    out.update(M=M, info={k: {kk: vv for kk, vv in v.items() if kk != "sol"}
                          for k, v in res.items()},
               default_vs_ref=d(res["default"]["sol"], res["ref"]["sol"]),
               ymatch_vs_ref=d(res["ymatch"]["sol"], res["ref"]["sol"]))
    print(out)
    dump(f"v5_misc_qaxis_M{M}", out)
