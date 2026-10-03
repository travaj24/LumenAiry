"""V7 -- the verifier's mutation matrix on its own fixtures.

usage: python v7_mutations.py <kind>     (kind from _ve1mut.KINDS or none)

Fixtures (M = 5 unless noted; slabs read the own (eps, mu) oracle, pillars
the change of the 36-vector against the correct arm, saved by kind=none):
  S1 sh4 DIRGEN normal          S2 sh4 DIRGEN oblique
  S3 sh4 DIRGEN + MU_GYRO obl.  S4 th3 GYROL + MU_GYROL conical (QZ)
  S5 none GYROL + MU_GYROL con. S6 sh4 DIRGEN slanted (0.17,-0.08) normal
  S7 same, oblique              S8 c3 DIRGEN oblique M6
  S9 th2b DIRGEN + MU_ANISO normal (stretch, chi33 visible?)
  S10 none DIRGEN + MU_SYM_LOSSY conical  S11 sh4, same, oblique
  P1 c3 pillar OOP normal       P2 c3 pillar eps 3.5 slanted 0.15 normal
  P3 c3 pillar OOP + mu-gyro disk (a magnetic OOP pillar) normal
"""
import sys
import warnings

import _ve1common as V
import _ve1mut as VM
import numpy as np

kind = sys.argv[1]
SL = (0.17, -0.08)
SLABS = {
    "S1": ("dirgen", None, "sh4", "n", None, 5),
    "S2": ("dirgen", None, "sh4", "o", None, 5),
    "S3": ("dirgen", "gyro", "sh4", "o", None, 5),
    "S4": ("gyrol", "gyrol", "th3", "c", None, 5),
    "S5": ("gyrol", "gyrol", "none", "c", None, 5),
    "S6": ("dirgen", None, "sh4", "n", SL, 5),
    "S7": ("dirgen", None, "sh4", "o", SL, 5),
    "S8": ("dirgen", None, "c3", "o", None, 6),
    "S9": ("dirgen", "aniso", "th2b", "n", None, 6),
    "S10": ("dirgen", "syml", "none", "c", None, 5),
    "S11": ("dirgen", "syml", "sh4", "o", None, 5),
}


def pillar_mag(M):
    cm, eps = V.disk_cell("c3", V.PIL_OOP)
    mu = np.broadcast_to(V.EYE, eps.shape).copy()
    mu[1, 1] = V.MU_GYRO
    f = V.PIL
    st = V.PMM2DStackPure(f["P"], f["P"], n_superstrate=1.0,
                          n_substrate=f["NSUB"], n_modes=M, n_orders=3,
                          cmap=cm)
    st.add_layer(f["DEP"], eps_cell=eps, mu_cell=mu)
    st.set_source(f["WL"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o, R, T, J = st.solve(jones=True)
    return V.vec36(o, R, T), float(np.abs(np.asarray(R).sum(1)
                                          + np.asarray(T).sum(1) - 1).max())


out = {}
with VM.vmutate(kind):
    for k, (e, m, mp, mo, sl, M) in SLABS.items():
        try:
            r = V.slab_run(V.TENSORS[e], V.make_map(mp, V.SLAB["P"]), M, mo,
                           slant=sl, mu=V.MUS[m or "none"])
            out[k] = {"worst": V.worst(r), "dRT": r["dRT"], "dJr": r["dJr"],
                      "dJt": r["dJt"], "clo": r["clo"]}
        except Exception as exc:
            out[k] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
        print(kind, k, out[k], flush=True)
    for k, fn in (("P1", lambda: V.pillar_run("c3", V.PIL_OOP, 5)),
                  ("P2", lambda: V.pillar_run("c3", 3.5 * V.EYE, 5,
                                              slant=(0.15, 0.0))),
                  ("P3", None)):
        try:
            if k == "P3":
                vec, clo = pillar_mag(5)
            else:
                _st, o, R, T, _w = fn()
                vec = V.vec36(o, R, T)
                clo = float(np.abs(R.sum(1) + T.sum(1) - 1).max())
            out[k] = {"vec": vec, "clo": clo}
        except Exception as exc:
            out[k] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
        print(kind, k, {kk: vv for kk, vv in out[k].items() if kk != "vec"},
              flush=True)
V.dump(f"v7_mut_{kind}.json", out)
