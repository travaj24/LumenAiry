"""E1-7 -- the generalized (Onsager-Casimir) reciprocity of a NON-reciprocal
medium: the reflection of eps for (incidence -> order (m, n)) and that of
eps^T for the reversed channel have the same singular values.  Measured on
the non-reciprocal OOP30 twin (``e5_pillar.NONREC30``), conical (25, 40 deg),
order (-1, 0), in three geometries so a convergence can be told from a
defect:

* ``stair``   -- the SHIPPED out-of-plane solver on the 4-step staircase (no
  map, no slant): the reference behaviour of the shipped code;
* ``curved``  -- the c3 circle map (Phase E1 generator), vertical;
* ``slanted`` -- the c3 circle map, slanted (0.2, 0) (the composite frame).

For each: |sv(eps; fwd) - sv(eps; rev)| (non-reciprocal: must NOT vanish),
|sv(eps; fwd) - sv(eps^T; rev)| (must converge to 0), and the reciprocal
OOP30 control |sv(fwd) - sv(rev)|.

usage: python e7_onsager.py <stair|curved|slanted> <M>   -> e7_<geom>_M<M>.json
       python e7_onsager.py summary
"""
import json
import os
import sys
import time
import warnings

import _e1common as E
import e5_pillar as E5
import numpy as np

TH, PH = 25.0, 40.0
ORDER = (-1, 0)


def solve(geom, t33, M, th, ph):
    if geom == "stair":
        w = np.array([0.0, 0.24, 0.96, E5.P])
        eps = np.broadcast_to(np.eye(3, dtype=complex), (3, 3, 3, 3)).copy()
        eps[1, 1] = t33
        st = E.PMM2DStackPure(E5.P, E5.P, n_superstrate=1.0,
                              n_substrate=E5.NSUB, n_modes=M, n_orders=3,
                              layer_grids="per-layer")
        st.add_layer(E5.DEP, eps_cell=eps, x_walls=w, y_walls=w)
    else:
        cm, eps = E5.disk_cells("c3", t33)
        st = E.PMM2DStackPure(E5.P, E5.P, n_superstrate=1.0,
                              n_substrate=E5.NSUB, n_modes=M, n_orders=3,
                              cmap=cm)
        st.add_layer(E5.DEP, eps_cell=eps,
                     slant=(0.2, 0.0) if geom == "slanted" else None)
    st.set_source(E5.WL, theta=np.deg2rad(th), phi=np.deg2rad(ph))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o, R, T, _J = st.solve(jones=True)
    r = E5.record(st, o, R, T, None, {})
    return r


def sv(r, order):
    k = E5.ORDS.index(order)
    return np.linalg.svd(E5.jones_block(r, k), compute_uv=False)


def run(geom, M):
    t0 = time.perf_counter()
    tr, pr = E5.reverse_angles(TH, PH, *ORDER)
    out = {"geom": geom, "M": M, "reverse": [tr, pr]}
    n = E5.MATS["nonrec30"]
    f = sv(solve(geom, n, M, TH, PH), ORDER)
    rn = sv(solve(geom, n, M, tr, pr), ORDER)
    rT = sv(solve(geom, n.T.copy(), M, tr, pr), ORDER)
    fo = sv(solve(geom, E5.MATS["oop30"], M, TH, PH), ORDER)
    ro = sv(solve(geom, E5.MATS["oop30"], M, tr, pr), ORDER)
    out.update({"nonrec_vs_same": float(np.abs(f - rn).max()),
                "nonrec_vs_transposed": float(np.abs(f - rT).max()),
                "reciprocal_control": float(np.abs(fo - ro).max()),
                "sv": {"f": f.tolist(), "rn": rn.tolist(), "rT": rT.tolist(),
                       "fo": fo.tolist(), "ro": ro.tolist()},
                "t": time.perf_counter() - t0})
    E.dump(f"e7_{geom}_M{M}.json", out)
    print(geom, M, {k: v for k, v in out.items() if k not in ("sv",)},
          flush=True)


def summary():
    here = os.path.dirname(os.path.abspath(__file__))
    out = {}
    for fn in sorted(os.listdir(here)):
        if fn.startswith("e7_") and fn.endswith(".json") and "summ" not in fn:
            r = json.load(open(os.path.join(here, fn)))
            out.setdefault(r["geom"], {})[str(r["M"])] = {
                k: r[k] for k in ("nonrec_vs_same", "nonrec_vs_transposed",
                                  "reciprocal_control")}
    E.dump("e7_summary.json", out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "summary":
        summary()
    else:
        run(sys.argv[1], int(sys.argv[2]))
