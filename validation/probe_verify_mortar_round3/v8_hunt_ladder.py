"""V8 -- the HUNT's two candidates, taken to a convergence ladder.

``v5_falsepos.py hunt`` found two geometries whose per-layer answer differs
from its common-refinement (mortar-free) twin by ~99 % while the generalized
mortar's residual reads 1e-14 and the band warning is silent:

* ``just_above_band``     -- an out-of-plane post 3.2e-2 of the period wide,
  i.e. JUST above the band's upper edge, so nothing warns;
* ``just_above_contract`` -- the same post 1.1e-3 wide, i.e. just above the
  width contract, which DOES warn.

A single modal count cannot call either of those WRONG: the two arms are
different discretisations of the same device.  This probe runs the ladder on
both arms and asks whether the gap CLOSES.  It also reports the lossless
closure of each arm, because a closure that is itself large says the rung is
not converged rather than that the mortar is broken.

Run: ``python v8_hunt_ladder.py <tag>``
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _path                                     # noqa: E402,F401,I001

import json                                                 # noqa: E402
import time                                                 # noqa: E402
import warnings                                             # noqa: E402

import numpy as np                                          # noqa: E402

import _vfix as F                                           # noqa: E402
from _capture import capture                                # noqa: E402
from lumenairy.elements.pmm import PMM2DStackPure           # noqa: E402
from lumenairy.elements.pmm import _core as _pc             # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
P = 0.87e-6
WB = (0.3106, 0.8039)

CASES = {
    # name: (the OUT-OF-PLANE layer's walls, per-layer Ms, union Ms)
    "just_above_band": ((0.30, 0.3320), (4, 5, 6, 7), (4, 5, 6)),
    "just_above_contract": ((0.30, 0.3011), (4, 5, 6, 7), (4, 5)),
    "ordinary_post": ((0.30, 0.45), (4, 5, 6, 7), (4, 5, 6)),
}


def _build(WA, M, union):
    U = tuple(sorted(set(WA) | set(WB)))
    wa, wb = (U, U) if union else (WA, WB)
    st = PMM2DStackPure(P, n_modes=M, n_orders=2, n_substrate=1.45,
                        layer_grids="per-layer")
    st.add_layer(0.118e-6, eps_cell=F.tensor_cell(wa, WA[0], WA[-1]),
                 x_walls=F.sc(wa, P), y_walls=F.sc(wa, P))
    st.add_layer(0.094e-6, eps_cell=F.scalar_cell(wb, WB[0], WB[-1],
                                                  eps_in=8.41),
                 x_walls=F.sc(wb, P), y_walls=F.sc(wb, P))
    st.set_source(0.73e-6, theta=0.17, phi=1.1)
    return st


def _run(WA, M, union):
    t0 = time.time()
    with capture() as recs:
        st = _build(WA, M, union)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            o, R, T = st.solve(jones=False)
    o, R, T = np.asarray(o), np.atleast_2d(R), np.atleast_2d(T)
    res = []
    for r in recs:
        X = np.linalg.solve(r["A"], r["B"])
        res.append(float(_pc._mortar_residual(r["A"], X, r["B"],
                                              probe=False)))
    return {"M": M, "union": union, "R00": F.R00(o, R),
            "sumR": float(R.sum(axis=1)[1]), "sumT": float(T.sum(axis=1)[1]),
            "closure": abs(float(R.sum(axis=1)[1] + T.sum(axis=1)[1]) - 1.0),
            "orders": o.tolist(), "R1": R[1].tolist(), "T1": T[1].tolist(),
            "worst_residual": max(res) if res else None,
            "n_gen_sites": len(recs),
            "band_warnings": len([w for w in ws
                                  if "degradation band" in str(w.message)]),
            "seconds": round(time.time() - t0, 1)}


def _far_field_gap(a, b):
    """max |R_a - R_b| + max |T_a - T_b| over the orders BOTH arms carry --
    an absolute comparison, so a tiny ``R00`` cannot inflate it."""
    oa = {tuple(x): i for i, x in enumerate(a["orders"])}
    ob = {tuple(x): i for i, x in enumerate(b["orders"])}
    keys = sorted(set(oa) & set(ob))
    dr = max(abs(a["R1"][oa[k]] - b["R1"][ob[k]]) for k in keys)
    dt = max(abs(a["T1"][oa[k]] - b["T1"][ob[k]]) for k in keys)
    return dr, dt, len(keys)


def main(tag):
    rows = []
    for name, (WA, Ms, uMs) in CASES.items():
        per, uni = {}, {}
        for M in Ms:
            per[M] = _run(WA, M, False)
            rows.append(dict(per[M], case=name, arm="per_layer"))
            print(f"  {name:22s} per M={M} R00={per[M]['R00']:.9f} "
                  f"clo={per[M]['closure']:.2e} warn={per[M]['band_warnings']}"
                  f" res={per[M]['worst_residual']:.2e} "
                  f"{per[M]['seconds']:.0f}s", flush=True)
        for M in uMs:
            uni[M] = _run(WA, M, True)
            rows.append(dict(uni[M], case=name, arm="union"))
            print(f"  {name:22s} uni M={M} R00={uni[M]['R00']:.9f} "
                  f"clo={uni[M]['closure']:.2e} sites={uni[M]['n_gen_sites']} "
                  f"{uni[M]['seconds']:.0f}s", flush=True)
        umax = uni[max(uMs)]
        gaps = {M: _far_field_gap(per[M], umax) for M in Ms}
        steps = {Ms[i + 1]: _far_field_gap(per[Ms[i + 1]], per[Ms[i]])
                 for i in range(len(Ms) - 1)}
        usteps = {uMs[i + 1]: _far_field_gap(uni[uMs[i + 1]], uni[uMs[i]])
                  for i in range(len(uMs) - 1)}
        rows.append({"what": "summary", "case": name,
                     "gap_vs_union": {str(k): v[:2] for k, v in gaps.items()},
                     "per_layer_step": {str(k): v[:2]
                                        for k, v in steps.items()},
                     "union_step": {str(k): v[:2] for k, v in usteps.items()},
                     "per_layer_closure": {str(M): per[M]["closure"]
                                           for M in Ms},
                     "union_closure": {str(M): uni[M]["closure"]
                                       for M in uMs}})
        print(f"  -> {name}: |dR| to union {[f'{gaps[M][0]:.3e}' for M in Ms]}"
              f"  per-layer steps "
              f"{[f'{steps[M][0]:.3e}' for M in list(steps)]}"
              f"  union steps {[f'{usteps[M][0]:.3e}' for M in list(usteps)]}",
              flush=True)
    out = {"env": F.env(), "tag": tag, "rows": rows}
    (HERE / f"v8_hunt_ladder_{tag}.json").write_text(
        json.dumps(out, indent=1), encoding="cp1252")
    print(f"wrote v8_hunt_ladder_{tag}.json ({len(rows)} rows)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "win")
