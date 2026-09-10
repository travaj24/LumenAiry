"""V7 -- does the MIXED per-layer arm CONVERGE to its mortar-free twin?

The residual screen accepts every mixed in-plane / out-of-plane stack round 2
refused.  Accepting is only right if the answers are RIGHT, and a single modal
count cannot say so: the per-layer arm and the common-refinement (union) arm
are two different discretisations of the SAME device, so at one ``M`` they
differ by whatever neither has converged out yet.

This probe runs the LADDER on both arms and asks whether the gap closes.  A
gap that does NOT close is a silent wrong answer of exactly the shape the
audit asks to be hunted; a gap that closes like the arms' own steps is
ordinary discretisation.

The union arm is CONFORMING -- both layers on the common refinement -- so it
takes the plain square modal match and builds no mortar at all: it is a
numerically independent path to the same device.

Run: ``python v7_convergence.py <tag>``
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
from lumenairy.elements.pmm import _core as _pc             # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent

LADDERS = [
    # (kind, per-layer Ms, union kind, union Ms)
    ("mix_pattern", (4, 5, 6, 7), "mix_pattern_union", (4, 5)),
    ("mix_spacer", (4, 5, 6, 7), "mix_spacer_conf", (4, 5, 6, 7)),
    ("mix_slant_inplane", (4, 5, 6), "mix_slant_inplane_conf", (4, 5, 6)),
]


def _run(kind, M):
    t0 = time.time()
    with capture() as recs:
        st = F.build(kind, M)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            o, R, T = st.solve(jones=False)
    res = []
    for r in recs:
        X = np.linalg.solve(r["A"], r["B"])
        res.append(float(_pc._mortar_residual(r["A"], X, r["B"],
                                              probe=False)))
    return {"kind": kind, "M": M, "R00": F.R00(o, R),
            "closure": abs(float(R.sum(axis=1)[1] + T.sum(axis=1)[1]) - 1.0),
            "n_gen_sites": len(recs),
            "worst_residual": max(res) if res else None,
            "n_warn": len(ws), "seconds": round(time.time() - t0, 2)}


def main(tag):
    rows = []
    for kind, Ms, ukind, uMs in LADDERS:
        per = {}
        for M in Ms:
            per[M] = _run(kind, M)
            rows.append(dict(per[M], arm="per_layer", pair=kind))
            print(f"  {kind:22s} M={M} R00={per[M]['R00']:.12f} "
                  f"clo={per[M]['closure']:.2e} "
                  f"res={per[M]['worst_residual']} "
                  f"{per[M]['seconds']:.0f}s", flush=True)
        uni = {}
        for M in uMs:
            uni[M] = _run(ukind, M)
            rows.append(dict(uni[M], arm="union", pair=kind))
            print(f"  {ukind:22s} M={M} R00={uni[M]['R00']:.12f} "
                  f"clo={uni[M]['closure']:.2e} sites={uni[M]['n_gen_sites']} "
                  f"{uni[M]['seconds']:.0f}s", flush=True)
        # the DECISION: does |per(M) - union(Mmax)| fall as M rises, and does
        # it fall INTO the per-layer arm's own step?
        umax = uni[max(uMs)]["R00"]
        gaps = {M: abs(per[M]["R00"] - umax) for M in Ms}
        steps = {Ms[i + 1]: abs(per[Ms[i + 1]]["R00"] - per[Ms[i]]["R00"])
                 for i in range(len(Ms) - 1)}
        rows.append({"what": "summary", "pair": kind,
                     "union_M": max(uMs), "union_R00": umax,
                     "gap_vs_union": gaps, "per_layer_step": steps,
                     "gap_ratio_first_to_last":
                     (gaps[Ms[0]] / gaps[Ms[-1]] if gaps[Ms[-1]] else None)})
        print(f"  -> {kind}: gaps {[f'{gaps[M]:.3e}' for M in Ms]}  "
              f"steps {[f'{steps[M]:.3e}' for M in list(steps)]}", flush=True)
    out = {"env": F.env(), "tag": tag, "rows": rows}
    (HERE / f"v7_convergence_{tag}.json").write_text(
        json.dumps(out, indent=1), encoding="cp1252")
    print(f"wrote v7_convergence_{tag}.json ({len(rows)} rows)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "win")
