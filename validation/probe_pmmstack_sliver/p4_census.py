"""P4 -- do the SHIPPED T3-4 instruments see the sliver?

Arms ``_MODE_CUT_CENSUS`` (and the disarmed ``PMM_MODE_CUT_GUARD``) around the
same sweep and reports, per delta/degree, the raw and residual growing-mode
counts, the classification margin, and ``q_excess`` -- alongside the answer's
own error against the ``delta -> 0`` reference.  It answers whether this defect
is the T3-4 CLASSIFICATION family (in which case the existing repair/guard is
the lever) or a separate CONDITIONING family.
"""
import json
import os
import warnings

import numpy as np
from p1_repro import PX, frames, orc

import lumenairy
from lumenairy.elements.pmm import _core as pc

print("lumenairy:", lumenairy.__file__, flush=True)
warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
MF_OFF = PX * 1e-10


def census(d, deg):
    pc._MODE_CUT_CENSUS = rows = []
    try:
        out = orc(frames(d), deg, min_feature=MF_OFF)
    finally:
        pc._MODE_CUT_CENSUS = None
    pat = [r for r in rows if r["patterned"]]
    return out, pat


if __name__ == "__main__":
    res = []
    for deg in (8, 12, 14, 16, 20):
        ref = orc(frames(0.0), deg, min_feature=MF_OFF)
        print(f"--- degree {deg} " + "-" * 60, flush=True)
        print(f"{'delta':>9} {'err':>9} {'clos':>9} {'ngrow':>6} {'npost':>6} "
              f"{'nrisk':>6} {'margin':>9} {'qexc':>9} {'nprop':>12}",
              flush=True)
        for d in (1e-3, 3e-4, 2e-4, 1.2e-4, 1e-4, 7e-5, 5e-5, 3e-5, 1e-5):
            (M, R, T), pat = census(d, deg)
            err = float(max(np.abs(R - ref[1]).max(), np.abs(T - ref[2]).max()))
            clos = abs(float(R.sum() + T.sum()) - 1.0)
            ng = sum(r["n_grow"] for r in pat)
            npo = sum(r["n_grow_post"] for r in pat)
            nr = sum(r["n_risk"] for r in pat)
            mg = min([r["margin"] for r in pat] or [float("inf")])
            qe = max([r["q_excess"] for r in pat] or [0.0])
            npr = [r["n_prop"] for r in pat]
            res.append(dict(degree=deg, delta=d, err=err, closure=clos,
                            n_grow=ng, n_grow_post=npo, n_risk=nr,
                            margin=mg, q_excess=qe, n_prop=npr))
            print(f"{d:9.2e} {err:9.2e} {clos:9.2e} {ng:6d} {npo:6d} "
                  f"{nr:6d} {mg:9.3g} {qe:9.3g} {str(npr):>12}", flush=True)
            json.dump(res, open(os.path.join(HERE, "p4_census.json"), "w"),
                      indent=1)
    print("\nwrote p4_census.json", flush=True)
