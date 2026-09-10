"""P3 -- ONSET MAP: where the union-grid sliver breaks, vs delta and degree.

The wall-snap is DISABLED here (``min_feature`` pushed below the union grid's
own 1e-9 float-noise tol) so the raw sliver hazard is the only variable.  The
``delta -> 0`` reference is the two layers with IDENTICAL walls -- an EXACT
reference, because the physical structure is continuous in ``delta``.
"""
import json
import os
import sys
import warnings

import numpy as np
from p1_repro import PX, frames, orc
from p2_mech import diag

import lumenairy

print("lumenairy:", lumenairy.__file__, flush=True)
warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
MF_OFF = PX * 1e-10          # below _pmm_union_grid's 1e-9 fractional tol

DELTAS = [3e-3, 1e-3, 5e-4, 3e-4, 2e-4, 1.5e-4, 1.2e-4, 1e-4, 7e-5, 5e-5,
          3e-5, 1e-5, 3e-6]
DEGREES = [8, 12, 14, 16, 20]


def run(deg):
    ref = orc(frames(0.0), deg, min_feature=MF_OFF)
    rows = []
    for d in DELTAS:
        fr = frames(d)
        M, R, T = orc(fr, deg, min_feature=MF_OFF)
        _m, R2, T2 = orc(fr, deg + 2, min_feature=MF_OFF)
        sg = float(max(np.abs(R - R2).max(), np.abs(T - T2).max()))
        shift = float(max(np.abs(R - ref[1]).max(), np.abs(T - ref[2]).max()))
        clos = abs(float(R.sum() + T.sum()) - 1.0)
        g = diag(d, degree=deg, min_feature=MF_OFF)
        rows.append(dict(degree=deg, delta=d, w_min=g["w_min_frac"],
                         k0J=g["k0J"], q_max=g["q_max_L1"],
                         cond_ifc=g["cond_ifc_L0_L1"], S_ifc=g["ifc_L0_L1_max"],
                         selfgap=sg, shift=shift, closure=clos))
        print(f"deg{deg:3d} d={d:8.2e} w={g['w_min_frac']:8.2e} "
              f"k0J={g['k0J']:8.2e} qmax={g['q_max_L1']:8.2e} "
              f"cnd={g['cond_ifc_L0_L1']:8.2e} |S|={g['ifc_L0_L1_max']:8.2e} "
              f"sg={sg:8.2e} shift={shift:8.2e} clos={clos:8.2e}", flush=True)
    return rows


if __name__ == "__main__":
    degs = ([int(a) for a in sys.argv[1:]] or DEGREES)
    out = []
    for deg in degs:
        out += run(deg)
        json.dump(out, open(os.path.join(HERE, "p3_onset.json"), "w"), indent=1)
    print("\nwrote p3_onset.json", flush=True)
