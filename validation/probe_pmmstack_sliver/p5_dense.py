"""P5 -- DENSE delta sweep: is the failure ENERGY-VISIBLE, and by how much?

O-11 was logged as the silent-wrongness class on the strength of the MORTAR
arm's closure (1.6e-08).  This measures the 1-D ``PMMStack``'s OWN closure on
the same fixture, densely, so the two populations (right / wrong) can be
scored against every candidate discriminator.

"WRONG" is defined without a fitted constant: the structure is CONTINUOUS in
``delta`` and the measured shift is linear (``err ~ 1.15 delta``), so an answer
further than ``100 * delta`` from the ``delta -> 0`` reference cannot be the
physical shift.
"""
import json
import os
import sys
import warnings

import numpy as np
from p1_repro import PX, frames, orc

import lumenairy

print("lumenairy:", lumenairy.__file__, flush=True)
warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
MF_OFF = PX * 1e-10


def run(deg, deltas):
    ref = orc(frames(0.0), deg, min_feature=MF_OFF)
    rows = []
    for d in deltas:
        M, R, T = orc(frames(d), deg, min_feature=MF_OFF)
        err = float(max(np.abs(R - ref[1]).max(), np.abs(T - ref[2]).max()))
        tot = float(R.sum() + T.sum())
        rows.append(dict(degree=deg, delta=float(d), err=err, total=tot,
                         closure=abs(tot - 1.0), wrong=bool(err > 100.0 * d)))
    return rows


if __name__ == "__main__":
    degs = [int(a) for a in sys.argv[1:]] or [12, 14, 20]
    deltas = np.geomspace(3e-3, 3e-6, 46)
    out = []
    for deg in degs:
        rows = run(deg, deltas)
        out += rows
        nb = sum(r["wrong"] for r in rows)
        gm = max([r["closure"] for r in rows if not r["wrong"]] or [0.0])
        bm = min([r["closure"] for r in rows if r["wrong"]] or [np.inf])
        gt = max([abs(r["total"] - 1.0) for r in rows if not r["wrong"]]
                 or [0.0])
        bt = min([r["total"] - 1.0 for r in rows if r["wrong"]] or [np.inf])
        print(f"degree {deg}: {nb}/{len(rows)} WRONG; "
              f"max closure among RIGHT = {gm:.3e}; "
              f"min closure among WRONG = {bm:.3e}; "
              f"min (R+T-1) among WRONG = {bt:.3e}", flush=True)
        for r in rows:
            print(f"   d={r['delta']:9.3e} err={r['err']:9.3e} "
                  f"R+T-1={r['total'] - 1.0:+10.3e} "
                  f"{'WRONG' if r['wrong'] else ''}", flush=True)
        json.dump(out, open(os.path.join(HERE, "p5_dense.json"), "w"), indent=1)
    print("\nwrote p5_dense.json", flush=True)
