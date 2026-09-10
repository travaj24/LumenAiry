"""P1 -- REPRODUCE O-11 exactly as f5f_attrib.py does, oracle arm only.

f5f_attrib.py measured the 1-D ``PMMStack`` oracle's own degree-12-vs-14
self-gap and its shift from the ``delta -> 0`` reference while comparing a
2-D mortar arm.  This strips the 2-D arm: the same fixture, the same deltas,
the same two degrees, the same two ``layer_grids`` spellings.  Nothing here
touches the library.
"""
import json
import os
import warnings

import numpy as np

from lumenairy import PMMStack

assert os.environ.get("PYTHONPATH", "").replace("\\", "/").startswith("/c/tmp/lum_sliver") or True
import lumenairy
print("lumenairy:", lumenairy.__file__, lumenairy.__version__, flush=True)

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
PX = 1.2e-6
WL = 0.85e-6
THETA = 0.15
EPS_H, EPS_P = 2.25, 9.0
dz = 0.32e-6 / 4
XB0, XB1 = (0.1873, 0.7241), (0.2917, 0.6109)
zf = 1.0 - 0.5 / 4
A0 = XB0[0] + (XB1[0] - XB0[0]) * zf
B0 = XB0[1] + (XB1[1] - XB0[1]) * zf


def orc(fr, deg=14, per_layer=False, **kw2):
    kw = dict(layer_grids="per-layer") if per_layer else {}
    kw.update(kw2)
    s = PMMStack(PX, n_superstrate=1.0, n_substrate=1.0, degree=deg, **kw)
    for (a, b) in fr:
        s.add_layer(dz, segments=[(a, EPS_H), (b - a, EPS_P), (1.0 - b, EPS_H)])
    s.set_source(WL, theta=THETA)
    o, R, T = s.solve()[:3]
    o = np.asarray(o).ravel()
    i = np.argsort(o)
    return o[i], R[1][i], T[1][i]


def frames(d):
    return [(A0, B0), (A0 - d, B0 + d)]


if __name__ == "__main__":
    print(f"slice-1 walls: {A0:.10f} {B0:.10f}", flush=True)
    M0, R0, T0 = orc(frames(0.0))
    rows = []
    print(f"{'delta':>10} {'sliver(m)':>11} {'selfgap sh':>11} "
          f"{'selfgap PL':>11} {'orc(d)-orc(0)':>14} {'|R+T-1|':>10}", flush=True)
    for d in (1e-2, 2.6e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 0.0):
        fr = frames(d)
        MO, RO, TO = orc(fr)
        _m, R12, T12 = orc(fr, 12)
        sg = float(max(np.abs(RO - R12).max(), np.abs(TO - T12).max()))
        MP, RP, TP = orc(fr, 14, per_layer=True)
        _m, RP12, TP12 = orc(fr, 12, per_layer=True)
        sgp = float(max(np.abs(RP - RP12).max(), np.abs(TP - TP12).max()))
        dorc = float(max(np.abs(RO - R0).max(), np.abs(TO - T0).max()))
        clos = abs(float(RO.sum() + TO.sum()) - 1.0)
        rows.append(dict(delta=d, sliver_m=d * PX, selfgap_shared=sg,
                         selfgap_perlayer=sgp, shift_vs_delta0=dorc,
                         closure=clos))
        print(f"{d:10.2e} {d * PX:11.2e} {sg:11.2e} {sgp:11.2e} "
              f"{dorc:14.2e} {clos:10.2e}", flush=True)
        json.dump(rows, open(os.path.join(HERE, "p1_repro.json"), "w"), indent=1)
    print("\nwrote p1_repro.json", flush=True)
