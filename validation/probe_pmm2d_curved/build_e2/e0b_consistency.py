"""E0b: the curved cross-mass is ONE integral; computed in either layer's
coordinates it must agree (different node sets, different weights), and the
plain tensor rule that ignores the other layer's walls must trend to it."""
import sys
import time

import numpy as np
from _common import R_CIRC, TS, P, dump

from lumenairy.elements.pmm import _curvemortar as CMM
from lumenairy.elements.pmm.shapes2d import Circle, SinusoidalWall, compile_shapes

M = int(sys.argv[1]) if len(sys.argv) > 1 else 4
tau = np.exp(-0.4j)


def grid(cm, M):
    return TS.StagGridOps(P, P, cm.u_bounds, cm.v_bounds, M, tau, tau,
                          cmap=cm)


_e, _x, _y, sinx = compile_shapes(P, P, [SinusoidalWall("x", 0.6, 0.12,
                                                        eps=2.25)], 1.0)
_e, _x, _y, siny = compile_shapes(P, P, [SinusoidalWall("y", 0.5, 0.1,
                                                        eps=2.25)], 1.0)
_e, _x, _y, circ = compile_shapes(P, P, [Circle(0.6, 0.6, R_CIRC, 4.0)], 1.0)
out = {}
# (1) two NON-singular maps: a vs b coordinates
ga, gb = grid(sinx, M), grid(siny, M + 1)
res = {}
for n in (8, 12, 16, 24, 32):
    t0 = time.perf_counter()
    Xa = CMM.curved_cross_mass(ga, gb, n, primary="a")
    ta = time.perf_counter() - t0
    Xb = CMM.curved_cross_mass(ga, gb, n, primary="b")
    res[n] = dict(Xa=Xa, Xb=Xb, ta=ta)
ref = res[32]["Xa"]
sc = np.abs(ref).max()
rows = []
for n, r in res.items():
    rows.append(dict(n=n, a_vs_ref=float(np.abs(r["Xa"] - ref).max() / sc),
                     b_vs_ref=float(np.abs(r["Xb"] - ref).max() / sc),
                     a_vs_b=float(np.abs(r["Xa"] - r["Xb"]).max() / sc),
                     t_a=r["ta"]))
    print("sinx|siny", rows[-1])
out["sinx_siny"] = rows
# (2) circle (a) vs sinusoid (b): auto (circle coords) vs the tensor rule
ga, gb = grid(circ, M), grid(sinx, M)
res = {}
for n in (8, 12, 16, 24, 32):
    t0 = time.perf_counter()
    X = CMM.curved_cross_mass(ga, gb, n)
    res[n] = (X, time.perf_counter() - t0)
ref = res[32][0]
sc = np.abs(ref).max()
rows = []
for n, (X, t) in res.items():
    rows.append(dict(n=n, cut_vs_ref=float(np.abs(X - ref).max() / sc), t=t))
    print("circle|sinx cut", rows[-1])
for n in (16, 32, 64, 128):
    X = CMM.curved_cross_mass(ga, gb, n, cut=False)
    rows.append(dict(n=n, nocut_vs_ref=float(np.abs(X - ref).max() / sc)))
    print("circle|sinx nocut", rows[-1])
for n in (16, 32):
    X = CMM.curved_cross_mass(ga, gb, n, primary="b")
    rows.append(dict(n=n, sinus_coords_vs_ref=float(np.abs(X - ref).max()
                                                    / sc)))
    print("circle|sinx in sinusoid coords", rows[-1])
out["circle_sinx"] = rows
dump(f"e0b_consistency_M{M}.json", out)
