"""E2-5: the curved mortar's OWN convergence -- the cross-mass quadrature
error against the node count n (per sub-interval per direction) at FIXED M,
for the E2-4 pair (circle over a crossing sinusoid: singular vertices,
tangencies, cut cells) and a non-singular pair (sinusoid-x over sinusoid-y),
each against its own n = 3 x (adaptive) reference, plus the node count the
adaptive rule stops at and its last relative change.

Usage: python e2_5_quadrature.py <M>"""
import sys
import time

import numpy as np
from _common import R_CIRC, TS, P, dump

from lumenairy.elements.pmm import _curvemortar as CMM
from lumenairy.elements.pmm.shapes2d import (
    Circle,
    SinusoidalWall,
    compile_shapes,
)

M = int(sys.argv[1])
tau = np.exp(-0.4j)
_e, _x, _y, circ = compile_shapes(P, P, [Circle(0.6, 0.6, R_CIRC, 4.0)], 1.0)
_e, _x, _y, sinx = compile_shapes(P, P, [SinusoidalWall("x", 0.6, 0.12,
                                                        eps=2.25)], 1.0)
_e, _x, _y, siny = compile_shapes(P, P, [SinusoidalWall("y", 0.5, 0.1,
                                                        eps=2.25)], 1.0)
Ms_q = -(-3 * (M - 1) // 2) + 1          # the q-matched sinusoid layer
out = {"M": M}
for name, (ca, Ma), (cb, Mb) in (("circle_sinx", (circ, M), (sinx, Ms_q)),
                                 ("sinx_siny", (sinx, M), (siny, M))):
    ga = TS.StagGridOps(P, P, ca.u_walls, ca.v_walls, Ma, tau, tau, cmap=ca)
    gb = TS.StagGridOps(P, P, cb.u_walls, cb.v_walls, Mb, tau, tau, cmap=cb)
    t0 = time.perf_counter()
    Xa, na, chg = CMM.curved_cross_mass_adaptive(ga, gb)
    ta = time.perf_counter() - t0
    ref = CMM.curved_cross_mass(ga, gb, 3 * na)
    sc = float(np.abs(ref).max())
    rows = []
    for n in sorted({4, 6, 8, 10, 12, 16, 20, 24, 32, na}):
        t0 = time.perf_counter()
        X = CMM.curved_cross_mass(ga, gb, n)
        rows.append(dict(n=n, rel=float(np.abs(X - ref).max() / sc),
                         t=time.perf_counter() - t0))
    out[name] = dict(Ma=Ma, Mb=Mb, adaptive_n=na, adaptive_change=chg,
                     adaptive_vs_ref=float(np.abs(Xa - ref).max() / sc),
                     adaptive_wall=ta, ladder=rows, ref_n=3 * na)
    print(name, {k: v for k, v in out[name].items() if k != "ladder"})
    for r in rows:
        print("   ", r)
dump(f"e2_5_quadrature_M{M}.json", out)
