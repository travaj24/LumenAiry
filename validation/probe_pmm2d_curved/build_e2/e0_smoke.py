"""E0 smoke: the curved cross-mass kernel against the two exact references
it must reproduce -- (a) the SAME map on both sides is the plain Gram, (b) two
UNMAPPED grids with different walls are the shipped separable cross-mass."""
import sys
import time

import numpy as np
from _common import CM, TS, P, dump

from lumenairy.elements.pmm import _curvemortar as CMM


def dense(pair):
    return np.kron(pair[0], pair[1])


def blocks_ref(ga, gb, cr):
    qa, qb = ga.qq, gb.qq
    X = np.zeros((2 * qb, 2 * qa), complex)
    X[:qb, :qa] = dense(cr.C1H())
    X[qb:, qa:] = dense(cr.C2H())
    return X


out = {}
M = int(sys.argv[1]) if len(sys.argv) > 1 else 4
cm, _w = CM._circle_map_3x3(P, 0.36)
for tau in (1.0, np.exp(-0.7j)):
    g = TS.StagGridOps(P, P, cm.u_walls, cm.v_walls, M, tau, tau, cmap=cm)
    t0 = time.perf_counter()
    X = CMM.curved_cross_mass(g, g, n=2 * M + 8)
    dt = time.perf_counter() - t0
    G = np.zeros_like(X)
    G[:g.qq, :g.qq] = dense(g.V1)
    G[g.qq:, g.qq:] = dense(g.V2)
    err = float(np.abs(X - G).max() / np.abs(G).max())
    out[f"same_circle_tau{tau}"] = dict(rel=err, wall=dt)
    print("same circle map, tau", tau, "rel", err, "t", dt)
# unmapped, different walls (through the kernel with identity views)
for tau in (1.0, np.exp(-0.7j)):
    ga = TS.StagGridOps(P, P, np.array([0, 0.3, 0.8, P]),
                        np.array([0, 0.5, 0.7, P]), M, tau, tau)
    gb = TS.StagGridOps(P, P, np.array([0, 0.45, P]), np.array([0, 0.2, P]),
                        M + 1, tau, tau)
    cr = TS.StagCrossOps(ga, gb)
    Xr = blocks_ref(ga, gb, cr)
    # route through the kernel by giving both sides an IDENTITY map object
    ga.cmap = CM.IdentityMap(ga.bx.xb, ga.by.xb)
    gb.cmap = CM.IdentityMap(gb.bx.xb, gb.by.xb)
    X = CMM.curved_cross_mass(ga, gb, n=2 * M + 8)
    err = float(np.abs(X - Xr).max() / np.abs(Xr).max())
    out[f"unmapped_walls_tau{tau}"] = dict(rel=err)
    print("identity maps, different walls, tau", tau, "rel", err)
dump(f"e0_smoke_M{M}.json", out)
