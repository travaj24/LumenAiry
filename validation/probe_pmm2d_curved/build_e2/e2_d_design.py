"""E2 DESIGN MEASUREMENT (deliverable 1): the three numbers the choice among
(i) a common fine parametrisation by inverting one map at quadrature nodes,
(ii) a composite map on the union grid, (iii) one merged map, rests on.

  (a) the cost of the Newton map inversion per quadrature node (circle and
      sinusoid maps, a node set of the cut-cell rule);
  (b) the conditioning of the non-separable cross-mass X = CrossE^H (its
      singular values against the SEPARABLE cross-mass of two unmapped
      grids of the same sizes, and the plain Gram);
  (c) whether (ii) is exact: the cross-mass by ONE tensor Gauss rule per cell
      of a composite parametrisation (the circle layer's cells), each layer's
      basis pulled back through it -- the other layer's walls are then
      curves INSIDE those cells -- against the cut-cell rule, vs node count.

Usage: python e2_d_design.py <M>"""
import sys
import time

import numpy as np
from _common import R_CIRC, TS, P, dump

from lumenairy.elements.pmm import _curvemortar as CMM
from lumenairy.elements.pmm.shapes2d import Circle, SinusoidalWall, compile_shapes

M = int(sys.argv[1]) if len(sys.argv) > 1 else 4
tau = np.exp(-0.4j)
_e, _x, _y, circ = compile_shapes(P, P, [Circle(0.6, 0.6, R_CIRC, 4.0)], 1.0)
_e, _x, _y, sinx = compile_shapes(P, P, [SinusoidalWall("x", 0.6, 0.12,
                                                        eps=2.25)], 1.0)
ga = TS.StagGridOps(P, P, circ.u_walls, circ.v_walls, M, tau, tau, cmap=circ)
gb = TS.StagGridOps(P, P, sinx.u_walls, sinx.v_walls, M, tau, tau, cmap=sinx)
out = {"M": M}
# (a) Newton inversion cost per node
A = CMM._MapView(circ, circ.u_bounds, circ.v_bounds, P, P)
B = CMM._MapView(sinx, sinx.u_bounds, sinx.v_bounds, P, P)
rows = {}
for nm, Mv in (("circle", A), ("sinusoid", B)):
    rng = np.random.default_rng(1)
    tot, cnt = 0.0, 0
    for sx in range(Mv.Nx):
        for sy in range(Mv.Ny):
            U = Mv.ub[sx] + rng.random(4000) * (Mv.ub[sx + 1] - Mv.ub[sx])
            V = Mv.vb[sy] + rng.random(4000) * (Mv.vb[sy + 1] - Mv.vb[sy])
            X, Y = Mv.geom(sx, sy, U, V)[:2]
            t0 = time.perf_counter()
            Ui, Vi, ok = Mv.invert(sx, sy, X, Y)
            tot += time.perf_counter() - t0
            cnt += U.size
            assert ok.all()
            err = float(max(np.abs(Ui - U).max(), np.abs(Vi - V).max()))
            rows.setdefault(nm + "_maxerr", 0.0)
            rows[nm + "_maxerr"] = max(rows[nm + "_maxerr"], err)
    rows[nm + "_us_per_node"] = 1e6 * tot / cnt
geom_t = []
for _ in range(20):
    U = np.linspace(0.35, 0.85, 4000)
    t0 = time.perf_counter()
    A.geom(1, 1, U, U)
    geom_t.append(time.perf_counter() - t0)
rows["circle_geom_us_per_point"] = 1e6 * min(geom_t) / 4000
out["inversion"] = rows
print("inversion", rows)
# (b) conditioning
X, n, chg = CMM.curved_cross_mass_adaptive(ga, gb)
s = np.linalg.svd(X, compute_uv=False)
g0a = TS.StagGridOps(P, P, circ.u_bounds, circ.v_bounds, M, tau, tau)
g0b = TS.StagGridOps(P, P, sinx.u_bounds, sinx.v_bounds, M, tau, tau)
cr = TS.StagCrossOps(g0a, g0b)
Xs = np.zeros_like(X)
Xs[:g0b.qq, :g0a.qq] = np.kron(*cr.C1H())
Xs[g0b.qq:, g0a.qq:] = np.kron(*cr.C2H())
ss = np.linalg.svd(Xs, compute_uv=False)
Gb = np.zeros((2 * gb.qq, 2 * gb.qq), complex)
Gb[:gb.qq, :gb.qq] = np.kron(*gb.V1)
Gb[gb.qq:, gb.qq:] = np.kron(*gb.V2)
sg = np.linalg.svd(Gb, compute_uv=False)
out["conditioning"] = dict(
    n=n, change=chg, curved_smax=float(s[0]), curved_smin=float(s[-1]),
    curved_cond=float(s[0] / s[-1]), separable_smax=float(ss[0]),
    separable_smin=float(ss[-1]), separable_cond=float(ss[0] / ss[-1]),
    gram_b_cond=float(sg[0] / sg[-1]), shape=list(X.shape))
print("conditioning", out["conditioning"])
# (c) composite parametrisation (one tensor rule per circle cell) vs cut
ref = CMM.curved_cross_mass(ga, gb, 2 * n)
sc = float(np.abs(ref).max())
rows = []
for nn in (8, 16, 32, 64, 128, 256):
    t0 = time.perf_counter()
    Xc = CMM.curved_cross_mass(ga, gb, nn, cut=False)
    rows.append(dict(n=nn, rel=float(np.abs(Xc - ref).max() / sc),
                     t=time.perf_counter() - t0))
    print("composite", rows[-1])
cut = []
for nn in (6, 9, 12, 18, 27):
    t0 = time.perf_counter()
    Xc = CMM.curved_cross_mass(ga, gb, nn)
    cut.append(dict(n=nn, rel=float(np.abs(Xc - ref).max() / sc),
                    t=time.perf_counter() - t0))
    print("cut", cut[-1])
out["composite_vs_cut"] = dict(composite=rows, cut=cut, ref_n=2 * n)
dump(f"e2_d_design_M{M}.json", out)
