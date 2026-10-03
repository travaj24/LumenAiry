"""V1 -- geometry of the transfinite maps, measured with THIS verifier's own
map constructions (from the primitives) and its own quadrature.

* my maps vs the builder's private builders, pointwise (positions + J);
* areas (Gauss-Legendre 64^2 per cell on det J, which is analytic);
* C0 on EVERY interior edge of every map (positions, tangent; and the size
  of the NORMAL-derivative jump, which is allowed);
* refusals: a mismatched vertex, a folding map, a curve whose s = 0 end is
  swapped;
* the ORDER of the zero of det J at every singular vertex (det J along rays
  from the vertex, fitted exponent) and the directional constant l(eta) of
  det J ~ rho l(eta), whose min / max over the quadrant controls the Duffy
  rule's convergence radius -- for the fillet as a function of r.

  python v1_geometry.py
Output: v1_geometry.json
"""
import _vcommon as C
import numpy as np
from numpy.polynomial.legendre import leggauss

CM = C.CM
res = {}


def pts_compare(a, b, n=9):
    xs = np.linspace(0.03, 0.97, n)
    worst = 0.0
    for sx in range(a.shape[0]):
        for sy in range(a.shape[1]):
            U = a.u_bounds[sx] + xs * (a.u_bounds[sx + 1] - a.u_bounds[sx])
            V = a.v_bounds[sy] + xs * (a.v_bounds[sy + 1] - a.v_bounds[sy])
            ga, gb = a.geom(sx, sy, U, V), b.geom(sx, sy, U, V)
            worst = max(worst, max(float(np.max(np.abs(x - y)))
                                   for x, y in zip(ga, gb)))
    return worst


# ---- my maps vs the builder's
r = 0.36
mine = {"c3": C.vcircle3(r)[0], "e3": C.vellipse3(0.40, 0.28)[0],
        "f5_0.2": C.vfillet(0.6, 0.12)[0], "f5_0.05": C.vfillet(0.6, 0.03)[0]}
theirs = {"c3": CM._circle_map_3x3(C.P, r)[0],
          "e3": CM._ellipse_map_3x3(C.P, (0.40, 0.28))[0],
          "f5_0.2": CM._fillet_map_5x5(C.P, 0.3, 0.12)[0],
          "f5_0.05": CM._fillet_map_5x5(C.P, 0.3, 0.03)[0]}
res["mine_vs_builder_max_abs"] = {k: pts_compare(mine[k], theirs[k])
                                  for k in mine}
res["fingerprint_equal"] = {k: mine[k].fingerprint == theirs[k].fingerprint
                            for k in mine}
print(res["mine_vs_builder_max_abs"], res["fingerprint_equal"], flush=True)

# ---- areas
xg, wg = leggauss(64)


def area(cm, cells):
    A = 0.0
    for sx, sy in cells:
        J1 = 0.5 * (cm.u_bounds[sx + 1] - cm.u_bounds[sx])
        J2 = 0.5 * (cm.v_bounds[sy + 1] - cm.v_bounds[sy])
        U = 0.5 * (cm.u_bounds[sx] + cm.u_bounds[sx + 1]) + J1 * xg
        V = 0.5 * (cm.v_bounds[sy] + cm.v_bounds[sy + 1]) + J2 * xg
        _X, _Y, xu, xv, yu, yv = cm.geom(sx, sy, U, V)
        A += float(np.sum(np.outer(wg, wg) * (xu * yv - xv * yu))) * J1 * J2
    return A


maps = {"c3": C.vcircle3(r), "c5_inner0.6": C.vcircle5(r, 0.6),
        "c3_r0.48": C.vcircle3(0.48), "c3_r0.24": C.vcircle3(0.24),
        "e3_0.42x0.24": C.vellipse3(0.42, 0.24),
        "f5_0.12": C.vfillet(0.6, 0.12), "f5_0.006": C.vfillet(0.6, 0.006),
        "f7_0.03_mid": C.vfillet(0.6, 0.03, split_mid=True),
        "f9_0.03_mid_out": C.vfillet(0.6, 0.03, True, True),
        "sine4_A0.08": C.vsine_ridge(0.35, 0.85, 0.08),
        "yuni_A0.1": C.vyuniform_curved(0.5, 0.9, 0.22, 0.1)}
exact_inner = {"c3": np.pi * r * r, "c5_inner0.6": np.pi * r * r,
               "c3_r0.48": np.pi * 0.48 ** 2, "c3_r0.24": np.pi * 0.24 ** 2,
               "e3_0.42x0.24": np.pi * 0.42 * 0.24,
               "f5_0.12": 0.36 - (4 - np.pi) * 0.12 ** 2,
               "f5_0.006": 0.36 - (4 - np.pi) * 0.006 ** 2,
               "f7_0.03_mid": 0.36 - (4 - np.pi) * 0.03 ** 2,
               "f9_0.03_mid_out": 0.36 - (4 - np.pi) * 0.03 ** 2,
               "sine4_A0.08": 0.5 * C.P, "yuni_A0.1": 0.4 * C.P}
res["area_rel_err"] = {}
res["singular_vertices"] = {}
for k, (cm, eps) in maps.items():
    nx, ny = cm.shape
    cells_in = [(i, j) for i in range(nx) for j in range(ny)
                if eps[i, j] != 1.0]
    allc = [(i, j) for i in range(nx) for j in range(ny)]
    res["area_rel_err"][k] = {
        "feature": abs(area(cm, cells_in) / exact_inner[k] - 1),
        "cell": abs(area(cm, allc) / C.P ** 2 - 1)}
    res["singular_vertices"][k] = cm.singular_vertices
print(res["area_rel_err"], flush=True)


# ---- C0 on every interior edge; the normal-derivative jump (allowed)
def c0(cm):
    nx, ny = cm.shape
    s = np.linspace(0.02, 0.98, 11)
    pos = tan = 0.0
    njump = 0.0
    for i in range(1, nx):          # vertical interior lines u = u_i
        for sy in range(ny):
            V = cm.v_bounds[sy] + s * (cm.v_bounds[sy + 1] - cm.v_bounds[sy])
            a = cm.geom(i - 1, sy, np.array([cm.u_bounds[i]]), V)
            b = cm.geom(i, sy, np.array([cm.u_bounds[i]]), V)
            pos = max(pos, float(np.max(np.abs(a[0] - b[0]))),
                      float(np.max(np.abs(a[1] - b[1]))))
            tan = max(tan, float(np.max(np.abs(a[3] - b[3]))),
                      float(np.max(np.abs(a[5] - b[5]))))
            njump = max(njump, float(np.max(np.abs(a[2] - b[2]))),
                        float(np.max(np.abs(a[4] - b[4]))))
    for j in range(1, ny):
        for sx in range(nx):
            U = cm.u_bounds[sx] + s * (cm.u_bounds[sx + 1] - cm.u_bounds[sx])
            a = cm.geom(sx, j - 1, U, np.array([cm.v_bounds[j]]))
            b = cm.geom(sx, j, U, np.array([cm.v_bounds[j]]))
            pos = max(pos, float(np.max(np.abs(a[0] - b[0]))),
                      float(np.max(np.abs(a[1] - b[1]))))
            tan = max(tan, float(np.max(np.abs(a[2] - b[2]))),
                      float(np.max(np.abs(a[4] - b[4]))))
            njump = max(njump, float(np.max(np.abs(a[3] - b[3]))),
                        float(np.max(np.abs(a[5] - b[5]))))
    return {"pos": pos, "tangent": tan, "normal_deriv_jump": njump}


res["C0"] = {k: c0(cm) for k, (cm, _e) in maps.items()}
print(res["C0"], flush=True)

# ---- refusals
ref = {}
cm, _ = C.vcircle3(r)
V = cm.vertex_images.copy()
for dv, tag in ((1e-9, "vertex_1e-9p"), (1e-11, "vertex_1e-11p"),
                (1e-13, "vertex_1e-13p")):
    V2 = V.copy()
    V2[2, 2, 1] += dv * C.P
    try:
        CM.TransfiniteMap(cm.u_bounds, cm.v_bounds, V2, dict(cm.curved_edges))
        ref[tag] = "accepted"
    except ValueError as e:
        ref[tag] = "refused: " + str(e)[:80]
# swapped orientation of one arc (s = 0 at the wrong vertex)
ed = dict(cm.curved_edges)
a = ed[("h", 1, 1)]
ed[("h", 1, 1)] = CM.Arc(a.center, a.radius, a.theta1, a.theta0)
try:
    CM.TransfiniteMap(cm.u_bounds, cm.v_bounds, V, ed)
    ref["swapped_arc"] = "accepted"
except ValueError as e:
    ref["swapped_arc"] = "refused: " + str(e)[:80]
# folding: an arc bulging the WRONG way (the long way round: centre mirrored)
ed = dict(cm.curved_edges)
P0, P1 = V[1, 1], V[2, 1]
cmir = np.array([C.P / 2, 2 * V[1, 1, 1] - C.P / 2])   # mirror of the centre
ed[("h", 1, 1)] = CM.Arc.through(P0, P1, cmir)
try:
    CM.TransfiniteMap(cm.u_bounds, cm.v_bounds, V, ed)
    ref["inward_bulge_bottom_arc"] = "accepted"
except ValueError as e:
    ref["inward_bulge_bottom_arc"] = "refused: " + str(e)[:80]
# folding: a circle so large its 45-degree points leave the cell
for rr in (0.55, 0.62, 0.8):
    try:
        C.vcircle3(rr)
        ref[f"circle_r{rr}"] = "accepted"
    except ValueError as e:
        ref[f"circle_r{rr}"] = "refused: " + str(e)[:80]
# a sinusoid of amplitude > the wall offset (the curve crosses x = 0)
try:
    C.vsine_ridge(0.10, 0.85, 0.15)
    ref["sine_A_gt_x1"] = "accepted"
except ValueError as e:
    ref["sine_A_gt_x1"] = "refused: " + str(e)[:80]
res["refusals"] = ref
print(ref, flush=True)


# ---- order of the zero of det J at the singular vertices
def detj_order(cm):
    out = []
    for sx, sy, cu, cv in cm.singular_vertices:
        u0 = cm.u_bounds[sx + cu]
        v0 = cm.v_bounds[sy + cv]
        du = cm.u_bounds[sx + 1] - cm.u_bounds[sx]
        dv = cm.v_bounds[sy + 1] - cm.v_bounds[sy]
        su, sv = (1 - 2 * cu), (1 - 2 * cv)      # into the cell
        expo, lmin, lmax = [], np.inf, 0.0
        for eta in np.linspace(0.0, 1.0, 21):      # direction across the quadrant
            d = np.array([su * du * (1 - eta), sv * dv * eta])
            rho = np.array([1e-3, 1e-4, 1e-5])
            dets = []
            for rr in rho:
                g = cm.geom_points(sx, sy, np.array([u0 + rr * d[0]]),
                                   np.array([v0 + rr * d[1]]))
                dets.append(float(g[2][0] * g[5][0] - g[3][0] * g[4][0]))
            dets = np.array(dets)
            p = np.polyfit(np.log(rho), np.log(np.abs(dets)), 1)[0]
            expo.append(p)
            ell = dets[-1] / rho[-1]
            lmin, lmax = min(lmin, ell), max(lmax, ell)
        out.append({"vertex": (sx, sy, cu, cv),
                    "exponent_min": float(min(expo)),
                    "exponent_max": float(max(expo)),
                    "l_min": float(lmin), "l_max": float(lmax),
                    "l_ratio": float(lmin / lmax)})
    return out


res["detJ_zero"] = {}
for k in ("c3", "c5_inner0.6", "c3_r0.48", "e3_0.42x0.24", "f5_0.12",
          "f5_0.006", "f9_0.03_mid_out"):
    res["detJ_zero"][k] = detj_order(maps[k][0])
    d = res["detJ_zero"][k]
    print(k, [(x["vertex"], round(x["exponent_min"], 6),
               round(x["exponent_max"], 6), round(x["l_ratio"], 4))
              for x in d], flush=True)
C.dump(f"v1_geometry_{C.build_tag()}.json", res)
