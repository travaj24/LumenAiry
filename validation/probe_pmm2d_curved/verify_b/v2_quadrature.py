"""V2 -- THE QUADRATURE DECISION, re-derived and re-measured independently.

Singularity.  At a singular vertex det J has a SIMPLE zero (v1: exponent
1.0000 on every ray, every map), det J = rho l(eta) + O(rho^2), so the five
geometric weights (1 / sqrt g and g_ij / sqrt g) are homogeneous of degree -1
at the corner: w ~ A(eta) / rho.  Consequences derived in the report:
  * tensor Gauss-Legendre: the inner integral over t of 1 / (a s + b t) is
    analytic for s > 0 but its result ~ log(1 / s) -- the 1-D marginal has a
    LOG endpoint singularity, and Gauss-Legendre on log(s) p(s) converges as
    n^-2 exactly;
  * a tensor Gauss-Jacobi rule (weight (1 - x^2)^a per axis) cannot absorb a
    log (no algebraic exponent matches) and the 2-D singularity is not a
    product of 1-D ones;
  * the Duffy collapse s = xi a(eta), t = xi b(eta) gives w dA =
    [A(eta) / l(eta) + O(xi)] dxi deta: the transformed integrand is
    ANALYTIC (the exponent of xi is exactly 0) -- i.e. Duffy + Gauss-Legendre
    IS the Gauss-Jacobi rule with the derived exponent (0) in the collapsed
    variable; a Jacobi weight xi^1 (using the Duffy Jacobian as the Jacobi
    weight instead of multiplying by it) leaves a 1 / xi pole and fails.

Rules (on the reference square [-1, 1]^2 of a cell with singular corners):
  gl        tensor Gauss-Legendre n x n
  gj<a>     tensor Gauss-Jacobi, weight (1 - x)^a (1 + x)^a per axis
  duffy     the BUILDER's _stag_duffy_points
  vduffy    THIS verifier's own Duffy rule (independent code: quadrant
            split, collapse onto the corner, Gauss-Legendre in xi and eta)
  vduffy_gj1  the same collapse with Gauss-Jacobi weight xi^1 in xi and the
            Jacobian xi NOT multiplied in (the 'Jacobi with the Jacobian as
            weight' variant)
  vduffy_sq the collapse xi -> xi^2 (over-collapsed: Jacobian xi^3)

Modes:
  moments                 -- moment error vs a vduffy n = 96 reference on the
                             singular cells of c3 / c5 / fillet / ellipse
  rt <map> <M> <rule> <n> -- a FULL library solve with the corner cells on the
                             given rule and the node count forced to n
Output: v2_moments_<build>.json, v2_rt_<map>_M<M>_<rule>_n<n>.json
"""
import sys

import _vcommon as C
import numpy as np
from numpy.polynomial.legendre import leggauss, legvander
from scipy.special import roots_jacobi

TS = C.TS


def rule_gl(corners, n):
    x, w = leggauss(n)
    return np.repeat(x, n), np.tile(x, n), np.outer(w, w).ravel()


def rule_gj(a):
    def f(corners, n):
        x, w = roots_jacobi(n, a, a)
        w = w / ((1 - x) ** a * (1 + x) ** a)
        return np.repeat(x, n), np.tile(x, n), np.outer(w, w).ravel()
    return f


def _collapse(corners, n, xi_rule="gl", power=1):
    """Own Duffy: split into quadrants when >1 corner; each piece that owns
    a corner c is cut by its diagonal into two triangles (c, A, o), each the
    image of [0,1]^2 under P = c + xi^power ((1 - eta)(A - c) + eta (o - c))
    -- Jacobian power xi^(2 power - 1) |det(A - c, o - c)|."""
    xg, wg = leggauss(n)
    e = 0.5 * (xg + 1)
    we = 0.5 * wg
    if xi_rule == "gl":
        xi, wxi = e, we
    else:                                  # Gauss-Jacobi weight xi^1 on [0,1]
        x, w = roots_jacobi(n, 0.0, 1.0)  # weight (1-x)^0 (1+x)^1 on [-1,1]
        xi = 0.5 * (x + 1)
        wxi = w / 4.0                      # (1+x) = 2 xi, dx = 2 dxi
    corners = [tuple(c) for c in corners]
    if len(corners) > 1:
        pieces = []
        for qs in (-1, 1):
            for qt in (-1, 1):
                lo = (min(0, qs), min(0, qt))
                hi = (max(0, qs), max(0, qt))
                pieces.append((lo, hi, [c for c in corners if c == (qs, qt)]))
    else:
        pieces = [((-1, -1), (1, 1), corners)]
    S, T, W = [], [], []
    for lo, hi, own in pieces:
        lo, hi = np.array(lo, float), np.array(hi, float)
        if not own:
            x0 = 0.5 * (lo + hi)
            hw = 0.5 * (hi - lo)
            S.append(np.repeat(x0[0] + hw[0] * xg, n))
            T.append(np.tile(x0[1] + hw[1] * xg, n))
            W.append(np.outer(wg, wg).ravel() * hw[0] * hw[1])
            continue
        cs, ct = own[0]
        c = np.array([hi[0] if cs > 0 else lo[0], hi[1] if ct > 0 else lo[1]])
        o = np.array([lo[0] if cs > 0 else hi[0], lo[1] if ct > 0 else hi[1]])
        for A in (np.array([o[0], c[1]]), np.array([c[0], o[1]])):
            d1, d2 = A - c, o - c
            area2 = abs(d1[0] * d2[1] - d1[1] * d2[0])
            XI, ET = np.meshgrid(xi, e, indexing="ij")
            WW = np.outer(wxi, we)
            Xp = XI ** power
            dirx = (1 - ET) * d1[0] + ET * d2[0]
            diry = (1 - ET) * d1[1] + ET * d2[1]
            S.append((c[0] + Xp * dirx).ravel())
            T.append((c[1] + Xp * diry).ravel())
            if xi_rule == "gl":
                jac = power * XI ** (2 * power - 1) * area2
            else:
                jac = area2 * np.ones_like(XI)   # the xi is in the weight
            W.append((WW * jac).ravel())
    return np.concatenate(S), np.concatenate(T), np.concatenate(W)


RULES = {"gl": rule_gl, "gj-0.5": rule_gj(-0.5), "gj-0.25": rule_gj(-0.25),
         "gj0.5": rule_gj(0.5),
         "duffy": TS._stag_duffy_points,
         "vduffy": lambda c, n: _collapse(c, n),
         "vduffy_gj1": lambda c, n: _collapse(c, n, "gj1"),
         "vduffy_sq": lambda c, n: _collapse(c, n, power=2)}


def maps():
    return {"c3": C.vcircle3(0.36), "c3_r0.48": C.vcircle3(0.48),
            "c5": C.vcircle5(0.36, 0.6), "f5_0.2": C.vfillet(0.6, 0.12),
            "f5_0.02": C.vfillet(0.6, 0.012),
            "e3": C.vellipse3(0.42, 0.24)}


def geom5(cm, sx, sy, s, t):
    du = 0.5 * (cm.u_bounds[sx + 1] - cm.u_bounds[sx])
    dv = 0.5 * (cm.v_bounds[sy + 1] - cm.v_bounds[sy])
    U = 0.5 * (cm.u_bounds[sx + 1] + cm.u_bounds[sx]) + du * s
    V = 0.5 * (cm.v_bounds[sy + 1] + cm.v_bounds[sy]) + dv * t
    _X, _Y, xu, xv, yu, yv = cm.geom_points(sx, sy, U, V)
    sg = xu * yv - xv * yu
    return [sg, 1 / sg, (xu * xu + yu * yu) / sg, (xu * xv + yu * yv) / sg,
            (xv * xv + yv * yv) / sg]


def moments(cm, cell, cs, rule, n, deg):
    s, t, w = RULES[rule](cs, n)
    f = geom5(cm, cell[0], cell[1], s, t)
    Ps = legvander(s, deg) * w[:, None]
    Pt = legvander(t, deg)
    return [Ps.T @ (fk[:, None] * Pt) for fk in f]


def run_moments():
    out = {}
    for name, (cm, _eps) in maps().items():
        cells = TS._stag_map_singular_corners(cm)
        for M in (6, 8):
            deg = 2 * M - 2
            refs = {cell: moments(cm, cell, cs, "vduffy", 96, deg)
                    for cell, cs in cells.items()}
            refs2 = {cell: moments(cm, cell, cs, "vduffy", 64, deg)
                     for cell, cs in cells.items()}
            ref_gap = max(max(float(np.max(np.abs(a - b)) / np.max(np.abs(b)))
                              for a, b in zip(refs2[c], refs[c]))
                          for c in cells)
            rows = {"ref_selfgap_64_vs_96": ref_gap}
            for rule, ns in (("gl", (10, 20, 40, 80, 160, 320, 640, 1280)),
                             ("gj-0.5", (10, 20, 40, 80, 160, 320, 640)),
                             ("gj-0.25", (10, 20, 40, 80, 160, 320, 640)),
                             ("gj0.5", (10, 20, 40, 80, 160, 320)),
                             ("duffy", (4, 6, 8, 10, 12, 16, 20, 24, 32)),
                             ("vduffy", (4, 6, 8, 10, 12, 16, 20, 24, 32)),
                             ("vduffy_gj1", (4, 8, 16, 32, 64, 128, 256)),
                             ("vduffy_sq", (4, 8, 12, 16, 24, 32, 48))):
                rows[rule] = {}
                for n in ns:
                    err = 0.0
                    for cell, cs in cells.items():
                        m = moments(cm, cell, cs, rule, n, deg)
                        err = max(err, max(
                            float(np.max(np.abs(a - b)) / np.max(np.abs(b)))
                            for a, b in zip(m, refs[cell])))
                    rows[rule][n] = err
                print(name, M, rule, {k: f"{v:.2e}" for k, v in
                                      rows[rule].items()}, flush=True)
            out[f"{name}_M{M}"] = rows
    # the analytic statement: Gauss-Legendre on log(s) on [0, 1]: n^-2
    lg = {}
    for n in (10, 20, 40, 80, 160, 320, 640):
        x, w = leggauss(n)
        s = 0.5 * (x + 1)
        lg[n] = abs(float(np.sum(0.5 * w * np.log(s))) + 1.0)
    out["gl_on_log_s"] = lg
    print("GL on log s:", {k: f"{v:.2e}" for k, v in lg.items()})
    C.dump(f"v2_moments_{C.build_tag()}.json", out)


def run_rt(mapname, M, rule, n):
    cm, eps = maps()[mapname]
    o_r, o_n = TS._stag_duffy_points, TS._stag_map_nodes
    TS._stag_duffy_points = RULES[rule]
    TS._stag_map_nodes = lambda *a, **k: int(n)
    try:
        o, R, T, st, warns = C.solve_map(cm, eps, M)
        s = TS.Granet2DTransverseE(C.P, C.P, cm.u_walls, cm.v_walls, M, eps,
                                   k0=C.K0, cmap=cm)
        ops = {"L": s.Lmat, "R": s.Rmat}
    finally:
        TS._stag_duffy_points, TS._stag_map_nodes = o_r, o_n
    np.savez_compressed(C.HERE + f"/_ops_{mapname}_M{M}_{rule}_n{n}.npz",
                        **ops)
    C.dump(f"v2_rt_{mapname}_M{M}_{rule}_n{n}.json",
           {"map": mapname, "M": M, "rule": rule, "n": n,
            "vec": C.vec(o, R, T), "warnings": warns})
    print(mapname, M, rule, n, "done", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "moments":
        run_moments()
    else:
        run_rt(sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5]))
