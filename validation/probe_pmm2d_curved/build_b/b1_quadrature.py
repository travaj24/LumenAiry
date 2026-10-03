"""B-Q -- THE FIRST DESIGN DECISION: the quadrature of the four singular
vertices (det J = 0) of a closed smooth curve on a tensor grid.

Phase A's adaptive rule (``_stag_map_nodes``: double nq from 2M + 8 until the
Legendre moments of the five geometric weights agree to 1e-13) is predicted
by Phase A finding F1 to run to its 256-node cap on the circle, because the
weights carry 1 / distance at the singular vertices.  This probe measures:

  demand M        -- per CELL, the moment error of the Phase-A criterion at
                     n = 2M+8, 2(2M+8), ... up to 4096 nodes per axis: what nq
                     the criterion demands, and how fast the error falls
  ladder K M      -- R / T of the full library solve (map K = c3 | c5) against
                     the per-axis node count FORCED to nq = 16 .. 512 (plain
                     tensor Gauss-Legendre, the Phase-A rule family); the
                     floor and the rate
  rules M         -- the singular cells' moment error per node budget for the
                     candidate rules: (a) plain tensor Gauss, (b1) a
                     geometrically graded tensor rule, (b2) a Duffy-collapsed
                     rule (the cell split at its centre into quadrants when it
                     owns several singular corners, each (sub)square owning a
                     singular corner cut into two triangles collapsed onto
                     that corner) -- reference: the Duffy rule at 200 x 200
                     nodes per triangle (its self-gap against 160 recorded).
Output: b1_quadrature_<mode>[_K]_M<M>.json
"""
import json
import sys
import time

import _common as C
import numpy as np
from numpy.polynomial.legendre import leggauss, legvander


def geom_pts(cmap, sx, sy, U, V):
    """``geom`` on a POINT list (U, V) of equal length, evaluated through the
    tensor protocol one row of equal U at a time."""
    out = [np.empty(U.size) for _ in range(6)]
    uniq, inv = np.unique(U, return_inverse=True)
    for k, u in enumerate(uniq):
        sel = np.nonzero(inv == k)[0]
        g = cmap.geom(sx, sy, np.array([u]), V[sel])
        for a, b in zip(out, g):
            a[sel] = b[0, :]
    return out


def weights5(xu, xv, yu, yv):
    sg = xu * yv - xv * yu
    return [sg, 1.0 / sg, (xu * xu + yu * yu) / sg,
            (xu * xv + yu * yv) / sg, (xv * xv + yv * yv) / sg]


def moments_tensor(cmap, sx, sy, n, deg):
    xg, wg = leggauss(n)
    Pv = legvander(xg, deg) * wg[:, None]
    u0, u1 = cmap.u_bounds[sx], cmap.u_bounds[sx + 1]
    v0, v1 = cmap.v_bounds[sy], cmap.v_bounds[sy + 1]
    U = 0.5 * (u0 + u1) + 0.5 * (u1 - u0) * xg
    V = 0.5 * (v0 + v1) + 0.5 * (v1 - v0) * xg
    _X, _Y, xu, xv, yu, yv = cmap.geom(sx, sy, U, V)
    return np.array([Pv.T @ f @ Pv for f in weights5(xu, xv, yu, yv)])


def moments_points(cmap, sx, sy, s, t, w, deg):
    """Moments sum_q w_q f(s_q, t_q) P_i(s_q) P_j(t_q) on reference
    [-1, 1]^2 points (s, t) with weights w (summing to 4)."""
    u0, u1 = cmap.u_bounds[sx], cmap.u_bounds[sx + 1]
    v0, v1 = cmap.v_bounds[sy], cmap.v_bounds[sy + 1]
    U = 0.5 * (u0 + u1) + 0.5 * (u1 - u0) * s
    V = 0.5 * (v0 + v1) + 0.5 * (v1 - v0) * t
    xu, xv, yu, yv = geom_pts(cmap, sx, sy, U, V)[2:]
    Ps = legvander(s, deg)
    Pt = legvander(t, deg)
    return np.array([np.einsum("q,qi,qj->ij", w * f, Ps, Pt)
                     for f in weights5(xu, xv, yu, yv)])


def rel_err(a, b):
    out = 0.0
    for x, y in zip(a, b):
        s = float(np.max(np.abs(y)))
        d = float(np.max(np.abs(x - y)))
        if s == 0.0:
            if d != 0.0:
                return np.inf
            continue
        out = max(out, d / s)
    return out


# ------------------------------------------------------------ candidate rules
def graded_1d(p, L, sigma, ends):
    """Composite Gauss-Legendre on [-1, 1], p nodes per piece, geometric
    breakpoints with ratio sigma and L layers toward each end in ``ends``
    (subset of {-1, +1})."""
    bks = [-1.0, 1.0]
    for e in ends:
        for k in range(1, L + 1):
            bks.append(e - np.sign(e) * 2.0 * sigma ** k)
    bks = np.unique(np.array(bks))
    xg, wg = leggauss(p)
    xs, ws = [], []
    for a, b in zip(bks[:-1], bks[1:]):
        xs.append(0.5 * (a + b) + 0.5 * (b - a) * xg)
        ws.append(0.5 * (b - a) * wg)
    return np.concatenate(xs), np.concatenate(ws)


def duffy_cell(corners, n):
    """Reference-square [-1, 1]^2 points of the Duffy rule for a cell whose
    singular corners are ``corners`` (list of (cs, ct) in {-1, +1}^2): split
    into the four quadrants when more than one corner is singular; every
    (sub)square owning a singular corner is cut along its diagonal from that
    corner into two triangles, each mapped from the unit square with the
    edge xi = 0 collapsed onto the corner (Jacobian ~ xi cancels 1/r); other
    (sub)squares get the n x n tensor rule."""
    xg, wg = leggauss(n)
    a = 0.5 * (xg + 1.0)
    wa = 0.5 * wg
    XI, ETA = np.meshgrid(a, a, indexing="ij")
    WXE = np.outer(wa, wa)
    if len(corners) > 1:
        subs = []
        for qs in (-1, 1):
            for qt in (-1, 1):
                lo = (min(0, qs), min(0, qt))
                hi = (max(0, qs), max(0, qt))
                own = [cc for cc in corners if tuple(cc) == (qs, qt)]
                subs.append((lo, hi, own))
    else:
        subs = [((-1, -1), (1, 1), list(corners))]
    S, T, W = [], [], []
    for (lo, hi, own) in subs:
        lo = np.array(lo, float)
        hi = np.array(hi, float)
        if not own:
            xs = 0.5 * (lo[0] + hi[0]) + 0.5 * (hi[0] - lo[0]) * xg
            ts = 0.5 * (lo[1] + hi[1]) + 0.5 * (hi[1] - lo[1]) * xg
            S.append(np.repeat(xs, n))
            T.append(np.tile(ts, n))
            W.append(np.outer(wg, wg).ravel() * 0.25 * (hi[0] - lo[0])
                     * (hi[1] - lo[1]))
            continue
        cs, ct = own[0]
        c = np.array([lo[0] if cs < 0 else hi[0], lo[1] if ct < 0 else hi[1]])
        o = np.array([hi[0] if cs < 0 else lo[0], hi[1] if ct < 0 else lo[1]])
        for A in (np.array([o[0], c[1]]), np.array([c[0], o[1]])):
            # P(xi, eta) = c + xi [(A - c) + eta (o - A)]
            e1 = A - c
            e2 = o - A
            jac = abs(e1[0] * e2[1] - e1[1] * e2[0])
            Pp = (c[None, None, :] + XI[..., None]
                  * (e1[None, None, :] + ETA[..., None] * e2[None, None, :]))
            S.append(Pp[..., 0].ravel())
            T.append(Pp[..., 1].ravel())
            W.append((WXE * XI * jac).ravel())
    return np.concatenate(S), np.concatenate(T), np.concatenate(W)


def singular_cells(cm):
    byc = {}
    for sx, sy, cu, cv in cm.singular_vertices:
        byc.setdefault((sx, sy), []).append((2 * cu - 1, 2 * cv - 1))
    return byc


def run_demand(M):
    res = {"env": C.env_record(), "M": M, "tol": 1e-13, "maps": {}}
    deg = 2 * M - 2
    for kind in ("c3", "c5"):
        cm = (C.circle3() if kind == "c3" else C.circle5())[0]
        Nx, Ny = cm.shape
        sing = set(singular_cells(cm))
        cells = {}
        for sx in range(Nx):
            for sy in range(Ny):
                ns = []
                n = 2 * M + 8
                prev = moments_tensor(cm, sx, sy, n, deg)
                demand = None
                while n <= 2048:
                    cur = moments_tensor(cm, sx, sy, 2 * n, deg)
                    e = rel_err(prev, cur)
                    ns.append([n, e])
                    if e <= 1e-13:
                        demand = n
                        break
                    n, prev = 2 * n, cur
                cells[f"{sx},{sy}"] = {"singular": (sx, sy) in sing,
                                       "demand": demand, "ladder": ns}
                print(kind, sx, sy, (sx, sy) in sing, demand,
                      [f"{a}:{b:.1e}" for a, b in ns], flush=True)
        res["maps"][kind] = cells
        C.dump(f"b1_quadrature_demand_M{M}.json", res)


def run_ladder(kind, M, nqs):
    cm, eps = C.circle3() if kind == "c3" else C.circle5()
    res = {"env": C.env_record(), "map": kind, "M": M, "rule": "plain tensor"
           " Gauss-Legendre, nq per axis per cell forced", "runs": []}
    for nq in nqs:
        t0 = time.perf_counter()
        with C.fixed_nodes(nq):
            o, R, T, _J = C.solve(cm, eps, M)
        v = C.vec(o, R, T)
        clo = float(np.max(np.abs(R.sum(1) + T.sum(1) - 1)))
        res["runs"].append({"nq": nq, "vec": v.tolist(), "closure": clo,
                            "t": time.perf_counter() - t0})
        print(f"{kind} M={M} nq={nq} R00={v[0]:.12f} clo={clo:.2e} "
              f"t={time.perf_counter() - t0:.1f}", flush=True)
        C.dump(f"b1_quadrature_ladder_{kind}_M{M}.json", res)
    ref = np.array(res["runs"][-1]["vec"])
    for r in res["runs"]:
        r["dist_to_top"] = float(np.max(np.abs(np.array(r["vec"]) - ref)))
    C.dump(f"b1_quadrature_ladder_{kind}_M{M}.json", res)
    for r in res["runs"]:
        print(r["nq"], f"{r['dist_to_top']:.2e}")


def run_rules(M):
    """Operator-level: the moment error of each candidate rule in the
    SINGULAR cells against the Duffy rule at 200 nodes per direction per
    triangle (the reference; its own self-gap vs 160 is recorded)."""
    deg = 2 * M - 2
    res = {"env": C.env_record(), "M": M, "deg": deg, "maps": {}}
    for kind in ("c3", "c5"):
        cm = (C.circle3() if kind == "c3" else C.circle5())[0]
        out = {}
        for (sx, sy), corners in singular_cells(cm).items():
            ref = moments_points(cm, sx, sy, *duffy_cell(corners, 200), deg)
            ref2 = moments_points(cm, sx, sy, *duffy_cell(corners, 160), deg)
            row = {"corners": corners, "ref_selfgap": rel_err(ref2, ref),
                   "plain": [], "graded": [], "duffy": []}
            for n in (16, 24, 32, 48, 64, 96, 128, 256, 512):
                m = moments_tensor(cm, sx, sy, n, deg)
                row["plain"].append([n, n * n, rel_err(m, ref)])
            ends = sorted({c[0] for c in corners})
            endt = sorted({c[1] for c in corners})
            for p, L, sig in ((8, 4, 0.15), (8, 8, 0.15), (12, 8, 0.15),
                              (12, 12, 0.15), (16, 12, 0.15), (16, 16, 0.15),
                              (M + 4, 10, 0.2), (2 * M, 14, 0.15)):
                xs, ws = graded_1d(p, L, sig, ends)
                xt, wt = graded_1d(p, L, sig, endt)
                S = np.repeat(xs, xt.size)
                T = np.tile(xt, xs.size)
                W = np.outer(ws, wt).ravel()
                m = moments_points(cm, sx, sy, S, T, W, deg)
                row["graded"].append([p, L, sig, xs.size, S.size,
                                      rel_err(m, ref)])
            for n in (8, 12, 16, 20, 24, 32, 48, 64):
                S, T, W = duffy_cell(corners, n)
                m = moments_points(cm, sx, sy, S, T, W, deg)
                row["duffy"].append([n, S.size, rel_err(m, ref)])
            out[f"{sx},{sy}"] = row
            print(kind, (sx, sy), "selfgap", f"{row['ref_selfgap']:.1e}",
                  flush=True)
            for k in ("plain", "graded", "duffy"):
                print("  ", k, [(r[-2], f"{r[-1]:.1e}") for r in row[k]],
                      flush=True)
        res["maps"][kind] = out
        C.dump(f"b1_quadrature_rules_M{M}.json", res)


def run_duffy(kind, M, ns):
    """The ADOPTED rule's own ladder: the full library solve (Duffy corner
    rule in the singular cells, tensor rule elsewhere) with the node count
    forced to n, plus the operator distance of the PLAIN rule at the same n
    to the Duffy operator at the top n (what the plain rule gets wrong)."""
    cm, eps = C.circle3() if kind == "c3" else C.circle5()
    k0 = 2 * np.pi / C.WL
    res = {"env": C.env_record(), "map": kind, "M": M,
           "rule": "Duffy corner rule in the singular cells (n per direction"
           " per piece), tensor Gauss n elsewhere", "runs": []}
    ops = {}
    for n in ns:
        t0 = time.perf_counter()
        with C.fixed_nodes(n):
            o, R, T, _J = C.solve(cm, eps, M)
            sol = C.TS.Granet2DTransverseE(C.P, C.P, cm.u_walls, cm.v_walls,
                                           M, eps, k0=k0, cmap=cm)
            orig = C.TS._stag_map_singular_corners
            C.TS._stag_map_singular_corners = lambda c: {}
            try:
                solp = C.TS.Granet2DTransverseE(C.P, C.P, cm.u_walls,
                                                cm.v_walls, M, eps, k0=k0,
                                                cmap=cm)
            finally:
                C.TS._stag_map_singular_corners = orig
        ops[n] = (sol.Lmat, sol.Rmat, solp.Lmat, solp.Rmat)
        v = C.vec(o, R, T)
        clo = float(np.max(np.abs(R.sum(1) + T.sum(1) - 1)))
        res["runs"].append({"n": n, "vec": v.tolist(), "closure": clo,
                            "t": time.perf_counter() - t0})
        print(f"duffy {kind} M={M} n={n} R00={v[0]:.14f} clo={clo:.2e}",
              flush=True)
    top = ns[-1]
    ref = np.array(res["runs"][-1]["vec"])
    Lr, Rr = ops[top][0], ops[top][1]
    sc = max(float(np.max(np.abs(Lr))), float(np.max(np.abs(Rr))))
    for r in res["runs"]:
        n = r["n"]
        r["dist_to_top"] = float(np.max(np.abs(np.array(r["vec"]) - ref)))
        r["op_duffy_to_top"] = max(float(np.max(np.abs(ops[n][0] - Lr))),
                                   float(np.max(np.abs(ops[n][1] - Rr)))) / sc
        r["op_plain_to_duffy_top"] = max(
            float(np.max(np.abs(ops[n][2] - Lr))),
            float(np.max(np.abs(ops[n][3] - Rr)))) / sc
        print(n, f"RT {r['dist_to_top']:.2e} opD {r['op_duffy_to_top']:.2e} "
              f"opPlain {r['op_plain_to_duffy_top']:.2e}", flush=True)
    C.dump(f"b1_quadrature_duffy_{kind}_M{M}.json", res)


def run_reparam(M):
    """Option (c): re-parametrise the singular cell's local coordinates per
    axis, s = q(s'), t = q(t'), and measure the plain tensor rule's moment
    self-gap (n vs 2n) of the five weights of the re-parametrised map
    (J' = J diag(q'(s'), q'(t'))):
      cubic  q = (3 s' - s'^3) / 2      (q' = 0 at the ends: degenerate)
      asin   q = (2 / pi) arcsin(s')    (q' ~ 1 / sqrt at the ends)
      none   q = s'                     (the map as built)."""
    deg = 2 * M - 2
    cm = C.circle3()[0]
    sx = sy = 1
    u0, u1 = cm.u_bounds[1], cm.u_bounds[2]
    qs = {"none": (lambda z: z, lambda z: np.ones_like(z)),
          "cubic": (lambda z: 0.5 * (3 * z - z ** 3),
                    lambda z: 1.5 * (1 - z ** 2)),
          "asin": (lambda z: (2 / np.pi) * np.arcsin(z),
                   lambda z: (2 / np.pi) / np.sqrt(1 - z ** 2))}
    res = {"env": C.env_record(), "M": M, "cell": [1, 1], "rows": {}}
    for name, (q, dq) in qs.items():
        prev = None
        rows = []
        for n in (20, 40, 80, 160, 320, 640, 1280):
            xg, wg = leggauss(n)
            sp = q(xg)
            U = 0.5 * (u0 + u1) + 0.5 * (u1 - u0) * sp
            _X, _Y, xu, xv, yu, yv = cm.geom(sx, sy, U, U)
            a = dq(xg)
            xu, yu = xu * a[:, None], yu * a[:, None]
            xv, yv = xv * a[None, :], yv * a[None, :]
            Pv = legvander(xg, deg) * wg[:, None]
            m = np.array([Pv.T @ f @ Pv for f in weights5(xu, xv, yu, yv)])
            if prev is not None:
                rows.append([n // 2, rel_err(prev, m)])
            prev = m
        res["rows"][name] = rows
        print(name, [(a, f"{b:.1e}") for a, b in rows], flush=True)
    C.dump(f"b1_quadrature_reparam_M{M}.json", res)


def run_summary():
    """The decision table: per map / M, the plain-rule ladder against its
    own top rung and against the Duffy limit, the planner's 24-vs-96 pair,
    the Duffy ladder, the operator distances, and the round-off floor."""
    out = {"env": C.env_record(), "rows": {}}
    for kind, M in (("c3", 6), ("c3", 8), ("c5", 5)):
        try:
            pl = C.load(f"b1_quadrature_ladder_{kind}_M{M}.json")
            du = C.load(f"b1_quadrature_duffy_{kind}_M{M}.json")
        except FileNotFoundError:
            continue
        pv = {r["nq"]: np.array(r["vec"]) for r in pl["runs"]}
        dv = {r["n"]: np.array(r["vec"]) for r in du["runs"]}
        lim = dv[max(dv)]
        row = {"plain_vs_duffy_limit": {n: float(np.max(np.abs(v - lim)))
                                        for n, v in pv.items()},
               "plain_vs_own_top": {r["nq"]: r["dist_to_top"]
                                    for r in pl["runs"]},
               "duffy_vs_own_top": {r["n"]: r["dist_to_top"]
                                    for r in du["runs"]},
               "duffy_op_to_top": {r["n"]: r["op_duffy_to_top"]
                                   for r in du["runs"]},
               "plain_op_to_duffy_top": {r["n"]: r["op_plain_to_duffy_top"]
                                         for r in du["runs"]}}
        if 24 in pv and 96 in pv:
            row["planner_pair_24_vs_96"] = float(np.max(np.abs(pv[24]
                                                                - pv[96])))
        out["rows"][f"{kind}_M{M}"] = row
        print(kind, M, json.dumps(row, indent=0)[:1500], flush=True)
    C.dump("b1_quadrature_summary.json", out)


def run_cost():
    """Operator ASSEMBLY time (best of three) with the corner rule and with
    the plain tensor rule at the same node count, plus the chosen count and
    the corner-cell point count.  Wall times on a shared box: upper bounds;
    the RATIO is the quantity."""
    k0 = 2 * np.pi / C.WL
    out = {"env": C.env_record(), "rows": []}
    for kind, M in (("c3", 6), ("c3", 8), ("c5", 5)):
        cm, eps = C.circle3() if kind == "c3" else C.circle5()
        row = {"map": kind, "M": M}
        for arm in ("corner", "plain"):
            ts = []
            for _ in range(3):
                t0 = time.perf_counter()
                if arm == "plain":
                    with C.plain_rule(), C.fixed_nodes(2 * M + 8):
                        s = C.TS.Granet2DTransverseE(
                            C.P, C.P, cm.u_walls, cm.v_walls, M, eps, k0=k0,
                            cmap=cm)
                else:
                    s = C.TS.Granet2DTransverseE(
                        C.P, C.P, cm.u_walls, cm.v_walls, M, eps, k0=k0,
                        cmap=cm)
                ts.append(time.perf_counter() - t0)
            row[f"t_{arm}"] = min(ts)
            if arm == "corner":
                row["nq"] = s._qrule.n
                row["corner_points"] = {f"{a},{b}": int(v[2].size) for (a, b),
                                        v in s._qrule.points.items()}
        row["ratio"] = row["t_corner"] / row["t_plain"]
        out["rows"].append(row)
        print(row, flush=True)
    C.dump("b1_quadrature_cost.json", out)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "demand":
        run_demand(int(sys.argv[2]))
    elif mode == "ladder":
        nqs = ([int(v) for v in sys.argv[4].split(",")] if len(sys.argv) > 4
               else [16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512])
        run_ladder(sys.argv[2], int(sys.argv[3]), nqs)
    elif mode == "rules":
        run_rules(int(sys.argv[2]))
    elif mode == "duffy":
        ns = ([int(v) for v in sys.argv[4].split(",")] if len(sys.argv) > 4
              else [8, 12, 16, 20, 24, 32, 48])
        run_duffy(sys.argv[2], int(sys.argv[3]), ns)
    elif mode == "reparam":
        run_reparam(int(sys.argv[2]))
    elif mode == "summary":
        run_summary()
    elif mode == "cost":
        run_cost()
    else:
        raise SystemExit(mode)
