"""V3 decisive check, limit (B2): a plan-view STAIRCASE of both outlines on
the SHIPPED unmapped pure 2-D PMM (no curved code): layer_grids='per-layer'
with each layer on its OWN straight walls, coupled by the shipped separable
mortar.  Layer 1: the disk as n EQUAL-AREA horizontal strips (uniform in y
over [c - r, c + r]; strip half-width = exact strip area / (2 dy)).  Layer 2:
the wall as n strips uniform in y over [0, P], each wall x = the strip MEAN
of xs(y) (equal area).  Wall lists are padded with dummy walls (largest gap
halved) to square grids (Nx == Ny).

  python v3m_stair.py <n> <M1,M2,...> [theta_deg phi_deg]
Output: v3m_stair_n<n>_th<t>_ph<p>_<build>.json (rewritten per rung)
"""
import sys
import time

import numpy as np
import v3m_geom as G
from _ve import closure, dump, solve

from lumenairy.elements.pmm import PMM2DStackPure

n = int(sys.argv[1])
Ms = [int(v) for v in sys.argv[2].split(",")]
th_d = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
ph_d = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
P, c, r = G.P, G.CX, G.R


def seg_area(y0, y1):
    """area of the disk between y0 and y1 (exact)."""
    def F(y):
        t = np.clip((y - c) / r, -1, 1)
        return r * r * (t * np.sqrt(1 - t * t) + np.arcsin(t))
    return F(y1) - F(y0)


def pad(xw, yw):
    xw, yw = sorted(set(xw)), sorted(set(yw))
    while len(xw) != len(yw):
        a = xw if len(xw) < len(yw) else yw
        full = [0.0] + a + [P]
        k = int(np.argmax(np.diff(full)))
        a.append(0.5 * (full[k] + full[k + 1]))
        a.sort()
    return np.array(xw), np.array(yw)


def disk_layer(n):
    ys = c - r + 2 * r * np.arange(n + 1) / n
    hw = [seg_area(ys[k], ys[k + 1]) / (2 * (ys[k + 1] - ys[k]))
          for k in range(n)]
    hw = [round(h, 12) for h in hw]
    xw = [c - h for h in hw] + [c + h for h in hw]
    xw, yw = pad(xw, list(ys))
    X = np.r_[0.0, xw, P]
    Y = np.r_[0.0, yw, P]
    eps = np.ones((len(X) - 1, len(Y) - 1), complex)
    area = 0.0
    for i in range(len(X) - 1):
        xm = 0.5 * (X[i] + X[i + 1])
        for j in range(len(Y) - 1):
            ym = 0.5 * (Y[j] + Y[j + 1])
            k = np.searchsorted(ys, ym) - 1
            if 0 <= k < n and abs(xm - c) < hw[k]:
                eps[i, j] = G.EPS_D
                area += (X[i + 1] - X[i]) * (Y[j + 1] - Y[j])
    return xw, yw, eps, area


def wall_layer(n):
    ys = P * np.arange(n + 1) / n
    kk = 2 * np.pi / P
    xm = [G.X0 - G.AMP * (np.cos(kk * ys[k + 1]) - np.cos(kk * ys[k]))
          / (kk * (ys[k + 1] - ys[k])) for k in range(n)]
    xm = [round(v, 12) for v in xm]
    xw, yw = pad(xm, list(ys[1:-1]))
    X = np.r_[0.0, xw, P]
    Y = np.r_[0.0, yw, P]
    eps = np.ones((len(X) - 1, len(Y) - 1), complex)
    area = 0.0
    for i in range(len(X) - 1):
        xc = 0.5 * (X[i] + X[i + 1])
        for j in range(len(Y) - 1):
            ym = 0.5 * (Y[j] + Y[j + 1])
            k = min(int(ym / (P / n)), n - 1)
            if xc > xm[k]:
                eps[i, j] = G.EPS_W
                area += (X[i + 1] - X[i]) * (Y[j + 1] - Y[j])
    return xw, yw, eps, area


x1, y1, e1, a1 = disk_layer(n)
x2, y2, e2, a2 = wall_layer(n)
out = {"n": n, "theta_deg": th_d, "phi_deg": ph_d,
       "disk_grid": e1.shape, "wall_grid": e2.shape,
       "disk_area_err": a1 - np.pi * r * r, "wall_area_err": a2 - 0.72,
       "runs": []}
print(out, flush=True)
for M in Ms:
    t0 = time.perf_counter()
    st = PMM2DStackPure(P, P, n_superstrate=G.N_SUP, n_substrate=G.N_SUB,
                        n_modes=M, n_orders=3, layer_grids="per-layer")
    st.add_layer(G.D1, eps_cell=e1, x_walls=x1, y_walls=y1,
                 max_pencil_dof=20000)
    st.add_layer(G.D2, eps_cell=e2, x_walls=x2, y_walls=y2,
                 max_pencil_dof=20000)
    o, R, T, J = solve(st, G.WL, np.deg2rad(th_d), np.deg2rad(ph_d))
    row = {"M": M, "orders": o, "R": R, "T": T, "closure": closure((o, R, T, J)),
           "pencils": [2 * (e1.shape[0] * (M - 1)) ** 2,
                       2 * (e2.shape[0] * (M - 1)) ** 2],
           "wall": time.perf_counter() - t0}
    out["runs"].append(row)
    print(n, M, row["closure"], row["pencils"], f"{row['wall']:.1f}s",
          flush=True)
    dump(f"v3m_stair_n{n}_th{th_d:g}_ph{ph_d:g}", out)
