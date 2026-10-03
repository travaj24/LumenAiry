"""V3 decisive check, limit (B1): the SHIPPED 2-D RCWA (FMM) on the E2-4
crossing device -- no curved code at all.  Layer 1: the disk through
RCWAStack shapes= (exact analytic form factor, Laurent).  Layer 2: the
sinusoidal wall as an eps_cell of Sx x Sy node samples, each the EXACT
x-area fraction of its pixel lying in (xs(y), P) (periodic; the straight
2.25 | air interface at x = 0 = P included), Laurent.  Ladder in n_orders.

  python v3m_rcwa.py <Sx> <Sy> <theta_deg> <phi_deg> <n1,n2,...>
Output: v3m_rcwa_S<Sx>x<Sy>_th<t>_ph<p>_<build>.json (rewritten per rung)
"""
import sys
import time

import numpy as np
import v3m_geom as G
from _ve import dump

from lumenairy.elements.rcwa.stack import RCWAStack

Sx, Sy = int(sys.argv[1]), int(sys.argv[2])
th_d, ph_d = float(sys.argv[3]), float(sys.argv[4])
ns = [int(v) for v in sys.argv[5].split(",")]
SUF = sys.argv[6] if len(sys.argv) > 6 else ""


def wall_cell(Sx, Sy):
    dx = G.P / Sx
    xj = np.arange(Sx) * dx
    yi = np.arange(Sy) * G.P / Sy
    w = G.xs(yi)                                   # (Sy,)
    lo = xj[:, None] - dx / 2
    hi = xj[:, None] + dx / 2

    def ov(a, b):                                  # overlap with [a, b]
        return np.clip(np.minimum(hi, b) - np.maximum(lo, a), 0.0, None)
    frac = (ov(w[None, :], G.P) + ov(w[None, :] - G.P, 0.0)
            + ov(w[None, :] + G.P, 2 * G.P)) / dx
    return (1.0 + (G.EPS_W - 1.0) * frac).astype(complex), frac


cell, frac = wall_cell(Sx, Sy)
out = {"Sx": Sx, "Sy": Sy, "theta_deg": th_d, "phi_deg": ph_d,
       "fill_area_err": float(frac.mean() * G.P ** 2 - 0.72), "runs": []}
for n in ns:
    t0 = time.perf_counter()
    st = RCWAStack(G.P, period_y=G.P, n_superstrate=G.N_SUP,
                   n_substrate=G.N_SUB, n_orders=n, n_orders_y=n)
    st.add_layer(G.D1, shapes=[{"shape": "disk", "eps": G.EPS_D,
                                "radius": G.R, "center": (G.CX, G.CY)}],
                 eps_background=1.0)
    st.add_layer(G.D2, eps_cell=cell)
    st.set_source(G.WL, theta=np.deg2rad(th_d), phi=np.deg2rad(ph_d))
    res = st.solve()
    o, R, T = res.efficiencies()
    o, R, T = np.asarray(o), np.asarray(R), np.asarray(T)
    keep = (np.abs(o[:, 0]) <= 3) & (np.abs(o[:, 1]) <= 3)
    row = {"n_orders": n, "orders": o[keep], "R": R[:, keep], "T": T[:, keep],
           "closure": np.abs(R.sum(1) + T.sum(1) - 1.0),
           "wall": time.perf_counter() - t0}
    out["runs"].append(row)
    print(n, row["closure"], f"{row['wall']:.1f}s", flush=True)
    dump(f"v3m_rcwa_S{Sx}x{Sy}{SUF}_th{th_d:g}_ph{ph_d:g}", out)
