"""Calibration of limit (B1): the RCWA ladder on the LONE disk layer (layer 1
only), to be compared with the lone disk on the shipped 3x3 circle map at
M = 10 (v3m_lone_b_M10) -- how close does RCWA + Richardson get on the disk
alone?   python v3m_rcwa_disk.py <n1,n2,...>"""
import sys
import time

import numpy as np
import v3m_geom as G
from _ve import dump

from lumenairy.elements.rcwa.stack import RCWAStack

ns = [int(v) for v in sys.argv[1].split(",")]
out = {"runs": []}
for n in ns:
    t0 = time.perf_counter()
    st = RCWAStack(G.P, period_y=G.P, n_superstrate=G.N_SUP,
                   n_substrate=G.N_SUB, n_orders=n, n_orders_y=n)
    st.add_layer(G.D1, shapes=[{"shape": "disk", "eps": G.EPS_D,
                                "radius": G.R, "center": (G.CX, G.CY)}],
                 eps_background=1.0)
    st.set_source(G.WL)
    o, R, T = st.solve().efficiencies()
    o, R, T = np.asarray(o), np.asarray(R), np.asarray(T)
    keep = (np.abs(o[:, 0]) <= 3) & (np.abs(o[:, 1]) <= 3)
    out["runs"].append({"n_orders": n, "orders": o[keep], "R": R[:, keep],
                        "T": T[:, keep], "wall": time.perf_counter() - t0})
    print(n, f"{time.perf_counter() - t0:.1f}s", flush=True)
    dump("v3m_rcwa_disk", out)
