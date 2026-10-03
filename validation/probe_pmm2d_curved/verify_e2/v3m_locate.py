"""Locate the order / input / port carrying the largest difference between two
v3m solves.  python v3m_locate.py <tag> <fileA> <fileB> [...pairs]"""
import json
import sys

import numpy as np

tag = sys.argv[1]
fs = sys.argv[2:]
res = {}
for a, b in zip(fs[0::2], fs[1::2]):
    A, B = json.load(open(a)), json.load(open(b))
    oa = [tuple(o) for o in A["orders"]]
    ob = [tuple(o) for o in B["orders"]]
    rows = []
    for port in ("R", "T"):
        Ra, Rb = np.asarray(A[port]), np.asarray(B[port])
        for k, o in enumerate(oa):
            j = ob.index(o)
            for pol in (0, 1):
                rows.append((abs(Ra[pol, k] - Rb[pol, j]), port, o, pol,
                             Ra[pol, k], Rb[pol, j]))
    rows.sort(reverse=True)
    key = f"{a} vs {b}"
    res[key] = [dict(d=r[0], port=r[1], order=r[2], input=r[3], a=r[4], b=r[5])
                for r in rows[:5]]
    print(key)
    for r in rows[:4]:
        print(f"   {r[0]:.3e} {r[1]} {r[2]} in{r[3]}  {r[4]:.6f} {r[5]:.6f}")
json.dump(res, open(f"v3m_locate_{tag}_win.json", "w"), indent=1, default=str)
