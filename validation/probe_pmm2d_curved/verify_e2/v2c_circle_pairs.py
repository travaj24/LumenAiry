"""V2c -- how often does the curved mortar REFUSE two crossing circles
(Phase E2 verifier, item 2/7)?  ``_assign_pairs`` refuses every cell pair
that touches a singular vertex (a 45-degree point) of BOTH maps.  Sweep: a
circle r_a at the cell centre (P 1.2) over a circle r_b whose centre is
offset by (dx, dy); classify (a) the stack-wide merge (crosses or not), (b)
the kernel: refused / solved, and for refused pairs the distance between the
two singular vertices involved.
"""
import itertools

import numpy as np
from _ve import dump
from v2_brute import circle_map, grid

from lumenairy.elements.pmm import _curvemortar as CMOR

P = 1.2
rows = []
for ra, rb, dx, dy in itertools.product(
        (0.36, 0.30), (0.34, 0.26, 0.18), (0.05, 0.12, 0.2, 0.3),
        (0.0, 0.07, 0.15)):
    ca = (0.6, 0.6)
    cb = (0.6 + dx, 0.6 + dy)
    if cb[0] + rb > P - 0.02 or cb[1] + rb > P - 0.02:
        continue
    d = np.hypot(dx, dy)
    crossing = abs(ra - rb) < d < ra + rb
    if not crossing:
        continue
    ga = grid(circle_map(ra, ca), 4)
    gb = grid(circle_map(rb, cb), 4)
    A = CMOR._MapView(ga.cmap, ga.bx.xb, ga.by.xb, P, P)
    B = CMOR._MapView(gb.cmap, gb.bx.xb, gb.by.xb, P, P)
    try:
        CMOR._assign_pairs(A, B)
        st = "solved"
    except NotImplementedError:
        st = "REFUSED"
    # min distance between a singular vertex of a and one of b
    sa = [p for v in A.sing.values() for p in v]
    sb = [p for v in B.sing.values() for p in v]
    dmin = min(np.hypot(p[0] - q[0], p[1] - q[1]) for p in sa for q in sb)
    rows.append(dict(ra=ra, rb=rb, dx=dx, dy=dy, status=st,
                     min_sing_sep=float(dmin)))
    print(ra, rb, dx, dy, st, f"{dmin:.3f}")
nref = sum(r["status"] == "REFUSED" for r in rows)
print(f"crossing pairs {len(rows)}, refused {nref}")
dump("v2c_circle_pairs", dict(rows=rows, n=len(rows), refused=nref))
