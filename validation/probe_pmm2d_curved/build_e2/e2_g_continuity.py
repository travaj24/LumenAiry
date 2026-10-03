"""E2-G4: the kernel's cross-mass is continuous ACROSS the circle graze --
the unmapped wall y = 0.24 + delta moved through the circle's bottom
(delta = -1e-3 .. +1e-3, M = 4): the relative change against delta = 0 must
be linear in delta (the wall position enters smoothly) with no jump at 0
(the sliver contributes ~ delta^1.5)."""
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "verify_e2"))
from _common import dump  # noqa: E402
from v2_brute import circle_map, grid  # noqa: E402

from lumenairy.elements.pmm import _curvemortar as CMM  # noqa: E402

res = {}
for d in (-1e-3, -1e-5, -1e-7, 0.0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3):
    ga = grid(circle_map(), 4)
    gb = grid(None, 4, walls=([0.0, 0.45, 1.2], [0.0, 0.24 + d, 1.2]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res[d] = CMM.curved_cross_mass_adaptive(ga, gb)[0]
ref = res[0.0]
sc = float(np.abs(ref).max())
rows = [dict(delta=d, rel_change=float(np.abs(X - ref).max() / sc),
             per_unit=(float(np.abs(X - ref).max() / sc / abs(d)) if d else
                       None)) for d, X in res.items()]
for r in rows:
    print(r)
dump("e2_g_continuity.json", dict(rows=rows))
