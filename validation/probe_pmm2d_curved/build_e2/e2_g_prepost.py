"""E2-G3: the kernel with and without the V-E2-D1 grazing refinement on the
circle graze (the wall y = 0.24 + delta just inside the circle's bottom,
M = 4) -- the size of what the fix adds there (the sliver's own share), to
set against the verifier's brute force, whose own n = 20 vs 28 change at
this graze is 4.2e-7 (e2_g_oracle notes)."""
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "verify_e2"))
from _common import dump  # noqa: E402
from v2_brute import circle_map, grid, sin_map  # noqa: E402

from lumenairy.elements.pmm import _curvemortar as CMM  # noqa: E402

rows = []
orig = CMM._grazing_refine
for kind, d in (("circ", 1e-7), ("circ", 1e-6), ("circ", 1e-5),
                ("sx", 1e-6)):
    if kind == "sx":
        ga = grid(sin_map(0.55, 0.12), 4)
        gb = grid(None, 4, walls=([0.0, 0.67 - d, 1.2], [0.0, 0.5, 1.2]))
    else:
        ga = grid(circle_map(), 4)
        gb = grid(None, 4, walls=([0.0, 0.45, 1.2], [0.0, 0.24 + d, 1.2]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Xf = CMM.curved_cross_mass_adaptive(ga, gb)[0]
        CMM._grazing_refine = lambda Pm, sx, sy, Om, e, tk: tk
        try:
            Xo = CMM.curved_cross_mass_adaptive(ga, gb)[0]
        finally:
            CMM._grazing_refine = orig
    r = dict(case=kind, delta=d,
             fix_moves=float(np.abs(Xf - Xo).max() / np.abs(Xf).max()))
    rows.append(r)
    print(r, flush=True)
dump("e2_g_prepost.json", dict(rows=rows))
