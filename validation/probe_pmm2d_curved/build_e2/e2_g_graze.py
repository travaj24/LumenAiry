"""E2-G: GRAZING cuts (the Phase E2 verifier's V-E2-D1) through the library
kernel against the verifier's independent physical brute force
(``verify_e2/v2_brute.py``): a straight x-wall of an unmapped layer grazing
the crest of a sinusoid wall map (graze depth delta = 0, 1e-7, 1e-6, 3e-5)
and a straight y-wall grazing the bottom of the circle map (delta = 1e-7,
1e-6, 1e-5), M = 4; and the circle / crossing-sinusoid pair, which the fix
must not move beyond round-off.  Output e2_g_graze_<tag>.json (arg 1: tag)."""
import os
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "verify_e2"))
from _common import dump  # noqa: E402
from v2_brute import (  # noqa: E402
    blocks_err,
    brute_cross,
    circle_map,
    grid,
    sin_map,
)

from lumenairy.elements.pmm import _curvemortar as CMM  # noqa: E402

tag = sys.argv[1] if len(sys.argv) > 1 else "post"
rows = []
cases = ([("sx", d) for d in (0.0, 1e-7, 1e-6, 3e-5)]
         + [("circ", d) for d in (1e-7, 1e-6, 1e-5)])
for kind, d in cases:
    if kind == "sx":
        ga = grid(sin_map(0.55, 0.12), 4)
        gb = grid(None, 4, walls=([0.0, 0.67 - d, 1.2], [0.0, 0.5, 1.2]))
    else:
        ga = grid(circle_map(), 4)
        gb = grid(None, 4, walls=([0.0, 0.45, 1.2], [0.0, 0.24 + d, 1.2]))
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        X, n, chg = CMM.curved_cross_mass_adaptive(ga, gb)
    tk = time.perf_counter() - t0
    Xb = brute_cross(ga, gb, n=20, verbose=False)[0]
    r = dict(case=kind, delta=d, err=blocks_err(X, Xb, ga.qq, gb.qq)["all"],
             n=n, change=chg, warned=len(wl) > 0, wall=tk)
    rows.append(r)
    print(r, flush=True)
ga, gb = grid(circle_map(), 4), grid(sin_map(), 4)
X18 = CMM.curved_cross_mass(ga, gb, 18)  # compared pre vs post, then replaced by the scalar
dump(f"e2_g_graze_{tag}.json", dict(rows=rows, circ_sin_n18=X18))
print("done")
