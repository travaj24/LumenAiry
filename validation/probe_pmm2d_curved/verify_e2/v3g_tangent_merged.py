"""Near-tangent (gap 1e-4) sinusoid beside a circle: the merged map (fast
path) vs the same stack forced through per-layer maps (curved mortar).
Usage: v3g_tangent_merged.py M"""
import sys

import numpy as np
from _ve import dump, solve
from v3g_fix import NREC, WL, stack
from v3g_geoms import awkward

M = int(sys.argv[1])
s1, s2 = awkward()["tangent_out"]
lay = [(0.25, s1, 1.0), (0.2, s2, 1.0)]
st = stack(lay, M)
assert st._perlayer_fast_ok()
o, R, T, J = solve(st, WL)
stf = stack(lay, M)
stf._e2_per_layer_maps = True
k0 = len(NREC)
o2, R2, T2, J2 = solve(stf, WL)
out = dict(M=M, forced_vs_merged=float(max(np.abs(R - R2).max(),
                                           np.abs(T - T2).max())),
           closure_merged=np.abs(R.sum(1) + T.sum(1) - 1),
           closure_forced=np.abs(R2.sum(1) + T2.sum(1) - 1),
           n=[r["n"] for r in NREC[k0:]], R=R, T=T, R_f=R2, T_f=T2)
print({k: v for k, v in out.items() if not k.startswith(("R", "T"))})
dump(f"v3g_tangent_merged_M{M}", out)
