"""V9 readings: the two decision quantities the verifier's gap-closing tests
assert, read on ANY tree (the tip or a mutant copy, named by VA_ROOT):

  shear_film  max |R/T - Airy| of a uniform film under the verifier's
              SHEARED map (g12 != 0), normal incidence, M = 6 and 7;
  sep_oracle  max relative error of the mapped Rmat / Lmat against the
              independent separable-factor oracle (v2_operators.oracle_ops),
              asymmetric two-harmonic stretch, stripe, M = 4.

usage: VA_ROOT=<tree> PYTHONPATH=<tree> v9_readings.py <tag>
"""
import sys

import _vcommon as C
import numpy as np

from lumenairy.elements.pmm import twod_staggered as TS

TAG = sys.argv[1]
out = {"tag": TAG}
cm = C.make_shear_map(0.06, 0.05)
for M in (6, 7):
    o, R, T, J, _ = C.stack_solve(cm, [C.cell("film", n=2)], M)
    ex = C.airy()
    i0 = C.i00(o)
    Rr, Tr = R.copy(), T.copy()
    Rr[:, i0] -= ex["s"][0]
    Tr[:, i0] -= ex["s"][1]
    out[f"shear_film_M{M}"] = float(max(np.abs(Rr).max(), np.abs(Tr).max()))
import v2_operators as V2  # noqa: E402

f = C.HarmonicStretch(0.10, 0.04, 0.9)
cmap = C.stretch_map(f)
eps = C.cell("stripe")
bx = TS.Basis1D(C.P, cmap.u_walls, 4)
by = TS.Basis1D(C.P, cmap.v_walls, 4)
ref = V2.oracle_ops(bx, by, f, eps[:, 0], C.K0, 1000)
s = TS.Granet2DTransverseE(C.P, C.P, cmap.u_walls, cmap.v_walls, 4, eps,
                           k0=C.K0, cmap=cmap)
o = V2.ops(s)
out["sep_oracle_M4"] = max(V2.rel(o[k], ref[k]) for k in ("Rmat", "Lmat"))
print(out)
C.dump(f"v9_readings_{TAG}", out)
