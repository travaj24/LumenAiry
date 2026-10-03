"""V5c -- convergence of the rotated ellipse's map: the uniform n = 2 film
under compile_shapes' map of Ellipse(0.6, 0.6, 0.40, 0.28, angle=20 deg)
(a case BOTH layouts lay out) vs Airy, M = 4..7, and the device's R00 / T00.
Run on HEAD (parametric corners) and on the scratch fix tree (LUM_TREE =
C:/tmp/vcc_fix, normal-45 corners, verify doc V-D2).
"""
import sys
import warnings

import numpy as np
from _vc import BUILD, ROOT, dump

from lumenairy.elements.pmm import Ellipse, PMM2DStackPure, compile_shapes

warnings.simplefilter("ignore")
P = 1.2
M = int(sys.argv[1])
e = Ellipse(0.6, 0.6, 0.40, 0.28, 4.0, angle=np.deg2rad(20.0))
cell, xw, yw, cm = compile_shapes(P, P, [e], 1.0)
k0 = 2 * np.pi
n = (1.0, 2.0, 1.45)
r01 = (n[0] - n[1]) / (n[0] + n[1])
r12 = (n[1] - n[2]) / (n[1] + n[2])
ph = np.exp(2j * n[1] * k0 * 0.5)
Rx = abs((r01 + r12 * ph) / (1 + r01 * r12 * ph)) ** 2
out = {"M": M, "tree": ROOT}
for kind in ("film", "device"):
    st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45,
                        n_modes=M, n_orders=3, cmap=cm)
    st.add_layer(0.5, eps_cell=np.full(cell.shape, 4.0 + 0j)
                 if kind == "film" else cell)
    st.set_source(1.0)
    o, R, T, J = st.solve()
    o, R, T = np.asarray(o), np.asarray(R), np.asarray(T)
    i0 = int(np.nonzero((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
    if kind == "film":
        R2, T2 = R.copy(), T.copy()
        R2[:, i0] -= Rx
        T2[:, i0] -= 1 - Rx
        out["film_err"] = float(max(np.abs(R2).max(), np.abs(T2).max()))
    else:
        out["R00"] = R[:, i0].tolist()
        out["T00"] = T[:, i0].tolist()
        out["closure"] = float(np.max(np.abs(R.sum(1) + T.sum(1) - 1)))
print(out, flush=True)
dump(f"v5c_ellipse_film_{'fix' if 'vcc_fix' in ROOT else 'head'}_M{M}_"
     f"{BUILD}.json", out)
