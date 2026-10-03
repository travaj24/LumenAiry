"""R1c: how close does an EXACT cluster (gap <= 1e-12 max|lam|) of the
V-E3-1 references sit to the consumer's branch points (relative to
max|lam|)?  Layer pencil: q = sqrt(g2), branch point g2 = 0.  Geometric
pencil: g2 = g2_geo + eps of each homogeneous region, branch points
g2_geo = -eps_sup, -eps_sub.

    python r1c_branch.py M
"""
import sys

import numpy as np
from _r2 import P, PMM2DStackPure, dump

from lumenairy.elements.pmm import Circle, FilletRect, Rect

M = int(sys.argv[1])
CASES = {"square": [Rect(0.6, 0.6, 0.5, 0.5, 3.5)],
         "circle": [Circle(0.6, 0.6, 0.33, 3.5)],
         "fillet_sq": [FilletRect(0.6, 0.6, 0.6, 0.6, 0.1, 3.5)]}
res = {"M": M}
for name, shp in CASES.items():
    st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45, n_modes=M,
                        n_orders=2, backend="jax")
    st.add_layer(0.45, shapes=shp, background_eps=1.0)
    st.set_source(1.0)
    tw = st.jax_twin()
    sh = tw.sol_h
    ref = tw.layers[0]["ref"]
    res[name] = {}
    for k, (L, G), pts in (("geom", (sh.Stt - sh.Schur, -sh.Rmat),
                            [-1.0, -1.45 ** 2]),
                           ("layer", (ref.Lmat, -ref.Rmat), [0.0])):
        lam = np.linalg.eigvals(np.linalg.solve(np.asarray(G), np.asarray(L)))
        s = np.max(np.abs(lam))
        D = np.abs(lam[:, None] - lam[None, :]) + np.eye(lam.size) * 1e300
        ex = np.nonzero(np.min(D, axis=1) <= 1e-12 * s)[0]
        dist = [float(np.min(np.abs(lam[ex] - p)) / s) for p in pts]
        allm = [float(np.min(np.abs(lam - p)) / s) for p in pts]
        res[name][k] = {"s": float(s), "n_exact_members": int(ex.size),
                        "cluster_to_branch_rel": dist,
                        "any_eig_to_branch_rel": allm}
        print(name, k, res[name][k], flush=True)
print(dump(f"r1c_branch_M{M}.json", res))
