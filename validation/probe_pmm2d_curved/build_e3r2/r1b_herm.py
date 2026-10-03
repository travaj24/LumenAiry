"""R1b: is each eig pencil of the V-E3-1 references Hermitian (L = L^H,
G = G^H > 0)?  Relative anti-Hermitian parts and the largest |Im lam| /
max|lam|; for the exact clusters, their |lam| / max|lam|.

    python r1b_herm.py M
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
    for k, (L, G) in {"geom": (sh.Stt - sh.Schur, -sh.Rmat),
                      "layer": (ref.Lmat, -ref.Rmat)}.items():
        L, G = np.asarray(L), np.asarray(G)
        lam = np.linalg.eigvals(np.linalg.solve(G, L))
        s = np.max(np.abs(lam))
        D = np.abs(lam[:, None] - lam[None, :]) + np.eye(lam.size) * 1e300
        ex = np.nonzero(np.min(D, axis=1) <= 1e-12 * s)[0]
        d = {"L_antiherm": float(np.linalg.norm(L - L.conj().T)
                                 / np.linalg.norm(L)),
             "G_antiherm": float(np.linalg.norm(G - G.conj().T)
                                 / np.linalg.norm(G)),
             "G_min_eig": float(np.min(np.linalg.eigvalsh(
                 0.5 * (G + G.conj().T)))),
             "max_abs_lam": float(s),
             "max_imag_rel": float(np.max(np.abs(lam.imag)) / s),
             "cluster_lam_rel_min": float(np.min(np.abs(lam[ex])) / s)
             if ex.size else None,
             "cluster_lams": sorted({round(float(x), 6) for x in
                                     lam[ex].real})[:12]}
        res[name][k] = d
        print(name, k, {a: (("%.2e" % b) if isinstance(b, float) else b)
                        for a, b in d.items()}, flush=True)
print(dump(f"r1b_herm_M{M}.json", res))
