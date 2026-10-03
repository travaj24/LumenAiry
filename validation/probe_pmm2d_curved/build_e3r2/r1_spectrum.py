"""R1: the cluster structure of the twin's eigs at the symmetric references
of V-E3-1 (square pillar, circle, square fillet) -- pairwise gaps of
eig(G^-1 L) relative to max|lam|, cluster sizes at several thresholds, and
the gap from each cluster to the rest of the spectrum.

    python r1_spectrum.py M
"""
import sys

import numpy as np
from _r2 import P, PMM2DStackPure, dump

from lumenairy.elements.pmm import Circle, FilletRect, Rect

M = int(sys.argv[1])
CASES = {"square": [Rect(0.6, 0.6, 0.5, 0.5, 3.5)],
         "circle": [Circle(0.6, 0.6, 0.33, 3.5)],
         "fillet_sq": [FilletRect(0.6, 0.6, 0.6, 0.6, 0.1, 3.5)]}


def clusters(lam, thr):
    s = np.max(np.abs(lam))
    n = lam.size
    C = np.abs(lam[:, None] - lam[None, :]) <= thr * s
    seen, out = np.zeros(n, bool), []
    for i in range(n):
        if seen[i]:
            continue
        comp, stack = [], [i]
        seen[i] = True
        while stack:
            k = stack.pop()
            comp.append(k)
            for j in np.nonzero(C[k] & ~seen)[0]:
                seen[j] = True
                stack.append(j)
        if len(comp) > 1:
            comp = np.array(comp)
            inner = np.max(np.abs(lam[comp][:, None] - lam[comp][None, :]))
            rest = np.setdiff1d(np.arange(n), comp)
            outer = np.min(np.abs(lam[rest][:, None] - lam[comp][None, :]))
            out.append((len(comp), inner / s, outer / s))
    return out


res = {"M": M}
for name, shp in CASES.items():
    st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45, n_modes=M,
                        n_orders=2, backend="jax")
    st.add_layer(0.45, shapes=shp, background_eps=1.0)
    st.set_source(1.0)
    tw = st.jax_twin()
    sh = tw.sol_h
    mats = {"geom": (sh.Stt - sh.Schur, -sh.Rmat),
            "layer": (tw.layers[0]["ref"].Lmat, -tw.layers[0]["ref"].Rmat)}
    res[name] = {}
    for k, (L, G) in mats.items():
        A = np.linalg.solve(G, L)
        lam = np.linalg.eigvals(A)
        d = {"n": int(lam.size), "max_abs": float(np.max(np.abs(lam)))}
        for thr in (1e-13, 1e-10, 1e-8, 1e-6, 1e-4):
            cl = clusters(lam, thr)
            sizes = sorted({c[0] for c in cl})
            d[f"thr={thr:g}"] = {
                "n_clusters": len(cl), "sizes": sizes,
                "max_inner": max((c[1] for c in cl), default=0.0),
                "min_outer": min((c[2] for c in cl), default=1.0)}
        res[name][k] = d
        print(name, k, d["n"], {t: (v["n_clusters"], v["sizes"],
                                    "%.1e" % v["max_inner"],
                                    "%.1e" % v["min_outer"])
                                for t, v in d.items() if t.startswith("thr")},
              flush=True)
print(dump(f"r1_spectrum_M{M}.json", res))
