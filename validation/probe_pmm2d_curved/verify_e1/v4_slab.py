"""V4 -- E1-3 / E1-3m on the verifier's own tensors, maps and oracle.

usage: python v4_slab.py <combo> <map> <M,M,...> [slant]

combo: dirgen (general director, lossless) | nrgen (non-reciprocal) |
       gyrol_gyrol (lossy gyrotropic rotated eps WITH a lossy gyrotropic mu:
       the QZ branch) | dirgen_gyro (lossless gyrotropic mu: the whitened
       branch with a non-symmetric chi) | dirgen_aniso
map:   none | th3 (asymmetric two-harmonic, 3 x 3) | th2b (2 x 2, strong) |
       sh4 (4 x 4 sheared transfinite) | c3 | c3off | c5
slant: optional "tx,ty" public slant (the composite frame null test).
Mounts normal / oblique (30, 0) / conical (22, 63); R/T, Jr, Jt against the
own (eps, mu) oracle (``_ve1common.eps_mu_slab``); closure; wall.
"""
import sys

import _ve1common as V

combo, mapname = sys.argv[1], sys.argv[2]
Ms = [int(m) for m in sys.argv[3].split(",")]
slant = (tuple(float(x) for x in sys.argv[4].split(","))
         if len(sys.argv) > 4 else None)
eps_name, _, mu_name = combo.partition("_")
t33 = V.TENSORS[eps_name]
mu = V.MUS[mu_name or "none"]
cm = V.make_map(mapname, V.SLAB["P"])
TAG = "" if slant is None else f"_s{slant[0]:+.2f}{slant[1]:+.2f}"
out = {}
for mo in ("n", "o", "c"):
    for M in Ms:
        try:
            r = V.slab_run(t33, cm, M, mo, slant=slant, mu=mu)
            out[f"{mo}_M{M}"] = {k: r[k] for k in ("dRT", "dJr", "dJt",
                                                   "clo", "wall")}
        except Exception as exc:
            out[f"{mo}_M{M}"] = {"error": repr(exc)[:300]}
        print(combo, mapname, mo, M, out[f"{mo}_M{M}"], flush=True)
        V.dump(f"v4_slab_{combo}_{mapname}{TAG}.json", out)   # incremental
