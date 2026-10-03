"""E1-3 -- a UNIFORM out-of-plane tensor slab under a map is the slab: R, T
AND both Jones matrices against the exact Berreman 4x4 oracle
(``berreman_jones_1d``), rung by rung.

The physical device is the plain slab (the map only redistributes the
solver's resolution), so the oracle is exact; inside the solver every region
-- vacuum included -- carries the map's chi_t = g / sqrt(g), chi33 =
1 / sqrt(g), and the slab's congruence-transformed eps' has varying
out-of-plane entries adj(J) e_t3, so every new block of the Phase E1
generator is exercised.

Tensors (_e1common): oop (tilted LC, tilt 35 / azimuth 25 deg), nonrec
(Hermitian, non-symmetric), lossy, lnonrec (lossy AND non-reciprocal).
Maps: none (the shipped solver, reference), s05 / s15 (separable sine
stretch of both axes, 2 x 2), shear (3 x 3, non-diagonal J), shear2 (2 x 2,
the centre vertex moved), c3 / c5 (the circle maps: E1-4).

A SLANTED uniform slab (``slant=tx,ty``, public tangents) is physically the
same slab (a shear of a homogeneous medium is a coordinate change), so the
same oracle gates the COMPOSITE map x = Phi(u, v) + t w of a slanted layer
under a map (E1-6's null test).

usage: python e3_slab.py ladder <map> <tensor> <normal|oblique|conical> [M,M,..] [slant=tx,ty]
       python e3_slab.py arms <map> <tensor> <angle> <M> [kinds] [slant=tx,ty]
                                   (the gauge fail-befores and the E1-9 mutations)
writes e3_slab_<map>_<tensor>_<angle>.json / e3_arms_<map>_<tensor>_<angle>_M<M>.json
"""
import sys
import time

import _e1common as E
import numpy as np


def _stag(slant):
    return "" if slant is None else f"_slant{slant[0]:+.2f}{slant[1]:+.2f}"


def ladder(mapname, tname, ang, Ms, slant=None):
    f = E.SLAB
    cm = E.make_map(mapname, f["P"])
    th, ph = E.ANG[ang]
    rows = []
    for M in Ms:
        t0 = time.perf_counter()
        r = E.slab_vs_berreman(E.TENSORS[tname], cm, M, theta=np.deg2rad(th),
                               phi=np.deg2rad(ph), slant=slant)
        r.update(M=M, t=time.perf_counter() - t0)
        rows.append(r)
        print(mapname, tname, ang, M,
              " ".join(f"{k}={r[k]:.2e}" for k in ("dRT", "dJr", "dJt",
                                                   "closure", "t")),
              flush=True)
    E.dump(f"e3_slab_{mapname}_{tname}_{ang}{_stag(slant)}.json",
           {"map": mapname, "tensor": tname, "theta": th, "phi": ph,
            "slant": slant, "rows": rows})


def arms(mapname, tname, ang, M, kinds, slant=None):
    f = E.SLAB
    cm = E.make_map(mapname, f["P"])
    th, ph = E.ANG[ang]
    out = {"map": mapname, "tensor": tname, "theta": th, "phi": ph, "M": M,
           "slant": slant}
    kw = dict(theta=np.deg2rad(th), phi=np.deg2rad(ph), slant=slant)
    out["correct"] = E.slab_vs_berreman(E.TENSORS[tname], cm, M, **kw)
    print("correct", out["correct"], flush=True)
    for kind in kinds:
        with E.mutate(kind):
            out[kind] = E.slab_vs_berreman(E.TENSORS[tname], cm, M, **kw)
        print(kind, out[kind], flush=True)
    E.dump(f"e3_arms_{mapname}_{tname}_{ang}{_stag(slant)}_M{M}.json", out)


if __name__ == "__main__":
    sl = None
    for a in sys.argv[5:]:
        if a.startswith("slant="):
            sl = tuple(float(x) for x in a[6:].split(","))
    argv = [a for a in sys.argv if not a.startswith("slant=")]
    if argv[1] == "ladder":
        Ms = ([int(x) for x in argv[5].split(",")] if len(argv) > 5
              else [4, 5, 6, 7, 8])
        ladder(argv[2], argv[3], argv[4], Ms, slant=sl)
    elif argv[1] == "arms":
        kinds = (argv[6].split(",") if len(argv) > 6 else
                 ["rot_flip", "hgauge_plus_i", "hgauge_one", "no_mu_blocks",
                  "chi33_no_sg", "g3_strong"])
        arms(argv[2], argv[3], argv[4], int(argv[5]), kinds, slant=sl)
    else:
        raise SystemExit(__doc__)
