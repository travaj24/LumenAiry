"""V4d -- the slanted slab under the 3 x 3 circle map at NORMAL incidence,
pushed past M = 7 (the composite weight tau = J^-1 t is ~1 / sg at the four
singular vertices): does the null test keep converging or plateau?"""
import sys

import _ve1common as V

out = {}
for M in [int(m) for m in sys.argv[1].split(",")]:
    for slant in (None, (0.17, -0.08)):
        r = V.slab_run(V.DIRGEN, V.make_map("c3", V.SLAB["P"]), M, "n",
                       slant=slant)
        key = f"{'slant' if slant else 'vertical'}_M{M}"
        out[key] = {k: r[k] for k in ("dRT", "dJr", "dJt", "clo", "wall")}
        print(key, out[key], flush=True)
        V.dump("v4d_slant_circle.json", out)
