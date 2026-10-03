"""V2 R / T ladder summary: for every full solve on a forced rule
(``v2_rt_*.json`` + the operator dumps ``_ops_*.npz``, which are NOT kept in
the repository -- 36 MB), the distance of R / T and of the operators (L, R)
to the verifier's own Duffy rule at n = 48.  Output: v2_rt_summary.json"""
import glob
import os

import _vcommon as C
import numpy as np

out = {}
for mp in ("c3", "c3_r0.48"):
    ref = np.array(C.load(f"v2_rt_{mp}_M6_vduffy_n48.json")["vec"])
    o0 = np.load(os.path.join(C.HERE, f"_ops_{mp}_M6_vduffy_n48.npz"))
    sc = max(abs(o0["L"]).max(), abs(o0["R"]).max())
    for f in sorted(glob.glob(os.path.join(C.HERE, f"v2_rt_{mp}_M6_*.json"))):
        d = C.load(os.path.basename(f))
        o = np.load(f.replace("v2_rt_", "_ops_").replace(".json", ".npz"))
        out[f"{mp}|{d['rule']}|{d['n']}"] = {
            "dRT": float(np.max(np.abs(np.array(d["vec"]) - ref))),
            "dops": float(max(abs(o["L"] - o0["L"]).max(),
                              abs(o["R"] - o0["R"]).max()) / sc)}
C.dump("v2_rt_summary.json", out)
print(len(out))
