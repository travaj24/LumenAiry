"""The V7 mutation matrix: per kind and fixture, the slab's worst error vs
the oracle (correct arm alongside) and the pillar's 36-vector change vs the
correct arm.  Output v7_summary.json."""
import glob
import json
import os

import _ve1common as V
import numpy as np

base = json.load(open(os.path.join(V.HERE, "v7_mut_none.json")))
out = {}
for fn in sorted(glob.glob(os.path.join(V.HERE, "v7_mut_*.json"))):
    kind = os.path.basename(fn)[7:-5]
    d = json.load(open(fn))
    row = {}
    for k, v in d.items():
        if k == "env":
            continue
        if "error" in v:
            row[k] = "RAISES: " + v["error"][:90]
        elif "worst" in v:
            row[k] = dict(worst=v["worst"], clo=v["clo"],
                          dJt=v["dJt"], dRT=v["dRT"])
        else:
            b = base[k]
            row[k] = dict(change=float(np.abs(np.asarray(v["vec"])
                                              - np.asarray(b["vec"])).max()),
                          clo=v["clo"])
    out[kind] = row
V.dump("v7_summary.json", out)
keys = [k for k in base if k != "env"]
print("kind".ljust(15) + "".join(k.ljust(10) for k in keys))
for kind, row in out.items():
    cells = []
    for k in keys:
        v = row.get(k)
        if v is None:
            cells.append("-")
        elif isinstance(v, str):
            cells.append("RAISE")
        elif "worst" in v:
            cells.append(f"{v['worst']:.1e}")
        else:
            cells.append(f"{v['change']:.1e}")
    print(kind.ljust(15) + "".join(c.ljust(10) for c in cells))
