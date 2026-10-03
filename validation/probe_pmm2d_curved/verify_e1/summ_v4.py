"""Print / collect the V4 slab ladders: worst of (R/T, Jr, Jt) per rung."""
import glob
import json
import os

out = {}
for fn in sorted(glob.glob("v4_slab_*.json")):
    d = json.load(open(fn))
    key = os.path.basename(fn)[8:-5]
    rows = {}
    for k, v in d.items():
        if k in ("env", "note"):
            continue
        mo, M = k.split("_M")
        rows.setdefault(mo, {})[int(M)] = (
            "ERR" if "error" in v else
            [v["dRT"], v["dJr"], v["dJt"], v["clo"]])
    out[key] = rows
    for mo in ("n", "o", "c"):
        if mo in rows:
            s = "  ".join(f"M{M}:" + ("ERR" if x == "ERR" else
                                      f"{max(x[:3]):.1e}")
                          for M, x in sorted(rows[mo].items()))
            print(f"{key:32s} {mo} {s}")
json.dump(out, open("v4_summary.json", "w"), indent=1)
