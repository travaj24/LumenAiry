"""Summarise V6: the duality ladder with each disk's own rung change (the
RATE question of F-D4), the after-quadrature-inverse arms, the merged stack.
Output v6_summary.json."""
import glob
import json
import os

import numpy as np
from _vdcommon import HERE, dump


def arr(x):
    if isinstance(x, dict):
        return np.array(x["re"]) + 1j * np.array(x["im"])
    return np.array(x)


out = {"dual": {}, "muafter": {}, "stack": {}}
duals = {}
for f in glob.glob(os.path.join(HERE, "v6_dual_*.json")):
    d = json.load(open(f))
    duals.setdefault((d["kind"], d["mu"]), {})[d["M"]] = d
for (kind, mu), rows in sorted(duals.items()):
    Ms = sorted(rows)
    tab = []
    for a, b in zip(Ms, Ms[1:] + [None]):
        r = rows[a]
        row = {"M": a, "duality": r["duality"],
               "noswap": r["noswap_control"], "closure": r["closure"]}
        if b is not None:
            vm = np.concatenate([arr(r["Rm"]).ravel(), arr(r["Tm"]).ravel()])
            vn = np.concatenate([arr(rows[b]["Rm"]).ravel(),
                                 arr(rows[b]["Tm"]).ravel()])
            vd = np.concatenate([arr(r["Rd"]).ravel(), arr(r["Td"]).ravel()])
            vdn = np.concatenate([arr(rows[b]["Rd"]).ravel(),
                                  arr(rows[b]["Td"]).ravel()])
            row["rung_change_mag"] = float(np.abs(vm - vn).max())
            row["rung_change_diel"] = float(np.abs(vd - vdn).max())
        tab.append(row)
    out["dual"][f"{kind}_{mu}"] = tab
for f in glob.glob(os.path.join(HERE, "v6_muafter_*.json")):
    d = json.load(open(f))
    out["muafter"].setdefault(d["map"], {}).setdefault(str(d["M"]), {})[
        d["arm"]] = [d["dRT"], d["dJ"]]
for f in glob.glob(os.path.join(HERE, "v6_stack_*.json")):
    d = json.load(open(f))
    out["stack"].setdefault(d["arm"], {})[str(d["M"])] = {
        k: d[k] for k in ("closure", "A_lossless_layers", "sumA_vs_balance")}
dump("v6_summary.json", out)
for k, tab in out["dual"].items():
    print(k)
    for r in tab:
        print("   M%d dual %.2e  rung(mag) %s  rung(diel) %s  noswap %.2e" % (
            r["M"], r["duality"],
            "%.2e" % r["rung_change_mag"] if "rung_change_mag" in r else "--",
            "%.2e" % r["rung_change_diel"] if "rung_change_diel" in r
            else "--", r["noswap"]))
print(json.dumps(out["muafter"], indent=0))
print(json.dumps(out["stack"], indent=0))
