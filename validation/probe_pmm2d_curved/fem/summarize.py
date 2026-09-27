"""Tabulate results.jsonl (per-order efficiencies per refinement) and write summary.json with the
best estimate + an error bar from the spread of the three highest-resolution independent meshes.
ASCII-only source."""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
recs = [json.loads(l) for l in open(os.path.join(HERE, "results.jsonl"))]
KR = ["0,0", "1,0", "0,1"]
KT = ["0,0", "1,0", "0,1", "1,1"]
hdr = "%-16s %2s %8s %6s %6s | %9s %9s %9s | %9s %9s %9s %9s | %9s %9s %10s | %9s %9s" % (
    "tag", "p", "ndof", "ntet", "t_s", "R00", "R10", "R01", "T00", "T10", "T01", "T11",
    "sumR", "sumT", "R+T-1", "fluxR", "fluxT")
print(hdr)
seen = set()
rows = []
for r in recs:
    key = (r["tag"], r["p"])
    if key in seen:
        continue
    seen.add(key)
    rows.append(r)
    print("%-16s %2d %8d %6d %6.0f | %s | %s | %9.6f %9.6f %10.2e | %9.6f %9.6f" % (
        r["tag"], r["p"], r["ndof"], r["ntet"], r["total_s"],
        " ".join("%9.6f" % r["R"][k]["eff"] for k in KR),
        " ".join("%9.6f" % r["T"][k]["eff"] for k in KT),
        r["sumR"], r["sumT"], r["RplusT"] - 1.0, np.mean(r.get("R_flux", [np.nan])), np.mean(r.get("T_flux", [np.nan]))))
best_keys = [("h1.0_e20", 4), ("h0.8_e30", 4), ("h1.0", 6)]
sel = [next(r for r in rows if (r["tag"], r["p"]) == k) for k in best_keys]
summ = {"best_from": ["%s p%d" % k for k in best_keys], "R": {}, "T": {}}
for side, keys in (("R", KR + ["1,1"]), ("T", KT)):
    for k in keys:
        v = np.array([s[side][k]["eff"] for s in sel])
        summ[side][k] = dict(value=float(v.mean()), spread_halfrange=float((v.max() - v.min()) / 2),
                             maxdev=float(np.max(np.abs(v - v.mean()))), propagating=sel[0][side][k]["propagating"])
for q in ("sumR", "sumT", "RplusT"):
    v = np.array([s[q] for s in sel])
    summ[q] = dict(value=float(v.mean()), maxdev=float(np.max(np.abs(v - v.mean()))))
summ["multiplicity"] = {"0,0": 1, "1,0": 2, "0,1": 2, "1,1": 4}
json.dump(summ, open(os.path.join(HERE, "summary.json"), "w"), indent=1)
print(json.dumps(summ, indent=1))
