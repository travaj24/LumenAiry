"""Summarise the V2 film runs: per (map, eps, mu, angle) the correct-arm
ladder in M (R,T / Jones error vs the verifier's own Berreman) and the
mutation arms at their fixed M with the ratio to the correct arm at the same
M.  Output v2_summary.json; prints a compact table."""
import glob
import json
import os
from collections import defaultdict

from _vdcommon import HERE, dump

runs = defaultdict(dict)
for f in glob.glob(os.path.join(HERE, "v2_*_M*.json")):
    if f.endswith("summary.json"):
        continue
    d = json.load(open(f))
    key = (d["map"], d["eps"], d["mu"], tuple(d["angle"]))
    runs[key][(d["M"], d["mutation"] or "ok")] = d
out = {}
for key in sorted(runs):
    r = runs[key]
    lad = {M: (d["dRT"], d["dJ"], d["dA_layer_absorption"])
           for (M, m), d in sorted(r.items()) if m == "ok"}
    muts = {}
    for (M, m), d in sorted(r.items()):
        if m == "ok":
            continue
        ok = r.get((M, "ok"))
        muts[f"{m}@M{M}"] = {
            "dRT": d["dRT"], "dJ": d["dJ"],
            "ok_dRT": ok["dRT"] if ok else None,
            "ok_dJ": ok["dJ"] if ok else None,
            "caught_RT": (d["dRT"] >= 10 * ok["dRT"] and d["dRT"] >= 1e-6)
            if ok else None,
            "caught_J": (d["dJ"] >= 10 * ok["dJ"] and d["dJ"] >= 1e-6)
            if ok else None}
    k = "_".join([key[0], key[1], key[2], f"{key[3][0]}_{key[3][1]}"])
    out[k] = {"ladder": {str(M): v for M, v in lad.items()},
              "mutations": muts}
dump("v2_summary.json", {"rows": out})
for k, v in out.items():
    lad = " ".join(f"M{M}:{a:.1e}/{b:.1e}" for M, (a, b, _c) in
                   v["ladder"].items())
    print(k, lad)
    for m, x in v["mutations"].items():
        print("    ", m, f"{x['dRT']:.1e}/{x['dJ']:.1e}",
              "RT" if x["caught_RT"] else "--", "J" if x["caught_J"] else "--")
