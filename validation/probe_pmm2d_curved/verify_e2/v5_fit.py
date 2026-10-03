"""V5 analysis: riding arms against the merged-map reference (highest ref M
of the same stack), q-matching ON/OFF on the crossing pair, misc probes.
Writes v5_fit_win.json."""
import glob
import json
import os

import numpy as np
from _ve import HERE, dump


def load(fn):
    with open(fn) as f:
        return json.load(f)


def arr(v):
    if isinstance(v, dict):
        return np.asarray(v["re"]) + 1j * np.asarray(v["im"])
    return np.asarray(v)


def d(a, b):
    return float(max(np.abs(arr(a["R"]) - arr(b["R"])).max(),
                     np.abs(arr(a["T"]) - arr(b["T"])).max()))


out = {}
runs = {}
for fn in glob.glob(os.path.join(HERE, "v5_ride_*_win.json")):
    r = load(fn)
    runs[(r["stack"], r["arm"], r["M"])] = r
refstack = {"CUS": "CUS", "SUC": "SUC", "CUSk": "CUS", "CVS": "CUS",
            "UCS": "UCS", "CSU": "CSU"}
for stk in ("CUS", "SUC", "UCS", "CSU", "CUSk", "CVS"):
    rs = refstack[stk]
    refMs = sorted(m for (s, a, m) in runs if s == rs and a == "ref")
    if not refMs:
        continue
    Mr = refMs[-1]
    ref = runs[(rs, "ref", Mr)]
    rec = dict(ref_stack=rs, ref_M=Mr,
               ref_ladder={m: d(runs[(rs, "ref", m)], ref)
                           for m in refMs[:-1]})
    rows = []
    for M in sorted({m for (s, a, m) in runs if s == stk}):
        row = dict(M=M)
        for arm in ("above", "below", "none", "ref"):
            r = runs.get((stk, arm, M))
            if r is None or (arm == "ref" and stk != rs):
                continue
            row[arm] = d(r, ref) if not (arm == "ref" and M == Mr) else 0.0
            if arm != "ref":
                row[arm + "_geoM"] = [g["M"] for g in r["geo"]]
                row[arm + "_closure"] = max(r["closure"])
        for a, b in (("above", "below"), ("above", "none")):
            if (stk, a, M) in runs and (stk, b, M) in runs:
                row[f"{a}_vs_{b}"] = d(runs[(stk, a, M)], runs[(stk, b, M)])
        rows.append(row)
    rec["rows"] = rows
    out[stk] = rec

# crossing pair q-matching ON / OFF
nat, off = {}, {}
for fn in glob.glob(os.path.join(HERE, "v4_ladder_nat_4_2.25_x0.6_M*_win.json")):
    r = load(fn)
    nat[r["M"]] = r
for fn in glob.glob(os.path.join(HERE,
                                 "v4_ladder_ploff_4_2.25_x0.6_M*_win.json")):
    r = load(fn)
    off[r["M"]] = r
if nat:
    Mr = max(nat)
    rows = []
    for m in sorted(set(nat) | set(off)):
        row = dict(M=m)
        if m in nat:
            row.update(on=d(nat[m], nat[Mr]), on_Ms=nat[m]["Ms"],
                       on_wall=nat[m]["wall"],
                       on_closure=max(nat[m]["closure"]))
        if m in off:
            row.update(off=d(off[m], nat[Mr]), off_Ms=off[m]["Ms"],
                       off_wall=off[m]["wall"],
                       off_closure=max(off[m]["closure"]))
        if m in nat and m in off:
            row["on_vs_off"] = d(nat[m], off[m])
        rows.append(row)
    out["crossing_qmatch"] = dict(ref="nat M%d" % Mr, rows=rows)
    if Mr - 1 in off:
        out["crossing_qmatch"]["off_ref_vs_on_ref"] = d(off[Mr - 1], nat[Mr])

# non-crossing pair ON / OFF against the merged reference
refs = {}
for fn in glob.glob(os.path.join(HERE, "v4_ladder_ref_4_2.25_x0.12_M*_win.json")):
    r = load(fn)
    refs[r["M"]] = r
if refs:
    Mr = max(refs)
    rows = []
    for m in range(4, 10):
        row = dict(M=m)
        for arm in ("pl", "ploff"):
            fn = os.path.join(HERE,
                              f"v4_ladder_{arm}_4_2.25_x0.12_M{m}_win.json")
            if os.path.exists(fn):
                r = load(fn)
                row[arm] = d(r, refs[Mr])
                row[arm + "_Ms"] = r["Ms"]
                row[arm + "_wall"] = r["wall"]
        if m in refs and m != Mr:
            row["merged"] = d(refs[m], refs[Mr])
        rows.append(row)
    out["noncrossing_qmatch"] = dict(ref="merged M%d" % Mr, rows=rows)

for fn in sorted(glob.glob(os.path.join(HERE, "v5_misc_*_win.json"))):
    r = load(fn)
    r.pop("env", None)
    out[os.path.basename(fn)[:-9]] = r
print(json.dumps(out, indent=1, default=str))
dump("v5_fit", out)

# the vacuum / pillar / vacuum sandwich (sup = sub = 1): ladders against M8
sw = {}
for fn in glob.glob(os.path.join(HERE, "v5_misc_sandwich_M*_win.json")):
    r = load(fn)
    if "alone_R" in r:
        sw[r["M"]] = r
if sw:
    Mr = max(sw)

    def pick(r, which):
        if which == "alone":
            return dict(R=r["alone_R"], T=r["alone_T"])
        if which == "ride":
            return dict(R=r["shared_R"], T=r["shared_T"])
        a = r["arms"]["forced_noride"]
        return dict(R=a["R"], T=a["T"])
    rows = []
    for m in sorted(sw):
        r = sw[m]
        row = dict(M=m)
        for w in ("alone", "ride", "noride"):
            for ref in ("alone", "noride"):
                row[f"{w}_vs_{ref}{Mr}"] = d(pick(r, w), pick(sw[Mr], ref))
        row["ride_closure"] = max(r["arms"]["forced"]["closure"])
        row["noride_closure"] = max(r["arms"]["forced_noride"]["closure"])
        row["alone_closure"] = max(r["alone_closure"])
        row["ride_bytes_eq_shared"] = [r["arms"][k]["bytes_eq_shared"]
                                       for k in ("nat", "forced",
                                                 "vacshapes")]
        rows.append(row)
    out["sandwich"] = dict(ref_M=Mr, rows=rows)
    print(json.dumps(out["sandwich"], indent=1))
    dump("v5_fit", out)
