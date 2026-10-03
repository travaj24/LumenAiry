"""V3 summary -- the circle ladders against the FEM at three radii, the two
topologies, the four-fold symmetry, and the staircases.

FEM: r = 0.36 the planner's saved ``fem/summary.json`` (provenance: this
verifier re-ran its 'h1.0 p4' mesh with the same runner and reproduced the
saved record to 1.4e-14, ``fem_r360_results.jsonl``); r = 0.48 / 0.24 this
verifier's own runs (``fem_r480_results.jsonl``, ``fem_r240_results.jsonl``)
of the same runner, best estimate = mean of the finest independent meshes,
error bar = largest per-order deviation from that mean.

  python v3_summary.py
Output: v3_summary.json
"""
import glob
import json
import os

import _vcommon as C
import numpy as np

KEYS = {"0,0": [(0, 0)], "1,0": [(1, 0), (-1, 0)], "0,1": [(0, 1), (0, -1)],
        "1,1": [(1, 1), (-1, 1), (1, -1), (-1, -1)]}
FINEST = [("h1.0_e20", 4), ("h0.8_e30", 4), ("h1.0", 6), ("h0.8_e20", 5),
          ("h1.0_e20", 5)]


def fem_ref(r):
    if abs(r - 0.36) < 1e-9:
        d = json.load(open(os.path.join(C.HERE, "..", "fem", "summary.json")))
        ref = {(s, k): d[s][k]["value"] for s in ("R", "T") for k in KEYS}
        bar = max(d[s][k]["maxdev"] for s in ("R", "T") for k in KEYS)
        return ref, bar, d["best_from"]
    recs = [json.loads(line) for line in open(os.path.join(
        C.HERE, f"fem_r{int(round(r * 1000))}_results.jsonl"))]
    sel = []
    for tag, p in FINEST:
        m = [x for x in recs if x["tag"] == tag and x["p"] == p]
        if m:
            sel.append(m[-1])
    ref, bar = {}, 0.0
    for s in ("R", "T"):
        for k in KEYS:
            v = np.array([x[s][k]["eff"] for x in sel])
            ref[(s, k)] = float(v.mean())
            bar = max(bar, float(np.max(np.abs(v - v.mean()))))
    rpt = [x["RplusT"] for x in sel]
    return ref, bar, [f"{x['tag']} p{x['p']}" for x in sel] + [
        f"R+T-1 max {max(abs(v - 1) for v in rpt):.1e}"]


def dist(d, ref):
    orders = [tuple(o) for o in d["orders"]]
    out = 0.0
    for (s, k), v in ref.items():
        arr = d["te_R"] if s == "R" else d["te_T"]
        for mn in KEYS[k]:
            out = max(out, abs(arr[orders.index(mn)] - v))
    return out


def vecof(d):
    return np.array(d["tm_R"] + d["te_R"] + d["tm_T"] + d["te_T"])


res = {}
for r in (0.36, 0.48, 0.24):
    ref, bar, src = fem_ref(r)
    row = {"fem_from": src, "fem_bar": bar, "c3": {}, "c5": {}, "stair": {}}
    for kind in ("c3", "c5"):
        prev = None
        for f in sorted(glob.glob(os.path.join(C.HERE, f"v3_{kind}_r{r}_M*.json")),
                        key=lambda x: int(x.split("_M")[-1][:-5])):
            d = json.load(open(f))
            v = vecof(d)
            row[kind][d["M"]] = {"to_fem": dist(d, ref),
                                 "closure": d["closure"],
                                 "sym": d["sym_te_tm"], "wall_s": d["wall_s"],
                                 "rung": (None if prev is None else
                                          float(np.max(np.abs(v - prev))))}
            prev = v
    # the two topologies at their top rungs
    t3 = max(row["c3"]) if row["c3"] else None
    t5 = max(row["c5"]) if row["c5"] else None
    if t3 and t5:
        a = vecof(json.load(open(os.path.join(C.HERE, f"v3_c3_r{r}_M{t3}.json"))))
        b = vecof(json.load(open(os.path.join(C.HERE, f"v3_c5_r{r}_M{t5}.json"))))
        row["topologies"] = {"c3_M": t3, "c5_M": t5,
                             "max_abs": float(np.max(np.abs(a - b)))}
    # staircases
    for f in glob.glob(os.path.join(C.HERE, f"v3_stair*_r{r}_M*.json")):
        d = json.load(open(f))
        row["stair"][f"k{d['k']}_M{d['M']}"] = {"to_fem": dist(d, ref),
                                                "vec": vecof(d).tolist()}
    res[str(r)] = row
    print(f"r = {r}: FEM bar {bar:.2e} from {src}")
    for kind in ("c3", "c5"):
        for M, x in sorted(row[kind].items()):
            print(f"  {kind} M={M:2d}  to FEM {x['to_fem']:.2e}  rung "
                  f"{(x['rung'] or 0):.1e}  closure {x['closure']:.1e}  sym "
                  f"{x['sym']:.1e}  {x['wall_s']:.0f} s")
    if "topologies" in row:
        print("  topologies", row["topologies"])
    for k, x in sorted(row["stair"].items()):
        print(f"  stair {k}: to FEM {x['to_fem']:.3e}")

# staircase direction cosines (r = 0.36 and 0.48): step k1 -> k2 against
# (curved - k1), and k1 -> k4 against (FEM - k1) on the 'te' FEM entries
for r, pairs in ((0.36, (("k1_M10", "k2_M7"), ("k2_M7", "k4_M4"),
                         ("k1_M8", "k2_M6"))),
                 (0.48, (("k1_M10", "k2_M7"), ("k2_M7", "k4_M4"),
                         ("k1_M8", "k2_M6")))):
    row = res[str(r)]
    top = max(row["c3"])
    cur = vecof(json.load(open(os.path.join(C.HERE, f"v3_c3_r{r}_M{top}.json"))))
    cos = {}
    for a, b in pairs:
        if a in row["stair"] and b in row["stair"]:
            va, vb = np.array(row["stair"][a]["vec"]), np.array(
                row["stair"][b]["vec"])
            step, aim = vb - va, cur - va
            cos[f"{a}->{b}"] = float(step @ aim / np.linalg.norm(step)
                                     / np.linalg.norm(aim))
    row["stair_direction_cos"] = cos
    print(r, "direction cosines", cos)
for r in res:
    for k in res[r]["stair"]:
        res[r]["stair"][k].pop("vec")
C.dump("v3_summary.json", res)
