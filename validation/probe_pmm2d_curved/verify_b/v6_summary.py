"""V6 summary -- the fillet ladders (base 5 x 5 and graded 7 x 7), the sharp
square references (plain 3 x 3 and graded 7 x 7), the shifts dR00 / dT00,
and the r -> 0 question: fits of the shift against r with several bases
(the build's c + a r^2 + b r^3; the corner-singularity exponent
c + a r^(2 lambda) + b r^2 with lambda = 0.806, the first Meixner exponent of
a 90-degree eps-4 / eps-1 dielectric wedge, computed in the report), and the
spread of the fitted constant over bases and references.

  python v6_summary.py
Output: v6_summary.json
"""
import glob
import json
import os

import _vcommon as C
import numpy as np

LAM = 0.806
SIDE = 0.6


def load(f):
    with open(f) as fh:
        return json.load(fh)


fil, sq = {}, {}
for f in glob.glob(os.path.join(C.HERE, "v6_fillet_*.json")):
    d = load(f)
    fil[(d["ratio"], d["M"], bool(d["grade"]))] = (d["R00"], d["T00"])
for f in glob.glob(os.path.join(C.HERE, "v6_square_*.json")):
    d = load(f)
    sq[(d["M"], bool(d["grade"]))] = (d["R00"], d["T00"])
res = {"fillet": {}, "square": {}, "refs": {}, "shifts": {}, "fits": {}}
for g in (False, True):
    Ms = sorted(M for (M, gg) in sq if gg == g)
    res["square"]["grade" if g else "plain"] = {
        M: {"R00": sq[(M, g)][0], "T00": sq[(M, g)][1],
            "rung_R": (abs(sq[(M, g)][0] - sq[(Mp, g)][0]) if Mp else None),
            "rung_T": (abs(sq[(M, g)][1] - sq[(Mp, g)][1]) if Mp else None)}
        for Mp, M in zip([None] + Ms[:-1], Ms)}
ratios = sorted({r for (r, M, g) in fil})
for r in ratios:
    for g in (False, True):
        Ms = sorted(M for (rr, M, gg) in fil if rr == r and gg == g)
        if not Ms:
            continue
        res["fillet"][f"{r}_{'grade' if g else 'base'}"] = {
            M: {"R00": fil[(r, M, g)][0], "T00": fil[(r, M, g)][1],
                "rung_R": (abs(fil[(r, M, g)][0] - fil[(r, Mp, g)][0])
                           if Mp else None),
                "rung_T": (abs(fil[(r, M, g)][1] - fil[(r, Mp, g)][1])
                           if Mp else None)}
            for Mp, M in zip([None] + Ms[:-1], Ms)}
# references: the best plain and graded squares
refs = {}
if sq:
    pm = max(M for (M, g) in sq if not g)
    refs[f"plain_M{pm}"] = sq[(pm, False)]
    gms = [M for (M, g) in sq if g]
    if gms:
        refs[f"grade_M{max(gms)}"] = sq[(max(gms), True)]
res["refs"] = refs
# the best fillet value per ratio: top base rung, and top graded rung
best = {}
for r in ratios:
    for g in (False, True):
        Ms = [M for (rr, M, gg) in fil if rr == r and gg == g]
        if Ms:
            best[(r, g)] = (max(Ms), fil[(r, max(Ms), g)])
for rk, rv in refs.items():
    for (r, g), (M, v) in best.items():
        res["shifts"][f"{r}_{'grade' if g else 'base'}_M{M}_vs_{rk}"] = {
            "dR00": v[0] - rv[0], "dT00": v[1] - rv[1]}
# fits over the base top rungs (ratio >= 0.05, like the build) and over all
bases = {"c + r^2 + r^3": lambda x: [np.ones_like(x), x ** 2, x ** 3],
         "c + r^2lam + r^2": lambda x: [np.ones_like(x), x ** (2 * LAM),
                                        x ** 2],
         "c + r^2lam + r^3": lambda x: [np.ones_like(x), x ** (2 * LAM),
                                        x ** 3]}
for rk, rv in refs.items():
    for sel_name, sel in (("r>=0.05", [r for r in ratios if r >= 0.05]),
                          ("all", ratios)):
        rr = [r for r in sel if (r, False) in best]
        if len(rr) < 3:
            continue
        x = np.array(rr) * SIDE
        for q, k in (("R00", 0), ("T00", 1)):
            y = np.array([best[(r, False)][1][k] - rv[k] for r in rr])
            for bn, bf in bases.items():
                A = np.array(bf(x)).T
                c, *_ = np.linalg.lstsq(A, y, rcond=None)
                res["fits"][f"{rk}|{sel_name}|{q}|{bn}"] = {
                    "c": float(c[0]),
                    "max_resid": float(np.max(np.abs(A @ c - y)))}
print(json.dumps(res, indent=1, default=float)[:6000])
C.dump("v6_summary.json", res)
