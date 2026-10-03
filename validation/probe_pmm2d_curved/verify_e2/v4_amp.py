"""V4 contrast discrimination at the AMPLITUDE level.

Efficiency differences are a biased measure across contrasts: a diffracted
order's efficiency is |a|^2 with a = O(delta eps), so an O(delta eps) error
in a shows up as an O(delta eps^2) efficiency error.  Here the measure is
the error of sqrt(efficiency) of every NON-SPECULAR propagating order
(= |a| up to a fixed per-order factor shared by both arms), normalised by
the largest sqrt(efficiency) of those orders in the reference.  A trace
non-smoothness that is FIRST order in delta eps gives a contrast-independent
normalised error; a second-order one gives an error falling like delta eps.
Writes v4_amp_win.json."""
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


ORD = np.asarray(load(glob.glob(os.path.join(
    HERE, "v4_ladder_ref_4_2.25_x0.12_M4_win.json"))[0])["orders"])
P0 = int(np.nonzero((ORD[:, 0] == 0) & (ORD[:, 1] == 0))[0][0])


def amp_err(a, b):
    """(normalised non-specular amplitude error, specular amp error)"""
    out = []
    num, den = 0.0, 0.0
    for k in ("R", "T"):
        A = np.sqrt(np.clip(arr(a[k]).real, 0, None))
        B = np.sqrt(np.clip(arr(b[k]).real, 0, None))
        m = np.ones(A.shape[1], bool)
        m[P0] = False
        num = max(num, float(np.abs(A[:, m] - B[:, m]).max()))
        den = max(den, float(B[:, m].max()))
        out.append(float(np.abs(A[:, P0] - B[:, P0]).max()))
    return num / den, num, den, max(out)


res = {}
# shipped spacer identity, per-arm R / T (the _amp reruns)
for E in ("4", "1.1", "1.02"):
    rows = []
    for fn in sorted(glob.glob(os.path.join(
            HERE, f"v4_spacer_shipped_{E}_M*_amp_win.json"))):
        r = load(fn)
        alone = dict(R=r["alone_R"], T=r["alone_T"])
        row = dict(M=r["M"])
        for arm in ("grid1", "offset3"):
            if arm in r and "R" in r[arm]:
                rel, num, den, spec = amp_err(r[arm], alone)
                row[arm] = dict(rel=rel, abs=num, scale=den, spec=spec)
        rows.append(row)
    rows.sort(key=lambda x: x["M"])
    if rows:
        res[f"spacer_shipped_{E}"] = rows

# per-layer (forced) vs the merged-map reference
for tag in ("4_2.25_x0.12", "1.1_1.1_x0.12"):
    refs = {load(f)["M"]: load(f) for f in glob.glob(os.path.join(
        HERE, f"v4_ladder_ref_{tag}_M*_win.json"))}
    if not refs:
        continue
    Mr = max(refs)
    rows = []
    for arm in ("pl", "ploff"):
        for f in glob.glob(os.path.join(HERE,
                                        f"v4_ladder_{arm}_{tag}_M*_win.json")):
            r = load(f)
            rel, num, den, spec = amp_err(r, refs[Mr])
            rows.append(dict(arm=arm, M=r["M"], rel=rel, abs=num, scale=den,
                             spec=spec))
    for m in sorted(refs):
        if m != Mr:
            rel, num, den, spec = amp_err(refs[m], refs[Mr])
            rows.append(dict(arm="merged", M=m, rel=rel, abs=num, scale=den,
                             spec=spec))
    rows.sort(key=lambda x: (x["arm"], x["M"]))
    res[f"ladder_{tag}_vs_merged_M{Mr}"] = rows

print(json.dumps(res, indent=1))
dump("v4_amp", res)
