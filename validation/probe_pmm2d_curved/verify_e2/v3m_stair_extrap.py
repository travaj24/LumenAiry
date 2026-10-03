"""First-order (in 1/n) Richardson of the staircase ladder at fixed M, against
the merged-map M = 8 answer, the per-layer top rung and the RCWA estimate.
python v3m_stair_extrap.py"""
import glob
import json

import numpy as np


def asd(o, R, T):
    return {tuple(x): (np.asarray(R)[:, k], np.asarray(T)[:, k])
            for k, x in enumerate(o)}


def dist(a, b):
    return max(max(float(np.abs(a[k][0] - b[k][0]).max()),
                   float(np.abs(a[k][1] - b[k][1]).max())) for k in b)


def L(fn):
    d = json.load(open(fn))
    return asd(d["orders"], d["R"], d["T"])


mg = L("v3m_merged_M8_th0_ph0_win.json")
pl = L("v3m_perlayer_M10_th0_ph0_win.json")
st = {}
for fn in glob.glob("v3m_stair_n*_th0_ph0_win.json"):
    d = json.load(open(fn))
    for r in d["runs"]:
        st[(d["n"], r["M"])] = asd(r["orders"], r["R"], r["T"])
out = []
for M in (4, 5):
    ns = sorted(n for (n, m) in st if m == M)
    for n0, n1 in zip(ns[:-1], ns[1:]):
        a0, a1 = st[(n0, M)], st[(n1, M)]
        ex = {k: tuple((n1 * a1[k][i] - n0 * a0[k][i]) / (n1 - n0)
                       for i in (0, 1)) for k in a1}
        row = dict(M=M, n_pair=[n0, n1], ex_vs_merged8=dist(ex, mg),
                   ex_vs_perlayer10=dist(ex, pl),
                   raw_n1_vs_merged8=dist(a1, mg))
        out.append(row)
        print(row)
json.dump(out, open("v3m_stair_extrap_win.json", "w"), indent=1)
