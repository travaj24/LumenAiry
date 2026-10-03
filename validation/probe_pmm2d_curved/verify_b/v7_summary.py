"""V7 summary -- oblique / conical: rung changes, closure, the two
topologies, RECIPROCITY (singular values of the power-normalised 2 x 2
reflection Jones block of a channel vs its reversal; coded here from the
plane-wave power metric G = I + k_t k_t^T / k_z^2), the wrong-pairing arm,
the film ladders (F-B6) and the y-momentum test.

  python v7_summary.py
Output: v7_summary.json
"""
import glob
import json
import os

import _vcommon as C
import numpy as np


def load(f):
    with open(f) as fh:
        return json.load(fh)


def jones_sv(d, m, n):
    md = d["modal"]
    o = [tuple(x) for x in md["orders"]]
    k = o.index((m, n))
    rx = np.array(md["rx"][0]) + 1j * np.array(md["rx"][1])
    ry = np.array(md["ry"][0]) + 1j * np.array(md["ry"][1])
    A = np.array([[rx[0, k], rx[1, k]], [ry[0, k], ry[1, k]]])
    kx0, ky0, kzi = md["kx0"], md["ky0"], md["kz_inc"]
    kxo = md["kx"][k]
    kyo = md["ky"][k]
    kzr = md["kz_ref"]
    kzo = float(kzr[0][k]) if isinstance(kzr[0], list) else float(kzr[k])

    def metric(kx, ky, kz):
        kt = np.array([kx, ky])
        return kz * (np.eye(2) + np.outer(kt, kt) / kz ** 2)

    def msqrt(S, p):
        w, V = np.linalg.eigh(S)
        return (V * w ** p) @ V.T
    N = msqrt(metric(kxo, kyo, kzo), 0.5) @ A @ msqrt(metric(kx0, ky0, kzi),
                                                      -0.5)
    return np.linalg.svd(N, compute_uv=False)


res = {"rungs": {}, "recip": {}, "film": {}, "momentum": {}}
files = glob.glob(os.path.join(C.HERE, "v7_rung_*.json"))
runs = {}
for f in files:
    d = load(f)
    runs[(d["kind"], d["M"], round(d["theta"], 4), round(d["phi"], 4))] = d
for kind in ("c3", "c5"):
    for ang in ((25.0, 0.0), (25.0, 40.0)):
        Ms = sorted(M for (k, M, t, p) in runs if k == kind
                    and (t, p) == ang)
        rows = {}
        prev = None
        for M in Ms:
            d = runs[(kind, M) + ang]
            v = np.array(d["vec"])
            rows[M] = {"closure": d["closure"],
                       "rung": None if prev is None else
                       float(np.max(np.abs(v - prev)))}
            prev = v
        res["rungs"][f"{kind}_{ang}"] = rows
pairs = {((25.0, 0.0), (-1, 0)): (24.2498, -0.0),
         ((25.0, 40.0), (-1, 0)): (35.2731, -28.0614),
         ((25.0, 40.0), (0, -1)): (40.4136, 119.9586)}
for (ang, mn), rev in pairs.items():
    for kind in ("c3", "c5"):
        out = {}
        for M in sorted(M for (k, M, t, p) in runs if k == kind
                        and (t, p) == ang):
            r = runs.get((kind, M) + rev)
            if r is None:
                continue
            f = runs[(kind, M) + ang]
            sf, sr = jones_sv(f, *mn), jones_sv(r, *mn)
            sw = jones_sv(r, 0, 0)
            out[M] = {"recip": float(np.max(np.abs(sf - sr))),
                      "wrong_pairing": float(np.max(np.abs(sf - sw)))}
        res["recip"][f"{kind}_{ang}_{mn}"] = out
# topologies at the top rungs
for ang in ((25.0, 0.0), (25.0, 40.0)):
    m3 = [M for (k, M, t, p) in runs if k == "c3" and (t, p) == ang]
    m5 = [M for (k, M, t, p) in runs if k == "c5" and (t, p) == ang]
    if not (m3 and m5):
        continue
    M3, M5 = max(m3), max(m5)
    a = np.array(runs[("c3", M3) + ang]["vec"])
    b = np.array(runs[("c5", M5) + ang]["vec"])
    res["rungs"][f"topologies_{ang}"] = {"c3_M": M3, "c5_M": M5,
                                         "max_abs": float(np.max(np.abs(a - b)))}
for f in glob.glob(os.path.join(C.HERE, "v7_film_*.json")):
    d = load(f)
    res["film"].setdefault(f"{d['kind']}_t{d['theta']}", {})[d["M"]] = d["err"]
for f in glob.glob(os.path.join(C.HERE, "v7_momentum_*.json")):
    d = load(f)
    res["momentum"].setdefault(d["map"], {})[d["M"]] = {
        k: d[k] for k in ("leak_max_per_order", "leak_unmapped_max_per_order",
                          "n0_orders_vs_unmapped", "closure")}
print(json.dumps(res, indent=1, default=float))
C.dump("v7_summary.json", res)
