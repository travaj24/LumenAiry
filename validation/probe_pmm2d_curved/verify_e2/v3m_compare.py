"""V3 decisive check -- fold the v3m_* JSONs into one comparison.

  python v3m_compare.py [theta_deg phi_deg]
Output: v3m_compare_th<t>_ph<p>_<build>.json
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
th_d = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
ph_d = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
tag = f"th{th_d:g}_ph{ph_d:g}"
BUILD = sys.argv[3] if len(sys.argv) > 3 else "win"


def load(fn):
    with open(fn) as f:
        return json.load(f)


def asd(o, R, T):
    o = np.asarray(o, int)
    return {(int(a), int(b)): (np.asarray(R)[:, k], np.asarray(T)[:, k])
            for k, (a, b) in enumerate(o)}


def dist(a, b):
    keys = set(a) | set(b)
    z = (np.zeros(2), np.zeros(2))
    m = 0.0
    for k in keys:
        ra, ta = a.get(k, z)
        rb, tb = b.get(k, z)
        m = max(m, float(np.abs(ra - rb).max()), float(np.abs(ta - tb).max()))
    return m


def ladder(mode):
    out = {}
    for fn in glob.glob(os.path.join(HERE, f"v3m_{mode}_M*_{tag}_{BUILD}.json")):
        d = load(fn)
        out[d["M"]] = dict(rt=asd(d["orders"], d["R"], d["T"]),
                           closure=max(d["closure"]), wall=d["wall"],
                           extra={k: d.get(k) for k in ("pencil",
                                                        "modal_counts")})
    return dict(sorted(out.items()))


mg = ladder("merged")
pl = ladder("perlayer")
res = {"theta_deg": th_d, "phi_deg": ph_d}
Mref = max(mg)
ref = mg[Mref]["rt"]
res["M_ref_merged"] = Mref
res["merged"] = [{"M": M, "d_vs_ref": dist(v["rt"], ref) if M != Mref else 0.0,
                  "d_vs_next": (dist(v["rt"], mg[M + 1]["rt"])
                                if M + 1 in mg else None),
                  "closure": v["closure"], "wall": v["wall"], **v["extra"]}
                 for M, v in mg.items()]
res["perlayer"] = [{"M": M, "d_vs_merged_ref": dist(v["rt"], ref),
                    "d_vs_merged_sameM": (dist(v["rt"], mg[M]["rt"])
                                          if M in mg else None),
                    "d_vs_next": (dist(v["rt"], pl[M + 1]["rt"])
                                  if M + 1 in pl else None),
                    "closure": v["closure"], "wall": v["wall"],
                    **v["extra"]} for M, v in pl.items()]
# builder's per-layer R/T (normal incidence only)
if th_d == 0 and ph_d == 0:
    bdir = os.path.join(HERE, "..", "build_e2")
    bb = []
    for M, v in pl.items():
        fn = os.path.join(bdir, f"e2_4_closure_M{M}.json")
        if os.path.exists(fn):
            d = load(fn)
            bb.append({"M": M, "d_mine_vs_builder": dist(
                v["rt"], asd(d["orders"], d["R"], d["T"])),
                "builder_closure": max(d["closure"])})
    res["builder_check"] = bb
# RCWA
rc = []
rc_ex = {}
rc_r1 = {}
_runs = {}
for fn in sorted(glob.glob(os.path.join(HERE, f"v3m_rcwa_S*_{tag}_{BUILD}.json"))):
    for r in load(fn)["runs"]:
        _runs[r["n_orders"]] = (fn, r)
if True:
    prev = prev2 = None
    for n_ in sorted(_runs):
        fn, r = _runs[n_]
        a = asd(r["orders"], r["R"], r["T"])
        row = {"file": os.path.basename(fn), "n_orders": r["n_orders"],
               "d_vs_merged_ref": dist(a, ref),
               "closure": max(r["closure"]), "wall": r["wall"]}
        if pl:
            row["d_vs_perlayer_top"] = dist(a, pl[max(pl)]["rt"])
        if prev is not None:
            # Richardson in h = 1 / (2 n + 1), first order
            (n0, a0) = prev
            h0, h1 = 1 / (2 * n0 + 1), 1 / (2 * r["n_orders"] + 1)
            ex = {k: tuple((h0 * a[k][i] - h1 * a0[k][i]) / (h0 - h1)
                           for i in (0, 1)) for k in a}
            row["richardson1_d_vs_merged_ref"] = dist(ex, ref)
            rc_r1[r["n_orders"]] = ex
            if pl:
                row["richardson1_d_vs_perlayer_top"] = dist(
                    ex, pl[max(pl)]["rt"])
        if prev2 is not None:
            # three-rung fit R(h) = R_inf + a h + b h^2, h = 1 / (2 n + 1)
            hs = [1 / (2 * prev2[0] + 1), 1 / (2 * prev[0] + 1),
                  1 / (2 * r["n_orders"] + 1)]
            V = np.vander(hs, 3, increasing=True)
            Vi = np.linalg.inv(V)[0]
            ex2 = {k: tuple(Vi[0] * prev2[1][k][i] + Vi[1] * prev[1][k][i]
                            + Vi[2] * a[k][i] for i in (0, 1)) for k in a}
            row["richardson2_d_vs_merged_ref"] = dist(ex2, ref)
            if pl:
                row["richardson2_d_vs_perlayer_top"] = dist(
                    ex2, pl[max(pl)]["rt"])
            rc_ex[r["n_orders"]] = ex2
        prev2 = prev
        prev = (r["n_orders"], a)
        rc.append(row)
res["rcwa"] = rc
if rc_r1:
    nb = max(rc_r1)
    best = rc_r1[nb]
    res["rcwa_best"] = f"Richardson-1 at n_orders {nb}"
    res["vs_rcwa_best"] = {
        "merged": {M: dist(v["rt"], best) for M, v in mg.items()},
        "perlayer": {M: dist(v["rt"], best) for M, v in pl.items()}}
st = []
for fn in sorted(glob.glob(os.path.join(HERE, f"v3m_stair_n*_{tag}_{BUILD}.json"))):
    d = load(fn)
    for r in d["runs"]:
        a = asd(r["orders"], r["R"], r["T"])
        row = {"n": d["n"], "M": r["M"], "d_vs_merged_ref": dist(a, ref),
               "closure": max(r["closure"]), "pencils": r["pencils"],
               "wall": r["wall"]}
        if pl:
            row["d_vs_perlayer_top"] = dist(a, pl[max(pl)]["rt"])
        st.append(row)
res["stair"] = st
# zero-order headline values of the reference
k0 = (0, 0)
res["ref_R00_T00"] = [list(ref[k0][0]), list(ref[k0][1])]


def fit(Ms, ds):
    Ms, ds = np.asarray(Ms, float), np.asarray(ds, float)
    ok = ds > 0
    if ok.sum() < 2:
        return None
    pe = np.polyfit(Ms[ok], np.log(ds[ok]), 1)
    pa = np.polyfit(np.log(Ms[ok]), np.log(ds[ok]), 1)
    return {"exp_rate_per_M": float(-pe[0]), "alg_exponent": float(-pa[0])}


res["fit_perlayer_vs_merged_ref"] = fit(
    [r["M"] for r in res["perlayer"] if r["M"] < Mref],
    [r["d_vs_merged_ref"] for r in res["perlayer"] if r["M"] < Mref])
res["fit_merged_vs_ref"] = fit(
    [r["M"] for r in res["merged"] if r["M"] < Mref],
    [r["d_vs_ref"] for r in res["merged"] if r["M"] < Mref])
fn = os.path.join(HERE, f"v3m_compare_{tag}_{BUILD}.json")
with open(fn, "w") as f:
    json.dump(res, f, indent=1)
for k in ("merged", "perlayer", "builder_check", "rcwa", "stair"):
    print("==", k)
    for r in res.get(k, []):
        print({a: (f"{b:.3e}" if isinstance(b, float) else b)
               for a, b in r.items() if a != "file"})
print("vs_rcwa_best", res.get("rcwa_best"), {k: {m: f"{x:.2e}" for m, x in v.items()} for k, v in res.get("vs_rcwa_best", {}).items()})
print("fits", res["fit_perlayer_vs_merged_ref"], res["fit_merged_vs_ref"])
print("ref R00/T00", res["ref_R00_T00"])
print("wrote", os.path.basename(fn))
