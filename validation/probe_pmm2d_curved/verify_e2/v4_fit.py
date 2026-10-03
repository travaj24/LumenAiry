"""V4 analysis: errors of the ladders and least-squares fits of the
algebraic exponent, err ~ c M^-p and err ~ c q^-p (log-log), and of a
spectral model err ~ c exp(-a M) for comparison.  Reads v4_ladder_* and
v4_spacer_* JSON; writes v4_fit_win.json."""
import glob
import json
import os

import numpy as np
from _ve import HERE, dump


def load(fn):
    with open(os.path.join(HERE, fn)) as f:
        return json.load(f)


def arr(v):
    if isinstance(v, dict):
        return np.asarray(v["re"]) + 1j * np.asarray(v["im"])
    return np.asarray(v)


def d(a, b):
    return float(max(np.abs(arr(a["R"]) - arr(b["R"])).max(),
                     np.abs(arr(a["T"]) - arr(b["T"])).max()))


def fit(xs, es):
    xs, es = np.asarray(xs, float), np.asarray(es, float)
    ok = es > 0
    if ok.sum() < 3:
        return None
    lx, le = np.log(xs[ok]), np.log(es[ok])
    p, c = np.polyfit(lx, le, 1)
    res_alg = float(np.sqrt(np.mean((le - (p * lx + c)) ** 2)))
    a, b = np.polyfit(xs[ok], le, 1)
    res_exp = float(np.sqrt(np.mean((le - (a * xs[ok] + b)) ** 2)))
    return dict(p=float(-p), rms_loglog=res_alg, a_exp=float(-a),
                rms_semilog=res_exp)


def flat_strength(r):
    """Scattering strength: max |R - R_flat|, |T - T_flat| over orders and
    inputs, R_flat the bare air / n_sub interface (specular only)."""
    o = np.asarray(r["orders"])
    nsub = 1.45
    Rf = ((1 - nsub) / (1 + nsub)) ** 2
    R, T = arr(r["R"]).real, arr(r["T"]).real
    Rf_ = np.zeros_like(R)
    Tf_ = np.zeros_like(T)
    p0 = int(np.nonzero((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
    Rf_[:, p0], Tf_[:, p0] = Rf, 1 - Rf
    return float(max(np.abs(R - Rf_).max(), np.abs(T - Tf_).max()))


ORD = load(os.path.basename(glob.glob(os.path.join(HERE, "v4_ladder_*_win.json"))[0]))["orders"]
out = {}
for tag in ("4_2.25_x0.12", "1.1_1.1_x0.12", "1.02_1.02_x0.12"):
    refs = {}
    for fn in glob.glob(os.path.join(HERE, f"v4_ladder_ref_{tag}_M*_win.json")):
        r = load(os.path.basename(fn))
        refs[r["M"]] = r
    if not refs:
        continue
    Mr = max(refs)
    ref = refs[Mr]
    S = flat_strength(ref)
    rec = dict(ref_M=Mr, strength=S,
               ref_ladder={m: d(refs[m], ref) for m in sorted(refs)
                           if m != Mr},
               ref_self_last=(d(refs[Mr - 1], ref) if Mr - 1 in refs
                              else None))
    for arm in ("pl", "ploff"):
        rows = []
        for fn in glob.glob(os.path.join(HERE,
                                         f"v4_ladder_{arm}_{tag}_M*_win.json")):
            r = load(os.path.basename(fn))
            q = [3 * (r["M"] - 1), 2 * (r["Ms"][1] - 1)]
            rows.append(dict(M=r["M"], Ms=r["Ms"], q_circ=q[0], q_sin=q[1],
                             err=d(r, ref), err_rel=d(r, ref) / S,
                             vs_same_M_ref=(d(r, refs[r["M"]])
                                            if r["M"] in refs else None),
                             closure=max(r["closure"]), wall=r["wall"]))
        rows.sort(key=lambda x: x["M"])
        if rows:
            rec[arm] = dict(rows=rows,
                            fit_M=fit([x["M"] for x in rows],
                                      [x["err"] for x in rows]),
                            fit_q=fit([min(x["q_circ"], x["q_sin"])
                                       for x in rows],
                                      [x["err"] for x in rows]),
                            fit_M_5up=fit([x["M"] for x in rows
                                           if x["M"] >= 5],
                                          [x["err"] for x in rows
                                           if x["M"] >= 5]))
    out[tag] = rec

# the crossing pair: q-matching ON (nat) vs OFF, reference = nat at max M
tag = "4_2.25_x0.6"
nat = {}
for fn in glob.glob(os.path.join(HERE, f"v4_ladder_nat_{tag}_M*_win.json")):
    r = load(os.path.basename(fn))
    nat[r["M"]] = r
off = {}
for fn in glob.glob(os.path.join(HERE, f"v4_ladder_ploff_{tag}_M*_win.json")):
    r = load(os.path.basename(fn))
    off[r["M"]] = r
if nat:
    Mr = max(nat)
    ref = nat[Mr]
    rows = []
    for m in sorted(set(nat) | set(off)):
        row = dict(M=m)
        if m in nat:
            row.update(on_err=d(nat[m], ref) if m != Mr else 0.0,
                       on_Ms=nat[m]["Ms"], on_wall=nat[m]["wall"],
                       on_closure=max(nat[m]["closure"]))
        if m in off:
            row.update(off_err=d(off[m], ref), off_Ms=off[m]["Ms"],
                       off_wall=off[m]["wall"],
                       off_closure=max(off[m]["closure"]))
        if m in off and m in nat:
            row["on_vs_off"] = d(nat[m], off[m])
        rows.append(row)
    out["crossing_on_off"] = dict(ref_M=Mr, rows=rows)

# spacer identity ladders
for fam, arms in (("shipped", ("grid1", "offset3b", "offset3", "offset3q",
                               "conforming")),
                  ("curved", ("noride",))):
    for E in ("4", "1.1", "1.02"):
        rows = []
        for fn in glob.glob(os.path.join(HERE,
                                         f"v4_spacer_{fam}_{E}_M*_win.json")):
            r = load(os.path.basename(fn))
            row = dict(M=r["M"])
            for a in arms:
                if a in r:
                    row[a] = r[a]["vs_alone"]
            S = flat_strength(dict(orders=r.get("orders") or ORD, R=r["alone_R"],
                                   T=r["alone_T"]))
            row["strength"] = S
            for a in arms:
                if a in r:
                    row[a + "_rel"] = r[a]["vs_alone"] / S
            rows.append(row)
        if not rows:
            continue
        rows.sort(key=lambda x: x["M"])
        fits = {}
        for a in arms:
            xs = [x["M"] for x in rows if a in x]
            es = [x[a] for x in rows if a in x]
            fits[a] = dict(fit_M=fit(xs, es),
                           fit_q=fit([3 * (m - 1) for m in xs], es))
        out[f"spacer_{fam}_{E}"] = dict(rows=rows, fits=fits)

print(json.dumps(out, indent=1, default=str))
dump("v4_fit", out)
