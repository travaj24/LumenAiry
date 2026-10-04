"""Tabulate the d1-d7 probes BEFORE (``*_post_<build>.json``, round 1, before
the round-2 routing) vs AFTER (``*_r2post_<build>.json``).

    python h_summary.py [before_tag after_tag]

Per configuration: rel_err of AD vs Richardson FD at each abscissa, the FD
premise median at the symmetric point, the mirror-identity defect (AD / FD)
where the probe reports one, the gauge change (max over seeds of the
gradient change under an in-cluster basis rotation, and of the value
change), the forward parity, and the exact-cluster membership (members with
gap <= 1e-12 / 1e-8, summed over the eig calls) at the symmetric point.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TAGS = sys.argv[1:3] if len(sys.argv) >= 3 else ["post", "r2post"]


def load(name, tag, build):
    p = os.path.join(HERE, f"{name}_{tag}_{build}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def g(x):
    return "-" if x is None else ("%.2e" % x if isinstance(x, float) else
                                  str(x))


def gauge_s(ga):
    if not isinstance(ga, dict):
        return str(ga)
    return "g%.1e/v%.1e" % (max(ga["grad_change_rel"]),
                            max(ga["value_change"]))


def clusters(sp):
    if sp is None:
        return "-"
    return "%d/%d" % (sum(r["members_below_1e-12"] for r in sp),
                      sum(r.get("members_below_1e-8", 0) for r in sp))


def sweep_row(sw):
    xs = list(sw)
    x0 = sw[xs[0]]
    rels = " ".join(f"{x}:{g(sw[x]['rel_err'])}" for x in xs)
    pm = x0["premise"]["median"] if isinstance(x0["premise"], dict) else None
    mir = ("AD %s FD %s" % (g(x0["mirror_defect_AD_rel"]),
                            g(x0["mirror_defect_FD_rel"]))
           if "mirror_defect_AD_rel" in x0 else "")
    return rels, pm, mir


def dgrad(d, name):
    """(config, sweep dict, gauge, parity, spectrum-at-0) rows."""
    rows = []
    if name in ("d4_berreman", "d5_pmmjones1d"):
        for k, sw in d["sweep"].items():
            x0 = list(sw)[0]
            sk = (f"{k}_{x0}" if name == "d4_berreman" else f"angle_{x0}")
            if name == "d5_pmmjones1d" and k == "depth":
                sk = None
            rows.append((k, sw, d["gauge"].get(k), d["parity"].get(k),
                         d["spectrum"].get(sk) if sk else None))
    else:
        for k, rec in d["sweep"].items():
            rows.append((k, rec["sweep"], rec.get("gauge"),
                         rec.get("parity"), d["spectrum"].get(f"{k}_0.0")))
            if "t1_sweep" in rec:
                rows.append((k + " [t1 ctrl]", rec["t1_sweep"],
                             rec.get("t1_gauge"), None, None))
    return rows


for build in ("win", "wsl"):
    print(f"\n######## build {build}  ({TAGS[0]} -> {TAGS[1]})")
    for name in ("d4_berreman", "d5_pmmjones1d", "d6_pmmstack",
                 "d7_pmmstack2d"):
        ds = [load(name, t, build) for t in TAGS]
        print(f"== {name}")
        if ds[1] is None:
            print("   AFTER missing")
            continue
        for tag, d in zip(TAGS, ds):
            if d is None:
                print(f"   [{tag}] missing")
                continue
            for k, sw, ga, par, sp in dgrad(d, name):
                rels, pm, mir = sweep_row(sw)
                print(f"   [{tag:6s}] {k:24s} rel {rels} | prem "
                      f"{g(pm)} | {mir} | gauge {gauge_s(ga)} | par "
                      f"{g(par)} | clus {clusters(sp)}")
    # controls
    print("== d1_bor / d2_borsem")
    for name, ks in (("d1_bor", ("n_r", "thk_control")),
                     ("d2_borsem", ("aniso_d", "iso_eps_control"))):
        for tag in TAGS:
            d = load(name, tag, build)
            if d is None:
                print(f"   {name} [{tag}] missing")
                continue
            for m, rec in d["m"].items():
                cl = "/".join(str(rec["spectrum"][s]["members_below_1e-12"])
                              for s in rec["spectrum"])
                print(f"   {name} [{tag:6s}] m={m} " + " ".join(
                    f"{k} {g(rec[k]['rel_err'])}" for k in ks)
                    + f" | parity {g(max(rec['numpy_parity_sumR_sumT']))}"
                    + f" | clus {cl}")
    print("== d3_eme")
    for tag in TAGS:
        d = load("d3_eme", tag, build)
        if d is None:
            print(f"   d3_eme [{tag}] missing")
            continue
        for fam, rec in d["fam"].items():
            xs = [k for k in rec if k.startswith("x0_")]
            print(f"   d3_eme [{tag:6s}] {fam:6s} " + " ".join(
                f"{x}: all {g(rec[x]['rel_err_all'])} sums "
                f"{g(rec[x]['rel_err_cluster_sums'])}" for x in xs)
                + f" | clus {rec['spectrum']['members_below_1e-12']}")
