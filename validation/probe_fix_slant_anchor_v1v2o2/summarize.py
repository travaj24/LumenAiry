"""Read every probe JSON of this directory and print the audit's tables.

Pure reporting: it re-reads what the probes measured on each arm and each
build and emits the comparisons the document quotes, so no number in
``docs/audits/FIX_SLANT_ANCHOR_V1_V2_O2_2026_09_11.md`` is transcribed by hand.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_R = os.path.join(_HERE, "results")


def load(name, suffix, build):
    fn = os.path.join(_R, "%s%s.slfix.%s.json" % (name, suffix, build))
    if not os.path.exists(fn):
        return None
    with open(fn, encoding="cp1252", errors="replace") as fh:
        return json.load(fh)


def _f(x, n=4):
    return "--" if x is None else ("%%.%de" % n) % x


def v1():
    print("\n" + "=" * 78 + "\nV1 -- the seven traced routes\n" + "=" * 78)
    for build in ("win", "wsl"):
        pre, post = load("v1_routes", "_prefix", build), \
            load("v1_routes", "_postfix", build)
        if not pre or not post:
            continue
        outs_pre = sorted({v["outcome"] for v in pre["routes"].values()})
        outs_post = sorted({v["outcome"] for v in post["routes"].values()})
        print("[%s] %d routes  pre=%s  post=%s" % (
            build, len(pre["routes"]), outs_pre, outs_post))
        r = pre["routes"]["depth"]
        print("      pre vs NumPy VERTICAL %s" % r["vs_numpy_VERTICAL"])
        print("      pre vs NumPy SLANTED  %s" % r["vs_numpy_SLANTED"])
        print("      the two NumPy calls   %s" % pre["numpy_slanted_vs_vertical"])
        print("      warnings on the silent rows: %r" % (r["warnings"],))
        for k in ("vertical_traced_depth", "const_tile_traced_depth"):
            same = pre["controls"][k]["hashes"] == post["controls"][k]["hashes"]
            print("      %-24s hashes identical pre/post: %s  %s"
                  % (k, same, post["controls"][k]["hashes"]))
        print("      const-tile NumPy no-op: %s"
              % post["controls"]["const_tile_numpy_slant_is_a_noop"])
        print("      grad AD/FD pre  %s" % pre["controls"]["grad_vertical_traced_depth"])
        print("      grad AD/FD post %s" % post["controls"]["grad_vertical_traced_depth"])


def v2_derive():
    print("\n" + "=" * 78 + "\nV2 -- the ladder against the engine's own "
          "z-staircase\n" + "=" * 78)
    for build in ("win", "wsl"):
        for suf in ("_prefix", "_postfix"):
            d = load("v2_derive", suf, build)
            if not d:
                continue
            print("\n[%s%s] surface=%s  P0 arg=%.6f  max||P|-1|=%.3e  "
                  "ptp(arg)=%.4f" % (
                      build, suf, d["transmission_surface"],
                      d["P0"]["arg"], d["P0"]["max_abs_dev"],
                      d["P0"]["ptp_arg"]))
            hdr = ("ns", "J as-ret", "J x P", "J x conj", "J / P",
                   "amps as-ret", "amps x P", "amps / P", "refl", "step J")
            print("   " + "".join("%-13s" % h for h in hdr))
            for k, v in d["ladder_vs_staircase"].items():
                row = [k, v["J_as_returned"], v["J_x_P"], v["J_x_conj_P"],
                       v.get("J_div_P"), v["amps_as_returned"], v["amps_x_P"],
                       v.get("amps_div_P"), v["J_reflection"],
                       v["stair_step_J"]]
                print("   %-13s" % row[0]
                      + "".join("%-13s" % _f(x, 4) for x in row[1:]))
            print("   uniform slanted film vs the VERTICAL film: "
                  + ", ".join("%s %s" % (k, _f(v, 4))
                              for k, v in d["uniform_slanted_film"].items()))
            n = d["normal_incidence"]
            print("   normal incidence: alpha0=%s |P0-1|=%s distinct P=%s "
                  "J as-ret %s  J x P0 %s  amps as-ret %s  amps x P %s"
                  % (n["alpha0"], n["P0_minus_1"], n["n_distinct_P"],
                     _f(n["J_as_returned"]), _f(n["J_x_P0"]),
                     _f(n["amps_as_returned"]), _f(n["amps_x_P"])))
            print("   seconds: %s" % d.get("seconds"))


def v2_cross():
    print("\n" + "=" * 78 + "\nV2 -- the CROSS-ENGINE arm\n" + "=" * 78)
    for build in ("win", "wsl"):
        for suf in ("_prefix", "_postfix"):
            d = load("v2_cross", suf, build)
            if not d:
                continue
            print("\n[%s%s] hybrid own n_orders step %s   pure own n_modes "
                  "step %s" % (build, suf,
                               _f(d.get("hybrid_own_n_orders_step")),
                               _f(d.get("pure_own_n_modes_step"))))
            for arm, rows in d["arms"].items():
                if isinstance(rows, str):
                    print("   %-16s %s" % (arm, rows))
                    continue
                for rn, r in rows.items():
                    print("   %-16s %-20s as-ret %s  x P0 %s  x conj %s  "
                          "/ P0 %s" % (arm, rn, _f(r["as_returned"]),
                                       _f(r["x_P0"]), _f(r["x_conj_P0"]),
                                       _f(r.get("div_P0"))))


def v2_compose():
    print("\n" + "=" * 78 + "\nV2 -- COMPOSITION and the LAYER SPLIT\n"
          + "=" * 78)
    for build in ("win", "wsl"):
        for suf in ("_prefix", "_postfix", "_smoke"):
            d = load("v2_compose", suf, build)
            if not d:
                continue
            print("\n[%s%s] %s" % (build, suf, d["fixture"]))
            for k, v in d["two_sheared_vs_staircase"].items():
                print("   %-5s " % k + "  ".join(
                    "%s %s" % (kk.replace("amps_", ""), _f(vv))
                    for kk, vv in sorted(v.items()) if kk.startswith("amps_")))
                print("         " + "  ".join(
                    "%s %s" % (kk, _f(vv))
                    for kk, vv in sorted(v.items()) if not kk.startswith("amps_")))
            print("   split: " + "  ".join(
                "%s %s" % (k, _f(v) if isinstance(v, float) else v)
                for k, v in sorted(d["layer_split"].items())))


def v2_census():
    print("\n" + "=" * 78 + "\nV2 -- the CENSUS of every PMMStack surface\n"
          + "=" * 78)
    for build in ("win", "wsl"):
        pre, post = load("v2_census", "_prefix", build), \
            load("v2_census", "_postfix", build)
        if not pre or not post:
            continue
        same = moved = 0
        moved_rows = []
        for fx in pre:
            if fx.startswith("_"):
                continue
            for key, val in pre[fx].items():
                if key not in post[fx]:
                    continue
                if post[fx][key] == val:
                    same += 1
                else:
                    moved += 1
                    moved_rows.append((fx, key))
        print("[%s] %d hashes identical, %d moved" % (build, same, moved))
        for fx, key in moved_rows:
            print("      MOVED  %-24s %s" % (fx, key))


def o2():
    print("\n" + "=" * 78 + "\nO2 -- the T22 population and the threshold\n"
          + "=" * 78)
    for build in ("win", "wsl"):
        d = load("o2_census", "_prefix", build)
        if not d:
            continue
        rows = []
        for k, v in d.items():
            if not isinstance(v, dict) or "T22" not in v:
                continue
            rs = v["T22"].get("rows") or []
            if not rs:
                continue
            if v["outcome"] == "RAISE":
                m = re.search(r"sum R\+T = ([0-9.e+-]+) exceeds (\d+)",
                              v.get("msg", ""))
                rt = float(m.group(1)) / float(m.group(2)) if m else float("inf")
            else:
                rt = v.get("RT", 0.0)
            rows.append((k, rt, rs))
        bad = [r for r in rows if r[1] > 1.10]
        good = [r for r in rows if r[1] <= 1.10]

        def band(pop):
            rc = [x["rcond_eq"] for _n, _t, rs in pop for x in rs]
            rd = [x["resid_eq"] for _n, _t, rs in pop for x in rs]
            cd = [x["cond"] for _n, _t, rs in pop for x in rs]
            return (len(pop), len(rc), min(rc), max(rc), min(rd), max(rd),
                    min(cd), max(cd))
        b, g = band(bad), band(good)
        wb = max(min(x["rcond_eq"] for x in rs) for _n, _t, rs in bad)
        wg = min(min(x["rcond_eq"] for x in rs) for _n, _t, rs in good)
        rb = min(max(x["resid_eq"] for x in rs) for _n, _t, rs in bad)
        rg = max(max(x["resid_eq"] for x in rs) for _n, _t, rs in good)
        print("\n[%s]  BROKEN  %d solves / %d ifc  rcond %.3e..%.3e  "
              "resid %.3e..%.3e  cond %.4g..%.4g" % (
                  build, b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]))
        print("      HEALTHY %d solves / %d ifc  rcond %.3e..%.3e  "
              "resid %.3e..%.3e  cond %.4g..%.4g" % (
                  g[0], g[1], g[2], g[3], g[4], g[5], g[6], g[7]))
        print("      per-SOLVE worst interface: broken max %.3e   healthy "
              "min %.3e   -> %.2f decades, geometric middle %.2e"
              % (wb, wg, math.log10(wg / wb), math.sqrt(wb * wg)))
        print("      per-SOLVE residual: broken min %.3e   healthy max %.3e"
              "   -> %.2f decades" % (rb, rg, math.log10(rb / rg)))
        ben = [(k, v["RT"], min(x["rcond_eq"] for x in v["T22"]["rows"]),
                max(x["resid_eq"] for x in v["T22"]["rows"]))
               for k, v in d.items()
               if isinstance(v, dict) and v.get("warned")
               and v.get("RT", 9e9) < 1.10 and (v["T22"].get("rows") or [])]
        if ben:
            print("      BENIGN warned rows: %d, RT %.4f..%.4f, rcond "
                  "%.3e..%.3e, resid <= %.3e" % (
                      len(ben), min(x[1] for x in ben), max(x[1] for x in ben),
                      min(x[2] for x in ben), max(x[2] for x in ben),
                      max(x[3] for x in ben)))

        post = load("o2_census", "_postfix", build)
        if post:
            ref = {k for k, v in post.items()
                   if isinstance(v, dict)
                   and v.get("exc") == "_ConditioningError"}
            still = [k for k, v in post.items()
                     if isinstance(v, dict) and v.get("outcome") == "SOLVED"]
            broken_names = {n for n, _t, _r in bad}
            print("      POST-FIX: %d refused, %d still solve; refused set == "
                  "BROKEN set: %s" % (len(ref), len(still),
                                      ref == broken_names))
            if ref != broken_names:
                print("        broken not refused: %s"
                      % sorted(broken_names - ref))
                print("        refused not broken: %s"
                      % sorted(ref - broken_names))
            print("      refused: %s" % sorted(ref))


def o2_identity():
    print("\n" + "=" * 78 + "\nO2 -- BIT-IDENTITY of the healthy fixtures, and "
          "the MORTAR\n" + "=" * 78)
    for build in ("win", "wsl"):
        pre, post = load("o2_identity", "_prefix", build), \
            load("o2_identity", "_postfix", build)
        if not pre or not post:
            continue
        same = moved = refused = 0
        moved_rows = []
        for k, v in pre.items():
            if k.startswith("_") or "R" not in v:
                continue
            w = post.get(k, {})
            if w.get("exc") == "_ConditioningError":
                refused += 1
                continue
            for key in ("orders", "R", "T", "J"):
                if key not in v or key not in w:
                    continue
                if v[key] == w[key]:
                    same += 1
                else:
                    moved += 1
                    moved_rows.append((k, key))
        print("[%s] %d hashes identical, %d moved, %d fixtures now refused"
              % (build, same, moved, refused))
        for r in moved_rows:
            print("      MOVED %s %s" % r)
        for k, v in post.get("_mortar", {}).items():
            print("      MORTAR %-30s %-7s RT=%-11.6g n=%-3s cond=%s"
                  % (k, v["outcome"], v.get("RT", float("nan")),
                     v["mortar"]["n"], v["mortar"]["max_cond"]))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in (("v1", v1), ("derive", v2_derive), ("cross", v2_cross),
                     ("compose", v2_compose), ("census", v2_census),
                     ("o2", o2), ("identity", o2_identity)):
        if which in ("all", name):
            fn()
