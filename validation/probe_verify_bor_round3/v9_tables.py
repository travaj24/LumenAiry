"""Aggregate every per-arm JSON in this directory into the tables the report
quotes, so each one is reproducible from the repository rather than transcribed.

``python v9_tables.py`` prints the four tables and writes
``v9_tables_<arm>.json``.
"""
from __future__ import annotations

import glob
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _vb3  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    out = []
    for p in sorted(glob.glob(os.path.join(HERE, name))):
        with io.open(p, encoding="cp1252") as fh:
            out.append((os.path.basename(p), json.load(fh)))
    return out


def gap2_ladder():
    rows = []
    for nm, d in _load("v1_ladder_*.json"):
        a, s = d["arm"], d["summary"]["by_where"]
        rows.append(dict(
            file=nm, tree=("BASE" if "BASE" in nm else "ROUND3"),
            build=a["build"], requested=a["requested_coretype"],
            loaded=a["loaded_kernel"], threads=a["threads"],
            by_where={w: dict(refused=s[w]["refused"], n=s[w]["n"],
                              energy=s[w]["by_detector"].get("energy", 0),
                              ceiling=s[w]["by_detector"].get("ceiling", 0))
                      for w in ("inc", "exit", "both")}))
    print("== GAP 2, the ladder (inc / exit / both, refused of 78)")
    for r in rows:
        cells = " ".join("%-4s %2d/%2d (e%d,c%d)" % (
            w, r["by_where"][w]["refused"], r["by_where"][w]["n"],
            r["by_where"][w]["energy"], r["by_where"][w]["ceiling"])
            for w in ("inc", "exit", "both"))
        print("  %-6s %-4s %-12s t%-2s %s" % (r["build"], r["tree"],
                                              r["loaded"], r["threads"],
                                              cells))
    return rows


def gap34_ladder():
    rows = []
    for nm, d in _load("v3_ladder_*.json"):
        a, s = d["arm"], d["summary"]
        for g in ("A", "B"):
            if g not in s:
                continue
            rows.append(dict(
                file=nm, tree=("BASE" if "BASE" in nm else "ROUND3"),
                build=a["build"], loaded=a["loaded_kernel"],
                threads=a["threads"], geom=g, Rbig=s[g]["Rbig"],
                disagreeing=s[g]["n_disagreeing_rungs"],
                edge=s[g]["edge_verdicts"],
                edge_ulp=s[g].get("edge_ulp_from_bar"),
                nonfinite=len(s[g]["nonfinite_rows"]),
                nonfinite_read_ok=len(s[g]["nonfinite_read_ok"])))
    print("== GAPS 3/4, the liner ladder")
    for r in rows:
        print("  %-4s %-6s %-12s t%-2s %s Rbig=%-6g dis=%d nf=%d nf_ok=%d "
              "edge=%s/%s/%s"
              % (r["tree"], r["build"], r["loaded"], r["threads"], r["geom"],
                 r["Rbig"], r["disagreeing"], r["nonfinite"],
                 r["nonfinite_read_ok"], r["edge"]["axis"],
                 r["edge"]["middle"], r["edge"]["outer"]))
    return rows


def gap5_family():
    rows = []
    for nm, d in _load("v4_family_*.json"):
        a, s = d["arm"], d["summary"]
        rows.append(dict(
            file=nm, build=a["build"], loaded=a["loaded_kernel"],
            threads=a["threads"],
            count_one_number=s["count_one_number_everywhere"],
            count_is_idx_plus_one=s["count_idx_plus_one_everywhere"],
            shallow=s["shallow_envelope"], deep=s["deep_envelope"],
            bar_shallow=s["bar_margin_shallow"], bar_deep=s["bar_margin_deep"],
            highest_offending_rung=s["highest_qn_over_floor_exceeding_1e_6"]))
    print("== GAP 5, the near-cutoff family")
    for r in rows:
        print("  %-4s %-12s t%-2s count=%s/%s shallow=%.6g deep=%.6g "
              "1e-5/shallow=%.1fx 2e-3/deep=%.1fx knee_hi=%.4gx"
              % (r["build"], r["loaded"], r["threads"], r["count_one_number"],
                 r["count_is_idx_plus_one"], r["shallow"], r["deep"],
                 r["bar_shallow"], r["bar_deep"],
                 r["highest_offending_rung"][0]))
    return rows


def identity():
    rows = []
    trees = {}
    for nm, d in _load("v6_identity_*.json"):
        a = d["arm"]
        arm_id = "%s/%s/t%s" % (a["build"], a["loaded_kernel"],
                                a["threads"])
        key = (arm_id, "BASE" if "BASE" in nm else "ROUND3")
        trees[key] = d["fixtures"]
    print("== bit identity against 1ac6de7e")
    for build in sorted({k[0] for k in trees}):
        b = trees.get((build, "BASE"))
        p = trees.get((build, "ROUND3"))
        if not (b and p):
            continue
        common = sorted(set(b) & set(p))
        ident = [k for k in common if b[k] == p[k]]
        moved = [k for k in common if b[k] != p[k]]
        legal = [k for k in moved
                 if b[k]["kind"] == "hash" and p[k]["kind"] == "raise"
                 and p[k]["value"] == "BORNodalPassivityError"]
        illegal = [k for k in moved if k not in legal]
        nb = len([k for k in common if not k.startswith("eme_")])
        rows.append(dict(build=build, common=len(common), bor=nb,
                         eme=len(common) - nb, identical=len(ident),
                         moved=len(moved), legal=len(legal),
                         illegal=len(illegal), movers=sorted(legal),
                         illegal_rows=sorted(illegal)))
        print("  %-18s common=%d (BOR %d / EME %d) identical=%d moved=%d "
              "legal=%d ILLEGAL=%d" % (build, len(common), nb,
                                       len(common) - nb, len(ident),
                                       len(moved), len(legal), len(illegal)))
    return rows


def main():
    a = _vb3.arm()
    _vb3.require_tree(a)
    print("ARM", a["build"], a["loaded_kernel"], "t%s" % a["threads"],
          a["lumenairy_file"])
    payload = dict(gap2_ladder=gap2_ladder(), gap34_ladder=gap34_ladder(),
                   gap5_family=gap5_family(), identity=identity())
    _vb3.dump("v9_tables", payload, a)


if __name__ == "__main__":
    main()
