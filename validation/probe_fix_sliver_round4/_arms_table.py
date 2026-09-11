"""Summarise the committed per-arm decision tables -- the numbers S7 of the
round-4 audit quotes, regenerated from the JSON rather than transcribed.

    python validation/probe_fix_sliver_round4/_arms_table.py
"""
import collections
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SLOPE_RIGHT = 10.0


def rows_of(t):
    return {(r["case"], "%.17g" % r["delta"]): r for r in t["rows"]}


def main():
    tabs = {}
    for fn in sorted(glob.glob(os.path.join(HERE, "p4_decisions_*.json"))):
        with open(fn, encoding="utf-8") as fh:
            d = json.load(fh)
        a = d["arm"]
        tabs[(a["build"], a["requested"], a["threads_requested"])] = d

    print("| build | requested | kernel dispatched | threads asked / run | "
          "rows | python / numpy | refused | wall | returned | "
          "wrong returned silently | worst slope-normalised score |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for key in sorted(tabs):
        d = tabs[key]
        a = d["arm"]
        c = collections.Counter(r["decision"] for r in d["rows"])
        silent = []
        for r in d["rows"]:
            if r.get("returned_kind") != "wrong" or r["decision"] != "returned":
                continue
            s = (r["d12"] / r["w_wide"]) if (r.get("d12") and r.get("w_wide")) \
                else None
            silent.append(r["returned_eod_both"] / s if s else float("inf"))
        print("| %s | %s | %s | %s / %s | %s | %s / %s | %d | %d | %d | %d "
              "| %.3g |"
              % (a["build"], a["requested"], a["kernel"],
                 a["threads_requested"], a["threads"],
                 ("%d (subset %s)" % (d["n"], ",".join(d["subset"])))
                 if d.get("subset") else str(d["n"]),
                 a["python"], a["numpy"], c["refused"], c["wall"],
                 c["returned"], len(silent),
                 max(silent) if silent else 0.0))

    full = {k: v for k, v in tabs.items() if not v.get("subset")}
    keys = sorted(set.intersection(*[set(rows_of(t)) for t in full.values()]))
    per = {k: {arm: rows_of(t)[k] for arm, t in full.items()} for k in keys}
    unstable = [k for k in keys
                if len({r["kind"] for r in per[k].values()}) > 1]
    disagree = [k for k in keys
                if len({r["kind"] for r in per[k].values()}) == 1
                and len({r["decision"] for r in per[k].values()}) > 1]
    fp = [(a, k) for k in keys for a, r in per[k].items()
          if r["kind"] == "right" and r["decision"] == "refused"]
    print()
    print("rows common to every arm      :", len(keys))
    print("answer-UNSTABLE across arms   :", len(unstable))
    print("decision differs where answers agree:", len(disagree), disagree[:6])
    print("false positives (any arm)     :", len(fp))

    spread = []
    for k in keys:
        if len({r["decision"] for r in per[k].values()}) != 1:
            continue
        v = [abs(r["worst"] - 1.0) for r in per[k].values()]
        if min(v) > 0.0:
            spread.append((max(v) / min(v), k))
    spread.sort(reverse=True)
    print("worst max R+T spread on an identically-decided row:",
          "%.4g" % spread[0][0], spread[0][1])
    print("  top 5:", [("%.4g" % s, k[0]) for s, k in spread[:5]])
    if unstable:
        print("answer-unstable rows (decision may differ):")
        for k in unstable[:12]:
            kinds = {a[1]: r["kind"] for a, r in per[k].items()}
            decs = {a[1]: r["decision"] for a, r in per[k].items()}
            print("   ", k, kinds, decs)

    # ------------------------------------------------------------------
    # THE SECOND LADDER: the thread count, one kernel at a time.
    # CI's fast lane leaves BLAS unpinned on 4-core runners while every
    # local measurement pinned one thread, so the thread count -- not the
    # kernel -- is the axis that separates this box from that one.
    # ------------------------------------------------------------------
    print()
    print("THREAD LADDER (same build AND same dispatched kernel, "
          "thread count varied)")
    print("| build | kernel | arms | thread settings | rows compared | "
          "answer differs | DECISION differs | worst err/delta spread |")
    print("|---|---|---|---|---|---|---|---|")
    by_kernel = collections.defaultdict(dict)
    for (build, req, thr), d in tabs.items():
        by_kernel[(build, d["arm"]["kernel"])][(req, thr)] = d
    total_thread_disagree = 0
    for bk in sorted(by_kernel):
        group = by_kernel[bk]
        if len(group) < 2:
            continue
        ks = sorted(set.intersection(*[set(rows_of(t))
                                       for t in group.values()]))
        ndiff = 0
        ddiff = []
        worst = 0.0
        for k in ks:
            rs = [rows_of(t)[k] for t in group.values()]
            if len({r["decision"] for r in rs}) > 1:
                ddiff.append(k)
            eods = [r["eod_both"] for r in rs]
            lo, hi = min(eods), max(eods)
            if hi > lo:
                ndiff += 1
                if lo > 0.0:
                    worst = max(worst, hi / lo)
        total_thread_disagree += len(ddiff)
        print("| %s | %s | %d | %s | %d | %d | %d | %.4g |"
              % (bk[0], bk[1], len(group),
                 ",".join(sorted(set(str(t) for _, t in group))),
                 len(ks), ndiff, len(ddiff), worst))
        if ddiff:
            print("    rows whose DECISION moved with the thread count:",
                  ddiff[:8])
    print("decision differences across the thread ladder (all kernels):",
          total_thread_disagree)


if __name__ == "__main__":
    main()
