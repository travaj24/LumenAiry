"""V7 -- the WINDOWS / WSL spread, decision by decision and digit by digit
(task 5).

Two different questions are asked of the same pair of JSONs.

* **Decisions.**  Every verdict, refusal, warning-set and classification this
  campaign records must be IDENTICAL on the two builds.  A single differing
  decision is a defect, not a tolerance -- the guard is a yes/no gate and a
  build-dependent gate is unusable.
* **Digits.**  The arbiter's evidence dict (``snapped_super_unity``, ``move``,
  ``w_wide``, ``mf_fix``, ``closure``, ``drop``, ``violation``) and the
  campaign's population statistics are floating-point reductions over a
  near-degenerate eigenproblem, so they are compared as RELATIVE spreads and
  the largest one is reported rather than pinned.

    python v7_crossbuild.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as fh:
        return json.load(fh)


def rel(a, b):
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return None
    if isinstance(a, bool) or isinstance(b, bool):
        return None
    m = max(abs(a), abs(b))
    return 0.0 if m == 0.0 else abs(a - b) / m


#: Entries below this magnitude are at the arithmetic's own noise floor -- a
#: relative spread between two numbers that are both ~1e-15 says nothing about
#: reproducibility, so they are reported separately rather than mixed in.
FLOOR = 1.0e-9

#: Keys that are wall-clock or build metadata, not measurements.
SKIP = ("wall", "secs", "build.python", "build.numpy", "build.platform",
        "build.lumenairy")


def skipped(tag):
    return any(tag.endswith(k) or f".{k}" in tag for k in SKIP)


def spread(pairs, floor=FLOOR):
    worst, who = 0.0, None
    for tag, a, b in pairs:
        if skipped(tag):
            continue
        if (isinstance(a, (int, float)) and isinstance(b, (int, float))
                and max(abs(a), abs(b)) < floor):
            continue
        r = rel(a, b)
        if r is not None and r > worst:
            worst, who = r, (tag, a, b)
    return worst, who


def main():
    out = {}

    # ---- v1: the arbiter's evidence dict, fixture by fixture --------------
    w = {r["name"]: r for r in load("v1_bitid_AFTER_win.json")["rows"]}
    s = {r["name"]: r for r in load("v1_bitid_AFTER_wsl.json")["rows"]}
    ev_pairs, dec_bad = [], []
    for nm in sorted(w):
        a, b = w[nm], s[nm]
        for k in ("refused", "arb", "screen", "kind", "kind_snap",
                  "ret_bitid"):
            if a.get(k) != b.get(k):
                dec_bad.append((nm, k, a.get(k), b.get(k)))
        if a.get("arb_ev") and b.get("arb_ev"):
            for k in sorted(a["arb_ev"]):
                ev_pairs.append((f"{nm}.{k}", a["arb_ev"][k],
                                 b["arb_ev"].get(k)))
        for k in ("worst", "su_snap", "drop", "move", "move_w", "w_wide",
                  "err_d", "err_snap_d"):
            if a.get(k) is not None and b.get(k) is not None:
                ev_pairs.append((f"{nm}.{k}", a[k], b[k]))
        # the returned buffers themselves are NOT expected to be identical
        # across builds (different BLAS kernels); the DECISION is.
    ws, who = spread(ev_pairs)
    out["v1"] = dict(fixtures=len(w), decisions_differing=len(dec_bad),
                     decision_diffs=dec_bad[:10],
                     max_rel_spread=ws, at=who,
                     n_compared=len(ev_pairs))

    # ---- v1 comparison summaries (BEFORE/AFTER) --------------------------
    cw, cs = load("v1_compare_win.json"), load("v1_compare_wsl.json")
    keys = [k for k in cw if isinstance(cw[k], (int, float))
            and not isinstance(cw[k], bool)]
    ws2, who2 = spread([(k, cw[k], cs[k]) for k in keys])
    out["v1_compare"] = dict(
        identical_int_fields=[k for k in keys
                              if isinstance(cw[k], int) and cw[k] == cs[k]],
        differing=[k for k in keys if cw[k] != cs[k]],
        max_rel_spread=ws2, at=who2)

    # ---- v2 / v3 / v4 / v5 summary statistics -----------------------------
    for tag, fw, fs in (("v2", "v2_closure_win.json", "v2_closure_wsl.json"),
                        ("v3", "v3_d5_win.json", "v3_d5_wsl.json"),
                        ("v4", "v4_move_win.json", "v4_move_wsl.json"),
                        ("v8", "v8_attack_win.json", "v8_attack_wsl.json")):
        try:
            a = load(fw)["summary"]
            b = load(fs)["summary"]
        except FileNotFoundError:
            continue
        num, dec = [], []
        for k in sorted(a):
            if isinstance(a[k], bool) or isinstance(b.get(k), bool):
                if a[k] != b.get(k):
                    dec.append((k, a[k], b.get(k)))
                continue
            if isinstance(a[k], int) and isinstance(b.get(k), int):
                if a[k] != b[k]:
                    dec.append((k, a[k], b[k]))
                continue
            if isinstance(a[k], float) and isinstance(b.get(k), float):
                num.append((k, a[k], b[k]))
        wsx, whox = spread(num)
        out[tag] = dict(integer_or_boolean_fields_differing=dec,
                        max_rel_spread=wsx, at=whox, n_compared=len(num))

    for tag, fw, fs in (("v5", "v5_testbars_win.json",
                         "v5_testbars_wsl.json"),
                        ("v6", "v6_open_items_win.json",
                         "v6_open_items_wsl.json"),
                        ("v9", "v9_falserefusal_win.json",
                         "v9_falserefusal_wsl.json")):
        try:
            a, b = load(fw), load(fs)
        except FileNotFoundError:
            continue
        num, dec = [], []

        def walk(x, y, path):
            if isinstance(x, dict) and isinstance(y, dict):
                for k in sorted(x):
                    if k in y:
                        walk(x[k], y[k], f"{path}.{k}")
            elif isinstance(x, list) and isinstance(y, list):
                for i, (u, v) in enumerate(zip(x, y)):
                    walk(u, v, f"{path}[{i}]")
            elif isinstance(x, bool) or isinstance(y, bool):
                if x != y:
                    dec.append((path, x, y))
            elif isinstance(x, str) or isinstance(y, str):
                if x != y and "msg" not in path and "note" not in path:
                    dec.append((path, x, y))
            elif isinstance(x, (int, float)) and isinstance(y, (int, float)):
                if isinstance(x, int) and isinstance(y, int):
                    if x != y:
                        dec.append((path, x, y))
                else:
                    num.append((path, x, y))

        walk(a, b, tag)
        num = [t for t in num if not t[0].endswith(".wall")]
        wsx, whox = spread(num)
        out[tag] = dict(decisions_differing=dec, max_rel_spread=wsx,
                        at=whox, n_compared=len(num))

    # ---- how far the DECISIONS sit from flipping ------------------------
    # For every arbitrated row of the independent box, the verdict turns on
    # two comparisons: ``su_snapped <= closure`` and ``move > 100 w_wide``.
    # The relative distance of the BINDING comparison from equality is what a
    # cross-build spread would have to exceed to move the verdict.
    try:
        rows = load("v2_closure_win.json")["rows"]
    except FileNotFoundError:
        rows = []
    margins = []
    for r in rows:
        if not r.get("arbitrated") or r.get("move_w") is None:
            continue
        su, worst = r["su_snap"], r["worst"]
        clo = max(1.0e-5, max(worst - 1.0, 0.0) * 1.0e-2)
        m_c = abs(su / clo - 1.0) if clo > 0 else float("inf")
        m_m = abs(r["move_w"] / 100.0 - 1.0)
        # the verdict is 'sliver' only if BOTH hold; the binding comparison is
        # the one that decides, i.e. the failing one when the verdict is
        # 'truncation' and the tighter one when it is 'sliver'
        both = (su <= clo) and (r["move_w"] > 100.0)
        margins.append(min(m_c, m_m) if both else
                       (m_c if su > clo else m_m))
    out["decision_margin_min"] = min(margins) if margins else None
    out["decision_margin_rows"] = len(margins)
    best = max((out[k]["max_rel_spread"] for k in out
                if isinstance(out[k], dict)
                and out[k].get("max_rel_spread") is not None), default=None)
    out["overall_max_rel_spread"] = best
    out["decision_headroom"] = (
        None if not margins or not best or best == 0
        else out["decision_margin_min"] / best)
    with open(os.path.join(HERE, "v7_crossbuild.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    for k in sorted(out):
        d = out[k]
        if not isinstance(d, dict):
            continue
        print(f"--- {k}: max rel spread {d.get('max_rel_spread')} at "
              f"{d.get('at')}")
        for f in ("decisions_differing", "integer_or_boolean_fields_differing",
                  "differing"):
            if d.get(f):
                print(f"    {f}: {d[f]}")
    print("OVERALL max relative cross-build spread:",
          out["overall_max_rel_spread"])
    print("smallest decision margin over", out["decision_margin_rows"],
          "arbitrated rows:", out["decision_margin_min"],
          "-> headroom", out["decision_headroom"])


if __name__ == "__main__":
    main()
