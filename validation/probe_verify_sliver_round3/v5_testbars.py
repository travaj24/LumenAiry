"""V5 -- every bar in ``tests/unit/test_fix_pmmstack_sliver_round3.py``,
re-measured on the running build (task 6).

The point is durability, not correctness of the physics: for each numeric
constant the test asserts against, this probe reports the value that was
actually measured, the margin the assertion carries, and -- where the
assertion is one-sided -- what the OTHER side reads, so the report can say
whether the bar is two-sided and whether it is a property of the SAMPLE the
test happens to enumerate or of the FAMILY it claims to describe.

The test module itself is imported, so the fixtures scored here are the
test's own and nothing is transcribed.

    python v5_testbars.py out.json
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


import v_fixtures as F  # noqa: E402

sys.path.insert(0, os.path.join(F.ROOT, "tests", "unit"))

import test_fix_pmmstack_sliver_round3 as T  # noqa: E402

from lumenairy.elements.pmm import stack as ps  # noqa: E402


def d5_arm():
    """Test (a): the D-5 reproducer."""
    ref = T._raw(T._gmr(0.0))
    ladder = [T._raw(T._gmr(0.0, deg))[3] - 1.0 for deg in (6, 8, 10, 12)]
    out = dict(ladder=ladder,
               ladder_monotone=bool(ladder[0] > ladder[1] > ladder[2]
                                    > ladder[3]),
               deg8_over_abs_closure=ladder[1] / ps._SLIVER_ATTRIB_CLOSURE,
               deg8_under_trigger=ps._SLIVER_TRIGGER_BAR / ladder[1],
               rows=[])
    for delta in T._D5_DELTAS:
        st = T._gmr(delta)
        cur = T._raw(st)
        snapped, hit = T._snapped(st, T._gmr, delta)
        err = T._move(cur, ref, pol=1) / delta
        err_snapped = T._move(snapped, ref, pol=1) / delta
        su = max(snapped[3] - 1.0, 0.0)
        v, ev = ps._sliver_arbiter(st, cur[3], cur[1], cur[2], None)
        out["rows"].append(dict(
            delta=delta, worst=cur[3], err_d=err, err_snap_d=err_snapped,
            su_snap=su, su_over_abs=su / ps._SLIVER_ATTRIB_CLOSURE,
            drop=T._drop(cur[3], su),
            drop_margin=T._drop(cur[3], su) * ps._SLIVER_CLOSURE_FRACTION,
            move_w=ev["move"] / ev["w_wide"],
            move_margin=(ev["move"] / ev["w_wide"]) / ps._SLIVER_MOVE_FACTOR,
            closure=ev["closure"],
            closure_over_abs=ev["closure"] / ps._SLIVER_ATTRIB_CLOSURE,
            verdict=v,
            premise_met=bool(err > 100.0 and err_snapped < 1.0),
            err_margin_over_100=err / 100.0,
            err_snap_margin_under_1=1.0 / err_snapped))
    out["n_premise_met"] = sum(1 for r in out["rows"] if r["premise_met"])
    return out


def box_arm():
    """Tests (a2) and (b): the staircase box and the two drop populations."""
    import itertools
    rows, corr = [], []
    n_rows = n_arb = 0
    for nsub, nsup, theta in itertools.product(
            (1.45 + 0.08j, 3.4 + 1.7j), (2.4, 3.2), (1.22, 1.44)):
        ref = T._raw(T._box(0.0, 6, nsub, nsup, theta, 2, 10.5))
        for d in (3e-3, 1e-3, 3e-4):
            st = T._box(d, 6, nsub, nsup, theta, 2, 10.5)
            cur = T._raw(st)
            e = T._move(cur, ref, pol=1)
            n_rows += 1
            arb = bool(T._screen(st) is not None
                       and cur[3] - 1.0 > ps._SLIVER_TRIGGER_BAR)
            n_arb += int(arb)
            rows.append(dict(err_d=e / d, kind=T._kind(e, d), arbitrated=arb,
                             worst=cur[3]))
    # the wider correct sample the bar test uses (three angles)
    for nsub, nsup, theta in itertools.product(
            (1.45 + 0.08j, 3.4 + 1.7j), (2.4, 3.2), (1.22, 1.33, 1.44)):
        ref = T._raw(T._box(0.0, 6, nsub, nsup, theta, 2, 10.5))
        for d in (3e-3, 1e-3, 3e-4):
            st = T._box(d, 6, nsub, nsup, theta, 2, 10.5)
            cur = T._raw(st)
            if (T._screen(st) is None
                    or cur[3] - 1.0 <= ps._SLIVER_TRIGGER_BAR):
                continue
            if T._kind(T._move(cur, ref, pol=1), d) != "right":
                continue
            snapped, _hit = T._snapped(st, T._box, d, 6, nsub, nsup, theta,
                                       2, 10.5)
            corr.append(T._drop(cur[3], max(snapped[3] - 1.0, 0.0)))
    d5 = []
    ref5 = T._raw(T._gmr(0.0))
    for delta in T._D5_DELTAS:
        st = T._gmr(delta)
        cur = T._raw(st)
        snapped, _hit = T._snapped(st, T._gmr, delta)
        if T._move(cur, ref5, pol=1) / delta <= 100.0:
            continue
        d5.append(T._drop(cur[3], max(snapped[3] - 1.0, 0.0)))
    bar = 1.0 / ps._SLIVER_CLOSURE_FRACTION
    return dict(n_rows=n_rows, n_arb=n_arb, n_arb_margin=n_arb / 8.0,
                all_right=all(r["kind"] == "right" for r in rows),
                n_corr=len(corr), n_d5=len(d5),
                corr_max=max(corr), corr_min=min(corr),
                corr_margin_under_bar=bar / max(corr),
                d5_min=min(d5), d5_max=max(d5),
                d5_margin_over_bar=min(d5) / bar,
                separation=min(d5) / max(corr),
                separation_margin_over_10=min(d5) / max(corr) / 10.0)


def round2_arm():
    """Test (c): the round-2 fixtures under the relative closure."""
    sliver, trunc = [], []
    for deg, d in ((14, 1e-4), (14, 3e-5), (12, 3e-5), (20, 3e-5),
                   (16, 1e-5)):
        st = T._o11(d, deg)
        cur = T._raw(st)
        v, ev = ps._sliver_arbiter(st, cur[3], cur[1], cur[2], None)
        sliver.append(dict(deg=deg, delta=d, verdict=v,
                           su_snap=ev["snapped_super_unity"],
                           move_w=ev["move"] / ev["w_wide"], drop=ev["drop"]))
    for nsub, th, deg, d in ((1.5 + 0.05j, 1.2, 6, 1e-3),
                             (3.0 + 2.0j, 1.45, 6, 1e-3),
                             (1.5 + 0.05j, 1.3, 8, 3e-4)):
        st = T._o11(d, deg, n_sup=2.5, n_sub=nsub, eps=12.0, theta=th,
                    ffo=31)
        cur = T._raw(st)
        got = ps._sliver_arbiter(st, cur[3], cur[1], cur[2], None)
        if got is None or cur[3] <= 1.0 + ps._SLIVER_TRIGGER_BAR:
            continue
        v, ev = got
        trunc.append(dict(nsub=str(nsub), theta=th, deg=deg, delta=d,
                          verdict=v, su_snap=ev["snapped_super_unity"],
                          move_w=ev["move"] / ev["w_wide"], drop=ev["drop"]))
    su_max = max(s["su_snap"] for s in sliver)
    return dict(
        sliver=sliver, trunc=trunc,
        sliver_su_max=su_max,
        sliver_su_margin_under_1e6=(ps._SLIVER_ATTRIB_CLOSURE / 10.0) / su_max
        if su_max > 0 else float("inf"),
        sliver_drop_min=min(s["drop"] for s in sliver),
        sliver_move_min=min(s["move_w"] for s in sliver),
        sliver_move_margin=min(s["move_w"] for s in sliver)
        / ps._SLIVER_MOVE_FACTOR,
        trunc_move_max=max(t["move_w"] for t in trunc),
        trunc_move_margin=ps._SLIVER_MOVE_FACTOR
        / max(t["move_w"] for t in trunc),
        trunc_su_min=min(t["su_snap"] for t in trunc),
        trunc_drop_max=max(t["drop"] for t in trunc),
        separation_move=min(s["move_w"] for s in sliver)
        / max(t["move_w"] for t in trunc))


def main():
    dest = sys.argv[1] if len(sys.argv) > 1 else "v5_testbars.json"
    t0 = time.perf_counter()
    doc = dict(build=F.build_info(), d5=d5_arm(), box=box_arm(),
               round2=round2_arm())
    doc["wall"] = time.perf_counter() - t0
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(F.jsonable(doc), fh, indent=1)
    print(json.dumps(F.jsonable({k: v for k, v in doc.items()
                                 if k != "round2"}), indent=1)[:4000])
    r = doc["round2"]
    print({k: v for k, v in r.items() if not isinstance(v, list)})
    print(f"-> {dest}")


if __name__ == "__main__":
    main()
