"""V2 -- the CLOSURE constant re-sized on the verifier's own box (task 2).

The box is :data:`v_fixtures.BOX`: 3 periods x 2 wavelengths x 2 superstrate
indices x 3 lossy substrates x 2 grazing angles x 2 degrees x 2 ridge
permittivities x 2 slice counts = **576 mounts**, each at 4 wall steps
(3e-3 / 1e-3 / 3e-4 / 1e-4 of a period) = **2,304 rows**.  Every axis value
differs from the fix's 576-configuration box, and the wall pair, the ridge
height permittivity and the slice thickness differ too, so what comes out is
an independent sample of the same populations rather than a re-reading.

Per row the probe records, all with the guard DISARMED so nothing the library
decides can feed back into the measurement:

* the unguarded solve and its super-unity;
* the widest manufactured cell, the ``min_feature`` the refusal would
  prescribe, and the UNGUARDED solve on that grid;
* the DROP factor ``(worst - 1) / su_snapped`` and ``move / w_wide``;
* the continuity classification against the exact ``delta -> 0`` solve of the
  same mount;
* what the LIBRARY actually did with the row (a full guarded solve), so the
  analytic verdicts can be checked against the decision rather than assumed.

The summary then answers the fix's own sizing question on this box: how many
CORRECT rows does the closure admit at the round-2 ABSOLUTE bar, and at
1e-1 / 3e-2 / 1e-2 / 3e-3 / 1e-3; what is the CORRECT population's FINITE
drop envelope and the margin the shipped 100x carries over it; and is there
any CORRECT row that both drops past 100x and moves past 100 widest cells --
a FALSE REFUSAL, which the fix claims is empty over 1,369 rows.

    python v2_closure.py out.json [--fast]
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import v_fixtures as F  # noqa: E402

FRACS = (1.0e-1, 3.0e-2, 1.0e-2, 3.0e-3, 1.0e-3)


def scan(fast=False):
    cfgs = F.box_configs()
    deltas = F.BOX_DELTAS
    if fast:
        cfgs = cfgs[::48]
        deltas = deltas[:2]
    rows, skipped = [], []
    for i, cfg in enumerate(cfgs):
        try:
            ref = F.unguarded(F.box_stack(0.0, **cfg))
        except (ValueError, NotImplementedError) as exc:
            # a mount the solver itself refuses (too few global nodes for the
            # propagating orders) is a hole in the population, recorded rather
            # than silently dropped
            skipped.append(dict(i=i, why=str(exc)[:120],
                                **{k: F.jsonable(v) for k, v in cfg.items()}))
            continue
        for d in deltas:
            st = F.box_stack(d, **cfg)
            cur = F.unguarded(st)
            pre = F.prescribed(st)
            e = F.move_shared(cur, ref, pol=1)
            rec = dict(i=i, delta=d, worst=cur["worst"], err=e, err_d=e / d,
                       kind=F.classify(e, d), screen=pre is not None,
                       ref_worst=ref["worst"],
                       **{k: F.jsonable(v) for k, v in cfg.items()})
            if pre is not None:
                sn = F.snapped(st, pre["mf"])
                su = max(sn["worst"] - 1.0, 0.0)
                mv = F.move_shared(cur, sn)
                es = F.move_shared(sn, ref, pol=1)
                rec.update(w_wide=pre["w_wide"], mf_fix=pre["mf"],
                           n_hit=pre["n_hit"], su_snap=su,
                           drop=F.drop(cur["worst"], su), move=mv,
                           move_w=(None if mv is None else mv / pre["w_wide"]),
                           err_snap_d=es / d, kind_snap=F.classify(es, d))
            rec["arbitrated"] = bool(
                rec["screen"] and cur["worst"] - 1.0 > F.ps._SLIVER_TRIGGER_BAR)
            refused, msg, _out, warns = F.guarded(F.box_stack(d, **cfg))
            rec["lib_refused"] = refused
            rec["lib_note"] = any("is NOT what moved this answer" in w
                                  for w in warns)
            rows.append(rec)
    return rows, skipped


def envelope(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return (max(v) if v else None), (min(v) if v else None), len(v)


def summarise(rows):
    arb = [r for r in rows if r["arbitrated"]]
    right = [r for r in arb if r["kind"] == "right"]
    grey = [r for r in arb if r["kind"] == "grey"]
    wrong = [r for r in arb if r["kind"] == "wrong"]
    fin = [r for r in right if r.get("drop") is not None
           and np.isfinite(r["drop"])]
    inf = [r for r in right if r.get("drop") is not None
           and not np.isfinite(r["drop"])]
    s = dict(
        rows=len(rows), arbitrated=len(arb),
        by_kind={k: sum(1 for r in rows if r["kind"] == k)
                 for k in ("right", "grey", "wrong")},
        arb_right=len(right), arb_grey=len(grey), arb_wrong=len(wrong),
        arb_right_finite_drop=len(fin), arb_right_infinite_drop=len(inf),
    )
    s["correct_finite_drop_envelope"] = envelope([r["drop"] for r in fin])[0]
    s["correct_finite_drop_floor"] = envelope([r["drop"] for r in fin])[1]
    s["correct_move_w_envelope"] = envelope([r.get("move_w")
                                             for r in right])[0]
    s["grey_move_w_envelope"] = envelope([r.get("move_w") for r in grey])[0]
    s["grey_finite_drop_envelope"] = envelope([r.get("drop")
                                               for r in grey])[0]
    # the round-2 ABSOLUTE closure, then the relative ladder
    adm2 = [r for r in right if r["su_snap"] <= 1.0e-5]
    s["correct_admitted_round2_absolute"] = len(adm2)
    s["correct_admitted_round2_all_infinite"] = all(
        not np.isfinite(r["drop"]) for r in adm2)
    s["correct_move_w_envelope_among_admitted"] = envelope(
        [r.get("move_w") for r in adm2])[0]
    for f in FRACS:
        adm = [r for r in right if F.closure_admits(r["worst"], r["su_snap"],
                                                    f)]
        att = [r for r in adm
               if r["move_w"] is not None and r["move_w"] > 100.0]
        s[f"correct_admitted_{f:g}"] = len(adm)
        s[f"correct_admitted_new_vs_round2_{f:g}"] = len(adm) - len(adm2)
        s[f"correct_attributed_{f:g}"] = len(att)
        s[f"correct_attributed_rows_{f:g}"] = [
            dict(i=r["i"], delta=r["delta"], drop=F.jsonable(r["drop"]),
                 move_w=r["move_w"], err_d=r["err_d"]) for r in att]
        s[f"correct_move_w_envelope_among_admitted_{f:g}"] = envelope(
            [r.get("move_w") for r in adm])[0]
        # the same on GREY rows, which the campaign does not call correct but
        # which no one would want refused either
        adm_g = [r for r in grey if F.closure_admits(r["worst"], r["su_snap"],
                                                     f)]
        s[f"grey_attributed_{f:g}"] = sum(
            1 for r in adm_g if r["move_w"] is not None and r["move_w"] > 100)
    # the row that would be newly admitted at the next coarser fraction
    newly = sorted([r for r in right
                    if F.closure_admits(r["worst"], r["su_snap"], 3.0e-2)
                    and not F.closure_admits(r["worst"], r["su_snap"], 1.0e-2)],
                   key=lambda r: -(r["drop"] if np.isfinite(r["drop"])
                                   else 0.0))
    s["correct_newly_admitted_at_3e-2"] = [
        dict(i=r["i"], delta=r["delta"], drop=F.jsonable(r["drop"]),
             move_w=r["move_w"], su_snap=r["su_snap"], worst=r["worst"],
             err_d=r["err_d"]) for r in newly[:8]]
    # the closest approach on both arms, among CORRECT rows
    top_drop = sorted(fin, key=lambda r: -r["drop"])[:5]
    top_move = sorted([r for r in right if r.get("move_w") is not None],
                      key=lambda r: -r["move_w"])[:5]
    s["correct_top_drop"] = [
        dict(i=r["i"], delta=r["delta"], drop=r["drop"], move_w=r["move_w"],
             err_d=r["err_d"], worst=r["worst"], su_snap=r["su_snap"],
             period=r["period"], wl=r["wl"], nsup=r["nsup"], nsub=r["nsub"],
             theta=r["theta"], degree=r["degree"], e_hi=r["e_hi"],
             nl=r["nl"]) for r in top_drop]
    s["correct_top_move"] = [
        dict(i=r["i"], delta=r["delta"], drop=F.jsonable(r["drop"]),
             move_w=r["move_w"], err_d=r["err_d"], worst=r["worst"],
             su_snap=r["su_snap"], period=r["period"], wl=r["wl"],
             nsup=r["nsup"], nsub=r["nsub"], theta=r["theta"],
             degree=r["degree"], e_hi=r["e_hi"], nl=r["nl"])
        for r in top_move]
    # what the LIBRARY did
    s["lib_refused"] = sum(1 for r in rows if r["lib_refused"])
    s["lib_false_positives"] = sum(1 for r in rows
                                   if r["lib_refused"] and r["kind"] == "right")
    s["lib_false_positive_rows"] = [
        dict(i=r["i"], delta=r["delta"], err_d=r["err_d"],
             drop=F.jsonable(r.get("drop")), move_w=r.get("move_w"))
        for r in rows if r["lib_refused"] and r["kind"] == "right"][:20]
    s["lib_refused_grey"] = sum(1 for r in rows
                                if r["lib_refused"] and r["kind"] == "grey")
    s["lib_truncation_notes"] = sum(1 for r in rows if r["lib_note"])
    # the analytic round-3 verdict vs the library, on arbitrated rows
    agree = sum(1 for r in arb
                if (F.v_round3(r["worst"], r["su_snap"], r["move"],
                               r["w_wide"], 1.0e-2) == "sliver")
                == r["lib_refused"])
    s["analytic_round3_matches_library"] = agree
    s["analytic_round3_rows"] = len(arb)
    # returned rows that are WRONG (the false-negative side of this box)
    s["lib_returned_wrong"] = sum(1 for r in rows
                                  if not r["lib_refused"]
                                  and r["kind"] == "wrong")
    return s


def main():
    fast = "--fast" in sys.argv
    dest = ([a for a in sys.argv[1:] if not a.startswith("--")] or
            ["v2_closure.json"])[0]
    t0 = time.perf_counter()
    rows, skipped = scan(fast)
    s = summarise(rows)
    s["skipped_mounts"] = len(skipped)
    s["wall"] = time.perf_counter() - t0
    doc = dict(build=F.build_info(), summary=s, skipped=skipped,
               rows=rows)
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(F.jsonable(doc), fh, indent=1)
    for k in sorted(s):
        if not isinstance(s[k], (list, dict)):
            print(f"{k:52s} {s[k]}")
    print(f"-> {dest}")


if __name__ == "__main__":
    main()
