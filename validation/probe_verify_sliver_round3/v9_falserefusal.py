"""V9 -- the FALSE REFUSAL v8 found, taken apart end to end.

`v8_attack.py` found three rows on one mount that meet BOTH arbiter arms while
the campaign's continuity rule calls the answer CORRECT.  This probe asks the
five questions that decide what that means:

1. **Does the LIBRARY actually refuse them?**  v8 scored the criteria
   analytically; here the guarded solve is run and the message captured.
2. **Would ROUND 2 have refused them too?**  Their snapped super-unity is
   exactly 0, so the round-2 ABSOLUTE closure admits them as well -- if the
   library refuses, this is an inherited false positive and not a round-3
   regression.  The BEFORE tree is asked directly by re-running this probe
   there.
3. **Is the answer CORRECT in the continuity sense** -- does it track the
   exact `delta -> 0` solve of the same mount at the same degree?
4. **Is the answer ACCURATE** -- how far is that degree from a converged one?
   A row can satisfy the continuity rule and still be a badly under-resolved
   answer, in which case refusing it is defensible even though the stated
   ATTRIBUTION is wrong.
5. **What does the refusal TELL the caller?**  The remedy it names first is
   `min_feature`; if the real limitation is the degree, the message is
   misdirecting.

    python v9_falserefusal.py out.json
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import v_fixtures as F  # noqa: E402

CFG = dict(period=1.02e-6, wl=0.633e-6, nsup=3.10, nsub=complex(2.90, 1.10),
           theta=1.35, e_hi=12.25, nl=3)
DELTAS = (1.6622244079925e-05, 1.2689610031679234e-05,
          7.395465531108587e-06)
DEGREE = 4
CONVERGED = 16


def main():
    dest = sys.argv[1] if len(sys.argv) > 1 else "v9_falserefusal.json"
    t0 = time.perf_counter()
    out = dict(build=F.build_info(), cfg=F.jsonable(CFG), degree=DEGREE)

    # (4) how far the working degree is from a converged one, on the
    #     SLIVER-FREE device -- the absolute truncation error
    ladder = {}
    for g in (4, 6, 8, 10, 12, CONVERGED):
        try:
            ladder[g] = F.unguarded(F.box_stack(0.0, degree=g, **CFG))
        except (ValueError, NotImplementedError) as exc:
            ladder[g] = None
            out.setdefault("ladder_errors", {})[g] = str(exc)[:100]
    conv = ladder[CONVERGED]
    out["ladder"] = {
        str(g): dict(worst=r["worst"],
                     abs_err_vs_converged=(
                         None if (r is None or conv is None or g == CONVERGED)
                         else F.move_shared(r, conv, pol=1)))
        for g, r in ladder.items() if r is not None}

    ref = ladder[DEGREE]
    rows = []
    for d in DELTAS:
        st = F.box_stack(d, degree=DEGREE, **CFG)
        cur = F.unguarded(st)
        pre = F.prescribed(st)
        sn = F.snapped(st, pre["mf"])
        su = max(sn["worst"] - 1.0, 0.0)
        mv = F.move_shared(cur, sn)
        e = F.move_shared(cur, ref, pol=1)
        es = F.move_shared(sn, ref, pol=1)
        # (1) what the library does
        refused, msg, _o, warns = F.guarded(
            F.box_stack(d, degree=DEGREE, **CFG))
        # (2) would round 2 have attributed it?
        r2 = F.v_round2(cur["worst"], su, mv, pre["w_wide"])
        r3 = F.v_round3(cur["worst"], su, mv, pre["w_wide"], 1.0e-2)
        # (4) the absolute distance of BOTH answers from the converged solve
        abs_cur = (None if conv is None
                   else F.move_shared(cur, conv, pol=1))
        abs_snap = (None if conv is None
                    else F.move_shared(sn, conv, pol=1))
        abs_ref = (None if conv is None
                   else F.move_shared(ref, conv, pol=1))
        rows.append(dict(
            delta=d, worst=cur["worst"], su_snap=su,
            drop=F.drop(cur["worst"], su),
            move=mv, w_wide=pre["w_wide"], move_w=mv / pre["w_wide"],
            err_d=e / d, kind=F.classify(e, d),
            err_snap_d=es / d, kind_snap=F.classify(es, d),
            lib_refused=refused, n_warns=len(warns),
            verdict_round2=r2, verdict_round3=r3,
            abs_err_returned=abs_cur, abs_err_snapped=abs_snap,
            abs_err_reference=abs_ref,
            snapped_closer_to_truth=(None if (abs_cur is None
                                              or abs_snap is None)
                                     else bool(abs_snap < abs_cur)),
            msg_head=(msg or "")[:160],
            msg_names_min_feature=("min_feature=" in (msg or "")),
            msg_attribution=("ATTRIBUTION, MEASURED ON THIS CALL"
                             in (msg or "")),
        ))
    out["rows"] = rows
    out["library_refusals"] = sum(1 for r in rows if r["lib_refused"])
    out["round2_would_attribute"] = sum(1 for r in rows
                                        if r["verdict_round2"] == "sliver")
    out["round3_attributes"] = sum(1 for r in rows
                                   if r["verdict_round3"] == "sliver")
    out["all_correct_by_continuity"] = all(r["kind"] == "right" for r in rows)
    out["snapped_closer_count"] = sum(1 for r in rows
                                      if r["snapped_closer_to_truth"])
    out["wall"] = time.perf_counter() - t0
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(F.jsonable(out), fh, indent=1)
    print(json.dumps(F.jsonable({k: v for k, v in out.items()
                                 if k not in ("rows",)}), indent=1))
    for r in rows:
        print({k: r[k] for k in ("delta", "worst", "su_snap", "move_w",
                                 "err_d", "kind", "lib_refused",
                                 "verdict_round2", "verdict_round3",
                                 "abs_err_returned", "abs_err_snapped",
                                 "abs_err_reference",
                                 "snapped_closer_to_truth")})
    print(f"-> {dest}")


if __name__ == "__main__":
    main()
