"""V8 -- a DIRECTED false-refusal attack on the box's own worst offenders.

`v2_closure.py` scans a 2,304-row grid on a fixed ladder of four wall steps.
This probe takes the mounts that came CLOSEST to a false refusal on that grid
-- the three whose CORRECT rows move furthest past the move bar, and the three
whose CORRECT rows drop furthest -- and attacks them directly: a fine
logarithmic sweep of the wall step at their own degree and at two neighbouring
degrees, which is where the two arms are most likely to be met at once.

The quantity reported is the JOINT approach
``min(drop / 100, move_w / 100)`` over the CORRECT rows: 1.0 or above IS a
false refusal, and the largest value below 1.0 is how close the attack came.
The arms are anti-correlated on this family -- a wall step small enough to
inflate the drop has already destroyed the answer, and a wall step large
enough to leave the answer correct leaves the drop near unity -- so the
interesting output is where that trade-off bottoms out.

    python v8_attack.py out.json
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import v_fixtures as F  # noqa: E402

#: The six mounts, read off `v2_closure_win.json`'s own summary (the three
#: `correct_top_move` mounts and the three `correct_top_drop` mounts).  They
#: are named here as explicit configurations so this probe does not depend on
#: that JSON being present.
MOUNTS = (
    ("move1", dict(period=1.35e-6, wl=1.064e-6, nsup=2.05,
                   nsub=complex(1.52, 0.03), theta=1.18, degree=6,
                   e_hi=6.76, nl=3)),
    ("move2", dict(period=1.35e-6, wl=0.633e-6, nsup=2.05,
                   nsub=complex(1.52, 0.03), theta=1.18, degree=8,
                   e_hi=12.25, nl=3)),
    ("move3", dict(period=1.02e-6, wl=0.633e-6, nsup=2.05,
                   nsub=complex(1.52, 0.03), theta=1.18, degree=6,
                   e_hi=12.25, nl=3)),
    ("drop1", dict(period=1.02e-6, wl=0.633e-6, nsup=3.10,
                   nsub=complex(3.45, 0.90), theta=1.35, degree=6,
                   e_hi=12.25, nl=2)),
    ("drop2", dict(period=1.02e-6, wl=0.633e-6, nsup=3.10,
                   nsub=complex(2.90, 1.10), theta=1.35, degree=6,
                   e_hi=12.25, nl=3)),
    ("drop3", dict(period=1.35e-6, wl=0.633e-6, nsup=3.10,
                   nsub=complex(3.45, 0.90), theta=1.35, degree=6,
                   e_hi=6.76, nl=2)),
)


def scan(name, cfg, deltas):
    try:
        ref = F.unguarded(F.box_stack(0.0, **cfg))
    except (ValueError, NotImplementedError):
        return []
    rows = []
    for d in deltas:
        d = float(d)
        try:
            st = F.box_stack(d, **cfg)
            cur = F.unguarded(st)
        except (ValueError, NotImplementedError):
            continue
        pre = F.prescribed(st)
        if pre is None:
            continue
        sn = F.snapped(st, pre["mf"])
        su = max(sn["worst"] - 1.0, 0.0)
        mv = F.move_shared(cur, sn)
        e = F.move_shared(cur, ref, pol=1)
        es = F.move_shared(sn, ref, pol=1)
        rows.append(dict(mount=name, delta=d, degree=cfg["degree"],
                         worst=cur["worst"], su_snap=su,
                         drop=F.drop(cur["worst"], su),
                         move_w=(None if mv is None else mv / pre["w_wide"]),
                         err_d=e / d, kind=F.classify(e, d),
                         err_snap_d=es / d, kind_snap=F.classify(es, d),
                         arbitrated=bool(cur["worst"] - 1.0
                                         > F.ps._SLIVER_TRIGGER_BAR)))
    return rows


def main():
    dest = sys.argv[1] if len(sys.argv) > 1 else "v8_attack.json"
    t0 = time.perf_counter()
    deltas = np.logspace(-2.2, -5.6, 30)
    rows = []
    for name, cfg in MOUNTS:
        for dd in (-2, 0):
            c = dict(cfg)
            c["degree"] = max(4, cfg["degree"] + dd)
            rows += scan(f"{name}_deg{c['degree']}", c, deltas)
    arb = [r for r in rows if r["arbitrated"] and r["move_w"] is not None]
    right = [r for r in arb if r["kind"] == "right"]
    grey = [r for r in arb if r["kind"] == "grey"]

    def approach(seq):
        best, who = 0.0, None
        for r in seq:
            dv = r["drop"] if np.isfinite(r["drop"]) else 1e300
            a = min(dv / 100.0, r["move_w"] / 100.0)
            if a > best:
                best, who = a, r
        return best, who

    a_r, who_r = approach(right)
    a_g, who_g = approach(grey)
    false_ref = [r for r in right
                 if r["move_w"] > 100.0
                 and (not np.isfinite(r["drop"]) or r["drop"] >= 100.0)]
    s = dict(
        rows=len(rows), arbitrated=len(arb), arb_right=len(right),
        arb_grey=len(grey),
        arb_wrong=sum(1 for r in arb if r["kind"] == "wrong"),
        correct_move_w_envelope=max((r["move_w"] for r in right), default=None),
        correct_move_w_over_100=sum(1 for r in right if r["move_w"] > 100.0),
        correct_finite_drop_envelope=max(
            (r["drop"] for r in right if np.isfinite(r["drop"])), default=None),
        correct_infinite_drop=sum(1 for r in right
                                  if not np.isfinite(r["drop"])),
        false_refusals=len(false_ref),
        false_refusal_rows=[F.jsonable(r) for r in false_ref[:10]],
        closest_approach_correct=a_r,
        closest_approach_correct_row=(None if who_r is None
                                      else F.jsonable(who_r)),
        closest_approach_grey=a_g,
        closest_approach_grey_row=(None if who_g is None
                                   else F.jsonable(who_g)),
        wall=time.perf_counter() - t0,
    )
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(F.jsonable(dict(build=F.build_info(), summary=s,
                                  rows=rows)), fh, indent=1)
    for k in sorted(s):
        if not isinstance(s[k], (list, dict)):
            print(f"{k:34s} {s[k]}")
    print("closest correct row:", s["closest_approach_correct_row"])
    print(f"-> {dest}")


if __name__ == "__main__":
    main()
