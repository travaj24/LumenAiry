"""V3 -- the D-5 family on the verifier's OWN mounts (task 3).

A **D-5 mount** is one whose SLIVER-FREE truncation super-unity -- the
``max(R+T) - 1`` its own degree leaves at ``delta = 0`` -- sits BETWEEN the
round-2 absolute closure bar (1e-5) and the trigger (1e-3).  On such a mount
the snapped solve can never reach the absolute bar, so round 2's closure
conjunct can never be met however completely the snap restores the answer.

Stage 1 SCREENS a list of candidate mounts by measuring that floor directly
(and the degree ladder around it, so a floor that is a second pathology rather
than ordinary truncation is visible).  Stage 2 scans the mounts that land in
the band over 26 log-spaced wall steps and records, per row: the unguarded
solve, the solve on the prescribed grid, the two arbiter quantities, the DROP
factor, the continuity classification of BOTH the returned and the snapped
answer against the exact ``delta -> 0`` reference, and what the LIBRARY did --
including the text of the ``truncation`` note when one was issued.

Stage 3 answers three questions:

* the RECOVERY rate of the shipped 1e-2 closure over this population, and over
  the whole 1e-1 .. 1e-3 ladder;
* the **R3-A edge** -- rows whose drop is below the demanded 100 that are
  WRONG and RETURNED, i.e. the band no setting of the constant can reach;
* whether the new ``truncation`` note is TRUE on every returned row.  The note
  asserts the sliver "is NOT what moved this answer".  It is FALSE on any row
  whose returned answer is WRONG by continuity while the answer on the
  prescribed grid is RIGHT -- because there the sliver is exactly what moved
  it.  Every returned noted row is scored that way.

    python v3_d5.py out.json [--fast]
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import v_fixtures as F  # noqa: E402

FRACS = (1.0e-1, 3.0e-2, 1.0e-2, 3.0e-3, 1.0e-3)
BAND = (1.0e-5, 1.0e-3)
NOTE = "is NOT what moved this answer"


def candidates():
    """``(name, builder(delta), degree)`` -- the mounts screened for the band.

    Deliberately five different physical mechanisms: a guided-mode resonance,
    a Fabry-Perot cavity, a near-Wood mount, a dense-superstrate grazing
    staircase and a high-index-contrast lossy-substrate staircase."""
    out = []
    for deg in (6, 8, 10):
        out.append((f"gmr_deg{deg}", (lambda d, g=deg: F.vgmr(d, degree=g)),
                    deg))
        out.append((f"gmr_duty38_deg{deg}",
                    (lambda d, g=deg: F.vgmr(d, degree=g, duty=0.38,
                                             wl=0.83e-6)), deg))
        out.append((f"fp_deg{deg}", (lambda d, g=deg: F.vfp(d, degree=g)),
                    deg))
        out.append((f"fp_thick_deg{deg}",
                    (lambda d, g=deg: F.vfp(d, degree=g, t_cav=1.46e-6,
                                            theta=0.81)), deg))
        out.append((f"wood_deg{deg}", (lambda d, g=deg: F.vwood(d, degree=g)),
                    deg))
        out.append((f"wood_m3_deg{deg}",
                    (lambda d, g=deg: F.vwood(d, degree=g, order=-3,
                                              period=2.05e-6, n_sup=2.10)),
                    deg))
        out.append((f"graze_deg{deg}",
                    (lambda d, g=deg: F.vstair(
                        d, period=1.18e-6, wl=0.72e-6, theta=1.31, a0=0.2870,
                        b0=0.7150, e_lo=3.24, e_hi=4.41, dz=0.19e-6, nl=2,
                        degree=g, nsup=2.28, nsub=complex(1.46, 0.06),
                        ffo=15)), deg))
        out.append((f"contrast_deg{deg}",
                    (lambda d, g=deg: F.vstair(
                        d, period=0.94e-6, wl=0.905e-6, theta=1.09,
                        a0=0.2260, b0=0.6810, e_lo=2.10, e_hi=16.0,
                        dz=0.22e-6, nl=2, degree=g, nsup=1.62,
                        nsub=complex(4.10, 1.40), ffo=15)), deg))
    return out


def screen():
    """Measure every candidate's sliver-free floor and keep the in-band
    mounts.  The ladder around the chosen degree is recorded so a floor that
    is NOT ordinary truncation is visible in the JSON."""
    seen, kept = [], []
    for name, build, deg in candidates():
        try:
            r = F.unguarded(build(0.0))
        except (ValueError, NotImplementedError) as exc:
            seen.append(dict(name=name, degree=deg, why=str(exc)[:100]))
            continue
        floor = r["worst"] - 1.0
        rec = dict(name=name, degree=deg, floor=floor,
                   in_band=bool(BAND[0] < floor < BAND[1]))
        seen.append(rec)
        if rec["in_band"]:
            kept.append((name, build, deg))
    return seen, kept


def scan_mount(name, build, deg, deltas):
    ref = F.unguarded(build(0.0))
    rows = []
    for d in deltas:
        try:
            st = build(d)
            cur = F.unguarded(st)
        except (ValueError, NotImplementedError):
            continue
        pre = F.prescribed(st)
        e = F.move_shared(cur, ref, pol=1)
        rec = dict(mount=name, degree=deg, delta=d, worst=cur["worst"],
                   err_d=e / d, kind=F.classify(e, d),
                   screen=pre is not None)
        if pre is not None:
            sn = F.snapped(st, pre["mf"])
            su = max(sn["worst"] - 1.0, 0.0)
            mv = F.move_shared(cur, sn)
            es = F.move_shared(sn, ref, pol=1)
            rec.update(w_wide=pre["w_wide"], su_snap=su,
                       drop=F.drop(cur["worst"], su), move=mv,
                       move_w=(None if mv is None else mv / pre["w_wide"]),
                       err_snap_d=es / d, kind_snap=F.classify(es, d),
                       abs_bar_rejects=bool(su > 1.0e-5))
        rec["arbitrated"] = bool(rec["screen"] and cur["worst"] - 1.0
                                 > F.ps._SLIVER_TRIGGER_BAR)
        refused, msg, _o, warns = F.guarded(build(d))
        rec["lib_refused"] = refused
        note = [w for w in warns if NOTE in w]
        rec["lib_note"] = bool(note)
        rec["note_text"] = (note[0][note[0].find("  A near-coincident"):]
                            if note else None)
        rec["n_warns"] = len(warns)
        rows.append(rec)
    return rows


def summarise(rows):
    arb = [r for r in rows if r["arbitrated"]]
    # THE D-5 POPULATION: the absolute bar rejects it, the answer is WRONG,
    # and the move criterion is met -- i.e. every piece of evidence an
    # attribution needs is present and only the closure's SHAPE stood in the
    # way.
    d5 = [r for r in arb if r.get("abs_bar_rejects")
          and r["kind"] == "wrong"
          and r.get("move_w") is not None and r["move_w"] > 100.0]
    s = dict(rows=len(rows), arbitrated=len(arb), d5_rows=len(d5),
             d5_mounts=sorted({r["mount"] for r in d5}),
             d5_drop_floor=(min(r["drop"] for r in d5) if d5 else None),
             d5_drop_ceiling=(max((r["drop"] for r in d5
                                   if np.isfinite(r["drop"])), default=None)),
             d5_infinite_drop=sum(1 for r in d5 if not np.isfinite(r["drop"])),
             d5_err_d_floor=(min(r["err_d"] for r in d5) if d5 else None),
             d5_err_d_ceiling=(max(r["err_d"] for r in d5) if d5 else None),
             d5_snapped_right=sum(1 for r in d5
                                  if r.get("kind_snap") == "right"))
    for f in FRACS:
        rec = [r for r in d5 if F.closure_admits(r["worst"], r["su_snap"], f)]
        s[f"d5_recovered_{f:g}"] = len(rec)
        left = [r for r in d5 if r not in rec]
        s[f"d5_left_returned_{f:g}"] = len(left)
        s[f"d5_worst_left_err_d_{f:g}"] = (max(r["err_d"] for r in left)
                                           if left else None)
    # R3-A: rows with drop BELOW the demanded 100 that are WRONG and RETURNED
    edge = [r for r in arb if r.get("drop") is not None
            and np.isfinite(r["drop"]) and r["drop"] < 100.0
            and r["kind"] == "wrong" and not r["lib_refused"]]
    s["r3a_edge_rows"] = len(edge)
    s["r3a_edge"] = [dict(mount=r["mount"], degree=r["degree"],
                          delta=r["delta"], drop=r["drop"],
                          err_d=r["err_d"], move_w=r.get("move_w"),
                          kind_snap=r.get("kind_snap"),
                          worst=r["worst"], su_snap=r.get("su_snap"))
                     for r in sorted(edge, key=lambda x: -x["err_d"])[:15]]
    s["r3a_edge_worst_err_d"] = (max(r["err_d"] for r in edge)
                                 if edge else None)
    s["r3a_edge_snapped_right"] = sum(1 for r in edge
                                      if r.get("kind_snap") == "right")
    # THE TRUNCATION NOTE'S TRUTH, scored on every RETURNED noted row
    noted = [r for r in rows if r["lib_note"] and not r["lib_refused"]]
    false_note = [r for r in noted if r["kind"] == "wrong"
                  and r.get("kind_snap") == "right"]
    soft_note = [r for r in noted if r["kind"] != "right"
                 and r.get("kind_snap") == "right" and r not in false_note]
    s["noted_returned_rows"] = len(noted)
    s["noted_false"] = len(false_note)
    s["noted_soft_false"] = len(soft_note)
    s["noted_false_rows"] = [
        dict(mount=r["mount"], degree=r["degree"], delta=r["delta"],
             err_d=r["err_d"], err_snap_d=r["err_snap_d"], drop=r.get("drop"),
             move_w=r.get("move_w"), worst=r["worst"],
             note=(r["note_text"] or "")[:400])
        for r in sorted(false_note, key=lambda x: -x["err_d"])[:10]]
    s["noted_by_kind"] = {k: sum(1 for r in noted if r["kind"] == k)
                          for k in ("right", "grey", "wrong")}
    s["lib_refused"] = sum(1 for r in rows if r["lib_refused"])
    s["lib_refused_right"] = sum(1 for r in rows
                                 if r["lib_refused"] and r["kind"] == "right")
    return s


def main():
    fast = "--fast" in sys.argv
    dest = ([a for a in sys.argv[1:] if not a.startswith("--")]
            or ["v3_d5.json"])[0]
    t0 = time.perf_counter()
    seen, kept = screen()
    deltas = list(np.logspace(-2.5, -6.6, 8 if fast else 30))
    rows = []
    for name, build, deg in (kept[:2] if fast else kept):
        rows += scan_mount(name, build, deg, deltas)
    s = summarise(rows)
    s["screened"] = len(seen)
    s["in_band_mounts"] = [r["name"] for r in seen if r.get("in_band")]
    s["wall"] = time.perf_counter() - t0
    doc = dict(build=F.build_info(), summary=s, screen=seen, rows=rows)
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(F.jsonable(doc), fh, indent=1)
    for k in sorted(s):
        if not isinstance(s[k], (list, dict)):
            print(f"{k:36s} {s[k]}")
    print("in-band mounts:", s["in_band_mounts"])
    print(f"-> {dest}")


if __name__ == "__main__":
    main()
