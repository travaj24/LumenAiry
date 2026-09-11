"""GAP 2 -- LADDER D, the decision that did not move in round 2, before and
after the round-3 change, plus its healthy counter-population.

Ladder D is the round-2 verification's own fixture shape: a damaging ring stack
between two ``eps = 2`` half-spaces with the loss placed on the INCIDENCE
half-space ALONE.  Round 2 reads 4 of 13 refused before and 4 of 13 after --
nothing moved -- while the div-conforming twin on the identical geometry
measures the legitimate flux budget at 8.75e-07 against a returned 2.41297.

LADDER H is the counter-population: HEALTHY stacks carrying the same loss on
the same layer.  A screen that refuses ladder D must refuse NONE of these.

WHAT "HEALTHY" HAS TO MEAN HERE, AND WHY IT IS SCREENED RATHER THAN ASSUMED.
An accurate ENERGY is not health: the round-2 census's own headline row
(``uniform``, ``m = 1``, ``N = 200``, ``rbl = 2``) closes ``R + T`` to 1 and
returns 12 channels where its div-conforming twin returns 10.  The ceiling
refuses it -- correctly, and already on the round-2 tree.  So ladder H's
geometries are SCREENED first: a candidate joins it only if, at zero loss, the
nodal cascade returns the SAME (incidence, exit) channel counts as its
staggered twin.  That is the channel-SET definition of damage the round-2 work
established, and it is the only definition under which "false refusal" means
anything.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _g3  # noqa: E402

K0 = 2.0
#: the 13 rungs, the round-2 verification's own ladder
RUNGS = (0.0, 1e-14, 1e-13, 1e-12, 3e-12, 1e-11, 1e-9, 1e-6,
         1e-4, 1e-3, 1e-2, 5e-2, 1e-1)


def _nodal_stack(im_sup, m=1, N=200, rbl=2.0, basis="nodal", family="ring"):
    return _g3.stack(basis, family, m, N, rbl, im_sup, "inc", K0)


def ladder(family, m, N, rbl, label):
    rows = []
    for im in RUNGS:
        lay = _nodal_stack(im, m, N, rbl, "nodal", family)
        twin = _nodal_stack(im, m, N, rbl, "staggered", family)
        raw = _g3.disarmed(lay, K0)
        stg = _g3.disarmed(twin, K0)
        en = np.asarray(raw["energy"], float)
        et = np.asarray(stg["energy"], float)
        v, det, nw = _g3.armed(lay, K0)
        ei, ex = _g3.ceiling_excess(lay, raw, K0)
        rows.append(dict(
            im=im, verdict=v, detector=det, warnings=nw,
            n_channels=int(en.size),
            nodal_max=float(np.max(en)) if en.size else None,
            nodal_min=float(np.min(en)) if en.size else None,
            twin_excess=(float(np.max(np.abs(et - 1.0))) if et.size else None),
            exc_inc=ei, exc_exit=ex))
        print("  %-8s im=%-8.3g n=%-3d nodal_max=%-12s twin=%-11s "
              "exc_inc=%-12s exc_exit=%-12s -> %s %s"
              % (label, im, rows[-1]["n_channels"],
                 "%.6g" % rows[-1]["nodal_max"] if en.size else "-",
                 "%.4e" % rows[-1]["twin_excess"] if et.size else "-",
                 "%.4e" % ei if ei is not None else "-",
                 "%.4e" % ex if ex is not None else "-", v, det))
    refused = sum(r["verdict"] == "REFUSED" for r in rows)
    return dict(label=label, family=family, m=m, N=N, rbl=rbl, rows=rows,
                refused=refused, n=len(rows))


#: candidates for ladder H -- screened, not assumed
H_CANDIDATES = tuple(
    (family, m, N, rbl)
    for family in ("uniform", "ring", "seg")
    for m in (0, 1, 2, 3)
    for N in (120, 200)
    for rbl in (0.5, 1.0, 2.0))


def is_healthy(family, m, N, rbl):
    """``True`` when, at ZERO loss, the nodal cascade is undamaged by EVERY
    definition the screen uses -- so a refusal of it at any rung of the loss
    ladder would be a FALSE one.

    ALL THREE CONJUNCTS ARE NEEDED, and the third was learned by measuring:
    screening on the channel SET alone admitted ``ring, m=1, N=200, rbl=0.5``,
    whose set MATCHES its staggered twin and whose nodal cascade nevertheless
    returns ``max(R + T) = 3.23828`` against a twin closing to 1.1e-12.  The
    energy screen refuses that correctly, and already did so before round 3.
    A "healthy" population defined by one detector is not a counter-population
    for the other.
    """
    import lumenairy.elements.bor.bor_solve as bs
    ln = _g3.stack("nodal", family, m, N, rbl, 0.0, "inc", K0)
    ls = _g3.stack("staggered", family, m, N, rbl, 0.0, "inc", K0)
    rn, rs = _g3.disarmed(ln, K0), _g3.disarmed(ls, K0)
    cn = (int(np.size(rn["inc"])), int(np.size(rn["out"])))
    cs = (int(np.size(rs["inc"])), int(np.size(rs["out"])))
    ei, ex = _g3.ceiling_excess(ln, rn, K0)
    sl = float(bs._BOR_INDEX_CEILING_SLACK)
    quiet = all((v is None or v <= sl) for v in (ei, ex))
    en = np.asarray(rn["energy"], float)
    closure = float(np.max(np.abs(en - 1.0))) if en.size else 0.0
    closes = closure <= float(bs._BOR_NODAL_SUPERUNITY_WARN)
    return (bool(cn == cs and quiet and closes and en.size),
            cn, cs, ei, ex, closure)


def main():
    a = _g3.arm()
    print("ARM", a["build"], a["loaded_kernel"], "t%s" % a["threads"],
          a["lumenairy_file"])
    t0 = time.time()
    D = ladder("ring", 1, 200, 2.0, "D(ring)")
    print("SCREENING ladder H candidates")
    healthy, screened = [], []
    for family, m, N, rbl in H_CANDIDATES:
        ok, cn, cs, ei, ex, closure = is_healthy(family, m, N, rbl)
        screened.append(dict(family=family, m=m, N=N, rbl=rbl, healthy=ok,
                             count_nodal=list(cn), count_stag=list(cs),
                             exc_inc=ei, exc_exit=ex, closure=closure))
        if ok:
            healthy.append((family, m, N, rbl))
        print("   %-8s m=%d N=%-4d rbl=%-4g nodal=%s stag=%s closure=%.3e -> %s"
              % (family, m, N, rbl, cn, cs, closure,
                 "HEALTHY" if ok else "damaged"))
    # STRIDE through the healthy set rather than taking the first few, so the
    # ladder is not four variants of one geometry: the candidate list is
    # family-major, so the first N entries would all be `uniform, m=0`.
    n_want = 4
    step = max(1, len(healthy) // n_want)
    chosen = healthy[::step][:n_want] if healthy else []
    print("screened %d candidates, %d healthy, running the ladder on %d"
          % (len(screened), len(healthy), len(chosen)))
    H = [ladder(f, m, N, rbl, "H(%s,m=%d,N=%d,rbl=%g)" % (f, m, N, rbl))
         for (f, m, N, rbl) in chosen]
    healthy_rows = sum(h["n"] for h in H)
    healthy_refused = sum(h["refused"] for h in H)
    summary = dict(
        ladder_D=dict(refused=D["refused"], n=D["n"],
                      by_detector={d: sum(r["detector"] == d for r in D["rows"])
                                   for d in ("energy", "ceiling")},
                      rows_with_channels=sum(bool(r["n_channels"])
                                             for r in D["rows"]),
                      refused_where_channels=sum(
                          1 for r in D["rows"]
                          if r["n_channels"] and r["verdict"] == "REFUSED")),
        ladder_H=dict(rows=healthy_rows, refused=healthy_refused,
                      refused_by={d: sum(r["detector"] == d
                                         for h in H for r in h["rows"])
                                  for d in ("energy", "ceiling")},
                      n_healthy_candidates=len(healthy),
                      geometries=[h["label"] for h in H],
                      screened=len(screened),
                      warnings=sum(r["warnings"] for h in H
                                   for r in h["rows"])),
        seconds=time.time() - t0,
    )
    for k in sorted(summary):
        print(" ", k, summary[k])
    _g3.dump("g2_ladder_d",
             dict(D=D, H=H, H_screen=screened, summary=summary), a)


if __name__ == "__main__":
    main()
