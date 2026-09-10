"""S4 -- the LOWER envelope of the drop factor, as a FAMILY property.

The round-3 closure bar is a bar on the super-unity DROP factor

    drop = (max(R+T) - 1) / (max(R+T) - 1 on the prescribed min_feature grid)

and the one population it MUST clear is the CORRECT one: a solve whose answer
already tracks the exact ``delta -> 0`` limit has, by construction, nothing
for the snap to remove, so its drop is O(1).  The census box of ``s1`` puts
291 arbitrated CORRECT rows at 0.90 .. 2.29 -- but 291 rows of ONE box is a
sample, and TESTING_STANDARDS rule 5 asks for the FAMILY's envelope.

This probe widens that population as far as the arbiter's own precondition
allows.  A row only enters it when the stack manufactures a sliver AND reads
super-unity above ``_SLIVER_TRIGGER_BAR`` AND its answer is still CORRECT by
the campaign's continuity rule, which happens on under-converged
dense-superstrate grazing mounts.  Two families are scanned:

  BOX  a much wider staircase box than ``s1``'s -- three periods, two
       wavelengths, two superstrate indices, three lossy substrates, two
       angles, two degrees, two ridge permittivities, two slice counts and
       four wall steps, i.e. 576 configurations against ``s1``'s 648 with a
       COMPLETELY different spread (``s1``'s box is one period, one wavelength
       and one ridge pair);
  GMR  the D-5 devices themselves at COARSE wall steps, where their answers
       are correct -- the most directly relevant control, because those are
       the mounts whose truncation floor sits above the absolute closure bar.

    python s4_lower_envelope.py [out.json] [--fast]
"""
import itertools
import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from f_fixtures import (  # noqa: E402
    TRIG,
    classify,
    drop_factor,
    gmr,
    prescribed,
    shared_move,
    snapped,
    unguarded,
    wbuild,
)

import lumenairy  # noqa: E402
from lumenairy.elements.pmm import stack as ps  # noqa: E402

FRACS = (1.0e-1, 3.0e-2, 1.0e-2, 3.0e-3, 1.0e-3)


def row_for(build, kw, d, deg, ref):
    st = build(float(d), deg, **kw)
    cur = unguarded(st)
    pre = prescribed(st)
    if pre is None or cur["worst"] - 1.0 <= TRIG:
        return None
    snp = snapped(st, pre["mf"])
    su = max(snp["worst"] - 1.0, 0.0)
    move = shared_move(cur, snp)
    e = shared_move(cur, ref, pol=1)
    return dict(delta=float(d), deg=deg, err=e, err_over_d=e / float(d),
                kind=classify(e, float(d)), worst=cur["worst"], su_snap=su,
                drop=drop_factor(cur["worst"], su), w_wide=pre["w_wide"],
                move=move, move_ratio=move / pre["w_wide"],
                err_snapped_over_d=shared_move(snp, ref, pol=1) / float(d))


def box(fast=False):
    """The wide staircase box."""
    rows, t0, refs = [], time.time(), {}
    periods = (1.2e-6,) if fast else (0.9e-6, 1.2e-6, 1.6e-6)
    wls = (0.85e-6,) if fast else (0.85e-6, 1.05e-6)
    thetas = (1.22,) if fast else (1.22, 1.38)
    degs = (8,) if fast else (6, 8)
    epss = (10.5,) if fast else (8.0, 14.0)
    subs = ((1.45 + 0.08j,) if fast
            else (1.45 + 0.08j, 2.0 + 0.35j, 3.4 + 1.7j))
    deltas = (1e-3,) if fast else (3e-3, 1e-3, 3e-4, 1e-4)
    nls = (2,) if fast else (2, 4)
    for period, wl, nsup, nsub, th, deg, eps, nl in itertools.product(
            periods, wls, (2.4, 3.2), subs, thetas, degs, epss, nls):
        kw = dict(period=period, wl=wl, th=th, a0=0.27865, b0=0.62505,
                  eh=2.25, ep=eps, dz=0.32e-6 / nl, nl=nl, nsub=nsub,
                  nsup=nsup, ffo=31)
        key = (period, wl, nsup, str(nsub), th, deg, eps, nl)
        if key not in refs:
            refs[key] = unguarded(wbuild(0.0, deg, **kw))
        for d in deltas:
            r = row_for(wbuild, kw, d, deg, refs[key])
            if r is not None:
                r.update(family="BOX", period=period, wl=wl, nsup=nsup,
                         nsub=str(nsub), th=th, eps=eps, nl=nl)
                rows.append(r)
    return rows, time.time() - t0


def gmr_box(fast=False):
    """The D-5 devices at COARSE wall steps, where their answers are still
    correct -- the mounts whose truncation floor sits ABOVE the absolute bar."""
    rows, t0, refs = [], time.time(), {}
    mounts = [dict(wl=9.3e-7, nsup=2.4, th=1.22, nsub=complex(1.45, 0.05)),
              dict(wl=9.3e-7, nsup=2.4, th=1.33, nsub=complex(1.45, 0.05)),
              dict(wl=8.6e-7, nsup=2.4, th=1.22, nsub=complex(1.45, 0.05)),
              dict(wl=9.3e-7, nsup=3.2, th=1.22, nsub=complex(2.0, 0.35)),
              dict(wl=1.02e-6, nsup=2.4, th=1.28, nsub=complex(1.45, 0.05),
                   duty=0.42)]
    if fast:
        mounts = mounts[:1]
    degs = (8,) if fast else (6, 8, 10, 12)
    deltas = ((1e-3,) if fast
              else [float(x) for x in np.geomspace(3e-3, 3e-5, 18)])
    for i, m in enumerate(mounts):
        for deg in degs:
            key = (i, deg)
            if key not in refs:
                refs[key] = unguarded(gmr(0.0, deg, **m))
            for d in deltas:
                r = row_for(gmr, m, d, deg, refs[key])
                if r is not None:
                    r.update(family="GMR", dev=f"d5_{i}",
                             **{k: str(v) for k, v in m.items()})
                    rows.append(r)
    return rows, time.time() - t0


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    fast = "--fast" in sys.argv
    out_path = args[0] if args else os.path.join(HERE, "s4_lower.json")
    lib = os.path.abspath(lumenairy.__file__)
    print("lumenairy:", lib)
    brows, bt = box(fast)
    print(f"  BOX arbitrated rows: {len(brows)}, {bt:.1f} s")
    grows, gt = gmr_box(fast)
    print(f"  GMR arbitrated rows: {len(grows)}, {gt:.1f} s")
    rows = brows + grows

    def env(rs, key="drop", how=max):
        vs = [r[key] for r in rs if np.isfinite(r[key])]
        return how(vs) if vs else None

    def pop(kind, fam=None):
        return [r for r in rows if r["kind"] == kind
                and (fam is None or r["family"] == fam)]

    s = dict(lumenairy=lib, python=sys.version.split()[0],
             numpy=np.__version__, box_wall=bt, gmr_wall=gt,
             n_arbitrated=len(rows), n_box=len(brows), n_gmr=len(grows))
    for fam in (None, "BOX", "GMR"):
        tag = fam or "ALL"
        for kind in ("right", "grey", "wrong"):
            sel = pop(kind, fam)
            selm = [r for r in sel
                    if r["move_ratio"] > ps._SLIVER_MOVE_FACTOR]
            s[f"{tag}_{kind}_n"] = len(sel)
            s[f"{tag}_{kind}_drop_max"] = env(sel)
            s[f"{tag}_{kind}_drop_min"] = env(sel, how=min)
            s[f"{tag}_{kind}_pastmove_n"] = len(selm)
            s[f"{tag}_{kind}_pastmove_drop_max"] = env(selm)
    right = pop("right")
    s["right_drop_top10"] = sorted((r["drop"] for r in right
                                    if np.isfinite(r["drop"])),
                                   reverse=True)[:10]
    s["right_worst_row"] = max(right, key=lambda r: r["drop"]) if right else None
    for f in FRACS:
        s[f"right_above_bar_{f:g}"] = sum(
            1 for r in right if r["drop"] >= 1.0 / f)
        s[f"right_above_bar_and_move_{f:g}"] = sum(
            1 for r in right if r["drop"] >= 1.0 / f
            and r["move_ratio"] > ps._SLIVER_MOVE_FACTOR)
    with open(out_path, "w") as fh:
        json.dump(dict(summary=s, rows=rows), fh, indent=1)
    for k, v in s.items():
        if k != "right_worst_row":
            print(f"  {k:30s} {v}")
    if s["right_worst_row"]:
        w = s["right_worst_row"]
        print(f"  worst CORRECT row: drop {w['drop']:.4g}, move/w "
              f"{w['move_ratio']:.4g}, err/delta {w['err_over_d']:.4g}, "
              f"R+T-1 {w['worst'] - 1:.4g}, family {w['family']}")
    print("->", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
