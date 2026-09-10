"""S3 -- the DROP-factor populations the relative closure has to separate.

The round-3 criterion replaces "the snapped super-unity reaches an ABSOLUTE
floor" with "the snap REMOVES most of the violation":

    su_snapped <= max(_SLIVER_ATTRIB_CLOSURE,
                      (worst - 1) * _SLIVER_CLOSURE_FRACTION)

which, whenever the relative arm binds, is exactly

    drop = (worst - 1) / su_snapped  >=  1 / _SLIVER_CLOSURE_FRACTION.

So the constant is sized on the DROP factor, and this probe measures its two
populations on three families at once, with the guard disarmed throughout:

  STAIR  the five two-slice staircases whose measured continuity slopes span
         0.47 .. 31.4 -- the ordinary sliver family;
  D5     the guided-mode-resonance grating in the dense-superstrate grazing
         mount whose sliver-FREE truncation floor sits ABOVE the absolute
         closure bar (the D-5 class), scanned over mounts, degrees and
         deltas;
  STEEP  the SAME grating on its guided-mode resonance flank, where the
         device's own ``dR/d(duty)`` is 70-170 -- the population whose CORRECT
         rows move past the MOVE bar (verification S5.3), i.e. the only rows
         for which the closure arm is the sole thing standing between a
         correct answer and a refusal.

Every row is classified BOTH ways: by the campaign's absolute continuity rule
(``err <= 10 delta`` RIGHT, ``> 100 delta`` WRONG) and by the
slope-NORMALISED rule the verification used on steep devices (``err <= 3 s
delta`` RIGHT, ``> 30 s delta`` WRONG, with ``s`` the device's own measured
smooth-regime slope).  The report gives the drop envelope of each population
under each rule, plus the flip census: which rows change verdict between
round 2 and round 3, and what kind they are.

    python s3_dropgap.py [out.json] [--fast]
"""
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
    FIXTURES,
    TRIG,
    classify,
    drop_factor,
    gmr,
    prescribed,
    shared_move,
    snapped,
    unguarded,
    verdict_round2,
    verdict_round3,
    wbuild,
)

import lumenairy  # noqa: E402
from lumenairy.elements.pmm import stack as ps  # noqa: E402

FRACS = (1.0e-1, 3.0e-2, 1.0e-2, 3.0e-3, 1.0e-3)

#: the D-5 mounts: dense superstrate, lossy substrate, grazing incidence --
#: the class whose sliver-free truncation floor lands above the absolute bar
D5_MOUNTS = (
    dict(wl=9.3e-7, nsup=2.4, th=1.22, nsub=complex(1.45, 0.05)),
    dict(wl=9.3e-7, nsup=2.4, th=1.33, nsub=complex(1.45, 0.05)),
    dict(wl=8.6e-7, nsup=2.4, th=1.22, nsub=complex(1.45, 0.05)),
    dict(wl=9.3e-7, nsup=3.2, th=1.22, nsub=complex(2.0, 0.35)),
    dict(wl=1.02e-6, nsup=2.4, th=1.28, nsub=complex(1.45, 0.05), duty=0.42),
)
#: the guided-mode-resonance FLANK, where dR/d(duty) is 70-170 (the R2-D
#: device): an ordinary mount, no dense superstrate, so its truncation floor
#: is tiny and only the steepness matters
STEEP_MOUNTS = (
    dict(wl=1.748880e-6, nsup=1.0, th=0.10, nsub=1.45),
    dict(wl=1.748920e-6, nsup=1.0, th=0.10, nsub=1.45),
    dict(wl=1.749000e-6, nsup=1.0, th=0.10, nsub=1.45),
    dict(wl=1.749040e-6, nsup=1.0, th=0.10, nsub=1.45),
)


def _geom(kw):
    """The build kwargs, without the bookkeeping ``tag``."""
    return {k: v for k, v in kw.items() if k != "tag"}


def device_slope(build, deg, kw, deltas=(3e-3, 1e-3, 3e-4)):
    """The device's OWN smooth-regime continuity slope ``s = err / delta``,
    the statistic the verification scored steep devices with."""
    g = _geom(kw)
    ref = unguarded(build(0.0, deg, **g))
    ss = []
    for d in deltas:
        cur = unguarded(build(d, deg, **g))
        ss.append(shared_move(cur, ref, pol=1) / d)
    return float(np.median(ss))


def scan(family, build, kw, degrees, deltas, slope):
    """Every arbitrated row of one device, with both classifications."""
    rows = []
    g = _geom(kw)
    for deg in degrees:
        ref = unguarded(build(0.0, deg, **g))
        for d in deltas:
            st = build(float(d), deg, **g)
            cur = unguarded(st)
            pre = prescribed(st)
            if pre is None or cur["worst"] - 1.0 <= TRIG:
                continue                      # the arbiter never runs here
            snp = snapped(st, pre["mf"])
            su = max(snp["worst"] - 1.0, 0.0)
            move = shared_move(cur, snp)
            e = shared_move(cur, ref, pol=1)
            e_snap = shared_move(snp, ref, pol=1)
            rows.append(dict(
                family=family, dev=kw.get("tag", ""), deg=deg, delta=float(d),
                slope=slope, err=e, err_over_d=e / d,
                err_snapped_over_d=e_snap / d,
                kind_abs=classify(e, float(d)),
                kind_slope=("wrong" if e > 30.0 * slope * d
                            else "right" if e <= 3.0 * slope * d else "grey"),
                worst=cur["worst"], su_snap=su,
                drop=drop_factor(cur["worst"], su),
                w_wide=pre["w_wide"], move=move,
                move_ratio=move / pre["w_wide"],
                v2=verdict_round2(cur["worst"], su, move, pre["w_wide"]),
                v3={f"{f:g}": verdict_round3(cur["worst"], su, move,
                                             pre["w_wide"], f)
                    for f in FRACS}))
    return rows


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    fast = "--fast" in sys.argv
    out_path = args[0] if args else os.path.join(HERE, "s3_dropgap.json")
    lib = os.path.abspath(lumenairy.__file__)
    print("lumenairy:", lib)
    n_d = 12 if fast else 40
    deltas = [float(x) for x in np.geomspace(3e-3, 1e-6, n_d)]
    degrees = (10, 14) if fast else (10, 14, 18)
    t0 = time.time()
    rows = []

    for name, kw in FIXTURES.items():
        t1 = time.time()
        s = device_slope(wbuild, 14, kw)
        rows += scan("STAIR", wbuild, dict(kw, tag=name), degrees, deltas, s)
        print(f"  STAIR {name:8s} slope {s:8.3g}  {len(rows)} rows, "
              f"{time.time() - t1:.1f} s")

    g_degrees = (6, 8, 10) if not fast else (8,)
    g_deltas = [float(x) for x in np.geomspace(3e-5, 1e-6,
                                               10 if fast else 26)]
    for i, m in enumerate(D5_MOUNTS):
        t1 = time.time()
        kw = dict(m, tag=f"d5_{i}")
        s = device_slope(gmr, 8, kw, deltas=(3e-4, 1e-4, 3e-5))
        rows += scan("D5", gmr, kw, g_degrees, g_deltas, s)
        print(f"  D5    {kw['tag']:8s} slope {s:8.3g}  {len(rows)} rows, "
              f"{time.time() - t1:.1f} s")

    s_degrees = (8, 12, 14, 18) if not fast else (12,)
    s_deltas = [float(x) for x in np.geomspace(3e-4, 1e-6,
                                               10 if fast else 24)]
    for i, m in enumerate(STEEP_MOUNTS):
        t1 = time.time()
        kw = dict(m, tag=f"steep_{i}")
        s = device_slope(gmr, 14, kw, deltas=(3e-4, 1e-4, 3e-5))
        rows += scan("STEEP", gmr, kw, s_degrees, s_deltas, s)
        print(f"  STEEP {kw['tag']:8s} slope {s:8.3g}  {len(rows)} rows, "
              f"{time.time() - t1:.1f} s")

    def env(rs, key="drop", how=max):
        vs = [r[key] for r in rs if np.isfinite(r[key])]
        return how(vs) if vs else None

    def pop(pred):
        return [r for r in rows if pred(r)]

    past_move = pop(lambda r: r["move_ratio"] > ps._SLIVER_MOVE_FACTOR)
    summary = dict(
        lumenairy=lib, python=sys.version.split()[0], numpy=np.__version__,
        wall=time.time() - t0, n_rows=len(rows),
        n_by_family={f: len(pop(lambda r, f=f: r["family"] == f))
                     for f in ("STAIR", "D5", "STEEP")},
    )
    for tag, key in (("abs", "kind_abs"), ("slope", "kind_slope")):
        for kind in ("right", "grey", "wrong"):
            sel = pop(lambda r, k=key, c=kind: r[k] == c)
            selm = [r for r in sel if r["move_ratio"] > ps._SLIVER_MOVE_FACTOR]
            summary[f"{tag}_{kind}_n"] = len(sel)
            summary[f"{tag}_{kind}_drop_max"] = env(sel)
            summary[f"{tag}_{kind}_drop_min"] = env(sel, how=min)
            summary[f"{tag}_{kind}_pastmove_n"] = len(selm)
            summary[f"{tag}_{kind}_pastmove_drop_max"] = env(selm)
            summary[f"{tag}_{kind}_pastmove_drop_min"] = env(selm, how=min)
    summary["past_move_n"] = len(past_move)
    # the D-5 class's own floor: WRONG rows on the mounts whose truncation
    # floor is ABOVE the absolute closure bar
    d5w = pop(lambda r: r["family"] == "D5" and r["kind_abs"] == "wrong"
              and r["su_snap"] > ps._SLIVER_ATTRIB_CLOSURE
              and r["move_ratio"] > ps._SLIVER_MOVE_FACTOR)
    summary["d5_blocked_by_absolute_n"] = len(d5w)
    summary["d5_blocked_by_absolute_drop_min"] = env(d5w, how=min)
    summary["d5_blocked_by_absolute_drop_max"] = env(d5w)
    # the flip census, per candidate fraction
    flips = {}
    for f in FRACS:
        k = f"{f:g}"
        ch = [r for r in rows if r["v2"] != r["v3"][k]]
        flips[k] = dict(
            n=len(ch),
            by_kind_abs={c: sum(1 for r in ch if r["kind_abs"] == c)
                         for c in ("right", "grey", "wrong")},
            by_kind_slope={c: sum(1 for r in ch if r["kind_slope"] == c)
                           for c in ("right", "grey", "wrong")},
            by_family={fam: sum(1 for r in ch if r["family"] == fam)
                       for fam in ("STAIR", "D5", "STEEP")},
            worst_err_over_d_flipped=min((r["err_over_d"] for r in ch),
                                         default=None),
            drop_min=env(ch, how=min))
    summary["flips"] = flips
    with open(out_path, "w") as fh:
        json.dump(dict(summary=summary, rows=rows), fh, indent=1)
    print("\n  --- drop-factor populations (absolute continuity rule) ---")
    for kind in ("right", "grey", "wrong"):
        print(f"    {kind:6s} n={summary[f'abs_{kind}_n']:5d}  drop "
              f"{summary[f'abs_{kind}_drop_min']} .. "
              f"{summary[f'abs_{kind}_drop_max']}   (past the MOVE bar: "
              f"n={summary[f'abs_{kind}_pastmove_n']}, drop max "
              f"{summary[f'abs_{kind}_pastmove_drop_max']})")
    print("  --- drop-factor populations (slope-normalised rule) ---")
    for kind in ("right", "grey", "wrong"):
        print(f"    {kind:6s} n={summary[f'slope_{kind}_n']:5d}  drop "
              f"{summary[f'slope_{kind}_drop_min']} .. "
              f"{summary[f'slope_{kind}_drop_max']}   (past the MOVE bar: "
              f"n={summary[f'slope_{kind}_pastmove_n']}, drop max "
              f"{summary[f'slope_{kind}_pastmove_drop_max']})")
    print(f"  D-5 class blocked by the ABSOLUTE bar: n="
          f"{summary['d5_blocked_by_absolute_n']}, drop "
          f"{summary['d5_blocked_by_absolute_drop_min']} .. "
          f"{summary['d5_blocked_by_absolute_drop_max']}")
    print("  --- flips, round 2 -> round 3 ---")
    for k, v in flips.items():
        print(f"    frac {k:7s} n={v['n']:4d}  abs {v['by_kind_abs']}  "
              f"slope {v['by_kind_slope']}  fam {v['by_family']}  "
              f"min err/delta among flipped {v['worst_err_over_d_flipped']}")
    print("->", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
