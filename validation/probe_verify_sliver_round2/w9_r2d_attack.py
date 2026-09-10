"""W9 -- the DIRECTED attack on R2-D.

The structural observation this probe is built on.  The prescribed snap moves
each colliding wall to the pair's MIDPOINT, i.e. by at most ``w_wide`` of a
period, so for a solve that is NOT corrupted

    move  ~  (dR/dx) * w_wide        =>      move / w_wide  ~  dR/dx

and ``move > 100 * w_wide`` is, to a factor of order one, a test of
``dR/dx > 100``.  The campaign's own ``err > 100 delta`` WRONG rule is the
same statistic with the exact ``delta -> 0`` solve in place of the snapped
one, which is why the two separate so cleanly -- and why a device whose OWN
physical ``dR/dx`` exceeds ~100 is where the criterion can misfire.

A false refusal therefore needs THREE things at once, and this probe searches
for all three:

  1. ``dR/dx`` above ~100                       (a resonant geometry)
  2. ``R+T - 1`` above the 1e-3 trigger from ORDINARY truncation
     (a dense superstrate at a grazing angle and a modest degree)
  3. the truncation super-unity CLOSING on the snapped grid
     (which needs the super-unity itself to be geometry-sensitive at the
     ``w_wide`` scale -- i.e. the same resonance)

For every REFUSED row it reports ``err / (s * delta)`` with ``s`` the
device's OWN measured smooth-regime slope: a value near 1 is an answer that
tracks this device's physical wall shift, i.e. a FALSE REFUSAL.

    python w9_r2d_attack.py [out.json]
"""
import itertools
import json
import os
import sys
import time
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from w_fixtures import prescribed, shared_move, snapped, unguarded  # noqa: E402

import lumenairy  # noqa: E402
from lumenairy.elements.pmm import PMMStack  # noqa: E402

NO_SNAP_FRAC = 1e-12


def build(d, deg, cfg, nl=2):
    p = cfg["period"]
    st = PMMStack(p, n_superstrate=cfg["nsup"], n_substrate=cfg["nsub"],
                  degree=deg, far_field_orders=31,
                  min_feature=p * NO_SNAP_FRAC)
    a0, b0 = cfg["a0"], cfg["b0"]
    for k in range(nl):
        dd = d * k / max(nl - 1, 1)
        st.add_layer(cfg["dz"], segments=[(a0 - dd, cfg["eh"]),
                                          (b0 + dd - (a0 - dd), cfg["ep"]),
                                          (1.0 - (b0 + dd), cfg["eh"])])
    st.set_source(cfg["wl"], theta=cfg["th"])
    return st


def smooth_slope(cfg, deg, ds=(3e-3, 1e-3, 3e-4)):
    """The device's own continuity slope where the answer is stationary in
    BOTH degree and delta -- measured, not assumed."""
    ref = unguarded(build(0.0, deg, cfg))
    out = []
    for d in ds:
        cur = unguarded(build(d, deg, cfg))
        out.append(shared_move(cur, ref, pol=1) / d)
    return float(np.median(out)), out


def guarded(cfg, d, deg):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        try:
            build(d, deg, cfg).solve()
        except ValueError as exc:
            return ("REFUSED" if "NEAR-COINCIDENT-WALL SLIVER" in str(exc)
                    else "raised"), [str(w.message) for w in rec]
    return "returned", [str(w.message) for w in rec]


def main():
    out_path = (sys.argv[1] if len(sys.argv) > 1
                else os.path.join(HERE, "w9_r2d_attack.json"))
    lib = os.path.abspath(lumenairy.__file__)
    print("lumenairy:", lib)
    t0 = time.time()
    rows = []
    deltas = [float(x) for x in np.geomspace(3e-3, 1e-6, 40)]
    # The mount is the false-positive box's (dense superstrate, grazing, lossy
    # substrate, modest degree -- the truncation super-unity regime); the
    # GEOMETRY is scanned for steepness.
    grid = itertools.product(
        (1.2e-6, 0.9e-6, 1.6e-6),                    # period
        (0.85e-6, 0.62e-6, 1.31e-6),                 # wavelength
        ((2.4, 1.22), (3.2, 1.33), (2.4, 1.44)),     # (n_sup, theta)
        (9.0, 12.0, 16.0),                           # eps_ridge
        ((0.27865, 0.62505), (0.19137, 0.71429), (0.4150, 0.5350)),
        (0.32e-6 / 4, 0.21e-6 / 3),                  # slice thickness
        (6, 8, 10))                                  # degree
    seen = 0
    for period, wl, (nsup, th), ep, (a0, b0), dz, deg in grid:
        cfg = dict(period=period, wl=wl, nsup=nsup, th=th, ep=ep, eh=2.25,
                   a0=a0, b0=b0, dz=dz, nsub=1.5 + 0.2j)
        seen += 1
        try:
            slope, ladder = smooth_slope(cfg, deg)
            ref = unguarded(build(0.0, deg, cfg))
        except (ValueError, NotImplementedError, RuntimeError):
            continue
        if not np.isfinite(slope) or slope <= 0:
            continue
        for d in deltas:
            try:
                st = build(d, deg, cfg)
                cur = unguarded(st)
            except (ValueError, NotImplementedError, RuntimeError):
                continue
            if cur["worst"] - 1.0 <= 1e-3:
                continue                       # the arbiter cannot even run
            pre = prescribed(st)
            if pre is None:
                continue
            snp = snapped(st, pre["mf"])
            su = max(snp["worst"] - 1.0, 0.0)
            mv = shared_move(cur, snp) / pre["w_wide"]
            if not (su <= 1e-5 and mv > 100.0):
                continue                       # not refused: not a candidate
            e = shared_move(cur, ref, pol=1)
            v, _w = guarded(cfg, d, deg)
            rows.append(dict(period=period, wl=wl, nsup=nsup, th=th, ep=ep,
                             a0=a0, b0=b0, dz=dz, deg=deg, delta=d,
                             slope=slope, slope_ladder=ladder, err=e,
                             err_over_d=e / d,
                             err_over_slope_d=e / max(slope * d, 1e-300),
                             worst=cur["worst"], su_snap=su, move_ratio=mv,
                             verdict=v))
    rows.sort(key=lambda r: r["err_over_slope_d"])
    summary = dict(lumenairy=lib, python=sys.version.split()[0],
                   numpy=np.__version__, configs=seen, n_refusals=len(rows),
                   wall=time.time() - t0,
                   min_err_over_slope_d=(rows[0]["err_over_slope_d"]
                                         if rows else None),
                   max_slope_among_refusals=(max(r["slope"] for r in rows)
                                             if rows else None),
                   n_false_refusals=sum(1 for r in rows
                                        if r["err_over_slope_d"] <= 3.0),
                   n_borderline=sum(1 for r in rows
                                    if r["err_over_slope_d"] <= 30.0))
    with open(out_path, "w") as fh:
        json.dump(dict(summary=summary, rows=rows[:400]), fh, indent=1)
    for k, v in summary.items():
        print(f"  {k:26s} {v}")
    print("  --- the 12 refusals CLOSEST to being false ---")
    for r in rows[:12]:
        print(f"    P={r['period']:.2e} wl={r['wl']:.2e} sup={r['nsup']} "
              f"th={r['th']} eps={r['ep']} deg={r['deg']} "
              f"d={r['delta']:.3e} slope={r['slope']:.3g} "
              f"err/(s d)={r['err_over_slope_d']:.2f} "
              f"move={r['move_ratio']:.1f} su_snap={r['su_snap']:.2e} "
              f"R+T={r['worst']:.6g} -> {r['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
