"""V6c -- two live re-measurements the reference study rests on.

1. REPRODUCIBILITY of the committed ladder data: re-run one staggered rung
   (M = 7) and one rcwa rung (n_orders = 5) of the ``corner / normal`` case on
   THIS build and compare, observable by observable, with the numbers in
   ``probe_pmm2d_staggered_oop_reference/results/t1_corner.json``.  If the
   committed ladders were not reproducible the whole re-fit would be moot.

2. The study's ONE correction to its brief: ``rcwa_jones_2d`` really does
   consume the OUT-OF-PLANE tensor entries.  Zero ``e13/e23/e31/e32`` on the
   corner cell and measure how far the answer moves (claimed dR 7.63e-04,
   dJones 2.41e-03 at n_orders = 5).

The corner fixture is rebuilt here from the study's own description (S1: a
(3,3) L of three ``uniaxial(1.5, 1.7, tilt 35 deg, azim 25 deg)`` pixels on an
air background, ``px = py = 1.2 lam``, ``depth = 0.4 lam``, ``n_sub = 1.5``).

Usage: python v6c_rcwa_oop.py
"""
import json
import math
import os
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REF = os.path.join(os.path.dirname(HERE), "probe_pmm2d_staggered_oop_reference",
                   "results")

import numpy as np  # noqa: E402
import vfix as F  # noqa: E402

F.assert_arm("C:/tmp/lum_vacc")

from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    pmm_jones_2d_staggered,
)
from lumenairy.elements.rcwa import rcwa_jones_2d  # noqa: E402
from lumenairy.elements.rcwa._core import uniaxial_tensor  # noqa: E402

PX = PY = 1.2e-6
WL = 1.0e-6
DEP = 0.4e-6
NSUB, NSUP = 1.5, 1.0


def corner_cell():
    er = uniaxial_tensor(1.5, 1.7, np.deg2rad(35.0), phi=np.deg2rad(25.0))
    e = np.zeros((3, 3, 3, 3), dtype=complex)
    e[:, :] = np.eye(3)
    for i, j in ((0, 0), (1, 0), (0, 1)):
        e[i, j] = er
    return e


def upsample(cell, f):
    return np.repeat(np.repeat(cell, f, axis=0), f, axis=1)


def observables(orders, R, T, J):
    """The study's observable dictionary, rebuilt from the entry point's
    return: per-order R/T for both incident polarizations, the two sums, and
    the eight Jones components."""
    out = {}
    o = np.asarray(orders)
    Ra, Ta = np.asarray(R), np.asarray(T)
    for i in range(o.shape[0]):
        key = (int(o[i, 0]), int(o[i, 1]))
        for p in (0, 1):
            out[f"R({key[0]}, {key[1]})p{p}"] = float(Ra[p, i])
            out[f"T({key[0]}, {key[1]})p{p}"] = float(Ta[p, i])
    for p in (0, 1):
        out[f"sumR{p}"] = float(np.sum(Ra[p, :]))
        out[f"sumT{p}"] = float(np.sum(Ta[p, :]))
    Jm = np.asarray(J)
    for a in (0, 1):
        for b in (0, 1):
            out[f"Jre{a}{b}"] = float(np.real(Jm[a, b]))
            out[f"Jim{a}{b}"] = float(np.imag(Jm[a, b]))
    return out


def committed(engine, rung):
    d = json.load(open(os.path.join(REF, "t1_corner.json")))
    case = [c for c in d["cases"] if c["mount"] == "normal"][0]
    lad = case["ladders"][engine]
    i = lad["x"].index(rung)
    return lad["obs"][i]


def part1():
    print("=" * 92)
    print("1. REPRODUCING committed ladder rungs of corner / normal on THIS "
          "build")
    print("=" * 92)
    cell = corner_cell()
    rows = []
    # --- staggered M = 7
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = pmm_jones_2d_staggered(PX, PY, cell, NSUB, NSUP, DEP, WL,
                                     degree=7, n_orders=5)
    dt = time.perf_counter() - t0
    mine = observables(*res)
    ref = committed("staggered", 7)
    common = [k for k in ref if k in mine]
    dd = {k: abs(mine[k] - ref[k]) for k in common}
    worst = max(dd, key=dd.get)
    print(f"staggered M=7 ({dt:.1f} s): {len(common)}/{len(ref)} observables "
          f"matched by key; worst |mine - committed| = {dd[worst]:.3e} "
          f"on {worst}")
    print(f"   sumR0 mine {mine['sumR0']:.12f}  committed {ref['sumR0']:.12f}")
    rows.append(dict(engine="staggered", rung=7, worst=dd[worst],
                     worst_key=worst, n=len(common), seconds=dt))
    # --- rcwa n_orders = 5, pixel-upsampled by ceil((4n+1)/3)
    n = 5
    f = math.ceil((4 * n + 1) / 3)
    up = upsample(cell, f)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = rcwa_jones_2d(PX, PY, up, NSUB, NSUP, DEP, WL, n_orders_x=n,
                            n_orders_y=n)
    dt = time.perf_counter() - t0
    mine = observables(*res)
    ref = committed("rcwa", 5)
    common = [k for k in ref if k in mine]
    dd = {k: abs(mine[k] - ref[k]) for k in common}
    worst = max(dd, key=dd.get) if dd else None
    print(f"rcwa n=5, upsample x{f} ({dt:.1f} s): {len(common)}/{len(ref)} "
          f"observables matched by key; worst |mine - committed| = "
          f"{dd[worst]:.3e} on {worst}")
    print(f"   sumR0 mine {mine['sumR0']:.12f}  committed {ref['sumR0']:.12f}")
    rows.append(dict(engine="rcwa", rung=5, worst=dd[worst], worst_key=worst,
                     n=len(common), seconds=dt, upsample=f))
    return rows


def part2():
    print()
    print("=" * 92)
    print("2. DOES rcwa_jones_2d CONSUME the out-of-plane tensor entries?")
    print("=" * 92)
    cell = corner_cell()
    rows = []
    for n in (5, 7):
        f = math.ceil((4 * n + 1) / 3)
        up = upsample(cell, f)
        flat = up.copy()
        flat[..., 0, 2] = flat[..., 1, 2] = 0.0
        flat[..., 2, 0] = flat[..., 2, 1] = 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = rcwa_jones_2d(PX, PY, up, NSUB, NSUP, DEP, WL, n_orders_x=n,
                              n_orders_y=n)
            b = rcwa_jones_2d(PX, PY, flat, NSUB, NSUP, DEP, WL, n_orders_x=n,
                              n_orders_y=n)
        dR = F.dmax(a[1], b[1])
        dT = F.dmax(a[2], b[2])
        dJ = F.dmax(a[3], b[3])
        ident = F.sha(*a) == F.sha(*b)
        rows.append(dict(n_orders=n, upsample=f, dR=dR, dT=dT, dJones=dJ,
                         identical=bool(ident)))
        print(f"n_orders = {n:2d} (upsample x{f}):  dR {dR:.3e}   dT {dT:.3e}"
              f"   dJones {dJ:.3e}   bit-identical {ident}")
    # and the same knock-out on the staggered arm, for scale
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = pmm_jones_2d_staggered(PX, PY, cell, NSUB, NSUP, DEP, WL,
                                   degree=6, n_orders=5)
        flat = cell.copy()
        flat[..., 0, 2] = flat[..., 1, 2] = 0.0
        flat[..., 2, 0] = flat[..., 2, 1] = 0.0
        b = pmm_jones_2d_staggered(PX, PY, flat, NSUB, NSUP, DEP, WL,
                                   degree=6, n_orders=5)
    print(f"staggered M = 6, same knock-out:  dR {F.dmax(a[1], b[1]):.3e}   "
          f"dT {F.dmax(a[2], b[2]):.3e}   dJones {F.dmax(a[3], b[3]):.3e}")
    rows.append(dict(engine="staggered", M=6, dR=F.dmax(a[1], b[1]),
                     dT=F.dmax(a[2], b[2]), dJones=F.dmax(a[3], b[3])))
    return rows


if __name__ == "__main__":
    p1 = part1()
    p2 = part2()
    with open(os.path.join(HERE, "results", "v6c_rcwa_oop.json"), "w") as fh:
        json.dump(dict(reproduce=p1, oop_knockout=p2), fh, indent=1)
