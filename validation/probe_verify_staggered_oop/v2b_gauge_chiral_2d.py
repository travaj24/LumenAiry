"""V2b -- the ROTATION gauge on a genuinely 2-D CHIRAL out-of-plane cell,
against the two Fourier oracles with their own truncation ladders.

The build's 2-D fixture is an L of three tilted-uniaxial pixels.  This one is a
different chiral shape with TWO DIFFERENT out-of-plane tensors plus an
isotropic pixel, so the 180-degree image is a different cell in both the
pattern and the tensor.  Oracles: ``rcwa_jones_2d`` (pixel-upsampled, n_orders
5/7/9 -- the ladder bounds its own drift) and ``pmm_jones_2d`` in BOTH ``E_z``
elimination rules (their mutual disagreement bounds the hybrid).

Usage:  PYTHONPATH=<root> python v2b_gauge_chiral_2d.py <root> <out.json>
"""
import json
import os
import sys
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

import lumenairy  # noqa: E402

ROOT = os.path.abspath(sys.argv[1])
assert os.path.abspath(lumenairy.__file__).startswith(ROOT), (
    f"lumenairy.__file__ = {lumenairy.__file__} is not under {ROOT}")

from lumenairy.elements.pmm import pmm_jones_2d  # noqa: E402
from lumenairy.elements.pmm import twod_staggered as TS  # noqa: E402
from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    pmm_jones_2d_staggered,
)
from lumenairy.elements.rcwa import rcwa_jones_2d, uniaxial_tensor  # noqa: E402

OUT = sys.argv[2]

PX = PY = 1.10e-6
WL = 0.68e-6
DEP = 0.36e-6
NSUB = 1.50
NSUP = 1.0

tA = uniaxial_tensor(1.46, 1.74, 0.58, phi=0.31)
tC = uniaxial_tensor(1.58, 1.82, 1.11, phi=2.05)
AIR = np.eye(3, dtype=complex)
ISO = 2.25 * np.eye(3, dtype=complex)

CELL = np.zeros((3, 3, 3, 3), dtype=complex)
CELL[:, :] = AIR
CELL[0, 0] = tA
CELL[1, 0] = tC
CELL[1, 2] = ISO


def upsample(cell, k):
    return np.repeat(np.repeat(cell, k, axis=0), k, axis=1)


def per_order(o_a, R_a, T_a, o_b, R_b, T_b):
    oa, ob = np.asarray(o_a), np.asarray(o_b)
    mb = {(int(m), int(n)): j for j, (m, n) in enumerate(ob)}
    dR = dT = 0.0
    n = 0
    for i, (m, nn) in enumerate(oa):
        k = mb.get((int(m), int(nn)))
        if k is None:
            continue
        dR = max(dR, float(np.max(np.abs(np.asarray(R_a)[:, i]
                                         - np.asarray(R_b)[:, k]))))
        dT = max(dT, float(np.max(np.abs(np.asarray(T_a)[:, i]
                                         - np.asarray(T_b)[:, k]))))
        n += 1
    return dR, dT, n


out = {"root": ROOT, "lumenairy": lumenairy.__file__, "cases": []}


def save():
    json.dump(out, open(OUT, "w"), indent=1)


for mount, th, ph in (("normal", 0.0, 0.0),
                      ("oblique25", np.deg2rad(25.0), 0.0),
                      ("conical25_40", np.deg2rad(25.0), np.deg2rad(40.0))):
    rec = {"mount": mount, "oracles": [], "staggered": []}
    arms = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for no in (5, 7, 9):
            t0 = time.perf_counter()
            arms[("rcwa", no)] = rcwa_jones_2d(
                PX, PY, upsample(CELL, int(np.ceil((4 * no + 1) / 3))),
                NSUB, NSUP, DEP, WL, theta=th, phi=ph,
                n_orders_x=no, n_orders_y=no)
            print(f"[{mount}] rcwa n_orders={no} {time.perf_counter()-t0:.1f}s",
                  flush=True)
        for form in ("laurent", "li"):
            t0 = time.perf_counter()
            arms[("hyb", form)] = pmm_jones_2d(
                PX, PY, CELL, NSUB, NSUP, DEP, WL, theta=th, phi=ph,
                degree=9, n_orders=13, formulation=form, stabilize=True)
            print(f"[{mount}] hybrid {form} {time.perf_counter()-t0:.1f}s",
                  flush=True)
    ref = arms[("rcwa", 9)]
    for key in (("rcwa", 5), ("rcwa", 7), ("hyb", "laurent"), ("hyb", "li")):
        a = arms[key]
        dR, dT, n = per_order(*a[:3], *ref[:3])
        dJ = float(np.max(np.abs(np.asarray(a[3]) - np.asarray(ref[3]))))
        rec["oracles"].append({"arm": f"{key[0]}-{key[1]}", "dR": dR, "dT": dT,
                               "dJones": dJ, "n": n})
        print(f"[{mount}] ORACLE {key[0]}-{key[1]} vs rcwa(9): dR={dR:.3e} "
              f"dT={dT:.3e} dJ={dJ:.3e}", flush=True)
    for rot, Ms in ((-1.0, (5, 6, 7)), (+1.0, (6,))):
        TS._OOP_ROT_SIGN = rot
        prev = None
        for M in Ms:
            t0 = time.perf_counter()
            s = pmm_jones_2d_staggered(PX, PY, CELL, NSUB, NSUP, DEP, WL,
                                       degree=M, n_orders=3, theta=th, phi=ph)
            dR, dT, n = per_order(*s[:3], *ref[:3])
            dJ = float(np.max(np.abs(np.asarray(s[3]) - np.asarray(ref[3]))))
            clo = float(np.max(np.abs(np.asarray(s[1]).sum(axis=1)
                                      + np.asarray(s[2]).sum(axis=1) - 1)))
            self_move = None
            if prev is not None:
                self_move = max(per_order(*s[:3], *prev[:3])[:2])
            rec["staggered"].append({"rot": rot, "M": M, "dR": dR, "dT": dT,
                                     "dJones": dJ, "closure": clo,
                                     "self_move": self_move,
                                     "t": time.perf_counter() - t0})
            print(f"[{mount}] staggered rot={rot:+.0f} M={M} dR={dR:.3e} "
                  f"dT={dT:.3e} dJ={dJ:.3e} |R+T-1|={clo:.2e} "
                  f"self_move={self_move} ({time.perf_counter()-t0:.1f}s)",
                  flush=True)
            prev = s
    TS._OOP_ROT_SIGN = -1.0
    # NO-FLOOR two-sided on this cell
    a = pmm_jones_2d_staggered(PX, PY, CELL, NSUB, NSUP, DEP, WL, degree=6,
                               n_orders=3, theta=th, phi=ph)
    b = pmm_jones_2d_staggered(PX, PY, CELL, NSUB, NSUP, DEP, WL, degree=6,
                               n_orders=8, theta=th, phi=ph)
    mv = max(per_order(*a[:3], *b[:3])[:2])
    mvJ = float(np.max(np.abs(np.asarray(a[3]) - np.asarray(b[3]))))
    hy = max(per_order(*arms[("hyb", "laurent")][:3],
                       *arms[("hyb", "li")][:3])[:2])
    rec["nofloor"] = {"staggered_orders_3_to_8": mv,
                      "staggered_jones": mvJ, "hybrid_two_rules": hy}
    print(f"[{mount}] NO-FLOOR staggered n_orders 3->8 moves {mv:.3e} "
          f"(Jones {mvJ:.3e}); hybrid's two E_z rules differ by {hy:.3e}",
          flush=True)
    out["cases"].append(rec)
    save()
print("DONE")
