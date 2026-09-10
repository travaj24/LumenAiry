"""V5 -- the rigorous 1-D lossless-closure ladder for the fff_nv stripe
fixture, coincident vs detuned (Task H.a).

Independently reconstructs the fixture from the test file's stated geometry
(period 0.7 um, wl 1.0 um, depth 0.5 um, normal incidence, n_sub = 1.5,
n_sup = 1.0, segments [(0.5, rot(35 deg, no=1.5, ne=2.3)), (0.5, eps_g * I)])
and reports |sum R + sum T - 2| over n_orders = 11..41 odd, plus the count
below 1e-9, for several groove permittivities.

    python v5_ladder.py <lumenairy-root> [tag]
"""
import json
import os
import sys
import time
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

import lumenairy  # noqa: E402

_ROOT = os.path.abspath(sys.argv[1])
assert os.path.abspath(lumenairy.__file__).startswith(_ROOT), lumenairy.__file__

from lumenairy.elements.rcwa.oned import rcwa_jones_1d_segments  # noqa: E402

TAG = sys.argv[2] if len(sys.argv) > 2 else "run"
PX, WL, DEPTH = 0.7e-6, 1.0e-6, 0.5e-6
NO_, NE_ = 1.5, 2.3
N_SUB, N_SUP = 1.5, 1.0


def rot(phi, no, ne):
    c, s = np.cos(phi), np.sin(phi)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return R @ np.diag([ne ** 2, no ** 2, no ** 2]).astype(complex) @ R.T


ER = rot(np.deg2rad(35.0), NO_, NE_)
print(f"director ordinary eps (no^2) = {NO_ ** 2!r}   "
      f"n_sub^2 = {N_SUB ** 2!r}   ER[1,1] = {ER[1, 1].real!r}")


def defect(eps_g, n):
    eg = np.diag([eps_g] * 3).astype(complex)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _o, R1, T1, _J = rcwa_jones_1d_segments(
            PX, [(0.5, ER), (0.5, eg)], N_SUB, N_SUP, DEPTH, WL, theta=0.0,
            n_orders=int(n))
    return float(np.sum(R1) + np.sum(T1) - 2.0), float(np.sum(R1))


LADDER = list(range(11, 42, 2))
OUT = {}
for eps_g, label in ((2.25, "coincident (no^2 = n_sub^2 = 2.25)"),
                     (2.10, "detuned 2.10 (the shipped fixture)"),
                     (2.40, "detuned 2.40"),
                     (2.2499999999, "detuned by 1e-10 relative")):
    t0 = time.perf_counter()
    rows = [defect(eps_g, n) for n in LADDER]
    d = [abs(r[0]) for r in rows]
    sound = sum(1 for x in d if x < 1e-9)
    OUT[str(eps_g)] = {"ladder": LADDER, "defect": d,
                       "sumR": [r[1] for r in rows], "sound": sound}
    print(f"\neps_groove = {eps_g}  -- {label}   [{time.perf_counter()-t0:.1f} s]")
    print(f"  worst = {max(d):.4e}   best = {min(d):.4e}   "
          f"sound(<1e-9) = {sound} of {len(LADDER)}")
    print("  " + "  ".join(f"{n}:{x:.2e}" for n, x in zip(LADDER, d)))
    print(f"  sum R at n=41: {rows[-1][1]:.12f}")

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       f"_out_v5_{TAG}.json"), "w", encoding="cp1252") as fh:
    json.dump(OUT, fh, indent=1)
