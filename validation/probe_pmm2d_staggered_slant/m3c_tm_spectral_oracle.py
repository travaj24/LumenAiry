"""M3c -- the TM channel against a CONVERGED oracle.

M3's TM residual (1e-4..1.3e-3) is the size of the 1-D scalar slant oracle's
OWN degree drift (M3b: 6.5e-4 at degree 22, falling ALGEBRAICALLY to 1.45e-4 at
degree 30) -- so M3 cannot resolve the 2-D slanted TM channel.  The library's
``pmm_jones_1d_slanted(factorization='covariant')`` is the Li-1999 oblique-
coordinate path whose TM converges SPECTRALLY (~1e-7 by degree ~24); driven with
isotropic tensors it is a converged TM oracle for exactly this cell.

T3c-1  the covariant oracle's own degree drift (is it converged?)
T3c-2  the covariant oracle vs the scalar oracle (do the two 1-D paths agree,
       and at what level?)
T3c-3  the 2-D staggered slanted arm vs the covariant oracle, M-ladder.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slant_lib import assert_worktree, solve_slant_stack  # noqa: E402
from lumenairy.elements.pmm import (  # noqa: E402
    pmm_efficiency_1d_slanted, pmm_jones_1d_slanted,
)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
PX = PY = 0.75
WL, DEPTH = 1.0, 0.30
NR, NG, NSUP, NSUB = 2.0, 1.0, 1.0, 1.5
NO = 4
ORD_CMP = [-2, -1, 0, 1, 2]
CELL = np.array([[NR ** 2, NR ** 2], [NG ** 2, NG ** 2]], dtype=complex)
res = {"lumenairy": assert_worktree()}
t0 = time.time()


def cov_oracle(phi_deg, theta, degree, fact="covariant"):
    o, R, T, _J = pmm_jones_1d_slanted(
        PX, (NR ** 2) * np.eye(3), (NG ** 2) * np.eye(3), NSUB, NSUP, DEPTH,
        0.5, WL, np.deg2rad(phi_deg), angle=theta, degree=degree,
        far_field_orders=15, factorization=fact)
    # row 0 = incident Ex (TM at phi_inc = 0), row 1 = incident Ey (TE)
    return {"tm": {int(m): (float(R[0, i]), float(T[0, i]))
                   for i, m in enumerate(o)},
            "te": {int(m): (float(R[1, i]), float(T[1, i]))
                   for i, m in enumerate(o)}}


def scal_oracle(phi_deg, theta, pol, degree=30):
    o, R, T = pmm_efficiency_1d_slanted(
        PX, NR, NG, NSUB, NSUP, DEPTH, 0.5, WL, np.deg2rad(phi_deg),
        angle=theta, polarization=pol, degree=degree, far_field_orders=15)
    return {int(m): (float(R[i]), float(T[i])) for i, m in enumerate(o)}


def two_d(t, theta, M):
    o, R, T, _Jr, _Jt, _i = solve_slant_stack(
        PX, PY, [{"thickness": DEPTH, "cell": CELL, "slant": (t, 0.0)}],
        NSUP, NSUB, WL, M=M, n_orders=NO, theta=theta, phi=0.0)
    idx = {int(m): i for i, (m, n) in enumerate(o) if n == 0}
    return {"tm": {m: (float(R[0, i]), float(T[0, i])) for m, i in idx.items()},
            "te": {m: (float(R[1, i]), float(T[1, i])) for m, i in idx.items()}}


def d(a, b):
    return max(max(abs(a[m][0] - b[m][0]), abs(a[m][1] - b[m][1]))
               for m in ORD_CMP if m in a and m in b)


MOUNTS = (("normal", 0.0), ("oblique25", np.deg2rad(25)))
PHIS = (10.0, 20.0, 35.0)

print("T3c-1  covariant oracle's OWN degree drift (ref degree 30)")
r1 = []
for phi in PHIS:
    for mn, th in MOUNTS:
        ref = cov_oracle(phi, th, 30)
        row = {"phi": phi, "mount": mn}
        for deg in (18, 22, 26):
            c = cov_oracle(phi, th, deg)
            row[f"tm{deg}"] = d(c["tm"], ref["tm"])
            row[f"te{deg}"] = d(c["te"], ref["te"])
        r1.append(row)
        print(f"  phi={phi:4.0f} {mn:9s} TM 18/22/26 "
              f"{row['tm18']:.1e} {row['tm22']:.1e} {row['tm26']:.1e} | "
              f"TE {row['te18']:.1e} {row['te22']:.1e} {row['te26']:.1e}")
res["T3c1_oracle_drift"] = r1

print("\nT3c-2  covariant vs scalar 1-D oracle (the two shipped 1-D paths)")
r2 = []
for phi in PHIS:
    for mn, th in MOUNTS:
        c = cov_oracle(phi, th, 30)
        row = {"phi": phi, "mount": mn,
               "tm": d(c["tm"], scal_oracle(phi, th, "tm")),
               "te": d(c["te"], scal_oracle(phi, th, "te"))}
        r2.append(row)
        print(f"  phi={phi:4.0f} {mn:9s} TM {row['tm']:.2e}  TE {row['te']:.2e}")
res["T3c2_cross_oracle"] = r2

print("\nT3c-3  2-D staggered SLANT vs the covariant oracle, M-ladder")
r3 = []
for phi in PHIS:
    for mn, th in MOUNTS:
        ref = cov_oracle(phi, th, 30)
        for M in (5, 6, 7, 8):
            d2 = two_d(-np.tan(np.deg2rad(phi)), th, M)
            row = {"phi": phi, "mount": mn, "M": M,
                   "tm": d(d2["tm"], ref["tm"]), "te": d(d2["te"], ref["te"])}
            r3.append(row)
            print(f"  phi={phi:4.0f} {mn:9s} M={M}  TM {row['tm']:.2e}  "
                  f"TE {row['te']:.2e}")
res["T3c3_2d_vs_covariant"] = r3
res["wall_s"] = time.time() - t0
with open(os.path.join(OUT, "m3c_tm_spectral_oracle.json"), "w") as f:
    json.dump(res, f, indent=1)
print(f"\nWROTE results/m3c_tm_spectral_oracle.json  ({res['wall_s']:.1f} s)")
