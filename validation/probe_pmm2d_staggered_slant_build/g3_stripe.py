"""B5 -- a y-uniform SLANTED SCALAR STRIPE vs ``pmm_efficiency_1d_slanted``,
per order, both polarizations, normal + oblique, slant 10 / 20 / 35 deg, with
the SLANT-SIGN arm scanned two-sided.

B6 -- a y-uniform SLANTED OUT-OF-PLANE stripe vs ``pmm_jones_1d_slanted``, the
combination NO other 2-D engine in the suite covers (the hybrid raises).

The bar for the slanted arm is the VERTICAL CONTROL: the same 2-D cell at
slant 0 against the vertical 1-D oracle is what this basis can do on this
geometry, and the slanted arm must TRACK it, not beat it.
"""
import time

import numpy as np
from _lib import uniaxial, write

from lumenairy.elements.pmm import (
    pmm_efficiency_1d,
    pmm_efficiency_1d_slanted,
    pmm_jones_1d,
    pmm_jones_1d_slanted,
    pmm_jones_2d_staggered,
)

PX = PY = 0.75
WL = 1.0
DEPTH = 0.30
NR, NGV = 2.0, 1.0
NSUP, NSUB = 1.0, 1.5
NO = 4
ORD_CMP = [-2, -1, 0, 1, 2]
DEG1D = 22

CELL = np.array([[NR ** 2, NR ** 2], [NGV ** 2, NGV ** 2]], dtype=complex)

res = {}
t00 = time.time()


def two_d(cell, slant, theta, M, n_orders=NO):
    o, R, T, _J = pmm_jones_2d_staggered(
        PX, PY, cell, NSUB, NSUP, DEPTH, WL, n_modes=M, n_orders=n_orders,
        theta=theta, phi=0.0, slant=slant)
    idx = {int(m): i for i, (m, n) in enumerate(o) if n == 0}
    out = {"tm": {}, "te": {}}
    for m, i in idx.items():
        out["tm"][m] = (float(R[0, i]), float(T[0, i]))
        out["te"][m] = (float(R[1, i]), float(T[1, i]))
    ylk = 0.0
    for i, (m, n) in enumerate(o):
        if n != 0:
            ylk = max(ylk, float(np.max(np.abs(R[:, i]))),
                      float(np.max(np.abs(T[:, i]))))
    clo = float(np.max(np.abs(R.sum(1) + T.sum(1) - 1.0)))
    return out, ylk, clo


def oracle_s(phi_deg, theta, pol):
    o, R, T = pmm_efficiency_1d_slanted(
        PX, NR, NGV, NSUB, NSUP, DEPTH, 0.5, WL, np.deg2rad(phi_deg),
        angle=theta, polarization=pol, degree=DEG1D, far_field_orders=15)
    return {int(m): (float(R[i]), float(T[i])) for i, m in enumerate(o)}


def oracle_v(theta, pol):
    o, R, T = pmm_efficiency_1d(
        PX, NR, NGV, NSUB, NSUP, DEPTH, 0.5, WL, angle=theta,
        polarization=pol, degree=DEG1D, far_field_orders=15)
    return {int(m): (float(R[i]), float(T[i])) for i, m in enumerate(o)}


def perorder(a, b):
    return max(max(abs(a[m][0] - b[m][0]), abs(a[m][1] - b[m][1]))
               for m in ORD_CMP if m in a and m in b)


# ------------------------------------------------------------------ B5a sign
print("B5a  SLANT-SIGN ARBITRATION vs the 1-D oracle (phi = 35 deg, M = 7)")
sign_rows = []
for mount, theta in (("normal", 0.0), ("oblique25", np.deg2rad(25.0))):
    vert2d, _y, _c = two_d(CELL, None, theta, 7)
    for pol in ("te", "tm"):
        ctrl = perorder(vert2d[pol], oracle_v(theta, pol))
        orc = oracle_s(35.0, theta, pol)
        row = {"mount": mount, "pol": pol, "ctrl": ctrl}
        for sgn, lab in ((+1.0, "plus_tan"), (-1.0, "minus_tan")):
            d2, _y, _c = two_d(CELL, (sgn * np.tan(np.deg2rad(35.0)), 0.0),
                               theta, 7)
            row[lab] = perorder(d2[pol], orc)
        sign_rows.append(row)
        print("  %-9s %s  ctrl %.2e | slant=+tan %.2e | slant=-tan %.2e"
              % (mount, pol, ctrl, row["plus_tan"], row["minus_tan"]))
res["B5a_sign"] = sign_rows
wp = max(r["plus_tan"] for r in sign_rows)
wm = max(r["minus_tan"] for r in sign_rows)
res["B5a_winner"] = "+tan" if wp < wm else "-tan"
res["B5a_worst_winner"] = min(wp, wm)
res["B5a_worst_loser"] = max(wp, wm)
print("  -> winner slant = %s(slant_angle)   (%.2e vs %.2e)"
      % (res["B5a_winner"], min(wp, wm), max(wp, wm)))
SGN = +1.0 if wp < wm else -1.0

# ------------------------------------------------------------------ B5b
print("")
print("B5b  PER-ORDER LADDER (winning sign)")
lad = []
for phi_deg in (0.0, 10.0, 20.0, 35.0):
    for mount, theta in (("normal", 0.0), ("oblique25", np.deg2rad(25.0))):
        orc = {p: (oracle_s(phi_deg, theta, p) if phi_deg > 0
                   else oracle_v(theta, p)) for p in ("te", "tm")}
        for M in (5, 6, 7, 8):
            sl = None if phi_deg == 0 else (SGN * np.tan(np.deg2rad(phi_deg)),
                                            0.0)
            d2, ylk, clo = two_d(CELL, sl, theta, M)
            row = {"phi": phi_deg, "mount": mount, "M": M,
                   "te": perorder(d2["te"], orc["te"]),
                   "tm": perorder(d2["tm"], orc["tm"]),
                   "yleak": ylk, "closure": clo}
            lad.append(row)
            print("  phi=%4.0f %-9s M=%d  te %.2e  tm %.2e  yleak %.1e  "
                  "|R+T-1| %.2e" % (phi_deg, mount, M, row["te"], row["tm"],
                                    ylk, clo))
res["B5b_ladder"] = lad
res["B5b_te_worst_M8"] = max(r["te"] for r in lad if r["M"] == 8)
res["B5b_te_worst_M8_vertical"] = max(r["te"] for r in lad
                                      if r["M"] == 8 and r["phi"] == 0.0)
res["B5b_te_worst_M8_slanted"] = max(r["te"] for r in lad
                                     if r["M"] == 8 and r["phi"] > 0.0)
res["B5b_yleak_worst"] = max(r["yleak"] for r in lad)

# ------------------------------------------------------------------ B6
print("")
print("B6  SLANT x OUT-OF-PLANE stripe vs pmm_jones_1d_slanted")
TIL = uniaxial(1.5, 1.7, np.deg2rad(35.0), np.deg2rad(25.0))
AIR = np.eye(3, dtype=complex)
OCELL = np.zeros((2, 2, 3, 3), dtype=complex)
OCELL[0, :] = TIL
OCELL[1, :] = AIR


def oracle_oop(phi_deg, theta, degree=30):
    if phi_deg == 0.0:
        o, R, T, J = pmm_jones_1d(
            PX, TIL, AIR, NSUB, NSUP, DEPTH, 0.5, WL, angle=theta,
            degree=degree, far_field_orders=15)
    else:
        o, R, T, J = pmm_jones_1d_slanted(
            PX, TIL, AIR, NSUB, NSUP, DEPTH, 0.5, WL, np.deg2rad(phi_deg),
            angle=theta, degree=degree, far_field_orders=15)
    return ({int(m): (float(R[0, i]), float(T[0, i]))
             for i, m in enumerate(o)},
            {int(m): (float(R[1, i]), float(T[1, i]))
             for i, m in enumerate(o)})


oop_rows = []
for phi_deg in (0.0, 20.0, 35.0):
    for mount, theta in (("normal", 0.0), ("oblique25", np.deg2rad(25.0))):
        ox, oy = oracle_oop(phi_deg, theta)
        oxd, oyd = oracle_oop(phi_deg, theta, degree=26)
        drift = max(max(abs(ox[m][0] - oxd[m][0]), abs(ox[m][1] - oxd[m][1]),
                        abs(oy[m][0] - oyd[m][0]), abs(oy[m][1] - oyd[m][1]))
                    for m in ORD_CMP if m in ox and m in oxd)
        sl = None if phi_deg == 0 else (SGN * np.tan(np.deg2rad(phi_deg)), 0.0)
        d2, ylk, clo = two_d(OCELL, sl, theta, 8)
        row = {"phi": phi_deg, "mount": mount, "M": 8,
               "ex": perorder(d2["tm"], ox), "ey": perorder(d2["te"], oy),
               "oracle_drift": drift, "yleak": ylk, "closure": clo}
        oop_rows.append(row)
        print("  phi=%4.0f %-9s  Ex %.2e  Ey %.2e  (oracle drift %.1e) "
              "yleak %.1e |R+T-1| %.2e"
              % (phi_deg, mount, row["ex"], row["ey"], drift, ylk, clo))
res["B6_rows"] = oop_rows
res["B6_worst_ex"] = max(r["ex"] for r in oop_rows)
res["B6_worst_ey"] = max(r["ey"] for r in oop_rows)
res["B6_worst_ex_vertical"] = max(r["ex"] for r in oop_rows if r["phi"] == 0)
res["B6_worst_ex_slanted"] = max(r["ex"] for r in oop_rows if r["phi"] > 0)
res["B6_yleak_worst"] = max(r["yleak"] for r in oop_rows)

# the coverage gap this closes: the hybrid REFUSES slant x out-of-plane
print("")
print("B6b  the shipped HYBRID on the same combination")
try:
    from lumenairy.elements.pmm import PMM2DStackHybrid
    hs = PMM2DStackHybrid(PX, PY, n_superstrate=NSUP, n_substrate=NSUB,
                          n_orders=5)
    hs.add_layer(DEPTH, eps_tensor_cell=OCELL,
                 slant=(SGN * np.tan(np.deg2rad(35.0)), 0.0))
    hs.set_source(WL, theta=0.0, phi=0.0)
    hs.solve()
    res["B6b_hybrid"] = "SOLVED (no refusal)"
except NotImplementedError as exc:
    res["B6b_hybrid"] = "NotImplementedError: " + str(exc)[:200]
except Exception as exc:                                    # noqa: BLE001
    res["B6b_hybrid"] = type(exc).__name__ + ": " + str(exc)[:200]
print("  " + res["B6b_hybrid"][:160])

res["wall_s"] = time.time() - t00
write("g3_stripe", res)
