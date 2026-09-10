"""The TEST FIXTURES, measured at the sizes ``tests/unit/
test_pmm2d_staggered_slant.py`` actually runs (grids <= (3,3), M <= 8), on BOTH
builds -- so every bar in that file is derived from a measured cross-build gap
rather than from the campaign's larger probe sizes.

Blocks, in the test's order:
  F1  slant-0 bit-identity + the prototype-sized null / phase rows
  F2  dispersion at the test's M
  F3  stripe vs the 1-D oracles at the test's M
  F4  census / depth / split / no-floor at the test's sizes
  F5  the 2-D PILLAR on a (3,3) grid: hybrid arm (both signs) + a 2-rung
      staircase (both marching directions)
  F6  cost ratio at the test's M
"""
import time
import warnings

import numpy as np
from _lib import uniaxial, write

from lumenairy.elements.pmm import (
    PMM2DStackHybrid,
    PMM2DStackPure,
    pmm_efficiency_1d,
    pmm_efficiency_1d_slanted,
    pmm_jones_1d,
    pmm_jones_1d_slanted,
    pmm_jones_2d_staggered,
)
from lumenairy.elements.pmm.twod_staggered import (
    Granet2DTransverseE,
    _region_modes,
    _region_modes_oop,
)

res = {}
t00 = time.time()
T35 = float(np.tan(np.deg2rad(35.0)))

# --------------------------------------------------------------- F3 stripe
PX = PY = 0.75
WL = 1.0
DEPTH = 0.30
NR, NGV = 2.0, 1.0
NSUP, NSUB = 1.0, 1.5
CELL = np.array([[NR ** 2, NR ** 2], [NGV ** 2, NGV ** 2]], dtype=complex)
ORD_CMP = [-1, 0, 1]


def two_d(cell, slant, theta, M, n_orders=3):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o, R, T, _J = pmm_jones_2d_staggered(
            PX, PY, cell, NSUB, NSUP, DEPTH, WL, n_modes=M,
            n_orders=n_orders, theta=theta, phi=0.0, slant=slant)
    idx = {int(m): i for i, (m, n) in enumerate(o) if n == 0}
    out = {"tm": {}, "te": {}}
    for m, i in idx.items():
        out["tm"][m] = (float(R[0, i]), float(T[0, i]))
        out["te"][m] = (float(R[1, i]), float(T[1, i]))
    return out


def perorder(a, b):
    return max(max(abs(a[m][0] - b[m][0]), abs(a[m][1] - b[m][1]))
               for m in ORD_CMP if m in a and m in b)


print("F3  y-uniform stripe vs the 1-D oracles (test sizes, M = 7)")
f3 = []
for phi_deg in (0.0, 20.0, 35.0):
    for mount, theta in (("normal", 0.0), ("oblique25", np.deg2rad(25.0))):
        for pol in ("te", "tm"):
            if phi_deg == 0.0:
                o, R, T = pmm_efficiency_1d(
                    PX, NR, NGV, NSUB, NSUP, DEPTH, 0.5, WL, angle=theta,
                    polarization=pol, degree=22, far_field_orders=15)
            else:
                o, R, T = pmm_efficiency_1d_slanted(
                    PX, NR, NGV, NSUB, NSUP, DEPTH, 0.5, WL,
                    np.deg2rad(phi_deg), angle=theta, polarization=pol,
                    degree=22, far_field_orders=15)
            orc = {int(m): (float(R[i]), float(T[i])) for i, m in enumerate(o)}
            row = {"phi": phi_deg, "mount": mount, "pol": pol}
            for sgn, lab in ((+1.0, "plus"), (-1.0, "minus")):
                sl = None if phi_deg == 0 else (
                    sgn * np.tan(np.deg2rad(phi_deg)), 0.0)
                row[lab] = perorder(two_d(CELL, sl, theta, 7)[pol], orc)
                if phi_deg == 0.0:
                    row["minus"] = row["plus"]
                    break
            f3.append(row)
            print("  phi=%4.0f %-9s %s  +tan %.2e   -tan %.2e"
                  % (phi_deg, mount, pol, row["plus"], row["minus"]))
res["F3"] = f3
res["F3_te_worst_plus"] = max(r["plus"] for r in f3 if r["pol"] == "te")
res["F3_te_worst_plus_slanted"] = max(r["plus"] for r in f3
                                      if r["pol"] == "te" and r["phi"] > 0)
res["F3_te_best_minus_slanted"] = min(r["minus"] for r in f3
                                      if r["pol"] == "te" and r["phi"] > 0)
res["F3_tm_worst_plus"] = max(r["plus"] for r in f3 if r["pol"] == "tm")

print("")
print("F3b  slant x OUT-OF-PLANE stripe vs pmm_jones_1d_slanted (M = 7)")
TIL = uniaxial(1.5, 1.7, np.deg2rad(35.0), np.deg2rad(25.0))
AIR = np.eye(3, dtype=complex)
OCELL = np.zeros((2, 2, 3, 3), dtype=complex)
OCELL[0, :] = TIL
OCELL[1, :] = AIR
f3b = []
for phi_deg in (0.0, 35.0):
    for mount, theta in (("normal", 0.0), ("oblique25", np.deg2rad(25.0))):
        if phi_deg == 0.0:
            o, R, T, J = pmm_jones_1d(PX, TIL, AIR, NSUB, NSUP, DEPTH, 0.5,
                                      WL, angle=theta, degree=30,
                                      far_field_orders=15)
        else:
            o, R, T, J = pmm_jones_1d_slanted(
                PX, TIL, AIR, NSUB, NSUP, DEPTH, 0.5, WL,
                np.deg2rad(phi_deg), angle=theta, degree=30,
                far_field_orders=15)
        ox = {int(m): (float(R[0, i]), float(T[0, i]))
              for i, m in enumerate(o)}
        oy = {int(m): (float(R[1, i]), float(T[1, i]))
              for i, m in enumerate(o)}
        sl = None if phi_deg == 0 else (np.tan(np.deg2rad(phi_deg)), 0.0)
        d2 = two_d(OCELL, sl, theta, 7)
        row = {"phi": phi_deg, "mount": mount,
               "ex": perorder(d2["tm"], ox), "ey": perorder(d2["te"], oy)}
        f3b.append(row)
        print("  phi=%4.0f %-9s  Ex %.2e   Ey %.2e"
              % (phi_deg, mount, row["ex"], row["ey"]))
res["F3b"] = f3b
res["F3b_worst_ex"] = max(r["ex"] for r in f3b)
res["F3b_worst_ex_vertical"] = max(r["ex"] for r in f3b if r["phi"] == 0)

# --------------------------------------------------------------- F5 pillar
print("")
print("F5  the 2-D PILLAR on a (3,3) grid (test sizes)")
PXP = PYP = 1.2
WLP = 1.0
DEPP = 0.8
TP = 2.0 * PXP / 3.0 / DEPP            # walk 2/3 of a period over the depth
NSUBP = 1.5
KEYS = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
print("     t = %.4f  (walk %.3f = 2 h on h = px/3)" % (TP, TP * DEPP))


def cell3(shift_cells):
    """Pillar occupying ONE grid cell in x and y, shifted by whole cells."""
    c = np.full((3, 3), 1.0, dtype=complex)
    c[int(shift_cells) % 3, 1] = 4.0
    return c


def pure(cells, slants, depths, M, n_orders, theta, phi):
    st = PMM2DStackPure(PXP, PYP, n_superstrate=1.0, n_substrate=NSUBP,
                        n_modes=M, n_orders=n_orders)
    for c, s, d in zip(cells, slants, depths):
        st.add_layer(d, eps_cell=c, slant=s)
    st.set_source(WLP, theta=theta, phi=phi)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return st.solve(jones=True)


def vec(o, R, T):
    idx = {(int(a), int(b)): i for i, (a, b) in enumerate(o)}
    return np.concatenate([np.array([R[:, idx[k]] for k in KEYS]).ravel(),
                           np.array([T[:, idx[k]] for k in KEYS]).ravel()])


MOUNTS = (("normal", 0.0, 0.0),
          ("conical", np.deg2rad(20.0), np.deg2rad(35.0)))
f5 = {"selfmove": [], "hybrid": [], "staircase": [], "vertical_gap": []}
ref = {}
for mn, th, ph in MOUNTS:
    o, R, T, J = pure([cell3(0)], [(TP, 0.0)], [DEPP], 4, 3, th, ph)
    ref[mn] = vec(o, R, T)
    for M in (3, 5):
        o2, R2, T2, J2 = pure([cell3(0)], [(TP, 0.0)], [DEPP], M, 3, th, ph)
        d = float(np.max(np.abs(vec(o2, R2, T2) - ref[mn])))
        f5["selfmove"].append({"mount": mn, "M": M, "vs_M4": d})
        print("     self-move %-8s M=%d vs M=4  %.3e" % (mn, M, d))
    ov, Rv, Tv, Jv = pure([cell3(0)], [None], [DEPP], 4, 3, th, ph)
    dv = float(np.max(np.abs(vec(ov, Rv, Tv) - ref[mn])))
    f5["vertical_gap"].append({"mount": mn, "vertical_vs_slanted": dv})
    print("     a VERTICAL pillar sits %.3e from the slanted answer (%s)"
          % (dv, mn))

for mn, th, ph in MOUNTS:
    for sgn in (+1.0, -1.0):
        prev = None
        for no in (3, 5, 7):
            hs = PMM2DStackHybrid(PXP, PYP, n_superstrate=1.0,
                                  n_substrate=NSUBP, n_orders=no)
            hs.add_layer(DEPP, eps_cell=cell3(0), slant=(sgn * TP, 0.0))
            hs.set_source(WLP, theta=th, phi=ph)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                oh, Rh, Th, Jh = hs.solve()
            v = vec(oh, Rh, Th)
            step = None if prev is None else float(np.max(np.abs(v - prev)))
            prev = v
            f5["hybrid"].append({"mount": mn,
                                 "sign": "+t" if sgn > 0 else "-t",
                                 "n_orders": no,
                                 "vs_pure": float(np.max(np.abs(v - ref[mn]))),
                                 "own_step": step})
            print("     hybrid %-8s %s n=%d  vs pure %.3e  own step %s"
                  % (mn, "+t" if sgn > 0 else "-t", no,
                     f5["hybrid"][-1]["vs_pure"],
                     "-" if step is None else "%.3e" % step))

for mn, th, ph in MOUNTS:
    o1, R1, T1, J1 = pure([cell3(0)], [(TP, 0.0)], [DEPP], 3, 3, th, ph)
    base = vec(o1, R1, T1)
    for direc in (+1, -1):
        for n in (1, 2):
            cells = [cell3(direc * k * (2 // n)) for k in range(n)]
            o2, R2, T2, J2 = pure(cells, [None] * n, [DEPP / n] * n, 3, 3,
                                  th, ph)
            d = float(np.max(np.abs(vec(o2, R2, T2) - base)))
            f5["staircase"].append({
                "mount": mn, "direction": "with" if direc > 0 else "against",
                "n": n, "vs_metric": d})
            print("     staircase %-8s march %-7s n=%d  %.3e"
                  % (mn, "with" if direc > 0 else "against", n, d))
res["F5"] = f5

# --------------------------------------------------------------- F6 cost
print("")
print("F6  cost ratio at the test's sizes")
K0 = 2.0 * np.pi
SCA = np.array([[4.0, 1.0], [1.0, 1.0]], dtype=complex)
OOPC = np.zeros((2, 2, 3, 3), dtype=complex)
OOPC[:, :] = np.eye(3)
OOPC[0, 0] = TIL
A0 = (0.25 * K0, 0.18 * K0)


def _t(fn, reps=3):
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


f6 = []
for M in (5, 6):
    def oop():
        _region_modes_oop(Granet2DTransverseE(
            1.2, 1.2, 2, 2, M, OOPC, alpha0x=A0[0], alpha0y=A0[1], k0=K0))

    def sl():
        _region_modes_oop(Granet2DTransverseE(
            1.2, 1.2, 2, 2, M, SCA, alpha0x=A0[0], alpha0y=A0[1], k0=K0,
            slant=(0.75, 0.0)))

    def ip():
        _region_modes(Granet2DTransverseE(
            1.2, 1.2, 2, 2, M, SCA, alpha0x=A0[0], alpha0y=A0[1], k0=K0))

    to, ts, ti = _t(oop), _t(sl), _t(ip)
    f6.append({"M": M, "oop_s": to, "slant_s": ts, "inplane_s": ti,
               "slant_over_oop": ts / to, "slant_over_inplane": ts / ti})
    print("  M=%d  slant/OOP %.2fx   slant/in-plane %.2fx" % (M, ts / to,
                                                              ts / ti))
res["F6"] = f6

res["wall_s"] = time.time() - t00
write("g7_testfixtures", res)
