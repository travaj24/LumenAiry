"""V7 -- every numeric bar in tests/unit/test_v5_20_12_rcwa_jones_2d_fff_nv.py,
re-measured (Task H.c durability audit).

Each row prints the quantity the test asserts, so the gap to the bar on BOTH
sides can be read off.  Run on Windows at 1 and 4 BLAS threads and on WSL.

    python v7_bars.py <lumenairy-root> [tag] [--fast]
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

from lumenairy.elements.rcwa import rcwa_jones_2d  # noqa: E402
from lumenairy.elements.rcwa import twod as _twod  # noqa: E402
from lumenairy.elements.rcwa.oned import rcwa_jones_1d_segments  # noqa: E402

TAG = sys.argv[2] if len(sys.argv) > 2 else "run"
FAST = "--fast" in sys.argv
PX, WL, DEPTH = 0.7e-6, 1.0e-6, 0.5e-6
OUT = {}


def rec(k, v):
    OUT[k] = float(v)
    print(f"  {k:44s} = {float(v):.6e}")


def _rot(phi, no, ne):
    c, s = np.cos(phi), np.sin(phi)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return R @ np.diag([ne ** 2, no ** 2, no ** 2]).astype(complex) @ R.T


def _stripe(er, eg, duty=0.5, Sx=64, Sy=8):
    xm = (np.arange(Sx) + 0.5) / Sx < duty
    c = np.zeros((Sx, Sy, 3, 3), complex)
    for ix in range(Sx):
        c[ix, :] = er if xm[ix] else eg
    return c


def q():
    w = warnings.catch_warnings()
    return w


# ---- 1. operator reduction: Cxx vs inv[[1/exx]]  (bar 1e-12) -------------
print("test_fff_nv_operator_reduces_to_li1996_on_stripe")
er2 = _rot(np.deg2rad(35.0), 1.5, 2.3)[:2, :2]
eg2 = np.diag([2.25, 2.25]).astype(complex)
Sx = 64
xm = (np.arange(Sx) + 0.5) / Sx < 0.5
cell2 = np.zeros((Sx, 8, 2, 2), complex)
for ix in range(Sx):
    cell2[ix, :] = er2 if xm[ix] else eg2
orders, _ = _twod._harmonic_orders_2d(9, 1)
Cxx, Cxy, Cyx, Cyy = _twod._li_convolutions_2d_tensor(
    cell2[:, :, 0, 0], cell2[:, :, 0, 1], cell2[:, :, 1, 0], cell2[:, :, 1, 1],
    orders, 9, 1, np)
inv_exx = np.linalg.inv(_twod._eps_convolution_2d(1.0 / cell2[:, :, 0, 0],
                                                 orders, 9, 1))
rec("t1 max|Cxx - inv_exx|  (bar 1e-12)", np.max(np.abs(Cxx - inv_exx)))

# ---- 2. the stripe reduction ---------------------------------------------
print("test_fff_nv_stripe_reduces_to_rigorous_1d")
ER = _rot(np.deg2rad(35.0), 1.5, 2.3)


def stripe_arms(eps_groove, No=11, n_ref=81):
    eg = np.diag([eps_groove] * 3).astype(complex)
    cell = _stripe(ER, eg)
    with q():
        warnings.simplefilter("ignore")
        _o, Rf, Tf, Jf = rcwa_jones_2d(PX, PX, cell, 1.5, 1.0, DEPTH, WL,
                                       n_orders_x=No, n_orders_y=1,
                                       formulation="fff_nv", symmetry=False)
        _o, Rl, Tl, Jl = rcwa_jones_2d(PX, PX, cell, 1.5, 1.0, DEPTH, WL,
                                       n_orders_x=No, n_orders_y=1,
                                       formulation="laurent", symmetry=False)
        _o, R1, T1, J1 = rcwa_jones_1d_segments(
            PX, [(0.5, ER), (0.5, eg)], 1.5, 1.0, DEPTH, WL, theta=0.0,
            n_orders=int(n_ref))
    return dict(
        ref_defect=abs(float(np.sum(R1) + np.sum(T1) - 2.0)),
        fff_defect=abs(float(np.sum(Rf) + np.sum(Tf)) - 2.0),
        minR=float(np.min(Rf)), minT=float(np.min(Tf)),
        ef=abs(float(np.sum(Rf) - np.sum(R1))),
        el=abs(float(np.sum(Rl) - np.sum(R1))),
        jf=float(np.max(np.abs(Jf - J1))),
        jl=float(np.max(np.abs(Jl - J1))))


a = stripe_arms(2.10)
rec("t2 ref 1-D closure defect @81 (bar 1e-9)", a["ref_defect"])
rec("t2 fff_nv closure defect @No=11 (bar 1e-9, then 6e-2)", a["fff_defect"])
rec("t2 min per-order Rf (bar >= 0)", a["minR"])
rec("t2 min per-order Tf (bar >= 0)", a["minT"])
rec("t2 ef", a["ef"])
rec("t2 el", a["el"])
rec("t2 ef/el  (bar 0.2)", a["ef"] / a["el"])
rec("t2 jf", a["jf"])
rec("t2 jl", a["jl"])
rec("t2 jf/jl  (bar 0.2)", a["jf"] / a["jl"])
# the same four on the COINCIDENT groove, for the record
b = stripe_arms(2.25)
rec("t2* coincident ef/el", b["ef"] / b["el"])
rec("t2* coincident jf/jl", b["jf"] / b["jl"])
rec("t2* coincident ref closure @81", b["ref_defect"])
rec("t2* coincident fff closure @11", b["fff_defect"])
# closure of the 2-D fff_nv arm at No = 9, 11, 13 (the bar's own spread)
for No in (9, 13):
    c = stripe_arms(2.10, No=No)
    rec(f"t2 fff_nv closure defect @No={No}", c["fff_defect"])

# ---- 3. the degeneracy test ----------------------------------------------
print("test_stripe_fixture_is_free_of_the_mode_match_degeneracy")


def worst(eps_groove):
    eg = np.diag([eps_groove] * 3).astype(complex)
    out = 0.0
    for n in range(11, 42, 2):
        with q():
            warnings.simplefilter("ignore")
            _o, R1, T1, _J = rcwa_jones_1d_segments(
                PX, [(0.5, ER), (0.5, eg)], 1.5, 1.0, DEPTH, WL, theta=0.0,
                n_orders=n)
        out = max(out, abs(float(np.sum(R1) + np.sum(T1) - 2.0)))
    return out


clean = worst(2.10)
degen = worst(2.25)
rec("t3 clean worst closure  (bar 1e-9)", clean)
rec("t3 degenerate worst closure", degen)
rec("t3 ratio degen/max(clean,1e-13)  (bar 1e5)",
    degen / max(clean, 1e-13))

# ---- 4. beats-laurent convergence ----------------------------------------
print("test_fff_nv_beats_laurent_convergence")
er4 = _rot(np.deg2rad(40.0), 1.6, 3.0)
eg4 = np.diag([1.0, 1.0, 1.0]).astype(complex)
cell4 = _stripe(er4, eg4)


def sumR4(No, form):
    with q():
        warnings.simplefilter("ignore")
        _o, R, _T, _J = rcwa_jones_2d(PX, PX, cell4, 1.5, 1.0, DEPTH, WL,
                                      n_orders_x=No, n_orders_y=1,
                                      formulation=form, symmetry=False)
    return float(np.sum(R))


with q():
    warnings.simplefilter("ignore")
    _o, Rref, _T, _J = rcwa_jones_1d_segments(
        PX, [(0.5, er4), (0.5, eg4)], 1.5, 1.0, DEPTH, WL, theta=0.0,
        n_orders=61)
ref4 = float(np.sum(Rref))
ef4 = abs(sumR4(9, "fff_nv") - ref4)
el4 = abs(sumR4(9, "laurent") - ref4)
rec("t4 ef", ef4)
rec("t4 el", el4)
rec("t4 ef/el  (bar 0.5)", ef4 / el4)

# ---- 5. lossy absorptance split ------------------------------------------
print("test_fff_nv_lossy_stripe_absorptance_split")
er5 = _rot(np.deg2rad(35.0), 1.5 + 0.15j, 2.3 + 0.15j)
eg5 = np.diag([2.25] * 3).astype(complex)
cell5 = _stripe(er5, eg5)


def absorp(form):
    with q():
        warnings.simplefilter("ignore")
        _o, R, T, _J = rcwa_jones_2d(PX, PX, cell5, 1.5, 1.0, DEPTH, WL,
                                     n_orders_x=11, n_orders_y=1,
                                     formulation=form, symmetry=False)
    return 1.0 - np.sum(R, 1) - np.sum(T, 1)


with q():
    warnings.simplefilter("ignore")
    _o, R1, T1, _J = rcwa_jones_1d_segments(
        PX, [(0.5, er5), (0.5, eg5)], 1.5, 1.0, DEPTH, WL, theta=0.0,
        n_orders=11)
A1 = float(np.mean(1.0 - np.sum(R1, 1) - np.sum(T1, 1)))
Af = float(np.mean(absorp("fff_nv")))
Al = float(np.mean(absorp("laurent")))
rec("t5 A1  (bars 0.05 / 0.95)", A1)
rec("t5 |Af - A1|", abs(Af - A1))
rec("t5 |Al - A1|", abs(Al - A1))
rec("t5 ratio  (bar 1.0)", abs(Af - A1) / abs(Al - A1))

# ---- 7. uniform routes to laurent ----------------------------------------
print("test_fff_nv_uniform_routes_to_laurent")
e7 = _rot(np.deg2rad(30.0), 1.5, 2.1)
cell7 = np.broadcast_to(e7, (32, 32, 3, 3)).copy()
with q():
    warnings.simplefilter("ignore")
    _o, Rf7, Tf7, Jf7 = rcwa_jones_2d(PX, PX, cell7, 1.5, 1.0, DEPTH, WL,
                                      n_orders_x=5, n_orders_y=5,
                                      formulation="fff_nv", symmetry=False)
    _o, Rl7, Tl7, Jl7 = rcwa_jones_2d(PX, PX, cell7, 1.5, 1.0, DEPTH, WL,
                                      n_orders_x=5, n_orders_y=5,
                                      formulation="laurent", symmetry=False)
rec("t7 max|Rf - Rl|  (bar 1e-12)", np.max(np.abs(Rf7 - Rl7)))
rec("t7 max|Jf - Jl|  (bar 1e-12)", np.max(np.abs(Jf7 - Jl7)))


# ---- 8/10. out-of-plane --------------------------------------------------
def _oop_stripe(No):
    th = np.deg2rad(35.0)
    c, s = np.cos(th), np.sin(th)
    Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    er = Ry @ np.diag([1.5 ** 2, 1.5 ** 2, 2.4 ** 2]).astype(complex) @ Ry.T
    eg = np.diag([2.1, 2.1, 2.1]).astype(complex)
    Sx = max(64, 4 * No + 4)
    xm = (np.arange(Sx) + 0.5) / Sx < 0.5
    cell = np.zeros((Sx, 8, 3, 3), complex)
    for ix in range(Sx):
        cell[ix, :] = er if xm[ix] else eg
    return cell


print("test_fff_nv_out_of_plane_converges_fast / same_limit_as_laurent")


def sumR8(No, form):
    with q():
        warnings.simplefilter("ignore")
        _o, R, T, _J = rcwa_jones_2d(PX, PX, _oop_stripe(No), 1.5, 1.0, DEPTH,
                                     WL, n_orders_x=No, n_orders_y=1,
                                     formulation=form, symmetry=False)
    return float(np.sum(R)), float(np.sum(R) + np.sum(T))


conv, e0 = sumR8(13, "fff_nv")
f7, _ = sumR8(7, "fff_nv")
l7, _ = sumR8(7, "laurent")
l15, _ = sumR8(15, "laurent")
rec("t8 |e0 - 2|  (bar 1e-9)", abs(e0 - 2.0))
rec("t8 |f7 - conv|  (bar 2e-5)", abs(f7 - conv))
rec("t8 |l7 - conv|", abs(l7 - conv))
rec("t8 ratio  (bar 0.5)", abs(f7 - conv) / abs(l7 - conv))
rec("t8 |l15 - conv|  (bar < |l7-conv|)", abs(l15 - conv))
g7 = abs(sumR8(7, "fff_nv")[0] - sumR8(7, "laurent")[0])
g15 = abs(sumR8(15, "fff_nv")[0] - sumR8(15, "laurent")[0])
g25 = abs(sumR8(25, "fff_nv")[0] - sumR8(25, "laurent")[0])
rec("t10 gap@7", g7)
rec("t10 gap@15", g15)
rec("t10 gap@25", g25)
rec("t10 gap25/gap7  (bar 0.5)", g25 / g7)

# ---- 9. berreman -----------------------------------------------------------
print("test_out_of_plane_matches_berreman_uniform")
from lumenairy.elements.berreman import berreman_jones_1d  # noqa: E402

th = np.deg2rad(35.0)
c9, s9 = np.cos(th), np.sin(th)
Ry = np.array([[c9, 0, s9], [0, 1, 0], [-s9, 0, c9]])
eps9 = Ry @ np.diag([1.5 ** 2, 1.5 ** 2, (2.4 + 0.05j) ** 2]) @ Ry.T
cell9 = np.broadcast_to(eps9, (8, 8, 3, 3)).copy()
worst9 = 0.0
for ang in (0.0, 25.0, 45.0):
    aa = np.deg2rad(ang)
    with q():
        warnings.simplefilter("ignore")
        _o, _R, _T, Jr = rcwa_jones_2d(0.6e-6, 0.6e-6, cell9, 1.5, 1.0, DEPTH,
                                       WL, theta=aa, phi=0.0, n_orders_x=3,
                                       n_orders_y=3, formulation="laurent",
                                       symmetry=False)
        _Rb, _Tb, Jb, _Jt = berreman_jones_1d([(eps9, DEPTH)], 1.5, 1.0, WL,
                                              theta=aa, phi=0.0)
    sv_r = np.sort(np.linalg.svd(Jr, compute_uv=False))
    sv_b = np.sort(np.linalg.svd(np.asarray(Jb), compute_uv=False))
    worst9 = max(worst9, float(np.max(np.abs(sv_r - sv_b))))
rec("t9 worst |sv_r - sv_b|  (bar 1e-11)", worst9)

# ---- 6. crossed cell (expensive) -----------------------------------------
if not FAST:
    print("test_fff_nv_crossed_cell_converges_and_beats_laurent")
    t0 = time.perf_counter()
    th6 = np.deg2rad(45.0)
    c6, s6 = np.cos(th6), np.sin(th6)
    R6 = np.array([[c6, -s6, 0.0], [s6, c6, 0.0], [0.0, 0.0, 1.0]])
    er6 = R6 @ np.diag([-8.0 + 1.2j, -2.0 + 0.8j,
                        -2.0 + 0.8j]).astype(complex) @ R6.T
    eg6 = np.diag([1.0, 1.0, 1.0]).astype(complex)

    def _sq(S):
        m = np.zeros((S, S), bool)
        m[S // 4:3 * S // 4, S // 4:3 * S // 4] = True
        cc = np.zeros((S, S, 3, 3), complex)
        for i in range(S):
            for j in range(S):
                cc[i, j] = er6 if m[i, j] else eg6
        return cc

    def sumR6(No, form):
        with q():
            warnings.simplefilter("ignore")
            _o, R, _T, _J = rcwa_jones_2d(0.5e-6, 0.5e-6,
                                          _sq(max(40, 4 * No + 4)), 1.5, 1.0,
                                          0.25e-6, WL, n_orders_x=No,
                                          n_orders_y=No, formulation=form,
                                          symmetry=False)
        return float(np.sum(R))

    ref6 = sumR6(17, "fff_nv")
    ef6 = [abs(sumR6(No, "fff_nv") - ref6) for No in (5, 7, 9, 11)]
    el6 = [abs(sumR6(No, "laurent") - ref6) for No in (5, 7, 9, 11)]
    for i, No in enumerate((5, 7, 9, 11)):
        rec(f"t6 ef[{No}]", ef6[i])
        rec(f"t6 el[{No}]", el6[i])
    rec("t6 ef[11]/el[11]  (bar 0.3)", ef6[-1] / el6[-1])
    rec("t6 ef[11]/ef[5]  (bar < 1)", ef6[-1] / ef6[0])
    rec("t6 monotone slack max(ef[i+1]-ef[i])  (bar 1e-9)",
        max(ef6[i + 1] - ef6[i] for i in range(3)))
    print(f"  [crossed cell took {time.perf_counter() - t0:.1f} s]")

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       f"_out_v7_{TAG}.json"), "w", encoding="cp1252") as fh:
    json.dump(OUT, fh, indent=1, sort_keys=True)
