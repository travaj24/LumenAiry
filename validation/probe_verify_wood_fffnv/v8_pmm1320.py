"""V8 -- test_v5_20_13's ladders, at the COINCIDENT and the DETUNED groove
(Follow-up A).

`tests/unit/test_v5_20_13_pmm_jones_2d_fff_nv.py` carries a two-stage RCWA
reference ladder (`_RCWA_REF_STAGES`), a per-rung closure `_scan` and a
`_corroborated_reference` picker.  All of that apparatus belongs to
`test_pmm_fff_nv_matches_rcwa_fff_nv`, which is the test that uses the
index-coincident cell (`rot(35 deg, 1.5, 2.3)` against `2.25 I`, n_sub = 1.5)
-- NOT to `test_pmm_fff_nv_stripe_reduces_to_rigorous_1d`, whose cell is
`rot(40 deg, 1.6, 3.0)` against air and carries no coincidence.

This probe scans both engines' ladders at eps_groove = 2.25 and 2.10 and
reports each rung's own lossless closure, so "the apparatus only exists to
survive the degeneracy" becomes a measurement.  It also records the OTHER
test's two bars.

    python v8_pmm1320.py <lumenairy-root> [tag]
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

from lumenairy.elements.pmm import pmm_jones_2d  # noqa: E402
from lumenairy.elements.rcwa import rcwa_jones_2d  # noqa: E402
from lumenairy.elements.rcwa._core import _EnergyError  # noqa: E402
from lumenairy.elements.rcwa.oned import rcwa_jones_1d_segments  # noqa: E402

TAG = sys.argv[2] if len(sys.argv) > 2 else "run"
PX, WL, DEPTH = 0.7e-6, 1.0e-6, 0.5e-6
OUT = {}


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


ER = _rot(np.deg2rad(35.0), 1.5, 2.3)


def scan(solve, ladder):
    rows = []
    for M in ladder:
        raised, sR, sT, close = None, float("nan"), float("nan"), float("inf")
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                R, T = solve(M)
                sR, sT = float(np.sum(R)), float(np.sum(T))
                close = abs(sR + sT - 2.0)
            except _EnergyError as exc:
                raised = type(exc).__name__
        rows.append(dict(M=M, sumR=sR, sumT=sT, close=close, raised=raised,
                         secs=time.perf_counter() - t0))
    return rows


print("A. the cross-solver test's ladders (test_pmm_fff_nv_matches_rcwa_fff_nv)")
for eps_g in (2.25, 2.10):
    eg = np.diag([eps_g] * 3).astype(complex)
    for Sx in (64, 128):
        rcell = _stripe(ER, eg, Sx=Sx)
        lad = ((7, 9, 11, 13, 15) if Sx == 64
               else (7, 11, 13, 15, 19, 21, 23, 25, 27))
        rows = scan(lambda M, c=rcell: rcwa_jones_2d(
            PX, PX, c, 1.5, 1.0, DEPTH, WL, n_orders_x=M, n_orders_y=1,
            formulation="fff_nv", symmetry=False)[1:3], lad)
        OUT[f"rcwa_{eps_g}_{Sx}"] = rows
        nclean = sum(1 for r in rows if r["close"] < 1e-3)
        print(f"  rcwa eps_g={eps_g} Sx={Sx}: clean {nclean}/{len(rows)}  "
              + "  ".join(f"{r['M']}:{r['close']:.2e}" for r in rows))
        print("      sumR: " + "  ".join(f"{r['M']}:{r['sumR']:.9f}"
                                         for r in rows))
    pcell = _stripe(ER, eg, Sx=64)
    prows = scan(lambda M, c=pcell: pmm_jones_2d(
        PX, PX, c, 1.5, 1.0, DEPTH, WL, degree=11, n_orders=M,
        formulation="fff_nv", symmetry=False)[1:3], (9, 11, 13))
    OUT[f"pmm_{eps_g}"] = prows
    nclean = sum(1 for r in prows if r["close"] < 1e-3)
    print(f"  pmm  eps_g={eps_g}        : clean {nclean}/{len(prows)}  "
          + "  ".join(f"{r['M']}:{r['close']:.2e}" for r in prows))
    print("      sumR: " + "  ".join(f"{r['M']}:{r['sumR']:.9f}"
                                     for r in prows)
          + "   secs " + " ".join(f"{r['secs']:.1f}" for r in prows))
    # cross-solver residual, every pmm rung against every rcwa Sx=64 rung
    rr = OUT[f"rcwa_{eps_g}_64"]
    worst_R = max(abs(p["sumR"] - r["sumR"]) for p in prows for r in rr
                  if r["close"] < 1e-3 and p["close"] < 1e-3) \
        if any(r["close"] < 1e-3 for r in rr) else float("nan")
    worst_T = max(abs(p["sumT"] - r["sumT"]) for p in prows for r in rr
                  if r["close"] < 1e-3 and p["close"] < 1e-3) \
        if any(r["close"] < 1e-3 for r in rr) else float("nan")
    OUT[f"cross_{eps_g}"] = dict(worst_R=worst_R, worst_T=worst_T)
    print(f"  cross-solver worst |dsumR| = {worst_R:.3e}   "
          f"|dsumT| = {worst_T:.3e}   (bar 4e-3)")

print()
print("B. test_pmm_fff_nv_stripe_reduces_to_rigorous_1d's own bars")
er2 = _rot(np.deg2rad(40.0), 1.6, 3.0)
eg2 = np.diag([1.0, 1.0, 1.0]).astype(complex)
cell2 = _stripe(er2, eg2)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    _o, Rref, Tref, _J = rcwa_jones_1d_segments(
        PX, [(0.5, er2), (0.5, eg2)], 1.5, 1.0, DEPTH, WL, theta=0.0,
        n_orders=61)
ref = float(np.sum(Rref))
refclose = abs(float(np.sum(Rref) + np.sum(Tref)) - 2.0)


def psumR(No, form):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _o, R, _T, _J = pmm_jones_2d(PX, PX, cell2, 1.5, 1.0, DEPTH, WL,
                                     degree=9, n_orders=No, formulation=form,
                                     symmetry=False)
    return float(np.sum(R))


t0 = time.perf_counter()
ef = abs(psumR(13, "fff_nv") - ref)
el = abs(psumR(13, "laurent") - ref)
OUT["t13_stripe"] = dict(refclose=refclose, ef=ef, el=el, ratio=ef / el,
                         secs=time.perf_counter() - t0)
print(f"  1-D reference closure (bar 1e-3) = {refclose:.4e}")
print(f"  ef = {ef:.6e}  (bar 1e-3)   el = {el:.6e}   "
      f"ef/el = {ef / el:.6e}  (bar 0.2)   [{time.perf_counter() - t0:.1f} s]")

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       f"_out_v8_{TAG}.json"), "w", encoding="cp1252") as fh:
    json.dump(OUT, fh, indent=1)
