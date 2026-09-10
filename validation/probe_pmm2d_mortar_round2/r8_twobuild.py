"""R8 -- the readings the SHIPPED gates quote, measured on BOTH builds.

``python r8_twobuild.py``

Several of the round-2 gates' comments state a number as "WIN / WSL".  This
probe measures exactly those numbers so the attribution is a measurement and
not a habit:

  * the D1 FAIL-BEFORE arm (ordinary vs sliver error against the exact 1-D
    oracle, and both closures);
  * the spurious-spectrum predictor constants;
  * the mortar-solve rcond census on the shipped taper;
  * the all-host mortar error against the analytic slab;
  * the plain-1-D interface readings behind the ~1850 decision;
  * the mortar operators' rcond at the two sliver widths the gate quotes.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
import time

import numpy as np

import lumenairy

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
assert os.path.abspath(lumenairy.__file__).startswith(_ROOT), lumenairy.__file__
print(f"[arm] lumenairy = {lumenairy.__file__}", flush=True)

sys.path.insert(0, os.path.join(_ROOT, "tests", "unit"))
import warnings  # noqa: E402

import scipy.linalg as sla  # noqa: E402
import test_fix_pmm2d_mortar_round2 as G  # noqa: E402

from lumenairy.elements.pmm import _core as _pc  # noqa: E402
from lumenairy.elements.pmm import twod_staggered as _ts  # noqa: E402

RES = {}
T0 = time.time()


def _log(m):
    print(f"[{time.time() - T0:6.1f}s] {m}", flush=True)


def main():
    # ---- D1 fail-before -----------------------------------------------------
    o1d, R1d, T1d = G._y_uniform_oracle(14)[:3]
    R1d, T1d = np.atleast_2d(R1d), np.atleast_2d(T1d)
    prev, prev_rc = _ts.PMM2D_STAG_MIN_SEG_GUARD, _pc._MORTAR_RCOND_REFUSE
    _ts.PMM2D_STAG_MIN_SEG_GUARD = False
    _pc._MORTAR_RCOND_REFUSE = 0.0
    try:
        fb = {}
        for lab, d in (("ordinary_0.30", 0.30), ("sliver_1e-5", 1e-5)):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                o, R, T = G._y_uniform_stack(d, 6).solve(jones=False)
            fb[lab] = {"err": G._score(o, R, T, o1d, R1d, T1d),
                       "closure": float(np.max(np.abs(R.sum(1) + T.sum(1)
                                                      - 1.0))),
                       "warnings": [str(x.message)[:60] for x in w]}
        fb["ratio"] = fb["sliver_1e-5"]["err"] / fb["ordinary_0.30"]["err"]
    finally:
        _ts.PMM2D_STAG_MIN_SEG_GUARD = prev
        _pc._MORTAR_RCOND_REFUSE = prev_rc
    RES["d1_fail_before"] = fb
    _log(f"D1 fail-before: ordinary {fb['ordinary_0.30']['err']:.4e} "
         f"(clo {fb['ordinary_0.30']['closure']:.2e}), sliver "
         f"{fb['sliver_1e-5']['err']:.4e} (clo "
         f"{fb['sliver_1e-5']['closure']:.2e}) -> {fb['ratio']:.2f}x")

    # ---- the rcond census on the shipped taper -----------------------------
    prevc, _pc._MORTAR_SOLVE_CENSUS = _pc._MORTAR_SOLVE_CENSUS, []
    try:
        st = G.PMM2DStackPure(G._P, n_modes=5, n_orders=1,
                              layer_grids="per-layer")
        st.add_tapered_pillar(0.24, eps_pillar=G._EPS_P, eps_host=G._EPS_H,
                              x_bounds_bottom=[0.1873 * G._P, 0.7241 * G._P],
                              y_bounds_bottom=[0.1873 * G._P, 0.7241 * G._P],
                              x_bounds_top=[0.2917 * G._P, 0.6109 * G._P],
                              y_bounds_top=[0.2917 * G._P, 0.6109 * G._P],
                              n_slices=4)
        st.set_source(G._WL, theta=G._TH, phi=G._PH)
        st.solve(jones=False)
        h = [r[2] for r in _pc._MORTAR_SOLVE_CENSUS if not r[3]]
    finally:
        _pc._MORTAR_SOLVE_CENSUS = prevc
    RES["taper_rcond"] = {"n": len(h), "min": min(h), "max": max(h)}
    _log(f"taper rcond: {len(h)} solves, {min(h):.4e} .. {max(h):.4e}")

    # ---- the sliver-width rcond readings the gate quotes -------------------
    _ts.PMM2D_STAG_MIN_SEG_GUARD = False
    try:
        rcs = {}
        for d in (1e-5, 1e-7):
            (L, _R), _ga, _gb = G._mortar_pair(d, 5)
            lu, piv = sla.lu_factor(L)
            gec = sla.get_lapack_funcs("gecon", (L,))
            rc, _i = gec(lu, float(np.max(np.sum(np.abs(L), axis=0))))
            rcs[f"{d:g}"] = float(rc)
    finally:
        _ts.PMM2D_STAG_MIN_SEG_GUARD = prev
    RES["sliver_rcond"] = rcs
    _log(f"sliver rcond (M=5): {rcs}")

    # ---- the spurious-spectrum predictor -----------------------------------
    _ts.PMM2D_STAG_MIN_SEG_GUARD = False
    try:
        k0 = 2 * np.pi / G._WL
        cs = {}
        for M in (4, 5):
            row = []
            for d in (1e-3, 1e-4, 1e-5, 1e-6):
                xb = np.array([0.0, (0.5 - d / 2) * G._P,
                               (0.5 + d / 2) * G._P, G._P])
                sol = G.Granet2DTransverseE(G._P, G._P, xb, xb, M, G._tile(),
                                            alpha0x=0.0, alpha0y=0.0, k0=k0)
                lam = G._region_modes(sol)[2]
                row.append(float(np.max(np.abs(lam))) * 4.0 * k0
                           * float(np.min(sol.bx.Jn)) / (M * (M + 1)))
            cs[M] = row
    finally:
        _ts.PMM2D_STAG_MIN_SEG_GUARD = prev
    RES["predictor"] = {str(k): v for k, v in cs.items()}
    _log("predictor c: " + "; ".join(
        f"M{M} " + " ".join(f"{v:.6f}" for v in row) for M, row in cs.items()))

    # ---- the all-host mortar error against the ANALYTIC slab ---------------
    import math
    k0 = 2 * np.pi / G._WL
    n0, nh = 1.0, math.sqrt(G._EPS_H)
    kx = k0 * n0 * math.sin(G._TH)
    kz0 = np.sqrt((k0 * n0) ** 2 - kx ** 2 + 0j)
    kzh = np.sqrt((k0 * nh) ** 2 - kx ** 2 + 0j)
    ph = np.exp(2j * kzh * (3.0 * 0.06))
    an = {}
    for pol, r01 in (("TE", (kz0 - kzh) / (kz0 + kzh)),
                     ("TM", (nh ** 2 * kz0 - n0 ** 2 * kzh)
                      / (nh ** 2 * kz0 + n0 ** 2 * kzh))):
        r = r01 * (1.0 - ph) / (1.0 - r01 * r01 * ph)
        an[pol] = float(abs(r) ** 2)
    _ts.PMM2D_STAG_MIN_SEG_GUARD = False
    try:
        ah = {}
        for d in (0.30, 1e-2):
            st = G.PMM2DStackPure(G._P, n_modes=5, n_orders=1,
                                  layer_grids="per-layer")
            yw = [0.27 * G._P, 0.61 * G._P]
            for xw in ([0.21 * G._P, 0.68 * G._P],
                       [(0.5 - d / 2) * G._P, (0.5 + d / 2) * G._P],
                       [0.33 * G._P, 0.79 * G._P]):
                st.add_layer(0.06, eps=G._EPS_H, x_walls=xw, y_walls=yw)
            st.set_source(G._WL, theta=G._TH, phi=0.0)
            o, R, T = st.solve(jones=False)
            p0 = int(np.where((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
            ah[f"{d:g}"] = max(abs(float(R[1, p0]) - an["TE"]),
                               abs(float(R[0, p0]) - an["TM"]))
    finally:
        _ts.PMM2D_STAG_MIN_SEG_GUARD = prev
    RES["allhost_vs_analytic_M5"] = ah
    _log(f"all-host vs analytic (M=5): {ah}")

    # ---- the plain 1-D interface readings behind the ~1850 decision ---------
    seen = []
    real = _pc._interface_smatrix
    import lumenairy.elements.pmm.stack as _st1d

    def _patched(Wa, Va, Wb, Vb):
        for A in (np.asarray(Wb), np.asarray(Vb)):
            lu, piv = sla.lu_factor(A)
            gec = sla.get_lapack_funcs("gecon", (A,))
            rc, _i = gec(lu, float(np.max(np.sum(np.abs(A), axis=0))))
            seen.append(float(rc))
        return real(Wa, Va, Wb, Vb)

    a0, a1 = 0.27865, 0.62505
    _st1d._interface_smatrix = _patched
    try:
        one_d = {}
        for d in (1e-4, 1e-5):
            seen.clear()
            st = G.PMMStack(G._P, degree=12, far_field_orders=5)
            st.add_layer(0.08, segments=[(a0, G._EPS_H), (a1 - a0, G._EPS_P),
                                         (1 - a1, G._EPS_H)])
            b0, b1 = a0 - d, a1 + d
            st.add_layer(0.08, segments=[(b0, G._EPS_H), (b1 - b0, G._EPS_P),
                                         (1 - b1, G._EPS_H)])
            st.set_source(G._WL, theta=G._TH)
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    _o, R, T = st.solve()[:3]
                one_d[f"{d:g}"] = {
                    "min_rcond": min(seen),
                    "RplusT": float(np.max(np.atleast_2d(R).sum(1)
                                           + np.atleast_2d(T).sum(1))),
                    "n_warnings": len(w)}
            except ValueError as exc:
                one_d[f"{d:g}"] = {"min_rcond": min(seen),
                                   "REFUSED": str(exc)[:70]}
    finally:
        _st1d._interface_smatrix = real
    RES["one_d_interface"] = one_d
    _log(f"1-D interface: {one_d}")

    tag = os.environ.get("R_TAG", "win")
    path = os.path.join(HERE, f"r8_twobuild_{tag}.json")
    with open(path, "w") as fh:
        json.dump(RES, fh, indent=1, sort_keys=True, default=float)
    _log(f"wrote {path}")


if __name__ == "__main__":
    main()
