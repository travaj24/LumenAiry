"""V8 -- re-derive every numeric constant in
``tests/unit/test_fix_rcwa_even_sector_wsl.py``, on both sides, at the thread
count the caller pinned.

For each of the file's five bars this reports the quantity the assertion reads,
on the POST arm (the shipped code) and on the PRE arm (the transcribed
``48c8747`` body, same process), so the two-sided margin the file claims can be
checked rather than read.  The thread count is the axis the whole defect turned
on, so the driver runs this once per setting.

Usage:  OPENBLAS_NUM_THREADS=<n> python v8_testbars.py <out.json>
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v_fixtures as V  # noqa: E402

# the test file's own fixture, rebuilt here from its stated parameters
_P, _WL, _DEPTH = 0.5e-6, 0.6e-6, 0.2e-6
_N_SUB, _N_SUP = 1.5, 1.0
_ETA_LADDER = (0.0, -1e-20, -5.7e-20, -1.6e-18, -2.7e-16, -2.9e-15, -5.5e-15,
               -1e-13, -1e-12, 1e-20, 2.9e-15, 1e-12)
_KZ = (0.9, 1.5, 2.0, 0.30, 8.7)


def _cell():
    return V.uniaxial_cell(S=48, twist=0.7, no=1.5, ne=1.7, bg=2.25)


def _solve(symmetry, theta=0.0, phi=0.0, n_orders=5):
    from lumenairy.elements.rcwa import rcwa_jones_2d
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return rcwa_jones_2d(_P, _P, _cell(), _N_SUB, _N_SUP, _DEPTH, _WL,
                             theta=theta, phi=phi, n_orders_x=n_orders,
                             n_orders_y=n_orders, symmetry=symmetry)


def bars():
    from lumenairy.elements.rcwa import _core as rc
    out = {}
    # ---- BAR 1: the selector, engineered input --------------------------
    lam2 = np.array([-kz ** 2 + 1j * eta for kz in _KZ
                     for eta in _ETA_LADDER], dtype=complex)
    lam = np.asarray(rc._sqrt_decay(lam2))
    out["bar1_n_incoming"] = int(np.sum(lam.imag < 0))
    out["bar1_n_total"] = int(lam.size)
    out["bar1_min_Re"] = float(np.min(lam.real))
    out["bar1_root_residual_rel"] = float(
        np.max(np.abs(lam ** 2 - lam2)) / np.max(np.abs(lam2)))
    # ---- BAR 1b: evanescent branch stays decaying ------------------------
    lam2e = np.array([g ** 2 + 1j * eta for g in (0.5, 3.0, 70.0)
                      for eta in _ETA_LADDER], dtype=complex)
    out["bar1b_min_Re"] = float(np.min(np.asarray(rc._sqrt_decay(lam2e)).real))
    # ---- BAR 2: a physically signed root must survive --------------------
    sig = {}
    for frac in (7.9e-2, 7.9e-3, 1.26e-2, 1e-4, 1e-6, 1e-7, 5e-8, 2e-8):
        a = frac / np.sqrt(1.0 - frac ** 2)
        r = np.array([a - 1j], dtype=complex)
        x = r ** 2
        got = np.asarray(rc._sqrt_decay(x))
        sig["%.3g" % frac] = dict(
            untouched=bool(np.array_equal(got, np.sqrt(x))),
            imag_still_negative=bool(got[0].imag < 0))
    out["bar2_signal_side"] = sig
    # ---- BAR 3 / 4: the solves ------------------------------------------
    def read():
        full, even = _solve(False), _solve(True)
        return dict(
            closure_full=V.closure_defect_jones(full),
            closure_even=V.closure_defect_jones(even),
            dR=float(np.max(np.abs(np.asarray(full[1])
                                   - np.asarray(even[1])))),
            dJ=float(np.max(np.abs(np.asarray(full[3])
                                   - np.asarray(even[3])))),
            fold_bit_identical=bool(np.array_equal(np.asarray(full[1]),
                                                   np.asarray(even[1]))),
            closure_obl_03=V.closure_defect_jones(
                _solve(True, theta=0.3, phi=0.0, n_orders=4)),
            closure_obl_02_07=V.closure_defect_jones(
                _solve(True, theta=0.2, phi=0.7, n_orders=4)),
        )
    out["post"] = read()
    with V.PreSqrtDecay():
        out["pre"] = read()
    # ---- BAR 5: the on-cut census on the real operator -------------------
    def census():
        eig = V.EigSpy()
        with eig:
            _solve(True)
            _solve(False)
        band = getattr(rc, "_CUT_BAND_REL", 1e-8)
        total = bad = 0
        for w in eig.seen:
            lm = np.asarray(rc._sqrt_decay(w))
            scale = max(float(np.max(np.abs(lm))), 1.0)
            oc = np.abs(lm.real) <= band * scale
            total += int(oc.sum())
            bad += int(np.sum(oc & (lm.imag < 0)))
        return dict(on_cut=total, incoming=bad)
    out["bar5_post"] = census()
    with V.PreSqrtDecay():
        out["bar5_pre"] = census()
    return out


def main():
    V.require_local_tree()
    o = sys.argv[1]
    V.claim_output(o)
    b = bars()
    b["openblas_num_threads"] = os.environ.get("OPENBLAS_NUM_THREADS",
                                               "unpinned")
    V.dump(o, b)
    p, q = b["post"], b["pre"]
    print("threads=%-8s BAR1 incoming=%d/%d minRe=%.1e resid=%.2e | "
          "BAR1b minRe=%.3g" % (
              b["openblas_num_threads"], b["bar1_n_incoming"],
              b["bar1_n_total"], b["bar1_min_Re"],
              b["bar1_root_residual_rel"], b["bar1b_min_Re"]))
    print("   BAR3 closure_full post=%+.3e pre=%+.3e | closure_even "
          "post=%+.3e pre=%+.3e   (bar 1e-9)" % (
              p["closure_full"], q["closure_full"], p["closure_even"],
              q["closure_even"]))
    print("   BAR3-obl t0.3 post=%+.3e pre=%+.3e | t0.2p0.7 post=%+.3e "
          "pre=%+.3e" % (p["closure_obl_03"], q["closure_obl_03"],
                         p["closure_obl_02_07"], q["closure_obl_02_07"]))
    print("   BAR4 dR post=%.3e pre=%.3e | dJ post=%.3e pre=%.3e  (bar 1e-11)"
          % (p["dR"], q["dR"], p["dJ"], q["dJ"]))
    print("   BAR5 on-cut incoming post=%d/%d pre=%d/%d" % (
        b["bar5_post"]["incoming"], b["bar5_post"]["on_cut"],
        b["bar5_pre"]["incoming"], b["bar5_pre"]["on_cut"]))
    fails_pre = []
    if abs(q["closure_full"]) < 1e-9:
        fails_pre.append("BAR3[False]")
    if abs(q["closure_even"]) < 1e-9:
        fails_pre.append("BAR3[True]")
    if abs(q["closure_obl_03"]) < 1e-9:
        fails_pre.append("BAR3-obl[0.3]")
    if abs(q["closure_obl_02_07"]) < 1e-9:
        fails_pre.append("BAR3-obl[0.2,0.7]")
    if q["dR"] < 1e-11 and q["dJ"] < 1e-11:
        fails_pre.append("BAR4")
    print("   PRE-ARM WOULD PASS (fail-before does NOT fire): %s"
          % (", ".join(fails_pre) if fails_pre else "none"))


if __name__ == "__main__":
    main()
