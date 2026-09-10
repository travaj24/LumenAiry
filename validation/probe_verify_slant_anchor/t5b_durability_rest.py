"""TASK 5 (continued) -- the remaining gate bars: tests 12, 13, 14, 15 and the
two O2 message/site tests, plus the ARMED-guard cost.

Split from :mod:`t5_durability` only for runtime: these four re-run the gate's
own composition, layer-split and cross-engine fixtures, which are its three
most expensive.
"""
from __future__ import annotations

import os
import sys
import time
import warnings

if os.environ.get("LUM_ARM_TREE"):
    sys.path.insert(0, os.environ["LUM_ARM_TREE"])
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))

import _lib  # noqa: E402
import numpy as np  # noqa: E402
from t5_durability import gate  # noqa: E402


def main():                                                  # noqa: C901
    g = gate()
    R, t0 = {}, time.time()

    def rec(name, reading, bar, sense, note=""):
        m = (None if reading is None or not bar else
             (bar / reading if sense == "<" else reading / bar))
        R[name] = dict(reading=reading, bar=bar, sense=sense, margin_x=m,
                       note=note)
        print(f"  {name:52s} {reading!r:>26} {sense} {bar!r:<8} "
              f"margin {m if m is None else round(m, 4)}  {note}")

    # ---- test 12: the zeroth-order exemption at normal incidence -------
    sh, _o = g._v2_sheared(theta=0.0)
    o_s, A_s, kx = g._amps(sh)
    i0 = int(np.where(o_s == 0)[0][0])
    ph = g._P(kx, g.V2_W)
    R["t12_kx0_is_exactly_zero"] = dict(reading=float(kx[i0]), bar=0.0,
                                        sense="==", margin_x=None)
    R["t12_P0_is_exactly_one"] = dict(reading=[float(np.real(ph[i0])),
                                               float(np.imag(ph[i0]))],
                                      bar=[1.0, 0.0], sense="==",
                                      margin_x=None)
    rec("t12_distinct_phases", float(len(np.unique(np.round(np.angle(ph), 9)))),
        5.0, ">")
    stc, _oc = g._v2_stair(8, theta=0.0)
    o_c, A_c, _k = g._amps(stc)
    shipped = g._align(o_s, A_s, o_c, A_c)
    none = g._align(o_s, A_s / ph, o_c, A_c)
    rec("t12_none_over_shipped", none / shipped, 20.0, ">",
        f"shipped={shipped:.4e}")

    # ---- test 13: the walks add ----------------------------------------
    st = g._v2_stack()
    st.add_sheared_grating(0.24e-6, eps_ridge=4.20, eps_groove=1.45,
                           duty=0.45, shear=0.25, centre=0.5)
    st.add_sheared_grating(0.16e-6, eps_ridge=2.90, eps_groove=1.60,
                           duty=0.35, shear=0.10, centre=0.5)
    st, _o = g._v2_solve(st)
    o_s, A_s, kx = g._amps(st)
    w1, w2 = 0.25 * g.V2_P, 0.10 * g.V2_P
    Pw, P1, P2 = g._P(kx, w1 + w2), g._P(kx, w1), g._P(kx, w2)
    prev, rows = None, {}
    for K in (4, 6):
        sc = g._v2_stack()
        sc.add_tapered_ridges(0.24e-6,
                              ridges=[(0.5 * g.V2_P, 0.45 * g.V2_P,
                                       0.45 * g.V2_P, 4.20)],
                              eps_groove=1.45, n_slices=K, shear=0.25)
        sc.add_tapered_ridges(0.16e-6,
                              ridges=[(0.75 * g.V2_P, 0.35 * g.V2_P,
                                       0.35 * g.V2_P, 2.90)],
                              eps_groove=1.60, n_slices=K, shear=0.10)
        sc, _oc = g._v2_solve(sc)
        o_c, A_c, _k = g._amps(sc)
        rows[K] = dict(full=g._align(o_s, A_s, o_c, A_c),
                       only1=g._align(o_s, A_s / Pw * P1, o_c, A_c),
                       only2=g._align(o_s, A_s / Pw * P2, o_c, A_c),
                       none=g._align(o_s, A_s / Pw, o_c, A_c),
                       conj=g._align(o_s, A_s / Pw * np.conj(Pw), o_c, A_c),
                       step=(None if prev is None
                             else g._align(prev[0], prev[1], o_c, A_c)))
        prev = (o_c, A_c)
    r = rows[6]
    rec("t13_full_vs_2xstep", r["full"] / (2.0 * r["step"]), 1.0, "<",
        f"full={r['full']:.4e} step={r['step']:.4e}")
    for arm in ("only1", "only2", "none", "conj"):
        rec(f"t13_{arm}_over_full", r[arm] / r["full"], 10.0, ">")
    rec("t13_K4_over_K6_full", rows[4]["full"] / r["full"], 1.5, ">")

    # ---- test 14: the layer split --------------------------------------
    st1 = g._v2_stack()
    st1.add_sheared_grating(0.24e-6, eps_ridge=4.20, eps_groove=1.45,
                            duty=0.45, shear=0.25, centre=0.5)
    st1, _o1 = g._v2_solve(st1)
    o1, A1, kx1 = g._amps(st1)
    st2 = g._v2_stack()
    for _ in range(2):
        # BOTH halves take the SAME passed centre -- the gate's own
        # construction.  In the LAB the second one stands at +W1, because the
        # cascade continues the frame (D1); passing the lab position instead
        # reads 0.67 on the split identity, which is how D1 was found.
        st2.add_sheared_grating(0.12e-6, eps_ridge=4.20, eps_groove=1.45,
                                duty=0.45, shear=0.125,
                                centre=0.5 - 0.0625)
    st2, _o2 = g._v2_solve(st2)
    o2, A2, _k = g._amps(st2)
    rec("t14_split_identity", g._align(o1, A1, o2, A2), 1e-12, "<")
    W1 = 0.25 * g.V2_P
    Pw1 = g._P(kx1, W1)
    sc = g._v2_stack()
    sc.add_tapered_ridges(0.24e-6,
                          ridges=[(0.5 * g.V2_P, 0.45 * g.V2_P,
                                   0.45 * g.V2_P, 4.20)],
                          eps_groove=1.45, n_slices=8, shear=0.25)
    sc, _oc = g._v2_solve(sc)
    oc, Ac, _k = g._amps(sc)
    full1 = g._align(o1, A1, oc, Ac)
    full2 = g._align(o2, A2, oc, Ac)
    half1 = g._align(o1, A1 / Pw1 * g._P(kx1, 0.5 * W1), oc, Ac)
    half2 = g._align(o2, A2 / Pw1 * g._P(kx1, 0.5 * W1), oc, Ac)
    none1 = g._align(o1, A1 / Pw1, oc, Ac)
    rec("t14_full_agreement_rel", abs(full1 - full2) / full1, 1e-6, "<")
    rec("t14_half_agreement_rel", abs(half1 - half2) / half1, 1e-6, "<")
    rec("t14_half_over_full", half1 / full1, 10.0, ">")
    rec("t14_none_over_full", none1 / full1, 10.0, ">")

    # ---- test 15: the cross-engine arm ---------------------------------
    import math

    from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure
    duty, shear, centre = 0.50, 0.25, 0.625
    tx, walk = shear * g.V2_P / g.V2_D, shear * g.V2_P
    stx = g._v2_stack(degree=12, n_orders=7)
    stx.add_sheared_grating(g.V2_D, eps_ridge=g.V2_ER, eps_groove=g.V2_EG,
                            duty=duty, shear=shear, centre=centre)
    stx, _o = g._v2_solve(stx)
    J = np.asarray(stx.jones_transmission())
    nx = 4
    cellp = np.full((nx, nx), g.V2_EG, dtype=complex)
    cellp[int(round((0.5 - duty / 2) * nx)):
          int(round((0.5 + duty / 2) * nx)), :] = g.V2_ER
    pu = PMM2DStackPure(g.V2_P, g.V2_P, n_superstrate=1.0, n_substrate=1.6,
                        n_modes=5, n_orders=3)
    pu.add_layer(g.V2_D, eps_cell=cellp, slant=(tx, 0.0))
    pu.set_source(g.V2_WL, theta=g.V2_TH, phi=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pu.solve()
    Rj = np.asarray(pu.jones_transmission())
    p0 = complex(np.exp(1j * (2.0 * np.pi / g.V2_WL)
                        * math.sin(g.V2_TH) * walk))
    sp = g._resid(J, Rj)
    rec("t15_shipped", sp, 0.05, "<")
    rec("t15_none_over_shipped", g._resid(J / p0, Rj) / sp, 20.0, ">")
    rec("t15_conj_over_shipped",
        g._resid(J * np.conj(p0) / p0, Rj) / sp, 20.0, ">")

    # ---- the ARMED guard's cost ---------------------------------------
    _rc = g._rc
    rng = np.random.default_rng(11)
    cost = {}
    for n in (66, 242, 450, 722):
        A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        A = A + n * np.eye(n)
        for _ in range(3):
            _rc._guarded_inverse(A, "w")
            _rc._guarded_inverse(A, "w", rcond_refuse=1e-10)
        bp = min(_time(lambda: _rc._guarded_inverse(A, "x"))
                 for _ in range(7))
        ba = min(_time(lambda: _rc._guarded_inverse(A, "x",
                                                    rcond_refuse=1e-10))
                 for _ in range(7))
        cost[str(n)] = dict(plain_s=bp, armed_s=ba, ratio=ba / bp)
        print(f"  guard_cost n={n:4d} plain={bp:.5f} armed={ba:.5f} "
              f"ratio={ba / bp:.3f}")
    R["armed_guard_cost"] = cost

    _lib.save("t5b_durability_rest",
              dict(rows=R, total_secs=round(time.time() - t0, 1)))


def _time(fn):
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t


if __name__ == "__main__":
    main()
