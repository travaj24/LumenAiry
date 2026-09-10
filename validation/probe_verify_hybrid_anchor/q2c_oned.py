"""Q2 (c) -- the 1-D engine as an independent arm.

Two questions, both answered by measurement:

1. does the 1-D ``PMMStack`` even EXPOSE a transmitted field on a SHEARED
   grating (``add_sheared_grating``), and on which factorization?
2. if it does, is that field LAB-referenced -- i.e. does the same frame-anchor
   defect exist one dimension down?  The oracle is the 1-D engine's OWN
   z-staircase of the identical parallelogram (``add_tapered_grating`` with
   ``duty_top == duty_bottom`` and the same ``shear``), which is lab-referenced
   by construction.

Then the 2-D hybrid is run on the SAME grating as a y-uniform stripe and its
zeroth-order transmitted Jones compared against whichever 1-D arm is
lab-referenced.  (The zeroth-order Jones is invariant under a lateral
translation of the whole structure, so the two engines' differing ``centre``
conventions cannot contaminate it.)
"""
from __future__ import annotations

import math
import time
import warnings

import _lib as L
import numpy as np

P, WL, D = L.PX, L.WL, L.DTHICK
ER, EG, DUTY = 3.05, 1.30, 0.5
SHEAR = 0.25                      # walk = P/4, the QUARTER walk
TH = math.radians(25.0)
NSUB, NSUP = L.NSUB, L.NSUP
DEG, NORD1 = 12, 7


def _oned(kind, ns=1, fac="convection"):
    from lumenairy.elements.pmm.stack import PMMStack
    st = PMMStack(P, n_superstrate=NSUP, n_substrate=NSUB, degree=DEG,
                  n_orders=NORD1, factorization=fac)
    if kind == "shear":
        st.add_sheared_grating(D, eps_ridge=ER, eps_groove=EG, duty=DUTY,
                               shear=SHEAR, centre=0.5)
    elif kind == "vertical":
        st.add_sheared_grating(D, eps_ridge=ER, eps_groove=EG, duty=DUTY,
                               shear=0.0, centre=0.5)
    else:
        st.add_tapered_grating(D, eps_ridge=ER, eps_groove=EG,
                               duty_bottom=DUTY, duty_top=DUTY, shear=SHEAR,
                               n_slices=ns)
    st.set_source(WL, theta=TH)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = st.solve()
    return st, out


def _jt(st):
    try:
        return np.asarray(st.jones_transmission())
    except Exception:                                  # noqa: BLE001
        return None


def _twod_stripe(nx=48, n_orders=7):
    """The SAME binary grating as a y-uniform 2-D cell, with the ridge at the
    TOP face centred where the 1-D ridge's zeta = 0 face is (centre 0.5 at
    mid-depth -> top centre 0.5 - shear/2 = 0.375)."""
    cell = np.full((nx, 4), EG, dtype=float)
    c0 = 0.375
    lo = int(round((c0 - DUTY / 2) * nx))
    hi = int(round((c0 + DUTY / 2) * nx))
    cell[lo:hi, :] = ER
    st = L.hybrid(n_orders=n_orders)
    st.add_layer(D, eps_cell=cell, slant=(SHEAR * P / D, 0.0))
    st.set_source(WL, theta=TH, phi=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        st.solve()
    return st


def main():
    t0 = time.time()
    k0 = 2.0 * np.pi / WL
    W = SHEAR * P
    a0 = math.sin(TH) * float(np.real(NSUP))
    P0 = complex(np.exp(1j * k0 * a0 * W))
    res = dict(P0=[P0.real, P0.imag], arg_P0=float(np.angle(P0)),
               walk_um=W * 1e6, alpha0=a0)

    # 1 -- which 1-D path exposes a transmitted field at all
    avail = {}
    for fac in ("auto", "convection", "covariant"):
        try:
            st, _o = _oned("shear", fac=fac)
            avail[fac] = ("retained" if _jt(st) is not None
                          else "NO per-order amplitudes retained")
        except Exception as e:                          # noqa: BLE001
            avail[fac] = "RAISE " + type(e).__name__
    res["1d_transmission_surface"] = avail

    # 2 -- the convection sheared grating against the 1-D engine's own
    #      z-staircase of the identical parallelogram
    sh, osh = _oned("shear", fac="convection")
    Js = _jt(sh)
    rows = {}
    prev = None
    for ns in (6, 12, 24):
        stc, oc = _oned("stair", ns=ns, fac="convection")
        Jc = _jt(stc)
        rows["ns%d" % ns] = dict(
            dJt_as_returned=L.jones_residual(Js, Jc),
            dJt_x_P0=L.jones_residual(Js * P0, Jc),
            dJt_x_conj_P0=L.jones_residual(Js * np.conj(P0), Jc),
            dJr=L.jones_residual(np.asarray(osh[3]), np.asarray(oc[3])),
            dR=float(np.max(np.abs(np.asarray(osh[1])
                                   - np.asarray(oc[1])))),
            dT=float(np.max(np.abs(np.asarray(osh[2])
                                   - np.asarray(oc[2])))),
            staircase_own_step=(None if prev is None
                                else L.jones_residual(prev, Jc)))
        prev = Jc
    res["1d_shear_vs_staircase"] = rows

    # 3 -- the 2-D hybrid on the same grating, against the 1-D staircase
    st2 = _twod_stripe()
    J2 = np.asarray(st2.jones_transmission())
    stc, _oc = _oned("stair", ns=24, fac="convection")
    Jc = _jt(stc)
    a = st2.per_order_amplitudes("transmission")
    p0 = int(np.where((np.asarray(a["orders"])[:, 0] == 0)
                      & (np.asarray(a["orders"])[:, 1] == 0))[0][0])
    P0h = complex(np.exp(1j * k0 * a["kx"][p0] * W))
    st2b = _twod_stripe(n_orders=9)
    res["2d_vs_1d_staircase"] = dict(
        dJt_shipped=L.jones_residual(J2, Jc),
        dJt_none=L.jones_residual(J2 / P0h, Jc),
        dJt_conj=L.jones_residual(J2 * np.conj(P0h) / P0h, Jc),
        dJr=L.jones_residual(np.asarray(_oned("stair", ns=24,
                                              fac="convection")[1][3]),
                             np.asarray(st2._modal["orders"][:0] if False
                                        else np.zeros((2, 2)))) if False
        else None,
        hybrid_own_n_orders_step=L.jones_residual(
            J2, np.asarray(st2b.jones_transmission())),
        P0_hybrid=[P0h.real, P0h.imag])
    res["seconds"] = round(time.time() - t0, 1)
    print(res["1d_transmission_surface"])
    for k, v in res["1d_shear_vs_staircase"].items():
        print(k, v)
    print("2d vs 1d:", res["2d_vs_1d_staircase"])
    L.dump("q2c_oned", res)


if __name__ == "__main__":
    main()
