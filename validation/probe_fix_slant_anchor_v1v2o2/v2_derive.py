"""V2 -- the 1-D frame anchor: DERIVATION, SIGN and the two-sided ladders.

The 1-D shear convention (``PMMStack.add_sheared_grating``): the frame is
``u = x - z tan(phi)`` anchored at the layer TOP, ``slant_angle = phi =
arctan(shear * period / thickness)``, and the LAB ridge centre is
``centre + shear * (zeta - 0.5)`` with ``zeta = 0`` at the top -- so
``slant_angle > 0`` walks the structure in ``+x`` with depth.  Specialising the
2-D rule ``A_lab(m) = exp(+i k0 alpha_m . W)`` to the x-only shear:

    A_lab(m) = exp(+i k0 alpha_m W) A_frame(m),
    W = sum over the layers that enter a sheared frame of tan(slant_j) * d_j

with ``alpha_m = kx_m / k0`` the (real) normalised transverse wavenumber and
``W`` in metres.  The sign is NOT taken on faith: three arms (as returned,
``x P``, ``x conj(P)``) are scored against a LAB-REFERENCED oracle -- the
engine's own z-staircase of the identical parallelogram -- at three rungs.

Also measured here, because they decide the RULE and not just the sign:

  * does a UNIFORM (unpatterned) slanted 1-D layer enter a frame?  In 2-D it
    does NOT (both ``_mode_key`` and ``_build_layer_modes`` short-circuit a
    constant tile to ``_homogeneous_modes`` BEFORE the slant is read).  The
    1-D metric generator has NO such short-circuit -- ``tan_conv * Dopx`` is
    added whenever ``|tan| > 1e-14`` -- so the answer must be MEASURED against
    the vertical film of the same eps;
  * the reflection side, which the derivation says carries NO anchor;
  * the ZEROTH-order exemption at normal incidence (``alpha_0 = 0``).
"""
from __future__ import annotations

import math
import sys
import time
import warnings

import _lib as L
import numpy as np

RUNGS = (6, 12, 24)


def align(a_orders, A, b_orders, B):
    """Relative Frobenius residual over the orders present in BOTH solves."""
    bmap = {int(o): j for j, o in enumerate(np.asarray(b_orders))}
    num = den = 0.0
    for i, o in enumerate(np.asarray(a_orders)):
        j = bmap.get(int(o))
        if j is None:
            continue
        num += float(np.sum(np.abs(A[..., i] - B[..., j]) ** 2))
        den += float(np.sum(np.abs(B[..., j]) ** 2))
    return math.sqrt(num) / math.sqrt(den) if den > 0 else math.sqrt(num)


def amps(st, port="transmission"):
    a = st.per_order_amplitudes(port)
    return (np.asarray(a["orders"]),
            np.stack([np.asarray(a["Ex"]), np.asarray(a["Ey"])], axis=0),
            np.asarray(np.real(a["kx"])))


def main():
    t0 = time.time()
    res = {}
    k0 = 2.0 * np.pi / L.WL1D
    W = L.SHEAR * L.P1D                    # = tan(slant_angle) * d, metres
    res["fixture"] = dict(period_um=L.P1D * 1e6, wl_um=L.WL1D * 1e6,
                          d_um=L.D1D * 1e6, eps_ridge=L.ER, eps_groove=L.EG,
                          duty=L.DUTY, shear=L.SHEAR, walk_um=W * 1e6,
                          walk_periods=L.SHEAR, theta_deg=25.0,
                          degree=L.DEG1D, n_orders=L.NORD1D,
                          n_sup=L.NSUP1D, n_sub=L.NSUB1D)

    # ---- 0. which 1-D factorizations retain a transmitted field ------------
    avail = {}
    for fac in ("auto", "convection", "covariant"):
        try:
            st, _o = L.oned("shear", fac=fac)
            avail[fac] = ("retained" if L.jt(st) is not None
                          else "NO per-order amplitudes retained")
        except Exception as e:                              # noqa: BLE001
            avail[fac] = "RAISE " + type(e).__name__
    res["transmission_surface"] = avail

    # ---- 1. the ladder against the engine's own z-staircase ----------------
    sh, osh = L.oned("shear")
    Js = L.jt(sh)
    o_s, A_s, kx_s = amps(sh)
    P_s = L.P_of(kx_s, W)
    a0 = float(np.real(L.NSUP1D) * math.sin(L.TH25))
    P0 = complex(np.exp(1j * k0 * a0 * W))
    res["P0"] = dict(re=P0.real, im=P0.imag, arg=float(np.angle(P0)),
                     alpha0=a0, walk_um=W * 1e6,
                     n_orders_retained=int(len(o_s)),
                     max_abs_dev=float(np.max(np.abs(np.abs(P_s) - 1.0))),
                     ptp_arg=float(np.ptp(np.angle(P_s))))
    rows, prev_J, prev_A, prev_o = {}, None, None, None
    for ns in RUNGS:
        stc, oc = L.oned("stair", ns=ns)
        Jc = L.jt(stc)
        o_c, A_c, _kx = amps(stc)
        rows["ns%d" % ns] = dict(
            J_as_returned=L.resid(Js, Jc),
            J_x_P=L.resid(Js * P0, Jc),
            J_x_conj_P=L.resid(Js * np.conj(P0), Jc),
            J_div_P=L.resid(Js / P0, Jc),
            amps_as_returned=align(o_s, A_s, o_c, A_c),
            amps_x_P=align(o_s, A_s * P_s, o_c, A_c),
            amps_x_conj_P=align(o_s, A_s * np.conj(P_s), o_c, A_c),
            amps_div_P=align(o_s, A_s / P_s, o_c, A_c),
            J_reflection=L.resid(np.asarray(osh[3]), np.asarray(oc[3])),
            dR=L.dmax(osh[1], oc[1]), dT=L.dmax(osh[2], oc[2]),
            stair_step_J=(None if prev_J is None else L.resid(prev_J, Jc)),
            stair_step_amps=(None if prev_A is None
                             else align(prev_o, prev_A, o_c, A_c)))
        prev_J, prev_A, prev_o = Jc, A_c, o_c
    res["ladder_vs_staircase"] = rows

    # ---- 2. does a UNIFORM slanted layer enter a frame? --------------------
    from lumenairy.elements.pmm.stack import PMMStack

    def film(tan_phi):
        st = PMMStack(L.P1D, n_superstrate=L.NSUP1D, n_substrate=L.NSUB1D,
                      degree=L.DEG1D, n_orders=L.NORD1D,
                      factorization="convection")
        st.add_layer(L.D1D, eps=2.60, slant_angle=math.atan(tan_phi))
        st.set_source(L.WL1D, theta=L.TH25)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = st.solve()
        return st, out

    stu, ou = film(W / L.D1D)
    stv, ov = film(0.0)
    Ju, Jv = L.jt(stu), L.jt(stv)
    o_u, A_u, kx_u = amps(stu)
    o_v, A_v, _ = amps(stv)
    P_u = L.P_of(kx_u, W)
    res["uniform_slanted_film"] = dict(
        J_as_returned=L.resid(Ju, Jv), J_x_P=L.resid(Ju * P0, Jv),
        J_x_conj_P=L.resid(Ju * np.conj(P0), Jv),
        amps_as_returned=align(o_u, A_u, o_v, A_v),
        amps_x_P=align(o_u, A_u * P_u, o_v, A_v),
        amps_x_conj_P=align(o_u, A_u * np.conj(P_u), o_v, A_v),
        J_reflection=L.resid(np.asarray(ou[3]), np.asarray(ov[3])),
        dR=L.dmax(ou[1], ov[1]), dT=L.dmax(ou[2], ov[2]))

    # ---- 3. the ZEROTH-order exemption at normal incidence -----------------
    shn, _oshn = L.oned("shear", theta=0.0)
    Jn = L.jt(shn)
    o_n, A_n, kx_n = amps(shn)
    P_n = L.P_of(kx_n, W)
    stcn, _ocn = L.oned("stair", ns=12, theta=0.0)
    Jcn = L.jt(stcn)
    o_cn, A_cn, _ = amps(stcn)
    i0 = int(np.where(np.asarray(o_n) == 0)[0][0])
    res["normal_incidence"] = dict(
        alpha0=float(kx_n[i0]),
        P0_minus_1=float(abs(complex(P_n[i0]) - 1.0)),
        n_distinct_P=int(len(np.unique(np.round(np.angle(P_n), 9)))),
        J_as_returned=L.resid(Jn, Jcn),
        J_x_P0=L.resid(Jn * complex(P_n[i0]), Jcn),
        amps_as_returned=align(o_n, A_n, o_cn, A_cn),
        amps_x_P=align(o_n, A_n * P_n, o_cn, A_cn),
        amps_x_conj_P=align(o_n, A_n * np.conj(P_n), o_cn, A_cn))

    res["seconds"] = round(time.time() - t0, 1)
    for k, v in res["ladder_vs_staircase"].items():
        print(k, {kk: (("%.5e" % vv) if isinstance(vv, float) else vv)
                  for kk, vv in v.items()})
    print("P0:", res["P0"])
    print("uniform film:", res["uniform_slanted_film"])
    print("normal:", res["normal_incidence"])
    print("surface:", res["transmission_surface"])
    L.dump("v2_derive", res, suffix=(sys.argv[1] if len(sys.argv) > 1 else ""))


if __name__ == "__main__":
    main()
