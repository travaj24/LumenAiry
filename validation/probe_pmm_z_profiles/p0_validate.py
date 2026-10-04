"""P0 -- validate the scratch route-B solver before any taper number is read.

V1  a VERTICAL ridge (no taper, identity map) against the shipped
    ``PMMStack`` (``add_layer(segments=...)``): every R / T, TE and TM.
V2  K-independence under a PURE SHEAR map (constant tilt, ``X_u = 1``): the
    virtual medium is z-invariant, so K = 1 and K = 16 must agree to round-off
    (the 1-D analogue of "a slanted layer expressed as a z-linear map").
V3  a UNIFORM film under the taper map against the exact Airy slab: NOT exact
    at finite K (the virtual medium varies with w), it must converge to Airy
    as K grows.

Run:  PYTHONPATH=<worktree> OMP_NUM_THREADS=2 python p0_validate.py
Writes p0_validate.json next to this file.
"""
from __future__ import annotations

import os
import warnings

import _zcommon as zc
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

P, WL, H = 0.8, 1.0, 0.6
EPS_R, EPS_G = 4.0, 1.0
EPS_SUP, EPS_SUB = 1.0, 1.45 ** 2
C = 0.5 * P


def walls_of(wd_top, wd_bot, h, shear=0.0):
    def xw(w):
        z = w / h
        wd = wd_top + (wd_bot - wd_top) * z
        s = shear * (w - 0.5 * h)       # a pure shear translates EVERY wall
        return np.array([0.0, C - 0.5 * wd, C + 0.5 * wd, P]) + s
    return xw


def library_vertical(duty, pol, degree):
    from lumenairy.elements.pmm import PMMStack
    st = PMMStack(P, n_substrate=np.sqrt(EPS_SUB), n_superstrate=1.0,
                  degree=degree, far_field_orders=5)
    edge = 0.5 * (1.0 - duty)
    st.add_layer(H, segments=[(edge, EPS_G), (duty, EPS_R), (edge, EPS_G)])
    st.set_source(WL, theta=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        orders, R, T, _J = st.solve()
    row = 1 if pol == "te" else 0
    o = list(np.asarray(orders).astype(int))
    return {int(m): (float(R[row][i]), float(T[row][i])) for i, m in
            enumerate(o)}


def airy(eps_f, pol, d):
    k0 = 2 * np.pi / WL
    n1, n2, n3 = 1.0, np.sqrt(eps_f + 0j), np.sqrt(EPS_SUB + 0j)
    if pol == "te":
        y1, y2, y3 = n1, n2, n3
    else:
        y1, y2, y3 = 1 / n1, 1 / n2, 1 / n3
    r12 = (y1 - y2) / (y1 + y2)
    r23 = (y2 - y3) / (y2 + y3)
    t12 = 2 * y1 / (y1 + y2)
    t23 = 2 * y2 / (y2 + y3)
    ph = np.exp(1j * n2 * k0 * d)
    r = (r12 + r23 * ph * ph) / (1 + r12 * r23 * ph * ph)
    t = t12 * t23 * ph / (1 + r12 * r23 * ph * ph)
    R = abs(r) ** 2
    # TE: E amplitude, T = |t|^2 n3/n1;  TM: H amplitude, T = |t|^2 (1/n3)/(1/n1)
    T = abs(t) ** 2 * np.real(y3) / np.real(y1)
    return float(R), float(T)


def main():
    info = zc.assert_tree()
    out = dict(info=info, fixture=dict(P=P, WL=WL, H=H, EPS_R=EPS_R,
                                        EPS_G=EPS_G, EPS_SUP=EPS_SUP,
                                        EPS_SUB=EPS_SUB))
    # ---- V1 vertical ridge vs the shipped PMMStack ------------------------
    duty = 0.5
    wd = duty * P
    uw = [0.0, C - 0.5 * wd, C + 0.5 * wd, P]
    v1 = {}
    for pol in ("te", "tm"):
        lib = library_vertical(duty, pol, 20)
        for deg in (12, 16, 20):
            r = zc.solve_taper(period=P, uw=uw, xw_of=walls_of(wd, wd, H),
                               h=H, eps_regions=[EPS_G, EPS_R, EPS_G],
                               eps_sup=EPS_SUP, eps_sub=EPS_SUB, wl=WL,
                               pol=pol, degree=deg, K=1)
            dmax = 0.0
            for i, m in enumerate(r["orders"]):
                if m in lib:
                    dmax = max(dmax, abs(r["R"][i] - lib[m][0]),
                               abs(r["T"][i] - lib[m][1]))
            v1[f"{pol}_deg{deg}"] = dict(max_dRT_vs_library=dmax,
                                         closure=r["closure"])
            print(f"V1 {pol} deg {deg}: max|dR,dT| vs PMMStack {dmax:.2e}, "
                  f"closure {r['closure']:.1e}")
    out["V1_vertical_vs_library"] = v1
    # ---- V2 pure shear: K independence ------------------------------------
    v2 = {}
    for pol in ("te", "tm"):
        xw = walls_of(wd, wd, H, shear=0.3)
        uw2 = list(xw(0.5 * H))
        rr = {}
        for K in (1, 4, 16):
            rr[K] = zc.solve_taper(period=P, uw=uw2, xw_of=xw, h=H,
                                   eps_regions=[EPS_G, EPS_R, EPS_G],
                                   eps_sup=EPS_SUP, eps_sub=EPS_SUB, wl=WL,
                                   pol=pol, degree=16, K=K)
        v2[pol] = dict(K1_vs_K4=zc.dist(rr[1], rr[4]),
                       K1_vs_K16=zc.dist(rr[1], rr[16]),
                       closure=rr[16]["closure"])
        print(f"V2 {pol} shear 0.3: K1 vs K16 {v2[pol]['K1_vs_K16']}")
    out["V2_shear_K_independence"] = v2
    # ---- V3 uniform film under the taper map vs Airy -----------------------
    v3 = {}
    wt, wb = 0.3 * P, 0.7 * P
    uw3 = [0.0, C - 0.25 * P, C + 0.25 * P, P]
    for pol in ("te", "tm"):
        Ra, Ta = airy(EPS_R, pol, H)
        lad = {}
        for K in (1, 2, 4, 8, 16, 32, 64):
            r = zc.solve_taper(period=P, uw=uw3, xw_of=walls_of(wt, wb, H),
                               h=H, eps_regions=[EPS_G, EPS_R, EPS_G],
                               eps_sup=EPS_SUP, eps_sub=EPS_SUB, wl=WL,
                               pol=pol, degree=14, K=K, eps_film=EPS_R)
            i0 = r["orders"].index(0)
            e = max(abs(r["R"][i0] - Ra), abs(r["T"][i0] - Ta),
                    max(abs(x) for j, x in enumerate(r["R"]) if j != i0),
                    max(abs(x) for j, x in enumerate(r["T"]) if j != i0))
            arms = {}
            for arm, kw in (("notilt", dict(tilt=False)),
                            ("m5", dict(m5=True))):
                try:
                    ra = zc.solve_taper(period=P, uw=uw3,
                                        xw_of=walls_of(wt, wb, H), h=H,
                                        eps_regions=[EPS_G, EPS_R, EPS_G],
                                        eps_sup=EPS_SUP, eps_sub=EPS_SUB,
                                        wl=WL, pol=pol, degree=14, K=K,
                                        eps_film=EPS_R, **kw)
                    arms[arm] = max(abs(ra["R"][i0] - Ra),
                                    abs(ra["T"][i0] - Ta))
                except Exception as exc:              # noqa: BLE001
                    arms[arm] = repr(exc)
            lad[K] = dict(err_vs_airy=e, closure=r["closure"], **arms)
            print(f"V3 {pol} film K {K}: err vs Airy {e:.2e}  "
                  f"notilt {arms['notilt']}  m5 {arms['m5']}")
        v3[pol] = lad
    out["V3_film_under_taper_map_vs_airy"] = v3
    zc.dump(os.path.join(HERE, "p0_validate.json"), out)


if __name__ == "__main__":
    main()
