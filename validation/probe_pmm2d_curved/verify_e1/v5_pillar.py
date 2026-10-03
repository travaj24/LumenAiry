"""V5 -- E1-5 / E1-6 / E1-7 on the verifier's own pillars.

usage:
  python v5_pillar.py curved <c3|c5> <M> <mat> <th_deg> <ph_deg> [sx]
  python v5_pillar.py stair <k> <M> <mat> [sx]
  python v5_pillar.py rcwa <n> <mat>              exact-disk tensor RCWA
  python v5_pillar.py zstair <Nz> <n> <sx>        exact-disk RCWA z-staircase
  python v5_pillar.py recip <c3|c5> <M> <sx>      reciprocity + Onsager

Pillar (``_ve1common.PIL``): period 1.1, r 0.33, depth 0.45, lambda 1,
air / 1.5.  mat: ``oop`` (director 30 deg out of the plane, azimuth 30 deg;
n_o 1.52, n_e 1.78), ``nr`` its non-reciprocal twin (e13 + 0.22i, e31 its
conjugate), ``nrT`` the transpose, ``eps35`` isotropic 3.5 (the slanted
circle).  ``sx`` the PUBLIC x-slant (0.15 / -0.15).

The exact-disk RCWA uses the verifier's OWN Bessel form factor
(``2 J1(G r) / (G r)``, centred, every component, Laurent), patched into
``rcwa.twod._eps_convolution_2d`` with a call counter (asserted > 0) and
self-checked against the shipped ``_shape_form_factor``.  The z-staircase
is the shipped ``RCWAStack`` with ``shapes=`` disks translated to
``c + slant * z_mid`` per slice.
"""
import sys
import time
import warnings
from contextlib import contextmanager

import _ve1common as V
import numpy as np
from scipy.special import j1

f = V.PIL
P, R0, DEP = f["P"], f["R0"], f["DEP"]
CTR = P / 2
MATS = {"oop": V.PIL_OOP, "nr": V.PIL_NR, "nrT": V.PIL_NR.T.copy(),
        "eps35": 3.5 * V.EYE}


def rec(o, R, T, extra, st=None):
    v = V.vec36(o, R, T)
    d = {"vec": v, "clo": float(np.abs(np.asarray(R).sum(1)
                                       + np.asarray(T).sum(1) - 1).max())}
    d.update(extra)
    return d


def sx_of(args, i):
    return float(args[i]) if len(args) > i else 0.0


mode = sys.argv[1]
t0 = time.perf_counter()

if mode == "curved":
    kind, M, mat = sys.argv[2], int(sys.argv[3]), sys.argv[4]
    th, ph = np.deg2rad(float(sys.argv[5])), np.deg2rad(float(sys.argv[6]))
    sx = sx_of(sys.argv, 7)
    st, o, R, T, wall = V.pillar_run(kind, MATS[mat], M, th, ph,
                                     slant=(sx, 0.0) if sx else None)
    res = rec(o, R, T, {"kind": kind, "M": M, "mat": mat, "th": th,
                        "ph": ph, "sx": sx, "wall": wall})
    V.dump(f"v5_curved_{mat}_{kind}_t{sys.argv[5]}_p{sys.argv[6]}"
           f"_s{sx:+.2f}_M{M}.json", res)
    print(mode, kind, mat, M, sx, f"clo={res['clo']:.2e} t={wall:.0f}")

if mode == "stair":
    k, M, mat = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    sx = sx_of(sys.argv, 5)
    st, o, R, T, wall = V.stair_run(k, MATS[mat], M,
                                    slant=(sx, 0.0) if sx else None)
    res = rec(o, R, T, {"k": k, "M": M, "mat": mat, "sx": sx,
                        "wall": wall})
    V.dump(f"v5_stair_{mat}_k{k}_s{sx:+.2f}_M{M}.json", res)
    print(mode, k, M, mat, sx, f"clo={res['clo']:.2e} t={wall:.0f}")


def bessel_ff(gx, gy, r, c):
    g = np.hypot(gx, gy)
    x = g * r
    small = x < 1e-12
    xs = np.where(small, 1.0, x)
    b = np.where(small, 1.0, 2.0 * j1(xs) / xs)
    return (np.pi * r * r / (P * P)) * b * np.exp(-1j * (gx * c[0]
                                                         + gy * c[1]))


CALLS = [0]


@contextmanager
def exact_disk():
    from lumenairy.elements.rcwa import twod as RT
    orig = RT._eps_convolution_2d

    def conv(eps_cell, orders, nx, ny):
        CALLS[0] += 1
        a = np.asarray(eps_cell)
        S = a.shape[0]
        bg, d = complex(a[0, 0]), complex(a[S // 2, S // 2])
        Mx, My = int(nx), int(ny)
        ks = np.arange(-2 * Mx, 2 * Mx + 1)
        ls = np.arange(-2 * My, 2 * My + 1)
        KK, LL = np.meshgrid(ks, ls, indexing="ij")
        F = bessel_ff(KK * 2 * np.pi / P, LL * 2 * np.pi / P, R0,
                      (CTR, CTR))
        c = (d - bg) * F
        c[2 * Mx, 2 * My] += bg
        dm = orders[:, 0][:, None] - orders[:, 0][None, :]
        dn = orders[:, 1][:, None] - orders[:, 1][None, :]
        return c[dm + 2 * Mx, dn + 2 * My]
    RT._eps_convolution_2d = conv
    try:
        yield
    finally:
        RT._eps_convolution_2d = orig


if mode == "rcwa":
    from lumenairy.elements.rcwa import rcwa_jones_2d, twod as RT
    n, mat = int(sys.argv[2]), sys.argv[3]
    th = np.deg2rad(float(sys.argv[4])) if len(sys.argv) > 4 else 0.0
    ph = np.deg2rad(float(sys.argv[5])) if len(sys.argv) > 5 else 0.0
    # self-check of the own form factor against the shipped one
    gx = np.linspace(-40, 40, 41)
    GX, GY = np.meshgrid(gx, gx * 0.7, indexing="ij")
    ffchk = float(np.abs(bessel_ff(GX, GY, R0, (CTR, CTR))
                         - RT._shape_form_factor(
                             {"shape": "disk", "radius": R0,
                              "center": (CTR, CTR)}, GX, GY, P, P)).max())
    S = 4 * n + 8
    S += S % 2
    x = np.arange(S) * P / S
    X, Y = np.meshgrid(x, x, indexing="ij")
    cell = np.broadcast_to(V.EYE, (S, S, 3, 3)).copy()
    cell[(X - CTR) ** 2 + (Y - CTR) ** 2 < R0 ** 2] = MATS[mat]
    with exact_disk():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o, R, T, J = rcwa_jones_2d(P, P, cell, f["NSUB"], f["NSUP"],
                                       DEP, f["WL"], n_orders_x=n,
                                       n_orders_y=n, symmetry=False,
                                       formulation="laurent", theta=th,
                                       phi=ph)
    assert CALLS[0] > 0, "the exact-disk patch was never called"
    o = np.asarray(o)
    res = rec(o, R, T, {"n": n, "mat": mat, "calls": CALLS[0],
                        "ffcheck": ffchk, "th": th, "ph": ph,
                        "wall": time.perf_counter() - t0})
    V.dump(f"v5_rcwa_{mat}_t{th:.3f}_p{ph:.3f}_n{n}.json", res)
    print(mode, n, mat, f"calls={CALLS[0]} ff={ffchk:.1e} "
          f"clo={res['clo']:.2e} t={res['wall']:.0f}")

if mode == "zstair":
    from lumenairy.elements.rcwa import RCWAStack
    Nz, n, sx = int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
    st = RCWAStack(P, period_y=P, n_superstrate=f["NSUP"],
                   n_substrate=f["NSUB"], n_orders=n, n_orders_y=n)
    dz = DEP / Nz
    for j in range(Nz):
        z = (j + 0.5) * dz
        cx = (CTR + sx * z) % P
        st.add_layer(dz, shapes=[{"shape": "disk", "eps": 3.5,
                                  "radius": R0, "center": (cx, CTR)}],
                     eps_background=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r_ = st.set_source(f["WL"], theta=0.0, phi=0.0).solve()
    o, R, T = r_.efficiencies()
    o = np.asarray(o)
    if o.ndim == 1 or o.shape[-1] != 2:
        o = np.asarray(r_._require_modal()["orders2d"])
    res = rec(o, R, T, {"Nz": Nz, "n": n, "sx": sx,
                        "wall": time.perf_counter() - t0})
    V.dump(f"v5_zstair_s{sx:+.2f}_Nz{Nz}_n{n}.json", res)
    print(mode, Nz, n, sx, f"clo={res['clo']:.2e} t={res['wall']:.0f}")

if mode == "recip":
    kind, M, sx = sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
    th, ph = np.deg2rad(22.0), np.deg2rad(38.0)
    out = {}
    for order in ((-1, 0), (0, -1)):
        tr, pr = V.reverse_angles(th, ph, *order)

        def sv(mat, a, b, o_):
            st, o, R, T, wall = V.pillar_run(kind, MATS[mat], M, a, b,
                                             slant=(sx, 0.0) if sx else None)
            return np.linalg.svd(V.jones_block(st, o, o_), compute_uv=False)
        f_oop = sv("oop", th, ph, order)
        r_oop = sv("oop", tr, pr, order)
        w_oop = sv("oop", tr, pr, (0, 0))
        f_nr = sv("nr", th, ph, order)
        r_nr = sv("nr", tr, pr, order)
        r_nrT = sv("nrT", tr, pr, order)
        key = f"{order[0]}{order[1]}"
        out[key] = dict(
            reciprocal=float(np.abs(f_oop - r_oop).max()),
            wrong_pairing=float(np.abs(f_oop - w_oop).max()),
            nonrec_fwd_rev=float(np.abs(f_nr - r_nr).max()),
            onsager_nr_vs_nrT=float(np.abs(f_nr - r_nrT).max()),
            sv_oop=f_oop, sv_nr=f_nr)
        print(kind, M, sx, key, {k: v for k, v in out[key].items()
                                 if not k.startswith("sv")}, flush=True)
    out["wall"] = time.perf_counter() - t0
    V.dump(f"v5_recip_{kind}_s{sx:+.2f}_M{M}.json", out)
