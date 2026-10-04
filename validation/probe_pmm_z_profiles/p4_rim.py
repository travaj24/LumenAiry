"""P4 -- a ROUNDED RIM in one dimension: a ridge whose vertical sidewalls round
over into its flat top with a quarter-circle of radius ``RF`` (a fillet in
the x-z section).

Under route B the rim is a lateral map whose half-width follows the arc,
``a(w) = a0 - RF + sqrt(RF^2 - (RF - w)^2)`` for ``0 <= w <= RF`` (``w = 0``
the top face), so the tilt ``da/dw = (RF - w) / sqrt(...)`` diverges like
``w^(-1/2)`` at the top face while ``X_u`` stays positive (the flat top keeps
a half-width ``a0 - RF > 0``).  The straight part below the rim is the
identity map (one exact slab).  The arc-to-line change at ``w = RF`` is a slab
edge (the z analogue of the in-plane rule that a change of curve type must
be a grid vertex).

Arms (TE and TM, normal incidence, degree 16):
  UNI    route B, K equal slabs on [0, RF], frozen at the slab midpoints
  GRADED route B, edges ``w_k = RF (k/K)^2`` frozen at ``RF ((k+1/2)/K)^2``
         (the midpoint rule in ``sigma = sqrt(w / RF)``, which turns the
         ``w^(-1/2)`` tilt into a smooth integrand)
  STAIR  the physical staircase of the same profile on the shipped
         ``PMMStack`` (``layer_grids='per-layer'``): ns vertical slices on
         [0, RF], each at the profile's width at the slice midpoint, plus the
         straight part as one layer
What is read: route B's HEALTH (lossless closure and the mirror identity
T(-1) = T(+1), both round-off on a sound solve), the spectral pairing of one
frozen rim slab against its tilt and degree, the physical staircase's
successive changes, and the size of the rounding against a sharp rim.

Run:  PYTHONPATH=<worktree> OMP_NUM_THREADS=2 python p4_rim.py
Writes p4_rim.json.
"""
from __future__ import annotations

import os
import time
import warnings

import _zcommon as zc
import numpy as np
from p2_taper_ladder import EPS_G, EPS_R, EPS_SUB, EPS_SUP, WL, C, H, P

HERE = os.path.dirname(os.path.abspath(__file__))
A0, RF = 0.20, 0.10
DEG = 16


def a_of(w):
    w = np.minimum(w, RF)
    return A0 - RF + np.sqrt(np.maximum(RF * RF - (RF - w) ** 2, 0.0))


def da_of(w):
    if w >= RF:
        return 0.0
    return (RF - w) / np.sqrt(RF * RF - (RF - w) ** 2)


def xw(w):
    a = a_of(w)
    return np.array([0.0, C - a, C + a, P])


def dxw(w):
    d = da_of(w)
    return np.array([0.0, -d, d, 0.0])


def route_b(pol, K, graded, degree=DEG):
    if graded:
        e = [RF * (k / K) ** 2 for k in range(K + 1)]
        mids = [RF * ((k + 0.5) / K) ** 2 for k in range(K)]
    else:
        e = [RF * k / K for k in range(K + 1)]
        mids = [RF * (k + 0.5) / K for k in range(K)]
    edges = e + [H]
    mids = mids + [0.5 * (RF + H)]
    uw = [0.0, C - A0, C + A0, P]
    return zc.solve_taper(period=P, uw=uw, xw_of=xw, dxw_of=dxw, h=H,
                          eps_regions=[EPS_G, EPS_R, EPS_G], eps_sup=EPS_SUP,
                          eps_sub=EPS_SUB, wl=WL, pol=pol, degree=degree,
                          K=K + 1, slab_w=edges, slab_mid=mids)


def sharp(pol, degree):
    uw = [0.0, C - A0, C + A0, P]
    return zc.solve_taper(period=P, uw=uw, xw_of=lambda w: np.array(uw),
                          h=H, eps_regions=[EPS_G, EPS_R, EPS_G],
                          eps_sup=EPS_SUP, eps_sub=EPS_SUB, wl=WL, pol=pol,
                          degree=degree, K=1)


def stair(pol, ns, degree=DEG):
    from lumenairy.elements.pmm import PMMStack
    st = PMMStack(P, n_substrate=np.sqrt(EPS_SUB), n_superstrate=1.0,
                  degree=degree, far_field_orders=5, layer_grids="per-layer")
    for k in range(ns):                      # top-down
        a = a_of(RF * (k + 0.5) / ns)
        d = 2 * a / P
        edge = 0.5 * (1.0 - d)
        st.add_layer(RF / ns, segments=[(edge, EPS_G), (d, EPS_R),
                                        (edge, EPS_G)])
    d = 2 * A0 / P
    edge = 0.5 * (1.0 - d)
    st.add_layer(H - RF, segments=[(edge, EPS_G), (d, EPS_R), (edge, EPS_G)])
    st.set_source(WL, theta=0.0)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        orders, R, T, _J = st.solve()
    row = 1 if pol == "te" else 0
    o = [int(m) for m in np.asarray(orders)]
    sel = [o.index(m) for m in (-2, -1, 0, 1, 2)]
    return dict(R=[float(R[row][i]) for i in sel],
                T=[float(T[row][i]) for i in sel],
                time=time.perf_counter() - t0)


def effd(a, b):
    return float(max(np.max(np.abs(np.subtract(a["R"], b["R"]))),
                     np.max(np.abs(np.subtract(a["T"], b["T"])))))


def health(r):
    """closure (lossless) and the mirror identity T(-1) = T(+1) of a
    symmetric ridge at normal incidence -- both must be round-off."""
    o = r["orders"]
    return dict(closure=r["closure"],
                mirror=float(abs(r["T"][o.index(-1)] - r["T"][o.index(1)])))


def pairing(pol, w, degree):
    """Spectral pairing of ONE frozen rim slab, companion (the cascade's) and
    first-order linearisations, relative to max |beta|."""
    import scipy.linalg as sla
    k0 = 2 * np.pi / WL
    uw = [0.0, C - A0, C + A0, P]
    mesh = zc.Mesh1D(P, uw, [EPS_G, EPS_R, EPS_G], degree)
    tm = zc.TaperMap(uw, xw, H, dxw)
    ops = zc.region_ops(mesh, tm, w, pol, k0)
    n = mesh.n
    I = np.eye(n, dtype=complex)
    Z = np.zeros((n, n), dtype=complex)
    bq = sla.eig(np.block([[Z, I], [ops["A0"], ops["A1"]]]),
                 np.block([[I, Z], [Z, ops["A2"]]]), right=False)
    bq = bq[np.isfinite(bq)]
    M = ops["A2"]
    C2 = ops["C2f"]
    C1 = C2.conj().T
    Mi = np.linalg.inv(M)
    G = np.block([[Mi @ C2, Mi], [-ops["A0"] - C1 @ Mi @ C2, -C1 @ Mi]])
    b1 = np.linalg.eigvals(G) / 1j

    def res(b):
        return float(np.max([np.min(np.abs(b + x)) for x in b])
                     / max(np.max(np.abs(b)), 1.0))
    return dict(tilt=float(da_of(w)), companion=res(bq), first_order=res(b1))


def main():
    info = zc.assert_tree()
    out = dict(info=info, fixture=dict(P=P, WL=WL, H=H, A0=A0, RF=RF,
                                       EPS_R=EPS_R, DEG=DEG))
    for pol in ("te", "tm"):
        rb = {}
        for graded in (False, True):
            for K in (4, 8, 16, 32, 64):
                h = health(route_b(pol, K, graded))
                rb[f"{'graded' if graded else 'uniform'}_K{K}"] = h
                print(f"{pol} route B {'graded' if graded else 'uniform'} "
                      f"K {K:3d}: closure {h['closure']:.1e} mirror "
                      f"{h['mirror']:.1e}", flush=True)
        pr = {}
        for deg in (10, 16):
            for w in (RF * 0.5 / 64, RF * 0.5 / 16, RF * 1.5 / 16, RF * 0.25):
                k = f"deg{deg}_w{w:.5f}"
                pr[k] = pairing(pol, w, deg)
                print(f"{pol} pairing {k}: tilt {pr[k]['tilt']:.2f} "
                      f"companion {pr[k]['companion']:.1e} first-order "
                      f"{pr[k]['first_order']:.1e}", flush=True)
        st = {ns: stair(pol, ns) for ns in (4, 8, 16, 32, 64, 128)}
        srows = [dict(ns=ns, step=effd(st[ns], st[2 * ns]))
                 for ns in (4, 8, 16, 32, 64)]
        for r in srows:
            print(f"{pol} STAIR ns {r['ns']:3d} -> {2 * r['ns']}: change "
                  f"{r['step']:.2e}", flush=True)
        sh = sharp(pol, DEG)
        geo = effd(st[128], dict(R=sh["R"], T=sh["T"]))
        print(f"{pol} rounded (staircase ns 128) vs sharp rim: {geo:.2e}")
        out[pol] = dict(route_b_health=rb, slab_pairing=pr,
                        stair_steps=srows, rounded_vs_sharp=geo,
                        stair_times=[st[ns]["time"] for ns in sorted(st)])
    zc.dump(os.path.join(HERE, "p4_rim.json"), out)


if __name__ == "__main__":
    main()
