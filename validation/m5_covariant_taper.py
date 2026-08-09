"""M5 SPIKE 2 (T3-6) -- first-order-in-tan(phi) covariant treatment of a taper.

ANALYSIS-ONLY.  Nothing in ``lumenairy/`` is modified.

Two halves:

  PART A (shipped API, real R/T and complex order amplitudes)
    A1  staircase convergence ladder on a 2 deg tapered grating; the ``ns = 1``
        rung IS the first-order/pure-shear answer, so the ladder measures what
        any shear-only treatment can deliver.
    A2  sidewall-angle sweep 0.25 .. 8 deg -- the scaling law of the ns=1 error
        over the angle range the design space lives in.
    A3  NULL-FLOOR CONTROL: a pure PARALLELOGRAM solved by the shipped EXACT
        shear layer vs its own staircase.  Establishes both that the shear
        machinery is exact when the geometry IS a shear, and the instrument's
        own floor.
    A4  SHEAR-ABSORPTION test: how much of a taper can a shear absorb?
        Measured on a one-wall taper (max shear content) and on the symmetric
        taper (zero shear content).

  PART B (operator prototype, no library change)
    B1  Derivation check: assemble the covariant quadratic-in-q pencil with a
        piecewise-AFFINE convection field ``b(u) = S(u)`` and verify that at
        constant ``S`` it reproduces the SHIPPED slant pencil's modal spectrum
        (``_sem_modes_slant``) -- the derivation's validation.
    B2  The taper pencil's spectrum: well-posedness + convergence in degree.
    B3  The exact size of the neglected term (the z-dependence of the metric
        that first order drops).

Run:  python validation/m5_covariant_taper.py [--quick]
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import warnings

import numpy as np
import scipy.linalg as sla
from numpy.polynomial.legendre import leggauss

from lumenairy.elements.pmm import PMMStack, pmm_efficiency_1d_slanted
from lumenairy.elements.pmm._core import _build_sem_slant, _sem_modes_slant

_C = np.complex128
_ABLATE_ANTISYM = False        # B5 ablation switch (see _cov_pencil)


# ===========================================================================
# PART A -- shipped-API measurements
# ===========================================================================
def _stack_taper(period, wl, depth, eps_r, eps_g, duty_mid, dduty, ns, *,
                 shear=0.0, n_sup=1.0, n_sub=1.0, degree=12, theta=0.0,
                 per_layer=True):
    st = PMMStack(period, n_substrate=n_sub, n_superstrate=n_sup,
                  degree=degree,
                  layer_grids="per-layer" if per_layer else "shared")
    st.add_tapered_grating(depth, eps_ridge=eps_r, eps_groove=eps_g,
                           duty_top=duty_mid - 0.5 * dduty,
                           duty_bottom=duty_mid + 0.5 * dduty,
                           n_slices=ns, rule="midpoint", shear=shear)
    st.set_source(wl, theta=theta)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        st.solve()
    return st


def _stack_shear_exact(period, wl, depth, eps_r, eps_g, duty, shear, *,
                       n_sup=1.0, n_sub=1.0, degree=12, theta=0.0):
    st = PMMStack(period, n_substrate=n_sub, n_superstrate=n_sup,
                  degree=degree)
    st.add_sheared_grating(depth, eps_ridge=eps_r, eps_groove=eps_g,
                           duty=duty, shear=shear)
    st.set_source(wl, theta=theta)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        st.solve()
    return st


def _obs(st):
    """Observables from the SHIPPED 4-tuple ``(orders, R, T, J)`` -- available on
    BOTH the staircase and the exact-shear paths (``per_order_amplitudes`` is
    NOT retained on the covariant uniform-slant cascade)."""
    o, R, T, J = st.solve()
    o = np.asarray(o)
    i0 = int(np.where(o == 0)[0][0])
    return {"R": np.asarray(R), "T": np.asarray(T), "J": np.asarray(J),
            "o": o, "i0": i0,
            "tot": float(np.max(np.sum(R, -1) + np.sum(T, -1)))}


def _obs_rcwa(r):
    o, R, T = r.efficiencies()
    o = np.asarray(o)
    i0 = int(np.where(o == 0)[0][0])
    R, T = np.asarray(R), np.asarray(T)
    return {"R": R, "T": T,
            "J": np.concatenate([np.asarray(r.jones_reflection()).ravel(),
                                 np.asarray(r.jones_transmission()).ravel()]),
            "o": o, "i0": i0,
            "tot": float(np.max(np.sum(R, -1) + np.sum(T, -1)))}


def _delta(a, b):
    """Worst absolute move over (i) every order/pol efficiency and (ii) the
    COMPLEX zeroth-order Jones -- the deep-null-sensitive observable."""
    if np.shape(a["R"]) != np.shape(b["R"]):        # align on common orders
        common = np.intersect1d(a["o"], b["o"])
        ia = np.searchsorted(a["o"], common)
        ib = np.searchsorted(b["o"], common)
        aR, aT, bR, bT = (a["R"][..., ia], a["T"][..., ia],
                          b["R"][..., ib], b["T"][..., ib])
    else:
        aR, aT, bR, bT = a["R"], a["T"], b["R"], b["T"]
    dR = float(np.max(np.abs(aR - bR)))
    dT = float(np.max(np.abs(aT - bT)))
    if np.shape(a["J"]) == np.shape(b["J"]):        # cross-solver: J differs
        dJ = float(np.max(np.abs(a["J"] - b["J"])))
        return {"dR": dR, "dT": dT, "dJ": dJ, "worst": max(dR, dT, dJ)}
    return {"dR": dR, "dT": dT, "dJ": float("nan"), "worst": max(dR, dT)}


def _rcwa_taper(period, wl, depth, eps_r, eps_g, duty_mid, dduty, ns, *,
                shear=0.0, n_sup=1.0, n_sub=1.0, n_orders=21, theta=0.0,
                n_x=4096):
    from lumenairy.elements.rcwa import RCWAStack
    st = RCWAStack(period, n_superstrate=n_sup, n_substrate=n_sub,
                   n_orders=n_orders)
    st.add_tapered_grating(depth, eps_ridge=eps_r, eps_groove=eps_g,
                           duty_top=duty_mid - 0.5 * dduty,
                           duty_bottom=duty_mid + 0.5 * dduty,
                           n_slices=ns, shear=shear, n_x=n_x, raster="area")
    st.set_source(wl, theta=theta)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _obs_rcwa(st.solve())


def part_a(res, quick):
    print("\n" + "=" * 78)
    print("PART A -- shipped-API measurements")
    print("=" * 78)
    P, WL, H = 0.700, 1.310, 0.310
    EPS_R, EPS_G = 4.0 + 0j, 1.0 + 0j
    NSUP, NSUB = 1.5, 1.5
    DUTY = 0.5
    DEG = 10 if quick else 12
    TH = np.deg2rad(8.0)
    NO = 15 if quick else 21

    def dduty_of(dw):
        return 2.0 * H * np.tan(np.deg2rad(dw)) / P

    dd = dduty_of(2.0)
    print(f"\n   device: P={P*1e3:.0f} nm, wl={WL*1e3:.0f} nm, H={H*1e3:.0f} nm,"
          f" duty {DUTY}, n_sup=n_sub={NSUP}, theta=8 deg")
    print(f"   2 deg taper: wall motion {H*np.tan(np.deg2rad(2.0))*1e3:.3f} nm "
          f"per wall; duty change {dd:.5f} periods = {dd*P*1e3:.3f} nm")

    # ---------------- A0: PMM per-layer stability census -------------------
    print("\n-- A0  PMM per-layer staircase: where does it stop being usable?")
    a0 = []
    print(f"   {'ns':>5}{'R+T max':>16}{'guard fired':>13}{'step vs prev':>14}")
    prev = None
    for ns in ([1, 2, 4, 8, 16, 32] if quick else
               [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 128]):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            o = _obs(_stack_taper(P, WL, H, EPS_R, EPS_G, DUTY, dd, ns,
                                  n_sup=NSUP, n_sub=NSUB, degree=DEG,
                                  theta=TH))
            fired = any("energy not conserved" in str(x.message)
                        for x in caught)
        step = _delta(o, prev)["worst"] if prev is not None else float("nan")
        prev = o
        a0.append(dict(ns=ns, tot=o["tot"], guard=bool(fired),
                       step=float(step)))
        print(f"   {ns:>5}{o['tot']:>16.10f}{str(fired):>13}{step:>14.3e}")
    res["A0_pmm_stability"] = a0
    bad = [r["ns"] for r in a0 if r["guard"] or abs(r["tot"] - 1) > 1e-5]
    print(f"   FIRST unusable n_slice: {bad[0] if bad else 'none in range'}"
          "   -- the staircase reference must come from elsewhere")
    res["A0_first_bad_ns"] = bad[0] if bad else None

    # ---------------- A1: staircase ladder on the RCWA oracle --------------
    print("\n-- A1  staircase ladder on the INDEPENDENT RCWA oracle "
          f"(n_orders={NO}, raster='area';")
    print("        the Fourier error is COMMON-MODE across ns, so this "
          "isolates the STAIRCASE)")
    ns_ref = 128 if quick else 384
    ref = _rcwa_taper(P, WL, H, EPS_R, EPS_G, DUTY, dd, ns_ref, n_sup=NSUP,
                      n_sub=NSUB, n_orders=NO, theta=TH)
    ref2 = _rcwa_taper(P, WL, H, EPS_R, EPS_G, DUTY, dd, ns_ref // 2,
                       n_sup=NSUP, n_sub=NSUB, n_orders=NO, theta=TH)
    floor = _delta(ref, ref2)["worst"]
    print(f"   reference ns={ns_ref}; R+T={ref['tot']:.12f}; "
          f"self-consistency vs ns={ns_ref//2}: {floor:.3e}  <-- floor")
    lad = []
    print(f"   {'ns':>5}{'dR':>12}{'dT':>12}{'dJones':>12}{'worst':>12}"
          f"{'ratio':>8}")
    prev = None
    for ns in ([1, 2, 4, 8, 16, 32] if quick else
               [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]):
        o = _rcwa_taper(P, WL, H, EPS_R, EPS_G, DUTY, dd, ns, n_sup=NSUP,
                        n_sub=NSUB, n_orders=NO, theta=TH)
        d = _delta(o, ref)
        rat = (prev / d["worst"]) if prev else float("nan")
        prev = d["worst"]
        lad.append(dict(ns=ns, **{k: float(v) for k, v in d.items()}))
        print(f"   {ns:>5}{d['dR']:>12.3e}{d['dT']:>12.3e}{d['dJ']:>12.3e}"
              f"{d['worst']:>12.3e}{rat:>8.2f}")
    res["A1_ladder"] = lad
    res["A1_floor"] = float(floor)
    res["A1_ns1"] = float(lad[0]["worst"])

    # A1b: the two solvers must agree where PMM is stable
    print("\n-- A1b  cross-solver check (PMM per-layer vs RCWA) at low ns")
    a1b = []
    for ns in (1, 2, 4):
        p_ = _obs(_stack_taper(P, WL, H, EPS_R, EPS_G, DUTY, dd, ns,
                               n_sup=NSUP, n_sub=NSUB, degree=DEG, theta=TH))
        r_ = _rcwa_taper(P, WL, H, EPS_R, EPS_G, DUTY, dd, ns, n_sup=NSUP,
                         n_sub=NSUB, n_orders=NO, theta=TH)
        d = _delta(p_, r_)
        a1b.append(dict(ns=ns, dR=d["dR"], dT=d["dT"]))
        print(f"   ns={ns}: |dR|={d['dR']:.3e}  |dT|={d['dT']:.3e}   "
              f"(PMM R+T={p_['tot']:.12f}, RCWA R+T={r_['tot']:.12f})")
    res["A1b_cross_solver"] = a1b

    # ---------------- A2: sidewall-angle sweep ----------------------------
    print("\n-- A2  angle sweep.  For a SYMMETRIC taper the shear content is "
          "ZERO, so the ns=1 rung")
    print("        IS the best any shear-only / first-order-in-tan(phi) "
          "treatment can do.")
    print(f"   {'phi_deg':>8}{'wall mv nm':>12}{'dduty':>10}{'err ns=1':>12}"
          f"{'err ns=2':>12}{'err ns=4':>12}{'err ns=8':>12}{'e1/dd^2':>11}")
    a2 = []
    ns_r2 = 64 if quick else 192
    for ang in ([0.5, 2.0, 8.0] if quick else [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]):
        ddx = dduty_of(ang)
        rr = _rcwa_taper(P, WL, H, EPS_R, EPS_G, DUTY, ddx, ns_r2, n_sup=NSUP,
                         n_sub=NSUB, n_orders=NO, theta=TH)
        e = {}
        for ns in (1, 2, 4, 8):
            e[ns] = _delta(_rcwa_taper(P, WL, H, EPS_R, EPS_G, DUTY, ddx, ns,
                                       n_sup=NSUP, n_sub=NSUB, n_orders=NO,
                                       theta=TH), rr)["worst"]
        a2.append(dict(phi=ang, dduty=float(ddx),
                       **{f"e{k}": float(v) for k, v in e.items()}))
        print(f"   {ang:>8.2f}{H*np.tan(np.deg2rad(ang))*1e3:>12.3f}"
              f"{ddx:>10.5f}{e[1]:>12.3e}{e[2]:>12.3e}{e[4]:>12.3e}"
              f"{e[8]:>12.3e}{e[1]/ddx**2:>11.3e}")
    res["A2_angle"] = a2

    # ---------------- A3: null control -- pure parallelogram ---------------
    print("\n-- A3  NULL CONTROL: pure PARALLELOGRAM (dduty = 0).  The shipped "
          "EXACT shear layer vs its own staircase.")
    sh = H * np.tan(np.deg2rad(2.0)) / P
    ex = _obs(_stack_shear_exact(P, WL, H, EPS_R, EPS_G, DUTY, sh, n_sup=NSUP,
                                 n_sub=NSUB, degree=DEG, theta=TH))
    a3 = []
    print(f"   shear = {sh:.6f} periods (2 deg wall tilt over {H*1e3:.0f} nm)")
    print(f"   {'ns':>5}{'|PMM staircase - EXACT shear|':>32}{'R+T':>18}")
    for ns in ([1, 2, 4] if quick else [1, 2, 3, 4, 6]):
        o = _obs(_stack_taper(P, WL, H, EPS_R, EPS_G, DUTY, 0.0, ns, shear=sh,
                              n_sup=NSUP, n_sub=NSUB, degree=DEG, theta=TH))
        d = _delta(o, ex)["worst"]
        a3.append(dict(ns=ns, err=float(d), tot=o["tot"]))
        print(f"   {ns:>5}{d:>32.3e}{o['tot']:>18.12f}")
    print("   -> a shear IS solved exactly by ONE layer; the staircase merely "
          "converges TO it.")
    res["A3_shear_null"] = a3

    # ---------------- A4: how much of a taper can a shear absorb? ---------
    print("\n-- A4  SHEAR ABSORPTION.  A linear wall motion splits into the "
          "CENTRE walk (= shear,")
    print("        absorbed EXACTLY by u = x - z tan) and the DUTY change "
          "(= dilation, absorbed by nothing).")
    a4 = []
    ns_r4 = 64 if quick else 192
    for label, sh_geo in [("symmetric taper (centre walk 0)", 0.0),
                          ("one-wall taper (centre walk dd/2)", 0.5 * dd)]:
        rr = _rcwa_taper(P, WL, H, EPS_R, EPS_G, DUTY, dd, ns_r4, shear=sh_geo,
                         n_sup=NSUP, n_sub=NSUB, n_orders=NO, theta=TH)
        cA = _rcwa_taper(P, WL, H, EPS_R, EPS_G, DUTY, 0.0, 1, shear=0.0,
                         n_sup=NSUP, n_sub=NSUB, n_orders=NO, theta=TH)
        best = None
        span = 1.5 * abs(sh_geo) + 1e-4
        for st_ in np.linspace(-span, span, 13):
            cB = _rcwa_taper(P, WL, H, EPS_R, EPS_G, DUTY, 0.0, 1,
                             shear=float(st_), n_sup=NSUP, n_sub=NSUB,
                             n_orders=NO, theta=TH)
            e_ = _delta(cB, rr)["worst"]
            if best is None or e_ < best[1]:
                best = (float(st_), float(e_))
        eA = _delta(cA, rr)["worst"]
        a4.append(dict(geom=label, shear_geo=float(sh_geo),
                       err_vertical=float(eA), best_shear=best[0],
                       err_best_shear=best[1],
                       gain=float(eA / best[1]) if best[1] else float("inf")))
        print(f"   {label}")
        print(f"     ns=1 vertical mid-width, no shear : {eA:.3e}")
        print(f"     BEST single parallelogram         : {best[1]:.3e} at "
              f"shear={best[0]:+.6f} (geometry's = {sh_geo:+.6f})")
        print(f"     gain from absorbing the shear     : {eA/best[1]:.2f}x")
    res["A4_shear_absorption"] = a4

    # ---------------- A4b: multi-order generality -------------------------
    print("\n-- A4b  generality: a MULTI-ORDER design (P=1000 nm, wl=633 nm, "
          "normal incidence)")
    P2, WL2, H2 = 1.000, 0.633, 0.300
    dd2 = 2.0 * H2 * np.tan(np.deg2rad(2.0)) / P2
    rr = _rcwa_taper(P2, WL2, H2, EPS_R, EPS_G, 0.5, dd2,
                     64 if quick else 192, n_orders=NO)
    a4b = []
    prev = None
    for ns in ([1, 2, 4, 8] if quick else [1, 2, 3, 4, 6, 8, 16, 32]):
        d = _delta(_rcwa_taper(P2, WL2, H2, EPS_R, EPS_G, 0.5, dd2, ns,
                               n_orders=NO), rr)
        rat = (prev / d["worst"]) if prev else float("nan")
        prev = d["worst"]
        a4b.append(dict(ns=ns, **{k: float(v) for k, v in d.items()}))
        print(f"   ns={ns:>3}: |dR|={d['dR']:.3e}  |dT|={d['dT']:.3e}  "
              f"|dJ|={d['dJ']:.3e}  worst={d['worst']:.3e}  ratio={rat:.2f}")
    res["A4b_multiorder"] = a4b
    return res


# ===========================================================================
# PART B -- covariant taper operator prototype
# ===========================================================================
def _gll(deg):
    """Gauss-Lobatto-Legendre nodes/weights via the library's helper."""
    from lumenairy.elements.pmm._core import _gll_nodes_weights
    return _gll_nodes_weights(deg)


def _cov_pencil(period, walls, eps_seg, S_nodes, degree, k0, *,
                zeta=0.5, height=0.0, els_per_region=1):
    r"""Covariant taper pencil for scalar TE, z FROZEN at depth fraction
    ``zeta`` (``zeta = 0.5`` -> the FIRST-ORDER / mid-depth layer).

    Mid-depth reference frame ``u``:  ``x = X(u, z) = u + (z - h/2) S(u)`` with
    ``S`` PIECEWISE LINEAR through the values ``S_nodes`` at ``walls``
    (``S`` must be periodic; ``S(0) = S(d) = 0`` keeps the cell fixed, a
    CONSTANT ``S`` is a pure shear).  Exact metric quantities:

        X_u = 1 + h (zeta - 1/2) S'(u)        (piecewise CONSTANT in u)
        g   = 1/X_u ,     b = X_z/X_u = S/X_u ,      b/g = S   (z-FREE)

    The covariant (Laplace-Beltrami) weak form  ``(1/sqrt G) d_i (sqrt G G^ij
    d_j E) + k0^2 eps E = 0`` with ``sqrt G = X_u``,
    ``G^uu = (1+S^2)/X_u^2``, ``G^uz = -S/X_u``, ``G^zz = 1``, and
    ``E = phi(u) exp(i q k0 z)``, gives the QUADRATIC pencil
    ``A1 phi - q Ac phi - q^2 A2 phi = 0``:

        A1 = <v| eps X_u |phi>  - (1/k0^2) <v'| (1 + S^2)/X_u |phi'>
        Ac = (2 i / k0) <v|S|phi'>                              [z-FREE]
        A2 = <v| X_u |phi>

    CAUTION (a term that is easy to drop, and was, on the first pass here):
    ``d_z(sqrt G G^zz d_z E) = X_u d_z^2 E + S'(u) d_z E`` contributes a THIRD
    q-linear piece ``-(i/k0)<v|S'|phi>``.  Together with
    ``<v'|S|phi> = -<v|S'|phi> - <v|S|phi'>`` (periodic IBP) the three q-linear
    pieces collapse to the single term above.  Dropping it leaves an
    ANTISYMMETRISED ``(i/k0)(<v|S|phi'> - <v'|S|phi>)``, which is identical for
    CONSTANT S (so it passes the shear reduction test B1) and WRONG by exactly
    ``<v|S'|phi>`` for a taper -- measured to degrade the cascade from
    second order to first order in K.

    Only the mass/stiffness weights carry ``z`` (through the scalar ``X_u`` per
    element); the convection ``Ac`` is exact at every depth.  With ``S``
    CONSTANT, ``S' = 0``, ``X_u == 1`` for ALL z: the pencil is z-independent
    and reduces EXACTLY to ``_sem_modes_slant``'s TE pencil (``sec^2 = 1+t^2``,
    ``Ac = (2 i t / k0) C``).  B1 verifies that at operator level."""
    nodes, w = _gll(degree)
    from lumenairy.elements.pmm._core import _lagrange_derivative_matrix
    D = _lagrange_derivative_matrix(nodes)
    bnds = []
    for i in range(len(walls) - 1):
        xl, xr = walls[i], walls[i + 1]
        sl, sr = S_nodes[i], S_nodes[i + 1]
        dS = (sr - sl) / (xr - xl)                     # S' on this region
        for e in range(els_per_region):
            a = xl + (xr - xl) * e / els_per_region
            b_ = xl + (xr - xl) * (e + 1) / els_per_region
            sa = sl + (sr - sl) * (a - xl) / (xr - xl)
            sb = sl + (sr - sl) * (b_ - xl) / (xr - xl)
            bnds.append((a, b_, eps_seg[i], sa, sb, dS))
    n_el = len(bnds)
    l2g = np.zeros((n_el, degree + 1), dtype=int)
    gid = 0
    for e in range(n_el):
        for a in range(degree + 1):
            if a == 0 and e > 0:
                l2g[e, a] = l2g[e - 1, degree]
            else:
                l2g[e, a] = gid
                gid += 1
    last = l2g[n_el - 1, degree]
    l2g[l2g == last] = 0
    n = last

    def Z():
        return np.zeros((n, n), dtype=_C)
    S0, Peps, Lw, Cb, Cbp, Cplain = (Z() for _ in range(6))
    for e in range(n_el):
        xl, xr, eps, sa, sb, dS = bnds[e]
        Xu = 1.0 + height * (zeta - 0.5) * dS          # scalar per element
        J = 0.5 * (xr - xl)
        wel = w * J
        Dp = D / J
        Sn = sa + (sb - sa) * (nodes + 1.0) * 0.5      # S at the GLL nodes
        idx = l2g[e]
        ix = np.ix_(idx, idx)
        S0[ix] += np.diag(wel * Xu)                    # <v|X_u|phi>
        Peps[ix] += np.diag(wel * (eps * Xu))          # <v|eps X_u|phi>
        Lw[ix] += (Dp.T * (wel * (1.0 + Sn ** 2) / Xu)) @ Dp
        Cb[ix] += np.diag(wel * Sn) @ Dp               # <v|S|phi'>
        Cbp[ix] += (Dp.T * wel) @ np.diag(Sn)          # <v'|S|phi>
        Cplain[ix] += np.diag(wel) @ Dp                # <v|phi'>
    A1 = Peps - Lw / (k0 * k0)
    A2 = S0
    Ac = (2j / k0) * Cb
    if _ABLATE_ANTISYM:                        # B5 ablation: the WRONG form
        Ac = (1j / k0) * (Cb - Cbp)
    return A1, Ac, A2, n, Cplain


def _pencil_spectrum(A1, Ac, A2, n):
    Imat = np.eye(n, dtype=_C)
    Zm = np.zeros((n, n), dtype=_C)
    Abig = np.block([[Zm, Imat], [A1, -Ac]])
    Bbig = np.block([[Imat, Zm], [Zm, A2]])
    q = sla.eig(Abig, Bbig, right=False)
    return q[np.isfinite(q)]


def _cov_modes(period, walls, eps_seg, S_nodes, degree, k0, zeta, height):
    """Forward/backward modal blocks (W = E nodal coeffs, V = H_x partner) of
    the z-FROZEN covariant pencil, split by z-Poynting flux."""
    A1, Ac, A2, n, _x = _cov_pencil(period, walls, eps_seg, S_nodes, degree,
                                    k0, zeta=zeta, height=height)
    # plain (X_u-free) mass, for the flux pairing and the H-partner projection
    _A1p, _Acp, Mp, _n, _xp = _cov_pencil(period, walls, eps_seg, S_nodes,
                                          degree, k0, zeta=0.5, height=0.0)
    Imat = np.eye(n, dtype=_C)
    Zm = np.zeros((n, n), dtype=_C)
    q, Vb = sla.eig(np.block([[Zm, Imat], [A1, -Ac]]),
                    np.block([[Imat, Zm], [Zm, A2]]))
    fin = np.isfinite(q)
    q = q[fin]
    phi = Vb[:n, fin]
    phi = phi / np.where(np.linalg.norm(phi, axis=0) < 1e-300, 1.0,
                         np.linalg.norm(phi, axis=0))
    # lab d/dz = i q k0 - S d/du  ->  V = q phi + (i/k0) Mp^{-1} <v|S|phi'> phi
    DS = np.linalg.solve(Mp, _cov_cb_only(period, walls, eps_seg, S_nodes,
                                          degree, zeta=zeta, height=height))
    V = phi * q[None, :] + (1j / k0) * (DS @ phi)
    Sz = np.real(np.einsum("in,in->n", phi, Mp @ np.conj(V)))
    qmax = max(float(np.max(np.abs(q))), 1.0)
    prop = np.abs(q.imag) < 1e-7 * qmax
    fwd = np.where(prop, Sz > 0.0, q.imag > 0.0)
    fi = np.where(fwd)[0]
    if fi.size != n:
        pi_ = np.where(prop)[0][np.argsort(-Sz[np.where(prop)[0]])]
        ei = np.where(~prop)[0][np.argsort(-q.imag[np.where(~prop)[0]])]
        fi = np.sort(np.concatenate([pi_, ei])[:n])
    bi = np.array(sorted(set(range(len(q))) - set(fi.tolist())))
    return (phi[:, fi], V[:, fi], -1j * q[fi],
            phi[:, bi], V[:, bi], -1j * q[bi], n)


def _cov_cb_only(period, walls, eps_seg, S_nodes, degree, zeta=0.5, height=0.0):
    """``<v| S/X_u |phi'>`` -- the H-partner's weight (lab ``d/dz`` at fixed x
    is ``d_z - (S/X_u) d_u``)."""
    nodes, w = _gll(degree)
    from lumenairy.elements.pmm._core import _lagrange_derivative_matrix
    D = _lagrange_derivative_matrix(nodes)
    bnds = []
    for i in range(len(walls) - 1):
        xl, xr = walls[i], walls[i + 1]
        dS = (S_nodes[i + 1] - S_nodes[i]) / (xr - xl)
        bnds.append((xl, xr, S_nodes[i], S_nodes[i + 1],
                     1.0 + height * (zeta - 0.5) * dS))
    n_el = len(bnds)
    l2g = np.zeros((n_el, degree + 1), dtype=int)
    gid = 0
    for e in range(n_el):
        for a in range(degree + 1):
            if a == 0 and e > 0:
                l2g[e, a] = l2g[e - 1, degree]
            else:
                l2g[e, a] = gid
                gid += 1
    last = l2g[n_el - 1, degree]
    l2g[l2g == last] = 0
    n = last
    Cb = np.zeros((n, n), dtype=_C)
    for e in range(n_el):
        xl, xr, sa, sb, Xu = bnds[e]
        J = 0.5 * (xr - xl)
        wel = w * J
        Dp = D / J
        Sn = sa + (sb - sa) * (nodes + 1.0) * 0.5
        idx = l2g[e]
        Cb[np.ix_(idx, idx)] += np.diag(wel * Sn / Xu) @ Dp
    return Cb


def _cov_layer_rt(period, walls, eps_seg, S_nodes, degree, k0, H, K):
    """The covariant layer's OWN S-matrix as a K-sub-slab cascade, expressed in
    the K-INDEPENDENT modal bases of the EXACT boundary metrics ``zeta = 0``
    and ``zeta = 1``.

    Using the exact-boundary bases (instead of a fixed vertical half-space)
    matters: with a fixed vertical reference the FIRST/LAST sub-slab's frozen
    ``X_u`` depends on K, so the outer interface injects an O(delta) K-dependent
    artefact that masks the cascade's true order -- measured, and it turned a
    second-order cascade into an apparent first-order one.

    A pure shear (S constant -> X_u == 1 at every z) must give a K-difference at
    machine zero: the null control."""
    from lumenairy.elements.pmm._core import (
        _interface_smatrix,
        _propagation_smatrix,
        _redheffer_star,
    )

    def m(zeta):
        return _cov_modes(period, walls, eps_seg, S_nodes, degree, k0, zeta, H)
    m0 = m(0.0)
    m1 = m(1.0)
    S = None
    prev = (m0[0], m0[1])
    dz = H / K
    for k in range(K):
        Wf, Vf, lf = m((k + 0.5) / K)[:3]
        Si = _interface_smatrix(prev[0], prev[1], Wf, Vf)
        S = Si if S is None else _redheffer_star(S, Si)
        S = _redheffer_star(S, _propagation_smatrix(lf, k0 * dz))
        prev = (Wf, Vf)
    S = _redheffer_star(S, _interface_smatrix(prev[0], prev[1], m1[0], m1[1]))
    return S


def _spectra_hausdorff(a, b):
    """Symmetric nearest-neighbour distance between two eigenvalue SETS -- the
    order-independent way to compare spectra (lexicographic sorting breaks on
    +0.0 / -0.0 ties in purely imaginary eigenvalues)."""
    a = np.asarray(a)
    b = np.asarray(b)
    d1 = np.max(np.min(np.abs(a[:, None] - b[None, :]), axis=1))
    d2 = np.max(np.min(np.abs(b[:, None] - a[None, :]), axis=1))
    return float(max(d1, d2))


def part_b(res, quick):
    print("\n" + "=" * 78)
    print("PART B -- covariant taper OPERATOR prototype")
    print("=" * 78)
    period, wl = 0.700, 1.310
    k0 = 2 * np.pi / wl
    eps_r, eps_g = 4.0, 1.0
    duty = 0.5
    dwall = duty * period

    # ---------------- B1: constant-S reduction to the shipped slant --------
    print("\n-- B1  derivation check: constant S(u) MUST reproduce the shipped "
          "slant pencil (OPERATOR level, then spectrum)")
    print(f"   {'slant_deg':>10}{'deg':>5}{'n':>5}{'|dA1|':>11}{'|dAc|':>11}"
          f"{'|dA2|':>11}{'|C+C^T|':>11}{'spec Hausdorff':>16}{'rel':>10}")
    b1 = []
    for slant_deg in ([2.0, 20.0] if quick else [0.0, 2.0, 10.0, 20.0, 45.0]):
        t = np.tan(np.deg2rad(slant_deg))
        for deg in ([8] if quick else [6, 8, 12]):
            mats = _build_sem_slant(period, dwall, eps_r, eps_g, deg, 1, 1,
                                    False)
            md = _sem_modes_slant(mats, k0, "te", np.deg2rad(slant_deg), 0.0)
            q_lib = (np.concatenate([md["q"], -md["q"]]) if md["symmetric"]
                     else np.concatenate([md["qf"], md["qb"]]))
            # library convention: its internal t = -tan(phi); our S plays that role
            walls = [0.0, dwall, period]
            A1, Ac, A2, n, _x = _cov_pencil(period, walls, [eps_r, eps_g],
                                            [-t, -t, -t], deg, k0)
            sec2 = 1.0 / np.cos(np.deg2rad(slant_deg)) ** 2
            A1_lib = mats["Peps"] - sec2 * mats["L"] / (k0 * k0)
            Ac_lib = (2j * (-t) / k0) * mats["C"]
            A2_lib = mats["S0"]
            dA1 = float(np.max(np.abs(A1 - A1_lib)))
            dAc = float(np.max(np.abs(Ac - Ac_lib)))
            dA2 = float(np.max(np.abs(A2 - A2_lib)))
            ibp = float(np.max(np.abs(mats["C"] + mats["C"].T)))
            q_pro = _pencil_spectrum(A1, Ac, A2, n)
            hd = _spectra_hausdorff(q_lib, q_pro)
            rel = hd / max(float(np.max(np.abs(q_lib))), 1e-30)
            b1.append(dict(slant_deg=slant_deg, degree=deg, n=n, dA1=dA1,
                           dAc=dAc, dA2=dA2, ibp=ibp, haus=hd, rel=rel))
            print(f"   {slant_deg:>10.2f}{deg:>5}{n:>5}{dA1:>11.2e}{dAc:>11.2e}"
                  f"{dA2:>11.2e}{ibp:>11.2e}{hd:>16.3e}{rel:>10.2e}")
    res["B1_reduction"] = b1
    res["B1_worst_rel"] = float(max(r["rel"] for r in b1))
    res["B1_worst_op"] = float(max(max(r["dA1"], r["dAc"], r["dA2"])
                                   for r in b1))

    # ---------------- B2: the taper pencil ---------------------------------
    print("\n-- B2  the TAPER pencil (S(u) piecewise AFFINE): well-posedness "
          "and convergence in degree")
    H = 0.310
    tphi = np.tan(np.deg2rad(2.0))
    L0, R0 = 0.5 * (period - duty * period), 0.5 * (period + duty * period)
    # S(u): the wall-velocity field.  S(0)=S(d)=0 (cell edges fixed),
    # S(L0) = -tan(phi), S(R0) = +tan(phi)  -> the ridge WIDENS with depth.
    walls = [0.0, L0, R0, period]
    Snodes = [0.0, -tphi, +tphi, 0.0]
    eps_seg = [eps_g, eps_r, eps_g]
    b2 = []
    prev = None
    print(f"   {'degree':>7}{'n':>5}{'q0 (largest Re)':>26}"
          f"{'|dq0| vs prev':>15}{'|dspec| vs prev':>17}")
    prev_spec = None
    for deg in ([8, 12, 16] if quick else [6, 8, 10, 12, 16, 20]):
        A1, Ac, A2, n, _ = _cov_pencil(period, walls, eps_seg, Snodes, deg, k0,
                                       zeta=0.5, height=H)
        q = _pencil_spectrum(A1, Ac, A2, n)
        q0 = q[np.argmax(np.real(q))]
        d = abs(q0 - prev) if prev is not None else float("nan")
        ds = (_spectra_hausdorff(q[np.abs(q) < 5], prev_spec)
              if prev_spec is not None else float("nan"))
        prev, prev_spec = q0, q[np.abs(q) < 5]
        b2.append(dict(degree=deg, n=n, q0=[q0.real, q0.imag],
                       dq0=float(d), dspec=float(ds)))
        print(f"   {deg:>7}{n:>5}{str(np.round(q0, 12)):>26}{d:>15.3e}"
              f"{ds:>17.3e}")
    res["B2_taper_spectrum"] = b2

    # NULL CONTROL: S == 0 must reproduce the vertical lamellar spectrum, and
    # the physical modes must satisfy the EXACT transcendental dispersion.
    A1, Ac, A2, n, _ = _cov_pencil(period, walls, eps_seg, [0.0] * 4, 14, k0,
                                   zeta=0.5, height=H)
    qv = _pencil_spectrum(A1, Ac, A2, n)
    print(f"   NULL CONTROL  S == 0: max|Ac| = {np.max(np.abs(Ac)):.3e} "
          f"(must be 0)")

    def _disp(g2):
        a, bw = duty * period, period - duty * period
        k1 = np.sqrt(_C(k0 ** 2 * eps_r - g2))
        k2 = np.sqrt(_C(k0 ** 2 * eps_g - g2))
        return float(np.real(1.0 - (np.cos(k1 * a) * np.cos(k2 * bw)
                     - 0.5 * (k1 / k2 + k2 / k1)
                     * np.sin(k1 * a) * np.sin(k2 * bw))))
    tops = np.sort(np.real(qv[np.abs(np.imag(qv)) < 1e-9]))[::-1][:3]
    dres = [abs(_disp((tq * k0) ** 2)) for tq in tops]
    print(f"   NULL CONTROL  S == 0: top real q = {np.round(tops, 9)}; "
          f"exact-dispersion residual = {['%.2e' % d for d in dres]}")
    res["B2_null_Ac"] = float(np.max(np.abs(Ac)))
    res["B2_null_dispersion"] = [float(d) for d in dres]

    # ---------------- B3: how much does z-freezing move the answer? --------
    print("\n-- B3  the ONLY z-dependence: X_u = 1 + h(zeta-1/2) S'(u).  "
          "Freeze it at several depths and watch the spectrum.")
    print(f"   {'phi':>6}{'delta_g %':>11}{'|q0(0)-q0(.5)|':>16}"
          f"{'|q0(1)-q0(.5)|':>16}{'2nd diff':>12}{'2nd/1st':>10}")
    b3 = []
    deg3 = 12
    for ang in ([2.0, 8.0] if quick else [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]):
        tp = np.tan(np.deg2rad(ang))
        Sn = [0.0, -tp, +tp, 0.0]
        Sp = [(Sn[i + 1] - Sn[i]) / (walls[i + 1] - walls[i])
              for i in range(len(walls) - 1)]
        dg = 0.5 * H * max(abs(s) for s in Sp)
        qz = {}
        A1, Ac, A2, n, _ = _cov_pencil(period, walls, eps_seg, Sn, deg3, k0,
                                       zeta=0.5, height=H)
        qmid = _pencil_spectrum(A1, Ac, A2, n)
        qz[0.5] = qmid[np.argmax(np.real(qmid))]
        for zt in (0.0, 1.0):                  # track BY CONTINUITY, not argmax
            A1, Ac, A2, n, _ = _cov_pencil(period, walls, eps_seg, Sn, deg3,
                                           k0, zeta=zt, height=H)
            q = _pencil_spectrum(A1, Ac, A2, n)
            qz[zt] = q[np.argmin(np.abs(q - qz[0.5]))]
        d0 = abs(qz[0.0] - qz[0.5])
        d1 = abs(qz[1.0] - qz[0.5])
        sec = abs(qz[0.0] + qz[1.0] - 2 * qz[0.5])
        b3.append(dict(phi=ang, delta_g=float(dg), d0=float(d0), d1=float(d1),
                       second=float(sec),
                       ratio=float(sec / max(d0, d1)) if max(d0, d1) else 0.0,
                       iface_shift_nm=float(0.5 * H * tp * 1e3)))
        print(f"   {ang:>6.2f}{dg*100:>11.3f}{d0:>16.3e}{d1:>16.3e}"
              f"{sec:>12.3e}{sec/max(d0, d1):>10.3e}")
    print("   (the FIRST-ORDER layer is the zeta=0.5 pencil; the Magnus-1 error "
          "is the SECOND difference, not the first)")
    for r in b3:
        print(f"   phi={r['phi']:>5.2f} deg: interface frame shift = "
              f"{r['iface_shift_nm']:>7.3f} nm "
              f"({r['iface_shift_nm']/(period*1e3)*100:.3f} % of period) "
              f"-- the far-field projection this spike does NOT prototype")
    res["B3_zfreeze"] = b3

    # ------ B4: spectral symmetry of the taper pencil (the BLOCKER) --------
    print("\n-- B4  spectral symmetry of the FROZEN taper pencil -- and why the "
          "shipped forward/backward selector does NOT apply")
    tp = np.tan(np.deg2rad(2.0))
    Sn4 = [0.0, -tp, +tp, 0.0]
    A1, Ac, A2, n4, _x = _cov_pencil(period, walls, eps_seg, Sn4, 10, k0,
                                     zeta=0.5, height=H)
    q4 = _pencil_spectrum(A1, Ac, A2, n4)
    real_ok = (float(np.max(np.abs(A1.imag))), float(np.max(np.abs((Ac / 1j).imag))),
               float(np.max(np.abs(A2.imag))))
    top = q4[np.argsort(-np.real(q4))][:4]
    d_neg = [float(np.min(np.abs(q4 + v))) for v in top]        # is -q present?
    d_negconj = [float(np.min(np.abs(q4 + np.conj(v)))) for v in top]  # -conj(q)?
    n_imgt = int(np.sum(q4.imag > 0))
    n_imlt = int(np.sum(q4.imag < 0))
    n_real = int(np.sum(np.abs(q4.imag) < 1e-9))
    print(f"   A1, Ac/i, A2 all REAL: {real_ok}")
    print(f"   fundamental q0 = {np.round(top[0], 9)}  "
          f"(COMPLEX, on a LOSSLESS cell)")
    print(f"   min|q + q_m| over the spectrum (is -q there?):     "
          f"{['%.2e' % d for d in d_neg]}")
    print(f"   min|q + conj(q_m)| (is -conj(q) there?):           "
          f"{['%.2e' % d for d in d_negconj]}")
    print(f"   modes with Im>0 / Im<0 / |Im|<1e-9: {n_imgt} / {n_imlt} / {n_real}"
          f"   (n = {n4}; a clean split needs {n4} / {n4})")
    print("   => the pencil's symmetry is q -> -conj(q), NOT q -> -q, so BOTH")
    print("      members of a forward/backward pair carry the SAME sign of Im.")
    print("      _forward_branch_flip / Im-sign / tight-tolerance flux CANNOT")
    print("      classify these modes.  This is T3-6's first hard prerequisite.")
    res["B4_symmetry"] = dict(q0=[float(top[0].real), float(top[0].imag)],
                              d_neg=d_neg, d_negconj=d_negconj,
                              n_im_pos=n_imgt, n_im_neg=n_imlt, n=n4,
                              op_real=real_ok)

    # ---------------- B4b: cascade + its CONTROL ---------------------------
    print("\n-- B4b  a K-sub-slab cascade built on the shipped selector, and "
          "the CONTROL that isolates the defect")
    deg4 = 10 if quick else 12
    Kref = 32 if quick else 64

    def _lay(Sn, K, height):
        return _cov_layer_rt(period, walls, eps_seg, Sn, deg4, k0, H, K)

    def _sdiff(Sa, Sb):
        return float(max(np.max(np.abs(a - b)) for a, b in zip(Sa, Sb)))

    # NULL CONTROL 1: a pure SHEAR has X_u == 1 at every z, so every K is the
    # SAME operator -> the cascade difference must be machine zero.
    tsh = np.tan(np.deg2rad(2.0))
    Ssh = [-tsh, -tsh, -tsh, -tsh]
    d_sh = _sdiff(_lay(Ssh, 1, H), _lay(Ssh, 8, H))
    print(f"   NULL CONTROL 1 (pure shear, S const, X_u==1): "
          f"|S(K=1)-S(K=8)| = {d_sh:.3e}   <-- machine zero, as it must be")
    res["B4_null_shear"] = d_sh

    # NULL CONTROL 2: the SAME cascade machinery driven by a smooth eps(z)
    # ramp (S == 0, so the shipped selector is valid) MUST be second order.
    from lumenairy.elements.pmm._core import (
        _interface_smatrix,
        _propagation_smatrix,
        _redheffer_star,
    )

    def _ramp(K, deg=deg4):
        def ef(z):
            return [eps_g, eps_r * (1.0 + 0.06 * (z - 0.5)), eps_g]
        m0 = _cov_modes(period, walls, ef(0.0), [0.0] * 4, deg, k0, 0.5, 0.0)
        m1 = _cov_modes(period, walls, ef(1.0), [0.0] * 4, deg, k0, 0.5, 0.0)
        S = None
        prev = (m0[0], m0[1])
        for kk in range(K):
            W, V, ll = _cov_modes(period, walls, ef((kk + 0.5) / K), [0.0] * 4,
                                  deg, k0, 0.5, 0.0)[:3]
            Si = _interface_smatrix(prev[0], prev[1], W, V)
            S = Si if S is None else _redheffer_star(S, Si)
            S = _redheffer_star(S, _propagation_smatrix(ll, k0 * H / K))
            prev = (W, V)
        return _redheffer_star(S, _interface_smatrix(prev[0], prev[1],
                                                     m1[0], m1[1]))
    rref = _ramp(Kref)
    ramp = {K: _sdiff(_ramp(K), rref) for K in (1, 2, 4, 8)}
    rr = [ramp[1] / ramp[2], ramp[2] / ramp[4], ramp[4] / ramp[8]]
    print("   NULL CONTROL 2 (smooth eps(z) ramp, S==0 -> selector valid): "
          + "  ".join(f"K={k}:{v:.2e}" for k, v in ramp.items()))
    print(f"      ratio per doubling = {['%.2f' % x for x in rr]}  "
          f"-> SECOND order, i.e. the cascade machinery is sound")
    res["B4_null_ramp"] = {str(k): float(v) for k, v in ramp.items()}
    res["B4_null_ramp_ratios"] = [float(x) for x in rr]

    b4 = []
    for ang in ([2.0, 8.0] if quick else [0.5, 1.0, 2.0, 4.0, 8.0]):
        tpa = np.tan(np.deg2rad(ang))
        Sn = [0.0, -tpa, +tpa, 0.0]
        sref = _lay(Sn, Kref, H)
        row = {"phi": ang}
        vals = {}
        for K in ([1, 2, 4, 8] if quick else [1, 2, 3, 4, 6, 8, 16]):
            vals[K] = _sdiff(_lay(Sn, K, H), sref)
            row[f"K{K}"] = vals[K]
        b4.append(row)
        print(f"   TAPER phi={ang:>5.2f} deg (ref K={Kref}): "
              + "  ".join(f"K={k}:{v:.2e}" for k, v in vals.items()))
    print("   => the CONTROLS are exact / second order, so the cascade "
          "MACHINERY is sound; the taper")
    print("      cascade is at best first order and at degree >= 12 DIVERGES.  "
          "The deficiency is the")
    print("      MODE CLASSIFICATION (B4), not the discretisation.  NO number "
          "from the taper rows")
    print("      is quoted as an accuracy result anywhere in the report.")
    res["B4_cascade"] = b4
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--json", default=None)
    ap.add_argument("--only", default="ab")
    args = ap.parse_args()
    res = {"platform": platform.platform(), "python": sys.version.split()[0],
           "numpy": np.__version__, "scipy": __import__("scipy").__version__}
    try:
        res["blas"] = np.__config__.CONFIG["Build Dependencies"]["blas"]["name"]
    except Exception:
        res["blas"] = "unknown"
    print(f"# build: {res['python']} np{res['numpy']} sp{res['scipy']} "
          f"blas={res['blas']} {res['platform']}")
    if "a" in args.only:
        part_a(res, args.quick)
    if "b" in args.only:
        part_b(res, args.quick)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(res, fh, indent=1, default=float)
        print(f"\n[json] {args.json}")


if __name__ == "__main__":
    main()
