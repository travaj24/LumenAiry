"""BUILD-time measurement of the shipped per-layer / non-uniform surface.

Every number in ``docs/audits/BUILD_PMM2D_STAGGERED_MORTAR_2026_09_11.md``
comes from this script (run on BOTH builds; see that doc's tables).  Unlike the
``m*``/``f*`` prototypes this drives the LIBRARY API only -- no research code.

    python validation/probe_pmm2d_staggered_mortar/b1_build_gates.py [gate ...]
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

WORKTREE = os.path.normcase(os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..")))
sys.path.insert(0, WORKTREE)

import lumenairy  # noqa: E402

_p = os.path.normcase(os.path.abspath(lumenairy.__file__))
if not _p.startswith(WORKTREE):
    raise SystemExit(f"REFUSED: lumenairy is {_p}, not under {WORKTREE}")

from lumenairy.elements.pmm import PMM2DStackPure, PMMStack  # noqa: E402
from lumenairy.elements.pmm import _core as _pmmcore  # noqa: E402
from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    Basis1D,
    Granet2DTransverseE,
    _stag_cross_mass_1d,
    _stag_kron_apply,
)

OUT = {}
P = 1.2
WL = 0.85


def _pillar(N, lo, hi, e_p=6.0, e_h=2.25):
    """A pillar occupying segments [lo, hi) of an N x N uniform lattice."""
    c = np.full((N, N), complex(e_h))
    c[lo:hi, lo:hi] = e_p
    return c


def _stripe(N, lo, hi, e_p=6.0, e_h=2.25):
    c = np.full((N, N), complex(e_h))
    c[lo:hi, :] = e_p
    return c


def _solve(st, **kw):
    t = time.perf_counter()
    out = st.solve(**kw)
    return out, time.perf_counter() - t


# ---------------------------------------------------------------- G2 + G3
def gate_g2_g3():
    """Conforming identity through the FORCED mortar, and the H-swap
    fail-before on a genuinely NON-conforming pair."""
    rows = []
    for name, cells, M, th, ph in (
            ("stripe|stripe", (_stripe(2, 0, 1), _stripe(2, 1, 2)), 5, 0.0, 0.0),
            ("stripe|stripe obl", (_stripe(2, 0, 1), _stripe(2, 1, 2)), 5, 0.20, 0.0),
            ("stripe|pillar", (_stripe(3, 0, 1), _pillar(3, 1, 2)), 6, 0.20, 0.0),
            ("pillar|pillar con", (_pillar(3, 0, 2), _pillar(3, 1, 3)), 6, 0.25, 0.7),
            ("3-region", (_stripe(2, 0, 1), None, _pillar(2, 0, 1)), 5, 0.15, 0.0),
    ):
        st_a = PMM2DStackPure(P, n_modes=M, n_orders=2)
        st_b = PMM2DStackPure(P, n_modes=M, n_orders=2, layer_grids="per-layer")
        for c in cells:
            if c is None:
                st_a.add_layer(0.2, eps=2.0)
                st_b.add_layer(0.2, eps=2.0, grid=cells[0].shape[0], n_modes=M)
            else:
                st_a.add_layer(0.3, eps_cell=c)
                st_b.add_layer(0.3, eps_cell=c)
        st_a.set_source(WL, theta=th, phi=ph)
        st_b.set_source(WL, theta=th, phi=ph)
        (_o, R1, T1, J1), _t = _solve(st_a)
        _o2, R2, T2, J2 = st_b._solve_per_layer(jones=True, retain_internal=False,
                                                force_mortar=True)
        sc = max(float(np.max(R1)), float(np.max(T1)))
        rows.append(dict(case=name, M=M,
                         dR=float(np.max(np.abs(R1 - R2))) / sc,
                         dT=float(np.max(np.abs(T1 - T2))) / sc,
                         dJ=float(np.max(np.abs(J1 - J2)))
                         / float(np.max(np.abs(J1)))))
    OUT["g2_forced_mortar_identity"] = rows

    # the Gram condition numbers that DERIVE the bar
    conds = []
    for N, M in ((2, 5), (3, 6), (2, 6), (3, 5)):
        sol = Granet2DTransverseE(P, P, N, N, M, np.full((N, N), 2.0 + 0j),
                                  alpha0x=0.2, alpha0y=0.0, k0=2 * np.pi / WL)
        G = -sol.Rmat
        qq = sol.q ** 2
        conds.append(dict(N=N, M=M,
                          cond_G1=float(np.linalg.cond(G[:qq, :qq])),
                          cond_G2=float(np.linalg.cond(G[qq:, qq:]))))
    OUT["gram_cond"] = conds

    # ---- G3: the H-row V1/V2 swap, on a NON-conforming pair --------------
    cA, cB = _pillar(2, 0, 1), _pillar(3, 1, 2)
    ref = PMM2DStackPure(P, n_modes=4, n_orders=2)
    ref.add_layer(0.30, eps_cell=np.kron(cA, np.ones((3, 3))))
    ref.add_layer(0.25, eps_cell=np.kron(cB, np.ones((2, 2))))
    ref.set_source(WL, theta=0.18, phi=0.35)
    (_o, Rr, Tr, Jr), _t = _solve(ref)

    def _arm():
        st = PMM2DStackPure(P, n_modes=7, n_orders=2, layer_grids="per-layer")
        st.add_layer(0.30, eps_cell=cA, n_modes=9)
        st.add_layer(0.25, eps_cell=cB, n_modes=7)
        st.set_source(WL, theta=0.18, phi=0.35)
        return st.solve()

    got = {}
    for swap in (True, False):
        _pmmcore.PMM2D_MORTAR_H_SWAP = swap
        try:
            _o, R, T, J = _arm()
        finally:
            _pmmcore.PMM2D_MORTAR_H_SWAP = True
        got["swap_on" if swap else "swap_off"] = dict(
            err=float(np.max(np.abs(R - Rr))) + float(np.max(np.abs(T - Tr))),
            closure=float(np.max(np.abs(R.sum(1) + T.sum(1) - 1.0))))
    OUT["g3_h_swap"] = got


# ---------------------------------------------------------------- G4
def gate_g4():
    """Transparent interface: one uniform slab SPLIT across grids, against the
    analytic Fresnel slab (scalar) and berreman_jones_1d (out-of-plane)."""
    from lumenairy.elements.berreman import berreman_jones_1d
    n, t = 2.0, 0.30
    rows = []
    for th in (0.0, 0.20):
        # analytic TE Fresnel slab
        k0 = 2 * np.pi / WL
        kz0 = k0 * np.cos(th)
        kz1 = k0 * np.sqrt(n ** 2 - np.sin(th) ** 2 + 0j)
        r01 = (kz0 - kz1) / (kz0 + kz1)
        ph = np.exp(2j * kz1 * t)
        r = (r01 + (-r01) * ph) / (1 + r01 * (-r01) * ph)
        R_ex = float(abs(r) ** 2)
        for M in (5, 7):
            for ga, gb in ((2, 2), (2, 4), (2, 3), (3, 4)):
                st = PMM2DStackPure(P, n_modes=M, n_orders=1,
                                    layer_grids="per-layer")
                st.add_layer(0.5 * t, eps=n ** 2, grid=ga, n_modes=M)
                st.add_layer(0.5 * t, eps=n ** 2, grid=gb, n_modes=M)
                st.set_source(WL, theta=th, phi=0.0)
                o, R, T, J = st.solve()
                p0 = int(np.where((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
                rows.append(dict(theta=th, M=M, grids=[ga, gb],
                                 dR=abs(float(R[1, p0]) - R_ex),
                                 closure=float(abs(R.sum(1)[1] + T.sum(1)[1] - 1))))
    OUT["g4_fresnel_split"] = rows

    # OUT-OF-PLANE: one uniform LC slab split across grids vs Berreman
    def _uni(no, ne, tilt, azim):
        c, s = np.cos(tilt), np.sin(tilt)
        ca, sa = np.cos(azim), np.sin(azim)
        d = np.array([s * ca, s * sa, c])
        return no ** 2 * np.eye(3) + (ne ** 2 - no ** 2) * np.outer(d, d)

    eps33 = _uni(1.5, 1.7, np.deg2rad(35.0), np.deg2rad(25.0))
    rows = []
    for th_deg, ph_deg in ((0.0, 0.0), (25.0, 40.0)):
        th, phz = np.deg2rad(th_deg), np.deg2rad(ph_deg)
        Rb, Tb, Jb, _Jt = berreman_jones_1d([(eps33, 0.35)], 1.5, 1.0, WL,
                                            angle=th, phi=phz)
        Jb = np.asarray(Jb)
        for M in (5, 7):
            for ga, gb in ((2, 2), (2, 3), (3, 4)):
                st = PMM2DStackPure(P, n_superstrate=1.0, n_substrate=1.5,
                                    n_modes=M, n_orders=1,
                                    layer_grids="per-layer")
                st.add_layer(0.175, eps=eps33, grid=ga, n_modes=M)
                st.add_layer(0.175, eps=eps33, grid=gb, n_modes=M)
                st.set_source(WL, theta=th, phi=phz)
                o, R, T, J = st.solve()
                rows.append(dict(theta_deg=th_deg, M=M, grids=[ga, gb],
                                 dJ=float(np.max(np.abs(J - Jb)))
                                 / float(np.max(np.abs(Jb))),
                                 dR=float(np.max(np.abs(R.sum(1) - Rb))),
                                 closure=float(np.max(np.abs(R.sum(1) + T.sum(1) - 1)))))
    OUT["g4_oop_split"] = rows


# ---------------------------------------------------------------- G5 / G6
def _oracle_1d(duties, ts, M1d=14, theta=0.20, per=0.9, wl=0.6):
    st = PMMStack(per, degree=M1d, far_field_orders=5)
    for duty, t in zip(duties, ts):
        st.add_layer(t, widths=[duty * per, (1 - duty) * per], eps=[6.0, 2.25])
    st.set_source(wl, theta=theta)
    return st.solve()


def gate_g5_g6():
    """Stripe pair per order against the exact 1-D PMMStack, and the EQUAL-DOF
    non-regression ratio."""
    per, wl, theta = 0.9, 0.6, 0.20
    duties, ts = (0.5, 1.0 / 3.0), (0.30, 0.22)

    def _oracle(deg):
        st = PMMStack(per, degree=deg, far_field_orders=5)
        for duty, t in zip(duties, ts):
            st.add_layer(t, widths=[duty * per, (1 - duty) * per],
                         eps=[6.0, 2.25])
        st.set_source(wl, theta=theta)
        return st.solve()

    o12, R12, T12 = _oracle(12)[:3]
    o14, R14, T14 = _oracle(14)[:3]
    R14 = np.atleast_2d(R14)
    T14 = np.atleast_2d(T14)
    R12 = np.atleast_2d(R12)
    T12 = np.atleast_2d(T12)
    keep = np.abs(np.asarray(o14)) <= 1
    self_gap = float(max(np.max(np.abs(R12[:, keep] - R14[:, keep])),
                         np.max(np.abs(T12[:, keep] - T14[:, keep]))))

    def _score(o2d, R, T, mirror=False):
        best = 0.0
        for m in (-1, 0, 1):
            sel = np.where((o2d[:, 0] == (-m if mirror else m))
                           & (o2d[:, 1] == 0))[0][0]
            j = int(np.where(np.asarray(o14) == m)[0][0])
            # TE row: incident E_y
            best = max(best, abs(float(R[1, sel]) - float(R14[1, j])),
                       abs(float(T[1, sel]) - float(T14[1, j])))
        return best

    def _cellA(N):
        c = np.full((N, N), 2.25 + 0j)
        c[:N // 2, :] = 6.0
        return c

    def _cellB(N):
        c = np.full((N, N), 2.25 + 0j)
        c[:N // 3, :] = 6.0
        return c

    rows = []
    for MA, MB in ((5, 5), (7, 7), (9, 9), (11, 11)):
        st = PMM2DStackPure(per, n_modes=max(MA, MB), n_orders=1,
                            layer_grids="per-layer")
        st.add_layer(ts[0], eps_cell=_cellA(2), n_modes=MA)
        st.add_layer(ts[1], eps_cell=_cellB(3), n_modes=MB)
        st.set_source(wl, theta=theta)
        (o, R, T, J), t = _solve(st)
        rows.append(dict(MA=MA, MB=MB, err=_score(o, R, T),
                         mirror=_score(o, R, T, True),
                         closure=float(abs(R.sum(1)[1] + T.sum(1)[1] - 1)),
                         wall=t))
    OUT["g5_vs_1d"] = dict(oracle_self_gap=self_gap, rows=rows)

    # ---- G6 EQUAL DOF: q = 6(M-1) union vs q_A = 2(MA-1), q_B = 3(MB-1) ----
    eq = []
    for Mu in (3, 4):
        q = 6 * (Mu - 1)
        stu = PMM2DStackPure(per, n_modes=Mu, n_orders=1)
        stu.add_layer(ts[0], eps_cell=_cellA(6))
        stu.add_layer(ts[1], eps_cell=_cellB(6))
        stu.set_source(wl, theta=theta)
        (ou, Ru, Tu, _), tu = _solve(stu)
        stm = PMM2DStackPure(per, n_modes=8, n_orders=1,
                             layer_grids="per-layer")
        stm.add_layer(ts[0], eps_cell=_cellA(2), n_modes=q // 2 + 1)
        stm.add_layer(ts[1], eps_cell=_cellB(3), n_modes=q // 3 + 1)
        stm.set_source(wl, theta=theta)
        (om, Rm, Tm, _), tm = _solve(stm)
        eu, em = _score(ou, Ru, Tu), _score(om, Rm, Tm)
        eq.append(dict(q=q, eig_dim=2 * q * q,
                       union=dict(M=Mu, err=eu, wall=tu,
                                  closure=float(abs(Ru.sum(1)[1] + Tu.sum(1)[1] - 1))),
                       mortar=dict(MA=q // 2 + 1, MB=q // 3 + 1, err=em, wall=tm,
                                   closure=float(abs(Rm.sum(1)[1] + Tm.sum(1)[1] - 1))),
                       ratio_err=em / eu))
    OUT["g6_equal_dof"] = eq


# ---------------------------------------------------------------- N-gates
def gate_nonuniform():
    """N3 (two exact-wall representations of one device, triangle-inequality
    bar), N4 (arbitrary walls vs the exact 1-D oracle), N6 (the conforming
    identity ON non-uniform grids)."""
    per, wl, theta = 0.9, 0.6, 0.20

    # ---- N3: a stripe at duty 1/3 as 2 NON-uniform segments vs 3 uniform ---
    def _oracle(deg):
        st = PMMStack(per, degree=deg, far_field_orders=5)
        st.add_layer(0.30, widths=[per / 3.0, 2.0 * per / 3.0], eps=[6.0, 2.25])
        st.set_source(wl, theta=theta)
        return st.solve()

    o14, R14, T14 = _oracle(14)[:3]
    o12, R12, T12 = _oracle(12)[:3]
    R14, T14 = np.atleast_2d(R14), np.atleast_2d(T14)
    keep = np.abs(np.asarray(o14)) <= 1
    self_gap = float(max(np.max(np.abs(np.atleast_2d(R12)[:, keep] - R14[:, keep])),
                         np.max(np.abs(np.atleast_2d(T12)[:, keep] - T14[:, keep]))))

    def _score(o2d, R, T):
        best = 0.0
        for m in (-1, 0, 1):
            sel = np.where((o2d[:, 0] == m) & (o2d[:, 1] == 0))[0][0]
            j = int(np.where(np.asarray(o14) == m)[0][0])
            best = max(best, abs(float(R[1, sel]) - float(R14[1, j])),
                       abs(float(T[1, sel]) - float(T14[1, j])))
        return best

    rows = []
    for M in (4, 5, 6, 7, 9):
        # NON-uniform, 2 segments, wall at per/3
        cnu = np.array([[6.0, 6.0], [2.25, 2.25]], dtype=complex)
        stn = PMM2DStackPure(per, n_modes=M, n_orders=1,
                             layer_grids="per-layer")
        stn.add_layer(0.30, eps_cell=cnu, x_walls=[per / 3.0],
                      y_walls=[per / 3.0])
        stn.set_source(wl, theta=theta)
        (on, Rn, Tn, _), tn = _solve(stn)
        # uniform, 3 segments
        cu = np.full((3, 3), 2.25 + 0j)
        cu[0, :] = 6.0
        stu = PMM2DStackPure(per, n_modes=M, n_orders=1)
        stu.add_layer(0.30, eps_cell=cu)
        stu.set_source(wl, theta=theta)
        (ou, Ru, Tu, _), tu = _solve(stu)
        en, eu = _score(on, Rn, Tn), _score(ou, Ru, Tu)
        gapv = 0.0
        for m in (-1, 0, 1):
            i1 = np.where((on[:, 0] == m) & (on[:, 1] == 0))[0][0]
            i2 = np.where((ou[:, 0] == m) & (ou[:, 1] == 0))[0][0]
            gapv = max(gapv, abs(float(Rn[1, i1]) - float(Ru[1, i2])),
                       abs(float(Tn[1, i1]) - float(Tu[1, i2])))
        rows.append(dict(M=M, q_nu=2 * (M - 1), q_u=3 * (M - 1),
                         err_nu=en, err_u=eu, gap=gapv, bar=en + eu,
                         inside=bool(gapv <= en + eu)))
    OUT["n3_two_representations"] = dict(oracle_self_gap=self_gap, rows=rows)

    # ---- N4: ARBITRARY walls, y-uniform, vs the exact 1-D oracle ----------
    w0, w1 = 0.2371 * per, 0.6183 * per

    def _oracle_arb(deg):
        st = PMMStack(per, degree=deg, far_field_orders=5)
        st.add_layer(0.30, widths=[w0, w1 - w0, per - w1],
                     eps=[2.25, 6.0, 2.25])
        st.set_source(wl, theta=theta)
        return st.solve()

    oa, Ra, Ta = _oracle_arb(14)[:3]
    ob, Rb, Tb = _oracle_arb(12)[:3]
    Ra, Ta = np.atleast_2d(Ra), np.atleast_2d(Ta)
    keep = np.abs(np.asarray(oa)) <= 1
    gap_arb = float(max(np.max(np.abs(np.atleast_2d(Rb)[:, keep] - Ra[:, keep])),
                        np.max(np.abs(np.atleast_2d(Tb)[:, keep] - Ta[:, keep]))))
    tile = np.array([[2.25, 2.25, 2.25], [6.0, 6.0, 6.0], [2.25, 2.25, 2.25]],
                    dtype=complex)
    rows = []
    for M in (4, 5, 6, 7, 9):
        st = PMM2DStackPure(per, n_modes=M, n_orders=1,
                            layer_grids="per-layer")
        st.add_layer(0.30, eps_cell=tile, x_walls=[w0, w1],
                     y_walls=[w0, w1])
        st.set_source(wl, theta=theta)
        (o, R, T, _), t = _solve(st)
        best = 0.0
        for m in (-1, 0, 1):
            sel = np.where((o[:, 0] == m) & (o[:, 1] == 0))[0][0]
            j = int(np.where(np.asarray(oa) == m)[0][0])
            best = max(best, abs(float(R[1, sel]) - float(Ra[1, j])),
                       abs(float(T[1, sel]) - float(Ta[1, j])))
        rows.append(dict(M=M, q=3 * (M - 1), err=best, wall=t,
                         closure=float(abs(R.sum(1)[1] + T.sum(1)[1] - 1))))
    OUT["n4_arbitrary_walls"] = dict(oracle_self_gap=gap_arb, rows=rows)

    # ---- N6: the conforming identity ON non-uniform grids -----------------
    rows = []
    for M in (5, 7):
        def _build():
            st = PMM2DStackPure(per, n_modes=M, n_orders=1,
                                layer_grids="per-layer")
            for _ in range(3):
                st.add_layer(0.10, eps_cell=tile, x_walls=[w0, w1],
                             y_walls=[w0, w1])
            st.set_source(wl, theta=theta)
            return st
        o1, R1, T1, J1 = _build().solve()
        o2, R2, T2, J2 = _build()._solve_per_layer(jones=True,
                                                   retain_internal=False,
                                                   force_mortar=True)
        rows.append(dict(M=M, dR=float(np.max(np.abs(R1 - R2))),
                         dT=float(np.max(np.abs(T1 - T2))),
                         dJ=float(np.max(np.abs(J1 - J2)))))
    OUT["n6_nonuniform_conforming_identity"] = rows


# ---------------------------------------------------------------- G7 memory
def gate_memory_cond():
    """The separable cross-mass, priced; and the mortar conditioning census."""
    rows = []
    for (Na, Nb), M in (((2, 3), 6), ((3, 6), 6), ((4, 6), 8), ((6, 12), 6)):
        ba = Basis1D(P, Na, M, 1.0 + 0j)
        bb = Basis1D(P, Nb, M, 1.0 + 0j)
        Cx = _stag_cross_mass_1d(ba, bb, "B")
        Cy = _stag_cross_mass_1d(ba, bb, "Btilde")
        dense_mb = Cx.size * Cy.size * 16 / 1e6
        fac_mb = (Cx.nbytes + Cy.nbytes) / 1e6
        X = np.random.default_rng(0).standard_normal(
            (Cy.shape[1] * Cx.shape[1], 4)) + 0j
        t0 = time.perf_counter()
        for _ in range(3):
            Y1 = _stag_kron_apply(Cy, Cx, X)
        t_fac = (time.perf_counter() - t0) / 3
        K = np.kron(Cy, Cx)
        t0 = time.perf_counter()
        for _ in range(3):
            Y2 = K @ X
        t_dense = (time.perf_counter() - t0) / 3
        rows.append(dict(grids=[Na, Nb], M=M, dense_MB=dense_mb,
                         factors_MB=fac_mb, mem_ratio=dense_mb / fac_mb,
                         speed_ratio=t_dense / t_fac,
                         identity=float(np.max(np.abs(Y1 - Y2)))
                         / float(np.max(np.abs(Y2)))))
    OUT["cross_mass_memory"] = rows

    # conditioning census through the shipped guard's own instrument
    from lumenairy.elements.rcwa import _core as _rc
    cens = []
    for name, (cA, cB, wa, wb) in {
        "pillar (2,3)": (_pillar(2, 0, 1), _pillar(3, 1, 2), None, None),
        "pillar (2,6)": (_pillar(2, 0, 1), _pillar(6, 2, 4), None, None),
        "stripe (2,3)": (_stripe(2, 0, 1), _stripe(3, 0, 1), None, None),
    }.items():
        for M in (4, 5, 6):
            prev, _rc._INV_CENSUS = _rc._INV_CENSUS, []
            try:
                st = PMM2DStackPure(P, n_modes=M, n_orders=2,
                                    layer_grids="per-layer")
                st.add_layer(0.30, eps_cell=cA, x_walls=wa, y_walls=wa)
                st.add_layer(0.25, eps_cell=cB, x_walls=wb, y_walls=wb)
                st.set_source(WL, theta=0.18, phi=0.35)
                st.solve()
                hits = [(si, n, rc)
                        for (si, n, rc, _res, _f) in _rc._INV_CENSUS]
            finally:
                _rc._INV_CENSUS = prev
            worst = min((rc for _s, _n, rc in hits), default=float("nan"))
            cens.append(dict(config=name, M=M, calls=len(hits),
                             worst_rcond=float(worst),
                             sites=sorted({si for si, _n, _rc in hits})))
    OUT["conditioning_census"] = cens


# ---------------------------------------------------------------- tapers
def gate_taper():
    """The taper: a per-layer NON-UNIFORM staircase vs the hybrid's own
    add_tapered_pillar staircase, and (y-uniform) vs the exact 1-D oracle."""
    from lumenairy.elements.pmm import PMM2DStackHybrid
    per, wl = 1.2, 0.85
    xb0, xb1 = (0.1873 * per, 0.7241 * per), (0.2917 * per, 0.6109 * per)
    nsl, thick = 4, 0.30
    hy = PMM2DStackHybrid(per, n_orders=9, degree=9)
    hy.add_tapered_pillar(thick, eps_pillar=6.0, eps_host=2.25,
                          x_bounds_bottom=xb0, y_bounds_bottom=xb0,
                          x_bounds_top=xb1, y_bounds_top=xb1, n_slices=nsl)
    hy.set_source(wl, theta=0.18, phi=0.35)
    (oh, Rh, Th, Jh), th_t = _solve(hy)
    hy2 = PMM2DStackHybrid(per, n_orders=9, degree=11)
    hy2.add_tapered_pillar(thick, eps_pillar=6.0, eps_host=2.25,
                           x_bounds_bottom=xb0, y_bounds_bottom=xb0,
                           x_bounds_top=xb1, y_bounds_top=xb1, n_slices=nsl)
    hy2.set_source(wl, theta=0.18, phi=0.35)
    oh2, Rh2, Th2, _ = hy2.solve()
    p0h = int(np.where((np.asarray(oh)[:, 0] == 0)
                       & (np.asarray(oh)[:, 1] == 0))[0][0])
    p0h2 = int(np.where((np.asarray(oh2)[:, 0] == 0)
                        & (np.asarray(oh2)[:, 1] == 0))[0][0])
    hyb_self = float(abs(Rh[0, p0h] - Rh2[0, p0h2]))
    rows = []
    for M in (4, 5, 6):
        st = PMM2DStackPure(per, n_modes=M, n_orders=1,
                            layer_grids="per-layer")
        st.add_tapered_pillar(thick, eps_pillar=6.0, eps_host=2.25,
                              x_bounds_bottom=xb0, y_bounds_bottom=xb0,
                              x_bounds_top=xb1, y_bounds_top=xb1,
                              n_slices=nsl)
        st.set_source(wl, theta=0.18, phi=0.35)
        (o, R, T, J), t = _solve(st)
        p0 = int(np.where((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
        rows.append(dict(M=M, q=3 * (M - 1),
                         R00=float(R[0, p0]),
                         err_vs_hybrid=float(abs(R[0, p0] - Rh[0, p0h])),
                         closure=float(np.max(np.abs(R.sum(1) + T.sum(1) - 1))),
                         wall=t))
    OUT["taper_vs_hybrid"] = dict(hybrid_self_gap=hyb_self,
                                  hybrid_closure=float(np.max(np.abs(
                                      np.atleast_2d(Rh).sum(1)
                                      + np.atleast_2d(Th).sum(1) - 1))),
                                  hybrid_wall=th_t, rows=rows)


GATES = {"g2g3": gate_g2_g3, "g4": gate_g4, "g5g6": gate_g5_g6,
         "nu": gate_nonuniform, "mem": gate_memory_cond, "taper": gate_taper}

if __name__ == "__main__":
    want = sys.argv[1:] or list(GATES)
    for nm in want:
        t0 = time.perf_counter()
        GATES[nm]()
        print(f"[{nm}] {time.perf_counter() - t0:.1f} s", flush=True)
    tag = os.environ.get("B1_TAG", "win")
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        f"b1_build_gates_{tag}.json")
    old = {}
    if os.path.exists(path):
        with open(path) as fh:
            old = json.load(fh)
    old.update(OUT)
    with open(path, "w") as fh:
        json.dump(old, fh, indent=1, sort_keys=True)
    print(json.dumps(OUT, indent=1, sort_keys=True))
