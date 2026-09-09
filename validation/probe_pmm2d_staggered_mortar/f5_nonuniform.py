"""F5 -- NON-UNIFORM SEGMENT BOUNDARIES: feasibility gates (a)-(d).

The taper enabler (roadmap item N-1).  ``nonuniform.py`` carries the
generalized basis; this script is its evidence.

  (a) UNIFORM boundaries reproduce the shipped ``Basis1D`` BIT-IDENTICALLY --
      1-D matrices, the per-segment tensors, the far-field projection, and the
      full 2-D operators on the in-plane, tensor and OUT-OF-PLANE paths.
  (b) a 2-SEGMENT non-uniform grid with its wall at 1/3 reproduces the
      3-SEGMENT uniform grid's single-layer R/T -- two exact-wall
      representations of ONE device, so they must converge to the same answer;
      the stripe twin is additionally scored against the EXACT 1-D ``PMMStack``.
  (c) a cell with walls at ARBITRARY positions on its own 3x3 non-uniform grid,
      against the hybrid PMM (exact walls) and RCWA -- the only oracles that
      can express such a cell at all.
  (d) a 4-SLICE TAPER as a per-layer non-uniform MORTAR cascade against the
      hybrid's ``add_tapered_pillar`` staircase with the same slices.

Usage:  python f5_nonuniform.py [a] [b] [c] [d]      (default: all)
"""
import json
import os
import sys
import time
import warnings

import numpy as np
from mortar2d import guard
print("lumenairy:", guard(), flush=True)
from nonuniform import (Basis1DNU, Granet2DTransverseE_NU, MortarStackNU,
                        global_pair_segmat_nu, stag_fourier_projection_nu)
from lumenairy import PMMStack
from lumenairy.elements.pmm.stack2d import PMM2DStackHybrid
from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure
from lumenairy.elements.pmm.twod import pmm_efficiency_2d_cell
from lumenairy.elements.pmm.twod_staggered import (
    Basis1D, Granet2DTransverseE, _global_pair_segmat,
    _stag_fourier_projection)
from lumenairy.elements.rcwa import rcwa_efficiency_2d
from lumenairy.elements.rcwa._core import uniaxial_tensor

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
WHICH = [a.lower() for a in sys.argv[1:]] or ["a", "b", "c", "c3", "d", "d2"]
res = {}

PX = PY = 1.2e-6
WL = 0.85e-6
EPS_H, EPS_P = 2.25, 9.0


def n0_row(orders, R, T, pol=1):
    """The (m, 0) row of a 2-D result, sorted by m."""
    o = np.asarray(orders)
    sel = o[:, 1] == 0
    m = o[sel, 0]
    i = np.argsort(m)
    return m[i], np.asarray(R)[pol][sel][i], np.asarray(T)[pol][sel][i]


# ======================================================================= (a)
if "a" in WHICH:
    print("\n=== GATE (a) -- uniform boundaries reproduce the shipped basis ===",
          flush=True)
    rows = []
    tau = np.exp(-0.37j)
    for N, M in ((2, 5), (3, 6), (4, 4), (6, 4)):
        a = Basis1D(PX, N, M, tau)
        b = Basis1DNU(PX, N, M, tau)
        checks = {
            "mass<til|til>": (a.mass(a.Btilde, a.Btilde),
                              b.mass(b.Btilde, b.Btilde)),
            "mass<B|B>": (a.mass(a.B, a.B), b.mass(b.B, b.B)),
            "mass eps-weighted": (a.mass(a.B, a.B, np.arange(1., N + 1)),
                                  b.mass(b.B, b.B, np.arange(1., N + 1))),
            "stiff": (a.stiff(a.Btilde, a.Btilde), b.stiff(b.Btilde, b.Btilde)),
            "mixed<B|d til>": (a.mixed(a.B, a.Btilde), b.mixed(b.B, b.Btilde)),
            "segmat(m_ref)": (_global_pair_segmat(a, a.m_ref, a.B, a.B),
                              global_pair_segmat_nu(b, b.m_ref, b.B, b.B)),
            "fourier proj": (_stag_fourier_projection(a, np.arange(-2, 3),
                                                      0.31)(a.B),
                             stag_fourier_projection_nu(b, np.arange(-2, 3),
                                                        0.31)(b.B)),
        }
        ok = all(np.array_equal(u, v) for u, v in checks.values())
        worst = max(float(np.max(np.abs(u - v))) for u, v in checks.values())
        rows.append(dict(kind="1D int-N", N=N, M=M, bit_identical=bool(ok),
                         worst_abs=worst))
        print(f"  1-D  N={N} M={M}  bit-identical over "
              f"{len(checks)} matrices: {ok}  (worst |d| {worst:.1e})",
              flush=True)
        # the EXPLICIT-linspace path: same lattice, different arithmetic
        c = Basis1DNU(PX, np.linspace(0.0, PX, N + 1), M, tau)
        fa = a.mass(a.Btilde, a.Btilde)
        fc = c.mass(c.Btilde, c.Btilde)
        rel = float(np.max(np.abs(fa - fc)) / np.max(np.abs(fa)))
        rows.append(dict(kind="1D linspace-walls", N=N, M=M,
                         bit_identical=bool(np.array_equal(fa, fc)),
                         rel=rel))
        print(f"       explicit linspace walls: bit-identical "
              f"{np.array_equal(fa, fc)}  rel {rel:.2e}", flush=True)

    # 2-D operators: scalar, in-plane tensor, out-of-plane
    k0 = 2 * np.pi / WL
    cell3 = np.array([[6.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 3.5]],
                     complex)
    T33 = np.asarray(uniaxial_tensor(1.5, 1.7, np.deg2rad(35.0),
                                     phi=np.deg2rad(25.0)), complex)
    tc = np.empty((2, 2, 3, 3), complex)
    tc[:, :] = np.eye(3) * 2.25
    tc[0, 0] = T33
    ip = np.empty((2, 2, 3, 3), complex)
    ip[:, :] = np.diag([2.25, 3.1, 2.7])
    ip[0, 0] = np.array([[6.0, 0.9j, 0], [-0.9j, 6.0, 0], [0, 0, 5.0]])
    for tag, args, attrs in (
            ("scalar 3x3", (3, 3, 5, cell3), ("Rmat", "Lmat", "Stt", "Schur")),
            ("in-plane tensor 2x2", (2, 2, 5, ip),
             ("Rmat", "Lmat", "Stt", "Schur")),
            ("OUT-OF-PLANE 2x2", (2, 2, 5, tc), ("Agen", "Bgen"))):
        nx, ny, M, cell = args
        A = Granet2DTransverseE(PX, PY, nx, ny, M, cell, alpha0x=0.31 * k0,
                                alpha0y=0.12 * k0, k0=k0)
        B = Granet2DTransverseE_NU(PX, PY, nx, ny, M, cell, alpha0x=0.31 * k0,
                                   alpha0y=0.12 * k0, k0=k0)
        ok = all(np.array_equal(getattr(A, n), getattr(B, n)) for n in attrs)
        rows.append(dict(kind="2D operators", case=tag, bit_identical=bool(ok)))
        print(f"  2-D  {tag:22s} {'/'.join(attrs)}  bit-identical: {ok}",
              flush=True)
    res["a"] = rows

# ======================================================================= (b)
if "b" in WHICH:
    print("\n=== GATE (b) -- 2 NON-uniform segments == 3 uniform segments ===",
          flush=True)
    THETA, PHI, NORD = 0.18, 0.0, 2
    DEP = 0.30e-6

    # -- (b1) 2-D pillar occupying [0, P/3] x [0, P/3] --------------------
    c2 = np.full((2, 2), EPS_H + 0j)
    c2[0, 0] = EPS_P
    c3 = np.full((3, 3), EPS_H + 0j)
    c3[0, 0] = EPS_P
    rows = []
    for M in (4, 5, 6, 7):
        s = MortarStackNU(PX, PY, n_modes=M, n_orders=NORD)
        s.add_layer(DEP, eps_cell=c2, x_walls=[PX / 3.0], y_walls=[PY / 3.0])
        s.set_source(WL, theta=THETA, phi=PHI)
        onu, Rnu, Tnu = s.solve(jones=False)
        u = PMM2DStackPure(PX, PY, n_modes=M, n_orders=NORD)
        u.add_layer(DEP, eps_cell=c3)
        u.set_source(WL, theta=THETA, phi=PHI)
        ou, Ru, Tu = u.solve(jones=False)
        sc = max(float(np.max(Ru)), float(np.max(Tu)))
        d = max(float(np.abs(Rnu - Ru).max()),
                float(np.abs(Tnu - Tu).max())) / sc
        rows.append(dict(part="b1 pillar", M=M, q_nu=2 * (M - 1),
                         q_unif=3 * (M - 1), rel_gap=d,
                         clo_nu=float(abs(Rnu[1].sum() + Tnu[1].sum() - 1)),
                         clo_unif=float(abs(Ru[1].sum() + Tu[1].sum() - 1))))
        print(f"  b1 pillar  M={M}  NU(2 seg, q={2*(M-1)}) vs "
              f"uniform(3 seg, q={3*(M-1)})  rel gap {d:9.2e}   "
              f"closure {rows[-1]['clo_nu']:8.1e} / "
              f"{rows[-1]['clo_unif']:8.1e}", flush=True)

    # -- (b2) the STRIPE twin, against the exact 1-D oracle ----------------
    st2 = np.array([[EPS_P, EPS_P], [EPS_H, EPS_H]], complex)
    st3 = np.array([[EPS_P] * 3, [EPS_H] * 3, [EPS_H] * 3], complex)
    o1 = PMMStack(PX, n_superstrate=1.0, n_substrate=1.0, degree=14)
    o1.add_layer(DEP, segments=[(1 / 3, EPS_P), (2 / 3, EPS_H)])
    o1.set_source(WL, theta=THETA)
    oo, Ro, To = o1.solve()[:3]
    oo = np.asarray(oo).ravel()
    i = np.argsort(oo)
    MO, RO, TO = oo[i], Ro[1][i], To[1][i]
    o1b = PMMStack(PX, n_superstrate=1.0, n_substrate=1.0, degree=12)
    o1b.add_layer(DEP, segments=[(1 / 3, EPS_P), (2 / 3, EPS_H)])
    o1b.set_source(WL, theta=THETA)
    _o, Rb, Tb = o1b.solve()[:3]
    selfgap = float(max(np.abs(RO - Rb[1][i]).max(), np.abs(TO - Tb[1][i]).max()))
    print(f"  1-D oracle deg12-vs-deg14 self-gap {selfgap:.2e}", flush=True)
    res.setdefault("b_oracle_selfgap", selfgap)

    def score1d(orders, R, T):
        m, r, t = n0_row(orders, R, T)
        keep = np.isin(MO, m)
        return max(float(np.abs(r - RO[keep]).max()),
                   float(np.abs(t - TO[keep]).max()))

    for M in (4, 5, 6, 7, 9):
        s = MortarStackNU(PX, PY, n_modes=M, n_orders=NORD)
        s.add_layer(DEP, eps_cell=st2, x_walls=[PX / 3.0], y_walls=[PY / 3.0])
        s.set_source(WL, theta=THETA, phi=0.0)
        onu, Rnu, Tnu = s.solve(jones=False)
        enu = score1d(onu, Rnu, Tnu)
        u = PMM2DStackPure(PX, PY, n_modes=M, n_orders=NORD)
        u.add_layer(DEP, eps_cell=st3)
        u.set_source(WL, theta=THETA, phi=0.0)
        ou, Ru, Tu = u.solve(jones=False)
        eu = score1d(ou, Ru, Tu)
        rows.append(dict(part="b2 stripe", M=M, q_nu=2 * (M - 1),
                         q_unif=3 * (M - 1), err_nu=enu, err_unif=eu))
        print(f"  b2 stripe  M={M}  NU(2 seg, q={2*(M-1):2d}) err {enu:9.2e} "
              f"| uniform(3 seg, q={3*(M-1):2d}) err {eu:9.2e}   "
              f"gap {abs(enu-eu):9.2e}", flush=True)
    res["b"] = rows

# ======================================================================= (c)
if "c" in WHICH:
    print("\n=== GATE (c) -- ARBITRARY walls on their own 3x3 NU grid ===",
          flush=True)
    THETA, PHI, NORD = 0.18, 0.35, 2
    DEP = 0.22e-6
    rows = []

    # -- (c1) walls exact on an 80-pixel grid: TWO independent oracles ------
    FX = (19 / 80, 49 / 80)                # 0.2375, 0.6125
    S = 80
    pix = np.full((S, S), EPS_H + 0j)
    i0, i1 = int(round(FX[0] * S)), int(round(FX[1] * S))
    pix[i0:i1, i0:i1] = EPS_P
    tile = np.full((3, 3), EPS_H + 0j)
    tile[1, 1] = EPS_P
    prevh = None
    for (deg, no) in ((11, 7), (11, 11), (13, 11)):
        r = pmm_efficiency_2d_cell(PX, PY, pix, 1.0, 1.0, DEP, WL, degree=deg,
                                   polarization="te", theta=THETA, phi=PHI,
                                   n_orders=no)
        m, Rh, Th = n0_row(r[0], np.stack([r[1], r[1]]),
                           np.stack([r[2], r[2]]))
        keepm = np.abs(m) <= NORD
        m, Rh, Th = m[keepm], Rh[keepm], Th[keepm]
        gap = (None if prevh is None else
               float(max(np.abs(Rh - prevh[0]).max(),
                         np.abs(Th - prevh[1]).max())))
        prevh = (Rh, Th)
        HM, HR, HT = m, Rh, Th
        print(f"  hybrid PMM (80-px cell) degree={deg} n_orders={no}: "
              f"R(0,0)={Rh[m == 0][0]:.9f}  T(0,0)={Th[m == 0][0]:.9f}"
              + ("" if gap is None else f"   self-gap {gap:.2e}"), flush=True)
        rows.append(dict(part="c1 oracle-ladder", engine="hybrid_pmm",
                         degree=deg, n_orders=no, R0=float(Rh[m == 0][0]),
                         T0=float(Th[m == 0][0]), selfgap=gap))
    prevr = None
    for no in (11, 15):
        rr = rcwa_efficiency_2d(PX, PY, pix, 1.0, 1.0, DEP, WL,
                                polarization="te", theta=THETA, phi=PHI,
                                n_orders_x=no, n_orders_y=no)
        m, Rr, Tr = n0_row(rr[0], np.stack([rr[1], rr[1]]),
                           np.stack([rr[2], rr[2]]))
        keepm = np.abs(m) <= NORD
        m, Rr, Tr = m[keepm], Rr[keepm], Tr[keepm]
        gap = (None if prevr is None else
               float(max(np.abs(Rr - prevr[0]).max(),
                         np.abs(Tr - prevr[1]).max())))
        prevr = (Rr, Tr)
        print(f"  RCWA (80-px cell) n_orders={no}: R(0,0)="
              f"{Rr[m == 0][0]:.9f}  T(0,0)={Tr[m == 0][0]:.9f}"
              + ("" if gap is None else f"   self-gap {gap:.2e}"), flush=True)
        rows.append(dict(part="c1 oracle-ladder", engine="rcwa", n_orders=no,
                         R0=float(Rr[m == 0][0]), T0=float(Tr[m == 0][0]),
                         selfgap=gap))
        RCR, RCT, RCM = Rr, Tr, m
    for M in (4, 5, 6, 7):
        t0 = time.perf_counter()
        s = MortarStackNU(PX, PY, n_modes=M, n_orders=NORD)
        s.add_layer(DEP, eps_cell=tile, x_walls=[FX[0] * PX, FX[1] * PX],
                    y_walls=[FX[0] * PY, FX[1] * PY])
        s.set_source(WL, theta=THETA, phi=PHI)
        o, R, T = s.solve(jones=False)
        m, Rp, Tp = n0_row(o, R, T)
        keep = np.isin(HM, m)
        d_pmm = max(float(np.abs(Rp - HR[keep]).max()),
                    float(np.abs(Tp - HT[keep]).max()))
        keep2 = np.isin(RCM, m)
        d_rcwa = max(float(np.abs(Rp - RCR[keep2]).max()),
                     float(np.abs(Tp - RCT[keep2]).max()))
        rows.append(dict(part="c1 pure-NU", M=M, q=3 * (M - 1),
                         d_hybrid=d_pmm, d_rcwa=d_rcwa,
                         closure=float(abs(R[1].sum() + T[1].sum() - 1)),
                         t=time.perf_counter() - t0))
        print(f"  pure NU 3x3  M={M} (q={3*(M-1)})  vs hybrid {d_pmm:9.2e} "
              f"| vs RCWA {d_rcwa:9.2e}  closure "
              f"{abs(R[1].sum()+T[1].sum()-1):8.1e}  "
              f"{rows[-1]['t']:5.1f}s", flush=True)

    # -- (c2) TRULY arbitrary walls (0.2371, 0.6183): the pure NU solver's
    #         own M-ladder, plus the hybrid at the SAME exact walls.
    from lumenairy.elements.pmm.twod import (_axis_elem_counts,
                                             _pmm2d_solve_core)
    FA = (0.2371, 0.6183)
    xw = [FA[0] * PX, FA[1] * PX]
    yw = [FA[0] * PY, FA[1] * PY]
    et = np.conj(tile)
    prev = None
    for (deg, no) in ((11, 7), (11, 11), (13, 11)):
        elx = _axis_elem_counts(PX, xw, deg, 1, "probe", "x")
        ely = _axis_elem_counts(PY, yw, deg, 1, "probe", "y")
        hr = _pmm2d_solve_core(PX, PY, xw, yw, et, np.conj(1.0 + 0j),
                               np.conj(1.0 + 0j), DEP, WL, deg, elx, ely,
                               False, "te", THETA, PHI, no, "li",
                               fn_name="probe")
        m, Rh, Th = n0_row(hr[0], np.stack([hr[1], hr[1]]),
                           np.stack([hr[2], hr[2]]))
        keepm = np.abs(m) <= NORD
        m, Rh, Th = m[keepm], Rh[keepm], Th[keepm]
        gap = (None if prev is None else
               float(max(np.abs(Rh - prev[0]).max(),
                         np.abs(Th - prev[1]).max())))
        prev = (Rh, Th)
        HM2, HR2, HT2 = m, Rh, Th
        print(f"  hybrid exact-wall degree={deg} n_orders={no}: "
              f"R(0,0)={Rh[m == 0][0]:.9f}"
              + ("" if gap is None else f"   self-gap {gap:.2e}"), flush=True)
        rows.append(dict(part="c2 oracle-ladder", degree=deg, n_orders=no,
                         R0=float(Rh[m == 0][0]), T0=float(Th[m == 0][0]),
                         selfgap=gap))
    for M in (4, 5, 6, 7):
        s = MortarStackNU(PX, PY, n_modes=M, n_orders=NORD)
        s.add_layer(DEP, eps_cell=tile, x_walls=xw, y_walls=yw)
        s.set_source(WL, theta=THETA, phi=PHI)
        o, R, T = s.solve(jones=False)
        m, Rp, Tp = n0_row(o, R, T)
        keep = np.isin(HM2, m)
        d = max(float(np.abs(Rp - HR2[keep]).max()),
                float(np.abs(Tp - HT2[keep]).max()))
        rows.append(dict(part="c2 pure-NU", M=M, q=3 * (M - 1), d_hybrid=d,
                         closure=float(abs(R[1].sum() + T[1].sum() - 1))))
        print(f"  pure NU 3x3 @ (0.2371, 0.6183)  M={M}  vs hybrid {d:9.2e}"
              f"  closure {abs(R[1].sum()+T[1].sum()-1):8.1e}", flush=True)
    res["c"] = rows

# ======================================================================= (d)
if "d" in WHICH:
    print("\n=== GATE (d) -- 4-slice TAPER: NU mortar cascade vs the hybrid ===",
          flush=True)
    THETA, PHI, NORD = 0.15, 0.30, 2
    THICK = 0.32e-6
    NSL = 4
    XB0 = (0.1873 * PX, 0.7241 * PX)
    XB1 = (0.2917 * PX, 0.6109 * PX)
    YB0 = (0.2135 * PY, 0.6907 * PY)
    YB1 = (0.3061 * PY, 0.5843 * PY)
    dz = THICK / NSL
    slices = []
    for s_ in range(NSL):
        zf = 1.0 - (s_ + 0.5) / NSL
        slices.append((
            [XB0[0] + (XB1[0] - XB0[0]) * zf, XB0[1] + (XB1[1] - XB0[1]) * zf],
            [YB0[0] + (YB1[0] - YB0[0]) * zf, YB0[1] + (YB1[1] - YB0[1]) * zf]))
    print("  slice walls (x, fraction of period):",
          [f"{w[0][0]/PX:.4f}-{w[0][1]/PX:.4f}" for w in slices], flush=True)
    tile = np.full((3, 3), EPS_H + 0j)
    tile[1, 1] = EPS_P

    rows = []
    HM = HR = HT = None
    HNORD = 9
    for deg in (9, 11):
        t0 = time.perf_counter()
        h = PMM2DStackHybrid(PX, PY, degree=deg, n_orders=HNORD)
        h.add_tapered_pillar(THICK, eps_pillar=EPS_P, eps_host=EPS_H,
                             x_bounds_bottom=XB0, y_bounds_bottom=YB0,
                             x_bounds_top=XB1, y_bounds_top=YB1,
                             n_slices=NSL, rule="midpoint")
        h.set_source(WL, theta=THETA, phi=PHI)
        out = h.solve()
        oh, Rh, Th = out[0], out[1], out[2]
        th = time.perf_counter() - t0
        m, Rr, Tr = n0_row(oh, Rh, Th)
        keepm = np.abs(m) <= NORD
        m, Rr, Tr = m[keepm], Rr[keepm], Tr[keepm]
        if HM is not None:
            print(f"  hybrid degree {deg}: self-gap vs previous "
                  f"{max(np.abs(Rr-HR).max(), np.abs(Tr-HT).max()):.2e}",
                  flush=True)
        HM, HR, HT = m, Rr, Tr
        rows.append(dict(arm="hybrid", degree=deg, R0=float(Rr[m == 0][0]),
                         T0=float(Tr[m == 0][0]), t=th,
                         closure=float(abs(Rh[1].sum() + Th[1].sum() - 1))))
        print(f"  hybrid staircase degree={deg}  R(0,0)={Rr[m==0][0]:.9f}  "
              f"closure {rows[-1]['closure']:8.1e}  {th:5.1f}s", flush=True)

    for M in (4, 5, 6, 7):
        t0 = time.perf_counter()
        s = MortarStackNU(PX, PY, n_modes=M, n_orders=NORD)
        for (xw, yw) in slices:
            s.add_layer(dz, eps_cell=tile, x_walls=xw, y_walls=yw)
        s.set_source(WL, theta=THETA, phi=PHI)
        o, R, T = s.solve(jones=False)
        tm = time.perf_counter() - t0
        m, Rp, Tp = n0_row(o, R, T)
        keep = np.isin(HM, m)
        d = max(float(np.abs(Rp - HR[keep]).max()),
                float(np.abs(Tp - HT[keep]).max()))
        rows.append(dict(arm="NU mortar", M=M, q=3 * (M - 1), d_hybrid=d,
                         closure=float(abs(R[1].sum() + T[1].sum() - 1)),
                         t=tm))
        print(f"  NU mortar cascade M={M} (q={3*(M-1)}, 4 slices, 5 mortar "
              f"interfaces)  vs hybrid {d:9.2e}  closure "
              f"{abs(R[1].sum()+T[1].sum()-1):8.1e}  {tm:6.1f}s", flush=True)
    res["d"] = rows


# ==================================================================== (c3)
# The SHARP twin of gate (c): the same ARBITRARY walls made y-uniform, so the
# EXACT 1-D ``PMMStack`` (arbitrary segment widths, spectral, no Fourier floor,
# no corner) is the oracle.  (c1)/(c2) are bounded by the hybrid/RCWA Fourier
# floor on a corner-dominated 2-D pillar; this one is not bounded by anything
# but the pure solver itself.
if "c3" in WHICH:
    print("\n=== GATE (c3) -- arbitrary walls, y-uniform: EXACT 1-D oracle ===",
          flush=True)
    THETA, NORD = 0.18, 2
    DEP = 0.22e-6
    FA = (0.2371, 0.6183)
    segs = [(FA[0], EPS_H), (FA[1] - FA[0], EPS_P), (1.0 - FA[1], EPS_H)]
    tile = np.empty((3, 3), complex)
    tile[0, :] = EPS_H
    tile[1, :] = EPS_P
    tile[2, :] = EPS_H

    def orc(deg):
        s = PMMStack(PX, n_superstrate=1.0, n_substrate=1.0, degree=deg)
        s.add_layer(DEP, segments=segs)
        s.set_source(WL, theta=THETA)
        o, R, T = s.solve()[:3]
        o = np.asarray(o).ravel()
        i = np.argsort(o)
        return o[i], R[1][i], T[1][i]

    MO, RO, TO = orc(14)
    _m, R12, T12 = orc(12)
    sg = float(max(np.abs(RO - R12).max(), np.abs(TO - T12).max()))
    print(f"  1-D oracle deg12-vs-deg14 self-gap {sg:.2e}", flush=True)
    rows = [dict(part="c3 oracle", selfgap=sg)]
    for M in (4, 5, 6, 7, 9):
        t0 = time.perf_counter()
        s = MortarStackNU(PX, PY, n_modes=M, n_orders=NORD)
        s.add_layer(DEP, eps_cell=tile, x_walls=[FA[0] * PX, FA[1] * PX],
                    y_walls=[FA[0] * PY, FA[1] * PY])
        s.set_source(WL, theta=THETA, phi=0.0)
        o, R, T = s.solve(jones=False)
        m, Rp, Tp = n0_row(o, R, T)
        keep = np.isin(MO, m)
        d = max(float(np.abs(Rp - RO[keep]).max()),
                float(np.abs(Tp - TO[keep]).max()))
        clo = float(abs(R[1].sum() + T[1].sum() - 1))
        rows.append(dict(part="c3 pure-NU", M=M, q=3 * (M - 1), err=d,
                         closure=clo, t=time.perf_counter() - t0))
        print(f"  pure NU 3x3 arbitrary walls  M={M} (q={3*(M-1):2d})  "
              f"err vs EXACT 1-D {d:9.2e}  closure {clo:8.1e}  "
              f"{rows[-1]['t']:6.1f}s", flush=True)
    res["c3"] = rows

# ==================================================================== (d2)
# The SHARP twin of gate (d): the SAME 4-slice taper made y-uniform, so the
# exact 1-D ``PMMStack`` is the oracle for the whole 4-layer cascade -- every
# slice on its OWN non-uniform 3-segment grid, four mortar interfaces between
# grids that share no wall.
if "d2" in WHICH:
    print("\n=== GATE (d2) -- 4-slice STRIPE taper vs the EXACT 1-D oracle ===",
          flush=True)
    THETA, NORD = 0.15, 2
    THICK = 0.32e-6
    NSL = 4
    XB0 = (0.1873, 0.7241)
    XB1 = (0.2917, 0.6109)
    dz = THICK / NSL
    fr = []
    for s_ in range(NSL):
        zf = 1.0 - (s_ + 0.5) / NSL
        fr.append((XB0[0] + (XB1[0] - XB0[0]) * zf,
                   XB0[1] + (XB1[1] - XB0[1]) * zf))
    print("  slice x-walls (fraction of period):",
          [f"{a:.4f}-{b:.4f}" for a, b in fr], flush=True)
    tile = np.empty((3, 3), complex)
    tile[0, :] = EPS_H
    tile[1, :] = EPS_P
    tile[2, :] = EPS_H

    def orc2(deg):
        s = PMMStack(PX, n_superstrate=1.0, n_substrate=1.0, degree=deg)
        for (a, b) in fr:
            s.add_layer(dz, segments=[(a, EPS_H), (b - a, EPS_P),
                                      (1.0 - b, EPS_H)])
        s.set_source(WL, theta=THETA)
        o, R, T = s.solve()[:3]
        o = np.asarray(o).ravel()
        i = np.argsort(o)
        return o[i], R[1][i], T[1][i]

    MO, RO, TO = orc2(14)
    _m, R12, T12 = orc2(12)
    sg = float(max(np.abs(RO - R12).max(), np.abs(TO - T12).max()))
    print(f"  1-D oracle deg12-vs-deg14 self-gap {sg:.2e}", flush=True)
    rows = [dict(part="d2 oracle", selfgap=sg)]
    for M in (4, 5, 6, 7, 9):
        t0 = time.perf_counter()
        s = MortarStackNU(PX, PY, n_modes=M, n_orders=NORD)
        for (a, b) in fr:
            s.add_layer(dz, eps_cell=tile, x_walls=[a * PX, b * PX],
                        y_walls=[a * PY, b * PY])
        s.set_source(WL, theta=THETA, phi=0.0)
        o, R, T = s.solve(jones=False)
        tm = time.perf_counter() - t0
        m, Rp, Tp = n0_row(o, R, T)
        keep = np.isin(MO, m)
        d = max(float(np.abs(Rp - RO[keep]).max()),
                float(np.abs(Tp - TO[keep]).max()))
        clo = float(abs(R[1].sum() + T[1].sum() - 1))
        rows.append(dict(part="d2 NU mortar", M=M, q=3 * (M - 1), err=d,
                         closure=clo, t=tm))
        print(f"  NU mortar taper M={M} (q={3*(M-1):2d}, 4 slices, 5 "
              f"interfaces)  err vs EXACT 1-D {d:9.2e}  closure {clo:8.1e}  "
              f"{tm:6.1f}s", flush=True)
    res["d2"] = rows

json.dump(res, open(os.path.join(HERE, "f5_nonuniform.json"), "w"), indent=1)
print("\nwrote f5_nonuniform.json", flush=True)
