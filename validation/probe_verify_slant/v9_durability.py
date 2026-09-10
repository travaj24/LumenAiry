"""V9 -- DURABILITY AUDIT of ``tests/unit/test_pmm2d_staggered_slant.py``.

Every numeric bar in that file is re-measured HERE, on the file's OWN fixtures
(imported from it, so the origin audit is about the shipped test and not about
a lookalike), and printed as ``reading | bar | margin``.  A bar whose margin is
under a decade on a quantity a build is entitled to move is a durability risk
even when it passes today; a bar whose reading does not reproduce the number in
its comment is a defect of the shape ``TESTING_STANDARDS.md`` calls
right-conclusion-wrong-numbers.

Run on BOTH builds (Windows and WSL) and diff the two tables: that is the
cross-build spread the standard requires each bar to clear.
"""
import time
import warnings

import numpy as np
import scipy.linalg as sla

from _lib import arm, dump  # noqa: I001

import tests.unit.test_pmm2d_staggered_slant as TT  # noqa: E402

from lumenairy.elements.pmm import (  # noqa: E402
    PMM2DStackHybrid,
    PMM2DStackPure,
    pmm_efficiency_1d,
    pmm_efficiency_1d_slanted,
    pmm_jones_1d,
    pmm_jones_1d_slanted,
)
from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    _STAG_BLOCK_TOL,
    Granet2DTransverseE,
    _region_modes,
    _region_modes_oop,
    _slant_congruence,
    _stag_block_eig,
    _stag_parity_gauge,
    pmm_jones_2d_staggered,
)
from lumenairy.elements.rcwa._core import _select_forward_flux  # noqa: E402

warnings.simplefilter("ignore")
R = {}


def rec(name, reading, bar, sense, comment_says=None):
    """``sense`` = 'lt' (reading must be BELOW bar) or 'gt' (above)."""
    margin = (bar / reading) if sense == "lt" else (reading / bar)
    R[name] = dict(reading=float(reading), bar=float(bar), sense=sense,
                   margin=float(margin), comment_says=comment_says)
    flag = "  <-- SUB-DECADE" if margin < 10 else ""
    doc = f"  [doc {comment_says}]" if comment_says is not None else ""
    print(f"{name:52s} {reading:.4e} {sense} {bar:.1e}  margin {margin:8.1f}x"
          f"{flag}{doc}")


def main():
    # ---- B1 congruence round-trip -------------------------------------
    rng = np.random.default_rng(20260910)
    e = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    worst = 0.0
    for tx, ty in ((TT.T35, 0.0), (0.0, TT.T45), TT.DIAG35, (1.7, -0.9)):
        cov = _slant_congruence(e, tx, ty)
        A = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]])
        worst = max(worst, float(np.max(np.abs(A @ cov @ A.T - e))
                                 / np.max(np.abs(e))))
    rec("b1_congruence_roundtrip", worst, 1e-10, "lt", "<= 1e-15")

    # ---- B2 uniform null ----------------------------------------------
    worst_all, leak_all, worst_normal, leak_normal = 0.0, 0.0, 0.0, 0.0
    for tname, tv in TT._NULL_TENSORS.items():
        for theta, phi in ((0.0, 0.0), (np.deg2rad(25.0), 0.0),
                           (np.deg2rad(25.0), np.deg2rad(40.0))):
            o0, R0, T0, J0 = TT._uniform(tv, None, 5, theta, phi)
            i0 = int(np.where((o0[:, 0] == 0) & (o0[:, 1] == 0))[0][0])
            for sname, sv in TT._NULL_SLANTS.items():
                o1, R1, T1, J1 = TT._uniform(tv, sv, 5, theta, phi)
                w = max(float(np.max(np.abs(R1 - R0))),
                        float(np.max(np.abs(T1 - T0))),
                        float(np.max(np.abs(J1 - J0))))
                lk = float(max(np.max(np.abs(np.delete(R1, i0, axis=1))),
                               np.max(np.abs(np.delete(T1, i0, axis=1)))))
                worst_all = max(worst_all, w)
                leak_all = max(leak_all, lk)
                if theta == 0.0:
                    worst_normal = max(worst_normal, w)
                    leak_normal = max(leak_normal, lk)
    rec("b2_uniform_null_worst_of_60", worst_all, 1e-04, "lt", "4.098e-06")
    rec("b2_order_leak_worst_of_60", leak_all, 1e-05, "lt", "1.766e-07")
    R["b2_worst_at_normal"] = dict(reading=worst_normal,
                                   comment_says="5.29e-15 (WIN)")
    R["b2_leak_at_normal"] = dict(reading=leak_normal,
                                  comment_says="1.95e-28 (WIN)")
    print(f"{'b2_worst_at_normal':52s} {worst_normal:.4e}  [doc 5.29e-15]")
    print(f"{'b2_leak_at_normal':52s} {leak_normal:.4e}  [doc 1.95e-28]")

    lad = []
    for M in (4, 5, 6, 7, 8):
        o0, R0, T0, J0 = TT._uniform(2.25, None, M, np.deg2rad(25.0), 0.0)
        o1, R1, T1, J1 = TT._uniform(2.25, (TT.T35, 0.0), M,
                                     np.deg2rad(25.0), 0.0)
        lad.append(float(np.max(np.abs(J1 - J0))))
    R["b2_ladder"] = dict(reading=lad,
                          comment_says="1.34e-04 2.95e-06 3.93e-08 3.39e-10 "
                                       "2.03e-12")
    print(f"{'b2_ladder':52s} " + " ".join(f"{v:.3e}" for v in lad)
          + "  [doc 1.34e-04 2.95e-06 3.93e-08 3.39e-10 2.03e-12]")
    rec("b2_ladder_floor_M8", lad[-1], 1e-10, "lt", "2.03e-12")
    rec("b2_ladder_worst_step_ratio", min(a / b for a, b in
                                          zip(lad, lad[1:])), 10.0, "gt")

    # ---- B3 frame anchor ----------------------------------------------
    ship_hi, none_lo, conj_ratio_lo = 0.0, np.inf, np.inf
    for tname, tv in (("iso", 2.25), ("oop", TT.TIL)):
        for theta, phi in ((np.deg2rad(25.0), 0.0),
                           (np.deg2rad(25.0), np.deg2rad(40.0))):
            (o0, R0, T0, J0), st0 = TT._stack(
                [(TT.DEP, {"eps": complex(tv) if np.ndim(tv) == 0
                           else np.asarray(tv, dtype=complex)})], 5,
                theta=theta, phi=phi)
            Jt0 = TT._transmission_jones(st0)
            for sname, sv in (("x10", (TT.T10, 0.0)), ("x35", (TT.T35, 0.0)),
                              ("diag35", TT.DIAG35)):
                kw = {"eps": complex(tv) if np.ndim(tv) == 0
                      else np.asarray(tv, dtype=complex), "slant": sv}
                (o1, R1, T1, J1), st1 = TT._stack([(TT.DEP, kw)], 5,
                                                  theta=theta, phi=phi)
                Jt = TT._transmission_jones(st1)
                shx = -sum(L.get("slant", (0.0, 0.0))[0] * L["thickness"]
                           for L in st1._layers)
                shy = -sum(L.get("slant", (0.0, 0.0))[1] * L["thickness"]
                           for L in st1._layers)
                k0 = 2.0 * np.pi / st1._modal["wavelength"]
                p0 = st1._modal["p0"]
                ph0 = np.exp(-1j * k0 * (st1._modal["kx"][p0] * shx
                                         + st1._modal["ky"][p0] * shy))
                ship_hi = max(ship_hi, float(np.max(np.abs(Jt - Jt0))))
                nn = float(np.max(np.abs(Jt / ph0 - Jt0)))
                pp = float(np.max(np.abs(Jt / ph0 / ph0 - Jt0)))
                none_lo = min(none_lo, nn)
                conj_ratio_lo = min(conj_ratio_lo, pp / nn)
    rec("b3_shipped_worst", ship_hi, 1e-03, "lt", "2.64e-05")
    rec("b3_none_smallest", none_lo, 1e-02, "gt", "1.43e-01")
    rec("b3_conj_over_none_smallest", conj_ratio_lo, 1.5, "gt", "~2x")

    # ---- B4 dispersion -------------------------------------------------
    phys_hi, wrong_lo, abl_lo = 0.0, np.inf, np.inf
    for tname, tv, sv in TT._DISP:
        qv = TT._uniform_spectrum(tv, sv)
        p, _r = TT._root_gap(qv, tv, sv, +1.0, +1.0)
        phys_hi = max(phys_hi, p)
        for agn, sgn, lab in ((+1.0, -1.0, "a+s-"), (-1.0, +1.0, "a-s+"),
                              (-1.0, -1.0, "a-s-")):
            if tname == "iso" and lab == "a-s-":
                continue
            w, _r = TT._root_gap(qv, tv, sv, agn, sgn)
            wrong_lo = min(wrong_lo, w)
        for mode in ("noblocks", "nocong"):
            a, _r = TT._root_gap(TT._uniform_spectrum(tv, sv, mode=mode),
                                 tv, sv)
            abl_lo = min(abl_lo, a)
    rec("b4_physical_worst", phys_hi, 1e-06, "lt", "2.26e-08 (iso row)")
    rec("b4_wrong_arm_smallest", wrong_lo, 1e-03, "gt", "3.53e-02")
    rec("b4_ablation_smallest", abl_lo, 1e-04, "gt", "1.58e-03")
    dl = [TT._root_gap(TT._uniform_spectrum(TT.TIL, TT.DIAG35, M=M),
                       TT.TIL, TT.DIAG35)[0] for M in (4, 5, 6)]
    R["b4_ladder"] = dict(reading=dl,
                          comment_says="3.55e-07 6.35e-10 7.6e-13")
    print(f"{'b4_ladder':52s} " + " ".join(f"{v:.3e}" for v in dl)
          + "  [doc 3.55e-07 6.35e-10 7.6e-13]")
    rec("b4_ladder_floor_M6", dl[2], 1e-11, "lt", "7.56e-13")

    # ---- B5 stripe -----------------------------------------------------
    te_hi, tm_hi, wrong_te_lo, ratio_lo = 0.0, 0.0, np.inf, np.inf
    for phi_deg in (0.0, 10.0, 20.0, 35.0):
        for theta in (0.0, np.deg2rad(25.0)):
            if phi_deg == 0.0:
                def orc(pol, theta=theta):
                    o, Rr, Tt = pmm_efficiency_1d(
                        TT._SPX, TT._SNR, TT._SNG, TT.NSUB, TT.NSUP, TT._SDEP,
                        0.5, TT._SWL, angle=theta, polarization=pol,
                        degree=22, far_field_orders=15)
                    return {int(m): (float(Rr[i]), float(Tt[i]))
                            for i, m in enumerate(o)}
                arms = [(None, "good")]
            else:
                def orc(pol, phi_deg=phi_deg, theta=theta):
                    o, Rr, Tt = pmm_efficiency_1d_slanted(
                        TT._SPX, TT._SNR, TT._SNG, TT.NSUB, TT.NSUP, TT._SDEP,
                        0.5, TT._SWL, np.deg2rad(phi_deg), angle=theta,
                        polarization=pol, degree=22, far_field_orders=15)
                    return {int(m): (float(Rr[i]), float(Tt[i]))
                            for i, m in enumerate(o)}
                t = float(np.tan(np.deg2rad(phi_deg)))
                arms = [((t, 0.0), "good"), ((-t, 0.0), "bad")]
            got = {}
            for sv, lab in arms:
                tm, te = TT._stripe_2d(TT._SCELL, sv, theta, 7)
                got[lab] = (TT._perorder(te, orc("te")),
                            TT._perorder(tm, orc("tm")))
            te_hi = max(te_hi, got["good"][0])
            tm_hi = max(tm_hi, got["good"][1])
            if "bad" in got:
                wrong_te_lo = min(wrong_te_lo, got["bad"][0])
                ratio_lo = min(ratio_lo, got["bad"][0] / got["good"][0])
    rec("b5_TE_worst", te_hi, 1e-04, "lt", "4.28e-06")
    rec("b5_TM_worst", tm_hi, 5e-03, "lt", "1.22e-03")
    rec("b5_wrong_TE_smallest", wrong_te_lo, 1e-03, "gt", "6.23e-03")
    rec("b5_wrongratio_smallest", ratio_lo, 100.0, "gt", "4.3e+03")

    st = PMM2DStackPure(TT._SPX, TT._SPX, n_superstrate=TT.NSUP,
                        n_substrate=TT.NSUB, n_modes=7, n_orders=3)
    st.add_layer(TT._SDEP, eps_cell=TT._SCELL,
                 slant=(-float(np.tan(np.deg2rad(35.0))), 0.0))
    st.set_source(TT._SWL, theta=0.0, phi=0.0)
    o, Rr, Tt, J = st.solve()
    rec("b5_wrongsign_closure", float(np.max(np.abs(Rr.sum(1) + Tt.sum(1)
                                                    - 1.0))),
        1e-04, "lt", "3.850e-07")

    # ---- B6 slant x out-of-plane ---------------------------------------
    dex_hi, dey_hi, ratio_hi = 0.0, 0.0, 0.0
    for phi_deg in (0.0, 35.0):
        for theta in (0.0, np.deg2rad(25.0)):
            if phi_deg == 0.0:
                o, Rr, Tt, J = pmm_jones_1d(
                    TT._SPX, TT.TIL, TT.AIR, TT.NSUB, TT.NSUP, TT._SDEP, 0.5,
                    TT._SWL, angle=theta, degree=30, far_field_orders=15)
                sv = None
            else:
                o, Rr, Tt, J = pmm_jones_1d_slanted(
                    TT._SPX, TT.TIL, TT.AIR, TT.NSUB, TT.NSUP, TT._SDEP, 0.5,
                    TT._SWL, np.deg2rad(phi_deg), angle=theta, degree=30,
                    far_field_orders=15)
                sv = (float(np.tan(np.deg2rad(phi_deg))), 0.0)
            ox = {int(m): (float(Rr[0, i]), float(Tt[0, i]))
                  for i, m in enumerate(o)}
            oy = {int(m): (float(Rr[1, i]), float(Tt[1, i]))
                  for i, m in enumerate(o)}
            ex2, ey2 = TT._stripe_2d(TT._OSTRIPE, sv, theta, 7)
            dex, dey = TT._perorder(ex2, ox), TT._perorder(ey2, oy)
            dex_hi, dey_hi = max(dex_hi, dex), max(dey_hi, dey)
            if phi_deg > 0.0:
                vx2, _v = TT._stripe_2d(TT._OSTRIPE, None, theta, 7)
                ov, Rv, Tv, Jv = pmm_jones_1d(
                    TT._SPX, TT.TIL, TT.AIR, TT.NSUB, TT.NSUP, TT._SDEP, 0.5,
                    TT._SWL, angle=theta, degree=30, far_field_orders=15)
                ctrl = TT._perorder(vx2, {int(m): (float(Rv[0, i]),
                                                   float(Tv[0, i]))
                                          for i, m in enumerate(ov)})
                ratio_hi = max(ratio_hi, dex / ctrl)
    rec("b6_Ex_worst", dex_hi, 1e-03, "lt", "6.60e-05")
    rec("b6_Ey_worst", dey_hi, 1e-04, "lt", "2.32e-06")
    rec("b6_slanted_over_control", ratio_hi, 3.0, "lt")

    # ---- B7 census -----------------------------------------------------
    k0 = 2.0 * np.pi
    th, ph = np.deg2rad(20.0), np.deg2rad(35.0)
    a0x, a0y = np.sin(th) * np.cos(ph) * k0, np.sin(th) * np.sin(ph) * k0
    splits_ok, minre_lossless, minre_lossy, ratio60 = True, np.inf, np.inf, 0.0
    for cname, cell in TT._CENSUS_CELLS.items():
        maxq = {}
        for sname, sv in [("vertical", None)] + list(
                TT._CENSUS_SLANTS.items()):
            sol = Granet2DTransverseE(1.2, 1.2, 2, 2, 6, cell, alpha0x=a0x,
                                      alpha0y=a0y, k0=k0, slant=sv)
            if not sol.offplane:
                continue
            qq = sol.q * sol.q
            Lc = np.linalg.cholesky(sol.Bgen)
            Ah = sla.solve_triangular(Lc, sol.Agen, lower=True)
            Ah = sla.solve_triangular(Lc, Ah.conj().T, lower=True).conj().T
            qv, Y = np.linalg.eig(Ah)
            X = sla.solve_triangular(Lc.conj().T, Y, lower=False)
            L1 = np.linalg.cholesky(sol.Bgen[:qq, :qq]).conj().T
            L2 = np.linalg.cholesky(sol.Bgen[qq:2 * qq,
                                             qq:2 * qq]).conj().T
            Vf = np.concatenate([L1 @ X[:qq], L2 @ X[qq:2 * qq],
                                 L2 @ X[2 * qq:3 * qq], L1 @ X[3 * qq:]],
                                axis=0)
            nrm = np.linalg.norm(Vf, axis=0)
            Vf = Vf / np.where(nrm == 0.0, 1.0, nrm)[None, :]
            fidx = np.asarray(_select_forward_flux(-1j * qv, Vf, qq))
            splits_ok &= (fidx.size == 2 * qq)
            _Wf, _Vf2, lam_f, _a, _b, _c = _region_modes_oop(sol)
            mn = float(np.min(np.real(lam_f)))
            if cname == "lossy":
                minre_lossy = min(minre_lossy, mn)
            else:
                minre_lossless = min(minre_lossless, mn)
            maxq[sname] = float(np.max(np.abs(qv)))
        if "vertical" in maxq and "x60" in maxq:
            ratio60 = max(ratio60, maxq["x60"] / maxq["vertical"])
    R["b7_split_all_exact"] = dict(reading=bool(splits_ok))
    print(f"{'b7_split_all_exact':52s} {splits_ok}")
    rec("b7_lossless_minRe(lam_f)", abs(minre_lossless), 1e-12, "lt",
        "-1.71e-14")
    rec("b7_lossy_minRe(lam_f)", minre_lossy, 1e-03, "gt", "+3.11e-02")
    rec("b7_maxq_ratio_60deg", ratio60, 3.0, "lt", "1.51")

    # ---- B8 / B9 / B10 -------------------------------------------------
    g_hi, clo_hi, split_hi, floor_norm, floor_con = 0.0, 0.0, 0.0, 0.0, 0.0
    for cname, cell, sv, th2, ph2 in TT._DEPTH_CFG:
        a0 = (np.sin(th2) * np.cos(ph2) * k0, np.sin(th2) * np.sin(ph2) * k0)
        sol = Granet2DTransverseE(1.2, 1.2, 2, 2, 5, cell, alpha0x=a0[0],
                                  alpha0y=a0[1], k0=k0, slant=sv)
        _W, _V, lam_f, _a, _b, _c = _region_modes_oop(sol)
        g_hi = max(g_hi, float(np.max(np.exp(-np.real(lam_f) * k0 * 3.0))))
        for d in (0.25, 1.0, 3.0):
            (o, Rr, Tt, J), _s = TT._stack(
                [(d, {"eps_cell": cell, "slant": sv})], 5, theta=th2, phi=ph2,
                px=1.2, py=1.2, wl=1.0)
            if d == 3.0:
                clo_hi = max(clo_hi, float(np.max(np.abs(Rr.sum(1)
                                                         + Tt.sum(1) - 1.0))))
        (o1, R1, T1, J1), _s = TT._stack(
            [(0.8, {"eps_cell": cell, "slant": sv})], 5, theta=th2, phi=ph2,
            px=1.2, py=1.2, wl=1.0)
        (o2, R2, T2, J2), _s = TT._stack(
            [(0.4, {"eps_cell": cell, "slant": sv}),
             (0.4, {"eps_cell": cell, "slant": sv})], 5, theta=th2, phi=ph2,
            px=1.2, py=1.2, wl=1.0)
        split_hi = max(split_hi, float(np.max(np.abs(R1 - R2))),
                       float(np.max(np.abs(T1 - T2))),
                       float(np.max(np.abs(J1 - J2))))
        keys = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)]
        base, move = None, 0.0
        for no in (3, 5, 8):
            (o, Rr, Tt, J), _s = TT._stack(
                [(0.8, {"eps_cell": cell, "slant": sv})], 5, n_orders=no,
                theta=th2, phi=ph2, px=1.2, py=1.2, wl=1.0)
            idx = {(int(a), int(b)): i for i, (a, b) in enumerate(o)}
            v = np.concatenate([np.array([Rr[:, idx[kk]]
                                          for kk in keys]).ravel(),
                                np.array([Tt[:, idx[kk]]
                                          for kk in keys]).ravel(),
                                J.ravel().view(float)])
            if base is None:
                base = v
            else:
                move = max(move, float(np.max(np.abs(v - base))))
        if th2 == 0.0:
            floor_norm = max(floor_norm, move)
        else:
            floor_con = max(floor_con, move)
    rec("b8_forward_growth_minus_1", abs(g_hi - 1.0), 1e-09, "lt", "1.5e-13")
    rec("b8_closure_at_3lam", clo_hi, 5e-02, "lt", "1.48e-02")
    rec("b9_layer_split_worst", split_hi, 1e-12, "lt", "1.67e-15")
    rec("b10_no_floor_normal", floor_norm, 1e-12, "lt", "1.55e-15")
    rec("b10_no_floor_conical", floor_con, 1e-04, "lt", "2.758e-06")

    # ---- B11 parity ----------------------------------------------------
    dA_hi = 0.0
    for sname, sv in (("vertical", None), ("x35", (TT.T35, 0.0)),
                      ("diag35", TT.DIAG35)):
        sol = TT._solver(TT.CENTRO, 6, slant=sv)
        gf = _stag_parity_gauge(TT._GeometryShim(sol))
        perm, r = gf
        A = sol.Agen
        rr = r[:, None] * r[None, :]
        dA_hi = max(dA_hi, float(np.max(np.abs(rr * A[np.ix_(perm, perm)]
                                               + A))) / float(np.max(np.abs(A))))
        assert _stag_block_eig(A, sol.Bgen, sol.q * sol.q, gf) is not None
    rec("b11_forced_structural_residual", dA_hi, _STAG_BLOCK_TOL, "lt",
        "2.31e-15")

    def run(sl, sym):
        return pmm_jones_2d_staggered(
            TT.PX, TT.PY, TT.CENTRO, TT.NSUB, TT.NSUP, TT.DEP, TT.WL,
            degree=5, n_orders=3, theta=0.0, phi=0.0, symmetry=sym, slant=sl)
    o_a, R_a, T_a, J_a = run((TT.T35, 0.0), "auto")
    o_v, R_v, T_v, J_v = run(None, "auto")
    rec("b11_slant_visible_dJones", float(np.max(np.abs(J_a - J_v))), 1e-03,
        "gt", "2.11e-02")

    # ---- M4 ------------------------------------------------------------
    for theta, phi, mount in ((0.0, 0.0, "normal"),
                              (np.deg2rad(20.0), np.deg2rad(35.0),
                               "conical")):
        ref = TT._pure_pillar([TT._pillar3()], [(TT._PT, 0.0)], [TT._PDEP], 4,
                              theta, phi)
        vert = TT._pure_pillar([TT._pillar3()], [None], [TT._PDEP], 4, theta,
                               phi)
        rec(f"m4_{mount}_slant_effect_scale",
            float(np.max(np.abs(vert - ref))), 1e-01, "gt",
            "3.75e-01 / 4.63e-01")
        got = {}
        for sgn, lab in ((+1.0, "+t"), (-1.0, "-t")):
            got[lab] = []
            for no in (3, 5, 7):
                hs = PMM2DStackHybrid(TT._PPX, TT._PPX, n_superstrate=TT.NSUP,
                                      n_substrate=TT.NSUB, n_orders=no)
                hs.add_layer(TT._PDEP, eps_cell=TT._pillar3(),
                             slant=(sgn * TT._PT, 0.0))
                hs.set_source(TT._PWL, theta=theta, phi=phi)
                oh, Rh, Th, Jh = hs.solve()
                got[lab].append(float(np.max(np.abs(TT._pvec(oh, Rh, Th)
                                                    - ref))))
        R[f"m4_{mount}_ladder_plus"] = dict(reading=got["+t"])
        R[f"m4_{mount}_ladder_minus"] = dict(reading=got["-t"])
        print(f"m4_{mount}_ladder_plus  " + " ".join(f"{v:.3e}"
                                                     for v in got["+t"]))
        print(f"m4_{mount}_ladder_minus " + " ".join(f"{v:.3e}"
                                                     for v in got["-t"]))
        rec(f"m4_{mount}_plus_monotone_worst_step_ratio",
            min(got["+t"][i] / got["+t"][i + 1] for i in range(2)), 1.0, "gt")
        rec(f"m4_{mount}_plus_final", got["+t"][-1], 5e-02, "lt")
        rec(f"m4_{mount}_minus_final", got["-t"][-1], 1e-01, "gt")
        rec(f"m4_{mount}_minus_no_improvement",
            got["-t"][-1] / got["-t"][0], 0.8, "gt")

        base = TT._pure_pillar([TT._pillar3()], [(TT._PT, 0.0)], [TT._PDEP],
                               3, theta, phi)
        st_got = {}
        for direc, lab in ((+1, "with"), (-1, "against")):
            st_got[lab] = []
            for n in (1, 2):
                cells = [TT._pillar3(direc * kk * (2 // n)) for kk in range(n)]
                v = TT._pure_pillar(cells, [None] * n, [TT._PDEP / n] * n, 3,
                                    theta, phi)
                st_got[lab].append(float(np.max(np.abs(v - base))))
        R[f"m4c_{mount}"] = dict(with_=st_got["with"],
                                 against=st_got["against"])
        print(f"m4c_{mount} with {st_got['with'][0]:.3e} "
              f"{st_got['with'][1]:.3e} | against "
              f"{st_got['against'][0]:.3e} {st_got['against'][1]:.3e}")
        rec(f"m4c_{mount}_with_ratio", st_got["with"][1] / st_got["with"][0],
            0.5, "lt", "0.358 (normal) / 0.453 (conical)")
        rec(f"m4c_{mount}_against_ratio",
            st_got["against"][1] / st_got["against"][0], 0.8, "gt",
            "0.875 (normal) / 0.986 (conical)")

    # ---- COST ----------------------------------------------------------
    a0 = (0.25 * k0, 0.18 * k0)
    ratios = []
    for M in (5, 6):
        def _t(fn, reps=3):
            ts = []
            for _ in range(reps):
                t0 = time.perf_counter()
                fn()
                ts.append(time.perf_counter() - t0)
            return float(np.median(ts))
        to = _t(lambda M=M: _region_modes_oop(Granet2DTransverseE(
            1.2, 1.2, 2, 2, M, TT.OOPC, alpha0x=a0[0], alpha0y=a0[1], k0=k0)))
        ts_ = _t(lambda M=M: _region_modes_oop(Granet2DTransverseE(
            1.2, 1.2, 2, 2, M, TT.SCA, alpha0x=a0[0], alpha0y=a0[1], k0=k0,
            slant=(0.75, 0.0))))
        ratios.append(ts_ / to)
    R["cost_ratios"] = dict(reading=ratios, comment_says="0.96x .. 1.07x")
    print(f"{'cost_ratios':52s} " + " ".join(f"{v:.3f}" for v in ratios)
          + "  [doc 0.96 .. 1.07]")
    rec("cost_min_ratio", min(ratios), 2.0, "lt")
    _ = _region_modes(Granet2DTransverseE(1.2, 1.2, 2, 2, 6, TT.SCA,
                                          alpha0x=a0[0], alpha0y=a0[1], k0=k0))

    dump("v9_durability", R)
    subs = [k for k, v in R.items()
            if isinstance(v, dict) and v.get("margin", 1e9) < 10]
    print(f"\nSUB-DECADE BARS ({len(subs)}): {subs}")
    print("arm", arm())


if __name__ == "__main__":
    main()
