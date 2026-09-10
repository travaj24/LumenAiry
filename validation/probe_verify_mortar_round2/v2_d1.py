"""VERIFY round 2, task 3 -- D1 RE-DERIVED on my OWN fixture.

Sections (``python v2_d1.py floor allhost spectrum exponents``):

``floor``     my own y-uniform 3-layer stack, ALL-HOST middle layer on its own
              grid whose two walls sit ``delta`` apart, PATTERNED neighbours,
              scored per order against the exact 1-D ``PMMStack`` (degree 14,
              self-gap measured against degree 12).  Is the M ladder floored?
``allhost``   three ALL-HOST layers on three DIFFERENT non-uniform grids vs the
              ANALYTIC Airy slab -- the mortar's OWN algebra, no oracle needed.
``spectrum``  the free spurious-spectrum predictor
              ``|gamma|/k0 = c(M) M(M+1)/(4 k0 J)`` and its constant ``c``;
              plus the UNIFORM-lattice overlap that kills a spectral bar.
``exponents`` cond_2 of the two mortar operators, W/V and the cross masses,
              with the exponent in ``1/delta`` FITTED over the tail.

The fixture is deliberately NOT the builder's: different period, wavelength,
angle, contrast, wall positions and an OFF-CENTRE sliver.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402

import lumenairy  # noqa: E402
from lumenairy.elements.pmm import _core as _pc  # noqa: E402
from lumenairy.elements.pmm import twod_staggered as _ts  # noqa: E402
from lumenairy.elements.pmm.stack import PMMStack  # noqa: E402
from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure  # noqa: E402
from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    Granet2DTransverseE,
    StagCrossOps,
    StagGridOps,
    _region_modes,
    _stag_kron_apply,
)

HERE = os.path.dirname(os.path.abspath(__file__))
print(f"[arm] lumenairy = {lumenairy.__file__} v{lumenairy.__version__}",
      flush=True)
TAG = os.environ.get("V2_TAG", "win")
T0 = time.time()
_C = complex

# ------------------------------------------------------------- MY fixture
PER = 1.05          # um
WL = 0.71
TH, PH = 0.26, 0.0  # phi = 0 so the analytic slab arm has clean s/p rows
EPSP, EPSH = 7.5, 2.0
TT = 0.12
W0 = (0.18, 0.61)   # patterned neighbour ABOVE
W2 = (0.29, 0.74)   # patterned neighbour BELOW
YW = (0.27, 0.73)   # identical on every layer: the ONLY non-conformity is x
SLIVER_C = 0.44     # OFF-CENTRE, so no accidental symmetry
DELTAS = (3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 1e-4, 1e-5, 1e-6)
RES = {}


def _log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def _disarm():
    """Lift BOTH round-2 guards (they land on the same width, so a
    measurement of what the guards prevent has to lift both).  A no-op on the
    pre-round-2 arm, where neither exists."""
    old = (getattr(_ts, "PMM2D_STAG_MIN_SEG_GUARD", None),
           getattr(_pc, "_MORTAR_RCOND_REFUSE", None))
    if old[0] is not None:
        _ts.PMM2D_STAG_MIN_SEG_GUARD = False
    if old[1] is not None:
        _pc._MORTAR_RCOND_REFUSE = 0.0
    return old


def _rearm(old):
    if old[0] is not None:
        _ts.PMM2D_STAG_MIN_SEG_GUARD = old[0]
    if old[1] is not None:
        _pc._MORTAR_RCOND_REFUSE = old[1]


# =================================================================== ORACLE
def _oracle(deg):
    st = PMMStack(PER, degree=deg, far_field_orders=5)
    st.add_layer(TT, segments=[(W0[0], EPSH), (W0[1] - W0[0], EPSP),
                               (1.0 - W0[1], EPSH)])
    st.add_layer(TT, segments=[(1.0, EPSH)])
    st.add_layer(TT, segments=[(W2[0], EPSH), (W2[1] - W2[0], EPSP),
                               (1.0 - W2[1], EPSH)])
    st.set_source(WL, theta=TH)
    return st.solve(stabilize=None)


def _oracle_pair():
    o14, R14, T14 = _oracle(14)[:3]
    o12, R12, T12 = _oracle(12)[:3]
    R14, T14 = np.atleast_2d(R14), np.atleast_2d(T14)
    R12, T12 = np.atleast_2d(R12), np.atleast_2d(T12)
    keep = np.abs(np.asarray(o14)) <= 1
    gap = float(max(np.max(np.abs(R12[:, keep] - R14[:, keep])),
                    np.max(np.abs(T12[:, keep] - T14[:, keep]))))
    return np.asarray(o14), R14, T14, gap


def _score(o2d, R, T, o1d, R1d, T1d):
    """Worst per-order deviation over |m| <= 1, n = 0, the TE row."""
    worst = 0.0
    for m in (-1, 0, 1):
        sel = np.where((o2d[:, 0] == m) & (o2d[:, 1] == 0))[0][0]
        j = int(np.where(o1d == m)[0][0])
        worst = max(worst, abs(float(R[1, sel]) - float(R1d[1, j])),
                    abs(float(T[1, sel]) - float(T1d[1, j])))
    return worst


def _stack(delta, M, *, mortar=True, walls=None):
    yw = [YW[0] * PER, YW[1] * PER]
    if walls is None:
        sw = [(SLIVER_C - delta / 2) * PER, (SLIVER_C + delta / 2) * PER]
    else:
        sw = [walls[0] * PER, walls[1] * PER]
    st = PMM2DStackPure(PER, n_modes=M, n_orders=1, layer_grids="per-layer")
    host = np.full((3, 3), _C(EPSH))

    def _tile(w):
        c = np.full((3, 3), _C(EPSH))
        c[1, :] = _C(EPSP)
        return c

    if mortar:
        st.add_layer(TT, eps_cell=_tile(W0),
                     x_walls=[W0[0] * PER, W0[1] * PER], y_walls=yw)
        st.add_layer(TT, eps_cell=host, x_walls=sw, y_walls=yw)
        st.add_layer(TT, eps_cell=_tile(W2),
                     x_walls=[W2[0] * PER, W2[1] * PER], y_walls=yw)
    else:
        # SAME device, every layer on the SLIVER's grid -> plain square match
        bx = [0.0] + list(sw) + [PER]
        for w in (W0, None, W2):
            if w is None:
                cell = host
            else:
                cell = np.zeros((3, 3), dtype=_C)
                for i in range(3):
                    mid = 0.5 * (bx[i] + bx[i + 1]) / PER
                    cell[i, :] = EPSP if w[0] < mid < w[1] else EPSH
            st.add_layer(TT, eps_cell=cell, x_walls=sw, y_walls=yw)
    st.set_source(WL, theta=TH, phi=PH)
    return st


def _solve(st):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        o, R, T = st.solve(jones=False)
    return (np.asarray(o), np.atleast_2d(R), np.atleast_2d(T),
            [f"{x.category.__name__}: {str(x.message)[:80]}" for x in w])


def _closure(R, T):
    return float(np.max(np.abs(R.sum(axis=1) + T.sum(axis=1) - 1.0)))


# ================================================================== floor
def sec_floor():
    o1d, R1d, T1d, gap = _oracle_pair()
    _log(f"oracle degree-14, its own 12->14 self-gap = {gap:.3e}")
    out = {"oracle_selfgap": gap, "fixture": dict(
        per=PER, wl=WL, theta=TH, epsp=EPSP, epsh=EPSH, t=TT,
        w0=W0, w2=W2, yw=YW, sliver_centre=SLIVER_C)}
    old = _disarm()
    try:
        grid = {}
        for delta in DELTAS:
            rec = {}
            for M in (4, 5, 6, 7, 8):
                t0 = time.time()
                try:
                    o, R, T, w = _solve(_stack(delta, M))
                    rec[str(M)] = {
                        "err": _score(o, R, T, o1d, R1d, T1d),
                        "closure": _closure(R, T),
                        "warnings": w, "wall": time.time() - t0}
                except Exception as exc:                    # noqa: BLE001
                    rec[str(M)] = {"REFUSED":
                                   f"{type(exc).__name__}: {str(exc)[:110]}",
                                   "wall": time.time() - t0}
            grid[f"{delta:g}"] = rec
            e = [rec[str(M)].get("err") for M in (4, 5, 6, 7, 8)]
            c = [rec[str(M)].get("closure") for M in (4, 5, 6, 7, 8)]
            nw = sum(len(rec[str(M)].get("warnings", [])) for M in
                     (4, 5, 6, 7, 8))
            _log(f"delta={delta:.0e}  err M=4..8 "
                 + " ".join("--" if x is None else f"{x:.3e}" for x in e)
                 + f"  | 7->8 {(e[3] / e[4] if e[3] and e[4] else float('nan')):.2f}x"
                 + "  clo " + " ".join("--" if x is None else f"{x:.2e}"
                                       for x in c)
                 + f"  warn {nw}")
        out["grid"] = grid
        # ORDINARY-wall control: the SAME layer count, ordinary partitions
        ctrl = {}
        for lo, hi in ((0.25, 0.75), (0.35, 0.65), (0.40, 0.60),
                       (0.44, 0.56), (0.46, 0.54)):
            rec = {}
            for M in (4, 5, 6, 7, 8):
                o, R, T, w = _solve(_stack(None, M, walls=(lo, hi)))
                rec[str(M)] = {"err": _score(o, R, T, o1d, R1d, T1d),
                               "closure": _closure(R, T)}
            ctrl[f"{hi - lo:.2f}"] = rec
            _log(f"ORDINARY partition width {hi - lo:.2f}: err M=4..8 "
                 + " ".join(f"{rec[str(M)]['err']:.3e}"
                            for M in (4, 5, 6, 7, 8)))
        out["ordinary_control"] = ctrl
        # NO-MORTAR attribution control at the extreme
        nm = {}
        for delta in (3e-1, 1e-3, 1e-6):
            rec = {}
            for M in (4, 5, 6):
                try:
                    o, R, T, w = _solve(_stack(delta, M, mortar=False))
                    rec[str(M)] = {"err": _score(o, R, T, o1d, R1d, T1d),
                                   "closure": _closure(R, T)}
                except Exception as exc:                    # noqa: BLE001
                    rec[str(M)] = {"REFUSED": f"{type(exc).__name__}"}
            nm[f"{delta:g}"] = rec
            _log(f"NO-MORTAR delta={delta:.0e}: err M=4..6 "
                 + " ".join(str(rec[str(M)].get("err")) for M in (4, 5, 6)))
        out["no_mortar"] = nm
    finally:
        _rearm(old)
    RES["floor"] = out


# ================================================================ allhost
def _airy_slab(n1, n2, t, wl, theta):
    """Symmetric-slab reflectance, s and p, from the interface coefficient
    alone (no t = 1 + r convention anywhere)."""
    k0 = 2.0 * np.pi / wl
    c1 = np.cos(theta)
    s1 = np.sin(theta)
    c2 = np.sqrt(1.0 - (n1 * s1 / n2) ** 2 + 0j)
    rs = (n1 * c1 - n2 * c2) / (n1 * c1 + n2 * c2)
    rp = (n2 * c1 - n1 * c2) / (n2 * c1 + n1 * c2)
    ph = np.exp(2j * k0 * n2 * c2 * t)
    out = {}
    for k, r in (("s", rs), ("p", rp)):
        R = r * (1.0 - ph) / (1.0 - r * r * ph)
        out[k] = float(abs(R) ** 2)
    return out


def sec_allhost():
    """THREE all-host layers on THREE different non-uniform grids.  The device
    is a homogeneous slab of thickness 3 TT: analytic."""
    ref = _airy_slab(1.0, np.sqrt(EPSH), 3.0 * TT, WL, TH)
    _log(f"analytic slab R_s = {ref['s']:.12f}  R_p = {ref['p']:.12f}")
    out = {"analytic": ref}
    old = _disarm()
    try:
        grid = {}
        for delta in DELTAS:
            rec = {}
            for M in (4, 5, 6, 7):
                yw = [YW[0] * PER, YW[1] * PER]
                sw = [(SLIVER_C - delta / 2) * PER,
                      (SLIVER_C + delta / 2) * PER]
                st = PMM2DStackPure(PER, n_modes=M, n_orders=1,
                                    layer_grids="per-layer")
                host = np.full((3, 3), _C(EPSH))
                st.add_layer(TT, eps_cell=host,
                             x_walls=[W0[0] * PER, W0[1] * PER], y_walls=yw)
                st.add_layer(TT, eps_cell=host, x_walls=sw, y_walls=yw)
                st.add_layer(TT, eps_cell=host,
                             x_walls=[W2[0] * PER, W2[1] * PER], y_walls=yw)
                st.set_source(WL, theta=TH, phi=PH)
                try:
                    o, R, T, w = _solve(st)
                    z = np.where((o[:, 0] == 0) & (o[:, 1] == 0))[0][0]
                    rec[str(M)] = {
                        "err_s": abs(float(R[1, z]) - ref["s"]),
                        "err_p": abs(float(R[0, z]) - ref["p"]),
                        "closure": _closure(R, T), "warnings": w}
                except Exception as exc:                    # noqa: BLE001
                    rec[str(M)] = {"REFUSED": f"{type(exc).__name__}"}
            grid[f"{delta:g}"] = rec
            _log(f"ALLHOST delta={delta:.0e}: err_s M=4..7 "
                 + " ".join(f"{rec[str(M)].get('err_s', float('nan')):.3e}"
                            for M in (4, 5, 6, 7))
                 + "  clo "
                 + f"{rec['7'].get('closure', float('nan')):.1e}")
        out["grid"] = grid
        # SHARED-grid cross-check of the analytic formula itself
        st = PMM2DStackPure(PER, n_modes=5, n_orders=1)
        for _ in range(3):
            st.add_layer(TT, eps=EPSH)
        st.set_source(WL, theta=TH, phi=PH)
        o, R, T, w = _solve(st)
        z = np.where((o[:, 0] == 0) & (o[:, 1] == 0))[0][0]
        out["analytic_selfcheck"] = {
            "shared_uniform_err_s": abs(float(R[1, z]) - ref["s"]),
            "shared_uniform_err_p": abs(float(R[0, z]) - ref["p"])}
        _log(f"analytic self-check on the SHARED uniform path: "
             f"err_s {out['analytic_selfcheck']['shared_uniform_err_s']:.3e} "
             f"err_p {out['analytic_selfcheck']['shared_uniform_err_p']:.3e}")
    finally:
        _rearm(old)
    RES["allhost"] = out


# =============================================================== spectrum
def _sliver_spectrum(delta, M, walls=None, N_uniform=None):
    k0 = 2.0 * np.pi / WL
    kx0 = k0 * np.sin(TH) * np.cos(PH)
    ky0 = k0 * np.sin(TH) * np.sin(PH)
    if N_uniform is not None:
        wb = np.linspace(0.0, PER, N_uniform + 1)
        cell = np.full((N_uniform, N_uniform), _C(EPSH))
    else:
        if walls is None:
            walls = (SLIVER_C - delta / 2, SLIVER_C + delta / 2)
        wb = np.array([0.0, walls[0] * PER, walls[1] * PER, PER])
        cell = np.full((3, 3), _C(EPSH))
    sb = Granet2DTransverseE(PER, PER, wb, wb, M, cell,
                             alpha0x=kx0, alpha0y=ky0, k0=k0)
    _W, _V, lam, *_ = _region_modes(sb)
    J = 0.5 * float(np.min(np.diff(wb)))
    # _region_modes' eigenvalues are ALREADY normalized by k0 -- cross-checked
    # against the builder's uniform-N=5 reading (9.84 at M=4 on a 1.2 um
    # period; this fixture's 0.21 um segments read 9.60), so ``lam`` IS
    # |gamma|/k0 and must NOT be divided again.
    gmax = float(np.max(np.abs(lam)))
    return dict(k0J=k0 * J, Jmin_over_d=2 * J / PER, gamma_max=gmax,
                gamma_over_k0=gmax,
                c=gmax * (4.0 * k0 * J) / (M * (M + 1)))


def sec_spectrum():
    old = _disarm()
    out = {"sliver": {}, "uniform": {}}
    try:
        for delta in DELTAS:
            for M in (4, 6):
                r = _sliver_spectrum(delta, M)
                out["sliver"][f"d{delta:g}_M{M}"] = r
            a = out["sliver"][f"d{delta:g}_M4"]
            b = out["sliver"][f"d{delta:g}_M6"]
            _log(f"SPECTRUM delta={delta:.0e}: k0J={a['k0J']:.3e} "
                 f"|g|/k0 M4 {a['gamma_over_k0']:.4e} c={a['c']:.4f} | "
                 f"M6 {b['gamma_over_k0']:.4e} c={b['c']:.4f}")
        for N in (3, 5, 8):
            r = _sliver_spectrum(None, 4, N_uniform=N)
            out["uniform"][f"N{N}_M4"] = r
            _log(f"UNIFORM N={N} (segments {1.0 / N:.3f} d): |g|/k0 M4 "
                 f"{r['gamma_over_k0']:.3f}  c={r['c']:.4f}")
        # the physical ceiling for comparison, and the OVERLAP argument
        out["physical_ceiling_n_max"] = float(np.sqrt(max(EPSP, EPSH)))
        # CROSS-FIXTURE: is c a universal constant, or fixture-dependent?
        # Re-measured on the BUILDER's own fixture 2 numbers
        # (per 1.2, wl 0.85, theta 0.15, phi 0.35, eps_h 2.25).
        out["cross_fixture"] = {}
        for (per, wl, th, ph, eh, tagf) in (
                (1.2, 0.85, 0.15, 0.35, 2.25, "builder_alt"),
                (0.9, 0.60, 0.20, 0.00, 2.25, "builder_r5"),
                (PER, WL, TH, PH, EPSH, "mine")):
            k0 = 2.0 * np.pi / wl
            kx0 = k0 * np.sin(th) * np.cos(ph)
            ky0 = k0 * np.sin(th) * np.sin(ph)
            for delta in (1e-3, 1e-5):
                for M in (4, 6):
                    wb = np.array([0.0, (0.5 - delta / 2) * per,
                                   (0.5 + delta / 2) * per, per])
                    cell = np.full((3, 3), _C(eh))
                    sb = Granet2DTransverseE(per, per, wb, wb, M, cell,
                                             alpha0x=kx0, alpha0y=ky0, k0=k0)
                    _W, _V, lam, *_ = _region_modes(sb)
                    J = 0.5 * float(np.min(np.diff(wb)))
                    g = float(np.max(np.abs(lam)))
                    out["cross_fixture"][f"{tagf}_d{delta:g}_M{M}"] = dict(
                        k0J=k0 * J, gamma_max=g, gamma_over_k0=g,
                        c=g * (4.0 * k0 * J) / (M * (M + 1)))
            a = out["cross_fixture"][f"{tagf}_d1e-05_M4"]
            b = out["cross_fixture"][f"{tagf}_d1e-05_M6"]
            _log(f"CROSS-FIXTURE {tagf}: c(M=4) = {a['c']:.4f}  "
                 f"c(M=6) = {b['c']:.4f}   (|g|/k0 {a['gamma_over_k0']:.4e})")
    finally:
        _rearm(old)
    RES["spectrum"] = out


# ============================================================== exponents
def _mortar_ops(delta, M):
    k0 = 2.0 * np.pi / WL
    kx0 = k0 * np.sin(TH) * np.cos(PH)
    ky0 = k0 * np.sin(TH) * np.sin(PH)
    taux, tauy = np.exp(-1j * kx0 * PER), np.exp(-1j * ky0 * PER)
    wa = np.array([0.0, W0[0] * PER, W0[1] * PER, PER])
    wb = np.array([0.0, (SLIVER_C - delta / 2) * PER,
                   (SLIVER_C + delta / 2) * PER, PER])
    yw = np.array([0.0, YW[0] * PER, YW[1] * PER, PER])
    ga = StagGridOps(PER, PER, wa, yw, M, taux, tauy)
    gb = StagGridOps(PER, PER, wb, yw, M, taux, tauy)
    cr = StagCrossOps(ga, gb)
    tl = np.full((3, 3), _C(EPSH))
    tl[1, :] = _C(EPSP)
    host = np.full((3, 3), _C(EPSH))
    sa = Granet2DTransverseE(PER, PER, wa, yw, M, tl,
                             alpha0x=kx0, alpha0y=ky0, k0=k0)
    sb = Granet2DTransverseE(PER, PER, wb, yw, M, host,
                             alpha0x=kx0, alpha0y=ky0, k0=k0)
    Wa, Va, lam_a, _ = _region_modes(sa)
    Wb, Vb, lam_b, _ = _region_modes(sb)
    lhsE = _pc._stag_blk2_apply(gb.V1, gb.V2, Wb, gb.qq, _stag_kron_apply)
    hb_a, _ = _pc._stag_h_blocks(ga, cr)
    lhsH_A = _pc._stag_blk2_apply(hb_a[0], hb_a[1], Va, ga.qq,
                                 _stag_kron_apply)
    hb_b, _ = _pc._stag_h_blocks(gb, cr)
    lhsH_B = _pc._stag_blk2_apply(hb_b[0], hb_b[1], Vb, gb.qq,
                                 _stag_kron_apply)

    def c2(A):
        try:
            return float(np.linalg.cond(np.asarray(A)))
        except Exception:                                # noqa: BLE001
            return float("inf")

    def rc(A):
        """LAPACK reciprocal 1-condition, the shipped screen's instrument."""
        A = np.asarray(A)
        try:
            import scipy.linalg as sla
            lu, piv = sla.lu_factor(A)
            anorm = float(np.max(np.sum(np.abs(A), axis=0)))
            gecon = sla.get_lapack_funcs("gecon", (A,))
            v, info = gecon(lu, anorm)
            return float(v) if int(info) == 0 else 0.0
        except Exception:                                # noqa: BLE001
            return 0.0

    return dict(
        delta=delta, M=M, n=int(np.asarray(lhsE).shape[0]),
        k0_Jmin=float(k0 * 0.5 * np.min(np.diff(wb))),
        cond_lhsE_B=c2(lhsE), cond_lhsH_A=c2(lhsH_A), cond_lhsH_B=c2(lhsH_B),
        rcond_lhsE_B=rc(lhsE), rcond_lhsH_A=rc(lhsH_A),
        rcond_lhsH_B=rc(lhsH_B),
        cond_Wb=c2(Wb), cond_Vb=c2(Vb), cond_Wa=c2(Wa), cond_Va=c2(Va),
        cond_C1x=c2(cr.C1[1]), cond_C2x=c2(cr.C2[1]),
        lam_max_sliver=float(np.max(np.abs(lam_b))),
        lam_max_plain=float(np.max(np.abs(lam_a))))


def sec_exponents():
    old = _disarm()
    rows = {}
    try:
        for delta in DELTAS:
            for M in (4, 6):
                rows[f"d{delta:g}_M{M}"] = _mortar_ops(delta, M)
            r = rows[f"d{delta:g}_M6"]
            _log(f"COND delta={delta:.0e} M=6: lhsE {r['cond_lhsE_B']:.3e} "
                 f"(rcond {r['rcond_lhsE_B']:.3e})  lhsH_A "
                 f"{r['cond_lhsH_A']:.3e} (rcond {r['rcond_lhsH_A']:.3e})  "
                 f"Wb {r['cond_Wb']:.3e} Vb {r['cond_Vb']:.3e} "
                 f"C1x {r['cond_C1x']:.3e} C2x {r['cond_C2x']:.3e}")
        fits = {}
        tail = [d for d in DELTAS if d <= 1e-2]
        for key in ("cond_lhsE_B", "cond_lhsH_A", "cond_lhsH_B", "cond_Wb",
                    "cond_Vb", "cond_C1x", "cond_C2x", "lam_max_sliver"):
            for M in (4, 6):
                y = np.array([rows[f"d{d:g}_M{M}"][key] for d in tail])
                x = np.array([1.0 / d for d in tail])
                g = np.isfinite(y) & (y > 0)
                if g.sum() >= 3:
                    fits[f"{key}_M{M}"] = float(
                        np.polyfit(np.log(x[g]), np.log(y[g]), 1)[0])
        _log("FITTED exponents in 1/delta (tail delta <= 1e-2): "
             + json.dumps({k: round(v, 3) for k, v in fits.items()}))
        rows["fits"] = fits
    finally:
        _rearm(old)
    RES["exponents"] = rows


SECTIONS = {"floor": sec_floor, "allhost": sec_allhost,
            "spectrum": sec_spectrum, "exponents": sec_exponents}

if __name__ == "__main__":
    want = sys.argv[1:] or list(SECTIONS)
    for s in want:
        _log(f"=== section {s} ===")
        SECTIONS[s]()
    p = os.path.join(HERE, f"v2_d1_{TAG}.json")
    old = {}
    if os.path.exists(p):
        try:
            old = json.load(open(p))
        except Exception:                                # noqa: BLE001
            old = {}
    old.update(RES)
    old["_lumenairy"] = lumenairy.__file__
    old["_numpy"] = np.__version__
    with open(p, "w") as fh:
        json.dump(old, fh, indent=1, default=str)
    _log(f"wrote {p}")
