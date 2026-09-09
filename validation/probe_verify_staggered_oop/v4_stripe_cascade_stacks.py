"""V4 -- the remaining build tables re-measured on FRESH fixtures:

  T5  a y-uniform out-of-plane STRIPE per order vs the two 1-D engines, with
      the y-momentum leak and the closure ladder;
  T7  cascade stability and closure vs DEPTH (0.25 / 1 / 3 wavelengths), the
      forward/backward split count -- INCLUDING the count BEFORE
      ``_select_forward_flux``'s defensive rebalance, which the shipped
      ``2 q^2 / 2 q^2`` guard cannot see -- and the max forward growth factor;
  T8a one 0.4-lam out-of-plane layer == two 0.2-lam layers;
  T8b a uniform out-of-plane MULTILAYER vs the Berreman multilayer;
  T8c ``retain_internal`` / ``layer_absorption`` budget on the generalized
      cascade, lossless and lossy.

Usage:  PYTHONPATH=<root> python v4_stripe_cascade_stacks.py <root> <out.json>
"""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

import lumenairy  # noqa: E402

ROOT = os.path.abspath(sys.argv[1])
assert os.path.abspath(lumenairy.__file__).startswith(ROOT), (
    f"lumenairy.__file__ = {lumenairy.__file__} is not under {ROOT}")

from lumenairy.elements.berreman import berreman_jones_1d  # noqa: E402
from lumenairy.elements.pmm import (
    PMM2DStackPure,  # noqa: E402
    pmm_jones_1d_segments,  # noqa: E402
)
from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    Granet2DTransverseE,
    _region_modes_oop,
    pmm_jones_2d_staggered,
)
from lumenairy.elements.rcwa import (  # noqa: E402
    rcwa_jones_1d_segments,
    uniaxial_tensor,
)

OUT = sys.argv[2]
WL = 0.64e-6
NSUB = 1.50
NSUP = 1.0

T_OOP = uniaxial_tensor(1.49, 1.75, 0.64, phi=0.28)
T_OOP2 = uniaxial_tensor(1.55, 1.66, 1.05, phi=1.90)
T_NR = T_OOP.copy()
T_NR[0, 2] = T_OOP[0, 2] + 0.19j
T_NR[2, 0] = np.conj(T_NR[0, 2])
T_LOSSY = uniaxial_tensor(1.51 + 0.06j, 1.77 + 0.06j, 0.80, phi=0.35)
ISO2 = 2.00 * np.eye(3, dtype=complex)

out = {"root": ROOT, "lumenairy": lumenairy.__file__,
       "T5": [], "T7": [], "T8a": [], "T8b": [], "T8c": []}


def save():
    json.dump(out, open(OUT, "w"), indent=1)


def align_m0(o2, A2, o1, A1):
    o2 = np.asarray(o2)
    sel = np.where(o2[:, 1] == 0)[0]
    m1 = {int(m): j for j, m in enumerate(np.asarray(o1))}
    d, n = 0.0, 0
    for j in sel:
        k = m1.get(int(o2[j, 0]))
        if k is not None:
            d = max(d, float(np.max(np.abs(np.asarray(A2)[:, j]
                                           - np.asarray(A1)[:, k]))))
            n += 1
    return d, n


# ============================ T5: the STRIPE ==============================
PX5 = 1.18e-6
DEP5 = 0.37e-6
SEGS = [(0.5, T_OOP), (0.5, ISO2)]
CELL5 = np.zeros((2, 2, 3, 3), dtype=complex)
CELL5[0, :] = T_OOP
CELL5[1, :] = ISO2

for mtag, th in (("normal", 0.0), ("oblique25", np.deg2rad(25.0))):
    op = pmm_jones_1d_segments(PX5, SEGS, NSUB, NSUP, DEP5, WL, angle=th,
                               degree=18, far_field_orders=31, stabilize=False)
    orc = rcwa_jones_1d_segments(PX5, SEGS, NSUB, NSUP, DEP5, WL, angle=th,
                                 n_orders=41)
    m1 = {int(m): j for j, m in enumerate(np.asarray(op[0]))}
    m2 = {int(m): j for j, m in enumerate(np.asarray(orc[0]))}
    common = sorted(set(m1) & set(m2))
    sp = {
        "dR": max(float(np.max(np.abs(op[1][:, m1[m]] - orc[1][:, m2[m]])))
                  for m in common),
        "dT": max(float(np.max(np.abs(op[2][:, m1[m]] - orc[2][:, m2[m]])))
                  for m in common),
        "dJones": float(np.max(np.abs(np.asarray(op[3]) - np.asarray(orc[3])))),
    }
    row = {"mount": mtag, "oracle_spread": sp, "ladder": []}
    print(f"[T5 {mtag}] ORACLE SPREAD dR={sp['dR']:.3e} dT={sp['dT']:.3e} "
          f"dJ={sp['dJones']:.3e}", flush=True)
    for M in (5, 6, 7, 8):
        o, R, T, J = pmm_jones_2d_staggered(PX5, PX5, CELL5, NSUB, NSUP, DEP5,
                                            WL, degree=M, n_orders=3, theta=th)
        oo = np.asarray(o)
        nz = np.where(oo[:, 1] != 0)[0]
        dR, n = align_m0(o, R, orc[0], orc[1])
        dT, _ = align_m0(o, T, orc[0], orc[2])
        dJ = float(np.max(np.abs(np.asarray(J) - np.asarray(orc[3]))))
        row["ladder"].append({
            "M": M, "dim": 4 * (2 * (M - 1)) ** 2, "dR": dR, "dT": dT,
            "dJones": dJ,
            "yleakR": float(np.max(np.abs(R[:, nz]))),
            "yleakT": float(np.max(np.abs(T[:, nz]))),
            "closure": float(np.max(np.abs(R.sum(axis=1) + T.sum(axis=1) - 1)))})
        d = row["ladder"][-1]
        print(f"  M={M} dim={d['dim']:5d} dR={dR:.3e} dT={dT:.3e} dJ={dJ:.3e} "
              f"yleak={d['yleakR']:.1e}/{d['yleakT']:.1e} "
              f"|R+T-1|={d['closure']:.2e}", flush=True)
    out["T5"].append(row)
    save()


# ==================== T7: cascade stability vs DEPTH =======================
def raw_split_count(lam, Vfull, N):
    """INDEPENDENT re-implementation of the forward classification rule that
    ``rcwa._core._select_forward_flux`` applies BEFORE its defensive rebalance,
    so the probe can see whether the rebalance had to engage (the shipped
    ``2 q^2 / 2 q^2`` guard is downstream of it and cannot)."""
    gre = np.real(lam)
    Ex, Ey = Vfull[:N], Vfull[N:2 * N]
    Hx, Hy = Vfull[2 * N:3 * N] / 1j, Vfull[3 * N:4 * N] / 1j
    Sz = np.real(np.sum(Ex * np.conj(Hy) - Ey * np.conj(Hx), axis=0))
    mx = max(1.0, float(np.max(np.abs(Sz))))
    carries = np.abs(Sz) > 1e-9 * mx
    carries &= ~((np.abs(Sz) < 3e-3 * mx) & (np.abs(gre) > 0.1))
    carries &= ~(np.abs(gre) > 0.5)
    fwd = np.where(carries, Sz > 0, gre > 0)
    return int(np.count_nonzero(fwd))


def modes_for(cell, px, th, ph, M):
    k0 = 2.0 * np.pi / WL
    kx0 = np.real(NSUP) * np.sin(th) * np.cos(ph)
    ky0 = np.real(NSUP) * np.sin(th) * np.sin(ph)
    s = Granet2DTransverseE(px, px, cell.shape[0], cell.shape[1], M, cell,
                            alpha0x=kx0 * k0, alpha0y=ky0 * k0, k0=k0)
    Wf, Vf, lf, Wb, Vb, lb = _region_modes_oop(s)
    qq = s.q * s.q
    # rebuild the whitened block vector exactly as the library does
    import scipy.linalg as _sla
    Lc = np.linalg.cholesky(s.Bgen)
    Ah = _sla.solve_triangular(Lc, s.Agen, lower=True)
    Ah = _sla.solve_triangular(Lc, Ah.conj().T, lower=True).conj().T
    qv, Y = np.linalg.eig(Ah)
    X = _sla.solve_triangular(Lc.conj().T, Y, lower=False)
    L1 = np.linalg.cholesky(s.Bgen[:qq, :qq]).conj().T
    L2 = np.linalg.cholesky(s.Bgen[qq:2 * qq, qq:2 * qq]).conj().T
    Vfull = np.concatenate([L1 @ X[:qq], L2 @ X[qq:2 * qq],
                            L2 @ X[2 * qq:3 * qq], L1 @ X[3 * qq:]], axis=0)
    nrm = np.linalg.norm(Vfull, axis=0)
    Vfull = Vfull / np.where(nrm == 0.0, 1.0, nrm)[None, :]
    raw = raw_split_count(-1j * qv, Vfull, qq)
    return lf, lb, qq, raw, 4 * qq


PX7 = 1.22e-6
CELL_U = np.zeros((2, 2, 3, 3), dtype=complex)
CELL_U[:] = T_OOP
CELL_P = np.zeros((2, 2, 3, 3), dtype=complex)
CELL_P[:] = T_OOP
CELL_P[0, 0] = ISO2
CELL_L = np.zeros((2, 2, 3, 3), dtype=complex)
CELL_L[:] = T_OOP
CELL_L[0, 0] = T_LOSSY

for cname, cell, herm in (("uniform_hermitian", CELL_U, True),
                          ("pillar_hermitian", CELL_P, True),
                          ("pillar_lossy", CELL_L, False)):
    for mtag, th, ph in (("normal", 0.0, 0.0),
                         ("conical25_40", np.deg2rad(25.0),
                          np.deg2rad(40.0))):
        lf, lb, qq, raw, ntot = modes_for(cell, PX7, th, ph, 7)
        row = {"cell": cname, "mount": mtag, "hermitian": herm,
               "split": [2 * qq, ntot - 2 * qq],
               "raw_forward_count_before_rebalance": raw,
               "rebalance_engaged": bool(raw != 2 * qq), "depths": []}
        k0 = 2.0 * np.pi / WL
        for dl in (0.25, 1.0, 3.0):
            dep = dl * WL
            o, R, T, J = pmm_jones_2d_staggered(
                PX7, PX7, cell, NSUB, NSUP, dep, WL, degree=7, n_orders=3,
                theta=th, phi=ph)
            clo = float(np.max(np.abs(R.sum(axis=1) + T.sum(axis=1) - 1)))
            grow = float(np.max(np.exp(-np.real(lf) * k0 * dep)))
            growb = float(np.max(np.exp(np.real(lb) * k0 * dep)))
            row["depths"].append({
                "depth_lam": dl, "closure": clo, "max_fwd_growth": grow,
                "max_bwd_growth": growb,
                "absorbed": float(np.max(1.0 - R.sum(axis=1) - T.sum(axis=1)))})
        out["T7"].append(row)
        d = row["depths"]
        print(f"[T7 {cname} {mtag}] split={row['split']} raw_fwd={raw} "
              f"(rebalanced={row['rebalance_engaged']})  closure "
              f"{d[0]['closure']:.2e}/{d[1]['closure']:.2e}/"
              f"{d[2]['closure']:.2e}  max_fwd_growth={d[2]['max_fwd_growth']:.4e}"
              f"  max_bwd_growth={d[2]['max_bwd_growth']:.4e}", flush=True)
        save()


# ============================ T8a / T8b / T8c =============================
def stack(layers, th, ph, M, n_orders=3, retain=False, px=1.05e-6):
    st = PMM2DStackPure(px, px, n_superstrate=NSUP, n_substrate=NSUB,
                        n_modes=M, n_orders=n_orders)
    for t, e in layers:
        e = np.asarray(e)
        if e.ndim == 2:
            st.add_layer(t, eps=e)
        else:
            st.add_layer(t, eps_cell=e)
    st.set_source(WL, theta=th, phi=ph)
    return st, st.solve(jones=True, retain_internal=retain)


for mtag, th, ph in (("normal", 0.0, 0.0),
                     ("conical20_35", np.deg2rad(20.0), np.deg2rad(35.0))):
    _s1, a = stack([(0.4 * WL, CELL_P)], th, ph, 6)
    _s2, b = stack([(0.2 * WL, CELL_P), (0.2 * WL, CELL_P)], th, ph, 6)
    row = {"mount": mtag,
           "dR": float(np.max(np.abs(a[1] - b[1]))),
           "dT": float(np.max(np.abs(a[2] - b[2]))),
           "dJones": float(np.max(np.abs(np.asarray(a[3])
                                         - np.asarray(b[3]))))}
    out["T8a"].append(row)
    print(f"[T8a {mtag}] 1x0.4 vs 2x0.2  dR={row['dR']:.3e} "
          f"dT={row['dT']:.3e} dJ={row['dJones']:.3e}", flush=True)
    save()

ML = [(0.21e-6, T_OOP), (0.14e-6, T_NR), (0.17e-6, T_LOSSY)]
for mtag, th, ph in (("oblique25", np.deg2rad(25.0), 0.0),
                     ("conical25_40", np.deg2rad(25.0), np.deg2rad(40.0))):
    Rb, Tb, Jrb, _ = berreman_jones_1d([(e, t) for t, e in ML], NSUB, NSUP, WL,
                                       theta=th, phi=ph)
    for M in (5, 6, 7, 8):
        _s, (o, R, T, J) = stack(ML, th, ph, M)
        oo = np.asarray(o)
        k = int(np.where((oo[:, 0] == 0) & (oo[:, 1] == 0))[0][0])
        row = {"mount": mtag, "M": M,
               "dR": float(np.max(np.abs(R[:, k] - Rb))),
               "dT": float(np.max(np.abs(T[:, k] - Tb))),
               "dJones": float(np.max(np.abs(np.asarray(J) - Jrb)))}
        out["T8b"].append(row)
        print(f"[T8b {mtag}] M={M} dR={row['dR']:.3e} dT={row['dT']:.3e} "
              f"dJ={row['dJones']:.3e}", flush=True)
    save()

STACKS_C = [
    ("lossless_mixed", [(0.19e-6, CELL_U), (0.13e-6, T_OOP2),
                        (0.09e-6, 2.25 * np.eye(3, dtype=complex))]),
    ("lossy_oop_plus_inplane", [(0.19e-6, CELL_L), (0.13e-6, T_OOP2),
                                (0.09e-6, 2.25 * np.eye(3, dtype=complex))]),
]
for sname, layers in STACKS_C:
    for M in (5, 6, 7, 8):
        st, (o, R, T, J) = stack(layers, np.deg2rad(20.0), np.deg2rad(35.0), M,
                                 retain=True)
        A = np.asarray(st.layer_absorption())
        dev = [float(abs(A[:, r].sum() - (1.0 - R[r].sum() - T[r].sum())))
               for r in (0, 1)]
        clo = [float(abs(R[r].sum() + T[r].sum() - 1)) for r in (0, 1)]
        row = {"stack": sname, "M": M, "shape": list(A.shape),
               "budget_dev": dev, "closure": clo,
               "A": A.tolist(),
               "max_abs_lossless_layers": float(np.max(np.abs(A[1:, :])))}
        out["T8c"].append(row)
        print(f"[T8c {sname}] M={M} |R+T-1|={clo[0]:.2e} budget_dev="
              f"{dev[0]:.3e}/{dev[1]:.3e} A0={A[0, 0]:.6f} "
              f"max|A_lossless|={row['max_abs_lossless_layers']:.2e}",
              flush=True)
    save()
print("DONE")
