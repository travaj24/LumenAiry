"""V5 -- deliberate attempts to BREAK the out-of-plane forward/backward split
and the generalized cascade.

Four stressors, each with the out-of-plane generator in the loop:

  B1  a LOSSY METAL region (``eps = -20 + 2i``) inside an out-of-plane LC host
      -- the case where genuine flux and deep decay coexist;
  B2  a wavelength walked onto a Rayleigh cutoff, from just OUTSIDE the
      library's own warning band to inside it;
  B3  a HIGH-CONTRAST pillar at ``M = 8`` (the largest allowed) -- the widest
      eigenvalue spread the budget permits;
  B4  oblique 60 degrees, where the transverse momentum is largest.

Measured for each: the forward/backward split, the forward count BEFORE
``_select_forward_flux``'s defensive rebalance (re-implemented here, because
the library's ``2 q^2 / 2 q^2`` guard sits downstream of the rebalance and can
never fire), the max forward growth ``exp(-Re(lam_f) k0 L)`` (any value above 1
is a growing mode classified forward), the max backward growth, and -- for the
Hermitian arms -- the energy closure.

Usage:  PYTHONPATH=<root> python v5_break_attempts.py <root> <out.json>
"""
import json
import os
import sys
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import scipy.linalg as sla  # noqa: E402

import lumenairy  # noqa: E402

ROOT = os.path.abspath(sys.argv[1])
assert os.path.abspath(lumenairy.__file__).startswith(ROOT), (
    f"lumenairy.__file__ = {lumenairy.__file__} is not under {ROOT}")

from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    Granet2DTransverseE,
    _region_modes_oop,
    pmm_jones_2d_staggered,
)
from lumenairy.elements.rcwa import uniaxial_tensor  # noqa: E402

OUT = sys.argv[2]
WL = 0.62e-6
NSUB = 1.50
NSUP = 1.0

LC = uniaxial_tensor(1.50, 1.75, 0.66, phi=0.42)
LC_HI = uniaxial_tensor(1.40, 3.60, 0.66, phi=0.42)        # huge birefringence
METAL = (-20.0 + 2.0j) * np.eye(3, dtype=complex)
HIGH = 16.0 * np.eye(3, dtype=complex)


def raw_forward_count(lam, Vfull, N):
    """Independent copy of the classification rule ``_select_forward_flux``
    applies BEFORE its defensive rebalance."""
    gre = np.real(lam)
    Ex, Ey = Vfull[:N], Vfull[N:2 * N]
    Hx, Hy = Vfull[2 * N:3 * N] / 1j, Vfull[3 * N:4 * N] / 1j
    Sz = np.real(np.sum(Ex * np.conj(Hy) - Ey * np.conj(Hx), axis=0))
    mx = max(1.0, float(np.max(np.abs(Sz))))
    carries = np.abs(Sz) > 1e-9 * mx
    carries &= ~((np.abs(Sz) < 3e-3 * mx) & (np.abs(gre) > 0.1))
    carries &= ~(np.abs(gre) > 0.5)
    return int(np.count_nonzero(np.where(carries, Sz > 0, gre > 0)))


def probe_modes(cell, px, wl, th, ph, M):
    k0 = 2.0 * np.pi / wl
    kx0 = float(np.real(NSUP) * np.sin(th) * np.cos(ph))
    ky0 = float(np.real(NSUP) * np.sin(th) * np.sin(ph))
    s = Granet2DTransverseE(px, px, cell.shape[0], cell.shape[1], M, cell,
                            alpha0x=kx0 * k0, alpha0y=ky0 * k0, k0=k0)
    Wf, Vf, lf, Wb, Vb, lb = _region_modes_oop(s)
    qq = s.q * s.q
    Lc = np.linalg.cholesky(s.Bgen)
    Ah = sla.solve_triangular(Lc, s.Agen, lower=True)
    Ah = sla.solve_triangular(Lc, Ah.conj().T, lower=True).conj().T
    qv, Y = np.linalg.eig(Ah)
    X = sla.solve_triangular(Lc.conj().T, Y, lower=False)
    L1 = np.linalg.cholesky(s.Bgen[:qq, :qq]).conj().T
    L2 = np.linalg.cholesky(s.Bgen[qq:2 * qq, qq:2 * qq]).conj().T
    Vfull = np.concatenate([L1 @ X[:qq], L2 @ X[qq:2 * qq],
                            L2 @ X[2 * qq:3 * qq], L1 @ X[3 * qq:]], axis=0)
    nrm = np.linalg.norm(Vfull, axis=0)
    Vfull = Vfull / np.where(nrm == 0.0, 1.0, nrm)[None, :]
    return dict(qq=qq, nf=int(lf.size), nb=int(lb.size),
                raw_fwd=raw_forward_count(-1j * qv, Vfull, qq),
                min_re_lam_f=float(np.min(np.real(lf))),
                max_re_lam_b=float(np.max(np.real(lb))),
                max_abs_lam=float(np.max(np.abs(np.concatenate([lf, lb])))),
                lam_f=lf, lam_b=lb, k0=k0)


def cell2(bg, pix=None, n=2):
    c = np.zeros((n, n, 3, 3), dtype=complex)
    c[:] = bg
    if pix is not None:
        c[0, 0] = pix
    return c


out = {"root": ROOT, "lumenairy": lumenairy.__file__, "cases": []}


def save():
    json.dump(out, open(OUT, "w"), indent=1)


def run(name, cell, px, wl, th, ph, M, dep, hermitian):
    m = probe_modes(cell, px, wl, th, ph, M)
    grow = float(np.max(np.exp(-np.real(m["lam_f"]) * m["k0"] * dep)))
    growb = float(np.max(np.exp(np.real(m["lam_b"]) * m["k0"] * dep)))
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        o, R, T, J = pmm_jones_2d_staggered(px, px, cell, NSUB, NSUP, dep, wl,
                                            degree=M, n_orders=3, theta=th,
                                            phi=ph)
    msgs = sorted({str(x.message)[:90] for x in wlist})
    rec = {"case": name, "M": M, "split": [m["nf"], m["nb"]],
           "expected": 2 * m["qq"],
           "raw_fwd_before_rebalance": m["raw_fwd"],
           "rebalance_engaged": bool(m["raw_fwd"] != 2 * m["qq"]),
           "min_Re_lam_f": m["min_re_lam_f"], "max_Re_lam_b": m["max_re_lam_b"],
           "max_abs_lam": m["max_abs_lam"],
           "max_fwd_growth": grow, "max_bwd_growth": growb,
           "sumRT": [float(R[r].sum() + T[r].sum()) for r in (0, 1)],
           "finite": bool(np.all(np.isfinite(R)) and np.all(np.isfinite(T))
                          and np.all(np.isfinite(np.asarray(J)))),
           "maxR": float(np.max(R)), "maxT": float(np.max(T)),
           "minR": float(np.min(R)), "minT": float(np.min(T)),
           "warnings": msgs}
    if hermitian:
        rec["closure"] = float(np.max(np.abs(R.sum(axis=1) + T.sum(axis=1)
                                             - 1)))
    out["cases"].append(rec)
    print(f"[{name}] split={rec['split']}/{rec['expected']} raw_fwd="
          f"{rec['raw_fwd_before_rebalance']} rebal="
          f"{rec['rebalance_engaged']} minRe(lam_f)={rec['min_Re_lam_f']:.3e} "
          f"fwd_growth={grow:.4e} bwd_growth={growb:.4e} "
          f"R+T={rec['sumRT'][0]:.6f} closure={rec.get('closure')} "
          f"maxR={rec['maxR']:.4f} minR={rec['minR']:.2e} warn={len(msgs)}",
          flush=True)
    for x in msgs:
        print("     WARN:", x, flush=True)
    save()


PX = 0.98e-6
# ---- B1: lossy METAL inside an out-of-plane LC host ----------------------
for mtag, th, ph in (("normal", 0.0, 0.0),
                     ("oblique60", np.deg2rad(60.0), 0.0),
                     ("conical60_40", np.deg2rad(60.0), np.deg2rad(40.0))):
    run(f"B1_metal_in_oop_lc_{mtag}", cell2(LC, METAL), PX, WL, th, ph, 7,
        0.30e-6, hermitian=False)
run("B1_metal_in_oop_lc_deep_3lam", cell2(LC, METAL), PX, WL,
    np.deg2rad(60.0), 0.0, 7, 3.0 * WL, hermitian=False)

# ---- B2: walk onto a Rayleigh cutoff -------------------------------------
# the (1,0) order of the SUBSTRATE cuts off at wl/px = n_sub - kx0.  Approach
# it from outside the library's 1e-4 (kt^2 units) warning band.
for frac in (3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 0.0):
    kt = NSUB - frac
    wl = kt * PX                              # kx = wl/px at normal incidence
    run(f"B2_cutoff_gap{frac:g}", cell2(LC, HIGH), PX, wl, 0.0, 0.0, 6,
        0.28e-6, hermitian=True)

# ---- B3: HIGH-CONTRAST pillar at M = 8 -----------------------------------
run("B3_high_contrast_M8_normal", cell2(LC_HI, HIGH), PX, WL, 0.0, 0.0, 8,
    0.30e-6, hermitian=True)
run("B3_high_contrast_M8_oblique60", cell2(LC_HI, HIGH), PX, WL,
    np.deg2rad(60.0), 0.0, 8, 0.30e-6, hermitian=True)
run("B3_high_contrast_33_M8_conical", cell2(LC_HI, HIGH, 3), PX, WL,
    np.deg2rad(60.0), np.deg2rad(40.0), 8, 0.30e-6, hermitian=True)

# ---- B4: oblique 60 on a plain out-of-plane pillar, depth ladder ---------
for dl in (0.25, 1.0, 3.0):
    run(f"B4_oblique60_depth{dl}lam", cell2(LC, 2.25 * np.eye(3,
                                                              dtype=complex)),
        PX, WL, np.deg2rad(60.0), 0.0, 7, dl * WL, hermitian=True)
print("DONE")
