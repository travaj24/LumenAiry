"""V2 -- (a) the STRUCTURE RESIDUAL on both sides of the 1e-10 bar, measured on
MY fixtures with MY own re-implementation of ``R M R``; and (b) ON vs OFF on
parity-symmetric OUT-OF-PLANE cells at NORMAL incidence: observables, the
eigenvalue SET, and the exact ``2 q^2 / 2 q^2`` forward/backward split on both
arms.

The residuals are computed here from the signed permutation the library
returns, but with a DENSE matrix ``R`` built explicitly (``R = P diag(r)``)
rather than by the library's row-blocked index trick -- two independent
routes to the same quantity, so a bug in the blocked indexing would show.

Usage: python v2_on_off.py
"""
import json
import os
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import vfix as F  # noqa: E402

F.assert_arm("C:/tmp/lum_vacc")

from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    Granet2DTransverseE,
    _region_modes_oop,
    _stag_block_eig,
    _stag_parity_gauge,
    pmm_jones_2d_staggered,
)

K0 = 2.0 * np.pi / F.WL


def solver(cell, M, theta=0.0, phi=0.0):
    a0x = float(np.real(F.NSUP)) * np.sin(theta) * np.cos(phi) * K0
    a0y = float(np.real(F.NSUP)) * np.sin(theta) * np.sin(phi) * K0
    n = np.asarray(cell).shape[0]
    return Granet2DTransverseE(F.PX, F.PY, n, n, M, cell,
                               alpha0x=a0x, alpha0y=a0y, k0=K0)


def dense_R(perm, r):
    """``R`` as an explicit dense matrix: ``R e_i = r_i e_{perm_i}``."""
    n = perm.size
    R = np.zeros((n, n))
    R[perm, np.arange(n)] = r
    return R


def residuals_dense(sol):
    """(dA, dB, R2err) via an EXPLICIT dense R -- independent of the library's
    row-blocked index arithmetic."""
    g = _stag_parity_gauge(sol)
    if g is None:
        return None
    perm, r = g
    R = dense_R(perm, r)
    A, B = sol.Agen, sol.Bgen
    dA = float(np.max(np.abs(R @ A @ R + A))) / float(np.max(np.abs(A)))
    dB = float(np.max(np.abs(R @ B @ R - B))) / float(np.max(np.abs(B)))
    r2 = float(np.max(np.abs(R @ R - np.eye(perm.size))))
    return dA, dB, r2


def carrying_cells():
    t = F.tensors()
    out = []
    # 1-4: uniform tilings (trivially their own parity image)
    out.append(("uniform tilted uniaxial (2,2)", F.tile(t["lc"], 2)))
    out.append(("uniform negative uniaxial (3,3)", F.tile(t["lc2"], 3)))
    # 5-8: mirror PAIRS
    out.append(("centro pair lc (2,2)", F.centro_pair(t["lc"], 2)))
    out.append(("centro pair generic dense tensor (2,2)",
                F.centro_pair(t["generic"], 2)))
    out.append(("centro pair NON-RECIPROCAL (2,2)",
                F.centro_pair(t["nonrec"], 2)))
    out.append(("centro pair LOSSY (2,2)", F.centro_pair(t["lossy"], 2)))
    out.append(("centro pair strongly-lossy METAL (2,2)",
                F.centro_pair(t["metal"], 2)))
    # 9-12: (3,3) with an INTERIOR feature (the parity-fixed centre pixel)
    out.append(("(3,3) ring lc + INTERIOR lossy centre",
                F.centro_interior(t["lc"], t["lossy"])))
    out.append(("(3,3) ring generic + INTERIOR non-reciprocal centre",
                F.centro_interior(t["generic"], t["nonrec"])))
    out.append(("(3,3) ring metal + INTERIOR lc2 centre",
                F.centro_interior(t["metal"], t["lc2"])))
    # 13-14: a 4-pixel orbit with DIFFERENT tensors on the two orbits
    out.append(("(3,3) cross orbit lc / nonrec",
                F.centro_cross(t["lc"], t["nonrec"], 3)))
    out.append(("(2,2) cross orbit lossy / generic",
                F.centro_cross(t["lossy"], t["generic"], 2)))
    # 15: centro pair PLUS a distinct centre, both out-of-plane
    out.append(("(3,3) pair lc2 + centre metal",
                F.centro_pair(t["lc2"], 3, centre=t["metal"])))
    return out


def violating_cells():
    t = F.tensors()
    out = []
    out.append(("off-centre pillar lc (2,2)", F.offcentre(t["lc"], 2), 0., 0.))
    out.append(("off-centre pillar generic (3,3)",
                F.offcentre(t["generic"], 3), 0., 0.))
    out.append(("parity-breaking tensor lc/lc2 (2,2)",
                F.parity_breaking_tensor(t["lc"], t["lc2"], 2), 0., 0.))
    out.append(("parity-breaking tensor lc/nonrec (3,3)",
                F.parity_breaking_tensor(t["lc"], t["nonrec"], 3), 0., 0.))
    # a MINIMAL parity break: the mirror pixel differs only in the 3rd digit
    t2 = t["lc"].copy()
    t2[0, 2] *= (1.0 + 1e-3)
    t2[2, 0] *= (1.0 + 1e-3)
    out.append(("parity-breaking tensor, 1e-3 relative (2,2)",
                F.parity_breaking_tensor(t["lc"], t2, 2), 0., 0.))
    t3 = t["lc"].copy()
    t3[0, 2] *= (1.0 + 1e-7)
    out.append(("parity-breaking tensor, 1e-7 relative (2,2)",
                F.parity_breaking_tensor(t["lc"], t3, 2), 0., 0.))
    out.append(("centro pair, OBLIQUE 25", F.centro_pair(t["lc"], 2),
                np.deg2rad(25.0), 0.0))
    out.append(("centro pair, CONICAL 25/40", F.centro_pair(t["lc"], 3),
                np.deg2rad(25.0), np.deg2rad(40.0)))
    out.append(("centro pair, OBLIQUE 1e-6 rad", F.centro_pair(t["lc"], 2),
                1e-6, 0.0))
    return out


def part_a():
    print("=" * 96)
    print("A. STRUCTURE RESIDUAL on the ASSEMBLED pencil (dense R, my own)")
    print("=" * 96)
    rows = []
    print(f"{'cell':56s} {'grid':6s} {'M':2s} {'dA':>10s} {'dB':>10s} "
          f"{'|R^2-I|':>9s} {'engages':7s}")
    for name, cell in carrying_cells():
        n = cell.shape[0]
        for M in (5, 6, 7, 8):
            sol = solver(cell, M)
            res = residuals_dense(sol)
            g = _stag_parity_gauge(sol)
            eng = _stag_block_eig(sol.Agen, sol.Bgen, sol.q * sol.q,
                                  g) is not None
            rows.append(dict(cell=name, grid=n, M=M, dA=res[0], dB=res[1],
                             r2=res[2], engages=bool(eng), carrying=True))
            print(f"{name:56s} ({n},{n}) {M:2d} {res[0]:10.3e} {res[1]:10.3e} "
                  f"{res[2]:9.1e} {str(eng):7s}")
    print()
    for name, cell, th, ph in violating_cells():
        n = cell.shape[0]
        for M in (6,):
            sol = solver(cell, M, th, ph)
            g = _stag_parity_gauge(sol)
            if g is None:
                rows.append(dict(cell=name, grid=n, M=M, dA=None, dB=None,
                                 r2=None, engages=False, carrying=False,
                                 gauge_refuses=True))
                print(f"{name:56s} ({n},{n}) {M:2d} "
                      f"{'GAUGE REFUSES':>21s} {'':>9s} {'False':7s}")
                continue
            res = residuals_dense(sol)
            eng = _stag_block_eig(sol.Agen, sol.Bgen, sol.q * sol.q,
                                  g) is not None
            rows.append(dict(cell=name, grid=n, M=M, dA=res[0], dB=res[1],
                             r2=res[2], engages=bool(eng), carrying=False,
                             gauge_refuses=False))
            print(f"{name:56s} ({n},{n}) {M:2d} {res[0]:10.3e} {res[1]:10.3e} "
                  f"{res[2]:9.1e} {str(eng):7s}")
    car = [r for r in rows if r["carrying"]]
    vio = [r for r in rows if not r["carrying"] and r.get("dA") is not None]
    print()
    print(f"carrying envelope dA : {min(r['dA'] for r in car):.3e} .. "
          f"{max(r['dA'] for r in car):.3e}   ({len(car)} rows, all engage="
          f"{all(r['engages'] for r in car)})")
    print(f"carrying envelope dB : {min(r['dB'] for r in car):.3e} .. "
          f"{max(r['dB'] for r in car):.3e}")
    print(f"smallest REAL violation dA: {min(r['dA'] for r in vio):.3e}   "
          f"(any engaged? {any(r['engages'] for r in vio)})")
    return rows


def part_b():
    print()
    print("=" * 96)
    print("B. ON vs OFF at NORMAL incidence on parity-symmetric OOP cells")
    print("=" * 96)
    rows = []
    hdr = (f"{'cell':56s} {'M':2s} {'4q^2':>6s} {'dR':>10s} {'dT':>10s} "
           f"{'dJones':>10s} {'d(eig set)':>11s} {'split':>11s} "
           f"{'min|q|/max':>10s}")
    print(hdr)
    for name, cell in carrying_cells():
        n = cell.shape[0]
        M = 6 if n == 3 else 7
        sol = solver(cell, M)
        qq = sol.q * sol.q
        g = _stag_parity_gauge(sol)
        fac = _stag_block_eig(sol.Agen, sol.Bgen, qq, g)
        assert fac is not None, f"{name} did not engage -- vacuous arm"
        on = _region_modes_oop(sol, symmetry=True)
        off = _region_modes_oop(sol, symmetry=False)
        split_ok = (on[2].size == off[2].size == 2 * qq
                    and on[5].size == off[5].size == 2 * qq)
        eigset = F.hausdorff(np.concatenate([on[2], on[5]]),
                             np.concatenate([off[2], off[5]]))
        qv = fac[0]
        rat = float(np.min(np.abs(qv)) / np.max(np.abs(qv)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = pmm_jones_2d_staggered(F.PX, F.PY, cell, F.NSUB, F.NSUP,
                                       F.DEP, F.WL, degree=M, n_orders=5,
                                       symmetry="auto")
            b = pmm_jones_2d_staggered(F.PX, F.PY, cell, F.NSUB, F.NSUP,
                                       F.DEP, F.WL, degree=M, n_orders=5,
                                       symmetry=False)
        dR, dT, dJ = (F.dmax(a[1], b[1]), F.dmax(a[2], b[2]),
                      F.dmax(a[3], b[3]))
        assert np.array_equal(np.asarray(a[0]), np.asarray(b[0]))
        rows.append(dict(cell=name, grid=n, M=M, q4=4 * qq, dR=dR, dT=dT,
                         dJ=dJ, eigset=eigset, split_ok=bool(split_ok),
                         fwd_on=int(on[2].size), fwd_off=int(off[2].size),
                         bwd_on=int(on[5].size), bwd_off=int(off[5].size),
                         two_qq=int(2 * qq), qratio=rat,
                         identical=(F.sha(*a) == F.sha(*b))))
        print(f"{name:56s} {M:2d} {4 * qq:6d} {dR:10.3e} {dT:10.3e} "
              f"{dJ:10.3e} {eigset:11.3e} {str(split_ok):>11s} {rat:10.3e}")
    print()
    print(f"worst dR        {max(r['dR'] for r in rows):.3e}")
    print(f"worst dT        {max(r['dT'] for r in rows):.3e}")
    print(f"worst dJones    {max(r['dJ'] for r in rows):.3e}")
    print(f"worst d(eigset) {max(r['eigset'] for r in rows):.3e}")
    print(f"split exact 2q^2/2q^2 on BOTH arms in "
          f"{sum(r['split_ok'] for r in rows)}/{len(rows)} rows")
    print(f"min|q|/max|q| range {min(r['qratio'] for r in rows):.2e} .. "
          f"{max(r['qratio'] for r in rows):.2e} "
          f"(floor _STAG_GAM_FLOOR = 1e-13)")
    print(f"bit-identical rows (should be 0 -- the arms are different "
          f"algorithms): {sum(r['identical'] for r in rows)}")
    return rows


if __name__ == "__main__":
    a = part_a()
    b = part_b()
    with open(os.path.join(HERE, "results", "v2_on_off.json"), "w") as fh:
        json.dump(dict(structure=a, agreement=b), fh, indent=1)
