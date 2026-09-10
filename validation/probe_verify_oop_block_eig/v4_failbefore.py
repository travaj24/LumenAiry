"""V4 -- FAIL-BEFORE, the BAR's own safety, and the eigenpair BACKWARD ERROR.

A. FAIL-BEFORE.  ``_STAG_BLOCK_TOL`` is monkeypatched to 1.0 (the one knob
   that decides) and the SHIPPED entry point is run on cells that do not carry
   the structure.  The observables must move by decades, else the runtime
   verification is decorative.

B. THE BAR'S OWN SAFETY -- the question the build doc does not ask.  The doc
   argues the 1e-10 bar has "8.8 decades below the smallest real violation",
   but that gap is a property of the four violating fixtures it chose, not of
   the gate.  Here the violation is made CONTINUOUS: one tensor entry of the
   mirror pixel is perturbed by a relative ``d``, sweeping ``dA`` across the
   bar from 1e-14 to 1e-1, and at each rung the ACCEPTED-or-REFUSED decision
   and the observable error of the (forced) reduction are recorded.  What the
   bar must guarantee is that any cell it ACCEPTS produces an answer within the
   agreement the tests assert.

C. BACKWARD ERROR of the factored eigenpairs against the dense ``zgeev`` on the
   SAME pencil, normwise |A x - q B x| / ((|A| + |q||B|) |x|).  The build
   claims a ratio 0.28 .. 7.20 and the test asserts <= 1e2 x; both sides of
   that bar are measured here.

Usage: python v4_failbefore.py
"""
import json
import os
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import scipy.linalg as sla  # noqa: E402
import vfix as F  # noqa: E402

F.assert_arm("C:/tmp/lum_vacc")

from lumenairy.elements.pmm import twod_staggered as TS  # noqa: E402
from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    Granet2DTransverseE,
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


def jones(cell, M, sym, n_orders=5):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pmm_jones_2d_staggered(F.PX, F.PY, cell, F.NSUB, F.NSUP, F.DEP,
                                      F.WL, degree=M, n_orders=n_orders,
                                      symmetry=sym)


def struct_dA(sol):
    g = _stag_parity_gauge(sol)
    if g is None:
        return None
    perm, r = g
    A = sol.Agen
    R = np.zeros((perm.size, perm.size))
    R[perm, np.arange(perm.size)] = r
    return float(np.max(np.abs(R @ A @ R + A))) / float(np.max(np.abs(A)))


def dense_eig(sol):
    A, B = sol.Agen, sol.Bgen
    Lc = np.linalg.cholesky(B)
    Ah = sla.solve_triangular(Lc, A, lower=True)
    Ah = sla.solve_triangular(Lc, Ah.conj().T, lower=True).conj().T
    qd, Y = np.linalg.eig(Ah)
    return qd, sla.solve_triangular(Lc.conj().T, Y, lower=False)


def backward_error(A, B, qv, X):
    na, nb = float(np.linalg.norm(A, 2)), float(np.linalg.norm(B, 2))
    res = A @ X - (B @ X) * qv[None, :]
    den = (na + np.abs(qv) * nb) * np.linalg.norm(X, axis=0)
    return float(np.max(np.linalg.norm(res, axis=0) / den))


# --------------------------------------------------------------------------- #
def part_a():
    print("=" * 100)
    print("A. FAIL-BEFORE -- _STAG_BLOCK_TOL disarmed (inf), SHIPPED entry "
          "point, structure-violating cells")
    print("=" * 100)
    t = F.tensors()
    cases = [
        ("off-centre pillar lc (2,2)", F.offcentre(t["lc"], 2), 6),
        ("off-centre pillar generic (3,3)", F.offcentre(t["generic"], 3), 6),
        ("parity-breaking lc/lc2 (2,2)",
         F.parity_breaking_tensor(t["lc"], t["lc2"], 2), 6),
        ("parity-breaking lc/nonrec (3,3)",
         F.parity_breaking_tensor(t["lc"], t["nonrec"], 3), 6),
        ("off-centre metal (2,2)", F.offcentre(t["metal"], 2), 7),
    ]
    rows = []
    print(f"{'cell':36s} {'M':2s} {'dA':>10s} {'d(eigset)':>11s} {'dR':>10s} "
          f"{'dT':>10s} {'dJones':>10s} {'shipped==dense':14s} {'tol=1 accepts':5s}")
    for name, cell, M in cases:
        sol = solver(cell, M)
        g = _stag_parity_gauge(sol)
        dA = struct_dA(sol)
        qq = sol.q * sol.q
        # NOTE: ``tol=1.0`` -- the value the shipped test monkeypatches --
        # is NOT a full disarm: ``dA`` can exceed 1 (it is bounded by 2), and
        # a cell with ``dA > 1`` is still REFUSED at tol=1.0.  ``np.inf``
        # is the actual disarm, and both are recorded.
        acc_at_1 = _stag_block_eig(sol.Agen, sol.Bgen, qq, g,
                                   tol=1.0) is not None
        forced = _stag_block_eig(sol.Agen, sol.Bgen, qq, g, tol=np.inf)
        assert forced is not None, "the disarmed gate must accept"
        qd, _Xd = dense_eig(sol)
        eigset = F.hausdorff(forced[0], qd)
        ref = jones(cell, M, False)
        saved = TS._STAG_BLOCK_TOL
        try:
            TS._STAG_BLOCK_TOL = np.inf
            bad = jones(cell, M, True)
        finally:
            TS._STAG_BLOCK_TOL = saved
        assert TS._STAG_BLOCK_TOL == saved
        shipped = jones(cell, M, True)
        dR, dT, dJ = (F.dmax(bad[1], ref[1]), F.dmax(bad[2], ref[2]),
                      F.dmax(bad[3], ref[3]))
        same = F.sha(*shipped) == F.sha(*ref)
        rows.append(dict(cell=name, M=M, dA=dA, eigset=eigset, dR=dR, dT=dT,
                         dJ=dJ, shipped_is_dense=bool(same),
                         accepted_at_tol_1=bool(acc_at_1)))
        print(f"{name:36s} {M:2d} {dA:10.3e} {eigset:11.3e} {dR:10.3e} "
              f"{dT:10.3e} {dJ:10.3e} {str(same):14s} {str(acc_at_1):5s}")
    print()
    print(f"forced-error envelope: dR {min(r['dR'] for r in rows):.2e} .. "
          f"{max(r['dR'] for r in rows):.2e};  dT "
          f"{min(r['dT'] for r in rows):.2e} .. {max(r['dT'] for r in rows):.2e}"
          f";  dJones {min(r['dJ'] for r in rows):.2e} .. "
          f"{max(r['dJ'] for r in rows):.2e}")
    print(f"eigenvalue-set error   {min(r['eigset'] for r in rows):.2e} .. "
          f"{max(r['eigset'] for r in rows):.2e}")
    print(f"the SHIPPED answer equals the dense one on "
          f"{sum(r['shipped_is_dense'] for r in rows)}/{len(rows)} rows")
    return rows


# --------------------------------------------------------------------------- #
def part_b():
    print()
    print("=" * 100)
    print("B. THE BAR'S OWN SAFETY -- a CONTINUOUS violation swept across "
          "_STAG_BLOCK_TOL = 1e-10")
    print("=" * 100)
    print("The mirror pixel's e13/e31 are scaled by (1 + d); dA is measured, "
          "the gate's DECISION recorded,")
    print("and the reduction is FORCED (tol=inf) so the error it WOULD have "
          "produced is visible on both sides.")
    t = F.tensors()
    M = 6
    rows = []
    print()
    print(f"{'d (relative)':>13s} {'dA':>11s} {'accepted':>9s} {'dR(forced)':>11s} "
          f"{'dT(forced)':>11s} {'dJ(forced)':>11s} {'dR/dA':>9s}")
    for d in (0.0, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 3e-9, 1e-8, 1e-7,
              1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
        t2 = t["lc"].copy()
        t2[0, 2] *= (1.0 + d)
        t2[2, 0] *= (1.0 + d)
        cell = F.tile(F.AIR, 2)
        cell[0, 0] = t["lc"]
        cell[1, 1] = t2
        sol = solver(cell, M)
        g = _stag_parity_gauge(sol)
        dA = struct_dA(sol)
        qq = sol.q * sol.q
        accepted = _stag_block_eig(sol.Agen, sol.Bgen, qq, g) is not None
        ref = jones(cell, M, False)
        saved = TS._STAG_BLOCK_TOL
        try:
            TS._STAG_BLOCK_TOL = np.inf
            forced = jones(cell, M, True)
        finally:
            TS._STAG_BLOCK_TOL = saved
        dR, dT, dJ = (F.dmax(forced[1], ref[1]), F.dmax(forced[2], ref[2]),
                      F.dmax(forced[3], ref[3]))
        # and the SHIPPED answer (gate armed) vs the dense one
        shipped = jones(cell, M, True)
        ship_same = F.sha(*shipped) == F.sha(*ref)
        ship_dR = F.dmax(shipped[1], ref[1])
        rows.append(dict(d=d, dA=dA, accepted=bool(accepted), dR=dR, dT=dT,
                         dJ=dJ, shipped_identical=bool(ship_same),
                         shipped_dR=ship_dR))
        print(f"{d:13.1e} {dA:11.3e} {str(accepted):>9s} {dR:11.3e} "
              f"{dT:11.3e} {dJ:11.3e} {(dR / dA if dA else np.nan):9.2e}")
    acc = [r for r in rows if r["accepted"]]
    ref_ = [r for r in rows if not r["accepted"]]
    print()
    print(f"ACCEPTED rungs ({len(acc)}): worst dR that the reduction would "
          f"introduce = {max(r['dR'] for r in acc):.3e}")
    print("   -- the test suite's agreement bar is 1e-11 and its measured "
          "agreement is <= 2.0e-14 (V2)")
    print(f"REFUSED rungs ({len(ref_)}): smallest dA refused = "
          f"{min(r['dA'] for r in ref_):.3e}, and every one of them is "
          f"bit-identical to dense: "
          f"{all(r['shipped_identical'] for r in ref_)}")
    slope = [r["dR"] / r["dA"] for r in rows if r["dA"] and r["dA"] > 1e-12]
    print(f"empirical amplification dR/dA over the sweep: {min(slope):.2e} .. "
          f"{max(slope):.2e}  ->  at the bar (dA = 1e-10) the accepted error "
          f"is <= {max(slope) * 1e-10:.2e}")
    return rows


# --------------------------------------------------------------------------- #
def part_c():
    print()
    print("=" * 100)
    print("C. EIGENPAIR BACKWARD ERROR -- factored vs the dense zgeev on the "
          "SAME pencil")
    print("=" * 100)
    t = F.tensors()
    cells = [("uniform tilted uniaxial", F.tile(t["lc"], 2), 2),
             ("centro pair generic", F.centro_pair(t["generic"], 2), 2),
             ("centro pair NON-RECIPROCAL", F.centro_pair(t["nonrec"], 2), 2),
             ("centro pair LOSSY", F.centro_pair(t["lossy"], 2), 2),
             ("centro pair METAL", F.centro_pair(t["metal"], 2), 2),
             ("(3,3) ring lc + interior lossy",
              F.centro_interior(t["lc"], t["lossy"]), 3),
             ("(3,3) cross orbit lc/nonrec",
              F.centro_cross(t["lc"], t["nonrec"], 3), 3)]
    rows = []
    print(f"{'cell':34s} {'grid':6s} {'M':2s} {'4q^2':>6s} {'dense':>11s} "
          f"{'factored':>11s} {'ratio':>7s} {'fwd d/f':>9s}")
    for name, cell, n in cells:
        for M in (5, 6, 7):
            if n == 3 and M == 7:
                continue
            sol = solver(cell, M)
            qq = sol.q * sol.q
            g = _stag_parity_gauge(sol)
            fac = _stag_block_eig(sol.Agen, sol.Bgen, qq, g)
            assert fac is not None
            qd, Xd = dense_eig(sol)
            bd = backward_error(sol.Agen, sol.Bgen, qd, Xd)
            bf = backward_error(sol.Agen, sol.Bgen, fac[0], fac[1])
            on = TS._region_modes_oop(sol, symmetry=True)
            off = TS._region_modes_oop(sol, symmetry=False)
            rows.append(dict(cell=name, grid=n, M=M, q4=4 * qq, dense=bd,
                             factored=bf, ratio=bf / bd,
                             fwd_dense=int(off[2].size),
                             fwd_fac=int(on[2].size)))
            print(f"{name:34s} ({n},{n}) {M:2d} {4 * qq:6d} {bd:11.3e} "
                  f"{bf:11.3e} {bf / bd:7.2f} "
                  f"{off[2].size:4d}/{on[2].size:<4d}")
    r = [x["ratio"] for x in rows]
    print()
    print(f"dense    envelope {min(x['dense'] for x in rows):.3e} .. "
          f"{max(x['dense'] for x in rows):.3e}   ({len(rows)} combinations)")
    print(f"factored envelope {min(x['factored'] for x in rows):.3e} .. "
          f"{max(x['factored'] for x in rows):.3e}")
    print(f"RATIO    envelope {min(r):.3f} .. {max(r):.3f}   "
          f"(the test's bar is 1e2 -> headroom "
          f"{np.log10(1e2 / max(r)):.2f} decades)")
    print(f"forward counts equal on "
          f"{sum(1 for x in rows if x['fwd_dense'] == x['fwd_fac'])}/"
          f"{len(rows)} rows")
    return rows


if __name__ == "__main__":
    a = part_a()
    b = part_b()
    c = part_c()
    with open(os.path.join(HERE, "results", "v4_failbefore.json"), "w") as fh:
        json.dump(dict(failbefore=a, bar_safety=b, backward=c), fh, indent=1)
