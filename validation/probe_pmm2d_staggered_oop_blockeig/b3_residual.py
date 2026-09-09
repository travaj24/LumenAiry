"""B3 -- eigenpair backward error of the reduced path against the DENSE path's
own backward error on the very same pencil, and the OBSERVABLE fail-before.

B6   normwise backward error ``|A x - q B x| / ((|A| + |q||B|) |x|)`` for both
     paths, plus the flux selector's classification on both.
B5b  the fail-before at the OBSERVABLE level: monkeypatch ``_STAG_BLOCK_TOL``
     to 1.0 so the runtime structure test cannot refuse, then run the SHIPPED
     entry point on cells that do not carry the structure and compare R / T /
     Jones against ``symmetry=False``.

Run:
  cd /c/tmp/lum_oopfast && PYTHONPATH=/c/tmp/lum_oopfast OMP_NUM_THREADS=1 \\
    OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
    python validation/probe_pmm2d_staggered_oop_blockeig/b3_residual.py
"""
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

import lumenairy  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
assert os.path.abspath(lumenairy.__file__).startswith(ROOT), (
    f"lumenairy.__file__ = {lumenairy.__file__} is not under {ROOT}")

import scipy.linalg as sla  # noqa: E402

from lumenairy.elements.pmm import twod_staggered as TS  # noqa: E402
from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    Granet2DTransverseE,
    _region_modes_oop,
    _stag_block_eig,
    _stag_parity_gauge,
    pmm_jones_2d_staggered,
)
from lumenairy.elements.rcwa import uniaxial_tensor  # noqa: E402

OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

PX = PY = 1.10e-6
WL = 0.68e-6
DEP = 0.34e-6
NSUB, NSUP = 1.50, 1.0
K0 = 2.0 * np.pi / WL
TIL = uniaxial_tensor(1.5, 1.7, np.deg2rad(35.0), phi=np.deg2rad(25.0))
NREC = TIL.copy()
NREC[0, 2] = TIL[0, 2] + 0.30j
NREC[2, 0] = TIL[2, 0] - 0.30j
AIR = np.eye(3, dtype=complex)
ISO = 2.25 * np.eye(3, dtype=complex)
R = {"lumenairy": lumenairy.__file__, "version": lumenairy.__version__}


def tile(t, n):
    c = np.zeros((n, n, 3, 3), dtype=complex)
    c[:, :] = t
    return c


def centro(t, n):
    c = tile(AIR, n)
    if n == 2:
        c[0, 0] = c[1, 1] = t
    else:
        c[1, 1] = t
        c[0, 0] = c[2, 2] = ISO
    return c


def offcentre(t, n):
    c = tile(AIR, n)
    c[0, 0] = t
    return c


def broken(n):
    c = tile(AIR, n)
    if n == 2:
        c[0, 0] = TIL
        c[1, 1] = uniaxial_tensor(1.5, 1.7, np.deg2rad(35.0),
                                  phi=np.deg2rad(-70.0))
    else:
        c[1, 1] = c[0, 0] = TIL
        c[2, 2] = uniaxial_tensor(1.58, 1.82, 1.11, phi=2.05)
    return c


def sol_for(cell, M):
    return Granet2DTransverseE(PX, PY, cell.shape[0], cell.shape[0], M, cell,
                               k0=K0)


def backward_error(A, B, qv, X):
    na, nb = float(np.linalg.norm(A, 2)), float(np.linalg.norm(B, 2))
    res = A @ X - (B @ X) * qv[None, :]
    num = np.linalg.norm(res, axis=0)
    den = (na + np.abs(qv) * nb) * np.linalg.norm(X, axis=0)
    return float(np.max(num / den))


print("\n## B6  eigenpair backward error, factored vs the dense zgeev on the "
      "SAME pencil")
print("   cell                    grid  M   4q^2   dense       factored    "
      "ratio   fwd count dense/factored")
rows = []
for Nx in (2, 3):
    for M in (5, 6, 7):
        for name, cell in (("uniform tilted uniaxial", tile(TIL, Nx)),
                           ("centro pillar", centro(TIL, Nx)),
                           ("centro NON-RECIPROCAL", centro(NREC, Nx))):
            sol = sol_for(cell, M)
            A, B = sol.Agen, sol.Bgen
            qq = sol.q * sol.q
            g = _stag_parity_gauge(sol)
            fac = _stag_block_eig(A, B, qq, g)
            assert fac is not None
            Lc = np.linalg.cholesky(B)
            Ah = sla.solve_triangular(Lc, A, lower=True)
            Ah = sla.solve_triangular(Lc, Ah.conj().T, lower=True).conj().T
            qd, Y = np.linalg.eig(Ah)
            Xd = sla.solve_triangular(Lc.conj().T, Y, lower=False)
            bd = backward_error(A, B, qd, Xd)
            bf = backward_error(A, B, fac[0], fac[1])
            n_on = _region_modes_oop(sol, symmetry=True)[2].size
            n_off = _region_modes_oop(sol, symmetry=False)[2].size
            print(f"   {name:23s} ({Nx},{Nx})  {M}   {4*qq:5d}  {bd:.3e}  "
                  f"{bf:.3e}  {bf/bd:6.2f}   {n_off} / {n_on}")
            rows.append(dict(cell=name, Nx=Nx, M=M, dim=4 * qq, dense=bd,
                             factored=bf, ratio=bf / bd, fwd_off=n_off,
                             fwd_on=n_on))
R["B6"] = rows
print(f"   dense {min(r['dense'] for r in rows):.2e} .. "
      f"{max(r['dense'] for r in rows):.2e}   factored "
      f"{min(r['factored'] for r in rows):.2e} .. "
      f"{max(r['factored'] for r in rows):.2e}   ratio "
      f"{min(r['ratio'] for r in rows):.2f} .. "
      f"{max(r['ratio'] for r in rows):.2f}")

print("\n## B5b  OBSERVABLE fail-before: _STAG_BLOCK_TOL monkeypatched to 1.0 "
      "so the structure test cannot refuse")
print("   case                     grid  M   |RAR+A|/|A|   dR         dT     "
      "    dJones     (vs symmetry=False)")
rows = []
saved = TS._STAG_BLOCK_TOL
for name, cell, Nx, M in (("OFF-CENTRE pillar", offcentre(TIL, 2), 2, 6),
                          ("OFF-CENTRE pillar", offcentre(TIL, 3), 3, 6),
                          ("parity-BREAKING tensor", broken(2), 2, 6),
                          ("parity-BREAKING tensor", broken(3), 3, 6)):
    sol = sol_for(cell, M)
    perm, r = _stag_parity_gauge(sol)
    rr = r[:, None] * r[None, :]
    dA = (float(np.max(np.abs(rr * sol.Agen[np.ix_(perm, perm)] + sol.Agen)))
          / float(np.max(np.abs(sol.Agen))))
    ref = pmm_jones_2d_staggered(PX, PY, cell, NSUB, NSUP, DEP, WL, degree=M,
                                 n_orders=5, symmetry=False)
    try:
        TS._STAG_BLOCK_TOL = 1.0
        bad = pmm_jones_2d_staggered(PX, PY, cell, NSUB, NSUP, DEP, WL,
                                     degree=M, n_orders=5, symmetry=True)
    finally:
        TS._STAG_BLOCK_TOL = saved
    dR = float(np.max(np.abs(np.asarray(bad[1]) - np.asarray(ref[1]))))
    dT = float(np.max(np.abs(np.asarray(bad[2]) - np.asarray(ref[2]))))
    dJ = float(np.max(np.abs(np.asarray(bad[3]) - np.asarray(ref[3]))))
    print(f"   {name:24s} ({Nx},{Nx})  {M}   {dA:.3e}   {dR:.3e}  {dT:.3e}  "
          f"{dJ:.3e}")
    rows.append(dict(case=name, Nx=Nx, M=M, dA=dA, dR=dR, dT=dT, dJones=dJ))
R["B5b"] = rows
assert TS._STAG_BLOCK_TOL == saved

with open(os.path.join(OUT, "b3_residual.json"), "w") as f:
    json.dump(R, f, indent=1, default=str)
print("\nwrote results/b3_residual.json")
