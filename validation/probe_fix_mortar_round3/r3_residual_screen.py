"""ROUND 3, DEFECT V1 -- the RESIDUAL as the generalized site's own instrument.

``r2_populations.py`` measured that ``rcond`` has NO two-sided gap at
``_interface_smatrix_general_mortar_2d``: the HEALTHY population (28 ordinary
stacks) runs 1.3e-14 .. 1.5e-04 and the SLIVER population runs 1.6e-24 ..
1.2e-05, so the two CROSS.  This probe measures the candidate that separates
what actually matters at that site -- CONSISTENT-but-rank-deficient (the V1
false positive) from SINGULAR / INCONSISTENT (the D2 failure that raised a bare
``LinAlgError``) -- and prices it:

  * ``resid_full``  ``||A X - B||_F / ||B||_F``            (one n^3 GEMM)
  * ``resid_probe`` ``||A (X v) - B v|| / ||B v||`` for ONE deterministic
    generic ``v`` (three n^2 matvecs -- the shipped form)
  * the WALL COST of each against the bare ``lu_factor`` + ``lu_solve``.

Populations: the ordinary stacks that reach the site (including a 3-layer stack
whose middle interface has BOTH sides promoted in-plane, and an ``M`` = 8
point), the sliver ladder with the width contract LIFTED, and synthetic
exactly-singular / inconsistent operands built FROM a real operand of the site.

``python r3_residual_screen.py [win|wsl]``
"""
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

import json  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import _fixtures as F  # noqa: E402
import numpy as np  # noqa: E402
import scipy.linalg as sla  # noqa: E402

import lumenairy  # noqa: E402
from lumenairy.elements.pmm import _core as _pc  # noqa: E402
from lumenairy.elements.pmm import twod_staggered as _ts  # noqa: E402
from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure  # noqa: E402

assert os.path.abspath(lumenairy.__file__).lower().startswith(_ROOT.lower()), (
    lumenairy.__file__)
TAG = (sys.argv[1] if len(sys.argv) > 1 else "win")
T0 = time.time()
_C = complex
print(f"[arm {TAG}] lumenairy = {lumenairy.__file__} v{lumenairy.__version__}",
      flush=True)


def _log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def _probe_vec(k):
    """The DETERMINISTIC generic probe.  A fixed-seed PCG64 draw is
    bit-reproducible on every platform numpy supports, so this is a constant of
    the library, not a random number."""
    r = np.random.default_rng(0x5EED)
    return (r.standard_normal(k) + 1j * r.standard_normal(k)).astype(_C)


def _measure(A, B):
    A = np.asarray(A)
    n = int(A.shape[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            lu, piv = sla.lu_factor(A)
        except (ValueError, sla.LinAlgError, np.linalg.LinAlgError,
                sla.LinAlgWarning):
            return dict(n=n, rcond=0.0, resid_full=float("inf"),
                        resid_probe=float("inf"), lu_failed=True)
        X = sla.lu_solve((lu, piv), B)
    anorm = float(np.max(np.sum(np.abs(A), axis=0)))
    gecon = sla.get_lapack_funcs("gecon", (A,))
    rcv, info = gecon(lu, anorm)
    rc = float(rcv) if int(info) == 0 else 0.0
    nB = float(np.linalg.norm(B))
    full = float(np.linalg.norm(A @ X - B) / nB)
    v = _probe_vec(B.shape[1])
    bv = B @ v
    nbv = float(np.linalg.norm(bv))
    probe = float(np.linalg.norm(A @ (X @ v) - bv) / nbv)
    return dict(n=n, rcond=rc, resid_full=full, resid_probe=probe,
                lu_failed=False)


class _Rec:
    def __init__(self):
        self.rows = []

    def __enter__(self):
        self._orig = _pc._guarded_mortar_solve

        def solve(A, B, site, ga=None, gb=None, hint=None, screen="rcond"):
            if "GENERALIZED" not in site:
                return self._orig(A, B, site, ga, gb, hint, screen)
            self.rows.append(_measure(A, B))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lu, piv = sla.lu_factor(np.asarray(A))
                return sla.lu_solve((lu, piv), B)

        _pc._guarded_mortar_solve = solve
        return self

    def __exit__(self, *a):
        _pc._guarded_mortar_solve = self._orig
        return False


def _run(st):
    with _Rec() as rec, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o, R, T = st.solve(jones=False)
    R2, T2 = np.atleast_2d(R), np.atleast_2d(T)
    return rec.rows, float(np.max(np.abs(R2.sum(1) + T2.sum(1) - 1.0))), \
        F.zeroth_R(o, R)


def _both_promoted_stack(M):
    """THREE layers: ONE out-of-plane (which puts the stack on the generalized
    cascade) and TWO in-plane patterned layers on DIFFERENT walls.  The middle
    interface then has BOTH sides promoted by ``_modes_as_general`` -- the
    deepest sub-population of the site."""
    st = PMM2DStackPure(F.P, n_modes=M, n_orders=2, n_substrate=1.5,
                        layer_grids="per-layer")
    st.add_layer(0.13e-6,
                 eps_cell=F.tensor_cell(F.WA, F.WA[0], F.WA[1], F.E_OOP,
                                        F.EPS_H),
                 x_walls=[w * F.P for w in F.WA],
                 y_walls=[w * F.P for w in F.WA])
    st.add_layer(0.09e-6,
                 eps_cell=F.scalar_cell(F.WB, F.WB[0], F.WB[1], 5.0, F.EPS_H),
                 x_walls=[w * F.P for w in F.WB],
                 y_walls=[w * F.P for w in F.WB])
    w3 = (0.19, 0.52)
    st.add_layer(0.10e-6,
                 eps_cell=F.scalar_cell(w3, w3[0], w3[1], F.EPS_B, F.EPS_H),
                 x_walls=[w * F.P for w in w3],
                 y_walls=[w * F.P for w in w3])
    st.set_source(F.WL, theta=F.THETA, phi=F.PHI)
    return st


def _sliver_stack(delta, M, centre=0.44):
    st = PMM2DStackPure(F.P, n_modes=M, n_orders=2, n_substrate=1.5,
                        layer_grids="per-layer")
    st.add_layer(0.13e-6,
                 eps_cell=F.tensor_cell(F.WA, F.WA[0], F.WA[1], F.E_OOP,
                                        F.EPS_H),
                 x_walls=[w * F.P for w in F.WA],
                 y_walls=[w * F.P for w in F.WA])
    sw = (centre, centre + delta)
    st.add_layer(0.09e-6, eps_cell=np.full((3, 3), _C(F.EPS_H)),
                 x_walls=[w * F.P for w in sw],
                 y_walls=[w * F.P for w in F.WA])
    st.add_layer(0.10e-6,
                 eps_cell=F.scalar_cell(F.WB, F.WB[0], F.WB[1], F.EPS_B,
                                        F.EPS_H),
                 x_walls=[w * F.P for w in F.WB],
                 y_walls=[w * F.P for w in F.WB])
    st.set_source(F.WL, theta=F.THETA, phi=F.PHI)
    return st


RES = {}

# ---- 1. HEALTHY -------------------------------------------------------
_log("HEALTHY population")
H = {}
for kind in ("oop_scalar", "oop_spacer", "oop_both", "slant_both"):
    for M in (4, 5, 6, 7):
        key = f"{kind}_M{M}"
        try:
            rows, clo, r00 = _run(F.build(kind, M))
        except Exception as exc:                            # noqa: BLE001
            H[key] = {"error": f"{type(exc).__name__}: {str(exc)[:110]}"}
            continue
        H[key] = dict(rows=rows, closure=clo, R00=r00)
        for r in rows:
            _log(f"  {key:20s} n={r['n']:5d} rcond {r['rcond']:.4e}  full "
                 f"{r['resid_full']:.3e}  probe {r['resid_probe']:.3e}")
for M in (4, 5, 6):
    key = f"both_promoted_M{M}"
    rows, clo, r00 = _run(_both_promoted_stack(M))
    H[key] = dict(rows=rows, closure=clo, R00=r00)
    for r in rows:
        _log(f"  {key:20s} n={r['n']:5d} rcond {r['rcond']:.4e}  full "
             f"{r['resid_full']:.3e}  probe {r['resid_probe']:.3e}")
# one M = 8 point, to show the residual does NOT degrade with the modal ladder
try:
    rows, clo, r00 = _run(F.build("oop_scalar", 8))
    H["oop_scalar_M8"] = dict(rows=rows, closure=clo, R00=r00)
    for r in rows:
        _log(f"  {'oop_scalar_M8':20s} n={r['n']:5d} rcond {r['rcond']:.4e}  "
             f"full {r['resid_full']:.3e}  probe {r['resid_probe']:.3e}")
except Exception as exc:                                    # noqa: BLE001
    H["oop_scalar_M8"] = {"error": f"{type(exc).__name__}: {str(exc)[:110]}"}
    _log(f"  oop_scalar_M8  {H['oop_scalar_M8']['error'][:100]}")
RES["healthy"] = H

# ---- 2. SLIVER (width contract LIFTED) --------------------------------
_log("SLIVER population, width contract LIFTED")
S = {}
_prev = _ts.PMM2D_STAG_MIN_SEG_GUARD
_ts.PMM2D_STAG_MIN_SEG_GUARD = False
try:
    for M in (4, 5):
        for delta in (3e-1, 1e-2, 1e-3, 1e-5, 1e-7, 1e-9, 1e-11):
            key = f"sliver_M{M}_d{delta:.0e}"
            try:
                rows, clo, r00 = _run(_sliver_stack(delta, M))
            except Exception as exc:                        # noqa: BLE001
                S[key] = {"error": f"{type(exc).__name__}: {str(exc)[:110]}"}
                _log(f"  {key:22s} {S[key]['error'][:90]}")
                continue
            S[key] = dict(rows=rows, closure=clo, R00=r00)
            _log(f"  {key:22s} worst rcond "
                 f"{min(r['rcond'] for r in rows):.4e}  worst full "
                 f"{max(r['resid_full'] for r in rows):.3e}  worst probe "
                 f"{max(r['resid_probe'] for r in rows):.3e}  R00 {r00:.10f}")
finally:
    _ts.PMM2D_STAG_MIN_SEG_GUARD = _prev
RES["sliver"] = S

# ---- 3. SINGULAR / INCONSISTENT synthetics ----------------------------
_log("SINGULAR / INCONSISTENT synthetics on a REAL operand of the site")
grab = {}
_o = _pc._guarded_mortar_solve


def _grab(A, B, site, ga=None, gb=None, hint=None, screen="rcond"):
    if "GENERALIZED" in site and "op" not in grab:
        grab["op"] = (np.array(A), np.array(B))
    return _o(A, B, site, ga, gb, hint, screen)


_pc._guarded_mortar_solve = _grab
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        F.build("oop_both", 4).solve(jones=False)
finally:
    _pc._guarded_mortar_solve = _o
A0, B0 = grab["op"]
K = {}
K["real"] = _measure(A0, B0)
A1 = A0.copy()
A1[:, 3] = A1[:, 7]                       # EXACTLY singular: repeated column
K["exact_singular_realB"] = _measure(A1, B0)
rng = np.random.default_rng(20260911)
Z = (rng.standard_normal(B0.shape) + 1j * rng.standard_normal(B0.shape))
K["rank_def_consistent"] = _measure(A1, A1 @ Z)
U1, _s1, _V1 = np.linalg.svd(A1)
cons = A1 @ Z
for scale, name in ((1.0, "1"), (1e-6, "1e-6"), (1e-10, "1e-10"),
                    (1e-13, "1e-13")):
    bad = cons + scale * float(np.linalg.norm(cons)) * np.outer(
        U1[:, -1], np.ones(B0.shape[1]) / np.sqrt(B0.shape[1]))
    K[f"inconsistent_{name}"] = _measure(A1, bad)
# a ZERO row/column (the shape a collapsed grid produces)
A2 = A0.copy()
A2[:, 11] = 0.0
K["zero_column"] = _measure(A2, B0)
for k, v in K.items():
    _log(f"  {k:24s} rcond {v['rcond']:.4e}  full {v['resid_full']:.4e}  "
         f"probe {v['resid_probe']:.4e}")
RES["synthetic"] = K

# ---- 4. COST ----------------------------------------------------------
_log("COST of the screen")
COST = {}
for n in (324, 576, 900, 1296):
    rg = np.random.default_rng(7)
    A = (rg.standard_normal((n, n)) + 1j * rg.standard_normal((n, n)))
    B = (rg.standard_normal((n, n)) + 1j * rg.standard_normal((n, n)))
    reps = 3

    def _t(fn):
        fn()
        t = time.perf_counter()
        for _ in range(reps):
            fn()
        return (time.perf_counter() - t) / reps

    def bare():
        lu, piv = sla.lu_factor(A)
        return sla.lu_solve((lu, piv), B)

    def with_gecon():
        lu, piv = sla.lu_factor(A)
        an = float(np.max(np.sum(np.abs(A), axis=0)))
        g = sla.get_lapack_funcs("gecon", (A,))
        g(lu, an)
        return sla.lu_solve((lu, piv), B)

    def with_probe():
        X = with_gecon()
        v = _probe_vec(B.shape[1])
        bv = B @ v
        float(np.linalg.norm(A @ (X @ v) - bv) / np.linalg.norm(bv))
        return X

    def with_full():
        X = with_gecon()
        float(np.linalg.norm(A @ X - B) / np.linalg.norm(B))
        return X

    tb, tg, tp, tf = _t(bare), _t(with_gecon), _t(with_probe), _t(with_full)
    COST[n] = dict(bare=tb, gecon=tg, probe=tp, full=tf,
                   x_probe=tp / tb, x_full=tf / tb, x_gecon=tg / tb)
    _log(f"  n={n:5d}  bare {tb * 1e3:8.1f} ms  +gecon {tg / tb:5.3f}x  "
         f"+probe {tp / tb:5.3f}x  +full-GEMM {tf / tb:5.3f}x")
RES["cost"] = COST

# ---- summary ----------------------------------------------------------
hr = [r for v in RES["healthy"].values() if v.get("rows") for r in v["rows"]]
sr = [r for v in RES["sliver"].values() if v.get("rows") for r in v["rows"]]
_log(f"HEALTHY {len(hr)} solves: rcond "
     f"{min(r['rcond'] for r in hr):.4e} .. {max(r['rcond'] for r in hr):.4e}"
     f"  full {min(r['resid_full'] for r in hr):.3e} .. "
     f"{max(r['resid_full'] for r in hr):.3e}"
     f"  probe {min(r['resid_probe'] for r in hr):.3e} .. "
     f"{max(r['resid_probe'] for r in hr):.3e}")
_log(f"SLIVER  {len(sr)} solves: rcond "
     f"{min(r['rcond'] for r in sr):.4e} .. {max(r['rcond'] for r in sr):.4e}"
     f"  full {min(r['resid_full'] for r in sr):.3e} .. "
     f"{max(r['resid_full'] for r in sr):.3e}"
     f"  probe {min(r['resid_probe'] for r in sr):.3e} .. "
     f"{max(r['resid_probe'] for r in sr):.3e}")
RES["summary"] = dict(
    healthy_n=len(hr),
    healthy_rcond=[min(r["rcond"] for r in hr), max(r["rcond"] for r in hr)],
    healthy_full=[min(r["resid_full"] for r in hr),
                  max(r["resid_full"] for r in hr)],
    healthy_probe=[min(r["resid_probe"] for r in hr),
                   max(r["resid_probe"] for r in hr)],
    sliver_n=len(sr),
    sliver_full=[min(r["resid_full"] for r in sr),
                 max(r["resid_full"] for r in sr)],
    sliver_probe=[min(r["resid_probe"] for r in sr),
                  max(r["resid_probe"] for r in sr)])
p = os.path.join(_HERE, f"r3_residual_screen_{TAG}.json")
with open(p, "w") as fh:
    json.dump({"tag": TAG, "lumenairy": lumenairy.__file__, **RES}, fh,
              indent=1, default=str)
_log(f"wrote {p}")
