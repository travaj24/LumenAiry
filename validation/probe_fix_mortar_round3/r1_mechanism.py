"""ROUND 3, DEFECT V1 -- the MECHANISM of the generalized mortar site's
near-null space, measured to closure.

The VERIFY audit (S6.3) refused ``_MORTAR_RCOND_REFUSE = 1e-12`` at
``_interface_smatrix_general_mortar_2d`` as a FALSE POSITIVE and gave a reading
of the mechanism it did NOT measure: "when one side of a generalized mortar is
an in-plane region promoted to the 6-tuple general form, its forward/backward
mode sets are related by +/-q symmetry, so the ``[[E1, E2], [H1, H2]]`` block
acquires a near-null space BY CONSTRUCTION, and the RHS lies in its range".

This probe measures exactly that, on the v8 fixture and its neighbours:

  * the SVD of every generalized-site operand, its numerical rank and the
    dimension of the near-null space, at ``M`` = 4..7;
  * WHICH side the near-null right vectors live on -- the block-column split
    ``||v_a|| / ||v||`` vs ``||v_b|| / ||v||`` -- and whether that side is the
    PROMOTED in-plane one (detected structurally: ``Wb is Wf`` and
    ``Vb == -Vf`` exactly, which is what :func:`_modes_as_general` writes);
  * the LEFT null space and the component of the right-hand side along it,
    ``||U_null^H B|| / ||B||`` -- the range-membership test;
  * the RESIDUAL ``||A X - B|| / ||B||`` of the answer ``lu_solve`` returns.

``python r1_mechanism.py [win|wsl]``
"""
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
# the WORKTREE's lumenairy, not whatever else is on PYTHONPATH: asserted below
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
from lumenairy.elements.pmm import stack2d_pure as _sp  # noqa: E402

HERE = _HERE
assert os.path.abspath(lumenairy.__file__).lower().startswith(_ROOT.lower()), (
    f"wrong lumenairy: {lumenairy.__file__} (want one under {_ROOT})")
TAG = (sys.argv[1] if len(sys.argv) > 1 else "win")
T0 = time.time()
print(f"[arm {TAG}] lumenairy = {lumenairy.__file__} v{lumenairy.__version__}",
      flush=True)


def _log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def _promoted(six):
    """Is this 6-tuple a SYMMETRIC in-plane region promoted by
    :func:`_modes_as_general` (``(W, V, lam, W, -V, -lam)``)?  Structural, and
    exact: the promotion shares ``W`` by identity and negates ``V`` bitwise."""
    Wf, Vf, lf, Wb, Vb, lb = six[:6]
    return bool(Wf is Wb or (Wf.shape == Wb.shape
                             and np.array_equal(Wf, Wb))) and bool(
        Vf.shape == Vb.shape and np.array_equal(Vf, -Vb))


def _analyse(A, B, meta):
    """SVD / rank / null space / residual / range membership of ONE operand."""
    A = np.asarray(A)
    n = A.shape[0]
    U, s, Vh = np.linalg.svd(A)
    eps = np.finfo(float).eps
    tol = max(A.shape) * eps * float(s[0])
    rank = int(np.sum(s > tol))
    ndef = n - rank
    # the answer the shipped site returns (guard lifted): lu_factor + lu_solve
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lu, piv = sla.lu_factor(A)
        X = sla.lu_solve((lu, piv), B)
    nB = float(np.linalg.norm(B))
    res = float(np.linalg.norm(A @ X - B)) / nB
    # RANGE MEMBERSHIP: the component of every RHS column along the LEFT null
    # space.  A CONSISTENT system has none.
    k = max(ndef, 1)
    left = U[:, n - k:]                 # the k smallest left singular vectors
    comp = float(np.linalg.norm(left.conj().T @ B)) / nB
    # WHERE the near-null right vectors live: split by block column.
    ma = int(meta["ma"])
    right = Vh[n - k:].conj().T         # (n, k)
    va = float(np.linalg.norm(right[:ma])) / float(np.linalg.norm(right))
    vb = float(np.linalg.norm(right[ma:])) / float(np.linalg.norm(right))
    # and, for the +/-q hypothesis, the block-column RANKS: is the deficiency
    # inside ONE side's columns?
    sa = np.linalg.svd(A[:, :ma], compute_uv=False)
    sb = np.linalg.svd(A[:, ma:], compute_uv=False)
    rk_a = int(np.sum(sa > max(A.shape) * eps * float(sa[0])))
    rk_b = int(np.sum(sb > max(A.shape) * eps * float(sb[0])))
    # LAPACK's own estimate, exactly as the shipped guard reads it
    anorm = float(np.max(np.sum(np.abs(A), axis=0)))
    gecon = sla.get_lapack_funcs("gecon", (A,))
    rcv, info = gecon(lu, anorm)
    rc = float(rcv) if int(info) == 0 else 0.0
    return dict(n=n, ma=ma, mb=n - ma, rank=rank, null_dim=ndef,
                s_max=float(s[0]), s_min=float(s[-1]),
                s_tail=[float(x) for x in s[-6:]],
                s_ratio=float(s[-1] / s[0]), tol=float(tol),
                rcond_gecon=rc, residual=res, range_defect=comp,
                null_on_a=va, null_on_b=vb,
                rank_a=rk_a, rank_b=rk_b, cols_a=ma, cols_b=n - ma,
                **{k2: meta[k2] for k2 in ("promoted_a", "promoted_b",
                                           "iface")})


def run(kind, M, n_orders=2):
    """Solve one fixture with the guard LIFTED, analysing every generalized
    mortar operand the stack builds."""
    rows = []
    state = {"iface": -1, "meta": None}
    orig_ifc = _sp._interface_smatrix_general_mortar_2d
    orig_solve = _pc._guarded_mortar_solve

    def ifc(six_a, six_b, ga, gb, cr, kron_apply):
        state["iface"] += 1
        state["meta"] = dict(iface=state["iface"],
                             ma=int(six_a[0].shape[1]),
                             promoted_a=_promoted(six_a),
                             promoted_b=_promoted(six_b))
        return orig_ifc(six_a, six_b, ga, gb, cr, kron_apply)

    def solve(A, B, site, ga=None, gb=None, hint=None, screen="rcond"):
        if "GENERALIZED" not in site:
            return orig_solve(A, B, site, ga, gb, hint, screen)
        rows.append(_analyse(A, B, state["meta"]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lu, piv = sla.lu_factor(np.asarray(A))
            return sla.lu_solve((lu, piv), B)

    _sp._interface_smatrix_general_mortar_2d = ifc
    _pc._guarded_mortar_solve = solve
    try:
        st = F.build(kind, M, n_orders=n_orders)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o, R, T = st.solve(jones=False)
        R2, T2 = np.atleast_2d(R), np.atleast_2d(T)
        clo = float(np.max(np.abs(R2.sum(1) + T2.sum(1) - 1.0)))
        r00 = F.zeroth_R(o, R)
    finally:
        _sp._interface_smatrix_general_mortar_2d = orig_ifc
        _pc._guarded_mortar_solve = orig_solve
    return rows, clo, r00


OUT = {}
for kind in ("oop_scalar", "oop_spacer", "oop_both", "slant_both"):
    for M in (4, 5, 6, 7):
        key = f"{kind}_M{M}"
        try:
            rows, clo, r00 = run(kind, M)
        except Exception as exc:                        # noqa: BLE001
            OUT[key] = {"error": f"{type(exc).__name__}: {str(exc)[:140]}"}
            _log(f"{key:22s} {OUT[key]['error'][:120]}")
            continue
        OUT[key] = dict(rows=rows, closure=clo, R00=r00)
        _log(f"{key:22s} closure {clo:.3e}  R00 {r00:.10f}  "
             f"{len(rows)} generalized solves")
        for r in rows:
            _log(f"    ifc {r['iface']}  n={r['n']}  "
                 f"promoted a/b {int(r['promoted_a'])}/{int(r['promoted_b'])}"
                 f"  rank {r['rank']}/{r['n']} (null {r['null_dim']})"
                 f"  s_min/s_max {r['s_ratio']:.3e}  rcond {r['rcond_gecon']:.3e}"
                 f"  resid {r['residual']:.3e}  range-defect "
                 f"{r['range_defect']:.3e}  null on a/b "
                 f"{r['null_on_a']:.3f}/{r['null_on_b']:.3f}"
                 f"  colrank a {r['rank_a']}/{r['cols_a']} b {r['rank_b']}/"
                 f"{r['cols_b']}")

p = os.path.join(HERE, f"r1_mechanism_{TAG}.json")
with open(p, "w") as fh:
    json.dump({"tag": TAG, "lumenairy": lumenairy.__file__,
               "version": lumenairy.__version__, "rows": OUT}, fh, indent=1)
_log(f"wrote {p}")
