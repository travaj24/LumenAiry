"""Capture every mortar operand a solve builds, WITHOUT the round-3 guard.

The guard is replaced by a recorder that always returns ``np.linalg.solve``,
so a stack the shipped guard would REFUSE still runs to completion and its
operand is available for analysis.  The generalized site's wrapper records,
for each interface, whether each side is an in-plane region PROMOTED by
``_modes_as_general`` -- STRUCTURALLY, not by tolerance: the promotion shares
``W`` by identity and negates ``V`` and ``lam`` bitwise, so the test is exact.
"""
from __future__ import annotations

import contextlib

import numpy as np

from lumenairy.elements.pmm import _core as _pc
from lumenairy.elements.pmm import stack2d_pure as _sp

GEN_SITE = "pmm2d staggered GENERALIZED mortar interface"


def _is_promoted(six):
    """``(W, V, lam, W, -V, -lam)`` exactly, as ``_modes_as_general`` writes
    it.  ``W3 is W0`` is the identity the function itself creates."""
    W, V, lam, W2, V2, lam2 = six[:6]
    if W2 is W and V2 is not None:
        return bool(np.array_equal(V2, -V) and np.array_equal(lam2, -lam))
    return bool(np.array_equal(W2, W) and np.array_equal(V2, -V)
                and np.array_equal(lam2, -lam))


@contextlib.contextmanager
def capture(record_all=False):
    """Yield a list that fills with dicts, one per guarded mortar solve."""
    out = []
    cur = {}
    orig_guard = _pc._guarded_mortar_solve
    orig_gen = _pc._interface_smatrix_general_mortar_2d

    def gen(six_a, six_b, ga, gb, cr, kron_apply):
        cur.clear()
        cur.update(prom_a=_is_promoted(six_a), prom_b=_is_promoted(six_b),
                   ma=int(np.asarray(six_a[0]).shape[1]),
                   mb=int(np.asarray(six_b[0]).shape[1]))
        try:
            return orig_gen(six_a, six_b, ga, gb, cr, kron_apply)
        finally:
            cur.clear()

    def guard(A, B, site, ga=None, gb=None, hint=None, screen="rcond"):
        rec = {"site": site, "n": int(A.shape[0]), "screen": screen,
               "A": np.array(A), "B": np.array(B)}
        rec.update({k: v for k, v in cur.items()})
        if record_all or site == GEN_SITE:
            out.append(rec)
        return np.linalg.solve(A, B)

    _pc._guarded_mortar_solve = guard
    _pc._interface_smatrix_general_mortar_2d = gen
    _sp._interface_smatrix_general_mortar_2d = gen
    try:
        yield out
    finally:
        _pc._guarded_mortar_solve = orig_guard
        _pc._interface_smatrix_general_mortar_2d = orig_gen
        _sp._interface_smatrix_general_mortar_2d = orig_gen


def x_functionals(X, k=12):
    """Deterministic bilinear functionals of the mortar answer, for a
    CROSS-BUILD comparison that does not need the matrix shipped in JSON.

    ``p_i^H X q_i`` for a fixed-seed pair of generic vectors.  Generic on
    purpose: the answer's error is dominated by the near-null direction, and a
    structured functional can be blind to it.
    """
    n = X.shape[0]
    r = np.random.default_rng(0xB0A7)
    out = []
    for _ in range(k):
        p = r.standard_normal(n) + 1j * r.standard_normal(n)
        q = r.standard_normal(X.shape[1]) + 1j * r.standard_normal(X.shape[1])
        z = complex(p.conj() @ (X @ q))
        out.append([z.real, z.imag])
    return out


def svd_facts(A, B, ma=None):
    """SVD, near-null participation, residual, range membership.

    ``ma`` is the width of the FIRST block column (side a's backward mode
    count); it is taken from the interface wrapper rather than assumed to be
    ``n // 2``, because a per-layer stack may give its two neighbours
    different modal counts."""
    n = A.shape[0]
    U, s, Vh = np.linalg.svd(A)
    v = Vh[-1].conj()
    u = U[:, -1]
    half = int(n // 2 if ma is None else ma)
    na = float(np.linalg.norm(v[:half]))
    nb = float(np.linalg.norm(v[half:]))
    X = np.linalg.solve(A, B)
    nB = float(np.linalg.norm(B))
    resid = float(np.linalg.norm(A @ X - B)) / nB if nB else 0.0
    rng = float(np.linalg.norm(u.conj() @ B)) / nB if nB else 0.0
    tol = max(A.shape) * np.finfo(float).eps * float(s[0])
    return {
        "n": int(n),
        "s_max": float(s[0]), "s_min": float(s[-1]),
        "s_ratio": float(s[-1] / s[0]),
        "numerical_rank": int(np.sum(s > tol)),
        "on_a": na, "on_b": nb,
        "residual": resid,
        "range_membership": rng,
        "predicted_rel_spread": (float(np.finfo(float).eps / rng)
                                 if rng > 0 else float("inf")),
        "A_norm": float(np.linalg.norm(A)),
        "B_norm": float(np.linalg.norm(B)),
        "X_norm": float(np.linalg.norm(X)),
        "x_func": x_functionals(X),
    }


def rcond_of(A):
    import warnings

    import scipy.linalg as sla
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            lu, piv = sla.lu_factor(np.array(A, dtype=complex))
        except (ValueError, sla.LinAlgError, np.linalg.LinAlgError):
            return 0.0
    anorm = float(np.max(np.sum(np.abs(A), axis=0)))
    gecon = sla.get_lapack_funcs(("gecon",), (A,))[0]
    rcv, info = gecon(lu, anorm, norm="1")
    return float(rcv) if int(info) == 0 else 0.0
