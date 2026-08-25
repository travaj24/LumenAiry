"""Two INDEPENDENT least-squares oracles for the C13 screen adjudication.

Both answer the same question -- what IS the least-squares solution of
``min ||b - A x||`` for a float64 ``A``, ``b``, to full working precision --
and they are built on different machinery so that agreeing proves each.

* :func:`ls_oracle_refine` -- Householder QR in float64 plus iterative
  refinement whose RESIDUAL is computed by the D14 two-product + ``math.fsum``
  route (exact products, correctly-rounded sum).  This is the extra-precise
  residual refinement of Bjorck; it converges to the working-precision LS
  solution while ``cond(A)^2 eps`` is small, which is the regime every fit here
  sits in.  The correction norm is returned per step so convergence is
  MEASURED, not assumed.
* :func:`ls_oracle_mp` -- the normal equations formed and solved at 60 decimal
  digits in ``mpmath``.  ``G = A^T A`` and ``A^T b`` are exact mathematics; the
  normal equations are only a NUMERICAL hazard, and 60 digits against a
  ``cond(G)`` of 1e10 leaves 50 digits of margin, so this route's answer is the
  exact one to far beyond float64.

Neither uses ``numpy.linalg.lstsq``, the Gram, or anything the library's
solver does, so a disagreement between them and the library is the library's.
"""
import math

import numpy as np

_SPLIT = float(2 ** 27 + 1)


def _two_product(a, b):
    """Exact ``a * b = hi + lo`` for float64 (Dekker; no FMA required).

    Same construction as ``tests/unit/test_niche_d14_deterministic_carrier_fit
    .py::_two_product``, which has its own ``test_the_oracle_is_an_oracle``.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    hi = a * b
    ca = _SPLIT * a
    a_hi = ca - (ca - a)
    a_lo = a - a_hi
    cb = _SPLIT * b
    b_hi = cb - (cb - b)
    b_lo = b - b_hi
    lo = a_lo * b_lo - (((hi - a_hi * b_hi) - a_lo * b_hi) - a_hi * b_lo)
    return hi, lo


def exact_residual(A, b, x):
    """``b - A x`` with every product exact and every row sum correctly
    rounded -- i.e. the row's exact value, rounded ONCE."""
    A = np.ascontiguousarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).ravel()
    x = np.asarray(x, dtype=np.float64).ravel()
    hi, lo = _two_product(A, -x[None, :])
    out = np.empty(A.shape[0], dtype=np.float64)
    for i in range(A.shape[0]):
        out[i] = math.fsum([b[i]] + hi[i].tolist() + lo[i].tolist())
    return out


def _two_sum(a, b):
    """Exact ``a + b = s + e`` for float64 (Knuth; no ordering assumption)."""
    s = a + b
    bb = s - a
    e = (a - (s - bb)) + (b - bb)
    return s, e


def exact_residual_dd(A, b, x):
    """``b - A x`` accumulated in DOUBLE-DOUBLE, vectorised over rows.

    Same exact products as :func:`exact_residual`; the row sum is a
    compensated (two-sum) accumulation instead of ``math.fsum``, so it is one
    vectorised pass per column rather than one Python ``fsum`` per row.  Not
    correctly rounded in the last bit, but its error is ``O(eps^2)`` where the
    residual's own is ``O(eps)`` -- proved against the ``fsum`` route on the
    small fits by ``adjudicate.py --mp`` (see ``oracle_gap_dd``).
    """
    A = np.ascontiguousarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).ravel()
    x = np.asarray(x, dtype=np.float64).ravel()
    hi, lo = _two_product(A, -x[None, :])
    s = b.copy()
    e = np.zeros_like(s)
    for j in range(A.shape[1]):
        s, e1 = _two_sum(s, hi[:, j])
        e += e1
        s, e2 = _two_sum(s, lo[:, j])
        e += e2
    return s + e


def _qr_factors(A):
    from scipy.linalg import qr
    return qr(np.asfortranarray(A, dtype=np.float64), mode='economic')


def ls_oracle_refine(A, b, steps=4, residual='auto'):
    """QR + extra-precise-residual iterative refinement.  Returns
    ``(x, [correction norms])``.

    ``residual='fsum'`` uses the correctly-rounded row sums; ``'dd'`` the
    vectorised double-double ones; ``'auto'`` (default) picks ``fsum`` while
    the Python loop is affordable (``n * M <= 2e5``) and ``dd`` above it.
    """
    from scipy.linalg import solve_triangular
    A = np.ascontiguousarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).ravel()
    if residual == 'auto':
        residual = ('fsum' if A.shape[0] * A.shape[1] <= 200_000 else 'dd')
    _res = exact_residual if residual == 'fsum' else exact_residual_dd
    Q, R = _qr_factors(A)
    x = solve_triangular(R, Q.T @ b, check_finite=False)
    corr = []
    for _ in range(steps):
        r = _res(A, b, x)
        d = solve_triangular(R, Q.T @ r, check_finite=False)
        x = x + d
        corr.append(float(np.max(np.abs(d))))
    return x, corr


def ls_oracle_mp(A, b, dps=60):
    """Normal equations formed and solved at ``dps`` decimal digits."""
    from mpmath import mp
    mp.dps = dps
    A = np.ascontiguousarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).ravel()
    n, m = A.shape
    cols = [[mp.mpf(float(v)) for v in A[:, j]] for j in range(m)]
    bb = [mp.mpf(float(v)) for v in b]
    G = mp.matrix(m, m)
    rhs = mp.matrix(m, 1)
    for i in range(m):
        for j in range(i, m):
            s = mp.fdot(zip(cols[i], cols[j]))
            G[i, j] = s
            G[j, i] = s
        rhs[i] = mp.fdot(zip(cols[i], bb))
    x = mp.lu_solve(G, rhs)
    return np.array([float(v) for v in x], dtype=np.float64)
