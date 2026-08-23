"""Adjudicate the candidate deterministic-Gram routes BY MEASUREMENT.

Routes:
  blas   G = A.T @ A ; rhs = A.T @ B                (shipped, nondeterministic)
  ufunc  the SHIPPED _det_normal_equations kernel   (np.multiply + np.sum,
         upper triangle only -- symmetry is already exploited)
  eins   per-block np.einsum('ri,rj->ij', T, T, optimize=False)
  einsT  per-block einsum on the TRANSPOSED tile ('ir,jr->ij')

Every route uses the SAME block partition (_det_block_rows) and the SAME
carry-stack pairwise tree, so only the per-block partial differs.
"""
import hashlib
import os
import sys
import time

import numpy as np

import lumenairy as la
from lumenairy.elements import _lens_traced as LT

_want = os.path.realpath(os.path.join(os.environ['LUMENAIRY_ROOT'],
                                      'lumenairy'))
_got = os.path.realpath(os.path.dirname(la.__file__))
assert _got == _want, 'imported %r, expected %r' % (_got, _want)


def _h(a):
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()[:10]


def _tree(parts_g, parts_r):
    """The shipped carry-stack fold, factored so every route shares it."""
    G = parts_g[-1]
    R = parts_r[-1]
    for i in range(len(parts_g) - 2, -1, -1):
        G = parts_g[i] + G
        R = parts_r[i] + R
    return G, R


def _carry(blocks):
    """Reproduce _det_normal_equations' carry-stack exactly."""
    stack = []
    for Gp, rp in blocks:
        stack.append((0, Gp, rp))
        while len(stack) >= 2 and stack[-1][0] == stack[-2][0]:
            d, G2, r2 = stack.pop()
            _, G1, r1 = stack.pop()
            stack.append((d + 1, G1 + G2, r1 + r2))
    G, R = stack[-1][1], stack[-1][2]
    for i in range(len(stack) - 2, -1, -1):
        G = stack[i][1] + G
        R = stack[i][2] + R
    return G, R


def r_blas(A, B):
    return A.T @ A, A.T @ B


def r_ufunc(A, B):
    return LT._det_normal_equations(A, B)


def _blocked(A, B, partial):
    A = np.ascontiguousarray(A, dtype=np.float64)
    Bm = np.asarray(B, dtype=np.float64)
    flat = (Bm.ndim == 1)
    Bm = Bm.reshape(Bm.shape[0], -1)
    n, M = A.shape
    blk = LT._det_block_rows(M)
    out = []
    for i0 in range(0, n, blk):
        i1 = min(i0 + blk, n)
        out.append(partial(A[i0:i1], Bm[i0:i1]))
    G, R = _carry(out)
    return G, (R.ravel() if flat else R)


def r_eins(A, B):
    def p(a, b):
        a = np.ascontiguousarray(a)
        b = np.ascontiguousarray(b)
        return (np.einsum('ri,rj->ij', a, a, optimize=False),
                np.einsum('ri,rk->ik', a, b, optimize=False))
    return _blocked(A, B, p)


def r_einsT(A, B):
    def p(a, b):
        T = np.ascontiguousarray(a.T)
        R = np.ascontiguousarray(b.T)
        return (np.einsum('ir,jr->ij', T, T, optimize=False),
                np.einsum('ir,kr->ik', T, R, optimize=False))
    return _blocked(A, B, p)


ROUTES = {'blas': r_blas, 'ufunc': r_ufunc, 'eins': r_eins, 'einsT': r_einsT}


def bench(fn, A, B, reps=3):
    best = float('inf')
    out = None
    for _ in range(reps):
        t = time.perf_counter()
        out = fn(A, B)
        best = min(best, time.perf_counter() - t)
    return best, out


def main():
    w = os.environ.get('OMP_NUM_THREADS')
    rng = np.random.default_rng(20260823)
    shapes = [(1337, 120, 3), (141471, 66, 1), (1000000, 120, 3),
              (119936, 5, 1)]
    if len(sys.argv) > 1 and sys.argv[1] == 'small':
        shapes = [(1337, 120, 3), (119936, 5, 1)]
    for n, M, K in shapes:
        A = np.ascontiguousarray(rng.normal(size=(n, M)) * 1e-2)
        B = np.ascontiguousarray(rng.normal(size=(n, K)))
        if K == 1:
            B = np.ascontiguousarray(B[:, 0])
        print('SHAPE n=%d M=%d K=%d  blk=%d  OMP=%s'
              % (n, M, K, LT._det_block_rows(M), w))
        for name, fn in ROUTES.items():
            try:
                t, (G, R) = bench(fn, A, B)
            except Exception as exc:                     # noqa: BLE001
                print('  %-6s FAILED %s' % (name, exc))
                continue
            print('  %-6s %8.4f s   G=%s rhs=%s' % (name, t, _h(G), _h(R)))
        sys.stdout.flush()


if __name__ == '__main__':
    main()
