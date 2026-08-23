"""(1) Where does the einsum partial overtake the ufunc partial, as a
function of the TERM COUNT alone?  (2) Oracle accuracy of both against a
correctly-rounded reference and against the legal-partition family.

The oracle machinery is the D14 one, imported from the niche's own test
module rather than re-derived -- a second copy of an instrument is a second
instrument.
"""
import importlib.util
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

_spec = importlib.util.spec_from_file_location(
    '_d14', os.path.join(os.environ['LUMENAIRY_ROOT'], 'tests', 'unit',
                         'test_niche_d14_deterministic_carrier_fit.py'))
_d14 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_d14)


def _carry(blocks):
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


def det_einsum(A, b):
    A = np.ascontiguousarray(A, dtype=np.float64)
    B = np.ascontiguousarray(np.asarray(b, dtype=np.float64))
    flat = (B.ndim == 1)
    B = B.reshape(B.shape[0], -1)
    n, M = A.shape
    blk = LT._det_block_rows(M)
    out = []
    for i0 in range(0, n, blk):
        i1 = min(i0 + blk, n)
        a = A[i0:i1]
        bb = B[i0:i1]
        out.append((np.einsum('ri,rj->ij', a, a, optimize=False),
                    np.einsum('ri,rk->ik', a, bb, optimize=False)))
    G, R = _carry(out)
    return G, (R.ravel() if flat else R)


def crossover():
    rng = np.random.default_rng(99)
    n = 200_000
    print('CROSSOVER  n=%d  OMP=%s' % (n, os.environ.get('OMP_NUM_THREADS')))
    print('  %-5s %-7s %10s %10s %10s %8s' %
          ('M', 'blk', 'blas', 'ufunc', 'einsum', 'e/u'))
    for M in (5, 6, 8, 10, 12, 16, 20, 28, 40, 66, 120):
        A = np.ascontiguousarray(rng.normal(size=(n, M)) * 1e-2)
        b = np.ascontiguousarray(rng.normal(size=n))
        ts = {}
        for name, fn in (('blas', lambda X, y: (X.T @ X, X.T @ y)),
                         ('ufunc', LT._det_normal_equations),
                         ('einsum', det_einsum)):
            best = float('inf')
            for _ in range(3):
                t0 = time.perf_counter()
                fn(A, b)
                best = min(best, time.perf_counter() - t0)
            ts[name] = best
        print('  %-5d %-7d %10.4f %10.4f %10.4f %8.2fx'
              % (M, LT._det_block_rows(M), ts['blas'], ts['ufunc'],
                 ts['einsum'], ts['einsum'] / ts['ufunc']))
        sys.stdout.flush()


def oracle():
    """Oracle-relative error of ufunc / einsum / the legal-partition family
    at the traced fits' own shapes."""
    rng = np.random.default_rng(20260823)
    print('ORACLE  OMP=%s' % os.environ.get('OMP_NUM_THREADS'))
    for n, M in ((1337, 120), (1457, 28), (20000, 66)):
        # column scales like a Chebyshev / total-degree design: O(1) columns
        A = np.ascontiguousarray(rng.uniform(-1.0, 1.0, size=(n, M)))
        A[:, 0] = 1.0
        b = np.ascontiguousarray(rng.normal(size=n) * 1e-3)
        Go, ro = _d14._oracle_normal_equations(A, b)
        xo = np.linalg.solve(Go, ro)

        def rel(x, ref):
            return float(np.max(np.abs(np.asarray(x) - ref))
                         / np.max(np.abs(ref)))

        def errs(G, r):
            return (rel(G, Go), rel(r, ro), rel(np.linalg.solve(G, r), xo))

        eu = errs(*LT._det_normal_equations(A, b))
        ee = errs(*det_einsum(A, b))
        fam = [errs(*_d14._partitioned_normal_equations(A, b, k))
               for k in _d14._KSPLITS]
        fb = [min(f[i] for f in fam) for i in range(3)]
        fw = [max(f[i] for f in fam) for i in range(3)]
        print('  n=%d M=%d' % (n, M))
        for i, q in enumerate(('G  ', 'rhs', 'coef')):
            print('    %-4s ufunc %.3e  einsum %.3e  family best %.3e '
                  'worst %.3e' % (q, eu[i], ee[i], fb[i], fw[i]))
        sys.stdout.flush()


if __name__ == '__main__':
    if 'oracle' in sys.argv:
        oracle()
    else:
        crossover()
