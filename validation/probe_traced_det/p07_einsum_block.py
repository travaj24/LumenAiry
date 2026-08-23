"""The einsum partial's block length, swept for SPEED and for ACCURACY.

p06 refuted the naive route: an einsum partial over the shipped 4096-row
block is 3-10x LESS accurate than the ufunc partial and 2-6x worse than the
legal-partition family's WORST draw, because einsum accumulates a block
SEQUENTIALLY where ``np.sum`` is pairwise.  Shortening the einsum run and
letting the carry-stack's pairwise tree cover the rest is the obvious repair;
whether it costs speed is the question this answers.
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


def det_einsum(A, b, blk):
    A = np.ascontiguousarray(A, dtype=np.float64)
    B = np.ascontiguousarray(np.asarray(b, dtype=np.float64))
    flat = (B.ndim == 1)
    B = B.reshape(B.shape[0], -1)
    n = A.shape[0]
    out = []
    for i0 in range(0, n, blk):
        i1 = min(i0 + blk, n)
        a = A[i0:i1]
        bb = B[i0:i1]
        out.append((np.einsum('ri,rj->ij', a, a, optimize=False),
                    np.einsum('ri,rk->ik', a, bb, optimize=False)))
    G, R = _carry(out)
    return G, (R.ravel() if flat else R)


BLKS = (32, 64, 128, 256, 512, 1024, 2048, 4096)


def main():
    rng = np.random.default_rng(20260823)
    print('EINSUM BLOCK SWEEP  OMP=%s' % os.environ.get('OMP_NUM_THREADS'))
    for n, M in ((1337, 120), (1457, 28), (200000, 66), (200000, 120)):
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

        fam = [errs(*_d14._partitioned_normal_equations(A, b, k))
               for k in _d14._KSPLITS]
        fw = [max(f[i] for f in fam) for i in range(3)]
        fb = [min(f[i] for f in fam) for i in range(3)]
        eu = errs(*LT._det_normal_equations(A, b))
        t0 = time.perf_counter()
        for _ in range(3):
            LT._det_normal_equations(A, b)
        tu = (time.perf_counter() - t0) / 3
        print('  n=%d M=%d   family best %.2e/%.2e/%.2e  worst '
              '%.2e/%.2e/%.2e' % (n, M, fb[0], fb[1], fb[2],
                                  fw[0], fw[1], fw[2]))
        print('    %-8s %9s  %10s %10s %10s' %
              ('blk', 'time', 'G', 'rhs', 'coef'))
        print('    %-8s %9.4f  %10.3e %10.3e %10.3e'
              % ('ufunc', tu, eu[0], eu[1], eu[2]))
        for blk in BLKS:
            e = errs(*det_einsum(A, b, blk))
            best = float('inf')
            for _ in range(3):
                t0 = time.perf_counter()
                det_einsum(A, b, blk)
                best = min(best, time.perf_counter() - t0)
            print('    %-8d %9.4f  %10.3e %10.3e %10.3e'
                  % (blk, best, e[0], e[1], e[2]))
        sys.stdout.flush()


if __name__ == '__main__':
    main()
