"""Is np.einsum BLAS-free and single-threaded ON THIS BUILD?

The D14 audit verified the analogous claim for ufunc reductions by hashing
across pinned widths.  Same instrument here, plus two things a hash alone
cannot show:

  * SCALING -- a threaded contraction gets FASTER with the width.  einsum's
    wall time must not.
  * optimize=True -- einsum's optimizer is allowed to route a contraction
    through ``tensordot`` -> BLAS ``dgemm``, which WOULD be threaded.  That
    is measured here so the shipped kernel's explicit ``optimize=False`` is
    a documented necessity rather than a stylistic default.
"""
import hashlib
import os
import time

import numpy as np

import lumenairy as la

_want = os.path.realpath(os.path.join(os.environ['LUMENAIRY_ROOT'],
                                      'lumenairy'))
_got = os.path.realpath(os.path.dirname(la.__file__))
assert _got == _want, 'imported %r, expected %r' % (_got, _want)

W = os.environ.get('OMP_NUM_THREADS')


def _h(a):
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()[:12]


def _t(fn, reps=3):
    best = float('inf')
    out = None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


rng = np.random.default_rng(31415)
for n, M in ((4096, 120), (500000, 120), (2000000, 66)):
    A = np.ascontiguousarray(rng.normal(size=(n, M)) * 1e-2)
    t_no, g_no = _t(lambda: np.einsum('ri,rj->ij', A, A, optimize=False))
    t_op, g_op = _t(lambda: np.einsum('ri,rj->ij', A, A, optimize=True))
    t_bl, g_bl = _t(lambda: A.T @ A)
    print('n=%-8d M=%-4d OMP=%s' % (n, M, W))
    print('   einsum optimize=False %8.4f s  %s' % (t_no, _h(g_no)))
    print('   einsum optimize=True  %8.4f s  %s' % (t_op, _h(g_op)))
    print('   A.T @ A               %8.4f s  %s' % (t_bl, _h(g_bl)))
    print('   optimize=True == BLAS bytes: %s'
          % np.array_equal(g_op, g_bl))
