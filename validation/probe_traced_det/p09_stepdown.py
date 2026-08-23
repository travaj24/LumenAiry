"""The C13 step-down, adjudicated.

MEASURED (p05/p08): on the traced chain EVERY non-carrier fit screens
numerically singular (Gram rcond 1.6e-9 at 28 terms, 9.6e-11 at 120, against
the 1e-8 screen) and therefore takes ``_solve_lstsq_qr`` -- a threaded
``dgeqrf`` over the full design matrix.  A deterministic GRAM alone therefore
changes nothing on that path: D14's declared "one hole" is the DEFAULT there,
not a corner.

Candidate replacement, all-deterministic: normal equations + ONE step of
iterative refinement (the textbook cure for the ``cond(A)^2`` loss, valid
while ``cond(A)^2 eps << 1`` -- here ``1/rcond`` = 1e10, eps = 2.2e-16,
product 2e-6).  This measures its residual against QR's on the fits' OWN
matrices, which is the only bar that matters: the step-down exists to lower
``||b - A x||``.
"""
import hashlib
import os
import sys
import time
import warnings

import numpy as np

import lumenairy as la
from lumenairy.elements import _lens_traced as LT

_want = os.path.realpath(os.path.join(os.environ['LUMENAIRY_ROOT'],
                                      'lumenairy'))
assert os.path.realpath(os.path.dirname(la.__file__)) == _want, la.__file__

LAM = 1.31e-6


def _h(a):
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()[:16]


def _det_matvec(A, X):
    return np.einsum('rj,jk->rk', A, X, optimize=False)


def _det_atb(A, B):
    """``A^T B`` alone, through the same fixed block tree as the Gram."""
    blk = int(LT._DET_EINSUM_BLOCK_ROWS)
    n = A.shape[0]
    stack = []
    for i0 in range(0, n, blk):
        i1 = min(i0 + blk, n)
        p = np.einsum('ri,rk->ik', A[i0:i1], B[i0:i1], optimize=False)
        stack.append((0, p))
        while len(stack) >= 2 and stack[-1][0] == stack[-2][0]:
            d, p2 = stack.pop()
            _, p1 = stack.pop()
            stack.append((d + 1, p1 + p2))
    out = stack[-1][1]
    for i in range(len(stack) - 2, -1, -1):
        out = stack[i][1] + out
    return out


def refined(A, b, steps=1):
    from scipy.linalg import cho_factor, cho_solve
    A = np.ascontiguousarray(A, dtype=np.float64)
    B = np.ascontiguousarray(np.asarray(b, dtype=np.float64))
    flat = (B.ndim == 1)
    B2 = B.reshape(B.shape[0], -1)
    G, rhs = LT._det_normal_equations(A, B2)
    cf = cho_factor(G, check_finite=False)
    x = cho_solve(cf, rhs, check_finite=False)
    for _ in range(steps):
        r = B2 - _det_matvec(A, x)
        x = x + cho_solve(cf, _det_atb(A, r), check_finite=False)
    return x.ravel() if flat else x


def resid(A, b, x):
    return float(np.linalg.norm(np.asarray(b) - np.asarray(A) @ np.asarray(x)))


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    DX = 30e-6
    sub = (None if len(sys.argv) > 2 and sys.argv[2] == 'none'
           else (int(sys.argv[2]) if len(sys.argv) > 2 else 2))
    x = (np.arange(N) - N / 2) * DX
    X, Y = np.meshgrid(x, x)
    w = int(0.15625 * N) * DX
    E = (np.exp(-(X * X + Y * Y) / (w * w))
         * np.exp(1j * (2 * np.pi / LAM) * (X * X + Y * Y) / 2.0)
         ).astype(np.complex128)
    presc = {'wavelength': LAM, 'aperture_diameter': 14e-3, 'surfaces': [
        {'radius': 51.68e-3, 'thickness': 4e-3, 'glass_before': 'air',
         'glass_after': 'N-BK7', 'semi_diameter': 7e-3},
        {'radius': -51.68e-3, 'thickness': 0.0, 'glass_before': 'N-BK7',
         'glass_after': 'air', 'semi_diameter': 7e-3}],
        'thicknesses': [4e-3], 'stop_index': 0}

    cap = []
    orig = LT._solve_lstsq_thread_safe

    def spy(A, b, deterministic=False):
        cap.append((np.ascontiguousarray(A, np.float64),
                    np.array(b, dtype=np.float64)))
        return orig(A, b, deterministic=deterministic)

    LT._solve_lstsq_thread_safe = spy
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            la.apply_real_lens_traced(E, prescription=presc, wavelength=LAM,
                                      dx=DX, carrier='auto',
                                      on_undersample='silent',
                                      **({} if sub is None
                                         else {'ray_subsample': sub}))
    finally:
        LT._solve_lstsq_thread_safe = orig

    print('OMP=%s  captured %d fits' % (os.environ.get('OMP_NUM_THREADS'),
                                        len(cap)))
    for A, b in cap:
        if A.shape[1] < 8:
            continue
        G, rhs = LT._det_normal_equations(A, b)
        rc = LT._gram_rcond(G)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            x_qr = LT._solve_lstsq_qr(A, b)
            x_ne = orig(A, b, deterministic=True)
        t0 = time.perf_counter()
        x_r1 = refined(A, b, 1)
        t1 = time.perf_counter() - t0
        x_r2 = refined(A, b, 2)
        t0 = time.perf_counter()
        LT._solve_lstsq_qr(A, b)
        t_qr = time.perf_counter() - t0
        t0 = time.perf_counter()
        LT._det_normal_equations(A, b)
        t_ne = time.perf_counter() - t0
        pk = float(np.max(np.abs(x_qr)))
        print('  A=%s rcond=%.3e' % (A.shape, rc))
        print('    resid   qr %.12e  ne(det) %.12e  ref1 %.12e  ref2 %.12e'
              % (resid(A, b, x_qr), resid(A, b, x_ne),
                 resid(A, b, x_r1), resid(A, b, x_r2)))
        print('    |x-x_qr|/peak   ne %.3e  ref1 %.3e  ref2 %.3e'
              % (float(np.max(np.abs(x_ne - x_qr))) / pk,
                 float(np.max(np.abs(x_r1 - x_qr))) / pk,
                 float(np.max(np.abs(x_r2 - x_qr))) / pk))
        print('    time    qr %.4f s  detNE %.4f s  ref1 %.4f s' %
              (t_qr, t_ne, t1))
        print('    hash    qr %s  ref1 %s  ref2 %s'
              % (_h(x_qr), _h(x_r1), _h(x_r2)))
        sys.stdout.flush()


if __name__ == '__main__':
    main()
