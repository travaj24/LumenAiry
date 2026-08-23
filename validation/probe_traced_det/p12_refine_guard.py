"""Where must the refinement REFUSE?

The D7 hard-mask fixture (test_c13_cures_the_hard_mask_fold_at_the_d7_order)
has an equilibrated Gram that has LOST POSITIVE-DEFINITENESS -- ``_gram_rcond``
returns exactly 0.0 -- while ``cho_factor`` on the raw Gram still succeeds.
Refinement on that system does not converge: measured, it returns a fit
missing the least-squares residual by 1.37e5x.

So the deterministic route needs a REFUSAL RULE, and it has to be
deterministic itself (it cannot score against the QR, whose residual moves
with the thread count).  Two candidates, measured here against both
populations -- the traced fits that must refine, and the singular fixtures
that must not:

  (a) the Gram's own rcond;
  (b) the SIZE OF THE CORRECTION relative to the answer, which is a direct
      measurement of whether refinement converged rather than a proxy for it.
"""
import os
import sys
import warnings

import numpy as np

import lumenairy as la
from lumenairy.elements import _lens_traced as LT

_want = os.path.realpath(os.path.join(os.environ['LUMENAIRY_ROOT'],
                                      'lumenairy'))
assert os.path.realpath(os.path.dirname(la.__file__)) == _want, la.__file__

LAM = 1.31e-6


def report(tag, A, b):
    from scipy.linalg import cho_factor, cho_solve
    A = np.ascontiguousarray(A, dtype=np.float64)
    B = np.ascontiguousarray(np.asarray(b, dtype=np.float64))
    flat = (B.ndim == 1)
    B2 = B.reshape(B.shape[0], -1)
    G, rhs = LT._det_normal_equations(A, B2)
    rc = LT._gram_rcond(G)
    try:
        cf = cho_factor(G, check_finite=False)
        x0 = cho_solve(cf, rhs, check_finite=False)
        chol = True
    except Exception:                                    # noqa: BLE001
        chol = False
        try:
            x0 = np.linalg.solve(G, rhs)

            def cho_solve(_c, _r, check_finite=False):   # noqa: F811
                return np.linalg.solve(G, _r)
            cf = None
        except np.linalg.LinAlgError:
            print('  %-26s %-14s NO FACTORISATION AT ALL -- the existing '
                  'rank-deficient exit already catches this'
                  % (tag, str(A.shape)))
            return
    d = cho_solve(cf, LT._det_at_b(A, B2 - LT._det_matvec(A, x0)),
                  check_finite=False)
    x1 = x0 + d
    ratio = (float(np.max(np.abs(d)))
             / max(float(np.max(np.abs(x0))), 1e-300))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        xq = LT._solve_lstsq_qr(A, b)
    r0 = LT._lstsq_residual(A, B, x0.ravel() if flat else x0)
    r1 = LT._lstsq_residual(A, B, x1.ravel() if flat else x1)
    rq = LT._lstsq_residual(A, B, xq)
    print('  %-26s %-14s chol=%d rcond=%.3e  |d|/|x0|=%.3e  '
          'resid ne/ref/qr = %.4gx / %.4gx / 1'
          % (tag, str(A.shape), chol, rc, ratio,
             (r0 / rq if rq > 0 else np.inf),
             (r1 / rq if rq > 0 else np.inf)))
    sys.stdout.flush()


def traced_fits(sub):
    N, dx = 512, 30e-6
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    w = 80 * dx
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
                                      dx=dx, carrier='auto',
                                      on_undersample='silent',
                                      **({} if sub is None
                                         else {'ray_subsample': sub}))
    finally:
        LT._solve_lstsq_thread_safe = orig
    return cap


print('MUST REFINE -- the traced chain\'s own fits')
for sub in (None, 2):
    for A, b in traced_fits(sub):
        report('traced sub=%s' % sub, A, b)

print('MUST REFUSE -- engineered / hard-mask singular systems')
rng = np.random.default_rng(1234)
n = 12_000
col = rng.normal(size=n)
b = np.ascontiguousarray(rng.normal(size=n))
report('duplicated column', np.ascontiguousarray(
    np.stack([np.ones(n), col, col, rng.normal(size=n)], axis=1)), b)
report('col perturbed 1e-6', np.ascontiguousarray(
    np.stack([np.ones(n), col, col + 1e-6 * rng.normal(size=n),
              rng.normal(size=n)], axis=1)), b)
report('col perturbed 1e-9', np.ascontiguousarray(
    np.stack([np.ones(n), col, col + 1e-9 * rng.normal(size=n),
              rng.normal(size=n)], axis=1)), b)
report('col perturbed 1e-12', np.ascontiguousarray(
    np.stack([np.ones(n), col, col + 1e-12 * rng.normal(size=n),
              rng.normal(size=n)], axis=1)), b)
