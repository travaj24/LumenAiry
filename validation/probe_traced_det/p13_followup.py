"""The 5.42.0 gate's three failures, instrumented at BOTH flag states.

(1)/(2) test_fix_runner_oom::test_the_forward_fit_coefficients_are_bit_identical
        -- a same-process TWO-ARM comparison of two design-matrix LAYOUTS
        whose reference arm hard-codes the solver's old default, so D15 moved
        one arm and not the other.
(3)     test_niche_d7::test_a_shrunken_basis_domain_is_a_liability_outside_itself
        -- a liability DEMONSTRATION whose magnitude was calibrated to the
        unrefined solve.

Prints, for each, what each arm reads with DETERMINISTIC_TRACED_FIT off and
on, plus whether the fit reaches the C13 screen at all (which is what decides
whether the refinement can touch it).
"""
import os
import warnings

import numpy as np

import lumenairy as la
from lumenairy.elements import _lens_traced as LT

_want = os.path.realpath(os.path.join(os.environ['LUMENAIRY_ROOT'],
                                      'lumenairy'))
assert os.path.realpath(os.path.dirname(la.__file__)) == _want, la.__file__


def _screen_census(fn):
    """Run ``fn`` and report every solve's Gram rcond and whether the C13
    screen fired (i.e. whether refinement is even reachable)."""
    rows = []
    orig = LT._solve_lstsq_thread_safe

    def spy(A, b, deterministic=False):
        G, _ = (LT._det_normal_equations(A, b) if deterministic
                else (np.ascontiguousarray(A, np.float64).T
                      @ np.ascontiguousarray(A, np.float64), None))
        rc = LT._gram_rcond(G)
        rows.append((np.shape(A), bool(deterministic), float(rc),
                     bool(rc < LT._LSTSQ_GRAM_RCOND_MIN)))
        return orig(A, b, deterministic=deterministic)

    LT._solve_lstsq_thread_safe = spy
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = fn()
    finally:
        LT._solve_lstsq_thread_safe = orig
    return out, rows


# ---------------------------------------------------------------- (1)/(2)
def _retired_cheb_coeffs(xs, ys, values, order, weights=None,
                         deterministic=False):
    mi = [(kx, ky) for kx in range(order + 1) for ky in range(order + 1 - kx)]
    K1 = np.asarray([m[0] for m in mi], dtype=np.int64)
    K2 = np.asarray([m[1] for m in mi], dtype=np.int64)
    xs_np = np.asarray(xs, dtype=np.float64)
    ys_np = np.asarray(ys, dtype=np.float64)
    vals = np.asarray(values, dtype=np.float64)
    xmin, xmax = float(xs_np.min()), float(xs_np.max())
    ymin, ymax = float(ys_np.min()), float(ys_np.max())
    X, Y = np.meshgrid(xs_np, ys_np, indexing='ij')
    u = (2.0 * X - (xmin + xmax)) / (xmax - xmin)
    v = (2.0 * Y - (ymin + ymax)) / (ymax - ymin)
    Tu = LT._cheb_vand_2d(u, order, np)
    Tv = LT._cheb_vand_2d(v, order, np)
    A_full = (Tu[K1] * Tv[K2]).reshape(len(mi), -1).T
    flat = vals.ravel()
    finite = np.isfinite(flat)
    if weights is not None:
        w_flat = np.asarray(weights, dtype=np.float64).ravel()
        keep = finite & np.isfinite(w_flat) & (w_flat > 0.0)
        _all = bool(keep.all())
        A = np.ascontiguousarray(A_full if _all else A_full[keep, :])
        w_keep = w_flat if _all else w_flat[keep]
        A = A * w_keep[:, None]
        rhs = (flat if _all else flat[keep]) * w_keep
    elif finite.all():
        A, rhs = A_full, flat
    else:
        A, rhs = A_full[finite, :], flat[finite]
    return LT._solve_lstsq_thread_safe(A, rhs, deterministic=deterministic)


print('=== (1)/(2) forward-fit layout bit-identity ===')
ax = np.linspace(-1.0e-3, 1.0e-3, 96)
X, Y = np.meshgrid(ax, ax, indexing='ij')
vals = (0.7 * X ** 2 - 1.3 * X * Y + 0.4 * Y ** 3
        + 2e-4 * np.cos(3.0e3 * X))
for weighted in (False, True):
    w = None
    if weighted:
        w = np.exp(-((X / 6e-4) ** 2 + (Y / 6e-4) ** 2))
    for flag in (False, True):
        LT.DETERMINISTIC_TRACED_FIT = flag
        (ev, rows) = _screen_census(
            lambda: LT._Cheb2DEvaluator(ax, ax, vals, order=6, weights=w))
        # reference arm at the SAME solver state
        ref_same = _retired_cheb_coeffs(ax, ax, vals, 6, weights=w,
                                        deterministic=flag)
        ref_old = _retired_cheb_coeffs(ax, ax, vals, 6, weights=w,
                                       deterministic=False)
        c = np.asarray(ev.coeffs)
        print('  weighted=%-5s flag=%-5s  rcond=%.3e screened=%s'
              % (weighted, flag, rows[0][2], rows[0][3]))
        print('     vs reference AT THE SAME STATE : identical=%s  maxdiff=%.3e'
              % (np.array_equal(c, ref_same),
                 float(np.abs(c - ref_same).max())))
        print('     vs reference at the OLD default: identical=%s  maxdiff=%.3e'
              % (np.array_equal(c, ref_old), float(np.abs(c - ref_old).max())))

# ---------------------------------------------------------------- (3)
print()
print('=== (3) D7 shrunken-basis liability demonstration ===')
rng = np.random.default_rng(20260729)
xs = np.linspace(-1.0, 1.0, 41)
Xg, Yg = np.meshgrid(xs, xs, indexing='ij')
vals3 = (np.exp(0.7 * Xg) * np.cos(1.3 * Yg)
         + 1e-9 * rng.standard_normal(Xg.shape))
w3 = np.where(((Xg - 0.5) ** 2 + Yg ** 2) <= 0.09, 1.0, 1e-4)
a, b = 1.0 / 0.3, -0.5 / 0.3
inside = np.array([[0.5, 0.0], [0.6, 0.1], [0.4, -0.1]])
corner = np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0]])
for flag in (False, True):
    LT.DETERMINISTIC_TRACED_FIT = flag
    (ev_wide, rw) = _screen_census(
        lambda: LT._Cheb2DEvaluator(xs, xs, vals3, order=12, weights=w3))
    (ev_tight, rt) = _screen_census(
        lambda: LT._Cheb2DEvaluator(a * xs + b, a * xs + b, vals3,
                                    order=12, weights=w3))
    d = {}
    for name, pts in (('inside', inside), ('corner', corner)):
        f0 = np.asarray(ev_wide.ev(pts[:, 0], pts[:, 1]))
        f1 = np.asarray(ev_tight.ev(a * pts[:, 0] + b, a * pts[:, 1] + b))
        d[name] = float(np.max(np.abs(f0 - f1)))
    # the SCALE the divergence should be read against: the function's own
    # magnitude at those points, not an absolute floor
    scale = float(np.max(np.abs(ev_wide.ev(corner[:, 0], corner[:, 1]))))
    print('  flag=%-5s wide rcond=%.3e screened=%s | tight rcond=%.3e '
          'screened=%s' % (flag, rw[0][2], rw[0][3], rt[0][2], rt[0][3]))
    print('     inside %.6e   corner %.6e   ratio %.4g   bar 1e3'
          % (d['inside'], d['corner'],
             d['corner'] / max(d['inside'], 1e-16)))
    print('     |f(corner)| scale %.6e   corner/scale %.3e'
          % (scale, d['corner'] / scale))
