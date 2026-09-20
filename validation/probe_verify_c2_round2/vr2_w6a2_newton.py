"""VERIFY-WP-C2 ROUND 2, the ``w6_a2`` restatement -- does the replacement
DECISION have a gap above its bar?

The retired assertion (``norm_offset <= 2 * bound``) was a theorem about a
linear solve and could not fail.  Its replacement is

    a SECOND Newton step taken from ``v*`` is < 1e-4 of the first.

``1e-4`` is a CHOSEN multiplier, and the test's message claims that "a root
that is not converged (a wrong Hessian, a missing prior term, a sign error)
gives a ratio of order 1, four decades above it".  That sentence is an
argument, not a measurement.  This probe MEASURES it: it reproduces the
test's arithmetic and then evaluates the same second-step ratio at five
deliberately UNCONVERGED expansion points, so the gap above the bar is a
number.

    shipped          the solver's own v*
    no_step          v* replaced by the pupil centre (zero Newton steps)
    half_step        half of the first Newton step
    no_prior         the step taken with the prior term dropped from H
    sign_error       the step taken with the wrong sign
    tenth_step       a tenth of the first Newton step (the SMALLEST
                     departure that still has to be caught)

Usage:  LUMENAIRY_ROOT=<root> python vr2_w6a2_newton.py <out.json>
"""
import json
import os
import sys

import numpy as np

_ROOT = os.environ['LUMENAIRY_ROOT']
sys.path.insert(0, _ROOT)

import lumenairy as la  # noqa: E402

_want = os.path.realpath(os.path.join(_ROOT, 'lumenairy'))
assert os.path.realpath(os.path.dirname(la.__file__)) == _want, la.__file__

sys.path.insert(0, os.path.join(_ROOT, 'tests', 'unit'))
import test_niche_audit_w6_asymptotic as W  # noqa: E402

from lumenairy.propagators.asymptotic import _solve_envelope_stationary_batch  # noqa: E402


def main(out_path):
    fit = W._fit()
    w_s, w_p = 20e-6, 0.02
    v_c = np.array([float(fit.v2x_centre), float(fit.v2y_centre)])
    vx, vy, _ = _solve_envelope_stationary_batch(
        fit, np.array([fit.s2x_centre]), np.array([fit.s2y_centre]),
        0.0, 0.0, w_s=w_s, w_p=w_p, v_cx=v_c[0], v_cy=v_c[1])
    v_star = np.array([float(vx[0]), float(vy[0])])

    def _rH(v, prior=True):
        s1x, s1y, jxx, jxy, jyx, jyy = fit.eval_s1_with_v2_grad(
            np.asarray(float(fit.s2x_centre)).reshape(()),
            np.asarray(float(fit.s2y_centre)).reshape(()),
            np.asarray(v[0]).reshape(()), np.asarray(v[1]).reshape(()))
        J = np.array([[float(jxx), float(jxy)], [float(jyx), float(jyy)]])
        ds1 = np.array([float(s1x), float(s1y)])
        r = (J.T @ ds1) / w_s ** 2
        H = (J.T @ J) / w_s ** 2
        if prior:
            r = r + (v - v_c) / w_p ** 2
            H = H + np.eye(2) / w_p ** 2
        return r, H

    r_c, H_c = _rH(v_c)
    predicted = -np.linalg.solve(H_c, r_c)
    n1 = float(np.linalg.norm(predicted))

    def ratio_at(v, prior=True):
        r2, H2 = _rH(np.asarray(v, dtype=float), prior=prior)
        return float(np.linalg.norm(np.linalg.solve(H2, r2))) / n1

    cases = {
        'shipped': ratio_at(v_star),
        'no_step': ratio_at(v_c),
        'half_step': ratio_at(v_c + 0.5 * predicted),
        'tenth_step': ratio_at(v_c + 0.1 * predicted),
        'sign_error': ratio_at(v_c - predicted),
        'no_prior_in_the_step': ratio_at(
            v_c - np.linalg.solve((H_c - np.eye(2) / w_p ** 2), r_c)),
    }
    # and the one that matters most: how far from v* does the ratio reach
    # the 1e-4 bar?  A bisection on the step fraction.
    lo, hi = 0.0, 1.0            # fraction of the first step NOT taken
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if ratio_at(v_c + (1.0 - mid) * predicted) < 1e-4:
            lo = mid
        else:
            hi = mid
    out = {
        'lumenairy_file': la.__file__,
        'python': sys.version.split()[0], 'numpy': np.__version__,
        'first_step_norm': n1,
        'v_star_minus_v_c': [float(x) for x in (v_star - v_c)],
        'second_step_ratios': cases,
        'bar': 1e-4,
        'shipped_headroom_under_the_bar': 1e-4 / cases['shipped'],
        'smallest_unconverged_ratio': min(
            v for k, v in cases.items() if k != 'shipped'),
        'gap_above_the_bar':
            min(v for k, v in cases.items() if k != 'shipped') / 1e-4,
        'fraction_of_the_first_step_that_may_be_missed_at_the_bar': lo,
    }
    with open(out_path, 'w') as fh:
        json.dump(out, fh, indent=1)
    for k, v in out.items():
        if isinstance(v, dict):
            print(k + ':')
            for kk, vv in v.items():
                print('   %-22s %.6e' % (kk, vv))
        else:
            print('%-52s %s' % (k, v))
    print('wrote', out_path)


if __name__ == '__main__':
    main(sys.argv[1])
