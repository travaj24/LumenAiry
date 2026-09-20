"""WP-C2 ROUND 3, VR2-D5 -- the ``w6_a2`` second-Newton-step bar, and the
mode its own message names that it does not catch.

``tests/unit/test_niche_audit_w6_asymptotic.py`` DECISION 3 asserts that a
second Newton step taken from ``v*`` is ``< 1e-4`` of the first, and its
message says that "a root that is not converged (a wrong Hessian, a
missing prior term, a sign error) gives a ratio of order 1, four decades
above it".

VERIFY-WP-C2 round 2 measured that the third of the three named modes --
the PRIOR TERM dropped from the Hessian -- reads about 2.5e-05 and so
PASSES a 1e-4 bar.  This probe re-measures the whole table independently
and adds the bisection that says how much of the first step may be missed
before a 1e-5 bar fires, so the tightened bar is chosen from a measurement
rather than from a round number.

Modes evaluated, all against the SAME first-step norm:

    shipped       the solver's own v*
    no_step       no Newton step taken at all (the pupil centre)
    half_step     half of the first step
    tenth_missed  nine tenths of the first step
    sign_error    the first step with the wrong sign
    no_prior      the step computed with the prior term dropped from H

Usage:  LUMENAIRY_ROOT=<root> python r3_w6a2_modes.py <out.json>
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

from lumenairy.propagators.asymptotic import (  # noqa: E402
    _solve_envelope_stationary_batch)

W_S, W_P = 20e-6, 0.02
BARS = (1e-4, 1e-5)


def main(out_path):
    fit = W._fit()
    v_c = np.array([float(fit.v2x_centre), float(fit.v2y_centre)])
    vx, vy, _conv = _solve_envelope_stationary_batch(
        fit, np.array([fit.s2x_centre]), np.array([fit.s2y_centre]),
        0.0, 0.0, w_s=W_S, w_p=W_P, v_cx=v_c[0], v_cy=v_c[1])
    v_star = np.array([float(vx[0]), float(vy[0])])

    def rH(v, prior=True):
        s1x, s1y, jxx, jxy, jyx, jyy = fit.eval_s1_with_v2_grad(
            np.asarray(float(fit.s2x_centre)).reshape(()),
            np.asarray(float(fit.s2y_centre)).reshape(()),
            np.asarray(v[0]).reshape(()), np.asarray(v[1]).reshape(()))
        J = np.array([[float(jxx), float(jxy)], [float(jyx), float(jyy)]])
        ds1 = np.array([float(s1x), float(s1y)])
        r = (J.T @ ds1) / W_S ** 2
        H = (J.T @ J) / W_S ** 2
        if prior:
            r = r + (v - v_c) / W_P ** 2
            H = H + np.eye(2) / W_P ** 2
        return r, H

    r_c, H_c = rH(v_c)
    predicted = -np.linalg.solve(H_c, r_c)
    n1 = float(np.linalg.norm(predicted))

    def ratio_at(v):
        r2, H2 = rH(np.asarray(v, dtype=float))
        return float(np.linalg.norm(np.linalg.solve(H2, r2))) / n1

    # the step a solver that forgot the prior term would take
    step_no_prior = -np.linalg.solve(H_c - np.eye(2) / W_P ** 2, r_c)

    modes = {
        'shipped': ratio_at(v_star),
        'no_step': ratio_at(v_c),
        'half_step': ratio_at(v_c + 0.5 * predicted),
        'tenth_missed': ratio_at(v_c + 0.9 * predicted),
        'sign_error': ratio_at(v_c - predicted),
        'no_prior': ratio_at(v_c + step_no_prior),
    }

    # how much of the first step may be MISSED before each bar fires --
    # the smallest departure the bar can still see, measured rather than
    # argued
    def missable(bar):
        lo, hi = 0.0, 1.0
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if ratio_at(v_c + (1.0 - mid) * predicted) < bar:
                lo = mid
            else:
                hi = mid
        return lo

    out = {
        'lumenairy_file': la.__file__,
        'python': sys.version.split()[0],
        'numpy': np.__version__,
        'first_step_norm': n1,
        'v_star_minus_v_c': [float(x) for x in (v_star - v_c)],
        'second_step_ratios': modes,
        'bars': {},
    }
    for bar in BARS:
        unconverged = {k: v for k, v in modes.items() if k != 'shipped'}
        out['bars'][repr(bar)] = {
            'shipped_passes': modes['shipped'] < bar,
            'shipped_headroom': bar / modes['shipped'],
            'modes_caught': sorted(k for k, v in unconverged.items()
                                   if v >= bar),
            'modes_missed': sorted(k for k, v in unconverged.items()
                                   if v < bar),
            'smallest_caught_ratio': min(
                [v for v in unconverged.values() if v >= bar] or [None]),
            'fraction_of_the_first_step_missable': missable(bar),
        }
    with open(out_path, 'w') as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print(json.dumps(out, indent=1, sort_keys=True))


if __name__ == '__main__':
    main(sys.argv[1])
