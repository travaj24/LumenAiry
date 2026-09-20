"""VERIFY-WP-C2 ROUND 2, item D8 -- the two d3 arms' one-ULP floors, all
four directions, re-measured, plus the COST of each extra direction.

Two questions:

1. do the four one-ULP floors reproduce, and does the maximum over
   ``('up', 'down')`` really BRACKET the maximum over all four on BOTH
   builds?  (If ``one_element`` or ``real`` could exceed them, barring
   against the two-direction max would be barring against a sample again.)
2. what does the second direction COST?  Each arm's floor evaluation is a
   full chain call, and the round-2 splice records the two d3 ids going
   43.8 / 10.2 s -> 93.6 / 33.0 s.  This probe times ONE floor evaluation
   of each arm so the question "is the second direction worth 50 s" has a
   measured answer, and records what the decision reads with one direction
   only, so the cheaper alternative can be judged rather than guessed.

Usage:  LUMENAIRY_ROOT=<root> python vr2_d3_floors.py <out.json>
"""
import json
import os
import sys
import time

import numpy as np

_ROOT = os.environ['LUMENAIRY_ROOT']
sys.path.insert(0, _ROOT)

import lumenairy as la  # noqa: E402

_want = os.path.realpath(os.path.join(_ROOT, 'lumenairy'))
assert os.path.realpath(os.path.dirname(la.__file__)) == _want, la.__file__

sys.path.insert(0, os.path.join(_ROOT, 'tests', 'unit'))
import test_niche_d3_guards as D  # noqa: E402

KINDS = ('up', 'down', 'real', 'one_element')


def main(out_path):
    out = {'lumenairy_file': la.__file__,
           'python': sys.version.split()[0], 'numpy': np.__version__}

    # ---------------- arm 1: the linearity-error floor -----------------
    t0 = time.perf_counter()
    bad6 = D._linearity_error(0.023)
    out['t_one_linearity_error_s'] = time.perf_counter() - t0
    good6 = D._linearity_error(0.0005)
    _deg = D._lens_traced._REMAP_RESID_EIKONAL_DEGREE
    D._lens_traced._REMAP_RESID_EIKONAL_DEGREE = 4
    try:
        bad4 = D._linearity_error(0.023)
    finally:
        D._lens_traced._REMAP_RESID_EIKONAL_DEGREE = _deg
    out['bad6'] = float(bad6)
    out['good6'] = float(good6)
    out['bad4'] = float(bad4)
    out['separation_bad6_over_good6'] = float(bad6 / good6)
    out['degree_effect'] = float(abs(bad4 - bad6) / max(bad6, 1e-300))

    arm1 = {}
    for k in KINDS:
        t = time.perf_counter()
        arm1[k] = float(abs(D._linearity_error(0.023, nudge=k) - bad6)
                        / max(bad6, 1e-300))
        arm1[k + '_seconds'] = time.perf_counter() - t
    out['arm1_floors'] = arm1
    f1 = {k: arm1[k] for k in KINDS}
    out['arm1_max_all_four'] = max(f1.values())
    out['arm1_max_up_down'] = max(f1['up'], f1['down'])
    out['arm1_up_down_brackets_all_four'] = (
        out['arm1_max_up_down'] >= out['arm1_max_all_four'])
    out['arm1_spread_all_four'] = max(f1.values()) / max(min(f1.values()),
                                                         1e-300)
    out['arm1_margin_at_worst'] = out['degree_effect'] / out['arm1_max_all_four']
    out['arm1_margin_at_up_only'] = out['degree_effect'] / f1['up']
    out['arm1_decision_10x_at_worst'] = bool(
        out['degree_effect'] > 10.0 * out['arm1_max_all_four'])
    out['arm1_decision_10x_at_up_only'] = bool(
        out['degree_effect'] > 10.0 * f1['up'])

    # ---------------- arm 2: the multiplexed-norm floor ----------------
    t0 = time.perf_counter()
    on6 = D._mux_chain_field(0.023, degree=6, launch=True)
    out['t_one_mux_chain_s'] = time.perf_counter() - t0
    ref = float(np.linalg.norm(on6))
    on4 = D._mux_chain_field(0.023, degree=4, launch=True)
    moved = float(np.linalg.norm(on4 - on6)) / ref
    out['moved'] = moved
    again = D._mux_chain_field(0.023, degree=6, launch=True)
    out['same_degree_twice'] = float(np.linalg.norm(again - on6)) / ref

    arm2 = {}
    for k in KINDS:
        t = time.perf_counter()
        arm2[k] = float(D._mux_last_bit_noise(0.023, degree=6, launch=True,
                                              kind=k))
        arm2[k + '_seconds'] = time.perf_counter() - t
    out['arm2_floors'] = arm2
    f2 = {k: arm2[k] for k in KINDS}
    out['arm2_max_all_four'] = max(f2.values())
    out['arm2_max_up_down'] = max(f2['up'], f2['down'])
    out['arm2_up_down_brackets_all_four'] = (
        out['arm2_max_up_down'] >= out['arm2_max_all_four'])
    out['arm2_spread_all_four'] = max(f2.values()) / max(min(f2.values()),
                                                         1e-300)
    out['arm2_margin_at_worst'] = moved / out['arm2_max_all_four']
    out['arm2_margin_at_up_only'] = moved / f2['up']
    out['arm2_decision_3x_at_worst'] = bool(
        moved > 3.0 * out['arm2_max_all_four'])
    out['arm2_decision_3x_at_up_only'] = bool(moved > 3.0 * f2['up'])

    # cost of the second direction, as the fraction of the test it is
    out['arm1_second_direction_seconds'] = arm1['down_seconds']
    out['arm2_second_direction_seconds'] = arm2['down_seconds']

    with open(out_path, 'w') as fh:
        json.dump(out, fh, indent=1)
    for k, v in out.items():
        if isinstance(v, dict):
            print(k + ':')
            for kk, vv in v.items():
                print('   %-18s %s' % (kk, vv))
        else:
            print('%-38s %s' % (k, v))
    print('wrote', out_path)


if __name__ == '__main__':
    main(sys.argv[1])
