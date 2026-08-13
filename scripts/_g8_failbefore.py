"""G8 FAIL-BEFORES -- the re-architected probe must still REFUSE.

A guard that stops refusing is not a guard.  Two models that are genuinely
worse than the incumbent, on niche C6's own fixture, both backends:

  A  the EXIT-DEGREE LADDER.  Degree 8 (45 terms) is the underfit
     ``BUILD_INVERSE_MAP`` S3's ladder puts at 6.25e-03 waves; it must be
     refused, and the ladder must show WHERE the guard's knee is.
  B  the S6.5b PRE-RESTRICTION model.  ``fit_radius_beam_factor=None`` leaves
     the model fitting the whole launch square unweighted -- the regime that
     read 4.5258e-01 waves against a restricted 1.9965e-05 on design 121.

Scratch driver for FIX_G8_PROBE_2026_08_12; not a test.
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys.path[0] != _ROOT:
    sys.path.insert(0, _ROOT)
sys.path.insert(1, os.path.join(_ROOT, 'tests', 'unit'))

import warnings

import numpy as np
import test_niche_c6_stationary_phase_launch as C6  # noqa: E402

from lumenairy.elements import _lens_imap as IM  # noqa: E402


def run(basis, **over):
    E, _X, _Y = C6._field()
    rec = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        C6._run(E, True, newton_fit=basis, _imap_out=rec, **over)
    return rec


def line(tag, r):
    v = 'ENGAGE' if r.get('engaged') else 'REFUSE ' + str(r.get('refused'))
    print('  %-28s %-11s a_opl %.4e  b_opl %.4e (%.3gx)  '
          'a_pos %.4e  b_pos %.4e (%.3gx)'
          % (tag, v,
             r.get('parity_map_opl_waves', np.nan),
             r.get('parity_incumbent_opl_waves', np.nan),
             r.get('parity_ratio_opl', np.nan),
             r.get('parity_map_pos_m', np.nan),
             r.get('parity_incumbent_pos_m', np.nan),
             r.get('parity_ratio_pos', np.nan)), flush=True)


if __name__ == '__main__':
    IM.TRACED_INVERSE_MAP = True
    IM._IMAP_CACHE_SIZE = 0
    print('A. THE EXIT-DEGREE LADDER (fit domain as shipped)')
    old = IM._IMAP_EXIT_DEGREE
    try:
        for deg in (6, 8, 10, 12, 14):
            IM._IMAP_EXIT_DEGREE = deg
            for basis in ('polynomial', 'spline'):
                line('degree %-2d %s' % (deg, basis), run(basis))
    finally:
        IM._IMAP_EXIT_DEGREE = old

    print('B. THE S6.5b PRE-RESTRICTION MODEL '
          '(fit_radius_beam_factor unset -> whole launch square, unweighted)')
    for basis in ('polynomial', 'spline'):
        line('unrestricted %s' % basis,
             run(basis, fit_radius_beam_factor=None))
