"""D5 fit-domain symmetry probe -- measures the c6 backend spread with the
inverse-map flag OFF and ON, and reports what the two bases hand the model.

Scratch driver for FIX_FIT_DOMAIN_SYMMETRY_2026_08_12; not a test.
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


def spread():
    E, _X, _Y = C6._field()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ref = np.abs(C6._run(E, True, newton_fit='spline'))
        got = np.abs(C6._run(E, True))
    scale = float(ref.max())
    return float(np.abs(got - ref).max()) / scale


if __name__ == '__main__':
    for flag in (False, True):
        IM.TRACED_INVERSE_MAP = flag
        print('TRACED_INVERSE_MAP=%-5s  d_ok = %.6e   (bar 5e-04)'
              % (flag, spread()), flush=True)
