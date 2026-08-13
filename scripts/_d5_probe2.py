"""Diagnose WHAT differs between the two newton_fit bases when the inverse map
is engaged: the model's own guard record (sample set, normalisation box, hull).
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys.path[0] != _ROOT:
    sys.path.insert(0, _ROOT)
sys.path.insert(1, os.path.join(_ROOT, 'tests', 'unit'))

import warnings

import test_niche_c6_stationary_phase_launch as C6  # noqa: E402

from lumenairy.elements import _lens_imap as IM  # noqa: E402

KEYS = ('engaged', 'gate_open', 'refused', 'refused_why', 'n_alive',
        'n_fit_samples', 'n_detj_census', 'det_j_range', 'exit_centre_mm',
        'exit_half_mm', 'n_out_of_domain', 'g8_map', 'g8_incumbent',
        'g8_ratio', 'fit_resid_max_waves')

if __name__ == '__main__':
    IM.TRACED_INVERSE_MAP = True
    E, _X, _Y = C6._field()
    recs = {}
    for basis in ('polynomial', 'spline'):
        out = {}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            C6._run(E, True, newton_fit=basis, _imap_out=out)
        recs[basis] = out
    for k in KEYS:
        a, b = recs['polynomial'].get(k), recs['spline'].get(k)
        flag = '' if repr(a) == repr(b) else '   <-- DIFFERS'
        print('%-22s poly=%-28r spline=%-28r%s' % (k, a, b, flag))
    print()
    print('all keys:', sorted(set(recs['polynomial']) | set(recs['spline'])))
