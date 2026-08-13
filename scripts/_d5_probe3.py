"""Is G8 DECIDABLE on an interpolating (spline) incumbent?

Arm B of G8 is the element's own Newton on the element's own forward fits.
``RectBivariateSpline`` with ``s=0`` INTERPOLATES: it reproduces every launch
node exactly, including the held-out ones.  So at the probe points -- which
ARE launch nodes -- arm B's error is the Newton loop's leftover residual and
not the incumbent's production accuracy.  Sweep the model's degree and see
whether any model can clear the bar.
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

if __name__ == '__main__':
    IM.TRACED_INVERSE_MAP = True
    IM._IMAP_CACHE_SIZE = 0
    E, _X, _Y = C6._field()
    old = IM._IMAP_EXIT_DEGREE
    print('%-8s %-6s %-10s %-12s %-12s %-8s %-12s %-12s %-8s'
          % ('basis', 'deg', 'engaged', 'a_pos', 'b_pos', 'r_pos',
             'a_opl', 'b_opl', 'r_opl'))
    try:
        for basis in ('polynomial', 'spline'):
            for deg in (10, 12, 14, 16, 18):
                IM._IMAP_EXIT_DEGREE = deg
                out = {}
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    C6._run(E, True, newton_fit=basis, _imap_out=out)
                print('%-8s %-6d %-10s %-12.4e %-12.4e %-8.3f %-12.4e '
                      '%-12.4e %-8.4f'
                      % (basis, deg, out.get('engaged'),
                         out.get('parity_map_pos_m', float('nan')),
                         out.get('parity_incumbent_pos_m', float('nan')),
                         out.get('parity_ratio_pos', float('nan')),
                         out.get('parity_map_opl_waves', float('nan')),
                         out.get('parity_incumbent_opl_waves', float('nan')),
                         out.get('parity_ratio_opl', float('nan'))),
                      flush=True)
    finally:
        IM._IMAP_EXIT_DEGREE = old
