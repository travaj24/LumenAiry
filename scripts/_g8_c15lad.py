"""Exit-degree ladder on the niche-C15 fixture, under the re-architected G8."""
from __future__ import annotations
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys.path[0] != _ROOT:
    sys.path.insert(0, _ROOT)
sys.path.insert(1, os.path.join(_ROOT, 'tests', 'unit'))
import numpy as np  # noqa: E402
import test_niche_c15_inverse_map as C15  # noqa: E402
from lumenairy.elements import _lens_imap as IM  # noqa: E402

if __name__ == '__main__':
    old = IM._IMAP_EXIT_DEGREE
    try:
        for deg in (4, 6, 8, 10, 12, 14):
            IM._IMAP_EXIT_DEGREE = deg
            _E, r, _m = C15._call(flag=True)
            print('deg %-3d %-12s n_probe %s/%s  a_opl %.4e  b_opl %.4e (%.4gx)'
                  '  a_pos %.4e  b_pos %.4e (%.4gx)'
                  % (deg,
                     'ENGAGE' if r.get('engaged') else 'REF ' + str(r.get('refused')),
                     r.get('n_probe_traced'), r.get('n_probe_requested'),
                     r.get('parity_map_opl_waves', np.nan),
                     r.get('parity_incumbent_opl_waves', np.nan),
                     r.get('parity_ratio_opl', np.nan),
                     r.get('parity_map_pos_m', np.nan),
                     r.get('parity_incumbent_pos_m', np.nan),
                     r.get('parity_ratio_pos', np.nan)), flush=True)
    finally:
        IM._IMAP_EXIT_DEGREE = old
