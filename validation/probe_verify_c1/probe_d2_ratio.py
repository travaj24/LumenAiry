"""VERIFY-C1 round 2, defect D2 -- the margin behind the ratio bar.

``tests/unit/test_audit2609_a8_verify.py::
test_verify_a8_e7_gray_edge_beats_hard_at_anamorphic_and_offset_rims``
asserted ``e_g4 <= e_hard``, which EQUALITY satisfies: with ``edge='hard'``
mutated to return the grey mask the two arms are the same array and the id
still passes (VERIFY-C1 measured 15 other ids going red under that mutant and
this not among them).  This probe measures the ratio ``e_hard / e_g4`` on that
id's own three fixtures, which is what the replacement bar is derived from.

Run:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONPATH=<tree> python validation/probe_verify_c1/probe_d2_ratio.py

There is no BLAS in any of it -- a mask sum is an integer count over
``n_sub**2`` -- so the readings are expected bit-identical across builds, and
the script prints enough digits to see if they are not.
"""
import json
import platform
import sys

import numpy as np

import lumenairy
from lumenairy.elements import elements as elem_mod

# The id's own fixture family, verbatim.
FIXTURES = [(37, 1.0, 0.37), (63, 2.5, 0.13), (145, 0.4, 0.29)]
N, DX = 512, 1e-6


def _errors(d_px, dy_ratio, offset):
    dy = DX * dy_ratio
    D = d_px * DX
    E = np.ones((N, N), dtype=np.complex128)
    analytic = np.pi * (D / 2) ** 2

    def _area(**kw):
        out = elem_mod.apply_aperture(E, DX, 'circular', {'diameter': D},
                                      xc=offset * DX, yc=-offset * dy, dy=dy,
                                      **kw)
        return float(np.sum(np.real(out))) * DX * dy

    e_hard = abs(_area(edge='hard') / analytic - 1.0)
    e_g4 = abs(_area(edge='gray') / analytic - 1.0)
    e_g16 = abs(_area(edge='gray', edge_samples=16) / analytic - 1.0)
    return e_hard, e_g4, e_g16


def main():
    print('lumenairy.__file__ =', lumenairy.__file__)
    print('python             =', sys.version.split()[0], platform.platform())
    print('numpy              =', np.__version__)
    rows = []
    for d_px, dy_ratio, offset in FIXTURES:
        e_hard, e_g4, e_g16 = _errors(d_px, dy_ratio, offset)
        ratio = e_hard / e_g4
        rows.append({'d_px': d_px, 'dy_ratio': dy_ratio, 'offset': offset,
                     'e_hard': e_hard, 'e_g4': e_g4, 'e_g16': e_g16,
                     'ratio_hard_over_g4': ratio})
        print(f'D/dx={d_px:4d} dy/dx={dy_ratio:3.1f} off={offset:.2f} px  '
              f'e_hard={e_hard:.6e}  e_g4={e_g4:.6e}  '
              f'ratio={ratio:.6f}  e_g16={e_g16:.6e}')
    worst = min(r['ratio_hard_over_g4'] for r in rows)
    print(f'\nsmallest ratio  = {worst:.6f}')
    print(f'proposed bar    = 1.5  ->  headroom {worst / 1.5:.4f}x')
    print('the value "grey IS hard" would give = 1.0 exactly '
          '(the equality the old <= bar admitted)')
    print(json.dumps({'rows': rows, 'smallest_ratio': worst, 'bar': 1.5,
                      'headroom': worst / 1.5,
                      'lumenairy': lumenairy.__file__,
                      'python': sys.version.split()[0],
                      'numpy': np.__version__}, indent=1))


if __name__ == '__main__':
    main()
