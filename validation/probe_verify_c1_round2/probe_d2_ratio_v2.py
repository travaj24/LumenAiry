"""VERIFY-WP-C1 ROUND 2 -- D2: the 1.5x ratio bar, re-measured independently.

Round 2 replaced ``assert e_g4 <= e_hard`` (satisfied by equality) with
``assert e_hard / e_g4 >= 1.5`` in
``test_audit2609_a8_verify.py::test_verify_a8_e7_gray_edge_beats_hard_at_anamorphic_and_offset_rims``
and claims the three readings 5.804555 / 8.815875 / 1.992823 are identical to
sixteen digits on both builds.

This probe re-measures those three with its own code, prints all sixteen
digits, and then does what round 2 did not: SCANS the binding fixture's
neighbourhood (offset and dy/dx) to see how close the ratio gets to the 1.5
bar on fixtures a re-pinning might plausibly choose, so the bar's margin is
known and not merely asserted for three points.

Run:  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      PYTHONPATH=<tree> python probe_d2_ratio_v2.py <out.json>
"""
import json
import sys

import numpy as np

import lumenairy  # noqa: F401
from lumenairy.elements.elements import apply_aperture

N, DX = 512, 1e-6


def _ratio(d_px, dy_ratio, offset, n_sub=4):
    dy = DX * dy_ratio
    D = d_px * DX
    E = np.ones((N, N), dtype=np.complex128)
    analytic = np.pi * (D / 2) ** 2

    def _area(**kw):
        out = apply_aperture(E, DX, 'circular', {'diameter': D},
                             xc=offset * DX, yc=-offset * dy, dy=dy, **kw)
        return float(np.sum(np.real(out))) * DX * dy

    e_hard = abs(_area(edge='hard') / analytic - 1.0)
    e_g = abs(_area(edge='gray', edge_samples=n_sub) / analytic - 1.0)
    return e_hard, e_g, (e_hard / e_g if e_g else float('inf'))


def main(out_path):
    shipped = []
    for d_px, dy_ratio, offset in [(37, 1.0, 0.37), (63, 2.5, 0.13),
                                   (145, 0.4, 0.29)]:
        e_hard, e_g4, r = _ratio(d_px, dy_ratio, offset)
        shipped.append({'d_px': d_px, 'dy_ratio': dy_ratio, 'offset': offset,
                        'e_hard': repr(e_hard), 'e_g4': repr(e_g4),
                        'ratio': repr(r), 'ratio_16sf': '{0:.16g}'.format(r)})

    # Neighbourhood of the BINDING fixture (145, 0.4): how close does the
    # ratio get to the 1.5 bar, and does it ever cross?
    scan = []
    for offset in [0.0, 0.05, 0.11, 0.17, 0.23, 0.29, 0.35, 0.41, 0.47, 0.5]:
        for dy_ratio in [0.4, 0.5, 1.0]:
            e_hard, e_g4, r = _ratio(145, dy_ratio, offset)
            scan.append({'d_px': 145, 'dy_ratio': dy_ratio, 'offset': offset,
                         'e_hard': e_hard, 'e_g4': e_g4, 'ratio': r})
    below_bar = [s for s in scan if s['ratio'] < 1.5]
    below_one = [s for s in scan if s['ratio'] < 1.0]

    out = {
        'lumenairy_file': lumenairy.__file__,
        'python': sys.version.split()[0],
        'numpy': np.__version__,
        'shipped_three': shipped,
        'neighbourhood_scan': scan,
        'scan_min_ratio': min(s['ratio'] for s in scan),
        'scan_n_below_1p5': len(below_bar),
        'scan_n_below_1p0': len(below_one),
        'scan_below_1p5': below_bar,
    }
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1, sort_keys=True, default=repr)
    print('lumenairy:', lumenairy.__file__)
    for s in shipped:
        print("  D/dx={0:4d} dy/dx={1:3.1f} off={2:.2f}  e_hard={3}  "
              "e_g4={4}  ratio={5}".format(
                  s['d_px'], s['dy_ratio'], s['offset'],
                  '{0:.16e}'.format(float(s['e_hard'])),
                  '{0:.16e}'.format(float(s['e_g4'])), s['ratio_16sf']))
    print("neighbourhood scan of the binding (145) fixture: n={0}  "
          "min ratio={1:.6f}  below 1.5: {2}  below 1.0: {3}".format(
              len(scan), min(s['ratio'] for s in scan), len(below_bar),
              len(below_one)))
    for s in sorted(scan, key=lambda s: s['ratio'])[:5]:
        print("   dy/dx={0:3.1f} off={1:.2f}  ratio={2:.6f}".format(
            s['dy_ratio'], s['offset'], s['ratio']))


if __name__ == '__main__':
    main(sys.argv[1])
