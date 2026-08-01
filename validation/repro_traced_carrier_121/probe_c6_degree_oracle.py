# Niche C6 loose end: the NON-MONOTONE degree response of the element-vs-oracle
# wavefront metric.
#
# ``_REMAP_RESID_EIKONAL_DEGREE``'s note records, and does not resolve:
#
#     "the ELEMENT-vs-oracle column is NOT monotone in the degree (3 beats 4
#      and 6, 0.0074 against 0.0140, while the model's own slope residual keeps
#      improving) ... The likely explanation is that the oracle's own
#      band-limited representation of ``a`` and this fit's differ at high
#      spatial frequency, i.e. it is a property of the comparison rather than
#      of the field.  Not resolved."
#
# probe_c6_split.py already put ALL of it in the MODEL leg, so it is not a
# coding defect.  This script tests the "property of the comparison" claim
# directly by moving the COMPARISON and leaving the field alone.  Three knobs,
# none of which is part of the physics:
#
#   UP     the oracle's own band-limited upsample of the measured residual
#          (wfe_probe_remap.ResidualField(..., up=UP)) -- the exact thing the
#          hypothesis names;
#   THRESH the amplitude floor for a pixel to enter the score;
#   HALF   the size of the scored exit patch.
#
# If the degree ORDERING moves with these, the ordering is a property of the
# instrument.  If it survives all three, it is a property of the field and the
# hypothesis on record is wrong.
#
# A monotone CONTROL is run first on the synthetic fixture in
# tests/unit/test_niche_c6_stationary_phase_launch.py, whose oracle is
# ANALYTIC (no band-limited residual at all): there the response is
# 2.065e-02 (off) -> 1.406e-02 (deg 2 and 3) -> 2.344e-05 (deg 4, 5, 6), i.e.
# perfectly ordered.  So the non-monotonicity is not in the fit.
#
# usage:  ORDERS='-4,-2' DEGS='2,3,4,6' UPS='4,8,16' python \
#             probe_c6_degree_oracle.py
import os
import sys
import time
import warnings

import numpy as np

import probe_c6_element as E6


def main():
    warnings.filterwarnings('ignore')
    m, n = (int(v) for v in os.environ.get('ORDERS', '-4,-2').split(','))
    degs = os.environ.get('DEGS', '2,3,4,6')
    ups = [int(v) for v in os.environ.get('UPS', '4,8,16').split(',')]
    threshs = [float(v) for v in os.environ.get('THRESHS', '0.02,0.10').split(',')]
    halves = [int(v) for v in os.environ.get('HALVES', '96,64').split(',')]
    rs = int(os.environ.get('RS', '4'))
    cases = E6.parse_cases('off,' + ','.join('deg' + d for d in degs.split(',')))

    print("order (%d,%d): does the degree ORDERING move with the COMPARISON?"
          % (m, n))
    print()
    hdr = ("%-4s %-7s %-5s %-16s %9s %9s %8s %8s" %
           ('up', 'thresh', 'half', 'config', 'sigmaF', 'nn p99', 'g resid',
            'npix'))
    print(hdr)
    print('-' * len(hdr))
    grid = ([(u, threshs[0], halves[0]) for u in ups]
            + [(ups[len(ups) // 2], t, halves[0]) for t in threshs[1:]]
            + [(ups[len(ups) // 2], threshs[0], h) for h in halves[1:]])
    for (up, th, hf) in grid:
        t0 = time.time()
        r = E6.analyse(m, n, cases, half=hf, thresh=th, up=up, rs=rs)
        best, bestv = None, np.inf
        for row in r['rows']:
            v = row['wfe']['sigma_fid_waves']
            print("%-4d %-7.2f %-5d %-16s %9.5f %9.4f %8.2e %8d"
                  % (up, th, hf, row['label'], v, row['wfe']['nn_step_p99'],
                     row['diag'].get('grad_a_residual_rms', np.nan),
                     row['wfe']['n_pix']))
            if row['label'] != 'OFF (shipped)' and v < bestv:
                best, bestv = row['label'], v
        print("     -> best degree: %s   [%.0fs]" % (best, time.time() - t0),
              flush=True)
    print()
    print("sigmaF is unwrap-free; legitimate only while nn p99 << pi = 3.1416 "
          "(amplitude-weighted p99, not a max).")


if __name__ == '__main__':
    sys.exit(main())
