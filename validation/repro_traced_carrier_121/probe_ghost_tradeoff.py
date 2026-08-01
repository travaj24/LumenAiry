# Niche C6 on-axis ghost: the TRADE-OFF the fix actually makes.
#
# probe_ghost_c6.py shows the D1 weighted restriction removes the on-axis ghost
# (1.03e-03 of Pin beyond 4 mm -> exactly 0).  But the weighted branch solves
# ILL-CONDITIONED normal equations (cond(Gram) up to 1.9e13 per the library's
# own docstring), so it can cost WAVEFRONT accuracy where the hard mask was
# exact -- on the synthetic free-leg fixture of
# tests/unit/test_niche_c6_stationary_phase_launch.py, whose map is affine and
# whose hard-mask fit is therefore exact, it costs 2.3e-05 -> 6.9e-04 waves.
#
# A halo fix that costs wavefront is not obviously a win for an ORACLE, so this
# script measures BOTH on design 121's real element call, against the same
# exact-ray oracle probe_c6_element.py uses, on axis (where the fix acts) and
# on a tilted order (where it must be inert).
#
# usage:  ORDERS='0,0 -4,-2' python probe_ghost_tradeoff.py
import os
import sys
import time
import warnings

import numpy as np

import probe_c6_element as E6
import lumenairy.elements._lens_traced as LT


def main():
    warnings.filterwarnings('ignore')
    orders = [tuple(int(v) for v in o.split(','))
              for o in os.environ.get('ORDERS', '0,0').split()]
    rs = int(os.environ.get('RS', '4'))
    cases = E6.parse_cases(os.environ.get('CASES', 'off,on'))
    import hashlib
    print("   lib sha256 %s   guard present %s"
          % (hashlib.sha256(open(LT.__file__, 'rb').read()).hexdigest()[:16],
             hasattr(LT, 'REMAP_STATIONARY_PHASE_FIT_GUARD')))
    hdr = ("%-8s %-6s %-16s %9s %9s %8s %10s %10s %6s" %
           ('order', 'guard', 'config', 'sigmaF', 'nn p99', 'P/Pin',
            'ghost>4mm', 'grad resid', 'fold'))
    print(hdr)
    print('-' * len(hdr))
    for (m, n) in orders:
        for guard in (False, True):
            old = LT.REMAP_STATIONARY_PHASE_FIT_GUARD
            LT.REMAP_STATIONARY_PHASE_FIT_GUARD = guard
            t0 = time.time()
            try:
                r = E6.analyse(m, n, cases, rs=rs)
            finally:
                LT.REMAP_STATIONARY_PHASE_FIT_GUARD = old
            for row in r['rows']:
                w_ = row['wfe']
                print("%-8s %-6s %-16s %9.5f %9.4f %8.5f %10.3e %10.3e %6d"
                      % (f'({m},{n})', str(guard), row['label'],
                         w_['sigma_fid_waves'], w_['nn_step_p99'],
                         row['power'], row['ghost'],
                         row['diag'].get('grad_a_residual_rms', np.nan),
                         row['flags']['fold']))
            print("         [null: two OFF runs array_equal=%s  %.0fs]"
                  % (r['null_eq'], time.time() - t0), flush=True)
    print()
    print("sigmaF = unwrap-free equivalent rms exit wavefront error (waves) "
          "vs the exact-ray oracle;")
    print("         legitimate only while nn p99 << pi = 3.1416 (stated as an "
          "amplitude-weighted p99, not a max).")


if __name__ == '__main__':
    sys.exit(main())
