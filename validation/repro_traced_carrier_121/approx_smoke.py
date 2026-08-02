# Smoke + differential-floor check for the approximation audit harness.
# Runs the shipped default twice (the second through an IDENTITY patch) and
# prints the null delta.  Anything below that delta is not a measurement.
import os

import numpy as np

import approx_common as A


def main():
    order = tuple(int(v) for v in os.environ.get('ORD', '-4,-2').split(','))
    post, env, R, dx, P_in, L, M, cen = A.setup(order)
    print("order %s  L,M = %.6f, %.6f  centre_out = (%.6f, %.6f) mm" %
          (order, L, M, cen[0] * 1e3, cen[1] * 1e3))
    print("grid: RN=%d dx=%.4f um  RS=%d  NFC=%d WF=%.1f  NOUT=%d DXO=%.3f um"
          % (A.RN, dx * 1e6, A.RS, A.NFC, A.WF, A.NOUT, A.DXO * 1e6))

    E0, st0, s0 = A.run_chain(post, env, R, dx, L, M, cen)
    m0 = A.metrics(E0, P_in)
    print("baseline  %.0f s  EE3 %.4f%%  EE6 %.4f%%  Ptile %.5f%%  "
          "off (%+.2f,%+.2f) um" %
          (s0, m0['EE3'] * 100, m0['EE6'] * 100, m0['P_tile'] * 100,
           m0['off_x'] * 1e6, m0['off_y'] * 1e6))
    for s in st0:
        if s.get('target'):
            continue
        extra = ''
        if 'na_exit_measured' in s:
            extra = ("  na_par %.4f na_meas %.4f na_nyq %.4f  P>nyq %.3e" %
                     (s['na_exit'], s['na_exit_measured'],
                      s['na_grid_nyquist'], s['exit_power_above_nyquist']))
        print("  %-12s dx %8.4f um  w %8.1f um  R_in %10.4f  R_out %10.4f%s" %
              (s['name'], s['dx'] * 1e6, s['w'] * 1e6, s['R_in'] * 1e3,
               s['R_out'] * 1e3, extra))

    ident = [(A.CM, '_sphere_parab_conversion', A.CM._sphere_parab_conversion)]
    E1, _st1, s1 = A.run_chain(post, env, R, dx, L, M, cen, patches=ident)
    d, ph = A.field_diff(E1, E0)
    m1 = A.metrics(E1, P_in)
    print("NULL (identity patch)  %.0f s  relL2 %.3e  dphi %.2e rad  "
          "dEE3 %+.3e pts" % (s1, d, ph, (m1['EE3'] - m0['EE3']) * 100))
    np.savez_compressed(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     '_approx_base_%d_%d.npz' % order),
        E=E0, P_in=P_in, cen=np.array(cen))


if __name__ == '__main__':
    main()
