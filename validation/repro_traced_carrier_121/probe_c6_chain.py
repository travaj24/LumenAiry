# Niche C6 END TO END: design 121's per-order spot at the chain's group-5
# exit, C6 OFF vs ON, on the SAME instrument the C5 changelog table used.
#
# The scorer is ``hybrid_localize_121.main()`` at NMIN = NMAX = 6 -- the chain
# does all six post-DOE groups and the exact-ray + Rayleigh-Sommerfeld oracle
# finishes the trailing leg -- so every number here is directly comparable with
# the C5 table (0,0 87.99 / -4,0 76.61 / -4,-2 73.66) and with the pre-C5 one
# (87.99 / 70.19 / 66.24).  ``UP=1``: the Fourier upsample in that script uses
# the fftshift-paste idiom whose off-by-one on odd crops was worth 2.6 rad, so
# it is left OFF.
#
# n_chain = 0 (the pure exact-ray oracle) is run once per order as the CEILING.
#
# usage:  ORDERS='0,0 -4,0 -4,-2' CASES='off,deg3,deg4' python probe_c6_chain.py
import os
import sys

import lumenairy.elements._lens_traced as LT
import hybrid_localize_121 as H


def main():
    orders = os.environ.get('ORDERS', '0,0 -4,0 -4,-2').split()
    cases = os.environ.get('CASES', 'oracle,off,deg3,deg4').split(',')
    os.environ.setdefault('NOUT', '61')
    os.environ.setdefault('DXO', '0.4')
    os.environ.setdefault('NL', '161')
    os.environ.setdefault('UP', '1')
    for o in orders:
        for c in cases:
            c = c.strip()
            os.environ['ORD'] = o
            if c == 'oracle':
                os.environ['NMIN'] = os.environ['NMAX'] = '0'
                flag, deg = False, None
            else:
                os.environ['NMIN'] = os.environ['NMAX'] = '6'
                flag = (c != 'off')
                deg = int(c[3:]) if c.startswith('deg') else None
            old = (LT.REMAP_STATIONARY_PHASE_LAUNCH,
                   LT._REMAP_RESID_EIKONAL_DEGREE,
                   LT._REMAP_RESID_DEGREE_CAP)
            LT.REMAP_STATIONARY_PHASE_LAUNCH = flag
            if deg is not None:
                LT._REMAP_RESID_EIKONAL_DEGREE = deg
                LT._REMAP_RESID_DEGREE_CAP = max(deg,
                                                 LT._REMAP_RESID_DEGREE_CAP)
            print(f"\n########## ORDER {o}   C6 = {c} "
                  f"(flag={flag}, degree={deg}) ##########", flush=True)
            try:
                H.main()
            finally:
                (LT.REMAP_STATIONARY_PHASE_LAUNCH,
                 LT._REMAP_RESID_EIKONAL_DEGREE,
                 LT._REMAP_RESID_DEGREE_CAP) = old


if __name__ == '__main__':
    sys.exit(main())
