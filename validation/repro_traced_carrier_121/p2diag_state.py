# P2 DID-NOT-WARN diagnosis -- AXIS 3/4: the PROCESS-STATE class, MEASURED.
#
# C11 S9.5/S9.6 closed the state-leak class by CONSTRUCTION (an autouse guard
# that snapshots and restores 91 module-level flags) without ever naming a
# leaker, and explicitly left open whether the same class carries S9.3's
# DID-NOT-WARN.  A guard proves the class is closed GOING FORWARD; it does not
# say which flags could ever have silenced this particular test, and that is
# the question a diagnosis has to answer.
#
# This script asks it directly: for each dispatch-global that an earlier test
# in the same PROCESS could plausibly leave dirty, does the P2 fixture's dx
# drift still exceed ``self_check_tol``, i.e. does the guard still FIRE?
#
# The flags swept are the ones C11 named as the guard's own blind spots at the
# time of the failing commit -- ``fft_infra`` was NOT in the discovered set
# until 1d340bb, so at 5af1edf a leaked ``USE_PYFFTW`` / ``DEFAULT_WAVE_
# PROPAGATOR`` / ``FFTW_MIN_SIZE`` would have survived from any earlier test in
# the shard straight into this one.
#
# usage:  python p2diag_state.py
import os
import platform
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     '..', '..'))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lumenairy.propagators.fft_infra as F                     # noqa: E402
from p2diag_route import run                                    # noqa: E402


CASES = [
    ('shipped                       ', {}),
    ('USE_PYFFTW=False              ', dict(USE_PYFFTW=False)),
    ('FFTW_MIN_SIZE=2**31           ', dict(FFTW_MIN_SIZE=2 ** 31)),
    ('USE_SCIPY_FFT=False           ', dict(USE_SCIPY_FFT=False)),
    ('SCIPY_FFT_WORKERS=1           ', dict(SCIPY_FFT_WORKERS=1)),
    ("DEFAULT_WAVE_PROPAGATOR='sas' ", dict(DEFAULT_WAVE_PROPAGATOR='sas')),
    ("DEFAULT_WAVE_PROPAGATOR='fresnel'", dict(DEFAULT_WAVE_PROPAGATOR='fresnel')),
    ("DEFAULT_WAVE_PROPAGATOR='rs'  ", dict(DEFAULT_WAVE_PROPAGATOR='rs')),
]


def main():
    print('=' * 96)
    print('P2 dx self-check vs LEAKABLE DISPATCH GLOBALS')
    print('platform :', platform.platform(), '| python', sys.version.split()[0],
          '| numpy', np.__version__)
    print('=' * 96)
    hdr = '%-34s %-9s %-40s %s' % ('leaked state', 'route', 'drift %', 'FIRES')
    print(hdr)
    print('-' * len(hdr))
    silent = []
    for label, over in CASES:
        old = {k: getattr(F, k) for k in over}
        try:
            for k, v in over.items():
                setattr(F, k, v)
            try:
                r = run('nonconv', verbose=False)
            except Exception as exc:            # a leak that CRASHES is also a finding
                print('%-34s %-9s %-40s %s' % (label, '-', type(exc).__name__ + ': '
                                               + str(exc)[:34], 'ERROR'))
                continue
        finally:
            for k, v in old.items():
                setattr(F, k, v)
        print('%-34s %-9s %-40s %s'
              % (label, r['route'],
                 ', '.join('%s %.3f' % (k, v) for k, v in sorted(r['drift'].items())),
                 'YES' if r['fired'] else '*** NO ***'))
        if not r['fired']:
            silent.append(label)
        sys.stdout.flush()
    print('-' * len(hdr))
    if silent:
        print('STATES THAT SILENCE THE GUARD: %s' % '; '.join(silent))
    else:
        print('no swept dispatch-global silences the guard')


if __name__ == '__main__':
    main()
