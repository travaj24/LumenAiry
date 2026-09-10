"""V17 -- is numpy's FFT still double-only?

``_fourier_upsample_crop``'s v5.44 note (and the restated pin in
``test_niche_perf_round2_2026_08_10.py``) says "numpy's FFT is still
double-only, so a complex64 input is transformed in complex128 and NARROWED
back on return".  That decides whether a complex64 envelope keeps double
precision through the transform pair or does not, so it is measured rather
than assumed.

Usage:  python v17_numpy_fft_dtype.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fix
import numpy as np

la = _fix.banner()

from lumenairy.propagators import fft_infra as FI
from lumenairy.propagators.fft_infra import _fft2

print('fft backend attrs:', [a for a in dir(FI) if 'backend' in a.lower() or 'PYFFTW' in a or 'SCIPY' in a][:20])
for f in ('_FFT_BACKEND', 'FFT_BACKEND', '_HAVE_PYFFTW', '_HAVE_SCIPY', 'get_fft_backend'):
    if hasattr(FI, f):
        v = getattr(FI, f)
        print(' ', f, '=', v() if callable(v) else v)
a = np.ones((8,8), dtype=np.complex64)
print('_fft2(c64).dtype =', _fft2(a).dtype)
print('np.fft.fft2(c64).dtype =', np.fft.fft2(a).dtype)
print('auto_promote =', la.get_fft_auto_promote())
la.set_fft_auto_promote(False)
print('after off: _fft2(c64).dtype =', _fft2(np.ones((8,8),dtype=np.complex64)).dtype)
