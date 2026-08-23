"""ONE COLD apply_real_lens_traced call per process.

The inverse map is CACHED after the first call, so an in-process A/B measures
the second call's cost, not the build's.  Every arm here is a fresh
interpreter with the width pinned before NumPy loads.
argv: N sub det tag
"""
import hashlib
import os
import sys
import time
import warnings

import numpy as np

import lumenairy as la
from lumenairy.elements import _lens_traced as LT

_want = os.path.realpath(os.path.join(os.environ['LUMENAIRY_ROOT'],
                                      'lumenairy'))
assert os.path.realpath(os.path.dirname(la.__file__)) == _want, la.__file__

LAM = 1.31e-6
N = int(sys.argv[1])
sub = None if sys.argv[2] == 'none' else int(sys.argv[2])
det = bool(int(sys.argv[3]))
tag = sys.argv[4] if len(sys.argv) > 4 else None
LT.DETERMINISTIC_TRACED_FIT = det

DX = 30e-6
x = (np.arange(N) - N / 2) * DX
X, Y = np.meshgrid(x, x)
w = int(0.15625 * N) * DX
E = (np.exp(-(X * X + Y * Y) / (w * w))
     * np.exp(1j * (2 * np.pi / LAM) * (X * X + Y * Y) / 2.0)
     ).astype(np.complex128)
presc = {'wavelength': LAM, 'aperture_diameter': 14e-3, 'surfaces': [
    {'radius': 51.68e-3, 'thickness': 4e-3, 'glass_before': 'air',
     'glass_after': 'N-BK7', 'semi_diameter': 7e-3},
    {'radius': -51.68e-3, 'thickness': 0.0, 'glass_before': 'N-BK7',
     'glass_after': 'air', 'semi_diameter': 7e-3}],
    'thicknesses': [4e-3], 'stop_index': 0}

log = []
orig = LT._solve_lstsq_thread_safe


def spy(A, b, deterministic=False):
    t0 = time.perf_counter()
    out = orig(A, b, deterministic=deterministic)
    log.append((np.shape(A), time.perf_counter() - t0))
    return out


LT._solve_lstsq_thread_safe = spy
kw = {} if sub is None else {'ray_subsample': sub}
t0 = time.perf_counter()
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    F = la.apply_real_lens_traced(E, prescription=presc, wavelength=LAM,
                                  dx=DX, carrier='auto',
                                  on_undersample='silent', **kw)
dt = time.perf_counter() - t0
st = sum(e[1] for e in log)
F = np.asarray(F)
print('COLD N=%d sub=%s det=%d OMP=%s  %.3f s  solves %.4f s (%.2f%%)  '
      'FIELD %s  shapes %s'
      % (N, sub, det, os.environ.get('OMP_NUM_THREADS'), dt, st,
         100.0 * st / dt,
         hashlib.sha256(np.ascontiguousarray(F).tobytes()).hexdigest()[:16],
         [e[0] for e in log]))
if tag:
    np.save(tag, F)
