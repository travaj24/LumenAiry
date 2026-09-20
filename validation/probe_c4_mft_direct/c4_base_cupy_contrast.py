import sys, numpy as np, lumenairy, os
print('bound:', os.path.realpath(lumenairy.__file__))
assert 'lum_c4_base' in os.path.realpath(lumenairy.__file__), 'WRONG TREE'
from lumenairy.propagators import _bluestein as B
print('has rule:', hasattr(B, '_auto_selects_direct'))
import cupy as cp
rng = np.random.default_rng(7)
for (ny, mx) in ((64, 2), (128, 4), (256, 8), (96, 3)):
    E = (rng.standard_normal((ny, ny)) + 1j*rng.standard_normal((ny, ny))).astype(np.complex128)
    a = 1e3/float(ny)**2
    try:
        B._bluestein_2d(cp.asarray(E), a, a, mx, mx, sign=-1, xp=cp,
                        fft2=cp.fft.fft2, ifft2=cp.fft.ifft2)
        print(f'BASE {ny}->{mx}: ran')
    except Exception as e:
        print(f'BASE {ny}->{mx}: {type(e).__name__}: {str(e)[:70]}')

# MEASURED 2026-09-20 on this box (broken cuFFT DLL), run from a neutral cwd
# with PYTHONPATH naming the BASE extraction:
#
#   bound: C:\tmp\lum_c4_base\lumenairy\__init__.py
#   has rule: False
#   BASE 64->2:  ImportError: DLL load failed while importing cufft
#   BASE 128->4: ImportError: DLL load failed while importing cufft
#   BASE 256->8: ImportError: DLL load failed while importing cufft
#   BASE 96->3:  ImportError: DLL load failed while importing cufft
#
# The BRANCH runs all four (c4_backends_win.json): the rule sends them to the
# dense route, which uses no FFT.  The chirp-side shapes still raise on both
# trees, which is the other half of the two-sided reading.
#
# THE TRAP THIS FILE IS A SCRIPT FOR.  Run as ``python - <<EOF`` from inside
# the worktree, the same code bound the WORKTREE's lumenairy and printed
# "has rule: True", because ``''`` (the cwd) precedes PYTHONPATH on sys.path
# for a stdin script.  The assert above is what turned that into a refusal.
