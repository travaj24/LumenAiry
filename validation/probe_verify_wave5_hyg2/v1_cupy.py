"""v1: the CuPy arm of ``_direct_matrix_2d`` -- agreement, FFT-freedom, and
WHERE the kernels are actually built.

Usage::  python v1_cupy.py <tree> <out.json>
"""
from __future__ import annotations

import inspect
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np                                            # noqa: E402
from vlib import anchor, build_tag, write_json                # noqa: E402

TREE = sys.argv[1]
OUT = sys.argv[2]
anchor(TREE)

from lumenairy.propagators._bluestein import (                # noqa: E402
    _bluestein_2d, _bluestein_centred_2d, _direct_matrix_2d)

R = {'build': build_tag(), 'tree': TREE}

# ---- structural: no FFT anywhere in the signature or the body -------------
sig = inspect.signature(_direct_matrix_2d)
src = inspect.getsource(_direct_matrix_2d)
body = src.split('"""')[-1]
R['signature'] = str(sig)
R['sig_has_fft2'] = 'fft2' in sig.parameters
R['sig_has_ifft2'] = 'ifft2' in sig.parameters
R['body_mentions'] = {k: (k in body) for k in
                      ('fft', 'next_fast_len', 'pad(', 'exp', 'matmul',
                       'rint')}
R['body_matmul_count'] = body.count('matmul')

# ---- WHERE are the kernels built?  host or device? ------------------------
# The builder is the nested ``_kernel``; read its text out of the source.
m = re.search(r"def _kernel\(.*?\n(?=\n    Wx_np)", src, re.S)
kern = m.group(0) if m else ''
R['kernel_source'] = kern
R['kernel_uses_np_arange'] = 'np.arange' in kern
R['kernel_uses_np_exp'] = 'np.exp' in kern
R['kernel_uses_xp'] = bool(re.search(r"\bxp\.", kern))
R['host_build_then_asarray'] = bool(
    re.search(r"Wx, Wy = xp\.asarray\(Wx_np\), xp\.asarray\(Wy_np\)", src))

# ---- numerical: CuPy vs NumPy --------------------------------------------
try:
    import cupy as cp
    R['cupy_version'] = cp.__version__
    R['cupy_device'] = str(cp.cuda.runtime.getDeviceProperties(0)['name'])
    try:
        cp.fft.fft2(cp.zeros((8, 8), dtype=cp.complex128))
        R['cufft_works'] = True
    except Exception as exc:                                  # noqa: BLE001
        R['cufft_works'] = False
        R['cufft_error'] = f'{type(exc).__name__}: {str(exc)[:200]}'
    rows = {}
    rng = np.random.default_rng(20260919)
    for (ny, nx, my, mx) in ((16, 16, 8, 8), (64, 64, 32, 32),
                             (128, 96, 48, 64)):
        E = (rng.standard_normal((ny, nx))
             + 1j * rng.standard_normal((ny, nx))).astype(np.complex128)
        Fn = _direct_matrix_2d(E, 0.013, 0.011, my, mx, sign=-1, xp=np)
        Fc = cp.asnumpy(_direct_matrix_2d(cp.asarray(E), 0.013, 0.011,
                                          my, mx, sign=-1, xp=cp))
        rows[f'{ny}x{nx}->{my}x{mx}'] = {
            'dtype_numpy': str(Fn.dtype), 'dtype_cupy': str(Fc.dtype),
            'rel_L2': float(np.linalg.norm(Fc - Fn)
                            / np.linalg.norm(Fn)),
            'max_abs': float(np.max(np.abs(Fc - Fn))),
            'rel_max': float(np.max(np.abs(Fc - Fn))
                             / np.max(np.abs(Fn))),
            'bit_identical': bool(np.array_equal(
                np.ascontiguousarray(Fn).view(np.float64),
                np.ascontiguousarray(Fc).view(np.float64))),
        }
    # centred convention on the device, and the chirp route for contrast
    E = (rng.standard_normal((32, 32))
         + 1j * rng.standard_normal((32, 32))).astype(np.complex128)
    Fn = _bluestein_centred_2d(E, 0.02, 0.02, 16, 16, sign=-1, xp=np,
                               fft2=None, ifft2=None, method='direct')
    Fc = cp.asnumpy(_bluestein_centred_2d(cp.asarray(E), 0.02, 0.02, 16, 16,
                                          sign=-1, xp=cp, fft2=None,
                                          ifft2=None, method='direct'))
    rows['centred.32->16'] = {
        'rel_L2': float(np.linalg.norm(Fc - Fn) / np.linalg.norm(Fn)),
        'note': 'fft2/ifft2 passed as None -- the dense route must not use '
                'them'}
    try:
        _bluestein_2d(cp.asarray(E), 0.02, 0.02, 16, 16, sign=-1, xp=cp,
                      fft2=cp.fft.fft2, ifft2=cp.fft.ifft2)
        rows['chirp_on_cupy'] = 'ran'
    except Exception as exc:                                  # noqa: BLE001
        rows['chirp_on_cupy'] = f'{type(exc).__name__}: {str(exc)[:160]}'
    R['cupy'] = rows
except Exception as exc:                                      # noqa: BLE001
    R['cupy'] = f'unavailable: {type(exc).__name__}: {str(exc)[:200]}'

write_json(R, OUT)
