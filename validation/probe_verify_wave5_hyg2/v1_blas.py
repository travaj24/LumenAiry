"""v1: the BLAS kernel ladder for ``_direct_matrix_2d``.

One process = one (OPENBLAS_CORETYPE, OPENBLAS_NUM_THREADS) rung.  The rung is
set by the CALLER in the environment; this module only records what BLAS it
actually got and what the dense route produced on it.

The fixtures are the AUTHOR'S OWN, seed for seed, so the question "would any of
the 29 ids fail on a different BLAS kernel?" is answered on the exact numbers
the assertions read -- not on a lookalike.

Usage::  python v1_blas.py <tree> <out.json>
"""
from __future__ import annotations

import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np                                            # noqa: E402
from vlib import anchor, build_tag, write_json                # noqa: E402

TREE = sys.argv[1]
OUT = sys.argv[2]
anchor(TREE)

from scipy.fft import next_fast_len                           # noqa: E402
from lumenairy.propagators._bluestein import (                # noqa: E402
    _bluestein_2d, _bluestein_centred_2d, _clear_h_fft_cache,
    _direct_matrix_2d)
from lumenairy.propagators.fft_infra import _fft2, _ifft2     # noqa: E402
from lumenairy.propagators.mft import (                       # noqa: E402
    angular_spectrum_propagate_mft, fraunhofer_propagate_mft,
    fresnel_propagate_mft)

EPS = float(np.finfo(np.float64).eps)
WL = 633e-9


# ---- the author's fixtures, rebuilt verbatim (same seeds) -----------------
def _rand(ny, nx, seed=20260915):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((ny, nx))
            + 1j * rng.standard_normal((ny, nx))).astype(np.complex128)


def _gauss(n, dx, w, seed=3):
    ax = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(ax, ax)
    rng = np.random.default_rng(seed)
    speckle = 1.0 + 0.1 * (rng.standard_normal((n, n))
                           + 1j * rng.standard_normal((n, n)))
    return (np.exp(-(X ** 2 + Y ** 2) / (w * w)) * speckle).astype(
        np.complex128)


def _pairwise_reference(E, ax_, ay_, M_y, M_x, sign,
                        c_in=(0.0, 0.0), c_out=(0.0, 0.0)):
    ny, nx = E.shape
    n_x = np.arange(nx, dtype=np.float64) - float(c_in[0])
    n_y = np.arange(ny, dtype=np.float64) - float(c_in[1])
    out = np.empty((M_y, M_x), dtype=np.complex128)
    for ky in range(M_y):
        ty = ay_ * (float(ky) - float(c_out[1])) * n_y
        wy = np.exp(1j * sign * 2.0 * np.pi * (ty - np.rint(ty)))
        for kx in range(M_x):
            tx = ax_ * (float(kx) - float(c_out[0])) * n_x
            wx = np.exp(1j * sign * 2.0 * np.pi * (tx - np.rint(tx)))
            out[ky, kx] = np.sum(E * (wy[:, None] * wx[None, :]))
    return out


def _bars(E, M_y, M_x):
    ny, nx = E.shape
    n = ny * nx
    L = float(next_fast_len(int(max(ny, nx) + max(M_y, M_x) - 1)))
    g_pair = float(np.log2(max(n / 128.0, 2.0)) + 8.0)
    g_chirp = float(3.0 * np.log2(L * L))
    g_dense = float(np.sqrt(n))
    s = float(np.sum(np.abs(E)))
    return ((g_chirp + g_pair) * EPS * s, (g_dense + g_pair) * EPS * s, s)


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(
        np.asarray(a, dtype=np.complex128)).tobytes()).hexdigest()


def blas_info():
    info = {'env_CORETYPE': os.environ.get('OPENBLAS_CORETYPE'),
            'env_OPENBLAS_NUM_THREADS':
                os.environ.get('OPENBLAS_NUM_THREADS')}
    try:
        import threadpoolctl
        info['threadpool'] = threadpoolctl.threadpool_info()
    except Exception as exc:                                  # noqa: BLE001
        info['threadpool'] = f'unavailable: {exc}'
    try:
        cfg = np.show_config(mode='dicts')
        info['numpy_blas'] = cfg.get('Build Dependencies', {}).get('blas')
    except Exception as exc:                                  # noqa: BLE001
        info['numpy_blas'] = f'unavailable: {exc}'
    return info


R = {'build': build_tag(), 'tree': TREE, 'blas': blas_info(), 'rows': {}}


# ---- A. raw dense-route digests at several shapes -------------------------
for tag, (ny, nx, my, mx, ax_, ay_, sgn) in {
        'raw.16x16->8x8':  (16, 16, 8, 8, 1 / 64., 1 / 64., -1),
        'raw.32x24->16x20': (32, 24, 16, 20, 1 / 64., 1 / 64., -1),
        'raw.48x12->40x6': (48, 12, 40, 6, 0.013, 0.011, -1),
        'raw.128x128->64x64': (128, 128, 64, 64, 1 / 64., 1 / 64., -1),
        'raw.256x256->64x64': (256, 256, 64, 64, 1 / 64., 1 / 64., +1),
}.items():
    E = _rand(ny, nx, seed=20260915)
    F = _direct_matrix_2d(E, ax_, ay_, my, mx, sign=sgn, xp=np)
    R['rows'][tag] = {'sha256': sha(F),
                      'sum_re': float(np.sum(F.real)),
                      'sum_im': float(np.sum(F.imag)),
                      'max_abs': float(np.max(np.abs(F)))}

# ---- B. the author's assertion-bearing quantities -------------------------
# B1 test_every_route_agrees_with_the_pairwise_reference (6 ids)
for shape in ((16, 16, 8, 8), (32, 24, 16, 20), (17, 9, 5, 23)):
    ny, nx, my, mx = shape
    for sign in (-1, +1):
        E = _rand(ny, nx)
        alpha = 1.0 / 64.0
        ref = _pairwise_reference(E, alpha, alpha, my, mx, sign)
        bar_chirp, bar_dense, _ = _bars(E, my, mx)
        row = {'bar_chirp': bar_chirp, 'bar_dense': bar_dense}
        got = {}
        for m, kw in (('bluestein', dict(separable=False, method='auto')),
                      ('separable', dict(separable=True, method='auto')),
                      ('direct', dict(method='direct'))):
            _clear_h_fft_cache()
            F = _bluestein_2d(E, alpha, alpha, my, mx, sign=sign, xp=np,
                              fft2=_fft2, ifft2=_ifft2, **kw)
            got[m] = F
            row[f'err_{m}'] = float(np.max(np.abs(F - ref)))
            row[f'sha_{m}'] = sha(F)
        for a in ('bluestein', 'separable'):
            row[f'pair_{a}_vs_direct'] = float(
                np.max(np.abs(got[a] - got['direct'])))
        row['bar_both'] = bar_chirp + bar_dense
        R['rows'][f'B1.{ny}x{nx}->{my}x{mx}.sign{sign}'] = row

# B2 test_the_centred_primitive_is_the_same_sum_on_the_dense_route
E = _rand(20, 24)
alpha = 1.0 / 48.0
my, mx = 12, 10
c_in = (24 / 2.0, 20 / 2.0)
c_out = (mx / 2.0 - 1.3, my / 2.0)
ref = _pairwise_reference(E, alpha, alpha, my, mx, -1, c_in=c_in, c_out=c_out)
bar_chirp, bar_dense, _ = _bars(E, my, mx)
kw = dict(n_centre_in_x=c_in[0], n_centre_in_y=c_in[1],
          k_centre_out_x=c_out[0], k_centre_out_y=c_out[1],
          sign=-1, xp=np, fft2=_fft2, ifft2=_ifft2)
F_b = _bluestein_centred_2d(E, alpha, alpha, my, mx, **kw)
F_d = _bluestein_centred_2d(E, alpha, alpha, my, mx, method='direct', **kw)
R['rows']['B2.centred'] = {
    'err_chirp': float(np.max(np.abs(F_b - ref))), 'bar_chirp': bar_chirp,
    'err_dense': float(np.max(np.abs(F_d - ref))), 'bar_dense': bar_dense,
    'sha_dense': sha(F_d), 'sha_chirp': sha(F_b)}

# B3 test_the_dense_kernel_serves_both_index_conventions (bit equality)
E = _rand(12, 14)
a = _direct_matrix_2d(E, 0.02, 0.03, 7, 9, sign=-1, xp=np)
b = _direct_matrix_2d(E, 0.02, 0.03, 7, 9, sign=-1, xp=np,
                      n_centre_in_x=0.0, n_centre_in_y=0.0,
                      k_centre_out_x=0.0, k_centre_out_y=0.0)
R['rows']['B3.both_conventions'] = {'bit_equal': bool(np.array_equal(
    np.ascontiguousarray(a).view(np.float64),
    np.ascontiguousarray(b).view(np.float64))), 'sha': sha(a)}

# B4 test_the_direct_route_reaches_the_same_physics (3 ids)
N, dx = 64, 8e-6
Eg = _gauss(N, dx, 60e-6)
z = 2e-2
dx_out = WL * z / (N * dx)
bar_chirp, bar_dense, s = _bars(Eg, N, N)
for fn in (fresnel_propagate_mft, fraunhofer_propagate_mft,
           angular_spectrum_propagate_mft):
    aa = fn(Eg, z, WL, dx, dx_out, N)
    bb = fn(Eg, z, WL, dx, dx_out, N, method='direct')
    rel = float(np.max(np.abs(aa - bb)) / np.max(np.abs(aa)))
    bar = (bar_chirp + bar_dense) / float(np.max(np.abs(Eg)))
    pa = float(np.sum(np.abs(aa) ** 2))
    pb = float(np.sum(np.abs(bb) ** 2))
    R['rows'][f'B4.{fn.__name__}'] = {
        'rel': rel, 'bar': bar, 'power_rel': abs(pa - pb) / pa,
        'power_bar': 1e-12, 'sha_direct': sha(bb), 'sha_default': sha(aa)}

# B5 test_the_association_order_is_a_function_of_the_shapes_alone
E = _rand(48, 12)
my, mx = 40, 6
ny, nx = E.shape
runs = [_direct_matrix_2d(E, 0.013, 0.011, my, mx, sign=-1, xp=np)
        for _ in range(3)]
cost_y_first = my * ny * nx + my * nx * mx
cost_x_first = ny * nx * mx + my * ny * mx


def _kernel(alpha_, n_in, n_out, c_in_, c_out_):
    n = np.arange(int(n_in), dtype=np.float64) - float(c_in_)
    k = np.arange(int(n_out), dtype=np.float64) - float(c_out_)
    t = float(alpha_) * k[:, None] * n[None, :]
    return np.exp(1j * -1 * 2.0 * np.pi * (t - np.rint(t)))


Wx = _kernel(0.013, nx, mx, 0.0, 0.0)
Wy = _kernel(0.011, ny, my, 0.0, 0.0)
y_first = (Wy @ E) @ Wx.T
x_first = Wy @ (E @ Wx.T)
cheaper = y_first if cost_y_first <= cost_x_first else x_first
dearer = x_first if cost_y_first <= cost_x_first else y_first


def _biteq(p, q):
    return bool(np.array_equal(np.ascontiguousarray(p).view(np.float64),
                               np.ascontiguousarray(q).view(np.float64)))


R['rows']['B5.association'] = {
    'runs_reproducible': all(_biteq(runs[0], r) for r in runs[1:]),
    'matches_cheaper': _biteq(runs[0], cheaper),
    'cheaper_ne_dearer': not _biteq(cheaper, dearer),
    'max_abs_cheaper_minus_dearer': float(np.max(np.abs(cheaper - dearer))),
    'sha_run0': sha(runs[0]), 'sha_cheaper': sha(cheaper),
    'sha_dearer': sha(dearer)}

# B6 test_the_chirp_phase_guard... (dense arm's bar)
Np, Mp = 24, 12
Ep = _rand(Np, Np, seed=77)
alpha = 1e17 / float(Np) ** 2
refp = _pairwise_reference(Ep, alpha, alpha, Mp, Mp, -1)
_, bar_dense_p, _ = _bars(Ep, Mp, Mp)
dense = _bluestein_2d(Ep, alpha, alpha, Mp, Mp, sign=-1, xp=np,
                      fft2=_fft2, ifft2=_ifft2, method='direct')
import warnings as _w                                         # noqa: E402
with _w.catch_warnings(record=True) as caught:
    _w.simplefilter('always')
    chirp = _bluestein_2d(Ep, alpha, alpha, Mp, Mp, sign=-1, xp=np,
                          fft2=_fft2, ifft2=_ifft2)
R['rows']['B6.phase_guard'] = {
    'err_dense': float(np.max(np.abs(dense - refp))),
    'bar_dense': bar_dense_p,
    'rel_chirp': float(np.linalg.norm(chirp - refp)
                       / np.linalg.norm(refp)),
    'warned': [str(w.message)[:60] for w in caught],
    'sha_dense': sha(dense)}

# B7 dtype contract
out32 = _bluestein_2d(_rand(16, 16).astype(np.complex64), 0.01, 0.01, 8, 8,
                      sign=-1, xp=np, fft2=_fft2, ifft2=_ifft2,
                      method='direct')
R['rows']['B7.dtype'] = {'c64': str(out32.dtype),
                         'sha_c64_as_c128': sha(out32)}

write_json(R, OUT)
