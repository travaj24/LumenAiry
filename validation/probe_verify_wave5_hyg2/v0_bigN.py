import os, sys, json
sys.path.insert(0, r'C:/tmp/lum_vhyg2/validation/probe_verify_wave5_hyg2')
import vlib
vlib.anchor(r'C:/tmp/lum_vhyg2/lumenairy')
import numpy as np
from scipy.fft import next_fast_len
from lumenairy.propagators._bluestein import _bluestein_2d, _clear_h_fft_cache
from lumenairy.propagators.fft_infra import _fft2, _ifft2
EPS = float(np.finfo(np.float64).eps)

def pairwise_ref(E, a, My, Mx, sign):
    ny, nx = E.shape
    n_x = np.arange(nx, dtype=np.float64); n_y = np.arange(ny, dtype=np.float64)
    out = np.empty((My, Mx), dtype=np.complex128)
    for ky in range(My):
        ty = a * float(ky) * n_y
        wy = np.exp(1j*sign*2.0*np.pi*(ty - np.rint(ty)))
        Ewy = E * wy[:, None]
        for kx in range(Mx):
            tx = a * float(kx) * n_x
            wx = np.exp(1j*sign*2.0*np.pi*(tx - np.rint(tx)))
            out[ky, kx] = np.sum(Ewy * wx[None, :])
    return out

rows = []
for (n_, m_) in ((192, 96), (256, 128)):
    rng = np.random.default_rng(20260915)
    E = (rng.standard_normal((n_, n_)) + 1j*rng.standard_normal((n_, n_))).astype(np.complex128)
    a = 1.0/64.0
    ref = pairwise_ref(E, a, m_, m_, -1)
    n = n_*n_
    L = float(next_fast_len(int(n_ + m_ - 1)))
    g_pair = float(np.log2(max(n/128.0, 2.0)) + 8.0)
    g_chirp = float(3.0*np.log2(L*L)); g_dense = float(np.sqrt(n))
    s = float(np.sum(np.abs(E))); sig = float(np.max(np.abs(ref)))
    for method, kw in (('bluestein', dict(separable=False, method='auto')),
                       ('separable', dict(separable=True, method='auto')),
                       ('direct', dict(method='direct'))):
        _clear_h_fft_cache()
        F = _bluestein_2d(E, a, a, m_, m_, sign=-1, xp=np, fft2=_fft2, ifft2=_ifft2, **kw)
        err = float(np.max(np.abs(F - ref)))
        bar = ((g_dense if method=='direct' else g_chirp) + g_pair)*EPS*s
        rows.append(dict(N=n_, M=m_, method=method, maxabs=err, bar=bar,
                         decades=float(np.log10(bar/err)), signal=sig,
                         bar_over_signal=bar/sig))
        print(f"N={n_} M={m_} {method:10s} err={err:.4e} bar={bar:.4e} dec={np.log10(bar/err):.3f} bar/sig={bar/sig:.2e}", flush=True)
json.dump(rows, open(r'C:/tmp/lum_vhyg2/validation/probe_verify_wave5_hyg2/v0_bigN_win.json','w'), indent=1)
