"""VERIFY-WP-C3 -- validate MY oracle and MY upsampler before either is used.

    python probe_oracle_validate.py <tree> <out.json>
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_TREE = os.path.abspath(sys.argv[1])
sys.path.insert(0, _TREE)

import numpy as np                                     # noqa: E402
import vclib as V                                      # noqa: E402

V.anchor(_TREE)

out = {}

# --- (0) the upsampler, on an analytic band-limited function ---------------
rows = []
rng = np.random.default_rng(7)
for n in (64, 256):
    d = 3.0e-6
    x = V.axis(n, d)
    L = n * d
    # a trigonometric polynomial of period L with |m| < n/2 -- exactly
    # band-limited on this lattice, so sinc interpolation is EXACT
    ms = np.arange(-(n // 2) + 1, n // 2)
    co = (rng.normal(size=ms.size) + 1j * rng.normal(size=ms.size))
    def f(u, ms=ms, co=co, L=L):
        return (co[None, :] * np.exp(2j * np.pi * ms[None, :] * u[:, None] / L)
                ).sum(axis=1)
    A = np.outer(f(x), np.ones(1)).T * 0 + f(x)[None, :] * np.ones((3, 1))
    for F in (2, 4, 8):
        fine = V._sinc_upsample_axis(A, F, -1)
        xf = V.axis(n * F, d / F)
        ref = np.ones((3, 1)) * f(xf)[None, :]
        rows.append({'n': n, 'F': F, 'rel_l2': V.rel_l2(fine, ref)})
out['upsampler'] = rows

# --- (1) paraxial Helmholtz residual --------------------------------------
out['pde'] = [V.validate_oracle_pde(N=n, h=h)
              for n, h in ((192, 2.0e-6), (256, 2.0e-6), (256, 4.0e-6),
                           (384, 2.0e-6), (512, 2.0e-6), (768, 2.0e-6))]

# --- (2) oracle vs an oversampled transfer-function propagation -----------
out['prop'] = {
    'their_fixture': V.validate_oracle_prop(
        w=0.30e-3, R=-40.0e-3, lam=1.064e-6, N=2048, span_w=8.0),
    'my_fixture': V.validate_oracle_prop(
        w=0.22e-3, R=-25.0e-3, lam=0.633e-6,
        legs=(3.125e-3, 12.5e-3, 25.625e-3, 37.5e-3), N=2048, span_w=8.0),
}

# --- (3) the truncation floor by quadrature, refined ----------------------
def _win(N, dx):
    return (-(N / 2.0) * dx, (N / 2.0 - 1.0) * dx)

w_t, w_m = 0.30e-3, 0.22e-3
dx_t = 6.0 * w_t / 512
dx_m = 6.0 * w_m / 768
out['floor'] = {
    'their_N512_6radii': V.truncation_floor(w_t, *_win(512, dx_t)),
    'their_N768_9radii': V.truncation_floor(w_t, *_win(768, dx_t)),
    'their_N1024_12radii': V.truncation_floor(w_t, *_win(1024, dx_t)),
    'mine_N768_6radii': V.truncation_floor(w_m, *_win(768, dx_m)),
    'mine_N1152_9radii': V.truncation_floor(w_m, *_win(1152, dx_m)),
    'mine_N1536_12radii': V.truncation_floor(w_m, *_win(1536, dx_m)),
}

V.write_json(sys.argv[2], out)
for r in out['upsampler']:
    print('upsample', r)
for r in out['pde']:
    print('pde', {k: r[k] for k in ('N', 'h', 'residual_rel')})
for tag, blk in out['prop'].items():
    for r in blk['rows']:
        print('prop', tag, r)
for tag, blk in out['floor'].items():
    print('floor', tag, 'closed', f"{blk['closed_form']:.6e}",
          'quad', [f"{lv['floor']:.6e}" for lv in blk['levels']])
