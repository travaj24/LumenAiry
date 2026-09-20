"""v1: re-derive the H2-1 agreement tolerance from scratch and measure it.

Two references are built here, not one:

* ``ref_pair``  -- the same sum, one output point at a time, with ``np.sum``
  (NumPy pairwise).  This is the SHAPE of the author's reference, rebuilt
  independently, so the author's table can be reproduced.
* ``ref_fsum``  -- the same sum with ``math.fsum`` on the real and imaginary
  parts separately.  ``fsum`` is CORRECTLY ROUNDED, so its summation growth
  factor is 1/2 ulp: it removes the reference's own summation error from the
  reading entirely.  Any departure measured against it belongs to the route.

Bars, in the author's ``g * eps * sum|E|`` convention with
``eps = np.finfo(float64).eps = 2u`` (``u = 2**-53`` is the unit roundoff):

* ``g_pair   = log2(n/128) + 8``          (author's, for the pairwise ref)
* ``g_chirp  = 3 * log2(L^2)``            (author's, three FFTs of length L^2)
* ``g_dense  = sqrt(n)``                  (author's, "two BLAS products")
* ``g_rig1   = n / 2``                    RIGOROUS Higham worst case for ONE
  length-``n`` inner product: ``|fl - exact| <= gamma_n * sum|terms|`` with
  ``gamma_n = n*u/(1-n*u)``; ``n*u = (n/2)*eps``.
* ``g_rig2   = (Nx + Ny) / 2``            RIGOROUS worst case for the dense
  route AS WRITTEN -- TWO chained products of lengths ``Nx`` and ``Ny``, not
  one of length ``n = Nx*Ny``.
* ``g_fsum   = 0.5``                      correctly-rounded reference.
"""
from __future__ import annotations

import json
import math
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
    _bluestein_2d, _clear_h_fft_cache)
from lumenairy.propagators.fft_infra import _fft2, _ifft2     # noqa: E402

EPS = float(np.finfo(np.float64).eps)
U = EPS / 2.0


def rand(ny, nx, seed):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((ny, nx))
            + 1j * rng.standard_normal((ny, nx))).astype(np.complex128)


def _weights(alpha, n_in, n_out, sign):
    n = np.arange(int(n_in), dtype=np.float64)
    k = np.arange(int(n_out), dtype=np.float64)
    t = float(alpha) * k[:, None] * n[None, :]
    return np.exp(1j * sign * 2.0 * np.pi * (t - np.rint(t)))


def references(E, alpha, my, mx, sign, do_fsum=True):
    """``(ref_pair, ref_fsum)`` -- the same terms, two summation rules."""
    Wy = _weights(alpha, E.shape[0], my, sign)
    Wx = _weights(alpha, E.shape[1], mx, sign)
    rp = np.empty((my, mx), np.complex128)
    rf = np.empty((my, mx), np.complex128) if do_fsum else None
    for ky in range(my):
        for kx in range(mx):
            term = E * (Wy[ky][:, None] * Wx[kx][None, :])
            rp[ky, kx] = np.sum(term)
            if do_fsum:
                flat = term.ravel()
                rf[ky, kx] = complex(math.fsum(flat.real.tolist()),
                                     math.fsum(flat.imag.tolist()))
    return rp, rf


CASES = ((16, 8), (32, 16), (48, 24), (96, 48), (128, 64))
ALPHA = 1.0 / 64.0
SIGN = -1
rows = []
for (N, M) in CASES:
    E = rand(N, N, seed=90210 + N)
    s = float(np.sum(np.abs(E)))
    n = N * N
    L = float(next_fast_len(int(N + M - 1)))
    g_pair = float(np.log2(max(n / 128.0, 2.0)) + 8.0)
    g_chirp = float(3.0 * np.log2(L * L))
    g_dense = float(np.sqrt(n))
    g_rig1 = n / 2.0
    g_rig2 = (N + N) / 2.0
    ref_pair, ref_fsum = references(E, ALPHA, M, M, SIGN)
    routes = {}
    for name, kw in (('bluestein', dict(separable=False, method='auto')),
                     ('separable', dict(separable=True, method='auto')),
                     ('direct', dict(method='direct'))):
        _clear_h_fft_cache()
        F = _bluestein_2d(E, ALPHA, ALPHA, M, M, sign=SIGN, xp=np,
                          fft2=_fft2, ifft2=_ifft2, **kw)
        g = g_dense if name == 'direct' else g_chirp
        bar_auth = (g + g_pair) * EPS * s
        bar_rig = ((g_rig2 if name == 'direct' else g_chirp) + g_pair) \
            * EPS * s
        bar_fsum = ((g_rig2 if name == 'direct' else g_chirp) + 0.5) * EPS * s
        e_pair = float(np.max(np.abs(F - ref_pair)))
        e_fsum = float(np.max(np.abs(F - ref_fsum)))
        routes[name] = dict(
            max_abs_vs_pairwise=e_pair,
            max_abs_vs_fsum=e_fsum,
            rel_L2_vs_fsum=float(np.linalg.norm(F - ref_fsum)
                                 / np.linalg.norm(ref_fsum)),
            bar_author=bar_auth,
            bar_rigorous=bar_rig,
            bar_rigorous_vs_fsum_ref=bar_fsum,
            decades_author=float(np.log10(bar_auth / e_pair)),
            decades_rigorous=float(np.log10(bar_rig / e_pair)),
            decades_rigorous_fsum=float(np.log10(bar_fsum / e_fsum)),
            crossed_author=bool(e_pair >= bar_auth),
            crossed_rigorous=bool(e_pair >= bar_rig),
        )
    rows.append(dict(
        N=N, M=M, n=n, L=L, sum_abs_E=s,
        max_abs_F=float(np.max(np.abs(ref_fsum))),
        min_abs_F=float(np.min(np.abs(ref_fsum))),
        kappa_max=float(np.max(s / np.abs(ref_fsum))),
        pairwise_ref_error_vs_fsum=float(np.max(np.abs(ref_pair - ref_fsum))),
        g_pair=g_pair, g_chirp=g_chirp, g_dense_author=g_dense,
        g_rigorous_one_product=g_rig1, g_rigorous_two_products=g_rig2,
        ratio_author_over_rigorous2=g_dense / g_rig2,
        bar_dense_author=(g_dense + g_pair) * EPS * s,
        bar_dense_rigorous=(g_rig2 + g_pair) * EPS * s,
        decades_bar_below_peak=float(
            np.log10(np.max(np.abs(ref_fsum))
                     / ((g_dense + g_pair) * EPS * s))),
        decades_bar_below_smallest=float(
            np.log10(np.min(np.abs(ref_fsum))
                     / ((g_dense + g_pair) * EPS * s))),
        routes=routes,
    ))
    print(f"N={N} M={M} done", file=sys.stderr)

# The AM-GM statement, on rectangular INPUT grids -- which is what the
# author's own fixtures use (32x24, 17x9, 20x24) and what a non-square
# readout would use.
amgm = []
for (ny, nx) in ((16, 16), (32, 24), (17, 9), (20, 24), (48, 12),
                 (1024, 4), (64, 64), (128, 32)):
    amgm.append(dict(Ny=ny, Nx=nx,
                     g_author_sqrt_n=float(np.sqrt(ny * nx)),
                     g_rigorous_two_products=(ny + nx) / 2.0,
                     author_over_rigorous=float(np.sqrt(ny * nx)
                                                / ((ny + nx) / 2.0))))

write_json(dict(build=build_tag(), tree=TREE, eps=EPS, u=U,
                alpha=ALPHA, sign=SIGN, cases=rows, amgm=amgm), OUT)
print(json.dumps([{k: r[k] for k in ('N', 'M')} for r in rows]))
