"""v1: ``tracemalloc`` peak for the three routes, cold caches, NO timing.

29 shapes: ``N`` in {64,128,256,512,1024} x ``M`` in {32,64,128,256,512},
plus 2048x1024, 1448x1448, 1024x1024 and 1536x768.

Every lumenairy cache in the central registry is dropped before each
measurement, together with SciPy's pocketfft plan cache and pyFFTW's wisdom /
plan caches where they exist, and a ``gc.collect()``.  No wall clock is read
anywhere in this file -- the timing ladder belongs to another stream.

Usage::  python v1_mem.py <tree> <out.json>
"""
from __future__ import annotations

import gc
import os
import sys
import tracemalloc

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np                                            # noqa: E402
from vlib import anchor, build_tag, write_json                # noqa: E402

TREE = sys.argv[1]
OUT = sys.argv[2]
anchor(TREE)

from lumenairy._cache_registry import (                       # noqa: E402
    clear_all_registered_caches, list_registered_cache_clearers)
from lumenairy.propagators._bluestein import (                # noqa: E402
    _bluestein_2d, _clear_h_fft_cache)
from lumenairy.propagators.fft_infra import _fft2, _ifft2     # noqa: E402


def cold():
    """Drop every cache this process can reach, then collect."""
    clear_all_registered_caches()
    _clear_h_fft_cache()
    try:
        import scipy.fft
        scipy.fft.set_global_backend(scipy.fft._backend._DEFAULT_BACKEND)
    except Exception:                                         # noqa: BLE001
        pass
    for mod, attr in (('scipy.fft._pocketfft.helper', '_cache_info'),):
        try:
            m = __import__(mod, fromlist=['x'])
            getattr(m, attr)
        except Exception:                                     # noqa: BLE001
            pass
    try:
        import pyfftw
        pyfftw.forget_wisdom()
        pyfftw.interfaces.cache.disable()
    except Exception:                                         # noqa: BLE001
        pass
    gc.collect()


SHAPES = [(N, M) for N in (64, 128, 256, 512, 1024)
          for M in (32, 64, 128, 256, 512)]
SHAPES += [(2048, 1024), (1448, 1448), (1024, 1024), (1536, 768)]

rng = np.random.default_rng(4242)
rows = []
for (N, M) in SHAPES:
    ax = (np.arange(N) - N // 2) * 8e-6
    X, Y = np.meshgrid(ax, ax)
    E = (np.exp(-(X ** 2 + Y ** 2) / (60e-6 ** 2))
         * (1.0 + 0.1 * (rng.standard_normal((N, N))
                         + 1j * rng.standard_normal((N, N))))
         ).astype(np.complex128)
    alpha = 1.0 / float(N)
    peaks = {}
    for name, kw in (('chirpz2d', dict(separable=False, method='auto')),
                     ('separable', dict(separable=True, method='auto')),
                     ('dense', dict(method='direct'))):
        cold()
        tracemalloc.start()
        tracemalloc.reset_peak()
        F = _bluestein_2d(E, alpha, alpha, M, M, sign=-1, xp=np,
                          fft2=_fft2, ifft2=_ifft2, **kw)
        cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        del F
        peaks[name] = peak / 1e6
    ok = peaks['dense'] < peaks['separable'] < peaks['chirpz2d']
    rows.append(dict(
        N=N, M=M, chirpz2d_MB=peaks['chirpz2d'],
        separable_MB=peaks['separable'], dense_MB=peaks['dense'],
        ordering_holds=bool(ok),
        margin_dense_vs_separable=peaks['separable'] / peaks['dense'],
        margin_separable_vs_chirpz=peaks['chirpz2d'] / peaks['separable'],
    ))
    print(f"N={N:5d} M={M:5d} chirp={peaks['chirpz2d']:9.2f} "
          f"sep={peaks['separable']:9.2f} dense={peaks['dense']:9.2f} "
          f"ok={ok}", file=sys.stderr)
    del E

worst = min(rows, key=lambda r: min(r['margin_dense_vs_separable'],
                                    r['margin_separable_vs_chirpz']))
write_json(dict(build=build_tag(), tree=TREE,
                registered_clearers=list_registered_cache_clearers(),
                n_shapes=len(rows),
                all_hold=all(r['ordering_holds'] for r in rows),
                exceptions=[r for r in rows if not r['ordering_holds']],
                worst_margin_shape=worst, rows=rows), OUT)
