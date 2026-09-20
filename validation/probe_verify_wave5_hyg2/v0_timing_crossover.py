"""The TIME crossover between the three routes through the MFT sum, and the
MEMORY ordering, re-measured independently.

Timing and tracemalloc are separate passes: tracemalloc charges per
allocation and the chirp-Z route allocates far more objects, so timing inside
it would bias the very comparison the table is for.

The box's load is recorded IN the JSON (process count and a measured
single-thread reference loop), so a contended reading is visible rather than
argued.
"""
from __future__ import annotations
import gc
import json
import os
import subprocess
import sys
import time
import tracemalloc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
vlib.anchor(os.path.join(ROOT, 'lumenairy'))

import numpy as np  # noqa: E402
from lumenairy.propagators._bluestein import (  # noqa: E402
    _bluestein_2d, _clear_h_fft_cache)
from lumenairy.propagators.fft_infra import _fft2, _ifft2  # noqa: E402

SHAPES = [(N, M) for N in (64, 128, 256, 512, 1024)
          for M in (32, 64, 128, 256, 512)] + \
         [(2048, 1024), (1448, 1448), (512, 512), (2048, 64)]

ROUTES = (('chirpz2d', dict(separable=False, method='auto')),
          ('separable', dict(separable=True, method='auto')),
          ('dense', dict(method='direct')))


def _load_snapshot():
    """How busy is the box?  Both a process census and a measured
    single-thread reference loop, so a contended reading is visible."""
    try:
        if sys.platform.startswith('win'):
            out = subprocess.run(['tasklist'], capture_output=True, text=True,
                                 timeout=60).stdout
            npy = sum(1 for ln in out.splitlines()
                      if ln.lower().startswith('python'))
            nproc = len(out.splitlines())
        else:
            out = subprocess.run(['ps', '-e'], capture_output=True, text=True,
                                 timeout=60).stdout
            npy = sum(1 for ln in out.splitlines() if 'python' in ln)
            nproc = len(out.splitlines())
    except Exception as exc:                        # noqa: BLE001 -- recorded
        npy, nproc = -1, f"{type(exc).__name__}"
    a = np.ones(1 << 16)
    t0 = time.perf_counter()
    for _ in range(200):
        a = a * 1.0000001
    ref = time.perf_counter() - t0
    return {'python_processes': npy, 'total_processes': nproc,
            'reference_loop_s': ref}


def _cold():
    _clear_h_fft_cache()
    try:
        import scipy.fft
        scipy.fft.set_global_backend(scipy.fft.get_global_backend())
    except Exception:                               # noqa: BLE001
        pass
    gc.collect()


def _time_one(E, N, M, kw, repeats=5):
    best = float('inf')
    for _ in range(repeats):
        _cold()
        t0 = time.perf_counter()
        _bluestein_2d(E, 1.0 / 64.0, 1.0 / 64.0, M, M, sign=-1, xp=np,
                      fft2=_fft2, ifft2=_ifft2, **kw)
        best = min(best, time.perf_counter() - t0)
    return best


def _peak_one(E, N, M, kw):
    _cold()
    tracemalloc.start()
    _bluestein_2d(E, 1.0 / 64.0, 1.0 / 64.0, M, M, sign=-1, xp=np,
                  fft2=_fft2, ifft2=_ifft2, **kw)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def main():
    out = {'build': vlib.build_tag(), 'load_before': _load_snapshot(),
           'time': [], 'memory': []}
    rng = np.random.default_rng(4)
    # -------- memory pass (no timing inside it) --------
    mem_shapes = ([] if os.environ.get('VHYG2_SKIP_MEM') else SHAPES)
    for (N, M) in mem_shapes:
        E = (rng.standard_normal((N, N))
             + 1j * rng.standard_normal((N, N))).astype(np.complex128)
        peaks = {}
        for name, kw in ROUTES:
            peaks[name] = _peak_one(E, N, M, kw)
        ok = peaks['dense'] < peaks['separable'] < peaks['chirpz2d']
        out['memory'].append({'N': N, 'M': M, 'peak_bytes': peaks,
                              'ordering_holds': bool(ok)})
        print(f"MEM N={N:5d} M={M:5d}  dense={peaks['dense']/1e6:9.2f} "
              f"sep={peaks['separable']/1e6:9.2f} "
              f"chirp={peaks['chirpz2d']/1e6:9.2f}  "
              f"{'OK' if ok else 'ORDERING BROKEN'}", flush=True)
        del E
        gc.collect()
    # -------- timing pass --------
    for (N, M) in SHAPES:
        E = (rng.standard_normal((N, N))
             + 1j * rng.standard_normal((N, N))).astype(np.complex128)
        ts = {name: _time_one(E, N, M, kw) for name, kw in ROUTES}
        win = min(ts, key=ts.get)
        out['time'].append({'N': N, 'M': M, 'best_of_5_s': ts, 'winner': win})
        print(f"TIME N={N:5d} M={M:5d}  dense={ts['dense']:.4f} "
              f"sep={ts['separable']:.4f} chirp={ts['chirpz2d']:.4f}  "
              f"-> {win}", flush=True)
        del E
        gc.collect()
    out['load_after'] = _load_snapshot()
    vlib.write_json(out, os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"v0_timing_crossover_{out['build'].split('-')[0].lower()}.json"))
    # the crossover, read off the table rather than asserted
    cross = {}
    for row in out['time']:
        cross.setdefault(row['N'], []).append((row['M'], row['winner']))
    print(json.dumps({str(k): v for k, v in cross.items()}, indent=1))


if __name__ == '__main__':
    main()
