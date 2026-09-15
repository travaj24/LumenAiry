"""VERIFY-B14 defect D2 -- ``test_z3_estimate_lens_memory_real_bounds_
apply_real_lens``'s new RETAIN guard is a claim about PROCESS HISTORY.

The guard added by WP-B14 asserts that the measured ``apply_real_lens`` call
retains 5.5 .. 7.0 full complex grids, "the 6.0 of N-sized FFT/ASM cache it
builds".  That is true only while those caches are COLD for this N.  Any
earlier test in the same pytest process that has already built them leaves
the measured call nothing to build, and the guard fires.

This measures the three states -- cold, warm, and drained -- and reports both
the RETAIN reading and the claim the test actually exists for
(``estimate_lens_memory`` must bound the measured peak), so the right repair
is a measurement rather than a guess.

Each state is measured in its own process (``--state cold|warm|drained``).
"""
from __future__ import annotations

import argparse
import gc
import json
import tracemalloc
import warnings

import numpy as np

import lumenairy as la

_N = 512
_WARM_N = 256


def _measure(state):
    wl, dx = 633e-9, 30e-3 / _N
    rx = la.make_singlet(R1=50e-3, R2=-50e-3, d=5e-3, glass='N-BK7',
                         aperture=25e-3)
    E, _, _ = la.create_gaussian_beam(_N, dx, wl, w0=5e-3,
                                      dtype=np.complex128)
    E = np.ascontiguousarray(E)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # the shipped warm-up: 256 px, which warms the deferred imports but
        # not the N = 512 FFT / ASM caches
        la.apply_real_lens(E[:_WARM_N, :_WARM_N].copy(), prescription=rx,
                           wavelength=wl, dx=dx)
        if state in ('warm', 'drained'):
            # what ANY earlier test in the process that touches this N does
            la.apply_real_lens(E.copy(), prescription=rx, wavelength=wl,
                               dx=dx)
        if state == 'drained':
            la.clear_asm_caches()
            try:
                from lumenairy.propagators import fft_infra as F
                F.clear_fft_plan_cache()
            except Exception:
                pass
    gc.collect()
    tracemalloc.start()
    tracemalloc.reset_peak()
    out = la.apply_real_lens(E, prescription=rx, wavelength=wl, dx=dx)
    retained, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del out
    gc.collect()
    held = retained / float(_N * _N * 16)
    est = la.estimate_lens_memory(_N, np.complex128, lens_model='real')
    return {'state': state, 'held_grids': held, 'peak_mb': peak / 1e6,
            'est_mb': est / 1e6, 'est_over_peak': est / peak,
            'retain_guard_5p5_7p0': bool(5.5 < held < 7.0),
            'claim_1p0_1p6': bool(1.0 <= est / peak <= 1.6)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--state', default='cold',
                    choices=('cold', 'warm', 'drained'))
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    r = _measure(a.state)
    r['lumenairy_file'] = la.__file__
    print('%-8s held=%5.2f grids  peak=%7.1f MB  est=%7.1f MB  '
          'est/peak=%.3f  RETAIN[5.5,7.0]=%s  CLAIM[1.0,1.6]=%s'
          % (r['state'], r['held_grids'], r['peak_mb'], r['est_mb'],
             r['est_over_peak'], r['retain_guard_5p5_7p0'],
             r['claim_1p0_1p6']))
    print('lumenairy:', la.__file__)
    if a.out:
        with open(a.out, 'w', encoding='cp1252') as fh:
            json.dump(r, fh, indent=1)


if __name__ == '__main__':
    main()
