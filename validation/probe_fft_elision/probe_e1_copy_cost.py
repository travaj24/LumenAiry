"""WAVE5-E item E1 -- the COST side of the fft_infra ping-pong decision.

VERIFY-B14 section 3 established the mechanism: with the pyFFTW ping-pong ON,
``_fft2`` hands back a NON-OWNING view of the plan workspace, so in
``_fft2(E) * H`` NumPy's ``temp_elide`` claims the RIGHT operand; with the
ping-pong OFF ``_fft2`` returns ``buf.copy()`` (a numpy-owned temporary) and the
LEFT operand is elided.  On the Linux numpy build the right-elided complex128
multiply moves the last bits.

The decision is between

  (a) the dispatchers returning a private copy at the shapes where it matters,
  (b) scoping the contract sentence only.

(a) costs one N^2 complex copy per transform.  This probe MEASURES that cost on
the propagator hot path -- ``angular_spectrum_propagate`` and
``fresnel_propagate`` at 512^2 / 1024^2 / 2048^2 -- by timing the shipped
ping-pong path against the already-shipped copy-per-FFT path
(``set_fft_double_buffer(False)``, which is exactly remedy (a)'s arithmetic:
the dispatcher returns ``buf.copy()``).  It also times the raw ``buf.copy()``
in isolation, and re-reads the elision A/B on the same process so the benefit
side is measured next to the cost side.

Usage:  python probe_e1_copy_cost.py <out.json>
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np


def _md5(a):
    if isinstance(a, tuple):
        a = a[0]
    return hashlib.md5(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()


def _best(fn, repeats):
    """Best-of-N wall clock of ``fn`` (best, not mean: contention is additive)."""
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts), float(np.median(ts))


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else 'e1_copy_cost.json'
    import lumenairy as la
    from lumenairy.propagators import fft_infra as fi

    res = {
        'lumenairy_file': la.__file__,
        'python': sys.version.split()[0],
        'numpy': np.__version__,
        'platform': sys.platform,
        'pyfftw_available': bool(fi.PYFFTW_AVAILABLE),
        'use_pyfftw': bool(fi.USE_PYFFTW),
        'fftw_min_size': int(fi.FFTW_MIN_SIZE),
        'threads_env': {k: os.environ.get(k) for k in
                        ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                         'MKL_NUM_THREADS')},
        'rows': [],
        'raw_copy': [],
        'elision': [],
    }
    print('lumenairy.__file__ =', la.__file__, flush=True)
    print('pyfftw available =', fi.PYFFTW_AVAILABLE, ' USE_PYFFTW =',
          fi.USE_PYFFTW, flush=True)

    wl, dx, z = 1.31e-6, 1.0e-6, 2.0e-3

    for n in (512, 1024, 2048):
        rng = np.random.default_rng(11)
        E = (rng.standard_normal((n, n))
             + 1j * rng.standard_normal((n, n))).astype(np.complex128)
        row = {'n': n}
        for name, fn in (
                ('asm', lambda: la.angular_spectrum_propagate(E, z, wl, dx)),
                ('fresnel', lambda: la.fresnel_propagate(E, z, wl, dx)),
        ):
            for mode, dbl in (('pingpong', True), ('copy', False)):
                fi.set_fft_double_buffer(dbl)
                la.clear_asm_caches()
                fn()          # warm the plan + the H cache at this shape
                fn()
                best, med = _best(fn, 7)
                row['%s_%s_best_s' % (name, mode)] = best
                row['%s_%s_median_s' % (name, mode)] = med
                row['%s_%s_md5' % (name, mode)] = _md5(fn())
            b_pp = row['%s_pingpong_best_s' % name]
            b_cp = row['%s_copy_best_s' % name]
            row['%s_copy_overhead_frac' % name] = (b_cp - b_pp) / b_pp
            row['%s_values_identical' % name] = (
                row['%s_pingpong_md5' % name] == row['%s_copy_md5' % name])
            print('n=%-5d %-8s pingpong %8.2f ms   copy %8.2f ms   '
                  'overhead %+7.2f %%   md5 same %s'
                  % (n, name, 1e3 * b_pp, 1e3 * b_cp,
                     1e2 * row['%s_copy_overhead_frac' % name],
                     row['%s_values_identical' % name]), flush=True)
        res['rows'].append(row)

        # The copy in isolation, against one forward transform at the same shape.
        fi.set_fft_double_buffer(True)
        la.clear_asm_caches()
        buf = np.empty((n, n), dtype=np.complex128)
        buf[:] = E
        c_best, _ = _best(lambda: buf.copy(), 11)
        fi._fft2(E)
        f_best, _ = _best(lambda: fi._fft2(E), 7)
        res['raw_copy'].append({
            'n': n, 'copy_best_s': c_best, 'fft2_best_s': f_best,
            'copy_frac_of_fft2': c_best / f_best,
            'nbytes': int(buf.nbytes)})
        print('n=%-5d raw copy %8.3f ms   _fft2 %8.3f ms   copy/fft2 %.3f'
              % (n, 1e3 * c_best, 1e3 * f_best, c_best / f_best), flush=True)

    # The benefit side, in this same process: does the running build show the
    # right-operand elision asymmetry at all?
    for n in (128, 256, 512, 1024):
        rng = np.random.default_rng(5)
        A = (rng.standard_normal((n, n))
             + 1j * rng.standard_normal((n, n))).astype(np.complex128)
        P = rng.standard_normal((n, n)) * 1e6
        a_named, h_named = A * 1.0, np.exp(1j * P)
        p_named = a_named * h_named
        p_right = a_named * np.exp(1j * P)
        p_left = (A * 1.0) * h_named
        sc = float(np.max(np.abs(p_named))) or 1.0
        res['elision'].append({
            'n': n,
            'right_eq_named': bool(np.array_equal(p_right, p_named)),
            'left_eq_named': bool(np.array_equal(p_left, p_named)),
            'right_vs_named_rel': float(np.max(np.abs(p_right - p_named))) / sc,
            'right_vs_named_ndiff': int(np.count_nonzero(
                p_right.view(np.float64) != p_named.view(np.float64))),
            'nvals': int(p_named.size * 2)})
    res['elision_changes_bits'] = any(not r['right_eq_named']
                                      for r in res['elision'])
    print('ELISION CHANGES BITS ON THIS BUILD:',
          res['elision_changes_bits'], flush=True)

    fi.set_fft_double_buffer(True)
    with open(out_path, 'w', encoding='cp1252') as fh:
        json.dump(res, fh, indent=1)
    print('wrote', out_path, flush=True)


if __name__ == '__main__':
    main()
