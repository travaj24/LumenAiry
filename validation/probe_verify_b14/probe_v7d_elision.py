"""VERIFY-B14 arm 7b (part 4) -- the MECHANISM under the WP report's
"``fft_infra`` determinism defect", isolated to four lines of NumPy.

The WP report records that with the pyFFTW ping-pong on,
``_ifft2(_fft2(E) * H)`` "is not a function of its input values alone", that
the difference vanishes with ``set_fft_double_buffer(False)`` and with
``USE_PYFFTW = False``, that it appears only at ``n >= FFTW_MIN_SIZE`` and
only on Linux, and that the mechanism below the A/B was NOT established.

It is not an FFT defect.  Measured here:

  1. ``_fft2`` and ``_ifft2`` ARE functions of their inputs -- eight identical
     evaluations give one byte image in every mode (``probe_v7_fft_determinism``).
  2. The two routes hand ``_ifft2`` DIFFERENT OPERANDS; the forward results
     they captured are bit-identical (``probe_v7b`` / ``v7c``).
  3. The operand difference is made by the MULTIPLY, and it depends on which
     operand NumPy's temporary elision claims -- which is what the ping-pong
     changes, because it returns a NON-OWNING aligned VIEW (not elidable)
     where the single-buffer path returns ``buf.copy()`` (a numpy-owned
     temporary, elidable).

This probe is (3) on its own: the same two complex128 arrays multiplied four
ways that differ only in which operand is an unreferenced temporary.  On the
Linux NumPy build the RIGHT-temporary form differs from the other three; on
the Windows build all four agree.  No lumenairy import is required.
"""
from __future__ import annotations

import hashlib
import json
import sys

import numpy as np


def _md5(a):
    return hashlib.md5(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else 'v7d_elision.json'
    res = {'numpy': np.__version__, 'python': sys.version.split()[0],
           'platform': sys.platform, 'rows': []}
    for n in (64, 128, 256, 512, 1024):
        rng = np.random.default_rng(5)
        A = (rng.standard_normal((n, n))
             + 1j * rng.standard_normal((n, n))).astype(np.complex128)
        P = rng.standard_normal((n, n)) * 1e6

        def mk_a():
            return A * 1.0

        def mk_h():
            return np.exp(1j * P)

        p_both = mk_a() * mk_h()
        a_named, h_named = mk_a(), mk_h()
        p_named = a_named * h_named
        p_left = mk_a() * h_named
        p_right = a_named * mk_h()
        sc = float(np.max(np.abs(p_named))) or 1.0
        row = {
            'n': n,
            'md5_both': _md5(p_both), 'md5_named': _md5(p_named),
            'md5_left': _md5(p_left), 'md5_right': _md5(p_right),
            'both_eq_named': bool(np.array_equal(p_both, p_named)),
            'left_eq_named': bool(np.array_equal(p_left, p_named)),
            'right_eq_named': bool(np.array_equal(p_right, p_named)),
            'right_vs_named_rel': float(np.max(np.abs(p_right - p_named))) / sc,
            'right_vs_named_ndiff': int(np.count_nonzero(
                p_right.view(np.float64) != p_named.view(np.float64))),
            'nvals': int(p_named.size * 2)}
        res['rows'].append(row)
        print('n=%-5d both==named %-5s left==named %-5s right==named %-5s  '
              'rel=%.3e  ndiff=%d/%d'
              % (n, row['both_eq_named'], row['left_eq_named'],
                 row['right_eq_named'], row['right_vs_named_rel'],
                 row['right_vs_named_ndiff'], row['nvals']), flush=True)
    res['elision_changes_bits'] = any(not r['right_eq_named']
                                      for r in res['rows'])
    print('ELISION CHANGES BITS ON THIS BUILD:', res['elision_changes_bits'])
    with open(out_path, 'w', encoding='cp1252') as fh:
        json.dump(res, fh, indent=1)


if __name__ == '__main__':
    main()
