"""WAVE5-E item E1 -- the spelling matrix behind ``NUMPY_ISSUE_DRAFT.md``.

lumenairy-free.  For one complex128 product it evaluates the six spellings that
differ only in which operand (if any) NumPy's temporary elision may claim, and
records which agree with the fully-named reference.  The load-bearing row is
``np.multiply(a, b, out=b)`` -- the explicit form the right-operand elision is
supposed to be a shorthand for: on the affected build the EXPLICIT form matches
the named reference while the ELIDED one does not, which is what makes this an
elision defect rather than a property of in-place complex arithmetic.

Also records the build axes the draft cites (numpy version, compiler, SIMD).

Usage:  python probe_e1_spellings.py <out.json>
"""
from __future__ import annotations

import json
import sys

import numpy as np


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else 'e1_spellings.json'
    cfg = np.show_config(mode='dicts')
    res = {'numpy': np.__version__, 'python': sys.version.split()[0],
           'platform': sys.platform,
           'compiler': cfg.get('Compilers', {}).get('c'),
           'simd': cfg.get('SIMD Extensions'),
           'blas': (cfg.get('Build Dependencies', {})
                    .get('blas', {}).get('openblas configuration')),
           'sizes': [], 'spellings': {}}

    for n in (64, 128, 256, 512, 1024):
        rng = np.random.default_rng(5)
        A = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
        P = rng.standard_normal((n, n)) * 1e6
        a, h = A * 1.0, np.exp(1j * P)
        named = a * h
        right = a * np.exp(1j * P)
        sc = float(np.max(np.abs(named))) or 1.0
        res['sizes'].append({
            'n': n,
            'right_eq_named': bool(np.array_equal(right, named)),
            'rel': float(np.max(np.abs(right - named))) / sc,
            'ndiff': int(np.count_nonzero(
                right.view(np.float64) != named.view(np.float64))),
            'nlanes': int(named.size * 2)})
        print('n=%-5d right==named %-5s rel=%.4e ndiff=%d/%d'
              % (n, res['sizes'][-1]['right_eq_named'], res['sizes'][-1]['rel'],
                 res['sizes'][-1]['ndiff'], res['sizes'][-1]['nlanes']),
              flush=True)

    n = 512
    rng = np.random.default_rng(5)
    A = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
    P = rng.standard_normal((n, n)) * 1e6
    a, h = A * 1.0, np.exp(1j * P)
    named = a * h

    def _inplace_right():
        b = h.copy()
        np.multiply(a, b, out=b)
        return b

    def _explicit_out():
        o = np.empty_like(h)
        np.multiply(a, h, out=o)
        return o

    def _inplace_left():
        c = a.copy()
        np.multiply(c, h, out=c)
        return c

    spellings = {
        'both_named__reference': lambda: a * h,
        'left_temporary_elidable': lambda: (A * 1.0) * h,
        'right_temporary_elidable': lambda: a * np.exp(1j * P),
        'explicit_inplace_on_right': _inplace_right,
        'explicit_third_array': _explicit_out,
        'explicit_inplace_on_left': _inplace_left,
    }
    for name, fn in spellings.items():
        got = fn()
        res['spellings'][name] = bool(np.array_equal(got, named))
        print('%-28s equals named: %s' % (name, res['spellings'][name]),
              flush=True)

    res['elision_defect_present'] = bool(
        not res['spellings']['right_temporary_elidable']
        and res['spellings']['explicit_inplace_on_right'])
    print('ELISION DEFECT PRESENT ON THIS BUILD:',
          res['elision_defect_present'], flush=True)
    with open(out_path, 'w', encoding='cp1252') as fh:
        json.dump(res, fh, indent=1)
    print('wrote', out_path, flush=True)


if __name__ == '__main__':
    main()
