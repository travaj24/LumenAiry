"""VERIFY-WAVE5-E / E1(d): the spelling matrix, lumenairy-free.

Five spellings of ONE complex128 product, each compared against the fully
NAMED form.  The load-bearing row is ``np.multiply(a, h, out=h)`` -- the
explicit in-place multiply the right-operand elision is supposed to be a
shorthand for.  If the explicit one MATCHES the named form while the elided
spelling does not, the elided path is doing something the explicit path does
not, and the divergence is not a documented consequence of in-place complex
arithmetic.

Swept over n so the onset is measured rather than assumed.
"""
import json
import platform
import sys

import numpy as np


def cell(n, seed=5):
    rng = np.random.default_rng(seed)
    A = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
    P = rng.standard_normal((n, n)) * 1e6
    a = A * 1.0
    h = np.exp(1j * P)
    named = a * h

    def rel(x):
        sc = float(np.max(np.abs(named))) or 1.0
        return float(np.max(np.abs(x - named))) / sc

    def ndiff(x):
        return int(np.count_nonzero(
            np.ascontiguousarray(x).view(np.float64)
            != np.ascontiguousarray(named).view(np.float64)))

    out = {'n': n, 'n_doubles': named.size * 2}

    # LEFT operand elidable: (A*1.0) is a fresh owned temporary
    left = (A * 1.0) * h
    out['left_elidable'] = dict(equal=bool(np.array_equal(left, named)),
                                rel=rel(left), ndiff=ndiff(left))
    # RIGHT operand elidable
    right = a * np.exp(1j * P)
    out['right_elidable'] = dict(equal=bool(np.array_equal(right, named)),
                                 rel=rel(right), ndiff=ndiff(right))
    # BOTH elidable
    both = (A * 1.0) * np.exp(1j * P)
    out['both_elidable'] = dict(equal=bool(np.array_equal(both, named)),
                                rel=rel(both), ndiff=ndiff(both))
    # EXPLICIT in-place into the right operand -- what the elision claims to be
    h2 = np.exp(1j * P)
    np.multiply(a, h2, out=h2)
    out['multiply_out_is_right'] = dict(equal=bool(np.array_equal(h2, named)),
                                        rel=rel(h2), ndiff=ndiff(h2))
    # EXPLICIT in-place into the LEFT operand
    a2 = A * 1.0
    np.multiply(a2, h, out=a2)
    out['multiply_out_is_left'] = dict(equal=bool(np.array_equal(a2, named)),
                                       rel=rel(a2), ndiff=ndiff(a2))
    # EXPLICIT into a fresh output
    o = np.empty_like(named)
    np.multiply(a, h, out=o)
    out['multiply_out_is_fresh'] = dict(equal=bool(np.array_equal(o, named)),
                                        rel=rel(o), ndiff=ndiff(o))
    return out


def main():
    cfg = {}
    try:
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            np.show_config()
        txt = buf.getvalue()
        cfg['simd'] = [ln.strip() for ln in txt.splitlines()
                       if 'baseline' in ln.lower() or 'found' in ln.lower()]
    except Exception as exc:                                 # noqa: BLE001
        cfg['error'] = str(exc)
    res = dict(numpy=np.__version__, python=sys.version.split()[0],
               platform=sys.platform, machine=platform.machine(),
               compiler=platform.python_compiler(), show_config=cfg,
               cells=[cell(n) for n in (64, 128, 256, 512, 1024)])
    res['right_elidable_differs_at'] = [
        c['n'] for c in res['cells'] if not c['right_elidable']['equal']]
    res['explicit_matches_named_everywhere'] = all(
        c['multiply_out_is_right']['equal'] for c in res['cells'])
    print(json.dumps(res, indent=1))


main()
