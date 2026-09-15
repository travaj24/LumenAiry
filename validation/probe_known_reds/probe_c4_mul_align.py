"""Decisive micro-probe: does numpy's complex128 multiply return DIFFERENT
BITS for two arrays that hold IDENTICAL VALUES but live at different
alignments / in a pyFFTW-allocated buffer?

If yes, ``_ifft2(_fft2(E) * H)`` is not a function of the VALUES alone: the
ping-pong slot the spectrum happens to land in decides the last bit, and two
callers of the same transport can disagree byte-wise.
"""
import json
import os
import sys

import numpy as np

import lumenairy
from lumenairy.propagators import fft_infra as F


def run(n):
    rng = np.random.default_rng(7)
    S = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
    H = np.exp(1j * (rng.standard_normal((n, n)) * 1e4))
    out = {'n': n}
    ref = S * H
    # a plain numpy copy at a different address
    S2 = S.copy()
    out['numpy_copy_equal'] = bool(np.array_equal((S2 * H).view(np.float64),
                                                  ref.view(np.float64)))
    # an offset (deliberately under-aligned) view holding the same values
    flat = np.empty(n * n + 8, dtype=np.complex128)
    off = flat[1:n * n + 1].reshape(n, n)
    off[:] = S
    out['offset_view_equal'] = bool(np.array_equal((off * H).view(np.float64),
                                                   ref.view(np.float64)))
    out['offset_addr_mod64'] = int(off.ctypes.data % 64)
    out['ref_addr_mod64'] = int(S.ctypes.data % 64)
    if F.PYFFTW_AVAILABLE:
        F._ensure_pyfftw_loaded()
        import pyfftw
        A = pyfftw.empty_aligned((n, n), dtype=np.complex128)
        A[:] = S
        out['pyfftw_buf_equal'] = bool(np.array_equal((A * H).view(np.float64),
                                                      ref.view(np.float64)))
        out['pyfftw_addr_mod64'] = int(A.ctypes.data % 64)
        out['simd_alignment'] = int(pyfftw.simd_alignment)
        # the real ping-pong slots at this key
        entry = F._build_plan_entry('fwd', (n, n), np.dtype(np.complex128),
                                    1, F._PYFFTW_PLAN_FLAGS[0])
        prods = []
        for b in entry['bufs']:
            b[:] = S
            prods.append((b * H).copy())
            out.setdefault('slot_addr_mod64', []).append(
                int(b.ctypes.data % 64))
        out['slot_products_equal'] = bool(np.array_equal(
            prods[0].view(np.float64), prods[-1].view(np.float64)))
        out['slot0_matches_ref'] = bool(np.array_equal(
            prods[0].view(np.float64), ref.view(np.float64)))
    return out


def main():
    arm = sys.argv[1] if len(sys.argv) > 1 else 'LOCAL'
    outdir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(
        os.path.abspath(__file__))
    assert 'lum_reds' in lumenairy.__file__, lumenairy.__file__
    env = dict(arm=arm, python=sys.version.split()[0], numpy=np.__version__,
               platform=sys.platform)
    rows = [run(n) for n in (63, 128, 256, 512)]
    path = os.path.join(outdir, 'probe_c4_mul_align_%s.json' % arm)
    with open(path, 'w') as fh:
        json.dump(dict(env=env, rows=rows), fh, indent=1)
    print(json.dumps(env))
    for r in rows:
        print(json.dumps(r))
    print('WROTE', path)


if __name__ == '__main__':
    main()
