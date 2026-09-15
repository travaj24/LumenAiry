"""VERIFY-B14 arm 7b -- ``_ifft2(_fft2(E) * H)`` is not a function of its
inputs while the pyFFTW ping-pong is on.  Characterisation, not a fix.

WHAT IS MEASURED, per grid size and per mode:

  * the SEQUENCE of results from N identical evaluations, reduced to the set
    of distinct byte images and to the index pattern that produced them.  A
    period-2 alternation is the signature of the two ping-pong SLOTS carrying
    two independently planned transforms; anything else is not.
  * the same for the two halves separately -- ``_fft2(E)`` alone and
    ``_ifft2(F)`` alone on a FIXED operand -- which says WHICH of the two
    calls is non-deterministic.
  * the per-slot plan metadata the cache holds (buffer address mod 64, the
    plan objects' identities, the planner flag, the number of slots), so the
    alternation can be attributed to a slot rather than inferred.
  * the same quantities with ``set_fft_double_buffer(False)`` and with
    ``USE_PYFFTW = False``.

No value is asserted here; the probe prints and stores.
"""
from __future__ import annotations

import hashlib
import json
import sys

import numpy as np

import lumenairy as la
from lumenairy.propagators import fft_infra as F


def _md5(a):
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()


def _pattern(hashes):
    """``[0, 1, 0, 1]``-style index pattern plus the number of distinct
    images."""
    order, idx = [], {}
    for h in hashes:
        if h not in idx:
            idx[h] = len(idx)
        order.append(idx[h])
    return order, len(idx)


def _slot_meta(shape, dtype):
    out = []
    try:
        for key, entry in list(F._PYFFTW_PLAN_CACHE.items()):
            bufs = entry.get('bufs', [])
            if not bufs or bufs[0].shape != tuple(shape):
                continue
            out.append({
                'key': repr(key)[:110],
                'n_slots': len(bufs),
                'flag': entry.get('flag'),
                'idx': entry.get('idx'),
                'calls': entry.get('calls'),
                'buf_addr_mod64': [int(b.__array_interface__['data'][0] % 64)
                                   for b in bufs],
                'buf_addr_mod4096': [
                    int(b.__array_interface__['data'][0] % 4096) for b in bufs],
                'plan_ids': [id(p) for p in entry.get('plans', [])]})
    except Exception as exc:                                # pragma: no cover
        out.append({'err': type(exc).__name__ + ': ' + str(exc)[:100]})
    return out


def _case(n, reps=8):
    rng = np.random.default_rng(12345)
    E = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
         ).astype(np.complex128)
    H = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
         ).astype(np.complex128)
    row = {'n': n}

    # round trip
    hs = []
    for _ in range(reps):
        out = F._ifft2(F._fft2(E) * H)
        hs.append(_md5(np.asarray(out)))
    row['roundtrip_pattern'], row['roundtrip_distinct'] = _pattern(hs)

    # forward alone, fixed operand
    hs = [_md5(np.asarray(F._fft2(E))) for _ in range(reps)]
    row['fft_pattern'], row['fft_distinct'] = _pattern(hs)

    # inverse alone, fixed operand (a private copy so the operand cannot be
    # a live workspace buffer)
    Fx = np.array(F._fft2(E) * H, copy=True)
    hs = [_md5(np.asarray(F._ifft2(Fx))) for _ in range(reps)]
    row['ifft_pattern'], row['ifft_distinct'] = _pattern(hs)

    # magnitude of the spread on the round trip
    vals = [np.asarray(F._ifft2(F._fft2(E) * H)).copy() for _ in range(4)]
    scale = float(np.max(np.abs(vals[0]))) or 1.0
    row['max_rel_spread'] = max(
        float(np.max(np.abs(vals[i] - vals[j]))) / scale
        for i in range(4) for j in range(i + 1, 4))
    row['frac_doubles_differing'] = float(
        np.mean(vals[0].view(np.float64) != vals[1].view(np.float64)))
    row['slots'] = _slot_meta((n, n), np.complex128)
    return row


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else 'v7_fft.json'
    sizes = [int(s) for s in
             (sys.argv[2] if len(sys.argv) > 2 else '128,256,512').split(',')]
    res = {'lumenairy_file': la.__file__, 'version': la.__version__,
           'python': sys.version.split()[0], 'numpy': np.__version__,
           'platform': sys.platform,
           'pyfftw_available': bool(F.PYFFTW_AVAILABLE),
           'use_pyfftw': bool(F.USE_PYFFTW),
           'fftw_min_size': int(F.FFTW_MIN_SIZE),
           'fftw_threads': getattr(F, 'FFTW_THREADS', None),
           'modes': {}}
    try:
        import pyfftw
        res['pyfftw_version'] = pyfftw.__version__
    except Exception:
        res['pyfftw_version'] = None

    for mode in ('shipped', 'double_buffer_off', 'no_pyfftw'):
        F.clear_fft_plan_cache() if hasattr(F, 'clear_fft_plan_cache') else None
        if mode == 'shipped':
            F.set_fft_double_buffer(True)
            F.USE_PYFFTW = bool(F.PYFFTW_AVAILABLE)
        elif mode == 'double_buffer_off':
            F.set_fft_double_buffer(False)
            F.USE_PYFFTW = bool(F.PYFFTW_AVAILABLE)
        else:
            F.set_fft_double_buffer(True)
            F.USE_PYFFTW = False
        rows = []
        for n in sizes:
            r = _case(n)
            rows.append(r)
            print('%-18s n=%-5d roundtrip %s (%d distinct)  fft %s  ifft %s  '
                  'rel=%.3e  diff_frac=%.3f'
                  % (mode, n, r['roundtrip_pattern'], r['roundtrip_distinct'],
                     r['fft_pattern'], r['ifft_pattern'],
                     r['max_rel_spread'], r['frac_doubles_differing']),
                  flush=True)
            for s in r['slots']:
                print('      slots', {k: v for k, v in s.items()
                                      if k != 'key'}, flush=True)
        res['modes'][mode] = rows
    F.set_fft_double_buffer(True)
    F.USE_PYFFTW = bool(F.PYFFTW_AVAILABLE)
    print('lumenairy:', la.__file__)
    with open(out_path, 'w', encoding='cp1252') as fh:
        json.dump(res, fh, indent=1)


if __name__ == '__main__':
    main()
