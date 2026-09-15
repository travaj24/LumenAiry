"""Probe: does the pyFFTW ping-pong give BIT-IDENTICAL results from its two
slots?

``_build_plan_entry`` allocates ``n_bufs`` independent ``pyfftw.empty_aligned``
workspaces and builds ONE ``pyfftw.FFTW`` PLAN PER BUFFER.  FFTW selects its
codelet from the problem spec *and the alignment of the arrays it is planned
against*, so two plans at the same (shape, dtype, threads) are only guaranteed
to agree to the bit if the two buffers land in the same alignment class.
Nothing in the allocator guarantees that.

``_fft2`` / ``_ifft2`` alternate slots, so two CONSECUTIVE calls with the same
input take DIFFERENT plans -- which is exactly the shape of
``test_the_tf_step_is_bit_identical``: the step under test takes slot 0 and the
whole-grid oracle takes slot 1.

Measured here, per shape and per thread count:

  * the byte alignment of every slot buffer (ptr % 16/32/64/128),
  * slot0 vs slot1 bit equality on the SAME input, forward and inverse,
  * the same comparison against a deliberately UNDER-aligned buffer, which
    shows whether FFTW's answer moves with alignment on this build at all.

Writes <out>/probe_c4_fft_slot_parity_<ARM>.json.
"""
import json
import os
import sys

import numpy as np

import lumenairy
from lumenairy.propagators import fft_infra as F


def _align(a):
    p = a.ctypes.data
    return {m: int(p % m) for m in (16, 32, 64, 128)}


def _ulp_max(a, b):
    av = np.ascontiguousarray(a).view(np.float64).ravel()
    bv = np.ascontiguousarray(b).view(np.float64).ravel()
    d = np.abs(av - bv)
    sp = np.abs(np.spacing(bv))
    sp[sp == 0] = np.finfo(np.float64).tiny
    return float(np.max(d / sp)), float(np.max(d))


def _run_slot(entry, slot, x):
    buf = entry['bufs'][slot]
    np.copyto(buf, x, casting='no')
    entry['plans'][slot]()
    return buf.copy()


def _one(n, threads):
    import pyfftw
    F._ensure_pyfftw_loaded()
    shape = (n, n)
    dt = np.dtype(np.complex128)
    rng = np.random.default_rng(11)
    x = (rng.standard_normal(shape)
         + 1j * rng.standard_normal(shape)).astype(dt)
    row = dict(n=n, threads=threads)
    for direction in ('fwd', 'inv'):
        entry = F._build_plan_entry(direction, shape, dt, threads,
                                    F._PYFFTW_PLAN_FLAGS[0])
        row[direction + '_n_bufs'] = len(entry['bufs'])
        row[direction + '_align'] = [_align(b) for b in entry['bufs']]
        row[direction + '_flag'] = entry['flag']
        outs = [_run_slot(entry, s, x) for s in range(len(entry['bufs']))]
        if len(outs) > 1:
            eq = bool(np.array_equal(outs[0].view(np.float64),
                                     outs[1].view(np.float64)))
            u, a = _ulp_max(outs[0], outs[1])
        else:
            eq, u, a = True, 0.0, 0.0
        row[direction + '_slot_bit_equal'] = eq
        row[direction + '_slot_ulp'] = u
        row[direction + '_slot_absmax'] = a

        # Does THIS build's FFTW answer move with alignment at all?  Plan the
        # same transform on a buffer deliberately offset by one complex128
        # (16 bytes) inside an over-aligned allocation.
        flat = pyfftw.empty_aligned(n * n + 4, dtype=dt, n=128)
        off = flat[1:n * n + 1].reshape(shape)
        try:
            plan_off = pyfftw.FFTW(
                off, off, axes=(0, 1),
                direction=('FFTW_FORWARD' if direction == 'fwd'
                           else 'FFTW_BACKWARD'),
                flags=(F._PYFFTW_PLAN_FLAGS[0],),
                threads=max(1, int(threads)))
            np.copyto(off, x, casting='no')
            plan_off()
            o = off.copy()
            row[direction + '_misaligned_align'] = _align(off)
            row[direction + '_misaligned_bit_equal'] = bool(
                np.array_equal(o.view(np.float64),
                               outs[0].view(np.float64)))
            u2, a2 = _ulp_max(o, outs[0])
            row[direction + '_misaligned_ulp'] = u2
            row[direction + '_misaligned_absmax'] = a2
        except Exception as e:
            row[direction + '_misaligned_error'] = repr(e)
    return row


def main():
    arm = sys.argv[1] if len(sys.argv) > 1 else 'LOCAL'
    outdir = sys.argv[2] if len(sys.argv) > 2 else (
        os.path.dirname(os.path.abspath(__file__)))
    assert 'lum_reds' in lumenairy.__file__, lumenairy.__file__
    env = dict(arm=arm, python=sys.version.split()[0], numpy=np.__version__,
               platform=sys.platform,
               PYFFTW_AVAILABLE=bool(F.PYFFTW_AVAILABLE),
               FFTW_MIN_SIZE=int(F.FFTW_MIN_SIZE),
               FFTW_THREADS=int(F.FFTW_THREADS),
               planner=F._PYFFTW_PLAN_FLAGS[0],
               OPENBLAS_CORETYPE=os.environ.get('OPENBLAS_CORETYPE', ''))
    if not F.PYFFTW_AVAILABLE:
        env['note'] = 'pyFFTW not installed -- probe is vacuous on this arm'
        rows = []
    else:
        F._ensure_pyfftw_loaded()
        import pyfftw
        env['pyfftw'] = pyfftw.__version__
        env['simd_alignment'] = int(pyfftw.simd_alignment)
        rows = []
        for threads in (1, int(F.FFTW_THREADS)):
            for n in (256, 384, 512, 1024):
                rows.append(_one(n, threads))
    out = dict(env=env, rows=rows)
    path = os.path.join(outdir, 'probe_c4_fft_slot_parity_%s.json' % arm)
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(env, indent=1))
    for r in rows:
        print("n=%4d thr=%d | fwd slot_eq=%-5s ulp=%.3g align0=%s align1=%s"
              " misalign_eq=%-5s ulp=%.3g | inv slot_eq=%-5s ulp=%.3g"
              % (r['n'], r['threads'], r['fwd_slot_bit_equal'],
                 r['fwd_slot_ulp'],
                 r['fwd_align'][0][64],
                 r['fwd_align'][1][64] if len(r['fwd_align']) > 1 else '-',
                 r.get('fwd_misaligned_bit_equal'),
                 r.get('fwd_misaligned_ulp', float('nan')),
                 r['inv_slot_bit_equal'], r['inv_slot_ulp']))
    print('WROTE', path)


if __name__ == '__main__':
    main()
