"""VERIFY-B14 arm 7b (part 3) -- the two ``_ifft2`` operands, side by side.

``probe_v7b`` established that with the ping-pong ON the two routes hand
``_ifft2`` DIFFERENT operands while handing ``_fft2`` the same input and
getting the same forward result back.  That leaves exactly two possibilities
and this separates them:

  A. the transfer functions differ (arithmetic upstream), or
  B. the forward result the step route multiplies by is no longer the array
     ``_fft2`` returned -- i.e. the LIVE ping-pong workspace was overwritten
     between the return and the multiply.

Case B is decided by re-deriving each route's transfer function from its own
``_ifft2`` operand and the forward result captured at return
(``H = operand / forward``) and comparing the two H's, and by comparing each
route's operand against ``forward_at_return * H``.

WHAT IT FOUND: neither.  The forward workspace is NOT clobbered (its hash
at return equals its hash at the ``_ifft2`` call) and the transfer functions
are equal; the difference is made by the MULTIPLY, and specifically by which
operand NumPy's temporary elision claims -- which is what the ping-pong
changes by returning a non-owning view instead of a copy.  See
``probe_v7d_elision.py``, which is that finding on its own.
"""
from __future__ import annotations

import json
import sys

import numpy as np

import lumenairy as la
from lumenairy.propagators import carrier as C, fft_infra as F

LAM = 1.55e-6
K0 = 2.0 * np.pi / LAM


def _phase(n, tilt, z_eff, dx, dy):
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(n, d=dy)
    k = K0
    L, M = float(tilt[0]), float(tilt[1])
    s2 = L * L + M * M
    nz = float(np.sqrt(1.0 - s2))
    KX, KY = kx[None, :], ky[:, None]
    ax, ay = k * L + KX, k * M + KY
    rad = k * k - (ax * ax + ay * ay)
    np.maximum(rad, 0.0, out=rad)
    root = np.sqrt(rad)
    root0 = float(np.sqrt(max(k * k * (1.0 - s2), 0.0)))
    lin = (L * KX + M * KY) / nz
    return (k * z_eff) + z_eff * (root - root0 + lin)


class _Tap:
    def __init__(self):
        self.fwd_out = []
        self.inv_in = []
        self.inv_out = []
        self._f, self._i = F._fft2, F._ifft2

    def __enter__(self):
        def fwd(x):
            out = self._f(x)
            self.fwd_out.append(np.array(out, copy=True))
            return out

        def inv(x):
            self.inv_in.append(np.array(x, copy=True))
            out = self._i(x)
            self.inv_out.append(np.array(out, copy=True))
            return out

        F._fft2, F._ifft2 = fwd, inv
        return self

    def __exit__(self, *a):
        F._fft2, F._ifft2 = self._f, self._i
        return False


def _cell(n, tilt, mode):
    z_eff, dx, dy = 5e-3, 2e-6, 2e-6
    rng = np.random.default_rng(11)
    e = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    E = np.ascontiguousarray(e, dtype=np.complex128)
    prev_use, prev_db = F.USE_PYFFTW, F.get_fft_double_buffer()
    try:
        if mode == 'no_pyfftw':
            F.USE_PYFFTW = False
        elif mode == 'single_buf':
            F.set_fft_double_buffer(False)
        with _Tap() as t1:
            C._exact_envelope_tf_step(e, z_eff, LAM, dx, dy, tilt=tilt)
        with _Tap() as t2:
            F._ifft2(F._fft2(E)
                     * np.exp(1j * _phase(n, tilt, z_eff, dx, dy))).copy()
    finally:
        F.USE_PYFFTW = prev_use
        F.set_fft_double_buffer(prev_db)

    f1, f2 = t1.fwd_out[-1], t2.fwd_out[-1]
    o1, o2 = t1.inv_in[-1], t2.inv_in[-1]
    with np.errstate(divide='ignore', invalid='ignore'):
        h1 = np.where(np.abs(f1) > 0, o1 / f1, 0.0)
        h2 = np.where(np.abs(f2) > 0, o2 / f2, 0.0)
    sc_f = float(np.max(np.abs(f1))) or 1.0
    sc_o = float(np.max(np.abs(o1))) or 1.0
    return {
        'n': n, 'tilt': list(tilt), 'mode': mode,
        'fwd_bit_equal': bool(np.array_equal(f1.view(np.float64),
                                             f2.view(np.float64))),
        'fwd_reldiff': float(np.max(np.abs(f1 - f2))) / sc_f,
        'operand_bit_equal': bool(np.array_equal(o1.view(np.float64),
                                                 o2.view(np.float64))),
        'operand_reldiff': float(np.max(np.abs(o1 - o2))) / sc_o,
        'operand_ndiff': int(np.count_nonzero(
            o1.view(np.float64) != o2.view(np.float64))),
        'operand_nvals': int(o1.size * 2),
        'H_bit_equal': bool(np.array_equal(h1.view(np.float64),
                                           h2.view(np.float64))),
        'H_reldiff': float(np.max(np.abs(h1 - h2))),
        'H_abs1_max_dev': float(np.max(np.abs(np.abs(h1) - 1.0))),
        'H_abs2_max_dev': float(np.max(np.abs(np.abs(h2) - 1.0))),
        # does each route's operand equal (its own captured forward) * (its
        # own H)?  If the step route's live buffer had been clobbered before
        # the multiply, this identity would break for it and hold for the
        # oracle.
        'o1_is_f1_times_h1': bool(np.allclose(o1, f1 * h1, rtol=0, atol=0)),
        'o2_is_f2_times_h2': bool(np.allclose(o2, f2 * h2, rtol=0, atol=0)),
        'o1_vs_f1h2': float(np.max(np.abs(o1 - f1 * h2))) / sc_o,
        'o2_vs_f2h1': float(np.max(np.abs(o2 - f2 * h1))) / sc_o,
    }


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else 'v7c.json'
    res = {'lumenairy_file': la.__file__, 'python': sys.version.split()[0],
           'numpy': np.__version__, 'platform': sys.platform,
           'pyfftw_available': bool(F.PYFFTW_AVAILABLE), 'rows': []}
    for mode in ('shipped', 'single_buf', 'no_pyfftw'):
        for n in (256, 512):
            for tilt in ((0.0, 0.0), (0.03, -0.02)):
                r = _cell(n, tilt, mode)
                res['rows'].append(r)
                print('%-11s n=%-4d tilt=%-14s fwd_eq=%-5s op_eq=%-5s '
                      'op_rel=%.3e op_ndiff=%d/%d  H_eq=%-5s H_rel=%.3e  '
                      'o1=f1*h1:%-5s o2=f2*h2:%-5s  o1-f1h2=%.3e'
                      % (mode, n, str(tilt), r['fwd_bit_equal'],
                         r['operand_bit_equal'], r['operand_reldiff'],
                         r['operand_ndiff'], r['operand_nvals'],
                         r['H_bit_equal'], r['H_reldiff'],
                         r['o1_is_f1_times_h1'], r['o2_is_f2_times_h2'],
                         r['o1_vs_f1h2']), flush=True)
    print('lumenairy:', la.__file__)
    with open(out_path, 'w', encoding='cp1252') as fh:
        json.dump(res, fh, indent=1)


if __name__ == '__main__':
    main()
