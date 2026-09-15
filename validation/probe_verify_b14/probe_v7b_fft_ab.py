"""VERIFY-B14 arm 7b (part 2) -- WHERE the 1-ulp A/B of the WP report's
``probe_c4_double_buffer`` actually comes from.

The WP report attributes it to the pyFFTW ping-pong on the grounds that it
vanishes when the ping-pong is off, and records that the mechanism below the
A/B was not established.  This decides it by capturing, for BOTH routes, the
exact array each route hands to ``_ifft2`` and the exact array each hands to
``_fft2``, and comparing those bitwise:

  * if the two ``_ifft2`` operands are bit-identical and the two OUTPUTS are
    not, the transform is not a function of its input -- a determinism defect
    in the FFT layer;
  * if the two ``_ifft2`` operands already differ, the FFT is innocent and the
    difference was made upstream (the transfer function, or the forward
    transform's own output), and "turning the ping-pong off makes it zero" has
    a different explanation -- the single-buffer path returns a COPY, which
    also removes an aliasing hazard.

Run on both builds.

WHAT IT FOUND: the second branch -- the two ``_ifft2`` operands already
differ while the two ``_fft2`` operands and results are bit-identical.  The
cause is in ``probe_v7d_elision.py``.
"""
from __future__ import annotations

import hashlib
import json
import sys

import numpy as np

import lumenairy as la
from lumenairy.propagators import carrier as C, fft_infra as F

LAM = 1.55e-6
K0 = 2.0 * np.pi / LAM


def _md5(a):
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()


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
    """Record every ``_fft2`` / ``_ifft2`` operand and result, by value."""

    def __init__(self):
        self.calls = []
        self._f, self._i = F._fft2, F._ifft2

    def __enter__(self):
        def fwd(x):
            xin = np.array(x, copy=True)
            out = self._f(x)
            self.calls.append(('fft2', _md5(xin), _md5(np.asarray(out)),
                               np.array(out, copy=True)))
            return out

        def inv(x):
            xin = np.array(x, copy=True)
            out = self._i(x)
            self.calls.append(('ifft2', _md5(xin), _md5(np.asarray(out)),
                               np.array(out, copy=True)))
            return out

        F._fft2, F._ifft2 = fwd, inv
        C._fft2 = fwd if hasattr(C, '_fft2') else None
        C._ifft2 = inv if hasattr(C, '_ifft2') else None
        return self

    def __exit__(self, *a):
        F._fft2, F._ifft2 = self._f, self._i
        if hasattr(C, '_fft2'):
            C._fft2 = self._f
        if hasattr(C, '_ifft2'):
            C._ifft2 = self._i
        return False


def _route_step(e, z_eff, dx, dy, tilt):
    return C._exact_envelope_tf_step(e, z_eff, LAM, dx, dy, tilt=tilt)


def _route_oracle(E, n, z_eff, dx, dy, tilt):
    return F._ifft2(F._fft2(E)
                    * np.exp(1j * _phase(n, tilt, z_eff, dx, dy))).copy()


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
            got = _route_step(e, z_eff, dx, dy, tilt)
        with _Tap() as t2:
            orc = _route_oracle(E, n, z_eff, dx, dy, tilt)
    finally:
        F.USE_PYFFTW = prev_use
        F.set_fft_double_buffer(prev_db)
    got = np.asarray(got)
    orc = np.asarray(orc)
    d = float(np.abs(got - orc).max())
    scale = float(np.abs(orc).max()) or 1.0

    def _last(calls, kind):
        for c in reversed(calls):
            if c[0] == kind:
                return c
        return None

    s_f, s_i = _last(t1.calls, 'fft2'), _last(t1.calls, 'ifft2')
    o_f, o_i = _last(t2.calls, 'fft2'), _last(t2.calls, 'ifft2')
    row = {
        'n': n, 'tilt': list(tilt), 'mode': mode,
        'bit_equal': bool(np.array_equal(got.view(np.float64),
                                         orc.view(np.float64))),
        'absmax': d, 'rel': d / scale,
        'n_fft_calls_step': sum(1 for c in t1.calls if c[0] == 'fft2'),
        'n_ifft_calls_step': sum(1 for c in t1.calls if c[0] == 'ifft2'),
        'n_fft_calls_oracle': sum(1 for c in t2.calls if c[0] == 'fft2'),
        'n_ifft_calls_oracle': sum(1 for c in t2.calls if c[0] == 'ifft2'),
    }
    if s_f and o_f:
        row['fft_operand_identical'] = (s_f[1] == o_f[1])
        row['fft_result_identical'] = (s_f[2] == o_f[2])
    if s_i and o_i:
        row['ifft_operand_identical'] = (s_i[1] == o_i[1])
        row['ifft_result_identical'] = (s_i[2] == o_i[2])
    return row, t1, t2


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else 'v7b.json'
    res = {'lumenairy_file': la.__file__, 'python': sys.version.split()[0],
           'numpy': np.__version__, 'platform': sys.platform,
           'pyfftw_available': bool(F.PYFFTW_AVAILABLE),
           'fftw_min_size': int(F.FFTW_MIN_SIZE),
           'fftw_threads': int(F.FFTW_THREADS), 'rows': []}
    try:
        import pyfftw
        res['pyfftw_version'] = pyfftw.__version__
    except Exception:
        res['pyfftw_version'] = None
    for mode in ('shipped', 'single_buf', 'no_pyfftw'):
        for n in (128, 256, 512):
            for tilt in ((0.0, 0.0), (0.03, -0.02)):
                row, _t1, _t2 = _cell(n, tilt, mode)
                res['rows'].append(row)
                print('%-11s n=%-4d tilt=%-14s equal=%-5s rel=%.3e  '
                      'fft_in_same=%-5s fft_out_same=%-5s  '
                      'ifft_in_same=%-5s ifft_out_same=%-5s  calls=%d/%d|%d/%d'
                      % (mode, n, str(tilt), row['bit_equal'], row['rel'],
                         row.get('fft_operand_identical'),
                         row.get('fft_result_identical'),
                         row.get('ifft_operand_identical'),
                         row.get('ifft_result_identical'),
                         row['n_fft_calls_step'], row['n_ifft_calls_step'],
                         row['n_fft_calls_oracle'],
                         row['n_ifft_calls_oracle']), flush=True)
    print('lumenairy:', la.__file__)
    with open(out_path, 'w', encoding='cp1252') as fh:
        json.dump(res, fh, indent=1)


if __name__ == '__main__':
    main()
