"""The decisive A/B for the C4 TF-step CI red.

``_exact_envelope_tf_step`` and the whole-grid oracle both compute
``_ifft2(_fft2(E) * H)`` from the SAME ``E`` and a bit-identical ``H``.  On
Linux they nevertheless disagree at ~1 ulp once the shape reaches
``FFTW_MIN_SIZE``.  This probe runs the same comparison three ways:

  shipped      pyFFTW on, two-buffer ping-pong on  (the shipped dispatch)
  single_buf   pyFFTW on, ``set_fft_double_buffer(False)`` -- one buffer, one
               plan, and ``_fft2``/``_ifft2`` hand back a PRIVATE COPY
  no_pyfftw    ``USE_PYFFTW = False`` -- scipy/pocketfft

and also measures both routes against the pocketfft answer, so it is visible
that NEITHER route is "right": they are two pyFFTW roundings of the same
transform.

Writes <out>/probe_c4_double_buffer_<ARM>.json.
"""
import json
import os
import sys

import numpy as np

import lumenairy
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


def _oracle(E, n, tilt, z_eff, dx, dy):
    return F._ifft2(F._fft2(E)
                    * np.exp(1j * _phase(n, tilt, z_eff, dx, dy))).copy()


def _pair(n, tilt):
    z_eff, dx, dy = 5e-3, 2e-6, 2e-6
    rng = np.random.default_rng(11)
    e = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    E = np.ascontiguousarray(e, dtype=np.complex128)
    got = C._exact_envelope_tf_step(e, z_eff, LAM, dx, dy, tilt=tilt)
    orc = _oracle(E, n, tilt, z_eff, dx, dy)
    return got, orc


def _one(n, tilt, mode):
    prev_use, prev_db = F.USE_PYFFTW, F.get_fft_double_buffer()
    try:
        if mode == 'no_pyfftw':
            F.USE_PYFFTW = False
        elif mode == 'single_buf':
            F.set_fft_double_buffer(False)
        got, orc = _pair(n, tilt)
    finally:
        F.USE_PYFFTW = prev_use
        F.set_fft_double_buffer(prev_db)
    d = float(np.abs(got - orc).max())
    scale = float(np.abs(orc).max())
    return dict(n=n, tilt=list(tilt), mode=mode,
                bit_equal=bool(np.array_equal(got.view(np.float64),
                                              orc.view(np.float64))),
                absmax=d, rel=d / scale,
                ndiff=int(np.count_nonzero(
                    got.view(np.float64) != orc.view(np.float64))),
                nvals=int(got.size * 2))


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
    rows = []
    for mode in ('shipped', 'single_buf', 'no_pyfftw'):
        for n in (128, 256, 512):
            for tilt in ((0.0, 0.0), (0.03, -0.02)):
                rows.append(_one(n, tilt, mode))
    # neither pyFFTW route is the pocketfft answer
    ref = []
    for n in (256, 512):
        for tilt in ((0.0, 0.0), (0.03, -0.02)):
            prev = F.USE_PYFFTW
            try:
                F.USE_PYFFTW = False
                rs, _ = _pair(n, tilt)
            finally:
                F.USE_PYFFTW = prev
            gs, os_ = _pair(n, tilt)
            ref.append(dict(
                n=n, tilt=list(tilt),
                step_vs_pocketfft=float(np.abs(gs - rs).max()),
                oracle_vs_pocketfft=float(np.abs(os_ - rs).max()),
                step_vs_oracle=float(np.abs(gs - os_).max())))
    out = dict(env=env, rows=rows, vs_pocketfft=ref)
    path = os.path.join(outdir, 'probe_c4_double_buffer_%s.json' % arm)
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(env))
    for r in rows:
        print('%-11s n=%4d tilt=%-14s bit_equal=%-5s ndiff=%-7d/%-7d '
              'absmax=%.3e rel=%.3e'
              % (r['mode'], r['n'], str(r['tilt']), r['bit_equal'],
                 r['ndiff'], r['nvals'], r['absmax'], r['rel']))
    for r in ref:
        print('vs pocketfft  n=%4d tilt=%-14s step %.3e  oracle %.3e  '
              'step-vs-oracle %.3e'
              % (r['n'], str(r['tilt']), r['step_vs_pocketfft'],
                 r['oracle_vs_pocketfft'], r['step_vs_oracle']))
    print('WROTE', path)


if __name__ == '__main__':
    main()
