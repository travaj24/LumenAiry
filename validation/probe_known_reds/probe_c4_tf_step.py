"""Probe for CI red: TestC4TransferFunction::test_the_tf_step_is_bit_identical
[tilt0-256] / [tilt1-256] (py3.14 shard 4).

Question: WHERE does the bit-identity between ``_exact_envelope_tf_step`` and
the test's whole-grid oracle break?  Three candidate layers, measured
separately:

  L1  the float64 ``phase`` grid (fast re-association vs oracle expression)
  L2  ``H`` = cos/sin written into real/imag  vs  ``np.exp(1j*phase)``
  L3  the FFT transport itself (``_ifft2(_fft2(E) * H)``) -- including
      whether it is even REPRODUCIBLE call-to-call on this build.

The shipped ``FFTW_MIN_SIZE = 256`` means n = 256 is the FIRST shape that
routes to pyFFTW, and 256 is exactly the shape CI fails at, so L3 is
measured with pyFFTW forced ON and forced OFF.

Writes <out>/probe_c4_tf_step_<ARM>.json.
"""
import json
import os
import sys

import numpy as np

import lumenairy
from lumenairy.propagators import carrier as C, fft_infra as F

LAM = 1.55e-6
K0 = 2.0 * np.pi / LAM


def _ulp_max(a, b):
    """Max |a-b| in ULPs of b, over the raw float64 views."""
    av = np.ascontiguousarray(a).view(np.float64).ravel()
    bv = np.ascontiguousarray(b).view(np.float64).ravel()
    d = np.abs(av - bv)
    sp = np.abs(np.spacing(bv))
    sp[sp == 0] = np.finfo(np.float64).tiny
    return float(np.max(d / sp)), float(np.max(d))


def _phase_fast(n, tilt, z_eff, dx, dy):
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(n, d=(dy if dy else dx))
    k = K0
    L, M = float(tilt[0]), float(tilt[1])
    s2 = L * L + M * M
    Nz = float(np.sqrt(1.0 - s2))
    root0 = float(np.sqrt(max(k * k * (1.0 - s2), 0.0)))
    if L == 0.0 and M == 0.0:
        phase = np.empty((n, n), dtype=np.float64)
        np.add((kx * kx)[None, :], (ky * ky)[:, None], out=phase)
        np.subtract(k * k, phase, out=phase)
        np.maximum(phase, 0.0, out=phase)
        np.sqrt(phase, out=phase)
        phase -= root0
        phase *= z_eff
        phase += k * z_eff
    else:
        KX, KY = kx[None, :], ky[:, None]
        ax, ay = k * L + KX, k * M + KY
        rad = k * k - (ax * ax + ay * ay)
        np.maximum(rad, 0.0, out=rad)
        root = np.sqrt(rad)
        lin = (L * KX + M * KY) / Nz
        phase = (k * z_eff) + z_eff * (root - root0 + lin)
    return phase


def _phase_orc(n, tilt, z_eff, dx, dy):
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(n, d=(dy if dy else dx))
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


def _tf_oracle(E, z_eff, dx, dy, tilt):
    E = np.ascontiguousarray(E, dtype=np.complex128)
    phase = _phase_orc(E.shape[-1], tilt, z_eff, dx, dy)
    return F._ifft2(F._fft2(E) * np.exp(1j * phase)).copy()


def _one(n, tilt, use_pyfftw):
    old = F.USE_PYFFTW
    F.USE_PYFFTW = use_pyfftw
    try:
        rng = np.random.default_rng(11)
        e = (rng.standard_normal((n, n))
             + 1j * rng.standard_normal((n, n)))
        z_eff, dx, dy = 5e-3, 2e-6, 2e-6
        pf = _phase_fast(n, tilt, z_eff, dx, dy)
        po = _phase_orc(n, tilt, z_eff, dx, dy)
        phase_eq = bool(np.array_equal(pf, po))
        p_ulp, p_abs = _ulp_max(pf, po)

        H = np.empty((n, n), dtype=np.complex128)
        np.cos(po, out=H.real)
        np.sin(po, out=H.imag)
        Hx = np.exp(1j * po)
        H_eq = bool(np.array_equal(H.view(np.float64), Hx.view(np.float64)))
        h_ulp, h_abs = _ulp_max(H, Hx)

        # L3: is the transport reproducible at all on this build?
        E = np.ascontiguousarray(e, dtype=np.complex128)
        f1 = F._fft2(E).copy()
        f2 = F._fft2(E).copy()
        fft_repro = bool(np.array_equal(f1, f2))
        f_ulp, f_abs = _ulp_max(f1, f2)

        o1 = _tf_oracle(e, z_eff, dx, dy, tilt)
        o2 = _tf_oracle(e, z_eff, dx, dy, tilt)
        orc_repro = bool(np.array_equal(o1, o2))
        or_ulp, or_abs = _ulp_max(o1, o2)

        got = C._exact_envelope_tf_step(e, z_eff, LAM, dx, dy, tilt=tilt)
        step_eq = bool(np.array_equal(got.view(np.float64),
                                      o1.view(np.float64)))
        s_ulp, s_abs = _ulp_max(got, o1)
        rel = float(s_abs / max(float(np.abs(o1).max()), 1e-300))

        backend = 'pyfftw' if (
            F.USE_PYFFTW and F.PYFFTW_AVAILABLE and n >= F.FFTW_MIN_SIZE
            and F._pyfftw_bad_key((n, n), np.dtype(np.complex128), 'fwd')
            not in F._PYFFTW_BAD_SHAPES) else (
            'scipy' if (F.USE_SCIPY_FFT and F.SCIPY_FFT_AVAILABLE)
            else 'numpy')
        return dict(n=n, tilt=list(tilt), use_pyfftw=bool(use_pyfftw),
                    backend=backend,
                    phase_bit_equal=phase_eq, phase_ulp=p_ulp,
                    phase_absmax=p_abs,
                    H_bit_equal=H_eq, H_ulp=h_ulp, H_absmax=h_abs,
                    fft_reproducible=fft_repro, fft_ulp=f_ulp,
                    fft_absmax=f_abs,
                    oracle_reproducible=orc_repro, oracle_ulp=or_ulp,
                    oracle_absmax=or_abs,
                    step_bit_equal=step_eq, step_ulp=s_ulp,
                    step_absmax=s_abs, step_rel=rel)
    finally:
        F.USE_PYFFTW = old


def main():
    arm = sys.argv[1] if len(sys.argv) > 1 else 'LOCAL'
    outdir = sys.argv[2] if len(sys.argv) > 2 else (
        os.path.dirname(os.path.abspath(__file__)))
    assert 'lum_reds' in lumenairy.__file__, lumenairy.__file__
    try:
        import threadpoolctl
        tpi = threadpoolctl.threadpool_info()
    except Exception as e:
        tpi = [{'error': repr(e)}]
    cpu = {}
    for mod in (getattr(np, '_core', None), getattr(np, 'core', None)):
        try:
            feats = mod._multiarray_umath.__cpu_features__
            cpu = {k: v for k, v in feats.items() if v}
            break
        except Exception:
            continue
    env = dict(
        arm=arm, python=sys.version.split()[0], numpy=np.__version__,
        lumenairy_file=lumenairy.__file__,
        platform=sys.platform,
        PYFFTW_AVAILABLE=bool(F.PYFFTW_AVAILABLE),
        USE_PYFFTW=bool(F.USE_PYFFTW),
        FFTW_THREADS=int(F.FFTW_THREADS),
        FFTW_MIN_SIZE=int(F.FFTW_MIN_SIZE),
        USE_SCIPY_FFT=bool(F.USE_SCIPY_FFT),
        SCIPY_FFT_AVAILABLE=bool(F.SCIPY_FFT_AVAILABLE),
        SCIPY_FFT_WORKERS=int(F.SCIPY_FFT_WORKERS),
        OPENBLAS_CORETYPE=os.environ.get('OPENBLAS_CORETYPE', ''),
        OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS', ''),
        threadpool=tpi, cpu_features=cpu)
    try:
        import scipy
        env['scipy'] = scipy.__version__
    except Exception:
        env['scipy'] = None
    rows = []
    for use in (True, False):
        for n in (63, 64, 65, 128, 256, 512):
            for tilt in ((0.0, 0.0), (0.03, -0.02)):
                rows.append(_one(n, tilt, use))
    out = dict(env=env, rows=rows)
    path = os.path.join(outdir, 'probe_c4_tf_step_%s.json' % arm)
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: v for k, v in env.items()
                      if k != 'cpu_features'}, indent=1)[:1800])
    for r in rows:
        print("use_pyfftw=%-5s n=%4d tilt=%s backend=%-6s phase_eq=%-5s "
              "H_eq=%-5s H_ulp=%.2f fft_repro=%-5s step_eq=%-5s "
              "step_ulp=%.3g step_rel=%.3e"
              % (r['use_pyfftw'], r['n'], r['tilt'], r['backend'],
                 r['phase_bit_equal'], r['H_bit_equal'], r['H_ulp'],
                 r['fft_reproducible'], r['step_bit_equal'],
                 r['step_ulp'], r['step_rel']))
    print('WROTE', path)


if __name__ == '__main__':
    main()
