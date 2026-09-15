"""Layer-by-layer localisation of the ``_exact_envelope_tf_step`` vs
whole-grid-oracle byte difference (CI red at n = 256, both tilts).

Unlike ``probe_c4_tf_step.py`` this does NOT re-implement the step: it wraps
``fft_infra._fft2`` / ``_ifft2`` and records the ACTUAL arrays the shipped
function hands them, so the comparison is against what the library really
computed rather than against a copy of its source.

Layers, in order:

  A  the array handed to ``_fft2``            (the input envelope)
  B  the array returned by ``_fft2``          (the spectrum)
  C  the array handed to ``_ifft2``           (spectrum * H)  <- H lives here
  D  the array returned by ``_ifft2``         (the transported field)

For each layer the step's array is compared BIT-WISE with the oracle's.  The
first layer that differs is the defect's layer.  Also recorded: whether the
same ``_ifft2`` input, replayed, reproduces its own output.

Writes <out>/probe_c4_tf_layers_<ARM>.json.
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
    av = np.ascontiguousarray(a).view(np.float64).ravel()
    bv = np.ascontiguousarray(b).view(np.float64).ravel()
    d = np.abs(av - bv)
    sp = np.abs(np.spacing(bv))
    sp[sp == 0] = np.finfo(np.float64).tiny
    return float(np.max(d / sp)), float(np.max(d))


class _Tap:
    """Record (input, output) of every _fft2 / _ifft2 call, as private
    copies, and report the buffer identity/alignment each call saw."""

    def __init__(self):
        self.calls = []
        self._f, self._i = F._fft2, F._ifft2

    def __enter__(self):
        def fft2(x):
            out = self._f(x)
            self.calls.append(('fwd', np.array(x, copy=True),
                               np.array(out, copy=True),
                               int(out.ctypes.data),
                               int(np.asarray(x).ctypes.data)))
            return out

        def ifft2(x):
            out = self._i(x)
            self.calls.append(('inv', np.array(x, copy=True),
                               np.array(out, copy=True),
                               int(out.ctypes.data),
                               int(np.asarray(x).ctypes.data)))
            return out

        F._fft2, F._ifft2 = fft2, ifft2
        return self

    def __exit__(self, *a):
        F._fft2, F._ifft2 = self._f, self._i
        return False


def _tf_oracle(E, z_eff, dx, dy, tilt):
    E = np.ascontiguousarray(E, dtype=np.complex128)
    ny, nx = E.shape[-2], E.shape[-1]
    k = K0
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=(dy if dy else dx))
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
    phase = (k * z_eff) + z_eff * (root - root0 + lin)
    return F._ifft2(F._fft2(E) * np.exp(1j * phase)).copy()


def _one(n, tilt, use_pyfftw):
    old = F.USE_PYFFTW
    F.USE_PYFFTW = use_pyfftw
    try:
        rng = np.random.default_rng(11)
        e = (rng.standard_normal((n, n))
             + 1j * rng.standard_normal((n, n)))
        z_eff, dx, dy = 5e-3, 2e-6, 2e-6
        with _Tap() as tap:
            got = C._exact_envelope_tf_step(e, z_eff, LAM, dx, dy, tilt=tilt)
            n_step = len(tap.calls)
            orc = _tf_oracle(e, z_eff, dx, dy, tilt)
        step = tap.calls[:n_step]
        orac = tap.calls[n_step:]
        row = dict(n=n, tilt=list(tilt), use_pyfftw=bool(use_pyfftw),
                   step_calls=[c[0] for c in step],
                   orac_calls=[c[0] for c in orac])
        for lab, i in (('fwd', 0), ('inv', 1)):
            if len(step) > i and len(orac) > i:
                si, so = step[i][1], step[i][2]
                oi, oo = orac[i][1], orac[i][2]
                row[lab + '_in_equal'] = bool(np.array_equal(
                    si.view(np.float64), oi.view(np.float64)))
                u, a = _ulp_max(si, oi)
                row[lab + '_in_ulp'], row[lab + '_in_absmax'] = u, a
                row[lab + '_out_equal'] = bool(np.array_equal(
                    so.view(np.float64), oo.view(np.float64)))
                u, a = _ulp_max(so, oo)
                row[lab + '_out_ulp'], row[lab + '_out_absmax'] = u, a
                row[lab + '_step_outbuf'] = step[i][3]
                row[lab + '_orac_outbuf'] = orac[i][3]
                row[lab + '_same_outbuf'] = bool(step[i][3] == orac[i][3])
        # WHICH H did each side actually multiply by?  Rebuild both candidate
        # kernels here from the phase the two paths share and re-do the
        # product on a PLAIN numpy copy of the (identical) spectrum.
        if len(step) > 1 and len(orac) > 1:
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
            ph = (k * z_eff) + z_eff * (root - root0 + lin)
            H_exp = np.exp(1j * ph)
            H_cs = np.empty((n, n), dtype=np.complex128)
            np.cos(ph, out=H_cs.real)
            np.sin(ph, out=H_cs.imag)
            row['H_cs_equals_H_exp'] = bool(np.array_equal(
                H_cs.view(np.float64), H_exp.view(np.float64)))
            S = step[0][2]              # plain numpy copy of the spectrum
            for lab, HH in (('exp', H_exp), ('cs', H_cs)):
                P = S * HH
                row['step_in_is_S_times_H%s' % lab] = bool(np.array_equal(
                    P.view(np.float64), step[1][1].view(np.float64)))
                row['orac_in_is_S_times_H%s' % lab] = bool(np.array_equal(
                    P.view(np.float64), orac[1][1].view(np.float64)))
            u, a = _ulp_max(step[1][1], orac[1][1])
            row['inv_in_ulp_detail'], row['inv_in_absmax_detail'] = u, a
            row['inv_in_ndiff'] = int(np.count_nonzero(
                step[1][1].view(np.float64) != orac[1][1].view(np.float64)))
        # replay: feed the step's own _ifft2 input back in twice
        if len(step) > 1:
            x = step[1][1]
            r1 = np.array(F._ifft2(x), copy=True)
            r2 = np.array(F._ifft2(x), copy=True)
            row['ifft2_replay_equal'] = bool(np.array_equal(
                r1.view(np.float64), r2.view(np.float64)))
            row['ifft2_replay_matches_step'] = bool(np.array_equal(
                r1.view(np.float64), step[1][2].view(np.float64)))
        row['final_equal'] = bool(np.array_equal(got.view(np.float64),
                                                 orc.view(np.float64)))
        u, a = _ulp_max(got, orc)
        row['final_ulp'], row['final_absmax'] = u, a
        return row
    finally:
        F.USE_PYFFTW = old


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
               SCIPY_FFT_WORKERS=int(F.SCIPY_FFT_WORKERS),
               planner=F._PYFFTW_PLAN_FLAGS[0],
               double_buffer=bool(getattr(F, '_PYFFTW_DOUBLE_BUFFER', True)),
               OPENBLAS_CORETYPE=os.environ.get('OPENBLAS_CORETYPE', ''))
    rows = []
    for use in (True, False):
        for n in (128, 256, 512):
            for tilt in ((0.0, 0.0), (0.03, -0.02)):
                rows.append(_one(n, tilt, use))
    out = dict(env=env, rows=rows)
    path = os.path.join(outdir, 'probe_c4_tf_layers_%s.json' % arm)
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(env, indent=1))
    for r in rows:
        print("pyfftw=%-5s n=%4d tilt=%-14s | fwd_in=%-5s fwd_out=%-5s "
              "(samebuf=%-5s) | inv_in=%-5s inv_out=%-5s (samebuf=%-5s) | "
              "replay_eq=%-5s replay==step=%-5s | final=%-5s ulp=%.3g"
              % (r['use_pyfftw'], r['n'], str(r['tilt']),
                 r.get('fwd_in_equal'), r.get('fwd_out_equal'),
                 r.get('fwd_same_outbuf'),
                 r.get('inv_in_equal'), r.get('inv_out_equal'),
                 r.get('inv_same_outbuf'),
                 r.get('ifft2_replay_equal'),
                 r.get('ifft2_replay_matches_step'),
                 r['final_equal'], r['final_ulp']))
    print('WROTE', path)


if __name__ == '__main__':
    main()
