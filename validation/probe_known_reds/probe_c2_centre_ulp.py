"""Probe for CI red
``TestC2FitRadiusCentre::test_the_on_axis_answer_moves_by_at_most_two_ulp``
(py3.14 shard 2: ``('increment', 0.05, 0.05000000000000001, 0.05)``).

The test asserts, for ``estimator='increment'`` on a CENTRED field, that the
residual-tilt projection (``centre='auto'``) and the unprojected fit
(``centre='origin'``) agree EXACTLY, byte for byte.

``_fit_carrier_inv._tilt_free_moment`` is

    num_c                      (the unprojected moment, = ``b``'s numerator)
    num_c - (mom/tot)*sum(w*C) (the projected moment, = ``a``'s numerator)

so ``a == b`` exactly iff ``(mom/tot)*sum(w*C)`` is small enough to vanish in
the subtraction, i.e. below half an ULP of ``num_c``.  On a centred field both
``mom = sum(w*slope)`` and ``sum(w*C) = sum(w*x)`` are ROUND-OFF sums whose
value depends on the summation ORDER -- which numpy's pairwise reduction ties
to the SIMD width the wheel dispatches to.  So whether the last bit moves is a
property of the build, not of the physics.

This probe reports, per arm and per estimator / radius:

  * a, b, their ULP distance and relative distance
  * the projection correction the library actually applied, per axis, as a
    fraction of the moment it corrects -- the quantity that decides whether
    the last bit moves
  * the raw round-off sums ``mom``, ``tot``, ``sum(w*x)``

Writes <out>/probe_c2_centre_ulp_<ARM>.json.
"""
import json
import os
import sys

import numpy as np

import lumenairy
from lumenairy.propagators import carrier as C

LAM = 1.31e-6
K0 = 2.0 * np.pi / LAM
N, DX, W = 512, 2e-6, 100e-6


def _field(r):
    # EXACTLY the test's construction, integer ``arange`` included -- the
    # int64 -> float64 promotion changes the last bits of the reading.
    g = (np.arange(N) - N / 2) * DX
    r2 = g[None, :] ** 2 + g[:, None] ** 2
    ph = (0.0 if np.isinf(r) else K0 * r2 / (2.0 * r))
    return np.exp(-r2 / W ** 2) * np.exp(1j * ph)


def _decentred_parabola(x0, R=50e-3):
    g = (np.arange(N, dtype=np.float64) - N / 2) * DX
    xx = g[None, :] - x0
    yy = g[:, None]
    r2 = xx * xx + yy * yy
    return np.exp(-r2 / W ** 2) * np.exp(1j * K0 * r2 / (2.0 * R))


def _ulp(a, b):
    if a == b:
        return 0.0
    if not (np.isfinite(a) and np.isfinite(b)):
        return float('inf')
    return float(abs(a - b) / np.spacing(abs(b)))


def _projection_terms(e, estimator):
    """The library's own ``_tilt_free_moment`` inputs, recomputed here from
    the same primitives, so the correction that decides the last bit is
    visible as a number."""
    x = (np.arange(N, dtype=np.float64) - N / 2) * DX
    y = x
    X, Y = x[None, :], y[:, None]
    out = {}
    if estimator == 'increment':
        # midpoint coordinates + midpoint |A| weights, x axis
        A = np.abs(e)
        ph = np.angle(e)
        dphi = np.angle(np.exp(1j * (ph[:, 1:] - ph[:, :-1])))
        wgt = 0.5 * (A[:, 1:] + A[:, :-1]) ** 2
        xm = 0.5 * (X[:, 1:] + X[:, :-1])
        q = wgt * (dphi / DX)
        num_c = float(np.sum(xm * q))
        tot = float(np.sum(wgt))
        mom = float(np.sum(q))
        swc = float(np.sum(wgt * xm))
        corr = (mom / tot) * swc if tot > 0 else 0.0
        out['x'] = dict(num_c=num_c, tot=tot, mom=mom, sum_w_C=swc,
                        corr=corr,
                        corr_over_ulp=(abs(corr) / np.spacing(abs(num_c))
                                       if num_c else float('inf')),
                        corr_rel=(abs(corr) / abs(num_c)
                                  if num_c else float('inf')))
    else:
        inten = np.abs(e) ** 2
        gy, gx = np.gradient(e, DX, DX)
        for lab, Cc, dE in (('x', X, gx), ('y', Y, gy)):
            q = np.imag(np.conj(e) * dE)
            num_c = float(np.sum(Cc * q))
            tot = float(np.sum(inten))
            mom = float(np.sum(q))
            swc = float(np.sum(inten * Cc))
            corr = (mom / tot) * swc if tot > 0 else 0.0
            out[lab] = dict(num_c=num_c, tot=tot, mom=mom, sum_w_C=swc,
                            corr=corr,
                            corr_over_ulp=(abs(corr) / np.spacing(abs(num_c))
                                           if num_c else float('inf')),
                            corr_rel=(abs(corr) / abs(num_c)
                                      if num_c else float('inf')))
    return out


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
    env = dict(arm=arm, python=sys.version.split()[0], numpy=np.__version__,
               platform=sys.platform, lumenairy_file=lumenairy.__file__,
               OPENBLAS_CORETYPE=os.environ.get('OPENBLAS_CORETYPE', ''),
               OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS', ''),
               threadpool=tpi)
    rows = []
    for est in ('gradient', 'increment'):
        for r in (50e-3, -20e-3, 1e9, np.inf):
            e = _field(r)
            a = C.carrier_referenced_fit_radius(
                e, LAM, DX, estimator=est, on_aliased='silent')
            b = C.carrier_referenced_fit_radius(
                e, LAM, DX, estimator=est, on_aliased='silent',
                centre='origin')
            rows.append(dict(
                kind='isotropic', estimator=est, r=(None if np.isinf(r)
                                                    else r),
                a=(None if np.isinf(a) else a),
                b=(None if np.isinf(b) else b),
                a_is_inf=bool(np.isinf(a)), b_is_inf=bool(np.isinf(b)),
                equal=bool(a == b), ulp=_ulp(a, b),
                rel=(0.0 if a == b else
                     (float('inf') if not np.isfinite(b)
                      else abs(a - b) / abs(b))),
                terms=_projection_terms(e, est)))
        e = _decentred_parabola(0.0)
        ua = C.carrier_referenced_fit_radius(
            e, LAM, DX, astigmatic=True, estimator=est, on_aliased='silent')
        ub = C.carrier_referenced_fit_radius(
            e, LAM, DX, astigmatic=True, estimator=est, on_aliased='silent',
            centre='origin')
        for i, (u, v) in enumerate(zip(ua, ub)):
            rows.append(dict(kind='astigmatic', estimator=est, axis=i,
                             a=float(u), b=float(v), equal=bool(u == v),
                             ulp=_ulp(float(u), float(v)),
                             rel=(0.0 if u == v
                                  else abs(u - v) / abs(v))))
    out = dict(env=env, rows=rows)
    path = os.path.join(outdir, 'probe_c2_centre_ulp_%s.json' % arm)
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1)
    print('arm=%s py=%s numpy=%s blas=%s'
          % (arm, env['python'], env['numpy'],
             tpi[0].get('architecture', '?') if tpi else '?'))
    for row in rows:
        extra = ''
        t = row.get('terms')
        if t:
            extra = ' corr/ulp=' + ','.join(
                '%s:%.3g' % (k, v['corr_over_ulp']) for k, v in t.items())
        print('%-10s %-10s %-12s a=%-22r b=%-22r eq=%-5s ulp=%.3g%s'
              % (row['kind'], row['estimator'],
                 row.get('r', row.get('axis')), row['a'], row['b'],
                 row['equal'], row['ulp'], extra))
    print('WROTE', path)


if __name__ == '__main__':
    main()
