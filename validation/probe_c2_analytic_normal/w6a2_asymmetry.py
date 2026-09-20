"""WP-C2 item 1 -- is the w6_a2 root's offset ROUNDING or the fit's own
asymmetry?

w6a2_oracle.py showed the library's ``v2*`` IS the exact root of the
fit's polynomial system (agreement 3e-22), and that the root is 4.6e-16
to 8.6e-16 off the pupil centre -- so ``|v2*| < 1e-15`` is not a
rounding floor, it is an accidental bound on how asymmetric the FITTED
coefficients happen to be.

The scale of that offset follows from the residual AT the pupil centre::

    |v* - v_c|  ~=  |r(v_c)| / sigma_min(H)

This probe measures ``|r(v_c)|`` both in float64 and at 60 digits.  If
the two agree, the residual at the centre is the fit's genuine asymmetry
(not rounding) and the bound is computable inside a test in float64.

Usage:  LUMENAIRY_ROOT=<root> python w6a2_asymmetry.py <out.json>
"""
import decimal
import json
import os
import sys
from decimal import Decimal as D

import numpy as np

_ROOT = os.environ['LUMENAIRY_ROOT']
sys.path.insert(0, _ROOT)

import lumenairy as la  # noqa: E402

_want = os.path.realpath(os.path.join(_ROOT, 'lumenairy'))
assert os.path.realpath(os.path.dirname(la.__file__)) == _want, la.__file__

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from w6a2_oracle import _cheb, _load, _poly, _set_defaults  # noqa: E402

decimal.getcontext().prec = 60


def _residual_f64(fit, s2x, s2y, srcx, srcy, w_s, w_p, vx, vy, vcx, vcy):
    s1x, s1y, a, b, c, d = fit.eval_s1_with_v2_grad(
        np.asarray(s2x).reshape(()), np.asarray(s2y).reshape(()),
        np.asarray(vx).reshape(()), np.asarray(vy).reshape(()))
    J = np.array([[float(a), float(b)], [float(c), float(d)]])
    ds1 = np.array([float(s1x) - srcx, float(s1y) - srcy])
    dv = np.array([vx - vcx, vy - vcy])
    r = (J.T @ ds1) / (w_s * w_s) + dv / (w_p * w_p)
    H = (J.T @ J) / (w_s * w_s) + np.eye(2) / (w_p * w_p)
    return r, H, J, ds1


def _residual_dec(fit, s2x, s2y, srcx, srcy, w_s, w_p, vx, vy, vcx, vcy):
    P = int(fit.poly_order)
    K = [tuple(int(v) for v in row) for row in np.asarray(fit.multi_indices)]
    cx = [D(repr(float(t))) for t in np.asarray(fit.coef_s1x)]
    cy = [D(repr(float(t))) for t in np.asarray(fit.coef_s1y)]
    u1 = ((D(repr(s2x)) - D(repr(float(fit.s2x_centre))))
          / D(repr(float(fit.s2x_halfrange))))
    u2 = ((D(repr(s2y)) - D(repr(float(fit.s2y_centre))))
          / D(repr(float(fit.s2y_halfrange))))
    v2xh = D(repr(float(fit.v2x_halfrange)))
    v2yh = D(repr(float(fit.v2y_halfrange)))
    u3 = (D(repr(vx)) - D(repr(float(fit.v2x_centre)))) / v2xh
    u4 = (D(repr(vy)) - D(repr(float(fit.v2y_centre)))) / v2yh
    T1, _ = _cheb(u1, P)
    T2, _ = _cheb(u2, P)
    T3, dT3 = _cheb(u3, P)
    T4, dT4 = _cheb(u4, P)
    s1x, gx3, gx4 = _poly(cx, K, T1, T2, T3, T4, dT3, dT4)
    s1y, gy3, gy4 = _poly(cy, K, T1, T2, T3, T4, dT3, dT4)
    j00, j01 = gx3 / v2xh, gx4 / v2yh
    j10, j11 = gy3 / v2xh, gy4 / v2yh
    iws = D(1) / (D(repr(w_s)) * D(repr(w_s)))
    iwp = D(1) / (D(repr(w_p)) * D(repr(w_p)))
    dx = s1x - D(repr(srcx))
    dy = s1y - D(repr(srcy))
    rx = iws * (j00 * dx + j10 * dy) + iwp * (D(repr(vx)) - D(repr(vcx)))
    ry = iws * (j01 * dx + j11 * dy) + iwp * (D(repr(vy)) - D(repr(vcy)))
    return rx, ry, s1x, s1y


def _measure(w6):
    w6._fit.cache_clear()
    fit = w6._fit()
    sx, sy = float(fit.s2x_centre), float(fit.s2y_centre)
    vcx, vcy = float(fit.v2x_centre), float(fit.v2y_centre)
    vx, vy, _ = w6._solve_envelope_stationary_batch(
        fit, np.array([sx]), np.array([sy]), 0.0, 0.0,
        w_s=20e-6, w_p=0.02, v_cx=vcx, v_cy=vcy)
    vbx, vby = float(vx[0]), float(vy[0])
    r64, H, J, ds1 = _residual_f64(fit, sx, sy, 0.0, 0.0, 20e-6, 0.02,
                                   vcx, vcy, vcx, vcy)
    rdx, rdy, s1x_d, s1y_d = _residual_dec(
        fit, sx, sy, 0.0, 0.0, 20e-6, 0.02, vcx, vcy, vcx, vcy)
    sv = np.linalg.svd(H, compute_uv=False)
    svmin = float(np.min(sv))
    rn64 = float(np.linalg.norm(r64))
    rndec = float((rdx * rdx + rdy * rdy).sqrt())
    return dict(
        v_batch=(vbx, vby), worst_batch=max(abs(vbx), abs(vby)),
        r_at_centre_f64=[float(r64[0]), float(r64[1])],
        r_at_centre_dec=[str(rdx), str(rdy)],
        rn_f64=rn64, rn_dec=rndec,
        rn_f64_vs_dec_rel=abs(rn64 - rndec) / max(rndec, 1e-300),
        svmin=svmin, svmax=float(np.max(sv)),
        bound_rn_over_svmin=rn64 / svmin,
        ratio_v_over_bound=max(abs(vbx), abs(vby)) / (rn64 / svmin),
        s1_at_centre=[float(np.asarray(ds1)[0]), float(np.asarray(ds1)[1])],
        halfrange=float(fit.v2x_halfrange),
        res_s1_rms_m=float(fit.res_s1_rms_m),
    )


def main():
    out = sys.argv[1]
    w6 = _load('_w6', os.path.join(_ROOT, 'tests', 'unit',
                                   'test_niche_audit_w6_asymptotic.py'))
    import importlib as _il
    tm = _il.import_module('lumenairy.raytrace.trace')
    shipped = (tm.trace.__defaults__[-2], tm.trace.__defaults__[-1])
    res = {}
    for renorm in ('surface', 'exit'):
        for sph in ('generic', 'analytic'):
            _set_defaults(renorm, sph)
            k = renorm + '/' + sph
            res[k] = _measure(w6)
            r = res[k]
            print('[%s] |v|=%.4e  |r(v_c)|f64=%.6e dec=%.6e (rel %.1e)  '
                  'svmin=%.4e  bound=%.4e  v/bound=%.3f'
                  % (k, r['worst_batch'], r['rn_f64'], r['rn_dec'],
                     r['rn_f64_vs_dec_rel'], r['svmin'],
                     r['bound_rn_over_svmin'], r['ratio_v_over_bound']),
                  flush=True)
    _set_defaults(*shipped)
    meta = dict(python=sys.version, numpy=np.__version__,
                lumenairy=la.__version__, platform=sys.platform)
    with open(out, 'w') as fh:
        json.dump(dict(meta=meta, results=res), fh, indent=1)
    print('wrote', out)


if __name__ == '__main__':
    main()
