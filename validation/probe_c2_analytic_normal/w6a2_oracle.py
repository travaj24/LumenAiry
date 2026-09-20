"""WP-C2 item 1 -- a 60-digit ``decimal`` oracle for the w6_a2 root.

``test_w6_a2_v2_star_is_untouched_by_the_verdict_fix`` asserts
``|v2*| < 1e-15`` at the fit centre, on the argument that "on-axis
source, on-axis image, rotationally symmetric singlet: the
envelope-stationary pupil point is the pupil centre".  That is an S4
floor bar with no derivation, and VERIFY-WP-B9 read 1.150e-15 there
under ``sphere_normal='analytic'``.

The premise is testable.  The solver's root is the root of a POLYNOMIAL
system whose coefficients are the fit's own float64 numbers::

    r(v) = J^T (s1(s2, v) - s_src) / w_s^2 + (v - v_c) / w_p^2 = 0
    s1   = sum_m c_m T_k1(u1) T_k2(u2) T_k3(u3) T_k4(u4)
    J    = d s1 / d v,  chain rule 1 / v2_halfrange

Nothing in it is transcendental, so the exact root of THAT system is
computable in ``decimal`` at 60 significant digits -- ~44 digits beyond
float64, so the oracle's own error is not measurable here.  This probe
computes it and compares it with what the library returns, under each of
the four (renormalize, sphere_normal) default combinations.

Usage:  LUMENAIRY_ROOT=<root> python w6a2_oracle.py <out.json>
"""
import decimal
import importlib
import importlib.util
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

trace_mod = importlib.import_module('lumenairy.raytrace.trace')
wtrace_mod = importlib.import_module('lumenairy.raytrace.world_trace')

decimal.getcontext().prec = 60


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _set_defaults(renorm, sph):
    for fn in (trace_mod.trace, wtrace_mod.trace_world):
        d = list(fn.__defaults__)
        d[-2], d[-1] = renorm, sph
        fn.__defaults__ = tuple(d)


def _cheb(u, n):
    """T_0..T_n and T'_0..T'_n at 60 digits (the library's recurrences)."""
    T = [D(1)]
    if n >= 1:
        T.append(u)
    for k in range(2, n + 1):
        T.append(2 * u * T[k - 1] - T[k - 2])
    U = [D(1)]
    if n >= 1:
        U.append(2 * u)
    for k in range(2, n + 1):
        U.append(2 * u * U[k - 1] - U[k - 2])
    Tp = [D(0)] + [D(k) * U[k - 1] for k in range(1, n + 1)]
    return T[:n + 1], Tp[:n + 1]


def _poly(coef, K, T1, T2, T3, T4, dT3, dT4):
    """f, df/du3, df/du4 at 60 digits."""
    f = D(0)
    d3 = D(0)
    d4 = D(0)
    for c, kk in zip(coef, K):
        k1, k2, k3, k4 = kk
        t12 = T1[k1] * T2[k2]
        f += c * t12 * T3[k3] * T4[k4]
        d3 += c * t12 * dT3[k3] * T4[k4]
        d4 += c * t12 * T3[k3] * dT4[k4]
    return f, d3, d4


def _oracle_root(fit, s2x, s2y, srcx, srcy, w_s, w_p, vcx, vcy, v0,
                 iters=60):
    P = int(fit.poly_order)
    K = [tuple(int(v) for v in row) for row in np.asarray(fit.multi_indices)]
    cx = [D(repr(float(c))) for c in np.asarray(fit.coef_s1x)]
    cy = [D(repr(float(c))) for c in np.asarray(fit.coef_s1y)]
    s2xc = D(repr(float(fit.s2x_centre)))
    s2xh = D(repr(float(fit.s2x_halfrange)))
    s2yc = D(repr(float(fit.s2y_centre)))
    s2yh = D(repr(float(fit.s2y_halfrange)))
    v2xc = D(repr(float(fit.v2x_centre)))
    v2xh = D(repr(float(fit.v2x_halfrange)))
    v2yc = D(repr(float(fit.v2y_centre)))
    v2yh = D(repr(float(fit.v2y_halfrange)))
    u1 = (D(repr(s2x)) - s2xc) / s2xh
    u2 = (D(repr(s2y)) - s2yc) / s2yh
    T1, _ = _cheb(u1, P)
    T2, _ = _cheb(u2, P)
    inv_ws2 = D(1) / (D(repr(w_s)) * D(repr(w_s)))
    inv_wp2 = D(1) / (D(repr(w_p)) * D(repr(w_p)))
    dsrcx = D(repr(srcx))
    dsrcy = D(repr(srcy))
    dvcx = D(repr(vcx))
    dvcy = D(repr(vcy))
    vx = D(repr(v0[0]))
    vy = D(repr(v0[1]))
    rn = None
    for _ in range(iters):
        u3 = (vx - v2xc) / v2xh
        u4 = (vy - v2yc) / v2yh
        T3, dT3 = _cheb(u3, P)
        T4, dT4 = _cheb(u4, P)
        s1x, gx3, gx4 = _poly(cx, K, T1, T2, T3, T4, dT3, dT4)
        s1y, gy3, gy4 = _poly(cy, K, T1, T2, T3, T4, dT3, dT4)
        j00 = gx3 / v2xh
        j01 = gx4 / v2yh
        j10 = gy3 / v2xh
        j11 = gy4 / v2yh
        dx = s1x - dsrcx
        dy = s1y - dsrcy
        rx = inv_ws2 * (j00 * dx + j10 * dy) + inv_wp2 * (vx - dvcx)
        ry = inv_ws2 * (j01 * dx + j11 * dy) + inv_wp2 * (vy - dvcy)
        rn = (rx * rx + ry * ry).sqrt()
        # Same Gauss-Newton Hessian MODEL the library uses -- it changes
        # the convergence RATE, never the root -- iterated to the
        # oracle's own floor rather than to a 1e-12 residual.
        h00 = inv_ws2 * (j00 * j00 + j10 * j10) + inv_wp2
        h01 = inv_ws2 * (j00 * j01 + j10 * j11)
        h11 = inv_ws2 * (j01 * j01 + j11 * j11) + inv_wp2
        det = h00 * h11 - h01 * h01
        sx = (h11 * rx - h01 * ry) / det
        sy = (h00 * ry - h01 * rx) / det
        vx = vx - sx
        vy = vy - sy
        if abs(sx) < D('1e-55') and abs(sy) < D('1e-55'):
            break
    return vx, vy, rn


def _measure(w6):
    w6._fit.cache_clear()
    fit = w6._fit()
    sx = float(fit.s2x_centre)
    sy = float(fit.s2y_centre)
    vcx = float(fit.v2x_centre)
    vcy = float(fit.v2y_centre)
    vx, vy, _ = w6._solve_envelope_stationary_batch(
        fit, np.array([sx]), np.array([sy]), 0.0, 0.0,
        w_s=20e-6, w_p=0.02, v_cx=vcx, v_cy=vcy)
    vbx = float(vx[0])
    vby = float(vy[0])
    ox, oy, rn = _oracle_root(fit, sx, sy, 0.0, 0.0, 20e-6, 0.02,
                              vcx, vcy, (vcx, vcy))
    hr = float(fit.v2x_halfrange)
    ox2, oy2, _ = _oracle_root(fit, sx, sy, 0.0, 0.0, 20e-6, 0.02,
                               vcx, vcy, (vcx + 0.1 * hr, vcy - 0.1 * hr))
    return dict(
        v_batch=(vbx, vby),
        worst_batch=max(abs(vbx), abs(vby)),
        v_oracle=(str(ox), str(oy)),
        worst_oracle=float(max(abs(ox), abs(oy))),
        oracle_self=float(max(abs(ox - ox2), abs(oy - oy2))),
        lib_vs_oracle=float(max(abs(D(repr(vbx)) - ox),
                                abs(D(repr(vby)) - oy))),
        halfrange=hr,
        coef_s1x_scale=float(np.max(np.abs(np.asarray(fit.coef_s1x)))),
        coef_s1y_scale=float(np.max(np.abs(np.asarray(fit.coef_s1y)))),
    )


def main():
    out = sys.argv[1]
    w6 = _load('_w6', os.path.join(_ROOT, 'tests', 'unit',
                                   'test_niche_audit_w6_asymptotic.py'))
    shipped = (trace_mod.trace.__defaults__[-2],
               trace_mod.trace.__defaults__[-1])
    res = {}
    for renorm in ('surface', 'exit'):
        for sph in ('generic', 'analytic'):
            _set_defaults(renorm, sph)
            k = renorm + '/' + sph
            res[k] = _measure(w6)
            r = res[k]
            print('[%s] lib=%.5e oracle=%.5e |lib-oracle|=%.3e '
                  'oracle_self=%.2e' % (k, r['worst_batch'],
                                        r['worst_oracle'],
                                        r['lib_vs_oracle'],
                                        r['oracle_self']), flush=True)
    _set_defaults(*shipped)
    meta = dict(python=sys.version, numpy=np.__version__,
                lumenairy=la.__version__, platform=sys.platform, prec=60)
    with open(out, 'w') as fh:
        json.dump(dict(meta=meta, results=res), fh, indent=1)
    print('wrote', out)


if __name__ == '__main__':
    main()
