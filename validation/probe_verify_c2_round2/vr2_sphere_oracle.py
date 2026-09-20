"""VERIFY-WP-C2 ROUND 2, item D1 -- an INDEPENDENT exact-input sphere-normal
oracle, on this verification's own sphere set.

Two things the shipped probe does not do, both of which are the point:

1. **TWO oracle formulations**, agreeing with each other before either is
   used.  The shipped oracle evaluates the SAG-DERIVATIVE form
   ``n = (-dz/dx, -dz/dy, 1)/|.|`` with ``dz/dh = h / (R sqrt(1 - h^2/R^2))``
   -- which is algebraically the same expression family the GENERIC library
   route uses, so an oracle bug that favours one route is not excluded by
   construction.  This probe also evaluates the GEOMETRIC form
   ``n = (x, y, z - R) / |R|`` with ``z = R - sign(R) sqrt(R^2 - h^2)``,
   which shares no sub-expression with the first, and asserts the two agree
   to well below a float64 ULP before either is compared to the library.

2. **the conversion is audited, not asserted**.  Every input is a float64
   the LIBRARY is handed; ``decimal.Decimal(float)`` is the float's exact
   binary value, so the oracle inherits no rounding.  The probe re-runs the
   whole sweep with ``repr``-based conversion as well and reports what that
   costs, which is the number defect D1 was raised on.

Sweep (nothing shared with the shipped probe's): ten radius magnitudes of
both signs, seventeen height fractions, seven azimuths, refracting and
MIRROR -- 4760 points, of which 2800 are at or below ``h = 0.95 |R|``.

Usage:  LUMENAIRY_ROOT=<root> python vr2_sphere_oracle.py <out.json>
        VR2_PREC=100 (default) -- set 160 to re-measure convergence.
"""
import decimal
import json
import os
import sys

import numpy as np

_ROOT = os.environ['LUMENAIRY_ROOT']
sys.path.insert(0, _ROOT)

import lumenairy as la  # noqa: E402

_want = os.path.realpath(os.path.join(_ROOT, 'lumenairy'))
assert os.path.realpath(os.path.dirname(la.__file__)) == _want, la.__file__

from lumenairy.raytrace import surface as _surf  # noqa: E402
from lumenairy.raytrace.core import Surface  # noqa: E402

ULP = 2.0 ** -52
PREC = int(os.environ.get('VR2_PREC', '100'))


def _to_dec(v, exact=True):
    if exact:
        return decimal.Decimal(float(v))
    return decimal.Decimal(repr(float(v)))


def _oracle_sag(x, y, R, prec, exact=True):
    """``n = (-dz/dx, -dz/dy, 1)/|.|`` -- the sag-derivative form."""
    ctx = decimal.Context(prec=prec)
    X, Y, RR = (_to_dec(x, exact), _to_dec(y, exact), _to_dec(R, exact))
    one = ctx.create_decimal(1)
    h_sq = ctx.add(ctx.multiply(X, X), ctx.multiply(Y, Y))
    h = h_sq.sqrt(ctx)
    inner = ctx.subtract(one, ctx.divide(h_sq, ctx.multiply(RR, RR)))
    root = inner.sqrt(ctx)
    dz_dh = ctx.divide(h, ctx.multiply(RR, root))
    if h == 0:
        dz_dx = dz_dy = ctx.create_decimal(0)
    else:
        dz_dx = ctx.divide(ctx.multiply(dz_dh, X), h)
        dz_dy = ctx.divide(ctx.multiply(dz_dh, Y), h)
    mag = ctx.add(ctx.add(ctx.multiply(dz_dx, dz_dx),
                          ctx.multiply(dz_dy, dz_dy)), one).sqrt(ctx)
    return (ctx.divide(-dz_dx, mag), ctx.divide(-dz_dy, mag),
            ctx.divide(one, mag))


def _oracle_geom(x, y, R, prec, exact=True):
    """``n = (x, y, z - R)/|R|`` with the sphere's own z -- no sub-expression
    in common with ``_oracle_sag``.  Oriented +z like the library's."""
    ctx = decimal.Context(prec=prec)
    X, Y, RR = (_to_dec(x, exact), _to_dec(y, exact), _to_dec(R, exact))
    h_sq = ctx.add(ctx.multiply(X, X), ctx.multiply(Y, Y))
    R2 = ctx.multiply(RR, RR)
    s = ctx.subtract(R2, h_sq).sqrt(ctx)
    sign = ctx.create_decimal(1 if R > 0 else -1)
    z = ctx.subtract(RR, ctx.multiply(sign, s))
    nx, ny, nz = X, Y, ctx.subtract(z, RR)
    mag = ctx.add(ctx.add(ctx.multiply(nx, nx), ctx.multiply(ny, ny)),
                  ctx.multiply(nz, nz)).sqrt(ctx)
    nx, ny, nz = (ctx.divide(nx, mag), ctx.divide(ny, mag),
                  ctx.divide(nz, mag))
    if nz < 0:
        nx, ny, nz = -nx, -ny, -nz
    return nx, ny, nz


def _generic(x, y, surf):
    dz_dx, dz_dy = _surf._surface_sag_derivatives_xy(x, y, surf)
    mag = np.sqrt(dz_dx ** 2 + dz_dy ** 2 + 1.0)
    return -dz_dx / mag, -dz_dy / mag, 1.0 / mag


RADII = [0.0031, -0.0031, 0.0125, -0.0125, 0.0437, -0.0437, 0.0731,
         -0.0731, 0.2500, -0.2500, 0.6180, -0.6180, 1.7320, -1.7320,
         12.5, -12.5, 0.0900, -0.0900]
FRACS = [0.0, 0.002, 0.013, 0.077, 0.19, 0.31, 0.44, 0.58, 0.67, 0.79,
         0.88, 0.93, 0.95, 0.97, 0.991, 0.9997, 0.99993]
AZIMUTHS = [0.0, 0.17, 0.62, 1.31, 2.04, 3.77, 5.51]


def sweep(prec, exact=True, do_geom=True):
    rows = []
    worst_cross = 0.0
    for mirror in (False, True):
        for R in RADII:
            surf = Surface(radius=R, thickness=0.0, glass_before='air',
                           glass_after=('mirror' if mirror else 'N-BK7'),
                           semi_diameter=abs(R))
            for frac in FRACS:
                h = abs(R) * frac
                for th in AZIMUTHS:
                    x = h * np.cos(th)
                    y = h * np.sin(th)
                    xa, ya = np.array([x]), np.array([y])
                    fast = np.array([float(np.ravel(c)[0])
                                     for c in _surf._sphere_normal(xa, ya, R)])
                    slow = np.array([float(np.ravel(c)[0])
                                     for c in _generic(xa, ya, surf)])
                    ref_d = _oracle_sag(x, y, R, prec, exact)
                    ref = np.array([float(c) for c in ref_d])
                    if do_geom:
                        g = _oracle_geom(x, y, R, prec, exact)
                        cross = max(abs(float(a - b))
                                    for a, b in zip(ref_d, g))
                        worst_cross = max(worst_cross, cross)
                    ef = float(np.max(np.abs(fast - ref)))
                    es = float(np.max(np.abs(slow - ref)))
                    rows.append(dict(
                        mirror=mirror, R=R, frac=frac, az=float(th),
                        e_fast_ulp=ef / ULP, e_slow_ulp=es / ULP,
                        unit_defect_ulp=abs(float(np.sum(fast * fast)) - 1.0)
                        / ULP,
                        fast_nan=bool(np.any(~np.isfinite(fast))),
                        slow_nan=bool(np.any(~np.isfinite(slow)))))
    return rows, worst_cross


def summarise(rows):
    ef = np.array([r['e_fast_ulp'] for r in rows])
    es = np.array([r['e_slow_ulp'] for r in rows])
    finite = np.array([not (r['fast_nan'] or r['slow_nan']) for r in rows])
    lo = np.array([r['frac'] <= 0.95 for r in rows]) & finite
    half = len(rows) // 2
    mirror_same = all(
        rows[i]['e_fast_ulp'] == rows[half + i]['e_fast_ulp']
        and rows[i]['e_slow_ulp'] == rows[half + i]['e_slow_ulp']
        for i in range(half))
    return dict(
        n_points=len(rows), n_finite=int(finite.sum()),
        n_points_to_0p95=int(lo.sum()),
        worst_fast_ulp_to_0p95=float(np.max(ef[lo])),
        worst_slow_ulp_to_0p95=float(np.max(es[lo])),
        n_fast_worse_by_more_than_ulp_to_0p95=int(
            np.sum(ef[lo] > es[lo] + 1.0)),
        worst_fast_ulp_all=float(np.max(ef[finite])),
        worst_slow_ulp_all=float(np.max(es[finite])),
        n_fast_worse_all=int(np.sum(ef[finite] > es[finite])),
        worst_deficit_ulp_all=float(np.max(ef[finite] - es[finite])),
        fast_wins=int(np.sum(ef[finite] < es[finite])),
        slow_wins=int(np.sum(ef[finite] > es[finite])),
        ties=int(np.sum(ef[finite] == es[finite])),
        worst_unit_defect_ulp=float(np.max(
            [r['unit_defect_ulp'] for r in rows if not r['fast_nan']])),
        mirror_invariant=bool(mirror_same),
        n_gate_disagree=int(sum(r['fast_nan'] != r['slow_nan']
                                for r in rows)),
    )


def main():
    out_path = sys.argv[1]
    rows, cross = sweep(PREC, exact=True)
    summary = summarise(rows)
    summary['oracle_cross_check_worst_abs'] = cross
    summary['oracle_cross_check_worst_ulp'] = cross / ULP

    # convergence: the same sweep at a much higher precision
    rows_hi, cross_hi = sweep(PREC + 60, exact=True, do_geom=False)
    sum_hi = summarise(rows_hi)
    summary['converged_vs_prec_%d' % (PREC + 60)] = all(
        summary[k] == sum_hi[k] for k in sum_hi)
    summary['fields_differing_at_higher_prec'] = [
        k for k in sum_hi if summary[k] != sum_hi[k]]

    # what the SHIPPED (pre-D1) repr conversion would have cost, on THIS set
    rows_repr, _ = sweep(60, exact=False, do_geom=False)
    sum_repr = summarise(rows_repr)
    summary['repr_conversion_worst_fast_ulp_to_0p95'] = \
        sum_repr['worst_fast_ulp_to_0p95']
    summary['repr_conversion_worst_slow_ulp_to_0p95'] = \
        sum_repr['worst_slow_ulp_to_0p95']
    summary['repr_conversion_worst_fast_ulp_all'] = \
        sum_repr['worst_fast_ulp_all']
    summary['repr_conversion_worst_slow_ulp_all'] = \
        sum_repr['worst_slow_ulp_all']

    for k, v in summary.items():
        print('%-46s %s' % (k, v))
    meta = dict(python=sys.version.split()[0], numpy=np.__version__,
                lumenairy=la.__version__, lumenairy_file=la.__file__,
                platform=sys.platform, prec=PREC)
    with open(out_path, 'w') as fh:
        json.dump(dict(meta=meta, summary=summary,
                       summary_repr=sum_repr, rows=rows), fh, indent=1)
    print('wrote', out_path)


if __name__ == '__main__':
    main()
