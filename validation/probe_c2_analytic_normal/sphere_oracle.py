"""WP-C2 item 2 -- the 60-digit ``decimal`` normal ladder, re-run on
WP-C2's own sphere set.

Independent of the library and of WP-B9's test: the oracle here
evaluates the textbook normal of the sphere
``x^2 + y^2 + (z - R)^2 = R^2``

    n = (-dz/dx, -dz/dy, 1) / |.|,   dz/dh = h / (R sqrt(1 - h^2/R^2))

in ``decimal`` at 60 significant digits (~44 digits beyond float64, so
the oracle's own error is not measurable here), and compares BOTH
library routes against it:

* ``surface._sphere_normal`` -- the closed form WP-C2 is making the
  default;
* the generic sag-derivative route, reproduced from the library's own
  ``_surface_sag_derivatives_xy`` the way ``_surface_normal`` composes
  it, so the comparison is two-sided rather than a floor bar.

Sweep: radii of both signs from 2 mm to 1 m, heights from the vertex up
to the ``0.99995 |R|`` domain clamp, several azimuths, and the same set
with the surface declared a MIRROR (``glass_after='mirror'``), which
must not change the normal at all.

Usage:  LUMENAIRY_ROOT=<root> python sphere_oracle.py <out.json>
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


def _oracle(x, y, R, prec=60):
    ctx = decimal.Context(prec=prec)
    dec = ctx.create_decimal
    X, Y, RR = dec(repr(float(x))), dec(repr(float(y))), dec(repr(float(R)))
    h_sq = ctx.add(ctx.multiply(X, X), ctx.multiply(Y, Y))
    h = h_sq.sqrt(ctx)
    one = dec(1)
    inner = ctx.subtract(one, ctx.divide(h_sq, ctx.multiply(RR, RR)))
    root = inner.sqrt(ctx)
    dz_dh = ctx.divide(h, ctx.multiply(RR, root))
    if h == 0:
        dz_dx = dz_dy = dec(0)
    else:
        dz_dx = ctx.divide(ctx.multiply(dz_dh, X), h)
        dz_dy = ctx.divide(ctx.multiply(dz_dh, Y), h)
    mag = ctx.add(ctx.add(ctx.multiply(dz_dx, dz_dx),
                          ctx.multiply(dz_dy, dz_dy)), one).sqrt(ctx)
    return (float(ctx.divide(-dz_dx, mag)),
            float(ctx.divide(-dz_dy, mag)),
            float(ctx.divide(one, mag)))


def _generic(x, y, surf):
    dz_dx, dz_dy = _surf._surface_sag_derivatives_xy(x, y, surf)
    mag = np.sqrt(dz_dx ** 2 + dz_dy ** 2 + 1.0)
    return -dz_dx / mag, -dz_dy / mag, 1.0 / mag


RADII = [0.002, -0.002, 0.0515, -0.0345, 0.080, -0.120, 0.5, -1.0]
FRACS = [0.0, 0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 0.99, 0.999, 0.9999,
         0.99994]
AZIMUTHS = [0.0, 0.3, np.pi / 4, 1.1, 2.7, 4.9]


def main():
    out_path = sys.argv[1]
    rows = []
    for mirror in (False, True):
        for R in RADII:
            surf = Surface(
                radius=R, thickness=0.0, glass_before='air',
                glass_after=('mirror' if mirror else 'N-BK7'),
                semi_diameter=abs(R))
            for frac in FRACS:
                h = abs(R) * frac
                for th in AZIMUTHS:
                    x = h * np.cos(th)
                    y = h * np.sin(th)
                    xa = np.array([x])
                    ya = np.array([y])
                    fast = np.array(
                        [float(np.ravel(c)[0])
                         for c in _surf._sphere_normal(xa, ya, R)])
                    slow = np.array(
                        [float(np.ravel(c)[0])
                         for c in _generic(xa, ya, surf)])
                    ref = np.array(_oracle(x, y, R))
                    ef = float(np.max(np.abs(fast - ref)))
                    es = float(np.max(np.abs(slow - ref)))
                    unit = abs(float(np.sum(fast * fast)) - 1.0)
                    rows.append(dict(
                        mirror=mirror, R=R, frac=frac, az=float(th),
                        e_fast_ulp=ef / ULP, e_slow_ulp=es / ULP,
                        e_fast=ef, e_slow=es,
                        unit_defect_ulp=unit / ULP,
                        fast_nan=bool(np.any(~np.isfinite(fast))),
                        slow_nan=bool(np.any(~np.isfinite(slow)))))
    ef = np.array([r['e_fast_ulp'] for r in rows])
    es = np.array([r['e_slow_ulp'] for r in rows])
    finite = np.array([not (r['fast_nan'] or r['slow_nan']) for r in rows])
    lo = np.array([r['frac'] <= 0.95 for r in rows]) & finite
    # mirror invariance: the normal cannot depend on glass_after
    half = len(rows) // 2
    mirror_same = all(
        rows[i]['e_fast'] == rows[half + i]['e_fast']
        and rows[i]['e_slow'] == rows[half + i]['e_slow']
        for i in range(half))
    summary = dict(
        n_points=len(rows),
        n_finite=int(finite.sum()),
        worst_fast_ulp_all=float(np.max(ef[finite])),
        worst_slow_ulp_all=float(np.max(es[finite])),
        worst_fast_ulp_to_0p95=float(np.max(ef[lo])),
        worst_slow_ulp_to_0p95=float(np.max(es[lo])),
        worst_unit_defect_ulp=float(
            np.max([r['unit_defect_ulp'] for r in rows if not r['fast_nan']])),
        fast_wins=int(np.sum(ef[finite] < es[finite])),
        slow_wins=int(np.sum(ef[finite] > es[finite])),
        ties=int(np.sum(ef[finite] == es[finite])),
        fast_never_worse_by_more_than_ulp=bool(
            np.all(ef[finite] <= es[finite] + 1.0)),
        mirror_invariant=mirror_same,
        n_fast_nan=int(sum(r['fast_nan'] for r in rows)),
        n_slow_nan=int(sum(r['slow_nan'] for r in rows)),
        n_gate_disagree=int(sum(r['fast_nan'] != r['slow_nan']
                                for r in rows)),
    )
    for k, v in summary.items():
        print(f'{k}: {v}')
    meta = dict(python=sys.version, numpy=np.__version__,
                lumenairy=la.__version__, platform=sys.platform, prec=60)
    with open(out_path, 'w') as fh:
        json.dump(dict(meta=meta, summary=summary, rows=rows), fh, indent=1)
    print('wrote', out_path)


if __name__ == '__main__':
    main()
