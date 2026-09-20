"""VERIFY-WP-C2 item 1 -- the sphere normal against an INDEPENDENT oracle.

Its own sphere set (nothing shared with ``probe_c2_analytic_normal``): eight
radius magnitudes of BOTH signs, fourteen heights from the vertex to the
domain clamp, five azimuths, each evaluated twice -- once on a refracting
surface and once with the surface declared a MIRROR -- 2240 points.

The oracle is ``decimal`` at 60 significant digits::

    n = (-x/R, -y/R, sqrt(1 - (x**2 + y**2)/R**2))

evaluated entirely in ``Decimal`` from the EXACT float64 inputs (``Decimal``
of a float is exact), so the only error being measured is the library's.

ULP is measured in ULPs of the correctly-rounded float64 of the oracle value.

Usage: ``python vc2_sphere_oracle.py OUT.json``
"""
import json
import math
import os
import sys
from decimal import Decimal, getcontext

ROOT = os.environ.get('LUMENAIRY_ROOT')
if ROOT:
    sys.path.insert(0, ROOT)

import numpy as np                                            # noqa: E402
import lumenairy                                              # noqa: E402
from lumenairy.raytrace.surface import (                      # noqa: E402
    _surface_normal, _sphere_normal, _is_pure_spherical)
from lumenairy.raytrace.trace import Surface                  # noqa: E402

assert ROOT is None or os.path.abspath(lumenairy.__file__).startswith(
    os.path.abspath(ROOT)), (lumenairy.__file__, ROOT)

#: the oracle's own working precision.  60 digits is ~44 beyond float64;
#: ``VC2_PREC=120`` re-runs the whole sweep at twice that so the oracle's
#: own convergence is a MEASURED number rather than an assumption.
getcontext().prec = int(os.environ.get('VC2_PREC', '60'))

EPS = 2.220446049250313e-16


def _oracle(x, y, R, exact=True):
    """60-digit ``(nx, ny, nz)``.

    ``exact=True`` feeds the oracle the EXACT float64 inputs
    (``Decimal(float)`` is exact).  ``exact=False`` reproduces the
    ``Decimal(repr(float))`` conversion ``probe_c2_analytic_normal/
    sphere_oracle.py`` uses, which is the shortest ROUND-TRIPPING decimal
    and therefore differs from the value actually handed to the library
    by up to half an ULP of the input.  The two are compared so the
    oracle's own contribution is a measured number, not an assumption.
    """
    if exact:
        dx, dy, dR = Decimal(float(x)), Decimal(float(y)), Decimal(float(R))
    else:
        dx = Decimal(repr(float(x)))
        dy = Decimal(repr(float(y)))
        dR = Decimal(repr(float(R)))
    nx = -dx / dR
    ny = -dy / dR
    u = (dx * dx + dy * dy) / (dR * dR)
    nz = (Decimal(1) - u).sqrt()
    return nx, ny, nz


def _ulp_err(got, exact):
    """``abs(got - exact)`` in ULPs of the nearest float64 to ``exact``."""
    ref = float(exact)
    if not math.isfinite(got):
        return float('inf')
    u = math.ulp(ref) if ref != 0.0 else math.ulp(0.0)
    return float(abs(Decimal(got) - exact) / Decimal(u))


RADII_MM = [1.5, 7.3, 25.0, 43.7, 62.9, 101.3, 250.0, 777.0]
FRACS = [0.0, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95,
         0.97, 0.99, 0.995, 0.999, 0.9999, 0.99994]
AZ_DEG = [0.0, 37.0, 91.0, 163.0, 251.0]


def _surf(R, mirror):
    return Surface(radius=R, conic=0.0, thickness=0.0,
                   glass_before='AIR',
                   glass_after=('MIRROR' if mirror else 'N-BK7'),
                   is_mirror=bool(mirror), semi_diameter=abs(R))


def main(out_path):
    rows = []
    for mag_mm in RADII_MM:
        for sign in (+1.0, -1.0):
            R = sign * mag_mm * 1e-3
            for mirror in (False, True):
                surf = _surf(R, mirror)
                assert _is_pure_spherical(surf)
                xs, ys, fr, azs = [], [], [], []
                for f in FRACS:
                    h = f * abs(R)
                    for a in AZ_DEG:
                        xs.append(h * math.cos(math.radians(a)))
                        ys.append(h * math.sin(math.radians(a)))
                        fr.append(f)
                        azs.append(a)
                xs = np.array(xs)
                ys = np.array(ys)
                ax, ay, az = _surface_normal(xs, ys, surf,
                                             analytic_sphere=True)
                gx, gy, gz = _surface_normal(xs, ys, surf,
                                             analytic_sphere=False)
                sx, sy, sz = _sphere_normal(xs, ys, R)
                for i in range(len(xs)):
                    ox, oy, oz = _oracle(xs[i], ys[i], R)
                    a_err = max(_ulp_err(float(ax[i]), ox),
                                _ulp_err(float(ay[i]), oy),
                                _ulp_err(float(az[i]), oz))
                    g_err = max(_ulp_err(float(gx[i]), ox),
                                _ulp_err(float(gy[i]), oy),
                                _ulp_err(float(gz[i]), oz))
                    # metric B: the C2 probe's own -- worst ABSOLUTE
                    # component error divided by a fixed 2**-52.
                    a_b = max(abs(float(ax[i]) - float(ox)),
                              abs(float(ay[i]) - float(oy)),
                              abs(float(az[i]) - float(oz))) / (2.0 ** -52)
                    g_b = max(abs(float(gx[i]) - float(ox)),
                              abs(float(gy[i]) - float(oy)),
                              abs(float(gz[i]) - float(oz))) / (2.0 ** -52)
                    # the oracle's OWN input-conversion contribution
                    rx, ry, rz = _oracle(xs[i], ys[i], R, exact=False)
                    o_b = max(abs(float(rx) - float(ox)),
                              abs(float(ry) - float(oy)),
                              abs(float(rz) - float(oz))) / (2.0 ** -52)
                    unit_a = abs(float(ax[i]) ** 2 + float(ay[i]) ** 2
                                 + float(az[i]) ** 2 - 1.0)
                    unit_g = abs(float(gx[i]) ** 2 + float(gy[i]) ** 2
                                 + float(gz[i]) ** 2 - 1.0)
                    same = (float(sx[i]) == float(ax[i])
                            and float(sy[i]) == float(ay[i])
                            and (float(sz[i]) == float(az[i])
                                 or (math.isnan(float(sz[i]))
                                     and math.isnan(float(az[i])))))
                    rows.append(dict(
                        R=R, mirror=bool(mirror), f=fr[i], az=azs[i],
                        x=float(xs[i]), y=float(ys[i]),
                        a_ulp=a_err, g_ulp=g_err,
                        a_nan=(not math.isfinite(float(az[i]))),
                        g_nan=(not math.isfinite(float(gz[i]))),
                        spy_same=same, unit_a=unit_a, unit_g=unit_g,
                        a_b=a_b, g_b=g_b, o_b=o_b))

    by_key = {}
    for r in rows:
        by_key.setdefault((r['R'], r['f'], r['az']), []).append(r)
    mirror_moves = sum(1 for v in by_key.values()
                       if len(v) == 2 and (v[0]['a_ulp'] != v[1]['a_ulp']
                                           or v[0]['g_ulp'] != v[1]['g_ulp']))

    fin = [r for r in rows if not r['a_nan'] and not r['g_nan']]
    low = [r for r in fin if r['f'] <= 0.95]
    hi = [r for r in fin if r['f'] > 0.95]

    def _mx(rs, k):
        return max((r[k] for r in rs), default=float('nan'))

    worse = [r for r in fin if r['a_ulp'] > r['g_ulp'] + 1.0]
    worse_low = [r for r in low if r['a_ulp'] > r['g_ulp'] + 1.0]

    per_f = {}
    for f in FRACS:
        sel = [r for r in fin if r['f'] == f]
        if sel:
            per_f[str(f)] = dict(n=len(sel), a=_mx(sel, 'a_ulp'),
                                 g=_mx(sel, 'g_ulp'),
                                 aB=_mx(sel, 'a_b'), gB=_mx(sel, 'g_b'),
                                 oracleB=_mx(sel, 'o_b'))

    res = dict(
        build=dict(python=sys.version.split()[0], numpy=np.__version__,
                   lumenairy=lumenairy.__version__,
                   lumenairy_file=lumenairy.__file__),
        n_points=len(rows), n_finite=len(fin),
        n_low=len(low), n_high=len(hi),
        analytic_worst_ulp_le_0p95=_mx(low, 'a_ulp'),
        generic_worst_ulp_le_0p95=_mx(low, 'g_ulp'),
        analytic_worst_ulp_all=_mx(fin, 'a_ulp'),
        generic_worst_ulp_all=_mx(fin, 'g_ulp'),
        n_analytic_worse_by_gt_1ulp_le_0p95=len(worse_low),
        n_analytic_worse_by_gt_1ulp_all=len(worse),
        worst_analytic_deficit_all=max(
            (r['a_ulp'] - r['g_ulp'] for r in fin), default=0.0),
        worst_analytic_deficit_le_0p95=max(
            (r['a_ulp'] - r['g_ulp'] for r in low), default=0.0),
        closer=sum(1 for r in fin if r['a_ulp'] < r['g_ulp']),
        further=sum(1 for r in fin if r['a_ulp'] > r['g_ulp']),
        tie=sum(1 for r in fin if r['a_ulp'] == r['g_ulp']),
        unit_defect_analytic_worst_ulp=max(r['unit_a'] / (2 * EPS)
                                           for r in fin),
        unit_defect_generic_worst_ulp=max(r['unit_g'] / (2 * EPS)
                                          for r in fin),
        per_height=per_f,
        # --- metric B: the C2 probe's own (abs error / 2**-52) ---
        metricB_analytic_worst_le_0p95=_mx(low, 'a_b'),
        metricB_generic_worst_le_0p95=_mx(low, 'g_b'),
        metricB_analytic_worst_all=_mx(fin, 'a_b'),
        metricB_generic_worst_all=_mx(fin, 'g_b'),
        metricB_worst_analytic_deficit_le_0p95=max(
            (r['a_b'] - r['g_b'] for r in low), default=0.0),
        metricB_n_analytic_worse_by_gt_1_le_0p95=sum(
            1 for r in low if r['a_b'] > r['g_b'] + 1.0),
        metricB_n_analytic_worse_by_gt_1_all=sum(
            1 for r in fin if r['a_b'] > r['g_b'] + 1.0),
        metricB_worst_analytic_deficit_all=max(
            (r['a_b'] - r['g_b'] for r in fin), default=0.0),
        # --- the oracle's own input-conversion error, same units ---
        oracle_repr_conversion_worst_le_0p95=_mx(low, 'o_b'),
        oracle_repr_conversion_worst_all=_mx(fin, 'o_b'),
        oracle_repr_conversion_at_0p95=_mx(
            [r for r in fin if r['f'] == 0.95], 'o_b'),
        oracle_repr_conversion_vs_analytic_le_0p95=(
            _mx(low, 'o_b') / max(_mx(low, 'a_b'), 1e-300)),
        n_domain_gate_disagreements=sum(1 for r in rows
                                        if r['a_nan'] != r['g_nan']),
        mirror_changes_normal=mirror_moves,
        surface_normal_equals_sphere_normal=all(r['spy_same'] for r in rows),
        n_high_points=len(hi),
    )
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    print(json.dumps({k: v for k, v in res.items() if k != 'build'},
                     indent=1, sort_keys=True))


if __name__ == '__main__':
    main(sys.argv[1])
