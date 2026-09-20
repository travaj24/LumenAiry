"""VERIFY-WP-C2 ROUND 2, items D3 and D9 -- the history-drift coefficient
and the rim paragraph's two facts, re-measured on this verification's own
ladder and its own radii.

D3.  The ``renormalize`` docstring's bound is ``n_surfaces * eps`` and its
     coefficient is claimed to run 1.000 at three surfaces to 0.615 at
     thirteen, with ``1e-15`` first exceeded at the SEVENTH.  Measured here
     on a ladder built from different radii and a different bundle.

D9.  Two facts:
     * the two normal routes' domain gates, LOCATED BY BISECTION on the
       meridian, are the SAME float at every radius tested (so the
       one-ULP band does not exist at ``y = 0``);
     * a ball lens / hemisphere whose clear semi-diameter IS ``|R|`` loses
       a measurable fraction of a rim-packed bundle to ``RAY_NAN`` on BOTH
       routes.

Usage:  LUMENAIRY_ROOT=<root> python vr2_d3_d9.py <out.json>
"""
import json
import os
import sys

import numpy as np

_ROOT = os.environ['LUMENAIRY_ROOT']
sys.path.insert(0, _ROOT)

import lumenairy as la  # noqa: E402

_want = os.path.realpath(os.path.join(_ROOT, 'lumenairy'))
assert os.path.realpath(os.path.dirname(la.__file__)) == _want, la.__file__

from lumenairy.raytrace.surface import RAY_NAN, RayBundle, Surface  # noqa: E402
from lumenairy.raytrace.trace import trace  # noqa: E402

WL = 587.5618e-9
EPS = float(np.finfo(np.float64).eps)


def ladder(n_pairs):
    """A refracting ladder with radii nothing like the shipped probe's."""
    out = []
    for j in range(n_pairs):
        out.append(Surface(radius=0.0623 + 0.0041 * j, thickness=0.0048,
                           glass_before='air', glass_after='N-SF5',
                           semi_diameter=0.0105))
        out.append(Surface(radius=-0.0771 - 0.0037 * j, thickness=0.0083,
                           glass_before='N-SF5', glass_after='air',
                           semi_diameter=0.0105))
    out.append(Surface(radius=0.3100, thickness=0.0250, glass_before='air',
                       glass_after='N-BK7', semi_diameter=0.0105))
    return out


def bundle(n, hmax, tilt_deg, seed=987654):
    rng = np.random.default_rng(seed)
    r = hmax * np.sqrt(rng.uniform(0, 1, n))
    th = rng.uniform(0, 2 * np.pi, n)
    L = np.full(n, np.sin(np.radians(tilt_deg)))
    return RayBundle(x=r * np.cos(th), y=r * np.sin(th), z=np.zeros(n),
                     L=L, M=np.zeros(n),
                     N=np.sqrt(np.maximum(1.0 - L ** 2, 0.0)),
                     wavelength=WL, alive=np.ones(n, dtype=bool),
                     opd=np.zeros(n))


def d3(out):
    rows = []
    for n_pairs in (1, 2, 3, 4, 5, 6):
        surfs = ladder(n_pairs)
        n_surf = len(surfs)
        row = {'n_surfaces': n_surf}
        for mode in ('exit', 'surface'):
            rb = bundle(2000, 0.0090, 2.0)
            res = trace(rb, surfs, WL, output_filter='all',
                        renormalize=mode, sphere_normal='analytic')
            worst = 0.0
            for b in list(res.ray_history)[:-1]:
                d = np.abs(np.sqrt(np.asarray(b.L) ** 2
                                   + np.asarray(b.M) ** 2
                                   + np.asarray(b.N) ** 2) - 1.0)
                alive = np.asarray(b.alive, dtype=bool)
                if alive.any():
                    worst = max(worst, float(np.max(d[alive])))
            row['history_drift_' + mode] = worst
            last = list(res.ray_history)[-1]
            dl = np.abs(np.sqrt(np.asarray(last.L) ** 2
                                + np.asarray(last.M) ** 2
                                + np.asarray(last.N) ** 2) - 1.0)
            al = np.asarray(last.alive, dtype=bool)
            row['final_drift_' + mode] = (float(np.max(dl[al]))
                                          if al.any() else 0.0)
        row['envelope_n_eps'] = n_surf * EPS
        row['ratio_exit_to_n_eps'] = (row['history_drift_exit']
                                      / row['envelope_n_eps'])
        row['exceeds_1e_15'] = row['history_drift_exit'] > 1e-15
        rows.append(row)
    out['d3_rows'] = rows
    out['d3_ratio_at_3_surfaces'] = next(
        (r['ratio_exit_to_n_eps'] for r in rows if r['n_surfaces'] == 3), None)
    out['d3_ratio_at_13_surfaces'] = next(
        (r['ratio_exit_to_n_eps'] for r in rows if r['n_surfaces'] == 13),
        None)
    out['d3_envelope_holds_on_every_rung'] = all(
        r['ratio_exit_to_n_eps'] <= 1.0 + 1e-12 for r in rows)
    first = [r['n_surfaces'] for r in rows if r['exceeds_1e_15']]
    out['d3_first_surface_count_exceeding_1e_15'] = min(first) if first else None
    out['d3_surface_drift_is_flat'] = (
        max(r['history_drift_surface'] for r in rows)
        <= 4.0 * EPS)


def gate(R, route, y=0.0):
    """Bisect the h/|R| at which this route starts returning NaN, on the
    meridian (y = 0)."""
    from lumenairy.raytrace.surface import _sphere_normal, _surface_normal

    surf = Surface(radius=R, thickness=0.0, glass_before='air',
                   glass_after='N-BK7', semi_diameter=abs(R))

    def alive(frac):
        x = np.array([abs(R) * frac])
        yy = np.array([y])
        if route == 'analytic':
            c = _sphere_normal(x, yy, R)
        else:
            c = _surface_normal(x, yy, surf)
        return bool(np.all(np.isfinite(np.asarray(c, dtype=float))))

    lo, hi = 0.5, 1.5
    assert alive(lo) and not alive(hi), (R, route)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid == lo or mid == hi:
            break
        if alive(mid):
            lo = mid
        else:
            hi = mid
    return lo


def d9(out):
    radii = [0.0020, -0.0020, 0.0125, -0.0125, 0.0515, -0.1200, 0.5, -1.0]
    gates = {}
    same = True
    for R in radii:
        a = gate(R, 'analytic')
        g = gate(R, 'generic')
        gates['%g' % R] = {'analytic': repr(a), 'generic': repr(g),
                           'identical': a == g}
        same = same and (a == g)
    out['d9_gates'] = gates
    out['d9_all_gates_identical_on_the_meridian'] = same
    out['d9_gate_values'] = sorted({v['analytic'] for v in gates.values()})

    # ---- the ball lens / hemisphere
    R = 0.0125
    n = 60000
    rng = np.random.default_rng(31415)
    # rim-packed: the outer 0.1 % of the aperture
    frac = 0.9990 + 0.001 * rng.uniform(0.0, 1.0, n)
    th = rng.uniform(0.0, 2 * np.pi, n)
    r = R * frac
    ball = [Surface(radius=R, thickness=2 * R, glass_before='air',
                    glass_after='N-BK7', semi_diameter=R),
            Surface(radius=-R, thickness=0.010, glass_before='N-BK7',
                    glass_after='air', semi_diameter=R)]
    hemi = [Surface(radius=R, thickness=R, glass_before='air',
                    glass_after='N-BK7', semi_diameter=R),
            Surface(radius=np.inf, thickness=0.010, glass_before='N-BK7',
                    glass_after='air', semi_diameter=R)]
    past = int(np.sum(frac ** 2 >= 0.9999))
    out['d9_n_rays'] = n
    out['d9_n_past_the_clamp'] = past
    res = {}
    for name, surfs in (('ball', ball), ('hemisphere', hemi)):
        for rn in ('surface', 'exit'):
            for sn in ('generic', 'analytic'):
                rb = RayBundle(x=r * np.cos(th), y=r * np.sin(th),
                               z=np.zeros(n), L=np.zeros(n), M=np.zeros(n),
                               N=np.ones(n), wavelength=WL,
                               alive=np.ones(n, dtype=bool),
                               opd=np.zeros(n))
                o = trace(rb, surfs, WL, output_filter='last',
                          renormalize=rn, sphere_normal=sn).image_rays
                code = np.asarray(o.error_code)
                res['%s/%s/%s' % (name, rn, sn)] = {
                    'n_ray_nan': int(np.sum(code == RAY_NAN)),
                    'n_alive': int(np.sum(np.asarray(o.alive, dtype=bool))),
                    'fraction_ray_nan': float(np.mean(code == RAY_NAN)),
                }
    out['d9_clamp'] = res
    vals = {v['n_ray_nan'] for v in res.values()}
    out['d9_ray_nan_identical_across_all_eight'] = len(vals) == 1
    out['d9_n_ray_nan'] = sorted(vals)


def main(out_path):
    out = {'lumenairy_file': la.__file__,
           'python': sys.version.split()[0], 'numpy': np.__version__,
           'eps': EPS}
    d3(out)
    d9(out)
    with open(out_path, 'w') as fh:
        json.dump(out, fh, indent=1)
    for k, v in out.items():
        if k == 'd3_rows':
            print('d3_rows:')
            for r in v:
                print('   n=%2d  exit=%.4e  ratio=%.4f  surface=%.4e  '
                      '>1e-15=%s' % (r['n_surfaces'], r['history_drift_exit'],
                                     r['ratio_exit_to_n_eps'],
                                     r['history_drift_surface'],
                                     r['exceeds_1e_15']))
        elif isinstance(v, dict):
            print(k + ':')
            for kk, vv in v.items():
                print('   %-28s %s' % (kk, vv))
        else:
            print('%-44s %s' % (k, v))
    print('wrote', out_path)


if __name__ == '__main__':
    main(sys.argv[1])
