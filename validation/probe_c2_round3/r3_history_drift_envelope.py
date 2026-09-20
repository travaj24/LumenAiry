"""WP-C2 ROUND 3, VR2-D2 -- is ``n_surfaces * eps`` a BOUND on the
history-bundle drift, or a reading of one ladder?

Under ``renormalize='exit'`` only the FINAL bundle is rescaled, so the
intermediate ``ray_history`` bundles carry ``| |d| - 1 |``.  The shipped
``trace`` docstring calls ``n_surfaces * eps`` "the BOUND" and steers a
consumer to size a tolerance from it.  VERIFY-WP-C2 round 2 measured an
ordinary 3-surface stack that exceeds it by 1.3333x.

This probe re-measures the claim over 90 combinations -- 5 surface counts
x 3 glasses x 2 radius pairs x 3 field angles -- and reports, per
combination, the worst drift, the ratio to ``n eps`` and the ratio to
``2 n eps``, plus the rung at which ``1e-15`` is first exceeded on each
(glass, radius pair, field angle) ladder.

Usage:  python r3_history_drift_envelope.py <out.json>
"""
import json
import pathlib
import sys

import numpy as np

WL = 587.6e-9
EPS = float(np.finfo(np.float64).eps)

#: (surface count) x (glass) x (radius pair) x (field angle, degrees)
COUNTS = (3, 5, 7, 9, 13)
GLASSES = ('N-BK7', 'N-SF5', 'N-SF11')
RADII = ((0.0515, -0.0515), (0.0623, -0.0771))
FIELDS = (0.0, 2.0, 5.0)


def bundle(n, semi, field_deg):
    from lumenairy.raytrace.trace import _make_bundle
    h = np.linspace(-semi, semi, n)
    z = np.zeros_like(h)
    t = np.deg2rad(field_deg)
    L = np.full_like(h, float(np.sin(t)))
    M = np.zeros_like(h)
    return _make_bundle(x=h, y=z, L=L, M=M, wavelength=WL)


def stack(n_surfaces, glass, radii):
    """``n_surfaces`` surfaces: alternating air->glass / glass->air pairs
    with a flat exit plane last, so the count is always odd."""
    from lumenairy.raytrace.core import Surface
    r1, r2 = radii
    S = []
    n_pairs = (n_surfaces - 1) // 2
    for k in range(n_pairs):
        # vary the third stack's radii so the ladder is not one element
        # repeated -- the shape VR2-D2 says the old reading came from
        f = 1.0 + 0.07 * k
        S.append(Surface(radius=r1 * f, thickness=0.004,
                         glass_before='air', glass_after=glass,
                         semi_diameter=0.0127))
        S.append(Surface(radius=r2 * f, thickness=0.006,
                         glass_before=glass, glass_after='air',
                         semi_diameter=0.0127))
    S.append(Surface(radius=np.inf, thickness=0.0, glass_before='air',
                     glass_after='air', semi_diameter=0.030))
    assert len(S) == n_surfaces, (len(S), n_surfaces)
    return S


def worst_history_drift(S, field_deg):
    from lumenairy.raytrace.trace import trace
    res = trace(bundle(2000, 0.010, field_deg), S, WL, output_filter='all')
    worst = 0.0
    for i in range(len(S) - 1):
        r = res.rays_at(i)
        m = np.asarray(r.alive)
        if not m.any():
            continue
        d = np.sqrt(np.asarray(r.L)[m] ** 2 + np.asarray(r.M)[m] ** 2
                    + np.asarray(r.N)[m] ** 2)
        worst = max(worst, float(np.max(np.abs(d - 1.0))))
    f = res.image_rays
    m = np.asarray(f.alive)
    d = np.sqrt(np.asarray(f.L)[m] ** 2 + np.asarray(f.M)[m] ** 2
                + np.asarray(f.N)[m] ** 2)
    return worst, float(np.max(np.abs(d - 1.0)))


def main(out_path):
    import lumenairy as la

    rows = []
    for glass in GLASSES:
        for ri, radii in enumerate(RADII):
            for field in FIELDS:
                for n in COUNTS:
                    S = stack(n, glass, radii)
                    worst, final = worst_history_drift(S, field)
                    rows.append({
                        'n_surfaces': n, 'glass': glass,
                        'radii': list(radii), 'radius_pair': ri,
                        'field_deg': field,
                        'worst_history_drift': worst,
                        'n_eps': n * EPS,
                        'ratio_to_n_eps': worst / (n * EPS),
                        'ratio_to_2n_eps': worst / (2 * n * EPS),
                        'final_bundle_drift': final,
                    })

    over_n = [r for r in rows if r['ratio_to_n_eps'] > 1.0]
    over_2n = [r for r in rows if r['ratio_to_2n_eps'] > 1.0]
    ladders = {}
    for glass in GLASSES:
        for ri in range(len(RADII)):
            for field in FIELDS:
                seq = sorted((r for r in rows
                              if r['glass'] == glass
                              and r['radius_pair'] == ri
                              and r['field_deg'] == field),
                             key=lambda r: r['n_surfaces'])
                first = next((r['n_surfaces'] for r in seq
                              if r['worst_history_drift'] > 1e-15), None)
                ladders[f'{glass}|pair{ri}|{field}deg'] = {
                    'drifts': [r['worst_history_drift'] for r in seq],
                    'ratios_to_n_eps': [r['ratio_to_n_eps'] for r in seq],
                    'first_surface_count_exceeding_1e_15': first,
                }

    firsts = sorted({v['first_surface_count_exceeding_1e_15']
                     for v in ladders.values()
                     if v['first_surface_count_exceeding_1e_15'] is not None})
    out = {
        'lumenairy_file': la.__file__,
        'python': sys.version.split()[0],
        'numpy': np.__version__,
        'eps': EPS,
        'n_combinations': len(rows),
        'n_exceeding_n_eps': len(over_n),
        'worst_ratio_to_n_eps': max(r['ratio_to_n_eps'] for r in rows),
        'worst_ratio_to_n_eps_at': max(
            rows, key=lambda r: r['ratio_to_n_eps']),
        'n_exceeding_2n_eps': len(over_2n),
        'worst_ratio_to_2n_eps': max(r['ratio_to_2n_eps'] for r in rows),
        'exceeding_n_eps': [
            {k: r[k] for k in ('n_surfaces', 'glass', 'radii', 'field_deg',
                               'worst_history_drift', 'n_eps',
                               'ratio_to_n_eps')}
            for r in sorted(over_n, key=lambda r: -r['ratio_to_n_eps'])],
        'first_1e_15_crossings': firsts,
        'max_final_bundle_drift': max(r['final_bundle_drift'] for r in rows),
        'ladders': ladders,
        'rows': rows,
    }
    pathlib.Path(out_path).write_text(
        json.dumps(out, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps({k: out[k] for k in (
        'lumenairy_file', 'python', 'numpy', 'n_combinations',
        'n_exceeding_n_eps', 'worst_ratio_to_n_eps', 'n_exceeding_2n_eps',
        'worst_ratio_to_2n_eps', 'first_1e_15_crossings',
        'max_final_bundle_drift')}, indent=2))


if __name__ == '__main__':
    main(sys.argv[1])
