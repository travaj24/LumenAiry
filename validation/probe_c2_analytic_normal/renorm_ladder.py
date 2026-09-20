"""WP-C2 item 3 -- the `renormalize='exit'` ladder.

``'surface'`` rescales the direction cosines to unit length after every
refraction and reflection; ``'exit'`` rescales once, on the bundle that
leaves the last surface.  Exact vector Snell with a UNIT normal returns
a unit vector identically, so each per-surface rescale only removes the
rounding that surface contributed -- but that surviving drift enters the
NEXT surface's ray-sphere quadratic, which assumes ``a = |d|^2 = 1``.

So the quantity to measure is not "is it small" but "does it ACCUMULATE
with surface count, and does it stay inside the envelope the mechanism
predicts".  The envelope is derivable: the drift after k surfaces is
O(k eps), the quadratic's root error is ``|t| * k eps / 2``, so the
positional envelope over an N-surface stack is ``N eps |t|``.

This probe builds stacks of 2, 4, 6, 8, 10 and 12 spherical surfaces
from one repeated cemented pair, traces each both ways, and reports:

* ``max | |d| - 1 |`` on every INTERMEDIATE history bundle under
  ``'exit'`` (the documented <= 1e-15 contract a history consumer
  relies on);
* the position / OPL / direction difference between the two modes, and
  its ratio to the derived ``N eps |t|`` envelope;
* the same on a conic stack (no closed-form normal involved), so the
  two switches' effects are not confused;
* the count of ``_normalize_directions`` calls, which must be exactly
  one under ``'exit'`` and zero under ``'surface'``.

Usage:  LUMENAIRY_ROOT=<root> python renorm_ladder.py <out.json>
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

from lumenairy.raytrace.core import Surface  # noqa: E402
from lumenairy.raytrace.trace import _make_bundle, trace  # noqa: E402

WL = 587.6e-9
EPS = float(np.finfo(np.float64).eps)


def _bundle(n, semi=0.010, seed=20260920):
    rng = np.random.default_rng(seed)
    r = semi * np.sqrt(rng.random(n))
    th = 2 * np.pi * rng.random(n)
    z = np.zeros(n)
    return _make_bundle(r * np.cos(th), r * np.sin(th), z, z.copy(), WL)


def _sphere_pair():
    return [
        Surface(radius=0.0515, thickness=0.004, glass_before='air',
                glass_after='N-BK7', semi_diameter=0.0127),
        Surface(radius=-0.0515, thickness=0.006, glass_before='N-BK7',
                glass_after='air', semi_diameter=0.0127),
    ]


def _conic_pair():
    return [
        Surface(radius=0.0515, conic=-0.6, thickness=0.004,
                glass_before='air', glass_after='N-BK7',
                semi_diameter=0.0127),
        Surface(radius=-0.0515, conic=-0.9, thickness=0.006,
                glass_before='N-BK7', glass_after='air',
                semi_diameter=0.0127),
    ]


def _stack(pair, n_pairs, tail=0.060):
    S = []
    for _ in range(n_pairs):
        for s in pair():
            S.append(Surface(**{f: getattr(s, f) for f in
                                ('radius', 'conic', 'thickness',
                                 'glass_before', 'glass_after',
                                 'semi_diameter')}))
    S[-1] = Surface(radius=S[-1].radius, conic=S[-1].conic,
                    thickness=tail, glass_before=S[-1].glass_before,
                    glass_after=S[-1].glass_after,
                    semi_diameter=S[-1].semi_diameter)
    S.append(Surface(radius=np.inf, thickness=0.0, glass_before='air',
                     glass_after='air', semi_diameter=0.030))
    return S


def _run(pair, n_pairs, sphere_normal):
    S = _stack(pair, n_pairs)
    rays = _bundle(4000)
    surf = trace(rays, S, WL, output_filter='all',
                 renormalize='surface', sphere_normal=sphere_normal)
    exit_ = trace(rays, S, WL, output_filter='all',
                  renormalize='exit', sphere_normal=sphere_normal)
    a, b = surf.image_rays, exit_.image_rays
    m = np.asarray(a.alive) & np.asarray(b.alive)
    worst = {}
    for f in ('x', 'y', 'z', 'opd', 'L', 'M', 'N'):
        worst[f] = float(np.max(np.abs(np.asarray(getattr(a, f))[m]
                                       - np.asarray(getattr(b, f))[m]))) \
            if m.any() else 0.0
    # the intermediate-history contract under 'exit'
    hist_drift = 0.0
    for i in range(len(S) - 1):
        r = exit_.rays_at(i)
        mm = np.asarray(r.alive)
        if not mm.any():
            continue
        d = np.sqrt(np.asarray(r.L)[mm] ** 2 + np.asarray(r.M)[mm] ** 2
                    + np.asarray(r.N)[mm] ** 2)
        hist_drift = max(hist_drift, float(np.max(np.abs(d - 1.0))))
    final = exit_.image_rays
    mm = np.asarray(final.alive)
    d = np.sqrt(np.asarray(final.L)[mm] ** 2
                + np.asarray(final.M)[mm] ** 2
                + np.asarray(final.N)[mm] ** 2)
    final_drift = float(np.max(np.abs(d - 1.0))) if mm.any() else 0.0
    t_scale = float(np.max(np.abs(np.asarray(a.z)[m]))) if m.any() else 0.0
    t_scale = max(t_scale, 0.11)      # the stack's own axial extent
    envelope = len(S) * EPS * t_scale
    return dict(
        n_surfaces=len(S), n_alive=int(m.sum()),
        worst=worst,
        worst_position=max(worst['x'], worst['y'], worst['z']),
        history_drift=hist_drift, final_drift=final_drift,
        envelope=envelope,
        position_over_envelope=(max(worst['x'], worst['y'], worst['z'])
                                / envelope),
        alive_equal=bool(np.array_equal(np.asarray(a.alive),
                                        np.asarray(b.alive))),
        code_equal=bool(np.array_equal(np.asarray(a.error_code),
                                       np.asarray(b.error_code))),
    )


def _call_counts():
    """``trace`` imports the helper BY NAME, so the count has to be
    taken in ``trace``'s own namespace -- patching
    ``intersection._normalize_directions`` sees nothing."""
    import importlib
    import unittest.mock as _mock

    _trace_mod = importlib.import_module('lumenairy.raytrace.trace')
    S = _stack(_sphere_pair, 3)
    out = {}
    for mode in ('surface', 'exit'):
        calls = []
        real = _trace_mod._normalize_directions

        def counting(r, _calls=calls, _real=real):
            _calls.append(1)
            return _real(r)

        with _mock.patch.object(_trace_mod, '_normalize_directions',
                                counting):
            trace(_bundle(64), S, WL, output_filter='last',
                  renormalize=mode)
        out[mode] = len(calls)
    return out


def main():
    out_path = sys.argv[1]
    results = {}
    for tag, pair in (('spherical', _sphere_pair),
                      ('conic', _conic_pair)):
        for sph in ('generic', 'analytic'):
            rows = []
            for n_pairs in (1, 2, 3, 4, 5, 6):
                rows.append(_run(pair, n_pairs, sph))
                r = rows[-1]
                print('%-9s/%-8s  %2d surf  dpos %.3e  dopd %.3e  '
                      'hist |d|-1 %.3e  final %.3e  pos/env %.3f  '
                      'alive eq %s'
                      % (tag, sph, r['n_surfaces'],
                         r['worst_position'], r['worst']['opd'],
                         r['history_drift'], r['final_drift'],
                         r['position_over_envelope'], r['alive_equal']),
                      flush=True)
            results[f'{tag}/{sph}'] = rows
    counts = _call_counts()
    print('_normalize_directions calls:', counts)
    summary = dict(
        normalize_calls=counts,
        history_drift_max=max(r['history_drift'] for v in results.values()
                              for r in v),
        final_drift_max=max(r['final_drift'] for v in results.values()
                            for r in v),
        position_over_envelope_max=max(r['position_over_envelope']
                                       for v in results.values()
                                       for r in v),
        alive_equal_everywhere=all(r['alive_equal']
                                   for v in results.values() for r in v),
        code_equal_everywhere=all(r['code_equal']
                                  for v in results.values() for r in v),
        worst_position=max(r['worst_position'] for v in results.values()
                           for r in v),
        worst_opd=max(r['worst']['opd'] for v in results.values()
                      for r in v),
        worst_direction=max(max(r['worst']['L'], r['worst']['M'],
                                r['worst']['N'])
                            for v in results.values() for r in v),
    )
    for k, v in summary.items():
        print(k, v)
    meta = dict(python=sys.version, numpy=np.__version__,
                lumenairy=la.__version__, platform=sys.platform, eps=EPS)
    with open(out_path, 'w') as fh:
        json.dump(dict(meta=meta, summary=summary, results=results),
                  fh, indent=1)
    print('wrote', out_path)


if __name__ == '__main__':
    main()
