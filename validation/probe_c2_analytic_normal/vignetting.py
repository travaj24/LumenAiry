"""WP-C2 item 2 -- the vignetting sweep VERIFY-WP-B9 section 3.3 asked for.

The two normal routes gate the domain from DIFFERENT expressions --
``(x*x + y*y)/(R*R) < 0.9999`` in the closed form against
``(1 + conic) * sqrt(x*x + y*y)**2 / R**2 < 0.9999`` in the generic one
-- which differ by up to 1 ULP, so within about 1 ULP of
``h**2 = 0.9999 R**2`` the two land on opposite sides of the threshold
and one route kills a ray the other refracts.  This probe measures:

1. **Where the gates straddle.**  A directed ``nextafter`` walk around
   ``h = |R| sqrt(0.9999)`` on a set of radii and azimuths, counting
   points where the two predicates disagree and confirming end to end
   through ``_refract`` that the disagreement reaches ``alive`` /
   ``error_code``.
2. **Whether real bundles reach that band.**  Dense uniform bundles on
   several prescriptions -- including ones whose clear aperture is
   deliberately opened to the full hemisphere, so the band is inside the
   traced set -- traced both ways, counting every ray whose ``alive`` or
   ``error_code`` moves.
3. **Whether any SHIPPED fixture's vignetting count changes.**  The
   package's own prescription builders at their own apertures, traced
   both ways, comparing the per-code census.

Usage:  LUMENAIRY_ROOT=<root> python vignetting.py <out.json>
"""
import json
import math
import os
import sys

import numpy as np

_ROOT = os.environ['LUMENAIRY_ROOT']
sys.path.insert(0, _ROOT)

import lumenairy as la  # noqa: E402

_want = os.path.realpath(os.path.join(_ROOT, 'lumenairy'))
assert os.path.realpath(os.path.dirname(la.__file__)) == _want, la.__file__

from lumenairy.raytrace import (
    intersection as _isect,  # noqa: E402
    )
from lumenairy.raytrace.core import Surface  # noqa: E402
from lumenairy.raytrace.trace import _make_bundle, trace  # noqa: E402

WL = 587.6e-9
RADII = [0.002, -0.002, 0.0515, -0.0345, 0.080, -0.120, 0.5, -1.0]
AZIMUTHS = [0.0, 0.3, math.pi / 4, 1.1, 2.7, 4.9]


def _gate_fast(x, y, R):
    return float((x * x + y * y) / (R * R)) < 0.9999


def _gate_slow(x, y, R, conic=0.0):
    h = math.sqrt(x * x + y * y)
    return float((1.0 + conic) * h ** 2 / (R * R)) < 0.9999


def _straddle_walk(steps=60):
    """Directed nextafter walk around h = |R| sqrt(0.9999)."""
    found = []
    for R in RADII:
        h0 = abs(R) * math.sqrt(0.9999)
        for az in AZIMUTHS:
            h = h0
            for _ in range(steps // 2):
                h = math.nextafter(h, 0.0)
            for _ in range(steps):
                x = h * math.cos(az)
                y = h * math.sin(az)
                f = _gate_fast(x, y, R)
                s = _gate_slow(x, y, R)
                if f != s:
                    found.append(dict(R=R, az=az, x=x, y=y,
                                      fast_valid=f, slow_valid=s,
                                      h_over_R=h / abs(R)))
                h = math.nextafter(h, math.inf)
    return found


def _endtoend(pt):
    """Does the gate disagreement reach alive / error_code?"""
    R = pt['R']
    surf = Surface(radius=R, thickness=0.0, glass_before='air',
                   glass_after='N-BK7', semi_diameter=abs(R) * 2)
    out = {}
    for name, kw in (('generic', 'generic'), ('analytic', 'analytic')):
        rays = _make_bundle(np.array([pt['x']]), np.array([pt['y']]),
                            np.array([0.0]), np.array([0.0]), WL)
        rays.z = np.array([0.0])
        _isect._refract(rays, surf, 1.0, 1.5168, sphere_normal=kw)
        out[name] = dict(alive=bool(rays.alive[0]),
                         code=int(rays.error_code[0]),
                         L=float(rays.L[0]), N=float(rays.N[0]))
    out['moves'] = (out['generic']['alive'] != out['analytic']['alive']
                    or out['generic']['code'] != out['analytic']['code'])
    return out


def _hemisphere_stack(R=0.0515):
    """ONE sphere with the aperture opened past 0.9999 |R|, so the band
    the two gates straddle is genuinely inside the traced set."""
    return [
        Surface(radius=R, thickness=0.050, glass_before='air',
                glass_after='N-BK7', semi_diameter=abs(R) * 0.99999),
        Surface(radius=np.inf, thickness=0.0, glass_before='N-BK7',
                glass_after='air', semi_diameter=abs(R) * 2),
    ]


def _spherical7():
    return [
        Surface(radius=0.0515, thickness=0.008, glass_before='air',
                glass_after='N-BK7', semi_diameter=0.0127),
        Surface(radius=-0.0345, thickness=0.003, glass_before='N-BK7',
                glass_after='N-SF5', semi_diameter=0.0127),
        Surface(radius=-0.120, thickness=0.010, glass_before='N-SF5',
                glass_after='air', semi_diameter=0.0127),
        Surface(radius=0.080, thickness=0.006, glass_before='air',
                glass_after='N-BK7', semi_diameter=0.0127),
        Surface(radius=-0.080, thickness=0.090, glass_before='N-BK7',
                glass_after='air', semi_diameter=0.0127),
        Surface(radius=np.inf, thickness=0.010, glass_before='air',
                glass_after='air', semi_diameter=0.02),
        Surface(radius=np.inf, thickness=0.0, glass_before='air',
                glass_after='air', semi_diameter=0.02),
    ]


def _cassegrain():
    return [
        Surface(radius=-0.400, thickness=-0.150, glass_before='air',
                glass_after='air', is_mirror=True, semi_diameter=0.050),
        Surface(radius=-0.120, thickness=0.250, glass_before='air',
                glass_after='air', is_mirror=True, semi_diameter=0.015),
        Surface(radius=np.inf, thickness=0.0, glass_before='air',
                glass_after='air', semi_diameter=0.030),
    ]


def _census(res):
    codes = np.asarray(res.image_rays.error_code)
    alive = np.asarray(res.image_rays.alive)
    return dict(n=int(alive.size), n_alive=int(alive.sum()),
                codes={int(c): int((codes == c).sum())
                       for c in np.unique(codes)})


def _bundle_sweep(label, S, semi, n=40000, tilt=0.0):
    rng = np.random.default_rng(20260920)
    r = semi * np.sqrt(rng.random(n))
    th = 2 * np.pi * rng.random(n)
    x = r * np.cos(th)
    y = r * np.sin(th)
    L = np.full(n, math.sin(math.radians(tilt)))
    M = np.zeros(n)
    rays = _make_bundle(x, y, L, M, WL)
    g = trace(rays, S, WL, output_filter='last', sphere_normal='generic')
    a = trace(rays, S, WL, output_filter='last', sphere_normal='analytic')
    ga, aa = np.asarray(g.image_rays.alive), np.asarray(a.image_rays.alive)
    gc = np.asarray(g.image_rays.error_code)
    ac = np.asarray(a.image_rays.error_code)
    return dict(label=label, n=n, tilt_deg=tilt, semi=semi,
                generic=_census(g), analytic=_census(a),
                n_alive_moved=int((ga != aa).sum()),
                n_code_moved=int((gc != ac).sum()),
                alive_count_delta=int(aa.sum()) - int(ga.sum()))


def _straddle_through_trace(pt):
    """The same point pushed through the PUBLIC ``trace``, so the claim
    is about the ray tracer and not about a private helper."""
    R = pt['R']
    # The downstream apertures are opened wide on purpose: a grazing
    # refracted ray leaves nearly sideways, and a tight second aperture
    # would kill it for a DIFFERENT reason and hide the alive flag the
    # rim band actually moves.
    S = [Surface(radius=R, thickness=0.001, glass_before='air',
                 glass_after='N-BK7', semi_diameter=abs(R) * 2),
         Surface(radius=np.inf, thickness=0.0, glass_before='N-BK7',
                 glass_after='air', semi_diameter=1.0)]
    out = {}
    for kw in ('generic', 'analytic'):
        rays = _make_bundle(np.array([pt['x']]), np.array([pt['y']]),
                            np.array([0.0]), np.array([0.0]), WL)
        res = trace(rays, S, WL, output_filter='last',
                    sphere_normal=kw).image_rays
        out[kw] = dict(alive=bool(res.alive[0]),
                       code=int(res.error_code[0]))
    out['moves'] = out['generic'] != out['analytic']
    return out


def main():
    out_path = sys.argv[1]
    straddle = _straddle_walk()
    reach = [_endtoend(p) for p in straddle[:12]]
    reach_trace = [_straddle_through_trace(p) for p in straddle[:12]]
    sweeps = []
    for label, S, semi, tilts in (
            ('hemisphere-aperture sphere', _hemisphere_stack(), 0.0515, (0.0,)),
            ('spherical7', _spherical7(), 0.0127, (0.0, 3.0, 8.0)),
            ('cassegrain (2 spherical mirrors)', _cassegrain(), 0.050,
             (0.0, 2.0)),
    ):
        for t in tilts:
            sweeps.append(_bundle_sweep(label, S, semi, tilt=t))
    # shipped fixtures, at their own apertures
    shipped = []
    for name, pres in (
            ('make_singlet R=20/-20 d=2 N-BK7',
             la.make_singlet(R1=20e-3, R2=-20e-3, d=2e-3, glass='N-BK7',
                             aperture=10e-3)),
            ('make_singlet R=51.5/inf d=4.1 N-BK7',
             la.make_singlet(51.5e-3, np.inf, 4.1e-3, 'N-BK7',
                             aperture=12.0e-3)),
            ('make_doublet 51.7/-34.5/-120',
             la.make_doublet(R1=51.7e-3, R2=-34.5e-3, R3=-120.0e-3,
                             d1=8e-3, d2=3e-3, glass1='N-BK7',
                             glass2='N-SF5', aperture=25.4e-3)),
    ):
        S = la.surfaces_from_prescription(pres)
        semi = max(float(getattr(s, 'semi_diameter', 0.0) or 0.0)
                   for s in S)
        for t in (0.0, 5.0):
            row = _bundle_sweep(name, S, semi * 0.999, n=20000, tilt=t)
            row['shipped'] = True
            shipped.append(row)
    summary = dict(
        n_straddle_points=len(straddle),
        n_straddle_reaching_refract=int(sum(1 for r in reach
                                            if r['moves'])),
        n_straddle_reaching_trace=int(sum(1 for r in reach_trace
                                          if r['moves'])),
        straddle_trace_example=(reach_trace[0] if reach_trace else None),
        straddle_example=(dict(straddle[0], **{'endtoend': reach[0]})
                          if straddle else None),
        sweeps_moved=[dict(label=s['label'], tilt=s['tilt_deg'],
                           alive_moved=s['n_alive_moved'],
                           code_moved=s['n_code_moved'])
                      for s in sweeps + shipped],
        total_alive_moved=int(sum(s['n_alive_moved']
                                  for s in sweeps + shipped)),
        total_code_moved=int(sum(s['n_code_moved']
                                 for s in sweeps + shipped)),
        total_rays=int(sum(s['n'] for s in sweeps + shipped)),
        shipped_fixture_counts_move=bool(
            any(s['n_alive_moved'] or s['n_code_moved'] for s in shipped)),
    )
    print('straddle points found:', summary['n_straddle_points'],
          '| reaching _refract:', summary['n_straddle_reaching_refract'],
          '| reaching trace():', summary['n_straddle_reaching_trace'])
    if reach_trace:
        print('   example:', summary['straddle_trace_example'])
    for s in sweeps + shipped:
        print('%-36s tilt %4.1f  alive_moved %d  code_moved %d  '
              'alive g/a %d/%d'
              % (s['label'], s['tilt_deg'], s['n_alive_moved'],
                 s['n_code_moved'], s['generic']['n_alive'],
                 s['analytic']['n_alive']))
    print('TOTAL rays', summary['total_rays'], 'alive moved',
          summary['total_alive_moved'], 'code moved',
          summary['total_code_moved'])
    print('shipped fixture vignetting counts move:',
          summary['shipped_fixture_counts_move'])
    meta = dict(python=sys.version, numpy=np.__version__,
                lumenairy=la.__version__, platform=sys.platform)
    with open(out_path, 'w') as fh:
        json.dump(dict(meta=meta, summary=summary, straddle=straddle,
                       endtoend=reach, endtoend_trace=reach_trace,
                       sweeps=sweeps, shipped=shipped),
                  fh, indent=1)
    print('wrote', out_path)


if __name__ == '__main__':
    main()
