"""VERIFY-WP-C2 item 4 -- the rim band, on VERIFY-C2's own sweep, and the
question the C2 report left open: can a REAL design land in it?

Three parts:

1. **Both gates LOCATED by bisection**, per radius, on the running build --
   the analytic route's ``(x*x + y*y)/(R*R) < 0.9999`` against the generic
   route's ``(1 + k) * sqrt(x*x + y*y)**2 / R**2 < 0.9999`` -- so the band's
   width is a measured number in metres, not a claim.
2. **A directed ``nextafter`` walk** that constructs a straddle point and
   pushes it through ``_refract`` AND through the public ``trace``.
3. **Sweeps**, including two the C2 report does not have: an OVER-FILLED
   aperture (clear semi-diameter set to ``|R|``, rays launched out to
   ``0.99999 |R|``), and the two real designs whose clear aperture actually
   REACHES the rim -- a BALL lens and a HEMISPHERE (half-ball), both
   catalogue parts, whose semi-diameter is ``|R|`` by construction.  A fast
   singlet is included for contrast: its marginal ray's ``h/|R|`` is a
   measured number, and the report's "can a real design land in the band"
   is answered from it.

Usage: ``LUMENAIRY_ROOT=<root> python vc2_vignetting.py OUT.json``
"""
import json
import math
import os
import sys

_ROOT = os.environ['LUMENAIRY_ROOT']
sys.path.insert(0, _ROOT)

import numpy as np                                            # noqa: E402
import lumenairy as la                                        # noqa: E402

_want = os.path.realpath(os.path.join(_ROOT, 'lumenairy'))
assert os.path.realpath(os.path.dirname(la.__file__)) == _want, la.__file__

from lumenairy.raytrace.surface import RayBundle, Surface     # noqa: E402
from lumenairy.raytrace.intersection import _refract          # noqa: E402
from lumenairy.raytrace.trace import trace                    # noqa: E402

WL = 587.5618e-9
CLAMP = 0.9999


def _surf(R, th=0.01, gb='air', ga='N-BK7', sd=np.inf):
    return Surface(radius=R, conic=0.0, thickness=th, glass_before=gb,
                   glass_after=ga, semi_diameter=sd)


def _nz(x, y, R, route):
    """``nz`` from one route at a single point (NaN outside its gate)."""
    s = _surf(R)
    from lumenairy.raytrace.surface import _surface_normal
    a = np.array([float(x)])
    b = np.array([float(y)])
    _nx, _ny, nzv = _surface_normal(a, b, s,
                                   analytic_sphere=(route == 'analytic'))
    return float(nzv[0])


def _threshold(R, route, lo=0.99, hi=1.0, steps=90):
    """Bisect for the largest ``h/|R|`` this route still accepts."""
    aR = abs(R)
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        if math.isfinite(_nz(mid * aR, 0.0, R, route)):
            lo = mid
        else:
            hi = mid
    return lo


def _bundle(x, y, n=None):
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(x)
    return RayBundle(x=x.copy(), y=y.copy(), z=np.zeros(n),
                     L=np.zeros(n), M=np.zeros(n), N=np.ones(n),
                     wavelength=WL, alive=np.ones(n, dtype=bool),
                     opd=np.zeros(n))


def _walk(R, n_az=8, n_steps=400):
    """Directed ``nextafter`` walk for a point the two gates straddle."""
    aR = abs(R)
    h0 = math.sqrt(CLAMP) * aR
    for k in range(n_az):
        th = 2.0 * math.pi * k / n_az + 0.11
        for sgn in (+1.0, -1.0):
            h = h0
            for _ in range(n_steps):
                x = h * math.cos(th)
                y = h * math.sin(th)
                fa = math.isfinite(_nz(x, y, R, 'analytic'))
                fg = math.isfinite(_nz(x, y, R, 'generic'))
                if fa != fg:
                    return dict(R=R, x=x, y=y, h_over_R=h / aR,
                                analytic_valid=fa, generic_valid=fg)
                h = math.nextafter(h, h + sgn)
    return None


def _through_refract(pt):
    """Does the straddle reach ``_refract``'s alive flag / error code?"""
    out = {}
    for route in ('generic', 'analytic'):
        rb = _bundle([pt['x']], [pt['y']])
        _refract(rb, _surf(pt['R']), 1.0, 1.5168, renormalize=True,
                 sphere_normal=route)
        out[route] = dict(alive=bool(rb.alive[0]),
                          code=int(rb.error_code[0]),
                          L=float(rb.L[0]), N=float(rb.N[0]))
    return out


def _through_trace(pt):
    surfs = [_surf(pt['R'], th=0.010, sd=np.inf),
             Surface(radius=np.inf, thickness=0.020, glass_before='N-BK7',
                     glass_after='air', semi_diameter=np.inf)]
    out = {}
    for route in ('generic', 'analytic'):
        rb = _bundle([pt['x']], [pt['y']])
        res = trace(rb, surfs, WL, output_filter='last',
                    sphere_normal=route, renormalize='exit')
        out[route] = dict(alive=bool(res.image_rays.alive[0]),
                          code=int(res.image_rays.error_code[0]))
    return out


# ------------------------------------------------------------- the sweeps

def _sweeps():
    """(name, surfaces, bundle-maker) for every combination swept."""
    from lumenairy.io.prescriptions_builders import (make_singlet,
                                                     make_doublet)
    from lumenairy.raytrace import surfaces_from_prescription as _sfp
    out = []

    def fan(hmax, n, tilt=0.0):
        def mk():
            rng = np.random.default_rng(90210)
            r = hmax * np.sqrt(rng.uniform(0.0, 1.0, n))
            th = rng.uniform(0, 2 * np.pi, n)
            x, y = r * np.cos(th), r * np.sin(th)
            L = np.full(n, math.sin(math.radians(tilt)))
            N = np.full(n, math.cos(math.radians(tilt)))
            return RayBundle(x=x, y=y, z=np.zeros(n), L=L, M=np.zeros(n),
                             N=N, wavelength=WL,
                             alive=np.ones(n, dtype=bool), opd=np.zeros(n))
        return mk

    def rim_fan(hmax, n, lo=0.9990):
        """Rays packed into the last 0.1 % of the aperture."""
        def mk():
            rng = np.random.default_rng(31337)
            r = hmax * rng.uniform(lo, 1.0, n)
            th = rng.uniform(0, 2 * np.pi, n)
            x, y = r * np.cos(th), r * np.sin(th)
            return RayBundle(x=x, y=y, z=np.zeros(n), L=np.zeros(n),
                             M=np.zeros(n), N=np.ones(n), wavelength=WL,
                             alive=np.ones(n, dtype=bool), opd=np.zeros(n))
        return mk

    R = 0.0125
    ball = [_surf(R, th=2 * R, ga='N-BK7', sd=R),
            _surf(-R, th=0.020, gb='N-BK7', ga='air', sd=R)]
    hemi = [_surf(R, th=R, ga='N-BK7', sd=R),
            Surface(radius=np.inf, thickness=0.020, glass_before='N-BK7',
                    glass_after='air', semi_diameter=R)]
    over = [_surf(0.0125, th=0.010, sd=0.0125 * 0.99999),
            Surface(radius=np.inf, thickness=0.020, glass_before='N-BK7',
                    glass_after='air', semi_diameter=np.inf)]
    out += [
        ('ball_lens_rimpacked', ball, rim_fan(R * 0.999999, 60000)),
        ('ball_lens_fullfan', ball, fan(R * 0.99999, 60000)),
        ('hemisphere_rimpacked', hemi, rim_fan(R * 0.999999, 60000)),
        ('overfilled_sphere', over, rim_fan(0.0125 * 0.999999, 60000)),
        ('overfilled_sphere_tilt5', over, fan(0.0125 * 0.99999, 60000,
                                              tilt=5.0)),
    ]
    s1 = _sfp(make_singlet(0.020, -0.020, 0.006, 'N-BK7', aperture=0.020))
    s2 = _sfp(make_singlet(0.0515, np.inf, 0.0035, 'N-BK7',
                           aperture=0.025))
    d1 = _sfp(make_doublet(0.0517, -0.0345, -0.120, 0.009, 0.0025,
                           'N-BK7', 'N-SF5', aperture=0.025))
    out += [
        ('make_singlet_20_-20_f0', s1, fan(0.0099, 40000)),
        ('make_singlet_20_-20_f8', s1, fan(0.0099, 40000, tilt=8.0)),
        ('make_singlet_515_inf_f0', s2, fan(0.0124, 40000)),
        ('make_doublet_f0', d1, fan(0.0124, 40000)),
        ('make_doublet_f5', d1, fan(0.0124, 40000, tilt=5.0)),
        ('make_singlet_20_-20_rim', s1, rim_fan(0.0099, 40000)),
        ('make_doublet_rim', d1, rim_fan(0.0124, 40000)),
    ]
    return out


def main(out_path):
    res = {'meta': dict(python=sys.version.split()[0], numpy=np.__version__,
                        lumenairy=la.__version__, file=la.__file__)}

    # --- 1. the two gates, located ---------------------------------
    gates = {}
    for R in (0.0020, -0.0020, 0.0125, -0.0125, 0.0515, -0.1200, 0.5, -1.0):
        ta = _threshold(R, 'analytic')
        tg = _threshold(R, 'generic')
        aR = abs(R)
        gates[f'{R:g}'] = dict(
            analytic_h_over_R=ta, generic_h_over_R=tg,
            delta_h_over_R=ta - tg,
            delta_h_m=(ta - tg) * aR,
            delta_h_ulp=((ta - tg) * aR / math.ulp(ta * aR)
                         if ta != tg else 0.0),
            band_width_m=abs(ta - tg) * aR,
            sqrt_clamp=math.sqrt(CLAMP))
    res['gates'] = gates

    # --- 2. the straddle ------------------------------------------
    straddles = []
    for R in (0.0020, -0.0125, 0.0515, -0.1200, 0.5, -1.0):
        pt = _walk(R)
        if pt is not None:
            pt['through_refract'] = _through_refract(pt)
            pt['through_trace'] = _through_trace(pt)
            straddles.append(pt)
    res['straddles'] = straddles
    res['n_radii_with_a_straddle'] = len(straddles)

    # --- 3. the sweeps --------------------------------------------
    sweeps = {}
    total_rays = 0
    total_moved_alive = 0
    total_moved_code = 0
    for name, surfs, mk in _sweeps():
        rb = mk()
        n = rb.n_rays
        total_rays += n
        a = trace(rb, surfs, WL, output_filter='last',
                  sphere_normal='analytic', renormalize='exit').image_rays
        g = trace(rb, surfs, WL, output_filter='last',
                  sphere_normal='generic', renormalize='exit').image_rays
        # ALSO the pre-5.49.0 pair, so the whole default move is covered
        g0 = trace(rb, surfs, WL, output_filter='last',
                   sphere_normal='generic', renormalize='surface').image_rays
        moved_alive = int(np.sum(a.alive != g.alive))
        moved_code = int(np.sum(a.error_code != g.error_code))
        moved_alive_full = int(np.sum(a.alive != g0.alive))
        moved_code_full = int(np.sum(a.error_code != g0.error_code))
        total_moved_alive += moved_alive
        total_moved_code += moved_code
        sweeps[name] = dict(
            n_rays=n, n_surfaces=len(surfs),
            n_alive_analytic=int(np.sum(a.alive)),
            n_alive_generic=int(np.sum(g.alive)),
            moved_alive=moved_alive, moved_code=moved_code,
            moved_alive_vs_pre549=moved_alive_full,
            moved_code_vs_pre549=moved_code_full,
            codes_analytic={int(k): int(v) for k, v in
                            zip(*np.unique(a.error_code,
                                           return_counts=True))},
            codes_generic={int(k): int(v) for k, v in
                           zip(*np.unique(g.error_code,
                                          return_counts=True))})
    res['sweeps'] = sweeps
    res['sweep_totals'] = dict(n_rays=total_rays,
                               moved_alive=total_moved_alive,
                               moved_code=total_moved_code)

    # --- 4. how close does a REAL design get to the clamp? ---------
    reach = {}
    for name, surfs, mk in _sweeps():
        rb = mk()
        res_t = trace(rb, surfs, WL, output_filter='all',
                      sphere_normal='analytic', renormalize='exit')
        worst = 0.0
        worst_surf = -1
        n_in_last_ulp = 0
        for si, (s, b) in enumerate(zip(surfs, res_t.ray_history)):
            if not np.isfinite(s.radius):
                continue
            h = np.sqrt(b.x ** 2 + b.y ** 2) / abs(s.radius)
            hm = float(np.nanmax(h))
            if hm > worst:
                worst, worst_surf = hm, si
            thr = math.sqrt(CLAMP)
            band = abs(np.asarray(h) - thr) <= 4.0 * np.spacing(thr)
            n_in_last_ulp += int(np.sum(band))
        reach[name] = dict(max_h_over_R=worst, at_surface=worst_surf,
                           clamp_h_over_R=math.sqrt(CLAMP),
                           reaches_clamp=bool(worst >= math.sqrt(CLAMP)),
                           n_rays_within_4ulp_of_the_gate=n_in_last_ulp)
    res['rim_reach'] = reach

    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    print(json.dumps(dict(gates=gates, sweep_totals=res['sweep_totals'],
                          n_straddles=len(straddles)),
                     indent=1, sort_keys=True))
    for k, v in sweeps.items():
        print(f"{k:28s} n={v['n_rays']:6d} moved_alive={v['moved_alive']} "
              f"moved_code={v['moved_code']} "
              f"vs_pre549={v['moved_alive_vs_pre549']}/"
              f"{v['moved_code_vs_pre549']}")
    for k, v in reach.items():
        print(f"{k:28s} max h/|R| = {v['max_h_over_R']:.10f} "
              f"reaches_clamp={v['reaches_clamp']} "
              f"in4ulp={v['n_rays_within_4ulp_of_the_gate']}")
    if straddles:
        print('straddle 0:', json.dumps(straddles[0], sort_keys=True))


if __name__ == '__main__':
    main(sys.argv[1])
