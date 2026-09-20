"""WP-C2 items 2 and 3 -- whole-trace wall clock, both switches.

Reports a RANGE over several prescriptions and several interleaved
repeats, never a single number, and records the box's load alongside it
so the reading can be read as what it is.  Pairs are interleaved
(generic, analytic, generic, analytic, ...) inside one process so a
drifting clock or a thermal ramp hits both arms equally; each reading is
the MINIMUM of its repeats, which is the standard estimator for "how
fast can this be" on a machine that only ever adds noise.

Prescriptions: an all-spherical seven-surface stack, an all-spherical
Cooke-like triplet of its own numbers, a two-mirror spherical
Cassegrain, a conic stack (no pure sphere -- the control: the switch
must buy nothing there) and an aspheric singlet (the other control).

Usage:  LUMENAIRY_ROOT=<root> python timing.py <out.json> [--n 200000]
"""
import cProfile
import json
import os
import pstats
import sys
import time

import numpy as np

_ROOT = os.environ['LUMENAIRY_ROOT']
sys.path.insert(0, _ROOT)

import lumenairy as la  # noqa: E402

_want = os.path.realpath(os.path.join(_ROOT, 'lumenairy'))
assert os.path.realpath(os.path.dirname(la.__file__)) == _want, la.__file__

from lumenairy.raytrace.core import Surface  # noqa: E402
from lumenairy.raytrace.trace import _make_bundle, trace  # noqa: E402

WL = 587.6e-9


def _bundle(n, semi=0.0126, seed=20260920):
    rng = np.random.default_rng(seed)
    r = semi * np.sqrt(rng.random(n))
    th = 2 * np.pi * rng.random(n)
    z = np.zeros(n)
    return _make_bundle(r * np.cos(th), r * np.sin(th), z, z.copy(), WL)


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


def _triplet6():
    """A Cooke-like all-spherical triplet with a stop-side flat."""
    return [
        Surface(radius=0.0223, thickness=0.0043, glass_before='air',
                glass_after='N-SK16', semi_diameter=0.0090),
        Surface(radius=-0.4351, thickness=0.0060, glass_before='N-SK16',
                glass_after='air', semi_diameter=0.0090),
        Surface(radius=-0.0224, thickness=0.0010, glass_before='air',
                glass_after='N-SF11', semi_diameter=0.0060),
        Surface(radius=0.0204, thickness=0.0053, glass_before='N-SF11',
                glass_after='air', semi_diameter=0.0060),
        Surface(radius=0.0796, thickness=0.0038, glass_before='air',
                glass_after='N-SK16', semi_diameter=0.0085),
        Surface(radius=-0.0184, thickness=0.0420, glass_before='N-SK16',
                glass_after='air', semi_diameter=0.0085),
    ]


def _cassegrain():
    """Two SPHERICAL mirrors -- the closed form has to serve reflection
    as well as refraction."""
    return [
        Surface(radius=-0.400, thickness=-0.150, glass_before='air',
                glass_after='air', is_mirror=True, semi_diameter=0.050),
        Surface(radius=-0.120, thickness=0.250, glass_before='air',
                glass_after='air', is_mirror=True, semi_diameter=0.015),
        Surface(radius=np.inf, thickness=0.0, glass_before='air',
                glass_after='air', semi_diameter=0.030),
    ]


def _conic3():
    return [
        Surface(radius=0.0515, conic=-0.6, thickness=0.008,
                glass_before='air', glass_after='N-BK7',
                semi_diameter=0.0127),
        Surface(radius=-0.0345, conic=-1.2, thickness=0.100,
                glass_before='N-BK7', glass_after='air',
                semi_diameter=0.0127),
        Surface(radius=np.inf, thickness=0.0, glass_before='air',
                glass_after='air', semi_diameter=0.030),
    ]


def _aspheric2():
    return [
        Surface(radius=0.0515, aspheric_coeffs={4: 1.0e3, 6: -2.0e5},
                thickness=0.008, glass_before='air', glass_after='N-BK7',
                semi_diameter=0.0127),
        Surface(radius=-0.0345, thickness=0.100, glass_before='N-BK7',
                glass_after='air', semi_diameter=0.0127),
        Surface(radius=np.inf, thickness=0.0, glass_before='air',
                glass_after='air', semi_diameter=0.030),
    ]


CASES = [
    ('spherical7', _spherical7, 0.0126, True),
    ('triplet6', _triplet6, 0.0085, True),
    ('cassegrain2mirror', _cassegrain, 0.045, True),
    ('conic3 (control)', _conic3, 0.0126, False),
    ('aspheric3 (control)', _aspheric2, 0.0126, False),
]


def _load():
    try:
        import psutil
        return dict(cpu_percent=psutil.cpu_percent(interval=0.4),
                    n_python=sum(
                        1 for p in psutil.process_iter(['name'])
                        if (p.info['name'] or '').lower().startswith(
                            'python')),
                    load_source='psutil')
    except Exception:
        try:
            return dict(loadavg=os.getloadavg(), load_source='getloadavg')
        except Exception:
            return dict(load_source='unavailable')


_NORMAL_BLOCK = ('_surface_normal', '_sphere_normal',
                 '_base_surface_sag_derivatives_xy',
                 '_surface_sag_derivatives_xy', '_surface_sag_derivative')


def _profile_shares(S, rays, kw, repeats=3):
    """cProfile tottime SHARE of the normal block.

    A SHARE is far more robust on a contended box than a wall clock:
    everything in the profile is slowed by the same contention, so the
    ratio survives what the absolute numbers do not.  This is the same
    measurement WP-B9 used for its 24.1 % premise.
    """
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(repeats):
        trace(rays, S, WL, output_filter='last', **kw)
    pr.disable()
    st = pstats.Stats(pr)
    total = 0.0
    block = 0.0
    for (_f, _l, name), (_cc, _nc, tt, _ct, _cal) in st.stats.items():
        total += tt
        if name in _NORMAL_BLOCK:
            block += tt
    return dict(total_tottime=total, normal_block=block,
                share=(block / total if total else float('nan')))


def main():
    out_path = sys.argv[1]
    N = 100000
    if '--n' in sys.argv:
        N = int(sys.argv[sys.argv.index('--n') + 1])
    reps = 9
    inner = 20          # traces per timing sample: Windows' CPU
    # clock ticks at 15.6 ms, so a single 0.25 s trace is quantised at
    # 6 % and the no-sphere CONTROLS read a 20 % 'speedup' off one tick.
    load_before = _load()
    results = {}
    for label, builder, semi, has_sphere in CASES:
        S = builder()
        rays = _bundle(N, semi=semi)
        # warm the glass / dispatch caches on both arms
        for kw in ({}, {'sphere_normal': 'analytic'},
                   {'renormalize': 'exit'},
                   {'renormalize': 'exit', 'sphere_normal': 'analytic'}):
            trace(rays, S, WL, output_filter='last', **kw)
        arms = {
            'generic_surface': {},
            'analytic_surface': {'sphere_normal': 'analytic'},
            'generic_exit': {'renormalize': 'exit'},
            'analytic_exit': {'renormalize': 'exit',
                              'sphere_normal': 'analytic'},
        }
        best = {k: float('inf') for k in arms}
        best_cpu = {k: float('inf') for k in arms}
        order = list(arms)
        for rep in range(reps):
            # alternate the order every repeat so no arm is permanently
            # first (a first-in-the-rep arm pays any per-rep cache cost)
            seq = order if rep % 2 == 0 else order[::-1]
            for name in seq:
                c0 = time.process_time()
                t0 = time.perf_counter()
                for _ in range(inner):
                    trace(rays, S, WL, output_filter='last', **arms[name])
                wall = (time.perf_counter() - t0) / inner
                cpu = (time.process_time() - c0) / inner
                best[name] = min(best[name], wall)
                best_cpu[name] = min(best_cpu[name], cpu)
        results[label] = dict(
            has_sphere=has_sphere, n_rays=N, n_surfaces=len(S),
            reps=reps, inner=inner, seconds={k: best[k] for k in arms},
            cpu_seconds={k: best_cpu[k] for k in arms},
            speedup_sphere_normal=best_cpu['generic_surface']
            / best_cpu['analytic_surface'],
            speedup_renormalize=best_cpu['generic_surface']
            / best_cpu['generic_exit'],
            speedup_both=best_cpu['generic_surface']
            / best_cpu['analytic_exit'],
            speedup_renormalize_on_analytic=best_cpu['analytic_surface']
            / best_cpu['analytic_exit'],
            wall_speedup_sphere_normal=best['generic_surface']
            / best['analytic_surface'],
            wall_speedup_renormalize=best['generic_surface']
            / best['generic_exit'],
        )
        r = results[label]
        print('%-22s %2d surf CPU generic %.4f  analytic %.4f (%.3fx)  '
              'exit %.4f (%.3fx)  both %.4f (%.3fx) | wall %.3fx / %.3fx'
              % (label, len(S), best_cpu['generic_surface'],
                 best_cpu['analytic_surface'], r['speedup_sphere_normal'],
                 best_cpu['generic_exit'], r['speedup_renormalize'],
                 best_cpu['analytic_exit'], r['speedup_both'],
                 r['wall_speedup_sphere_normal'],
                 r['wall_speedup_renormalize']), flush=True)
    # Structural, contention-immune: the normal block's share of the
    # profile on the seven-surface spherical stack.
    S = _spherical7()
    rays = _bundle(100000, semi=0.0126)
    profile = {
        'generic': _profile_shares(S, rays, {}),
        'analytic': _profile_shares(S, rays,
                                    {'sphere_normal': 'analytic'}),
    }
    print('normal-block share: generic %.3f %%  analytic %.3f %%'
          % (100 * profile['generic']['share'],
             100 * profile['analytic']['share']))
    load_after = _load()
    sph = [v for v in results.values() if v['has_sphere']]
    summary = dict(
        sphere_normal_range=[
            min(v['speedup_sphere_normal'] for v in sph),
            max(v['speedup_sphere_normal'] for v in sph)],
        sphere_normal_median=float(np.median(
            [v['speedup_sphere_normal'] for v in sph])),
        renormalize_range=[
            min(v['speedup_renormalize'] for v in results.values()),
            max(v['speedup_renormalize'] for v in results.values())],
        renormalize_median=float(np.median(
            [v['speedup_renormalize'] for v in results.values()])),
        renormalize_on_analytic_range=[
            min(v['speedup_renormalize_on_analytic']
                for v in results.values()),
            max(v['speedup_renormalize_on_analytic']
                for v in results.values())],
        both_range=[min(v['speedup_both'] for v in sph),
                    max(v['speedup_both'] for v in sph)],
        control_sphere_normal=[
            v['speedup_sphere_normal'] for k, v in results.items()
            if not v['has_sphere']],
        control_wall_sphere_normal=[
            v['wall_speedup_sphere_normal'] for k, v in results.items()
            if not v['has_sphere']],
        normal_block_share_generic=profile['generic']['share'],
        normal_block_share_analytic=profile['analytic']['share'],
        normal_block_profile=profile,
    )
    for k, v in summary.items():
        print(k, v)
    meta = dict(python=sys.version, numpy=np.__version__,
                lumenairy=la.__version__, platform=sys.platform,
                load_before=load_before, load_after=load_after,
                threads={k: os.environ.get(k) for k in
                         ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                          'MKL_NUM_THREADS')})
    with open(out_path, 'w') as fh:
        json.dump(dict(meta=meta, summary=summary, results=results),
                  fh, indent=1)
    print('wrote', out_path)


if __name__ == '__main__':
    main()
