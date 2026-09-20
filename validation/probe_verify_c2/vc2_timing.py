"""VERIFY-WP-C2 items 2 and 3 -- the two switches' cost, on three
contention-immune instruments plus a batched CPU clock.

Instruments, weakest to strongest:

1. **Batched CPU time.**  ``time.process_time()``.  Windows' process clock
   ticks at 15.6 ms, so each sample is a BATCH of traces auto-sized to at
   least ``VC2_MIN_SAMPLE`` seconds (default 2.0), which puts the tick at
   under 1 % of a sample.  Arms are interleaved and the order alternates
   every repeat; the reported number is the MINIMUM over repeats, which is
   the contention-robust estimator (load can only add time).
2. **Profile share.**  ``cProfile`` tottime of the normal block and of the
   refract / reflect bodies as a FRACTION of the traced total.  A share is
   a ratio of two quantities measured in the same profiled run, so
   box load cancels out of it.
3. **The structural microbenchmark** (``renormalize`` only).  The hoist
   removes, per surface, one ``np.maximum`` and three in-place divisions on
   an N-array, and adds one ``_normalize_directions`` at the end.  Both are
   timed directly, so the PREDICTED saving is arithmetic rather than a
   difference of two noisy wall times -- and it bounds what the switch can
   possibly be worth.

Usage: ``LUMENAIRY_ROOT=<root> python vc2_timing.py OUT.json``
"""
import cProfile
import json
import os
import pstats
import sys
import time

_ROOT = os.environ['LUMENAIRY_ROOT']
sys.path.insert(0, _ROOT)

import numpy as np                                            # noqa: E402
import lumenairy as la                                        # noqa: E402

_want = os.path.realpath(os.path.join(_ROOT, 'lumenairy'))
assert os.path.realpath(os.path.dirname(la.__file__)) == _want, la.__file__

from lumenairy.raytrace.surface import RayBundle, Surface     # noqa: E402
from lumenairy.raytrace.trace import trace                    # noqa: E402

WL = 587.5618e-9
N_RAYS = int(os.environ.get('VC2_N', '100000'))
MIN_SAMPLE = float(os.environ.get('VC2_MIN_SAMPLE', '2.0'))
REPEATS = int(os.environ.get('VC2_REPEATS', '7'))

NORMAL_FUNCS = ('_surface_normal', '_sphere_normal',
                '_surface_sag_derivatives_xy', '_axis_deriv',
                '_surface_sag_derivative')
RENORM_FUNCS = ('_refract', '_reflect', '_normalize_directions')


def _s(R, th, gb, ga, sd, conic=0.0, asph=None, mirror=False):
    return Surface(radius=R, conic=conic, aspheric_coeffs=asph, thickness=th,
                   glass_before=gb, glass_after=ga, semi_diameter=sd,
                   is_mirror=mirror)


def _ladder(n_pairs):
    out = []
    for j in range(n_pairs):
        out.append(_s(0.0800 + 0.003 * j, 0.0055, 'air', 'N-BK7', 0.011))
        out.append(_s(-0.0900 - 0.003 * j, 0.0090, 'N-BK7', 'air', 0.011))
    out.append(_s(0.2500, 0.0300, 'air', 'N-BK7', 0.011))
    return out


PRES = {
    # --- sphere-bearing: the switch bites ---
    'doublet3': [_s(0.0517, 0.0090, 'air', 'N-BK7', 0.0125),
                 _s(-0.0345, 0.0025, 'N-BK7', 'N-SF5', 0.0125),
                 _s(-0.1200, 0.0400, 'N-SF5', 'air', 0.0125)],
    'stack7': _ladder(3),
    'ladder13': _ladder(6),
    'two_mirror': [_s(-0.3000, -0.1000, 'air', 'air', 0.060, mirror=True),
                   _s(-0.0900, 0.2000, 'air', 'air', 0.020, mirror=True)],
    # --- CONTROLS: no pure sphere anywhere, so sphere_normal cannot
    #     change anything.  They measure the method's own resolution.
    'conic3_control': [_s(0.0517, 0.0090, 'air', 'N-BK7', 0.0125,
                          conic=-0.6),
                       _s(-0.0345, 0.0025, 'N-BK7', 'N-SF5', 0.0125,
                          conic=-1.2),
                       _s(-0.1200, 0.0400, 'N-SF5', 'air', 0.0125,
                          conic=0.4)],
    'asphere3_control': [_s(0.0517, 0.0090, 'air', 'N-BK7', 0.0125,
                            asph={4: 1e3}),
                         _s(-0.0345, 0.0025, 'N-BK7', 'N-SF5', 0.0125,
                            asph={4: -5e2}),
                         _s(-0.1200, 0.0400, 'N-SF5', 'air', 0.0125,
                            asph={4: 2e2})],
}


def _bundle(n):
    rng = np.random.default_rng(20260920)
    h = rng.uniform(-0.0075, 0.0075, n)
    g = rng.uniform(-0.0075, 0.0075, n)
    return RayBundle(x=h, y=g, z=np.zeros(n), L=np.zeros(n), M=np.zeros(n),
                     N=np.ones(n), wavelength=WL,
                     alive=np.ones(n, dtype=bool), opd=np.zeros(n))


def _calibrate(fn):
    """Repetitions that make one sample at least ``MIN_SAMPLE`` seconds.

    The first call of any arm is COLD (glass-index caches, numpy's first
    touch of each buffer), so calibrating on it returns a repetition count
    sized for a cost the warm loop never pays -- and the warm sample then
    falls under the 15.6 ms Windows process tick and reads 0.  Warm up
    first, then GROW until the measured sample actually clears the target.
    """
    for _ in range(3):
        fn()
    reps = 1
    while True:
        t0 = time.process_time()
        for _ in range(reps):
            fn()
        dt = time.process_time() - t0
        if dt >= MIN_SAMPLE:
            return reps, dt
        if reps >= 1 << 20:
            raise RuntimeError('cannot reach the minimum sample size')
        reps *= 2 if dt > 0.25 * MIN_SAMPLE else 4


def _timed(fns, reps):
    """One interleaved sample per arm: (cpu, wall) each."""
    out = {}
    for name, fn in fns:
        t0c, t0w = time.process_time(), time.perf_counter()
        for _ in range(reps):
            fn()
        out[name] = (time.process_time() - t0c, time.perf_counter() - t0w)
    return out


def _bench(fns):
    reps, _cal = _calibrate(fns[0][1])
    best = {n: (float('inf'), float('inf')) for n, _ in fns}
    for k in range(REPEATS):
        order = fns if k % 2 == 0 else list(reversed(fns))
        got = _timed(order, reps)
        for n, (c, w) in got.items():
            best[n] = (min(best[n][0], c / reps), min(best[n][1], w / reps))
    return reps, {n: dict(cpu=c, wall=w) for n, (c, w) in best.items()}


N_PROF = int(os.environ.get('VC2_N_PROF', '15'))


def _profile_shares(rb, surfs, **kw):
    trace(rb, surfs, WL, output_filter='last', **kw)      # warm
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(N_PROF):
        trace(rb, surfs, WL, output_filter='last', **kw)
    pr.disable()
    st = pstats.Stats(pr)
    tot = 0.0
    per = {}
    for (fpath, _line, fname), row in st.stats.items():
        tt = row[2]
        tot += tt
        if fname in NORMAL_FUNCS or fname in RENORM_FUNCS or fname == 'trace':
            per[fname] = per.get(fname, 0.0) + tt
    normal = sum(per.get(f, 0.0) for f in NORMAL_FUNCS)
    return dict(total_tottime=tot,
                normal_block=normal,
                normal_share=normal / tot if tot else 0.0,
                refract_reflect=sum(per.get(f, 0.0)
                                    for f in ('_refract', '_reflect')),
                refract_reflect_share=(
                    sum(per.get(f, 0.0) for f in ('_refract', '_reflect'))
                    / tot if tot else 0.0),
                normalize_directions=per.get('_normalize_directions', 0.0),
                normalize_directions_share=(
                    per.get('_normalize_directions', 0.0) / tot
                    if tot else 0.0),
                per_func={k: v for k, v in sorted(per.items())})


def _micro(n):
    """Cost of the block the hoist removes, and of the pass it adds.

    Both are written EXACTLY as the library writes them -- ``rays.L /= mag``
    is an IN-PLACE divide, and an allocating ``a = L / m`` costs a fresh
    N-array each time.  A first version of this probe used the allocating
    form and over-predicted the hoist by up to 1.53x on WSL against a
    measured 1.00x; the in-place form is the one the switch actually
    removes.  ``mag`` itself is computed in BOTH modes (the degenerate-ray
    diagnosis needs it), so it is not part of what the hoist saves.
    """
    rng = np.random.default_rng(7)
    L = rng.normal(size=n)
    M = rng.normal(size=n)
    Nd = rng.normal(size=n)
    mag0 = np.sqrt(L ** 2 + M ** 2 + Nd ** 2)
    buf = np.empty(n)
    box = [L, M, Nd]

    def removed():
        L, M, Nd = box
        # ``mag = np.maximum(mag, 1e-30); rays.L /= mag; ...`` per surface
        np.maximum(mag0, 1e-30, out=buf)
        L /= buf
        M /= buf
        Nd /= buf
        L *= buf
        M *= buf
        Nd *= buf          # restore, so the arrays do not drift to zero
        return buf[0]

    def added():
        L, M, Nd = box
        # the single exit pass: the sqrt is EXTRA here (the per-surface
        # ``mag`` already existed), then maximum + three in-place divides
        np.sqrt(L * L + M * M + Nd * Nd, out=buf)
        np.maximum(buf, 1e-30, out=buf)
        L /= buf
        M /= buf
        Nd /= buf
        L *= buf
        M *= buf
        Nd *= buf
        return buf[0]

    _r, got = _bench([('removed', removed), ('added', added)])
    out = {k: 0.5 * v['cpu'] for k, v in got.items()}   # halve: the
    # restore multiply doubles each body, and costs the same as the divide
    out['note'] = ('each body does the divide AND an identical multiply to '
                   'restore the arrays; the reported number is half the '
                   'measured body, i.e. the divide half alone')
    return out


def main(out_path):
    rb = _bundle(N_RAYS)
    res = {'meta': dict(python=sys.version.split()[0], numpy=np.__version__,
                        lumenairy=la.__version__, file=la.__file__,
                        n_rays=N_RAYS, repeats=REPEATS,
                        min_sample_s=MIN_SAMPLE,
                        cpu_count=os.cpu_count())}
    micro = _micro(N_RAYS)
    res['microbench_cpu_s'] = micro
    # A DETERMINISTIC bound alongside the timed one: count the N-sized
    # array passes each mode makes.  'surface' spends, per refracting
    # surface, one ``np.maximum`` and three in-place divides = 4 passes.
    # 'exit' spends 0 per surface and, once, ``_normalize_directions``:
    # 3 squares + 2 adds + 1 sqrt + 1 maximum + 3 divides = 10 passes.
    res['passes_saved_formula'] = '4 * n_refracting - 10'

    sphere = {}
    renorm = {}
    for name, surfs in PRES.items():
        n_ref = sum(1 for s in surfs if not s.is_coordbrk)

        def mk(sn, rn):
            return lambda: trace(rb, surfs, WL, output_filter='last',
                                 sphere_normal=sn, renormalize=rn)

        # --- sphere_normal, renormalize held at 'exit' (the new default)
        reps, got = _bench([('generic', mk('generic', 'exit')),
                            ('analytic', mk('analytic', 'exit'))])
        pg = _profile_shares(rb, surfs, sphere_normal='generic',
                             renormalize='exit')
        pa = _profile_shares(rb, surfs, sphere_normal='analytic',
                             renormalize='exit')
        sphere[name] = dict(
            n_surfaces=len(surfs), reps=reps,
            cpu_generic=got['generic']['cpu'],
            cpu_analytic=got['analytic']['cpu'],
            speedup_cpu=got['generic']['cpu'] / got['analytic']['cpu'],
            speedup_wall=got['generic']['wall'] / got['analytic']['wall'],
            normal_share_generic=pg['normal_share'],
            normal_share_analytic=pa['normal_share'],
            profile=dict(generic=pg, analytic=pa))

        # --- renormalize, sphere_normal held at 'analytic'
        reps2, got2 = _bench([('surface', mk('analytic', 'surface')),
                              ('exit', mk('analytic', 'exit'))])
        ps = _profile_shares(rb, surfs, sphere_normal='analytic',
                             renormalize='surface')
        pe = _profile_shares(rb, surfs, sphere_normal='analytic',
                             renormalize='exit')
        pred_saving = n_ref * micro['removed'] - micro['added']
        passes_saved = 4 * n_ref - 10
        t_surface = got2['surface']['cpu']
        renorm[name] = dict(
            n_surfaces=len(surfs), n_refracting=n_ref, reps=reps2,
            cpu_surface=t_surface, cpu_exit=got2['exit']['cpu'],
            speedup_cpu=t_surface / got2['exit']['cpu'],
            speedup_wall=got2['surface']['wall'] / got2['exit']['wall'],
            refl_share_surface=ps['refract_reflect_share'],
            refl_share_exit=pe['refract_reflect_share'],
            normdir_share_exit=pe['normalize_directions_share'],
            profile_share_of_hoisted_block=(
                ps['refract_reflect_share'] - pe['refract_reflect_share']),
            predicted_saving_s=pred_saving,
            predicted_speedup=(t_surface / (t_surface - pred_saving)
                               if t_surface > pred_saving else float('inf')),
            predicted_saving_fraction=pred_saving / t_surface,
            passes_saved=passes_saved)
    res['sphere_normal'] = sphere
    res['renormalize'] = renorm

    def _med(vals):
        v = sorted(vals)
        return v[len(v) // 2] if len(v) % 2 else 0.5 * (v[len(v) // 2 - 1]
                                                        + v[len(v) // 2])

    sph_real = [k for k in sphere if not k.endswith('_control')]
    res['summary'] = dict(
        sphere_speedup_range=[min(sphere[k]['speedup_cpu']
                                  for k in sph_real),
                              max(sphere[k]['speedup_cpu']
                                  for k in sph_real)],
        sphere_speedup_median=_med([sphere[k]['speedup_cpu']
                                    for k in sph_real]),
        control_speedup=[sphere[k]['speedup_cpu'] for k in sphere
                         if k.endswith('_control')],
        normal_share_generic_range=[
            min(sphere[k]['normal_share_generic'] for k in sph_real),
            max(sphere[k]['normal_share_generic'] for k in sph_real)],
        normal_share_analytic_range=[
            min(sphere[k]['normal_share_analytic'] for k in sph_real),
            max(sphere[k]['normal_share_analytic'] for k in sph_real)],
        renorm_speedup_range=[min(renorm[k]['speedup_cpu'] for k in renorm),
                              max(renorm[k]['speedup_cpu'] for k in renorm)],
        renorm_speedup_median=_med([renorm[k]['speedup_cpu']
                                    for k in renorm]),
        renorm_predicted_speedup_range=[
            min(renorm[k]['predicted_speedup'] for k in renorm),
            max(renorm[k]['predicted_speedup'] for k in renorm)],
        renorm_profile_share_of_hoisted_block=[
            renorm[k]['profile_share_of_hoisted_block'] for k in renorm],
    )
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    print(json.dumps(res['summary'], indent=1, sort_keys=True))
    for k in sphere:
        print(f"sphere {k:18s} cpu {sphere[k]['speedup_cpu']:.4f}x  "
              f"wall {sphere[k]['speedup_wall']:.4f}x  share "
              f"{sphere[k]['normal_share_generic']:.4%} -> "
              f"{sphere[k]['normal_share_analytic']:.4%}")
    for k in renorm:
        print(f"renorm {k:18s} cpu {renorm[k]['speedup_cpu']:.4f}x  "
              f"wall {renorm[k]['speedup_wall']:.4f}x  predicted "
              f"{renorm[k]['predicted_speedup']:.4f}x  block share "
              f"{renorm[k]['profile_share_of_hoisted_block']:.4%}")
    print('micro', micro)


if __name__ == '__main__':
    main(sys.argv[1])
