"""VERIFY-WP-C2 item 2 -- what the ``renormalize`` hoist can possibly be
worth, measured on the LIBRARY'S OWN code rather than on a re-implementation.

The hoist removes, per refracting surface, exactly the difference between
``_refract(..., renormalize=True)`` and ``_refract(..., renormalize=False)``
-- the ``np.maximum`` and the three in-place divides.  It adds exactly one
``_normalize_directions`` call.  Both are timed here directly, so the
predicted saving needs no model of what the block costs:

    saving(S) = S * [t(_refract, True) - t(_refract, False)]
                - t(_normalize_directions)

and the BREAK-EVEN surface count is ``t(_normalize_directions)`` divided by
the per-surface difference.  Below it the hoist COSTS time, because the one
exit pass recomputes the magnitude with its own ``sqrt`` and three squares
while the per-surface block it replaces only divides by a magnitude the
degenerate-ray diagnosis had already computed.

Every number is a MINIMUM over repeats of a batch sized past the 15.6 ms
Windows process tick, so box load can only make it larger.

Usage: ``LUMENAIRY_ROOT=<root> python vc2_renorm_cost.py OUT.json``
"""
import json
import os
import sys
import time

_ROOT = os.environ['LUMENAIRY_ROOT']
sys.path.insert(0, _ROOT)

import numpy as np                                            # noqa: E402
import lumenairy as la                                        # noqa: E402

_want = os.path.realpath(os.path.join(_ROOT, 'lumenairy'))
assert os.path.realpath(os.path.dirname(la.__file__)) == _want, la.__file__

from lumenairy.raytrace.surface import RayBundle, Surface     # noqa: E402
from lumenairy.raytrace.intersection import (                 # noqa: E402
    _normalize_directions, _refract)
from lumenairy.raytrace.trace import trace                    # noqa: E402

WL = 587.5618e-9
N = int(os.environ.get('VC2_N', '100000'))
MIN_SAMPLE = float(os.environ.get('VC2_MIN_SAMPLE', '1.5'))
REPEATS = int(os.environ.get('VC2_REPEATS', '9'))


def _bundle(n=None):
    n = n or N
    rng = np.random.default_rng(555)
    x = rng.uniform(-0.0075, 0.0075, n)
    y = rng.uniform(-0.0075, 0.0075, n)
    L = rng.uniform(-0.05, 0.05, n)
    M = rng.uniform(-0.05, 0.05, n)
    Nd = np.sqrt(1.0 - L ** 2 - M ** 2)
    return RayBundle(x=x, y=y, z=np.zeros(n), L=L, M=M, N=Nd,
                     wavelength=WL, alive=np.ones(n, dtype=bool),
                     opd=np.zeros(n))


def _calibrate(fn):
    for _ in range(3):
        fn()
    reps = 1
    while True:
        t0 = time.process_time()
        for _ in range(reps):
            fn()
        dt = time.process_time() - t0
        if dt >= MIN_SAMPLE:
            return reps
        if reps >= 1 << 22:
            raise RuntimeError('cannot reach the minimum sample')
        reps *= 2 if dt > 0.25 * MIN_SAMPLE else 4


def _bench(fns):
    """Minimum per-call CPU AND wall time over interleaved repeats.

    Windows' process clock ticks at 15.6 ms, which on a 3.5 ms body with
    512 repetitions quantises at 0.9 % -- coarse enough that the 3 %
    difference this probe is looking for reads as exactly ZERO.  The
    wall clock has nanosecond resolution and, taken as a MINIMUM over
    interleaved repeats on a CPU-bound single-threaded loop, resolves it.
    Both are reported; the wall minimum is the one the verdict uses and
    the CPU minimum is carried as the cross-check.
    """
    reps = _calibrate(fns[0][1])
    best = {n: (float('inf'), float('inf')) for n, _ in fns}
    for k in range(REPEATS):
        order = fns if k % 2 == 0 else list(reversed(fns))
        for name, fn in order:
            t0c, t0w = time.process_time(), time.perf_counter()
            for _ in range(reps):
                fn()
            c = (time.process_time() - t0c) / reps
            w = (time.perf_counter() - t0w) / reps
            best[name] = (min(best[name][0], c), min(best[name][1], w))
    return reps, {n: v[1] for n, v in best.items()},         {n: v[0] for n, v in best.items()}


def main(out_path):
    surf = Surface(radius=0.0517, conic=0.0, thickness=0.009,
                   glass_before='air', glass_after='N-BK7',
                   semi_diameter=np.inf)
    rb_t = _bundle()
    rb_f = _bundle()
    rb_n = _bundle()
    # A second, NON-BENDING pair: n1 == n2 makes Snell the identity, so
    # no ray ever TIRs and no direction ever degenerates however many
    # times the body is repeated.  It isolates the renormalise block from
    # any drift the repeated-refraction arms could accumulate.
    rb_t2 = _bundle()
    rb_f2 = _bundle()

    def refract_true_flat():
        _refract(rb_t2, surf, 1.0, 1.0, renormalize=True,
                 sphere_normal='analytic')

    def refract_false_flat():
        _refract(rb_f2, surf, 1.0, 1.0, renormalize=False,
                 sphere_normal='analytic')

    def refract_true():
        _refract(rb_t, surf, 1.0, 1.5168, renormalize=True,
                 sphere_normal='analytic')

    def refract_false():
        _refract(rb_f, surf, 1.0, 1.5168, renormalize=False,
                 sphere_normal='analytic')

    def normalize():
        _normalize_directions(rb_n)

    reps, got, got_cpu = _bench([('refract_true', refract_true),
                                 ('refract_false', refract_false),
                                 ('refract_true_flat', refract_true_flat),
                                 ('refract_false_flat', refract_false_flat),
                                 ('normalize_directions', normalize)])
    per_surface = got['refract_true'] - got['refract_false']
    added = got['normalize_directions']
    break_even = added / per_surface if per_surface > 0 else float('inf')

    res = dict(
        meta=dict(python=sys.version.split()[0], numpy=np.__version__,
                  lumenairy=la.__version__, file=la.__file__, n_rays=N,
                  reps=reps, repeats=REPEATS,
                  min_sample_s=MIN_SAMPLE, cpu_count=os.cpu_count()),
        t_refract_renormalize_true=got['refract_true'],
        t_refract_renormalize_false=got['refract_false'],
        t_normalize_directions=added,
        removed_per_surface_s=per_surface,
        removed_per_surface_fraction_of_refract=(
            per_surface / got['refract_true']),
        cpu_clock_cross_check=got_cpu,
        cpu_clock_removed_per_surface_s=(got_cpu['refract_true']
                                         - got_cpu['refract_false']),
        added_once_s=added,
        break_even_surface_count=break_even,
        nonbending_cross_check=dict(
            t_true=got['refract_true_flat'],
            t_false=got['refract_false_flat'],
            removed_per_surface_s=(got['refract_true_flat']
                                   - got['refract_false_flat']),
            removed_fraction=((got['refract_true_flat']
                               - got['refract_false_flat'])
                              / got['refract_true_flat']),
            break_even_surface_count=(
                added / (got['refract_true_flat']
                         - got['refract_false_flat'])
                if got['refract_true_flat'] > got['refract_false_flat']
                else float('inf'))))

    # --- turn it into a predicted speed-up per surface count -------
    def _ladder(n_pairs):
        out = []
        for j in range(n_pairs):
            out.append(Surface(radius=0.0800 + 0.003 * j, thickness=0.0055,
                               glass_before='air', glass_after='N-BK7',
                               semi_diameter=0.011))
            out.append(Surface(radius=-0.0900 - 0.003 * j, thickness=0.0090,
                               glass_before='N-BK7', glass_after='air',
                               semi_diameter=0.011))
        out.append(Surface(radius=0.2500, thickness=0.0300,
                           glass_before='air', glass_after='N-BK7',
                           semi_diameter=0.011))
        return out

    pred = {}
    rb = _bundle()
    for n_pairs in (1, 2, 3, 4, 5, 6):
        surfs = _ladder(n_pairs)
        s = len(surfs)

        def whole():
            trace(rb, surfs, WL, output_filter='last',
                  sphere_normal='analytic', renormalize='surface')

        r2, g2, _c2 = _bench([('whole', whole)])
        t_whole = g2['whole']
        saving = s * per_surface - added
        pred[f'{s}_surfaces'] = dict(
            t_surface_mode_s=t_whole,
            predicted_saving_s=saving,
            predicted_saving_fraction=saving / t_whole,
            predicted_speedup=(t_whole / (t_whole - saving)
                               if t_whole > saving else float('inf')))
    res['predicted_per_surface_count'] = pred

    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    print(json.dumps(res, indent=1, sort_keys=True))


if __name__ == '__main__':
    main(sys.argv[1])
