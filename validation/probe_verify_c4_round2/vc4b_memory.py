"""VERIFY-WP-C4 ROUND 2, item 3 -- the memory half, re-measured and re-derived.

Three separate questions, kept apart:

1.  **Is the dense route the cheapest at every shape the rule CAPTURES?**
    Measured with ``tracemalloc`` over this verification's own 35-shape ladder,
    cold, one route per trace, every route's answer digested so a row cannot be
    cheap because a route did not run.
2.  **Do the committed round-2 readings say what the report says they say?**
    The committed ``r2_memcensus_all_{win,wsl}.json`` are re-counted here
    (42 of 51 dense-cheapest, the captured-and-not-cheapest set empty, 0 of 51
    identical readings across builds, 51 of 51 identical cheapest route)
    instead of being quoted.
3.  **Are the byte counts DERIVED from the code, or read off the box?**  The
    three derived counts are recomputed here from the array shapes
    ``_bluestein.py`` actually allocates, and compared against the measured
    ordering on this ladder.  A derivation that predicts the ordering is an
    argument; one that only agrees with the run it came from is not.

    PYTHONPATH=<tree> python vc4b_memory.py <tree>

Author:  Andrew Traverso
"""
from __future__ import annotations

import gc
import json
import os
import sys
import tracemalloc

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vc4blib as L                                              # noqa: E402
from vc4b_ladder import KW, LADDER, alpha_for, work_per_entry    # noqa: E402

ROUTES = ('chirpz2d', 'separable', 'dense')


def dense_live(ny, nx, my, mx):
    """Live bytes at the dense route's peak, from ``_direct_matrix_2d``.

    ``_kernel`` forms a float64 ``t`` of ``M x N`` (24 bytes per entry once
    ``t``, ``rint(t)`` and the difference coexist) and then a complex128ted
    ``W``; the two kernels, one intermediate and the output coexist at the
    matmul.  The larger of the two is the peak.
    """
    build = 24 * max(my * ny, mx * nx)
    kernels = 16 * (my * ny + mx * nx)
    y_first = (my * ny * nx + my * nx * mx) <= (ny * nx * mx + my * ny * mx)
    inter = 16 * (my * nx if y_first else ny * mx)
    return max(build, kernels + inter + 16 * my * mx)


def sep_live(ny, nx, my, mx):
    """Largest ``(N_in x L)`` working array of the separable chirp-Z route."""
    from scipy.fft import next_fast_len
    ly, lx = int(next_fast_len(ny + my - 1)), int(next_fast_len(nx + mx - 1))
    return 16 * max(ny * lx, my * ly)


def chirp2d_live(ny, nx, my, mx):
    """The ``Ly x Lx`` padded plane of the 2-D chirp-Z route."""
    from scipy.fft import next_fast_len
    ly, lx = int(next_fast_len(ny + my - 1)), int(next_fast_len(nx + mx - 1))
    return 16 * ly * lx


def recount_committed(repo):
    """Re-count the branch's committed census instead of quoting it."""
    probe = os.path.join(repo, 'validation', 'probe_c4_round2')
    data = {}
    for build in ('win', 'wsl'):
        p = os.path.join(probe, "r2_memcensus_all_%s.json" % build)
        with open(p, encoding='cp1252') as fh:
            data[build] = {r['name']: r for r in json.load(fh)['rows']}
    names = sorted(set(data['win']) & set(data['wsl']))
    out = {'n_shapes': len(names)}
    for build in ('win', 'wsl'):
        rows = data[build]
        out['dense_cheapest_%s' % build] = sum(
            1 for n in names if rows[n]['cheapest'] == 'dense')
        out['captured_%s' % build] = sum(
            1 for n in names if rows[n]['rule_says_direct'])
        out['captured_and_not_cheapest_%s' % build] = [
            n for n in names if rows[n]['rule_says_direct']
            and rows[n]['cheapest'] != 'dense']
        out['not_cheapest_%s' % build] = [
            n for n in names if rows[n]['cheapest'] != 'dense']
    out['identical_readings_across_builds'] = [
        n for n in names
        if data['win'][n]['peak_bytes'] == data['wsl'][n]['peak_bytes']]
    out['identical_cheapest_across_builds'] = sum(
        1 for n in names
        if data['win'][n]['cheapest'] == data['wsl'][n]['cheapest'])
    # the shape the whole D1/D3 memory argument rests on
    thin = 'w0400_2048x64b'
    if thin in data['win']:
        out['w0400_2048x64b'] = {
            b: {'peak_MB': data[b][thin].get('peak_MB'),
                'cheapest': data[b][thin]['cheapest'],
                'rule_says_direct': data[b][thin]['rule_says_direct']}
            for b in ('win', 'wsl')}
    return out


def main(tree):
    import numpy as np
    la = L.anchor(tree)
    L.single_thread_ffts()
    from lumenairy.propagators._bluestein import (
        _auto_selects_direct, _bluestein_2d)
    from lumenairy.propagators.fft_infra import _fft2, _ifft2

    repo = os.path.dirname(os.path.dirname(HERE))
    out = {'build': L.build(), 'python': sys.version.split()[0],
           'numpy': np.__version__, 'lumenairy_version': la.__version__,
           'rows': [], 'committed_recount': recount_committed(repo)}
    rng = np.random.default_rng(20260920)
    for (name, ny, nx, my, mx, grp) in LADDER:
        E = (rng.standard_normal((ny, nx))
             + 1j * rng.standard_normal((ny, nx))).astype(np.complex128)
        a = alpha_for(ny, nx, my, mx)
        peak, dig = {}, {}
        for r in ROUTES:
            L.cold()
            tracemalloc.start()
            F = _bluestein_2d(E, a, a, my, mx, sign=-1, xp=np, fft2=_fft2,
                              ifft2=_ifft2, **KW[r])
            _cur, pk = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak[r] = int(pk)
            dig[r] = L.digest(F)
            del F
            gc.collect()
        derived = {'dense': dense_live(ny, nx, my, mx),
                   'separable': sep_live(ny, nx, my, mx),
                   'chirpz2d': chirp2d_live(ny, nx, my, mx)}
        cheapest = min(peak, key=lambda k: peak[k])
        cheapest_derived = min(derived, key=lambda k: derived[k])
        row = {'name': name, 'group': grp, 'Ny_in': ny, 'Nx_in': nx,
               'My': my, 'Mx': mx,
               'work_per_entry': work_per_entry(ny, nx, my, mx),
               'rule_says_direct': bool(_auto_selects_direct(ny, nx, my, mx)),
               'peak_bytes': peak,
               'peak_MB': {k: round(v / 2 ** 20, 3) for k, v in peak.items()},
               'derived_bytes': derived,
               'cheapest': cheapest, 'cheapest_derived': cheapest_derived,
               'derivation_predicts_cheapest': cheapest == cheapest_derived,
               'distinct_route_digests': len(set(dig.values()))}
        out['rows'].append(row)
        print("%-14s %dx%d->%dx%-4d w/e=%8.2f rule=%s peak MB d=%8.3f "
              "s=%8.3f c=%8.3f cheapest=%-10s derived=%-10s %s"
              % (name, ny, nx, my, mx, row['work_per_entry'],
                 'D' if row['rule_says_direct'] else 'c',
                 row['peak_MB']['dense'], row['peak_MB']['separable'],
                 row['peak_MB']['chirpz2d'], cheapest, cheapest_derived,
                 'OK' if row['derivation_predicts_cheapest'] else 'MISS'),
              flush=True)
        del E
        gc.collect()

    cap = [r for r in out['rows'] if r['rule_says_direct']]
    out['n_shapes'] = len(out['rows'])
    out['dense_cheapest_count'] = sum(1 for r in out['rows']
                                      if r['cheapest'] == 'dense')
    out['CAPTURED_AND_NOT_CHEAPEST'] = [
        (r['name'], r['cheapest'], r['peak_MB']) for r in cap
        if r['cheapest'] != 'dense']
    out['derivation_predicts_at'] = sum(
        1 for r in out['rows'] if r['derivation_predicts_cheapest'])
    print()
    print("shapes                        :", out['n_shapes'])
    print("dense cheapest at             :", out['dense_cheapest_count'])
    print("captured                      :", len(cap))
    print("CAPTURED AND NOT CHEAPEST     :", out['CAPTURED_AND_NOT_CHEAPEST'])
    print("derived counts predict at     :", out['derivation_predicts_at'])
    print("committed recount             :",
          json.dumps({k: v for k, v in out['committed_recount'].items()
                      if not isinstance(v, dict)}, default=str))
    L.write(out, os.path.join(HERE, "vc4b_memory_%s.json" % L.tag()))


if __name__ == '__main__':
    main(sys.argv[1])
