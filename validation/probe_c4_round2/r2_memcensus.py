"""WP-C4 round 2, V-C4-D1 / V-C4-D3 -- the memory census, re-run over the
work-ratio ladder with the byte counts RE-DERIVED from the code.

Two claims are settled here.

* **V-C4-D1's memory half.**  WP-C4's report says the dense route is the
  cheapest of the three "at 42 of 42 shapes" and that "the memory half never
  argues against the rule anywhere".  VERIFY-WP-C4 found the exception at
  ``2048x64 -> 64x2`` (dense 5.264 MB against separable 4.399 MB).  This probe
  measures the whole 51-shape round-2 ladder and asks the one question that
  matters after the second condition lands: is the dense route the cheapest at
  every shape the rule STILL CAPTURES?
* **V-C4-D3's cross-build claim.**  The same report says the two builds'
  readings are "identical to the byte at every shape".  They are not; the
  ORDERING is.  Both are recorded here per shape, so the corrected sentence
  has its own measurement.

``tracemalloc`` peak, cold (every registered cache dropped and
``gc.collect()`` before each route), one route per trace, each route's answer
digested so a row cannot be cheap because a route did not run.  The DERIVED
byte counts come from the two routes' own array sizes in
``_bluestein.py`` -- the same expressions
``tests/unit/test_verify_c4_mft_direct.py::
test_the_derived_byte_counts_predict_where_the_memory_half_fails`` asserts.

    PYTHONPATH=<tree> python r2_memcensus.py <tree> [--family thin|square|all]
"""
from __future__ import annotations

import gc
import os
import sys
import tracemalloc

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'probe_verify_c4'))
import v4lib  # noqa: E402
from r2_workladder import (  # noqa: E402
    ROUTES, SQUARE, THIN, alpha_for, run_once, work_per_kernel_entry,
)


def dense_live(ny, nx, my, mx) -> int:
    """Live bytes at the dense route's peak, from ``_direct_matrix_2d``: the
    larger kernel's float64 ``t`` plus its complex128 ``W`` while it is being
    built, against the two finished kernels plus the intermediate plus the
    output."""
    build = 24 * max(my * ny, mx * nx)
    kernels = 16 * (my * ny + mx * nx)
    y_first = (my * ny * nx + my * nx * mx) <= (ny * nx * mx + my * ny * mx)
    inter = 16 * (my * nx if y_first else ny * mx)
    return max(build, kernels + inter + 16 * my * mx)


def sep_live(ny, nx, my, mx) -> int:
    """Live bytes at the separable chirp-Z route's peak: the largest
    ``(N_in x L)`` working array, ``L = next_fast_len(N + M - 1)`` per axis."""
    from scipy.fft import next_fast_len
    ly = int(next_fast_len(ny + my - 1))
    lx = int(next_fast_len(nx + mx - 1))
    return 16 * max(ny * lx, my * ly)


def chirp2d_live(ny, nx, my, mx) -> int:
    """Live bytes at the 2-D chirp-Z route's peak: the ``Ly x Lx`` padded
    plane, several copies of which the convolution holds at once."""
    from scipy.fft import next_fast_len
    ly = int(next_fast_len(ny + my - 1))
    lx = int(next_fast_len(nx + mx - 1))
    return 16 * ly * lx


def main(tree, family):
    import numpy as np
    from lumenairy.propagators import fft_infra
    from lumenairy.propagators._bluestein import _auto_selects_direct
    v4lib.anchor(tree)
    fft_infra.SCIPY_FFT_WORKERS = 1
    shapes = (THIN if family == 'thin' else
              SQUARE if family == 'square' else THIN + SQUARE)
    out = {'build': v4lib.build_tag(), 'family': family,
           'python': sys.version.split()[0], 'numpy': np.__version__,
           'load_before': v4lib.load_census(), 'rows': []}
    rng = np.random.default_rng(20260921)
    for (name, ny, nx, my, mx) in shapes:
        E = (rng.standard_normal((ny, nx))
             + 1j * rng.standard_normal((ny, nx))).astype(np.complex128)
        a = alpha_for(ny, nx, my, mx)
        peak = {}
        dig = {}
        for r in ROUTES:
            v4lib.cold()
            gc.collect()
            tracemalloc.start()
            F = run_once(E, a, my, mx, r)
            peak[r] = int(tracemalloc.get_traced_memory()[1])
            tracemalloc.stop()
            dig[r] = v4lib.digest_array(F)
            del F
        cheapest = min(peak, key=lambda k: peak[k])
        rec = {'name': name, 'Ny_in': ny, 'Nx_in': nx, 'My': my, 'Mx': mx,
               'ratio_max': max(my / ny, mx / nx),
               'work_per_kernel_entry': work_per_kernel_entry(ny, nx, my, mx),
               'rule_says_direct': bool(_auto_selects_direct(ny, nx, my, mx)),
               'peak_bytes': peak,
               'peak_MB': {k: v / 1e6 for k, v in peak.items()},
               'cheapest': cheapest,
               'dense_over_cheapest_fallback':
                   peak['dense'] / min(peak['separable'], peak['chirpz2d']),
               'derived_bytes': {
                   'dense': dense_live(ny, nx, my, mx),
                   'separable': sep_live(ny, nx, my, mx),
                   'chirpz2d': chirp2d_live(ny, nx, my, mx)},
               'distinct_route_digests': len(set(dig.values()))}
        rec['derived_cheapest'] = min(rec['derived_bytes'],
                                      key=lambda k: rec['derived_bytes'][k])
        rec['derived_predicts_measured'] = bool(
            rec['derived_cheapest'] == cheapest)
        out['rows'].append(rec)
        print(f"{name:16s} {ny}x{nx}->{my}x{mx} "
              f"w/e={rec['work_per_kernel_entry']:8.2f} "
              f"rule={'D' if rec['rule_says_direct'] else 'c'} "
              f"dense={peak['dense'] / 1e6:9.3f} "
              f"sep={peak['separable'] / 1e6:9.3f} "
              f"chirp={peak['chirpz2d'] / 1e6:10.3f} MB "
              f"cheapest={cheapest:9s} "
              f"derived={rec['derived_cheapest']:9s}", flush=True)
        del E
        gc.collect()
    out['load_after'] = v4lib.load_census()
    n = len(out['rows'])
    out['dense_cheapest_count'] = sum(1 for r in out['rows']
                                      if r['cheapest'] == 'dense')
    out['CAPTURED_AND_NOT_CHEAPEST'] = [
        r['name'] for r in out['rows']
        if r['rule_says_direct'] and r['cheapest'] != 'dense']
    out['REFUSED_AND_NOT_CHEAPEST'] = [
        r['name'] for r in out['rows']
        if not r['rule_says_direct'] and r['cheapest'] != 'dense']
    out['derived_predicts_measured_count'] = sum(
        1 for r in out['rows'] if r['derived_predicts_measured'])
    print(f"dense cheapest at {out['dense_cheapest_count']} of {n}", flush=True)
    print("CAPTURED and NOT cheapest:", out['CAPTURED_AND_NOT_CHEAPEST'],
          flush=True)
    print("REFUSED and not cheapest :", out['REFUSED_AND_NOT_CHEAPEST'],
          flush=True)
    print(f"derived counts predict the measured cheapest at "
          f"{out['derived_predicts_measured_count']} of {n}", flush=True)
    v4lib.write_json(out, os.path.join(
        HERE, f"r2_memcensus_{family}_{v4lib.short_tag()}.json"))


def _arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


if __name__ == '__main__':
    main(sys.argv[1], _arg('--family', 'all'))
