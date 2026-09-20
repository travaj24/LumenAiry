"""VERIFY-WP-C4 claim 2b -- PEAK RSS from the OS, one child process per
(shape, route), beside the byte counts DERIVED from the code.

``tracemalloc`` charges Python allocations; the operating system's peak working
set is what a caller's box actually has to hold, and it includes whatever the
FFT library allocates outside the Python allocator.  One route per PROCESS, so
the peak is that route's and not the maximum over a sequence.

The DERIVED counts come from reading the two routes:

* dense (``_direct_matrix_2d``) -- ``_kernel`` builds a float64 ``t`` of
  ``M x N`` and then a complex128 ``W`` of ``M x N`` before the cast, so the
  build peak is ``24*M*N`` per axis; the two kernels are ``16*(My*Ny + Mx*Nx)``;
  the association order is chosen from the shapes, and the intermediate is
  ``16*My*Nx`` (y first) or ``16*Ny*Mx`` (x first); the output is ``16*My*Mx``.
* chirp-Z 2-D -- ``L = next_fast_len(N + M - 1)`` per axis and the working
  arrays are ``Ly x Lx``: ``g_pad``, ``G_FFT``, ``h_2d``, ``H_FFT`` and
  ``CONV``, so at least ``3 * 16 * Ly * Lx`` live at once.
* separable -- the largest array is ``(N_in x L)`` per pass.

    PYTHONPATH=<tree> python v4_rss.py <tree> --shape NY,NX,MY,MX --route NAME
    PYTHONPATH=<tree> python v4_rss.py <tree> --drive            # spawns all
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v4lib  # noqa: E402
from v4_ladder import alpha_for, run_once  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROUTES = ('chirpz2d', 'separable', 'dense')
SHAPES = [(2048, 2048, 64, 64), (1024, 1024, 32, 32), (2048, 2048, 16, 16),
          (1024, 1024, 128, 128), (2048, 512, 64, 16), (512, 512, 16, 16),
          (1024, 1024, 512, 512)]


def derived(np, ny, nx, my, mx):
    from scipy.fft import next_fast_len
    Ly = int(next_fast_len(ny + my - 1))
    Lx = int(next_fast_len(nx + mx - 1))
    cost_y_first = my * ny * nx + my * nx * mx
    cost_x_first = ny * nx * mx + my * ny * mx
    y_first = cost_y_first <= cost_x_first
    inter = 16 * (my * nx if y_first else ny * mx)
    return {
        'L': [Ly, Lx], 'y_first': bool(y_first),
        'dense_kernels_bytes': 16 * (my * ny + mx * nx),
        'dense_kernel_build_peak_bytes': 24 * max(my * ny, mx * nx),
        'dense_intermediate_bytes': inter,
        'dense_out_bytes': 16 * my * mx,
        'dense_total_live_bytes': 16 * (my * ny + mx * nx) + inter
                                  + 16 * my * mx,
        'chirpz2d_one_L2_bytes': 16 * Ly * Lx,
        'chirpz2d_min_live_bytes': 3 * 16 * Ly * Lx,
        'separable_largest_bytes': 16 * max(ny * Lx, my * Ly),
        'dense_flops': min(cost_y_first, cost_x_first),
        'dense_kernel_entries': my * ny + mx * nx,
        'work_per_kernel_entry': min(cost_y_first, cost_x_first)
                                 / float(my * ny + mx * nx),
    }


def one(tree, shape, route):
    import numpy as np
    v4lib.anchor(tree)
    ny, nx, my, mx = shape
    before = v4lib.peak_rss_bytes()
    rng = np.random.default_rng(31337)
    E = (rng.standard_normal((ny, nx))
         + 1j * rng.standard_normal((ny, nx))).astype(np.complex128)
    after_alloc = v4lib.peak_rss_bytes()
    a = alpha_for(ny, nx, my, mx)
    v4lib.cold()
    F = run_once(E, a, my, mx, route)
    peak = v4lib.peak_rss_bytes()
    print(json.dumps({'shape': list(shape), 'route': route,
                      'rss_start': before, 'rss_after_input': after_alloc,
                      'rss_peak': peak,
                      'route_delta': peak - after_alloc,
                      'input_bytes': int(E.nbytes),
                      'digest': v4lib.digest_array(F),
                      'derived': derived(np, ny, nx, my, mx)}))


def drive(tree):
    import numpy as np
    v4lib.anchor(tree)
    out = {'build': v4lib.build_tag(), 'rows': [],
           'load': v4lib.load_census()}
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', PYTHONPATH=tree)
    for shape in SHAPES:
        rec = {'shape': list(shape)}
        for route in ROUTES:
            p = subprocess.run(
                [sys.executable, os.path.abspath(__file__), tree,
                 '--shape', ','.join(str(v) for v in shape),
                 '--route', route],
                capture_output=True, text=True, env=env,
                cwd=os.path.dirname(os.path.abspath(__file__)))
            line = [ln for ln in p.stdout.splitlines()
                    if ln.startswith('{')]
            if not line:
                rec[route] = {'failed': p.stderr[-400:]}
                continue
            rec[route] = json.loads(line[-1])
        d = rec.get('dense', {}).get('derived')
        if d:
            rec['derived'] = d
        peaks = {r: rec[r].get('route_delta') for r in ROUTES
                 if 'route_delta' in rec.get(r, {})}
        if len(peaks) == 3:
            rec['rss_route_delta'] = peaks
            rec['dense_cheapest_rss'] = bool(
                peaks['dense'] <= min(peaks['separable'], peaks['chirpz2d']))
            rec['dense_smaller_by_rss'] = (
                min(peaks['separable'], peaks['chirpz2d'])
                / max(peaks['dense'], 1))
            print(f"{tuple(shape)} RSS delta MB dense="
                  f"{peaks['dense']/1e6:9.1f} sep={peaks['separable']/1e6:9.1f}"
                  f" chirp={peaks['chirpz2d']/1e6:9.1f}  "
                  f"derived dense_live={d['dense_total_live_bytes']/1e6:8.2f} "
                  f"chirp_min_live={d['chirpz2d_min_live_bytes']/1e6:9.2f} "
                  f"work/entry={d['work_per_kernel_entry']:8.1f}", flush=True)
        out['rows'].append(rec)
    v4lib.write_json(out, os.path.join(
        HERE, f"v4_rss_{v4lib.short_tag()}.json"))


def _arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


if __name__ == '__main__':
    tree = sys.argv[1]
    if '--drive' in sys.argv:
        drive(tree)
    else:
        one(tree, tuple(int(v) for v in _arg('--shape').split(',')),
            _arg('--route'))
