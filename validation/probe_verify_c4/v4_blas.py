"""VERIFY-WP-C4 claim 7 -- the BLAS sweep: the dense route's BITS move with the
kernel, the DECISION does not, and the distance to an exactly-reduced reference
stays inside the derived bar in every cell.

One CHILD PROCESS per cell, because ``OPENBLAS_CORETYPE`` and
``OPENBLAS_NUM_THREADS`` are read when the BLAS is loaded and cannot be
changed afterwards.  Each child prints one JSON line; the driver collects them.

    PYTHONPATH=<tree> python v4_blas.py <tree> --cell        # one cell
    PYTHONPATH=<tree> python v4_blas.py <tree> --drive       # spawns them all
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v4lib  # noqa: E402
from v4_accuracy import GENERIC, derived_bars, exact_reference  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CORETYPES = ('HASWELL', 'NEHALEM', 'SANDYBRIDGE', 'KATMAI')
THREADS = ('1', '4')
SHAPES = [(96, 96, 3, 3), (128, 128, 4, 4), (64, 64, 8, 8)]


def cell(tree):
    import numpy as np
    v4lib.anchor(tree)
    from lumenairy.propagators._bluestein import (_auto_selects_direct,
                                                  _bluestein_2d)
    from lumenairy.propagators.fft_infra import _fft2, _ifft2
    rec = {'coretype': os.environ.get('OPENBLAS_CORETYPE', ''),
           'threads': os.environ.get('OPENBLAS_NUM_THREADS', ''),
           'build': v4lib.build_tag(), 'rows': []}
    try:
        import threadpoolctl
        rec['threadpool'] = [
            {k: d.get(k) for k in ('user_api', 'internal_api',
                                   'architecture', 'num_threads')}
            for d in threadpoolctl.threadpool_info()]
    except Exception as exc:                              # noqa: BLE001
        rec['threadpool'] = f"{type(exc).__name__}: {exc}"
    rng = np.random.default_rng(90210)
    for (ny, nx, my, mx) in SHAPES:
        E = (rng.standard_normal((ny, nx))
             + 1j * rng.standard_normal((ny, nx))).astype(np.complex128)
        a = GENERIC * 1.0e3 / float(max(ny, nx, my, mx)) ** 2
        kw = dict(sign=-1, xp=np, fft2=_fft2, ifft2=_ifft2)
        auto = _bluestein_2d(E, a, a, my, mx, **kw)
        dense = _bluestein_2d(E, a, a, my, mx, method='direct', **kw)
        chirp = _bluestein_2d(E, a, a, my, mx, method='bluestein', **kw)
        ref = exact_reference(np, E, a, a, my, mx, -1)
        bars = derived_bars(np, ny, my, a, float(np.abs(E).sum()))
        says = bool(_auto_selects_direct(ny, nx, my, mx))
        rec['rows'].append({
            'shape': [ny, nx, my, mx], 'rule_says_direct': says,
            'dense_digest': v4lib.digest_array(dense),
            'chirp_digest': v4lib.digest_array(chirp),
            'auto_digest': v4lib.digest_array(auto),
            'auto_is_dense': bool(v4lib.digest_array(auto)
                                  == v4lib.digest_array(dense)),
            'auto_is_chirp': bool(v4lib.digest_array(auto)
                                  == v4lib.digest_array(chirp)),
            'dense_maxabs_vs_ref': float(np.max(np.abs(dense - ref))),
            'chirp_maxabs_vs_ref': float(np.max(np.abs(chirp - ref))),
            'bar_dense': bars['bar_dense'], 'bar_chirp': bars['bar_chirp'],
            'dense_inside_bar': bool(np.max(np.abs(dense - ref))
                                     <= bars['bar_dense']),
            'chirp_inside_bar': bool(np.max(np.abs(chirp - ref))
                                     <= bars['bar_chirp']),
        })
    print(json.dumps(rec))


def drive(tree):
    v4lib.anchor(tree)
    out = {'build': v4lib.build_tag(), 'cells': []}
    for ct in CORETYPES:
        for th in THREADS:
            env = dict(os.environ)
            env.update(OMP_NUM_THREADS=th, MKL_NUM_THREADS=th,
                       OPENBLAS_NUM_THREADS=th, OPENBLAS_CORETYPE=ct,
                       PYTHONPATH=tree)
            p = subprocess.run(
                [sys.executable, os.path.abspath(__file__), tree, '--cell'],
                capture_output=True, text=True, env=env,
                cwd=os.path.dirname(os.path.abspath(__file__)))
            lines = [ln for ln in p.stdout.splitlines() if ln.startswith('{')]
            if not lines:
                out['cells'].append({'coretype': ct, 'threads': th,
                                     'failed': p.stderr[-400:]})
                print(f"{ct:12s} th={th}  FAILED: {p.stderr[-160:]}",
                      flush=True)
                continue
            rec = json.loads(lines[-1])
            out['cells'].append(rec)
            arch = ''
            if isinstance(rec.get('threadpool'), list):
                arch = ','.join(str(d.get('architecture'))
                                for d in rec['threadpool'])
            ok = all(r['dense_inside_bar'] and r['chirp_inside_bar']
                     for r in rec['rows'])
            route_ok = all(r['auto_is_dense'] == r['rule_says_direct']
                           for r in rec['rows'])
            print(f"{ct:12s} th={th}  arch={arch:24s} "
                  f"dense_digests="
                  f"{[r['dense_digest'][:8] for r in rec['rows']]} "
                  f"in_bars={ok} route_follows_rule={route_ok}", flush=True)
    good = [c for c in out['cells'] if 'rows' in c]
    per_shape = {}
    for c in good:
        for r in c['rows']:
            per_shape.setdefault(tuple(r['shape']), {
                'dense': set(), 'chirp': set(), 'route': set()})
            per_shape[tuple(r['shape'])]['dense'].add(r['dense_digest'])
            per_shape[tuple(r['shape'])]['chirp'].add(r['chirp_digest'])
            per_shape[tuple(r['shape'])]['route'].add(
                'dense' if r['auto_is_dense'] else
                'chirp' if r['auto_is_chirp'] else 'NEITHER')
    out['summary'] = {
        'cells_run': len(good), 'cells_total': len(out['cells']),
        'per_shape': {str(k): {'distinct_dense_digests': len(v['dense']),
                               'distinct_chirp_digests': len(v['chirp']),
                               'distinct_routes': sorted(v['route'])}
                      for k, v in per_shape.items()},
        'all_inside_bars': all(r['dense_inside_bar'] and r['chirp_inside_bar']
                               for c in good for r in c['rows']),
        'route_follows_rule_everywhere': all(
            r['auto_is_dense'] == r['rule_says_direct']
            for c in good for r in c['rows']),
    }
    print('SUMMARY', json.dumps(out['summary'], indent=1), flush=True)
    v4lib.write_json(out, os.path.join(
        HERE, f"v4_blas_{v4lib.short_tag()}.json"))


if __name__ == '__main__':
    if '--cell' in sys.argv:
        cell(sys.argv[1])
    else:
        drive(sys.argv[1])
