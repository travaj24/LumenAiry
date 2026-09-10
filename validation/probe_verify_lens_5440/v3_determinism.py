"""V3 -- the D14/D15 determinism contract on the BANDED ray-density path.

The whole-grid traced fit was made thread-count independent in 5.41/5.42
(BUILD_DETERMINISTIC_TRACED_FIT_2026_08_23, BUILD_DETERMINISTIC_CARRIER_FIT_
2026_08_22).  A band loop that re-ordered any reduction would regress it, so
this hashes the SAME banded call at OPENBLAS_NUM_THREADS 1 / 2 / 4 -- the
env var is read at process start, so each arm is its own subprocess.

Usage:  python v3_determinism.py <out.json>            (driver)
        python v3_determinism.py --worker <tag>        (one arm; internal)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fix                                                    # noqa: E402

N, DX, W, SUB = 384, 13e-6, 1.0e-3, 4

CASES = {
    'rd_imap_band32': dict(rows=32, inverse_map=None,
                           amplitude_model='ray_density'),
    'rd_imap_band7': dict(rows=7, inverse_map=None,
                          amplitude_model='ray_density'),
    'rd_imap_whole': dict(rows=0, inverse_map=None,
                          amplitude_model='ray_density'),
    'rd_newton_band32': dict(rows=32, inverse_map=False,
                             amplitude_model='ray_density'),
    'rd_remapfull_band32': dict(rows=32, inverse_map=None,
                                amplitude_model='ray_density',
                                preserve_input_phase='remap',
                                remap_sampling='full'),
    'screen_band32': dict(rows=32, inverse_map=None,
                          amplitude_model='screen'),
}


def worker():
    la = _fix.banner()
    E = _fix.gauss(N, DX, W)
    out = {}
    for name, spec in CASES.items():
        spec = dict(spec)
        rows = spec.pop('rows')
        inv = spec.pop('inverse_map')
        kw = dict(prescription=_fix.presc_singlet(ap=4.4e-3),
                  wavelength=_fix.WL, dx=DX, ray_subsample=SUB, n_workers=1,
                  on_undersample='silent', on_noncollimated='off',
                  on_aperture_beam='silent', parallel_amp=False, **spec)
        if inv is not None:
            kw['inverse_map'] = inv
        f, rec, msgs = _fix.run(la, E, kw, rows)
        out[name] = {'hash': _fix.h(f),
                     'sum_abs2': float(np.sum(np.abs(f) ** 2)),
                     'engaged': bool(rec.get('engaged', False))}
        print(f"  {name:24s} {out[name]['hash']}", flush=True)
    print('RESULT_JSON ' + json.dumps(out))


def main():
    if '--worker' in sys.argv:
        worker()
        return
    res = {}
    for nt in (1, 2, 4):
        env = dict(os.environ)
        for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                  'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            env[v] = str(nt)
        print(f'--- threads={nt} ---', flush=True)
        p = subprocess.run([sys.executable, os.path.abspath(__file__),
                            '--worker'], env=env, capture_output=True,
                           text=True)
        if p.returncode:
            print(p.stdout[-2000:], p.stderr[-2000:])
            raise SystemExit(f'worker failed at threads={nt}')
        line = [ln for ln in p.stdout.splitlines()
                if ln.startswith('RESULT_JSON ')][0]
        res[nt] = json.loads(line[len('RESULT_JSON '):])
        print(p.stdout.split('RESULT_JSON')[0].strip(), flush=True)
    verdict = {}
    for name in CASES:
        hs = {nt: res[nt][name]['hash'] for nt in res}
        verdict[name] = {'hashes': hs,
                         'identical': len(set(hs.values())) == 1}
        print(f"{name:24s} identical={verdict[name]['identical']} {hs}")
    with open(sys.argv[1], 'w') as fh:
        json.dump({'by_threads': res, 'verdict': verdict}, fh, indent=1)
    print('wrote', sys.argv[1])


if __name__ == '__main__':
    main()
