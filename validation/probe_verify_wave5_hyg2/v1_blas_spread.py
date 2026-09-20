"""v1: the exact inter-rung VALUE spread of ``_direct_matrix_2d``'s output.

Reads the per-rung ``.npz`` dumps and reports, for each fixture and each
build, the number of distinct byte images and ``max |F_i - F_j|`` over every
pair of rungs -- the headline durability number for the BLAS kernel ladder.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import sys

import numpy as np

DUMP = sys.argv[1] if len(sys.argv) > 1 else (
    r'C:\Users\Tesla\AppData\Local\Temp\claude'
    r'\C--Users-Tesla\372a2d1f-acbe-4b57-a148-eeae3fe1d729'
    r'\scratchpad\blasdump')
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'v1_blas_spread.json')

files = {f[:-4]: os.path.join(DUMP, f)
         for f in os.listdir(DUMP) if f.endswith('.npz')}
data = {k: dict(np.load(v)) for k, v in files.items()}
fixtures = sorted(next(iter(data.values())))
res = {}
for build in ('win', 'wsl'):
    rungs = sorted(k for k in data if k.startswith(build + '_'))
    if not rungs:
        continue
    row = {}
    for fx in fixtures:
        shas = {r: hashlib.sha256(
            np.ascontiguousarray(data[r][fx]).tobytes()).hexdigest()
            for r in rungs}
        worst = 0.0
        worst_pair = None
        for a, b in itertools.combinations(rungs, 2):
            d = float(np.max(np.abs(data[a][fx] - data[b][fx])))
            if d > worst:
                worst, worst_pair = d, (a, b)
        peak = float(np.max(np.abs(data[rungs[0]][fx])))
        row[fx] = {
            'n_rungs': len(rungs),
            'distinct_digests': len(set(shas.values())),
            'digest_groups': {s: sorted(r for r in rungs if shas[r] == s)
                              for s in sorted(set(shas.values()))},
            'max_abs_spread': worst,
            'worst_pair': worst_pair,
            'peak_abs': peak,
            'relative_spread': worst / peak,
        }
    res[build] = row

# cross-build: does the same rung give the same bytes on the two builds?
cross = {}
for fx in fixtures:
    same = {}
    for ct in ('HASWELL', 'NEHALEM', 'KATMAI', 'SANDYBRIDGE'):
        for nt in ('1', '4'):
            a, b = f'win_{ct}_t{nt}', f'wsl_{ct}_t{nt}'
            if a in data and b in data:
                same[f'{ct}/t{nt}'] = {
                    'bit_equal': bool(np.array_equal(data[a][fx],
                                                     data[b][fx])),
                    'max_abs': float(np.max(np.abs(data[a][fx]
                                                   - data[b][fx])))}
    cross[fx] = same
res['cross_build_same_rung'] = cross

json.dump(res, open(OUT, 'w'), indent=1, sort_keys=True, default=str)
for build in ('win', 'wsl'):
    if build not in res:
        continue
    print('=====', build)
    for fx, r in res[build].items():
        print(f"  {fx:10s} rungs={r['n_rungs']} distinct={r['distinct_digests']}"
              f" max|Fi-Fj|={r['max_abs_spread']:.3e} peak={r['peak_abs']:.4g}"
              f" rel={r['relative_spread']:.3e}")
        for s, g in r['digest_groups'].items():
            print(f"       {s[:12]} <- {[x.replace(build + '_', '') for x in g]}")
print('cross-build same-rung bit equality:')
for fx, s in cross.items():
    print(' ', fx, {k: v['bit_equal'] for k, v in s.items()})
print('wrote', OUT)
