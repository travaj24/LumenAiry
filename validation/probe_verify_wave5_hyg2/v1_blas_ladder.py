"""v1: aggregate the BLAS-kernel-ladder runs written by ``v1_blas.py``."""
from __future__ import annotations

import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CTS = ('HASWELL', 'NEHALEM', 'KATMAI', 'SANDYBRIDGE')
NTS = ('1', '4')
agg = {}
for build in ('win', 'wsl'):
    runs = {}
    for ct in CTS:
        for nt in NTS:
            p = os.path.join(HERE, f'v1_blas_{build}_{ct}_t{nt}.json')
            if os.path.exists(p):
                runs[f'{ct}/t{nt}'] = json.load(open(p))
    if not runs:
        continue
    arch = {k: (v['blas']['threadpool'][0].get('architecture')
                if isinstance(v['blas']['threadpool'], list)
                and v['blas']['threadpool'] else None)
            for k, v in runs.items()}
    nthr = {k: (v['blas']['threadpool'][0].get('num_threads')
                if isinstance(v['blas']['threadpool'], list)
                and v['blas']['threadpool'] else None)
            for k, v in runs.items()}
    keys = sorted(next(iter(runs.values()))['rows'])
    table = {}
    for key in keys:
        row = {}
        for field in sorted(next(iter(runs.values()))['rows'][key]):
            vals = {k: runs[k]['rows'][key][field] for k in runs}
            if field.startswith('sha'):
                row[field] = {'distinct': len(set(vals.values())),
                              'by_rung': vals}
            elif isinstance(next(iter(vals.values())), (int, float)) \
                    and not isinstance(next(iter(vals.values())), bool):
                lo = min(vals.values())
                hi = max(vals.values())
                row[field] = {'min': lo, 'max': hi, 'spread': hi - lo,
                              'by_rung': vals}
            else:
                row[field] = {'distinct': len(set(map(str, vals.values()))),
                              'by_rung': vals}
        table[key] = row
    agg[build] = {'architecture_reported': arch,
                  'num_threads_reported': nthr,
                  'rungs': sorted(runs), 'table': table}

json.dump(agg, open(os.path.join(HERE, 'v1_blas_ladder.json'), 'w'),
          indent=1, sort_keys=True, default=str)

for build, d in agg.items():
    print('=' * 70)
    print(build, 'architecture:', d['architecture_reported'])
    print(build, 'num_threads :', d['num_threads_reported'])
    for key, row in d['table'].items():
        shas = {f: v for f, v in row.items() if f.startswith('sha')}
        nd = {f: v['distinct'] for f, v in shas.items()}
        extra = []
        for f, v in row.items():
            if 'spread' in v:
                extra.append(f"{f}=[{v['min']:.6e}..{v['max']:.6e}] "
                             f"spread={v['spread']:.3e}")
            elif not f.startswith('sha'):
                extra.append(f"{f}:distinct={v['distinct']}")
        print(f"  {key}: distinct_digests={nd}")
        for e in extra:
            print(f"      {e}")
print('wrote v1_blas_ladder.json')
