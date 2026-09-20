"""v1: compare the base / branch digest maps written by ``v1_bitid.py``."""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
out = {}
for build in ('win', 'wsl'):
    b = json.load(open(os.path.join(HERE, f'v1_bitid_{build}_base.json')))
    n = json.load(open(os.path.join(HERE, f'v1_bitid_{build}_branch.json')))
    b.pop('__build__', None)
    n.pop('__build__', None)
    shared = sorted(set(b) & set(n))
    same = [k for k in shared if b[k] == n[k]]
    diff = [k for k in shared if b[k] != n[k]]
    row = {
        'keys': len(set(b) | set(n)),
        'shared': len(shared),
        'identical': len(same),
        'differing': len(diff),
        'only_base': sorted(set(b) - set(n)),
        'only_branch': sorted(set(n) - set(b)),
        'differing_keys': diff,
        'LEG_total': len([k for k in shared if k.startswith('LEG::')]),
        'LEG_identical': len([k for k in same if k.startswith('LEG::')]),
        'LEG_differing': [k for k in diff if k.startswith('LEG::')],
        'NEW_total': len([k for k in shared if k.startswith('NEW::')]),
        'NEW_identical': len([k for k in same if k.startswith('NEW::')]),
        'NEW_differing': len([k for k in diff if k.startswith('NEW::')]),
    }
    # cross-key: does branch's method='auto' reproduce the BASE tree's
    # keyword-free answer, byte for byte?
    xk = {}
    for tag in ('fres', 'frau', 'asm'):
        xk[f'base_LEGtwin_vs_branch_AUTO.{tag}'] = (
            b[f'LEG::autotwin.{tag}'] == n[f'NEW::prop.{tag}.auto'])
        xk[f'branch_LEGtwin_vs_branch_AUTO.{tag}'] = (
            n[f'LEG::autotwin.{tag}'] == n[f'NEW::prop.{tag}.auto'])
        xk[f'base_LEGtwin_vs_branch_DIRECT.{tag}'] = (
            b[f'LEG::autotwin.{tag}'] == n[f'NEW::prop.{tag}.direct'])
    row['crosskey'] = xk
    out[build] = row
    print(build, json.dumps({k: v for k, v in row.items()
                             if k not in ('differing_keys',)}, indent=1))
json.dump(out, open(os.path.join(HERE, 'v1_bitid_compare.json'), 'w'),
          indent=1, sort_keys=True)
print('wrote v1_bitid_compare.json', file=sys.stderr)
