"""VERIFY-WP-B12 -- cross-build agreement of every probe reading.

Walks the two JSON files a probe wrote (``*_win32_314.json`` and
``*_linux_312.json``), pairs every numeric leaf by its path, and reports the
largest absolute and relative disagreement plus every boolean/string leaf that
differs.  Nothing in the report may sit inside that spread.
"""
from __future__ import annotations

import json
import os

SKIP = ('env', 'seconds', 'cost', 'wall')


def leaves(obj, path=''):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if any(s in str(k) for s in SKIP):
                continue
            yield from leaves(v, f'{path}/{k}')
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from leaves(v, f'{path}[{i}]')
    else:
        yield path, obj


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    names = sorted({f.rsplit('_win32_314.json', 1)[0]
                    for f in os.listdir(here) if f.endswith('_win32_314.json')})
    grand = 0.0
    for nm in names:
        a = os.path.join(here, f'{nm}_win32_314.json')
        b = os.path.join(here, f'{nm}_linux_312.json')
        if not os.path.exists(b):
            print(f'{nm}: no linux twin -- skipped')
            continue
        A = dict(leaves(json.load(open(a, encoding='cp1252'))))
        B = dict(leaves(json.load(open(b, encoding='cp1252'))))
        keys = sorted(set(A) & set(B))
        worst_abs = worst_rel = 0.0
        wa = wr = ''
        mism = []
        for k in keys:
            x, y = A[k], B[k]
            if isinstance(x, bool) or isinstance(y, bool) \
                    or isinstance(x, str) or isinstance(y, str):
                if x != y:
                    mism.append((k, x, y))
                continue
            if x is None or y is None:
                continue
            d = abs(float(x) - float(y))
            r = d / max(abs(float(y)), 1e-300)
            if d > worst_abs:
                worst_abs, wa = d, k
            if r > worst_rel and abs(float(y)) > 1e-12:
                worst_rel, wr = r, k
        only = sorted(set(A) ^ set(B))
        print(f'{nm}: {len(keys)} paired readings; worst |d| = {worst_abs:.3e} '
              f'at {wa}; worst rel = {worst_rel:.3e} at {wr}')
        for k, x, y in mism:
            print(f'   MISMATCH {k}: win={x!r} linux={y!r}')
        if only:
            print(f'   present on one build only: {only[:6]}'
                  f'{" ..." if len(only) > 6 else ""}')
        grand = max(grand, worst_rel)
    print(f'GRAND worst relative disagreement across all probes: {grand:.3e}')


if __name__ == '__main__':
    main()
