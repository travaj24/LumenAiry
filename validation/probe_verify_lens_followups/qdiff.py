"""Structural diff of two probe JSONs, ignoring wall-clock and arm metadata.

Usage: python qdiff.py <a.json> <b.json> [--allow KEYFRAGMENT ...]

Prints every leaf whose value differs, the count of leaves compared, and
exits 1 on any difference that is not covered by ``--allow``.
"""
from __future__ import annotations

import argparse
import json
import sys

IGNORE_LEAF = ('secs', '_arm', 'wall', 'seconds', 'free_gb', 'elapsed')


def leaves(o, path=''):
    if isinstance(o, dict):
        for k in sorted(o):
            if k in IGNORE_LEAF:
                continue
            yield from leaves(o[k], f'{path}.{k}' if path else k)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from leaves(v, f'{path}[{i}]')
    else:
        yield path, o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('a')
    ap.add_argument('b')
    ap.add_argument('--allow', nargs='*', default=[])
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()
    A = dict(leaves(json.load(open(args.a, encoding='cp1252'))))
    B = dict(leaves(json.load(open(args.b, encoding='cp1252'))))
    keys = sorted(set(A) | set(B))
    bad, allowed, n = [], [], 0
    for k in keys:
        if k not in A or k not in B:
            (allowed if any(f in k for f in args.allow) else bad).append(
                (k, A.get(k, '<MISSING>'), B.get(k, '<MISSING>')))
            continue
        n += 1
        if A[k] != B[k]:
            (allowed if any(f in k for f in args.allow) else bad).append(
                (k, A[k], B[k]))
    print(f"leaves compared: {n}   differing: {len(bad)}   "
          f"allowed: {len(allowed)}")
    for k, a, b in bad[:60]:
        print(f"  DIFF {k}\n    A={a!r}\n    B={b!r}")
    if not args.quiet:
        for k, a, b in allowed[:40]:
            print(f"  (allowed) {k}: A={str(a)[:60]!r} B={str(b)[:60]!r}")
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
