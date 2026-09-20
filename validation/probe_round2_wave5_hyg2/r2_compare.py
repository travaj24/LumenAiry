"""Compare two round-2 digest maps key by key.

Prints, and writes, the four counts that make a bit-identity claim a claim:
identical, differing, only-in-base, only-in-branch.  A key that is missing on
one side is reported as its own category rather than silently dropped -- a
probe that stopped producing a key would otherwise read as a pass.

    python r2_compare.py base.json branch.json out.json
"""
from __future__ import annotations

import json
import sys


def main(a_path, b_path, out_path):
    with open(a_path, encoding='utf-8') as fh:
        a = json.load(fh)
    with open(b_path, encoding='utf-8') as fh:
        b = json.load(fh)
    common = sorted(set(a) & set(b))
    same = [k for k in common if a[k] == b[k]]
    diff = [k for k in common if a[k] != b[k]]
    res = {
        'keys_base': len(a), 'keys_branch': len(b),
        'identical': len(same), 'differing': len(diff),
        'only_base': sorted(set(a) - set(b)),
        'only_branch': sorted(set(b) - set(a)),
        'differing_keys': diff,
    }
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    print(f"keys base={len(a)} branch={len(b)}  identical={len(same)}  "
          f"differing={len(diff)}  only-base={len(res['only_base'])}  "
          f"only-branch={len(res['only_branch'])}")
    for k in diff[:20]:
        print('  DIFFER  ' + k)
    for k in res['only_base'][:10]:
        print('  ONLY-BASE  ' + k)
    for k in res['only_branch'][:10]:
        print('  ONLY-BRANCH  ' + k)
    return 0 if (diff or res['only_base'] or res['only_branch']) == [] \
        and not diff and not res['only_base'] and not res['only_branch'] else 1


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1], sys.argv[2], sys.argv[3]))
