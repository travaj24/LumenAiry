"""Compare two v2_bitid digest maps key by key.

    python v2_bitid_compare.py base.json branch.json out.json
"""
from __future__ import annotations

import json
import sys


def main():
    a = json.load(open(sys.argv[1], encoding='utf-8'))
    b = json.load(open(sys.argv[2], encoding='utf-8'))
    ka, kb = set(a), set(b)
    both = sorted(ka & kb)
    diff = [k for k in both if a[k] != b[k]]
    res = {
        'base_file': sys.argv[1], 'branch_file': sys.argv[2],
        'keys_base': len(ka), 'keys_branch': len(kb),
        'compared': len(both), 'identical': len(both) - len(diff),
        'differing': len(diff), 'differing_keys': diff,
        'only_base': sorted(ka - kb), 'only_branch': sorted(kb - ka),
        'base_build': a.get('meta.build'), 'branch_build': b.get('meta.build'),
    }
    json.dump(res, open(sys.argv[3], 'w', encoding='utf-8'), indent=1,
              sort_keys=True)
    print(json.dumps({k: v for k, v in res.items()
                      if k != 'differing_keys'}, indent=1))
    if diff:
        print('DIFFERING:', diff)


if __name__ == '__main__':
    main()
