"""Compare two :mod:`v4_bitid` digest maps and CHECK the split against the
rule, key by key.

    python v4_bitid_compare.py BASE.json BRANCH.json [OUT.json]
"""
from __future__ import annotations

import json
import sys


def main(base_p, branch_p, out_p=None):
    base = json.load(open(base_p, encoding='cp1252'))
    branch = json.load(open(branch_p, encoding='cp1252'))
    bk, rk = base['keys'], branch['keys']
    assert set(bk) == set(rk), sorted(set(bk) ^ set(rk))[:10]
    rule = branch['rule_says']
    groups = {}
    for k in sorted(bk):
        grp = k.split('/')[0]
        tag = k.split('/')[2]
        same = bk[k] == rk[k]
        g = groups.setdefault(grp, {'identical': 0, 'differ': 0,
                                    'agrees_with_rule': 0,
                                    'disagrees': [], 'raised': 0})
        if str(bk[k]).startswith('RAISED') or str(rk[k]).startswith('RAISED'):
            g['raised'] += 1
        if same:
            g['identical'] += 1
        else:
            g['differ'] += 1
        if grp == 'nokw':
            # a key differs base-to-branch exactly when the rule says 'direct'
            if (not same) == bool(rule[tag]):
                g['agrees_with_rule'] += 1
            else:
                g['disagrees'].append(k)
    out = {'base': base_p, 'branch': branch_p, 'build': branch['build'],
           'groups': groups}
    for grp, g in sorted(groups.items()):
        n = g['identical'] + g['differ']
        line = (f"{grp:8s} identical {g['identical']:4d}/{n:<4d} "
                f"differ {g['differ']:4d}  raised {g['raised']}")
        if grp == 'nokw':
            line += (f"  split agrees with the rule "
                     f"{g['agrees_with_rule']}/{n}  "
                     f"disagreements {len(g['disagrees'])}")
        print(line)
    if out_p:
        with open(out_p, 'w', encoding='cp1252', errors='replace') as fh:
            json.dump(out, fh, indent=1, sort_keys=True)
        print('[wrote]', out_p)


if __name__ == '__main__':
    main(*sys.argv[1:])
