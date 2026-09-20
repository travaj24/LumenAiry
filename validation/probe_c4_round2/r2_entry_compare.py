"""WP-C4 round 2, V-C4-D2 -- the base-vs-branch verdict, per entry point.

Reads ``r2_entry_base_<tag>.json`` and ``r2_entry_branch_<tag>.json`` and
answers three questions per entry point, from the digests alone:

* ``moves_at_captured``  -- the no-keyword call differs between ``49ddf4bd``
  and this branch at a shape the rule captures.  It MUST, or that entry point
  never reached the rule and the row is vacuous.
* ``identical_at_refused`` -- the no-keyword call is byte-identical above the
  boundary.  It MUST be, or the flip is wider than the rule says.
* ``way_back`` -- WHICH ``mft_method`` spelling reproduces the base bytes
  exactly.  ``None`` here is the defect V-C4-D2 reported.

    python r2_entry_compare.py <win|wsl>
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main(tag):
    base = json.load(open(os.path.join(HERE, f"r2_entry_base_{tag}.json"),
                          encoding='cp1252'))
    branch = json.load(open(os.path.join(HERE, f"r2_entry_branch_{tag}.json"),
                            encoding='cp1252'))
    names = sorted({k.split('|')[0] for k in base['keys']})
    out = {'tag': tag, 'base_build': base['build'],
           'branch_build': branch['build'],
           'base_version': base.get('lumenairy_version'),
           'branch_version': branch.get('lumenairy_version'),
           'rows': []}
    for name in names:
        b_cap = base['keys'].get(f"{name}|captured")
        b_ref = base['keys'].get(f"{name}|refused")
        n_cap = branch['keys'].get(f"{name}|captured")
        n_ref = branch['keys'].get(f"{name}|refused")
        ways = [s for s in ('bluestein', 'separable')
                if branch['keys'].get(f"{name}|captured|{s}") == b_cap]
        row = {'entry_point': name,
               'moves_at_captured': bool(b_cap != n_cap),
               'identical_at_refused': bool(b_ref == n_ref),
               'way_back_spellings': ways,
               'way_back': ways[0] if ways else None,
               'base_captured': b_cap, 'branch_captured': n_cap,
               'base_refused': b_ref, 'branch_refused': n_ref}
        out['rows'].append(row)
        print(f"{name:42s} moves={row['moves_at_captured']!s:5s} "
              f"refused_identical={row['identical_at_refused']!s:5s} "
              f"way_back={row['way_back']}", flush=True)
    out['n_entry_points'] = len(out['rows'])
    out['n_moving'] = sum(1 for r in out['rows'] if r['moves_at_captured'])
    out['n_identical_at_refused'] = sum(
        1 for r in out['rows'] if r['identical_at_refused'])
    out['n_with_a_way_back'] = sum(1 for r in out['rows'] if r['way_back'])
    out['WITHOUT_A_WAY_BACK'] = [r['entry_point'] for r in out['rows']
                                 if not r['way_back']]
    print(f"{out['n_entry_points']} entry points: "
          f"{out['n_moving']} move at the captured shape, "
          f"{out['n_identical_at_refused']} identical at the refused one, "
          f"{out['n_with_a_way_back']} have a ONE-KEYWORD way back",
          flush=True)
    print("WITHOUT A WAY BACK:", out['WITHOUT_A_WAY_BACK'], flush=True)
    with open(os.path.join(HERE, f"r2_entry_compare_{tag}.json"), 'w',
              encoding='cp1252', errors='replace') as fh:
        json.dump(out, fh, indent=1, sort_keys=True, default=str)
    print(f"[wrote] r2_entry_compare_{tag}.json", flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
