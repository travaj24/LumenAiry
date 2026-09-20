"""VERIFY-WP-C4 ROUND 2, item 4/6 -- reduce the two entry-point digests to a
verdict per entry point.

Per entry point, on each build:

* ``moves_at_captured`` -- the no-keyword call differs between ``49ddf4bd``
  and this branch.  A row that does not move witnesses nothing.
* ``identical_at_refused`` -- the no-keyword call is byte-identical at a shape
  the rule refuses.  A row that also moved there means the flip is wider than
  the rule says.
* ``spelling`` -- which ``mft_method`` value reproduces the base bytes EXACTLY
  (``bluestein`` / ``separable`` / both / neither).
* ``none_is_inert`` -- ``mft_method=None`` is byte-identical to omitting the
  keyword.
* ``auto_is_the_default_today`` -- ``mft_method='auto'`` is byte-identical to
  omitting it, which is true today and is NOT the same statement as the one
  above.

    python vc4b_entry_compare.py <win|wsl>

Author:  Andrew Traverso
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vc4blib as L                                              # noqa: E402


def main(build):
    def load(tag):
        p = os.path.join(HERE, "vc4b_entry_%s_%s.json" % (tag, build))
        with open(p, encoding='cp1252') as fh:
            return json.load(fh)

    base, branch = load('base'), load('branch')
    b, r = base['keys'], branch['keys']
    names = sorted({k.split('|')[0] for k in b})
    rows = []
    for n in names:
        cap_b, cap_r = b.get(n + '|captured'), r.get(n + '|captured')
        ref_b, ref_r = b.get(n + '|refused'), r.get(n + '|refused')
        blu, sep = r.get(n + '|captured|bluestein'), r.get(n + '|captured|separable')
        none, auto = r.get(n + '|captured|none'), r.get(n + '|captured|auto')
        spell = [s for s, v in (('bluestein', blu), ('separable', sep))
                 if v is not None and v == cap_b]
        rows.append({
            'entry_point': n,
            'moves_at_captured': bool(cap_b and cap_r and cap_b != cap_r),
            'identical_at_refused': bool(ref_b and ref_r and ref_b == ref_r),
            'spelling_reproducing_base': spell,
            'way_back': bool(spell) or (cap_b == cap_r),
            'bluestein_matches_base': blu == cap_b if blu else None,
            'separable_matches_base': sep == cap_b if sep else None,
            'none_is_inert': (none == cap_r) if none else None,
            'auto_equals_no_keyword': (auto == cap_r) if auto else None,
        })
        print("%-42s moves=%-5s same@refused=%-5s spelling=%-20s none_inert=%s"
              % (n, rows[-1]['moves_at_captured'],
                 rows[-1]['identical_at_refused'],
                 ','.join(spell) or '(none needed)' if not
                 rows[-1]['moves_at_captured'] else ','.join(spell) or 'NONE',
                 rows[-1]['none_is_inert']))
    out = {'build': build, 'n_base_keys': len(b), 'n_branch_keys': len(r),
           'base_errors': base.get('errors', {}),
           'branch_errors': branch.get('errors', {}),
           'N_in': branch['N_in'], 'N_captured': branch['N_captured'],
           'N_refused': branch['N_refused'], 'rows': rows}
    out['moved_without_a_way_back'] = [x['entry_point'] for x in rows
                                       if x['moves_at_captured']
                                       and not x['spelling_reproducing_base']]
    out['moved_at_a_refused_shape'] = [x['entry_point'] for x in rows
                                       if not x['identical_at_refused']]
    out['none_not_inert'] = [x['entry_point'] for x in rows
                             if x['none_is_inert'] is False]
    out['did_not_move_at_captured'] = [x['entry_point'] for x in rows
                                       if not x['moves_at_captured']]
    print()
    for k in ('moved_without_a_way_back', 'moved_at_a_refused_shape',
              'none_not_inert', 'did_not_move_at_captured'):
        print("%-30s %s" % (k, out[k]))
    L.write(out, os.path.join(HERE, "vc4b_entry_compare_%s.json" % build))


if __name__ == '__main__':
    main(sys.argv[1])
