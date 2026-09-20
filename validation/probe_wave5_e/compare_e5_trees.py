"""WAVE5-E item E5 -- PRE vs POST comparison of ``probe_e5_dead_ray_freeze``.

The claim the mask has to survive is that no LIVE bit moved.  This reads the
two trees' JSON and compares, cell by cell, the md5 of the alive rows' state
and Jacobian, and reports the dead-row drift the fix removed.

Usage:  python compare_e5_trees.py <pre.json> <post.json> [<out.json>]
"""
from __future__ import annotations

import json
import sys


def main():
    pre = json.load(open(sys.argv[1], encoding='cp1252'))
    post = json.load(open(sys.argv[2], encoding='cp1252'))
    out_path = sys.argv[3] if len(sys.argv) > 3 else None
    idx = {(r['fixture'], r['backend']): r for r in pre['rows']}
    res = {'pre_tree': pre['lumenairy_file'], 'post_tree': post['lumenairy_file'],
           'platform': post.get('platform'), 'python': post.get('python'),
           'numpy': post.get('numpy'), 'rows': []}
    for q in post['rows']:
        key = (q['fixture'], q['backend'])
        p = idx.get(key)
        if p is None or 'err' in p or 'err' in q:
            res['rows'].append({'fixture': key[0], 'backend': key[1],
                                'skipped': True,
                                'pre_err': (p or {}).get('err'),
                                'post_err': q.get('err')})
            continue
        row = {
            'fixture': key[0], 'backend': key[1],
            'alive_state_md5_same': p['alive_md5_state'] == q['alive_md5_state'],
            'alive_jac_md5_same': p['alive_md5_jac'] == q['alive_md5_jac'],
            'pre_dead_frozen': p['dead_frozen'],
            'post_dead_frozen': q['dead_frozen'],
            'pre_dead_dopd': p.get('dead_dopd', p.get('jax_dead_dopd')),
            'post_dead_dopd': q.get('dead_dopd', q.get('jax_dead_dopd')),
            'pre_dead_djac': p.get('dead_djac', p.get('jax_dead_djac')),
            'post_dead_djac': q.get('dead_djac', q.get('jax_dead_djac')),
            'n_dead': q['n_dead'], 'n_alive': q['n_alive'],
            'freeze_set_empty': q.get('freeze_set_empty'),
            'n_companion_only_dead': q.get('n_companion_only_dead'),
        }
        res['rows'].append(row)
        print('%-18s %-9s alive state/jac identical %-5s/%-5s  dead opd '
              '%.3e -> %.3e  jac %.3e -> %.3e'
              % (row['fixture'], row['backend'],
                 row['alive_state_md5_same'], row['alive_jac_md5_same'],
                 row['pre_dead_dopd'], row['post_dead_dopd'],
                 row['pre_dead_djac'], row['post_dead_djac']), flush=True)
    live = [r for r in res['rows'] if not r.get('skipped')]
    res['n_cells'] = len(live)
    res['all_alive_bits_identical'] = bool(live) and all(
        r['alive_state_md5_same'] and r['alive_jac_md5_same'] for r in live)
    scored = [r for r in live if not r.get('freeze_set_empty')]
    res['n_cells_vacuous'] = len(live) - len(scored)
    res['all_post_frozen'] = bool(scored) and all(
        r['post_dead_frozen'] for r in scored)
    res['n_pre_unfrozen'] = sum(1 for r in scored
                                if not r['pre_dead_frozen'])
    res['max_pre_dead_dopd'] = max((r['pre_dead_dopd'] for r in live),
                                   default=None)
    res['max_pre_dead_djac'] = max((r['pre_dead_djac'] for r in live),
                                   default=None)
    print('CELLS %d (%d scored, %d vacuous)   ALIVE BITS IDENTICAL %s   '
          'ALL POST FROZEN %s   PRE UNFROZEN %d of %d   '
          'max pre dead opd drift %.3e'
          % (res['n_cells'], len(scored), res['n_cells_vacuous'],
             res['all_alive_bits_identical'], res['all_post_frozen'],
             res['n_pre_unfrozen'], len(scored),
             res['max_pre_dead_dopd']), flush=True)
    if out_path:
        with open(out_path, 'w', encoding='cp1252') as fh:
            json.dump(res, fh, indent=1)
        print('wrote', out_path, flush=True)


if __name__ == '__main__':
    main()
