"""WP-C4 -- compare two ``c4_bitid.py`` digest maps key by key.

    python c4_compare.py BASE.json BRANCH.json OUT.json

THREE CLAIMS, counted separately, so a pass cannot hide behind an aggregate:

1. ``live/bl/*`` and ``live/sep/*`` -- the PREVIOUS route named explicitly.
   Every key byte-identical base to branch, at every shape.  This is the
   "one keyword away, and byte-identical under it" claim.
2. ``live/nokw/*`` -- NO keyword.  Byte-identical exactly at the shapes the
   rule leaves on the previous route, and DIFFERENT exactly at the shapes it
   sends to the dense route.  Both counts are reported and the split is
   checked against the rule's own answer for each key's shape, which
   ``c4_bitid.py`` records beside the digest.
3. ``never/nokw/*`` -- the same no-keyword fixtures with
   ``_MFT_DIRECT_MAX_RATIO = _MFT_DIRECT_NEVER``.  Every key byte-identical to
   the BASE tree's ``live/nokw/*``, at every shape: the process-wide way back.

A key that is missing on one side is reported, never silently skipped.
"""
from __future__ import annotations

import json
import sys


def load(path):
    with open(path, encoding='cp1252') as fh:
        return json.load(fh)


def compare(base, branch, group, base_group=None):
    bg = base_group or group
    bkeys = {k[len(bg):]: v for k, v in base['keys'].items()
             if k.startswith(bg)}
    nkeys = {k[len(group):]: v for k, v in branch['keys'].items()
             if k.startswith(group)}
    both = sorted(set(bkeys) & set(nkeys))
    same = [k for k in both if bkeys[k] == nkeys[k]]
    diff = [k for k in both if bkeys[k] != nkeys[k]]
    return {'n': len(both), 'identical': len(same), 'differing': len(diff),
            'only_base': sorted(set(bkeys) - set(nkeys)),
            'only_branch': sorted(set(nkeys) - set(bkeys)),
            'differing_keys': diff}


def main(base_path, branch_path, out_path):
    base = load(base_path)
    branch = load(branch_path)
    out = {'base': {k: base.get(k) for k in
                    ('build', 'tree', 'lumenairy_file', 'version', 'has_rule',
                     'ratio')},
           'branch': {k: branch.get(k) for k in
                      ('build', 'tree', 'lumenairy_file', 'version',
                       'has_rule', 'ratio')}}
    out['wayback_bluestein'] = compare(base, branch, 'live/wayback_bl/')
    out['wayback_separable'] = compare(base, branch, 'live/wayback_sep/')
    out['no_keyword'] = compare(base, branch, 'live/nokw/')
    out['constant_never'] = compare(base, branch, 'never/nokw/',
                                    base_group='live/nokw/')
    out['resample_leg'] = compare(base, branch, 'live/resample/')
    out['carrier_leg'] = compare(base, branch, 'live/carrier/')

    # --- the no-keyword SPLIT, checked against the rule ------------------
    # Every primitive / propagator key names the shape it was driven at, and
    # ``c4_bitid.py`` records what ``_auto_selects_direct`` answers for that
    # shape.  A key must DIFFER base-to-branch exactly when the rule says
    # 'direct' -- counting the split is not the same as checking it.
    rule = branch.get('rule_says', {})
    split = {'agree': 0, 'disagree': [], 'unclassified': 0}
    for key, bv in base['keys'].items():
        if not key.startswith('live/nokw/'):
            continue
        rest = key[len('live/nokw/'):]
        parts = rest.split('/')
        kind, tag = parts[0], parts[1]
        if kind in ('plain', 'centred', 'offcentre'):
            says = rule.get(f'plain/{tag}')
        elif kind in ('fresnel', 'fraunhofer', 'asm'):
            says = rule.get(f'prop/{tag}')
        else:
            says = None
        if says is None:
            split['unclassified'] += 1
            continue
        moved = branch['keys'].get(key) != bv
        if moved == says:
            split['agree'] += 1
        else:
            split['disagree'].append(
                {'key': key, 'rule_says_direct': says, 'bytes_moved': moved})
    out['no_keyword_split_vs_rule'] = split
    print(f"no_keyword split vs rule: {split['agree']} agree, "
          f"{len(split['disagree'])} disagree, "
          f"{split['unclassified']} unclassified")

    ok = True
    for name in ('wayback_bluestein', 'wayback_separable', 'constant_never'):
        r = out[name]
        print(f"{name:22s} {r['identical']:4d}/{r['n']:4d} identical, "
              f"{r['differing']} differing, only_base {len(r['only_base'])}, "
              f"only_branch {len(r['only_branch'])}")
        if r['differing'] or r['only_base'] or r['only_branch']:
            ok = False
    for name in ('no_keyword', 'resample_leg', 'carrier_leg'):
        r = out[name]
        print(f"{name:22s} {r['identical']:4d}/{r['n']:4d} identical, "
              f"{r['differing']} differing  (a SPLIT is expected here)")
        if r['only_base'] or r['only_branch']:
            ok = False
    if out['no_keyword_split_vs_rule']['disagree']:
        ok = False
    out['all_byte_identity_claims_hold'] = ok
    print('BYTE-IDENTITY CLAIMS:', 'HOLD' if ok else 'BROKEN')
    with open(out_path, 'w', encoding='cp1252') as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print(f"-> {out_path}")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3]))
