"""Round 3 -- compare two ``r3_routes_bitid.py`` outputs key by key.

    python r3_compare.py BASE.json BRANCH.json [OUT.json]

Prints the VALUE differing count (which must be 0: this round changes no
arithmetic) and the WARNING differing count (which is allowed to be non-zero
and is the point of the change), with the differing keys named.
"""
import json
import sys


def load(p):
    with open(p, encoding='cp1252') as fh:
        return json.load(fh)


def main(base_p, branch_p, out_p=None):
    a, b = load(base_p), load(branch_p)
    res = {'base': a['lumenairy_file'], 'branch': b['lumenairy_file'],
           'platform': a['platform'], 'n_base': a['n'], 'n_branch': b['n']}
    if set(a['value']) != set(b['value']):
        only_a = sorted(set(a['value']) - set(b['value']))
        only_b = sorted(set(b['value']) - set(a['value']))
        res['key_sets_differ'] = {'only_base': only_a[:20],
                                  'only_branch': only_b[:20],
                                  'n_only_base': len(only_a),
                                  'n_only_branch': len(only_b)}
    keys = sorted(set(a['value']) & set(b['value']))
    vdiff = [k for k in keys if a['value'][k] != b['value'][k]]
    wdiff = [k for k in keys if a['warn'][k] != b['warn'][k]]
    res.update({'n_common': len(keys), 'n_value_differing': len(vdiff),
                'n_warn_differing': len(wdiff),
                'value_differing': vdiff, 'warn_differing': wdiff})
    print(f"platform      {a['platform']}")
    print(f"base          {a['lumenairy_file']}  ({a['n']} keys)")
    print(f"branch        {b['lumenairy_file']}  ({b['n']} keys)")
    print(f"common keys   {len(keys)}")
    print(f"VALUE differing   {len(vdiff)}"
          + ("" if not vdiff else "  " + ", ".join(vdiff[:10])))
    print(f"WARNING differing {len(wdiff)}"
          + ("" if not wdiff else "  e.g. " + ", ".join(wdiff[:4])))
    if out_p:
        with open(out_p, 'w', encoding='cp1252') as fh:
            json.dump(res, fh, indent=1, sort_keys=True)
        print(f"-> {out_p}")
    return 0 if not vdiff else 1


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
