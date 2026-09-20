"""Compare two vh3_bitid.py JSON readings key by key.

    python vh3_compare.py A.json B.json [OUT.json]

Prints the number of keys that AGREE and that DIFFER, splits the difference
by prefix, and (with OUT.json) writes the verdict.  Never imports lumenairy,
so it cannot be confused about which tree produced which file.
"""
import collections
import json
import sys


def load(p):
    with open(p, encoding='cp1252') as fh:
        return json.load(fh)


def main(pa, pb, out=None):
    A, B = load(pa), load(pb)
    ka, kb = set(A['keys']), set(B['keys'])
    common = sorted(ka & kb)
    same = [k for k in common if A['keys'][k] == B['keys'][k]]
    diff = [k for k in common if A['keys'][k] != B['keys'][k]]
    kindflip = [k for k in common
                if A.get('kinds', {}).get(k) != B.get('kinds', {}).get(k)]
    bypfx = collections.Counter(k.split('.')[0] for k in diff)
    tot = collections.Counter(k.split('.')[0] for k in common)
    res = {
        'a': {'file': pa, 'lumenairy_file': A['lumenairy_file'],
              'python': A['python'], 'numpy': A['numpy'],
              'platform': A['platform']},
        'b': {'file': pb, 'lumenairy_file': B['lumenairy_file'],
              'python': B['python'], 'numpy': B['numpy'],
              'platform': B['platform']},
        'n_common': len(common),
        'n_same': len(same),
        'n_diff': len(diff),
        'only_in_a': sorted(ka - kb),
        'only_in_b': sorted(kb - ka),
        'kind_flips': kindflip,
        'diff_by_prefix': {p: [bypfx.get(p, 0), tot[p]] for p in sorted(tot)},
        'diff_keys': diff,
    }
    print(f"A {A['lumenairy_file']}  ({A['platform']} py{A['python']})")
    print(f"B {B['lumenairy_file']}  ({B['platform']} py{B['python']})")
    print(f"common {len(common)}  SAME {len(same)}  DIFF {len(diff)}  "
          f"only-A {len(ka - kb)}  only-B {len(kb - ka)}  "
          f"kind-flips {len(kindflip)}")
    for p in sorted(tot):
        print(f"  {p}: {bypfx.get(p, 0)}/{tot[p]} differ")
    if diff:
        print("  first differing keys:", diff[:12])
    if out:
        with open(out, 'w', encoding='cp1252') as fh:
            json.dump(res, fh, indent=1, sort_keys=True)
    return 0


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
