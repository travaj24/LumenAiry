"""Compare two WP-C5 probe JSONs key by key and write the verdict.

    python validation/probe_c5_three_defaults/c5_compare.py BASE.json HEAD.json OUT.json

Never imports ``lumenairy``: it only reads the two files the probes wrote, so
it cannot be confused by a pinned tree and it is safe to run from anywhere.
"""
import json
import sys


def main(base_p, head_p, out_p):
    base = json.load(open(base_p, encoding='utf-8'))
    head = json.load(open(head_p, encoding='utf-8'))
    b, h = base.get('digests', {}), head.get('digests', {})
    same = sorted(k for k in b if k in h and b[k] == h[k])
    moved = sorted(k for k in b if k in h and b[k] != h[k])
    only_b = sorted(set(b) - set(h))
    only_h = sorted(set(h) - set(b))
    res = {
        'base_file': base['_env']['lumenairy_file'],
        'head_file': head['_env']['lumenairy_file'],
        'base_python': base['_env']['python'],
        'head_python': head['_env']['python'],
        'n_keys_base': len(b), 'n_keys_head': len(h),
        'n_identical': len(same), 'n_moved': len(moved),
        'moved': moved, 'only_in_base': only_b, 'only_in_head': only_h,
    }
    json.dump(res, open(out_p, 'w', encoding='utf-8'), indent=1,
              sort_keys=True)
    print('identical %d / moved %d / base-only %d / head-only %d'
          % (len(same), len(moved), len(only_b), len(only_h)))
    for k in moved:
        print('  MOVED', k)
    for k in only_b:
        print('  BASE-ONLY', k)
    for k in only_h:
        print('  HEAD-ONLY', k)


if __name__ == '__main__':
    main(*sys.argv[1:4])
