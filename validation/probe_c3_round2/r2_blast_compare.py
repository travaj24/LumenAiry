"""WP-C3 round 2 -- classify a blast-width sweep base vs branch.

    python r2_blast_compare.py <base.json> <branch.json>

Prints the IDENTICAL / MOVED / OK->RAISED / RAISED->OK / BOTH-RAISED census
and the Kelly-warning totals, which is the measurement VERIFY-WP-C3 D5's
claim has to be written from.
"""
import collections
import json
import sys


def main():
    base = json.load(open(sys.argv[1], encoding='utf-8'))
    head = json.load(open(sys.argv[2], encoding='utf-8'))
    cls = collections.Counter()
    moved, raised = [], []
    for k, b in base['cells'].items():
        h = head['cells'][k]
        if b['outcome'] == 'raised' and h['outcome'] == 'raised':
            cls['BOTH-RAISED'] += 1
        elif b['outcome'] == 'raised':
            cls['RAISED->OK'] += 1
        elif h['outcome'] == 'raised':
            cls['OK->RAISED'] += 1
            raised.append((k, h.get('exc'), (h.get('msg') or '')[:80]))
        elif b['sha'] == h['sha']:
            cls['IDENTICAL'] += 1
        else:
            cls['MOVED'] += 1
            moved.append(k)
    n = len(base['cells'])
    print(f"base default = {base['default_transport']}  "
          f"branch default = {head['default_transport']}  cells = {n}")
    for k in ('IDENTICAL', 'MOVED', 'OK->RAISED', 'RAISED->OK',
              'BOTH-RAISED'):
        print(f"  {k:12s} {cls[k]:4d}  {100.0 * cls[k] / n:6.2f} %")
    kb = sum(v.get('kelly', 0) for v in base['cells'].values())
    kh = sum(v.get('kelly', 0) for v in head['cells'].values())
    cb = sum(1 for v in base['cells'].values() if v.get('kelly', 0))
    ch = sum(1 for v in head['cells'].values() if v.get('kelly', 0))
    print(f"  Kelly warnings: base {kb} over {cb} cells | "
          f"branch {kh} over {ch} cells")
    if raised:
        print('  OK->RAISED cells:')
        for r in raised[:30]:
            print('   ', r)
    if moved:
        print(f"  MOVED cells ({len(moved)}):")
        for m in moved[:40]:
            print('   ', m)


if __name__ == '__main__':
    main()
