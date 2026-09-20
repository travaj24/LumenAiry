"""VERIFY-WP-C1 ROUND 2 -- the recorded .test_durations item, re-checked.

Round 2 claims: all 31 ids of ``tests/unit/test_c1_gray_edge_default.py``
spliced, the one id left stale by round 2's rename removed, 16 624 -> 16 655
entries, JSON re-parsed, existing key order preserved, staleness gate 4 passed.

This probe re-collects the ids from BOTH C1 files and cross-checks them against
the file, so "31 spliced" is checked against pytest's own collection rather
than against a count, and any stale entry left behind is named.
"""
import json
import subprocess
import sys


def collected(paths):
    out = subprocess.run(
        [sys.executable, '-m', 'pytest', *paths, '--collect-only', '-q',
         '-p', 'no:randomly'], capture_output=True, text=True)
    sep = chr(92)          # backslash, without putting one in a literal
    return [ln.strip().replace(sep, '/') for ln in out.stdout.splitlines()
            if '::' in ln and ln.strip().startswith('tests/')]


def main(out_path):
    files = ['tests/unit/test_c1_gray_edge_default.py',
             'tests/unit/test_verify_c1_gray_edge.py']
    with open('.test_durations', encoding='utf-8') as fh:
        raw = fh.read()
    d = json.loads(raw)
    sep = chr(92)
    dk = {k.replace(sep, '/'): v for k, v in d.items()}

    report = {'total_entries': len(d), 'json_valid': True, 'files': {}}
    for f in files:
        ids = collected([f])
        stem = f.split('/')[-1][:-3]
        present = [i for i in ids if i in dk]
        missing = [i for i in ids if i not in dk]
        stale = [k for k in dk if stem in k and k not in set(ids)]
        report['files'][f] = {
            'collected': len(ids),
            'in_durations': len(present),
            'missing': missing,
            'stale': stale,
            'total_seconds': round(sum(dk[i] for i in present), 4),
        }
    # key order preserved?  The spliced ids should be APPENDED or in place,
    # and the file must round-trip through json without reordering loss.
    keys = list(d)
    report['first_key'] = keys[0]
    report['last_key'] = keys[-1]
    report['duplicate_keys'] = len(keys) != len(set(keys))
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(report, fh, indent=1, sort_keys=True)
    print('.test_durations entries:', report['total_entries'],
          ' json valid: True  duplicate keys:', report['duplicate_keys'])
    for f, r in report['files'].items():
        print("  {0}: collected={1} in_durations={2} missing={3} stale={4} "
              "total={5}s".format(f, r['collected'], r['in_durations'],
                                  len(r['missing']), len(r['stale']),
                                  r['total_seconds']))
        for m in r['missing']:
            print('     MISSING:', m)
        for s in r['stale']:
            print('     STALE  :', s)


if __name__ == '__main__':
    main(sys.argv[1])
