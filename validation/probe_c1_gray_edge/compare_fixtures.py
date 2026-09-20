"""WP-C1 probe 3 -- the byte-identity comparison, archive to archive.

Reads two ``fixtures_*.json`` written by ``probe_fixtures.py`` and reports, per
fixture, whether the SHA-256 of the returned array's raw bytes is the same.
Never compares within one tree: the reference side is always a run against the
``git archive`` extraction of the parent commit.

Usage::

    python validation/probe_c1_gray_edge/compare_fixtures.py \
        <base.json> <new.json> <label>

Writes ``validation/probe_c1_gray_edge/identity_<label>.json`` and prints the
counts the report quotes.
"""
import json
import os
import sys


def load(path):
    with open(path, encoding='cp1252') as f:
        return json.load(f)


def compare(base, new, section):
    same, moved, only_base, only_new = [], [], [], []
    for name, d in sorted(base[section].items()):
        if name not in new[section]:
            only_base.append(name)
            continue
        n = new[section][name]
        if (d['sha256'] == n['sha256'] and d['dtype'] == n['dtype']
                and d['shape'] == n['shape']):
            same.append(name)
        else:
            moved.append({'name': name, 'base': d, 'new': n})
    for name in sorted(new[section]):
        if name not in base[section]:
            only_new.append(name)
    return {'identical': same, 'moved': moved,
            'only_in_base': only_base, 'only_in_new': only_new}


def main():
    base_path, new_path, label = sys.argv[1], sys.argv[2], sys.argv[3]
    base, new = load(base_path), load(new_path)
    out = {
        'label': label,
        'base': {k: base[k] for k in ('tag', 'arm', 'lumenairy_file',
                                      'python', 'platform', 'numpy')},
        'new': {k: new[k] for k in ('tag', 'arm', 'lumenairy_file',
                                    'python', 'platform', 'numpy')},
        'base_unavailable': base['unavailable'],
        'new_unavailable': new['unavailable'],
        'aperture': compare(base, new, 'aperture'),
        'plain': compare(base, new, 'plain'),
    }
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, f'identity_{label}.json')
    with open(path, 'w', encoding='cp1252') as f:
        json.dump(out, f, indent=2, sort_keys=True)
    for section in ('aperture', 'plain'):
        r = out[section]
        print(f"{section:>10}: {len(r['identical'])} byte-identical, "
              f"{len(r['moved'])} moved, "
              f"{len(r['only_in_base'])} base-only, "
              f"{len(r['only_in_new'])} new-only")
        for m in r['moved']:
            print(f"    MOVED {m['name']}: {m['base']['sha256'][:12]} -> "
                  f"{m['new']['sha256'][:12]}")
        for n in r['only_in_base']:
            print(f"    base-only {n}")
        for n in r['only_in_new']:
            print(f"    new-only {n}")
    print(f"wrote {path}")
    # Exit code carries the decision so a shell caller can gate on it.
    bad = (out['aperture']['moved'] or out['plain']['moved'])
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
