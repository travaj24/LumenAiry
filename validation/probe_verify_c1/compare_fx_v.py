"""VERIFY-C1 -- compare two fixture digest files, archive to archive.

Usage::

    python -P validation/probe_verify_c1/compare_fx_v.py \
        <base.json> <new.json> <label>

Writes ``validation/probe_verify_c1/id_<label>.json`` and prints the counts.
Both sides must be the SAME build: nothing is ever compared across builds.
"""
import json
import os
import sys


def load(path):
    with open(path) as f:
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
            moved.append({'name': name,
                          'base_sha': d['sha256'][:16],
                          'new_sha': n['sha256'][:16],
                          'base_dtype': d['dtype'], 'new_dtype': n['dtype'],
                          'base_neg_zero': [d.get('neg_zero_real'),
                                            d.get('neg_zero_imag')],
                          'new_neg_zero': [n.get('neg_zero_real'),
                                           n.get('neg_zero_imag')]})
    for name in sorted(new[section]):
        if name not in base[section]:
            only_new.append(name)
    return {'identical': same, 'moved': moved,
            'only_base': only_base, 'only_new': only_new,
            'n_identical': len(same), 'n_moved': len(moved),
            'n_total': len(same) + len(moved)}


def main():
    base_p, new_p, label = sys.argv[1], sys.argv[2], sys.argv[3]
    base, new = load(base_p), load(new_p)
    out = {
        'label': label,
        'base': {'file': os.path.basename(base_p),
                 'lumenairy_file': base['lumenairy_file'],
                 'arm': base['arm'], 'python': base['python'].split()[0]},
        'new': {'file': os.path.basename(new_p),
                'lumenairy_file': new['lumenairy_file'],
                'arm': new['arm'], 'python': new['python'].split()[0]},
        'aperture': compare(base, new, 'aperture'),
        'plain': compare(base, new, 'plain'),
    }
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, 'id_%s.json' % label)
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    a, p = out['aperture'], out['plain']
    print('[%s] base=%s(%s) new=%s(%s)' % (
        label, out['base']['file'], base['arm'],
        out['new']['file'], new['arm']))
    print('  aperture: %d/%d identical, %d moved'
          % (a['n_identical'], a['n_total'], a['n_moved']))
    print('  plain   : %d/%d identical, %d moved'
          % (p['n_identical'], p['n_total'], p['n_moved']))
    for m in a['moved']:
        print('    MOVED ap  ', m['name'], m['base_sha'], '->', m['new_sha'],
              'neg0', m['base_neg_zero'], '->', m['new_neg_zero'])
    for m in p['moved']:
        print('    MOVED pln ', m['name'], m['base_sha'], '->', m['new_sha'])
    if a['only_base'] or a['only_new'] or p['only_base'] or p['only_new']:
        print('  only_base ap', a['only_base'], 'only_new ap', a['only_new'])
        print('  only_base pl', p['only_base'], 'only_new pl', p['only_new'])
    print('wrote', path)


if __name__ == '__main__':
    main()
