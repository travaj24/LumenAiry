"""Compare the four ``r2_wayback_entrypoints.py`` runs of one build.

Prints the per-entry-point table WP-C2 round 2's defect D4 needs:

* ``pre`` (the ``git archive 49ddf4bd`` tree, no keyword) against ``post``
  (this tree, ``renormalize='surface', sphere_normal='generic'``) -- the way
  back.  Must be identical on every array;
* ``pre`` against ``post_default`` -- the CONTRAST: the entry points that
  actually moved.  An entry point identical here simply does not sample the
  flip on this fixture, and is reported as such so a 16/16 way back cannot
  be read as "the keyword reaches nothing";
* ``post_default`` against ``post_none`` -- "``None`` stamps nothing".
"""
from __future__ import annotations

import argparse
import json


def load(path):
    with open(path, encoding='ascii') as fh:
        return json.load(fh)


def compare(a, b):
    da, db = a['digests'], b['digests']
    rows = []
    for name in sorted(set(da) | set(db)):
        ka, kb = da.get(name, {}), db.get(name, {})
        keys = sorted(set(ka) | set(kb))
        same = sum(1 for k in keys if ka.get(k) == kb.get(k) is not None)
        rows.append((name, same, len(keys)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pre', required=True)
    ap.add_argument('--post', required=True)
    ap.add_argument('--post-default', required=True)
    ap.add_argument('--post-none', required=True)
    ap.add_argument('--out')
    args = ap.parse_args()

    pre, post = load(args.pre), load(args.post)
    pdef, pnone = load(args.post_default), load(args.post_none)
    assert not pre['errors'] and not post['errors'], (
        pre['errors'], post['errors'])

    wayback = compare(pre, post)
    contrast = compare(pre, pdef)
    nones = compare(pdef, pnone)
    by = {n: (s, t) for n, s, t in contrast}
    bn = {n: (s, t) for n, s, t in nones}

    tot_s = tot_t = 0
    moved_eps = 0
    print(f'{"entry point":28s} {"way back":>14s} {"moved at default":>18s} '
          f'{"None==omitted":>15s}')
    for name, s, t in wayback:
        cs, ct = by[name]
        ns, nt = bn[name]
        tot_s += s
        tot_t += t
        if cs < ct:
            moved_eps += 1
        print(f'{name:28s} {s:6d}/{t:<7d} {ct - cs:6d}/{ct:<11d} '
              f'{ns:6d}/{nt:<8d}')
    print(f'{"TOTAL":28s} {tot_s:6d}/{tot_t:<7d}  '
          f'entry points that moved at the default: {moved_eps}/'
          f'{len(wayback)}')
    ok = tot_s == tot_t
    print('WAY BACK BYTE-IDENTICAL:', ok)
    print('None == omitted:', all(ns == nt for _n, ns, nt in nones))

    if args.out:
        payload = {
            'pre': {k: pre[k] for k in ('root', 'python', 'numpy',
                                        'lumenairy_file')},
            'post': {k: post[k] for k in ('root', 'python', 'numpy',
                                          'lumenairy_file')},
            'way_back': {n: {'identical': s, 'total': t}
                         for n, s, t in wayback},
            'moved_at_default': {n: {'moved': t - s, 'total': t}
                                 for n, s, t in contrast},
            'none_equals_omitted': {n: {'identical': s, 'total': t}
                                    for n, s, t in nones},
            'way_back_identical_total': tot_s,
            'way_back_arrays_total': tot_t,
            'way_back_byte_identical': ok,
            'entry_points': len(wayback),
            'entry_points_that_moved_at_the_default': moved_eps,
        }
        with open(args.out, 'w', encoding='ascii') as fh:
            json.dump(payload, fh, indent=1, sort_keys=True)
        print('wrote', args.out)


if __name__ == '__main__':
    main()
