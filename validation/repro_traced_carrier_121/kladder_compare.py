# Compare the k-ladder arms written by kladder_121.py.
#
# THE ACCEPTANCE IS BIT-IDENTITY (the niche-D8 contract): the parent
# accumulates congruence results in ASCENDING k regardless of the order the
# workers finish in, so the complex sum -- and therefore every scalar derived
# from it -- must be FP-identical to serial at every worker count.  This
# script checks that on the artifacts rather than on a printed banner:
#
#   * sha256 of the whole accumulated field;
#   * sha256 of each order's readout tile;
#   * every per-order scalar, at rel=0 abs=0;
#   * np.array_equal on the two full tiles each arm dumped.
#
# and then prints the speed table -- wall/order, efficiency vs k=1, peak RSS
# of the tree and of the largest single worker.
#
# Usage:  python kladder_compare.py <dir> <base_tag> <tag> [<tag> ...]
import hashlib
import json
import os
import sys

import numpy as np

_SCALARS = ('power_in', 'power_exit', 'power_out', 'throughput', 'capture',
            'cellP', 'field_pct', 'fwhm_um', 'ee3', 'ee6', 'ee12',
            'chief_x_um', 'chief_y_um')


def load(d, tag):
    with open(os.path.join(d, '%s.json' % tag), 'r', encoding='cp1252') as fh:
        return json.load(fh)


def main(argv):
    d = argv[0]
    tags = argv[1:]
    docs = {t: load(d, t) for t in tags}
    base = tags[0]
    b = docs[base]

    print('=' * 78)
    print('K-LADDER -- %s' % d)
    print('=' * 78)
    print('  base arm %s: CW=%s NFC=%s RAMB=%s orders=%d NOUT=%d'
          % (base, b['CW'], b['nfc'], b['ramb'], b['orders'], b['nout']))
    print()
    print('  %-10s %3s %8s %10s %10s %8s %10s %10s %6s'
          % ('tag', 'k', 'chainB s', 's/order', 'speed-up', 'eff %',
             'peak GB', 'worker GB', 'exit'))
    t1 = float(b['chainB_per_order_s'])
    for t in tags:
        a = docs[t]
        sp = t1 / float(a['chainB_per_order_s'])
        print('  %-10s %3d %8.1f %10.2f %10.3f %8.1f %10.2f %10.2f %6d'
              % (t, a['CW'], a['chainB_s'], a['chainB_per_order_s'], sp,
                 100.0 * sp / max(1, a['CW']), a['peak_tree_gb'],
                 a['peak_child_gb'] or a['peak_tree_gb'], a['exit']))

    print()
    print('  BIT-IDENTITY vs %s (the D8 contract)' % base)
    ok_all = True
    for t in tags[1:]:
        a = docs[t]
        bad = []
        if a['field_sha256'] != b['field_sha256']:
            bad.append('accumulated field sha256')
        if len(a['rows']) != len(b['rows']):
            bad.append('order count')
        else:
            for ra, rb in zip(a['rows'], b['rows']):
                if ra['order'] != rb['order']:
                    bad.append('order ORDER (%s vs %s)'
                               % (ra['order'], rb['order']))
                    continue
                if ra['tile_sha256'] != rb['tile_sha256']:
                    bad.append('%s tile sha256' % ra['order'])
                for s in _SCALARS:
                    if ra[s] != rb[s]:
                        bad.append('%s %s (%.17g vs %.17g)'
                                   % (ra['order'], s, ra[s], rb[s]))
        print('    %-10s %s' % (t, 'IDENTICAL' if not bad
                                else 'DIFFERS: ' + '; '.join(bad[:6])))
        ok_all &= not bad

    print()
    print('  FULL-ARRAY CHECK on the dumped tiles (np.array_equal)')
    fb = os.path.join(d, '%s_tiles.npz' % base)
    if os.path.exists(fb):
        zb = np.load(fb)
        for t in tags[1:]:
            fa = os.path.join(d, '%s_tiles.npz' % t)
            if not os.path.exists(fa):
                print('    %-10s (no tile dump)' % t)
                continue
            za = np.load(fa)
            for key in sorted(zb.files):
                if key not in za.files:
                    print('    %-10s %s MISSING' % (t, key))
                    ok_all = False
                    continue
                same = np.array_equal(za[key], zb[key])
                mx = float(np.abs(za[key] - zb[key]).max())
                print('    %-10s %-16s array_equal=%s  max|delta|=%.3e  '
                      'sha=%s' % (t, key, same, mx,
                                  hashlib.sha256(np.ascontiguousarray(
                                      za[key]).tobytes()).hexdigest()[:16]))
                ok_all &= same
    print()
    print('  VERDICT: %s' % ('BIT-IDENTICAL AT EVERY k' if ok_all
                             else 'NOT IDENTICAL -- see above'))
    return 0 if ok_all else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
