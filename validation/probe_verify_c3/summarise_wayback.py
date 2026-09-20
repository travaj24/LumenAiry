"""Print the VERIFY-WP-C3 4a/4b tables from the driver's compare JSONs."""
from __future__ import annotations

import io
import json
import sys

PREFIX_TO_ENTRY = [
    ('S1-', 'propagate_carrier_referenced'),
    ('S2-', 'propagate_carrier_referenced'),
    ('S3-', 'propagate_carrier_referenced'),
    ('S4-', 'propagate_carrier_referenced'),
    ('S5-', 'propagate_carrier_referenced'),
    ('S6-', 'propagate_carrier_referenced'),
    ('S7-', 'propagate_carrier_referenced'),
    ('J0-', 'propagate_carrier_referenced (JAX array)'),
    ('J1-', 'propagate_carrier_referenced (JAX array)'),
    ('P1-', 'carrier_referenced_focus_readout'),
    ('P2-', 'carrier_referenced_exact_focus_readout'),
    ('P3-', 'carrier_referenced_* helpers'),
    ('C1-', 'propagate_traced_carrier_chain'),
    ('C2-', 'propagate_traced_carrier_chain'),
    ('C3-', 'propagate_traced_carrier_chain'),
    ('C4-', 'propagate_traced_carrier_chain'),
    ('C5-', 'propagate_traced_carrier_chain'),
    ('M1-', 'propagate_traced_carrier_chain_multi'),
    ('M2-', 'propagate_traced_carrier_chain_multi'),
    ('M3-', 'propagate_traced_carrier_chain_multi'),
    ('M4-', 'propagate_traced_carrier_chain_multi'),
]


def entry_of(k):
    for p, e in PREFIX_TO_ENTRY:
        if k.startswith(p):
            return e
    return '?'


def main():
    for path in sys.argv[1:]:
        d = json.load(io.open(path, encoding='utf-8'))
        print('=' * 78)
        print(path)
        for n, v in sorted(d['arms'].items()):
            print('  arm %-12s %-22s spelling=%-8s %-12s keys=%d  %.1fs'
                  % (n, v['tree'], v['spelling'], v['build'], v['n_keys'],
                     v['seconds']))
        for p, c in d['comparisons'].items():
            print('  --- %s : shared %d identical %d differ %d only_left %r '
                  'only_right %r  -> %s'
                  % (p, c['n_shared'], c['n_identical'], c['n_differ'],
                     c['only_left'], c['only_right'], c['verdict']))
            by = {}
            for k in c['differ']:
                by.setdefault(entry_of(k), []).append(k)
            allby = {}
            for k in sorted(set(list(c['differ'])
                                + [x for x in c['differ']])):
                pass
            for e, ks in sorted(by.items()):
                print('      %-42s %d moved: %s' % (e, len(ks),
                                                    ', '.join(sorted(ks))))
            # outcome flips (ok -> raised and back)
            flips = []
            for k in c['differ']:
                L = c['differ_detail'][k]['left']
                R = c['differ_detail'][k]['right']
                if L['outcome'] != R['outcome']:
                    flips.append((k, L['outcome'], R['outcome'],
                                  (R['detail'] or L['detail'])[:130]))
            if flips:
                print('      OUTCOME FLIPS (%d):' % len(flips))
                for k, lo, ro, det in flips:
                    print('        %-38s %s -> %s  %s' % (k, lo, ro, det))
            # warning-count changes
            wch = []
            for k in c['differ']:
                L = c['differ_detail'][k]['left']
                R = c['differ_detail'][k]['right']
                lw = [w[0] + ':' + w[1][:60] for w in L['warnings']]
                rw = [w[0] + ':' + w[1][:60] for w in R['warnings']]
                if lw != rw:
                    wch.append((k, len(L['warnings']), len(R['warnings'])))
            if wch:
                print('      WARNING SETS CHANGED (%d): %s'
                      % (len(wch), ', '.join('%s %d->%d' % x for x in wch)))


if __name__ == '__main__':
    main()
