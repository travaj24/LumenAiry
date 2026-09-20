"""Emit the report's tables from the JSONs, so no number in
``WP-B7c_ROUND3_REPORT.md`` is typed by hand.

Usage:
    python r3tables.py <joined.json> [<joined-at-shallower-ladder.json> ...]
"""
from __future__ import annotations

import collections
import json
import sys

import r3fixtures as FX


def load(p):
    with open(p, encoding='cp1252') as f:
        return json.load(f)


def population_table(rows):
    by = collections.defaultdict(list)
    for r in rows:
        by[r['fixture']].append(r)
    print('| optic | provenance | NA | grid | planes | fold ring | fallback |')
    print('|---|---|---|---|---|---|---|')
    import r3ladder as LAD
    G = LAD.geom()
    tot = fr = fb = 0
    for nm in sorted(by, key=lambda k: (k.rstrip('_alt'), k)):
        rs = by[nm]
        g = G.get(nm) or G.get(nm[:-4] if nm.endswith('_alt') else nm) or {}
        fx = FX.FIXTURES[nm]
        nfr = sum(1 for r in rs if r.get('reason') == 'fold_ring'
                  and not r.get('fell_back'))
        nfb = sum(1 for r in rs if r.get('fell_back'))
        tot += len(rs)
        fr += nfr
        fb += nfb
        print('| `%s` | %s | %.3f | %d x %.2f um | %d | %d | %d |' % (
            nm, FX.PROVENANCE.get(nm.replace('_alt', ''), '--'),
            g.get('na', float('nan')), fx['N'], fx['dx'] * 1e6,
            len(rs), nfr, nfb))
    print('| **total** | | | | **%d** | **%d** | **%d** |' % (tot, fr, fb))


def cost_table(summary, pop):
    c = summary['cost'][pop]
    bands = ['fid >= 0.99', '0.95 <= fid < 0.99', '0.883 <= fid < 0.95',
             '0.75 <= fid < 0.883', 'fid < 0.75']
    print('| bar | refused | returned | margin above | margin below | '
          + ' | '.join('REFUSED %s' % b for b in bands) + ' | '
          + ' | '.join('RETURNED %s' % b for b in bands) + ' |')
    print('|' + '---|' * (5 + 2 * len(bands)))
    for k in sorted(c, key=float):
        v = c[k]
        print('| %s | %d | %d | %s | %s | %s | %s |' % (
            k, v['n_refused'], v['n_returned'],
            ('%.5f' % v['margin_above']) if 'margin_above' in v else '--',
            ('%.5f' % v['margin_below']) if 'margin_below' in v else '--',
            ' | '.join(str(v['refused_by_band'][b]) for b in bands),
            ' | '.join(str(v['returned_by_band'][b]) for b in bands)))


def extremes(rows, bar=1.06, pop='fold_ring'):
    if pop == 'fold_ring':
        sel = [r for r in rows if r.get('reason') == 'fold_ring'
               and not r.get('fell_back')]
    elif pop == 'fallback':
        sel = [r for r in rows if r.get('fell_back')]
    else:
        sel = list(rows)
    ret = [r for r in sel if r['pixel_continuity'] <= bar]
    ref = [r for r in sel if r['pixel_continuity'] > bar]
    print('| population | n | continuity | oracle fidelity |')
    print('|---|---|---|---|')
    for nm, p in (('RETURNED', ret), ('REFUSED', ref)):
        if not p:
            continue
        print('| **%s** | %d | %.5f .. %.5f | %.4f .. %.4f |' % (
            nm, len(p), min(r['pixel_continuity'] for r in p),
            max(r['pixel_continuity'] for r in p),
            min(r['fidelity'] for r in p), max(r['fidelity'] for r in p)))
    if ret and ref:
        wr = min(r['fidelity'] for r in ret)
        br = max(r['fidelity'] for r in ref)
        print()
        print('worst returned fidelity %.4f against best refused %.4f -- '
              'populations %s' % (wr, br,
                                  'OVERLAP' if wr <= br else 'disjoint'))
    print()
    print('| | optic | z [um] | C | fidelity | power / oracle |')
    print('|---|---|---|---|---|---|')
    for r in sorted(ret, key=lambda r: r['fidelity'])[:6]:
        print('| worst RETURNED | `%s` | %.3f | %.5f | **%.4f** | %.4f |' % (
            r['fixture'], r['z_um'], r['pixel_continuity'], r['fidelity'],
            r.get('power_over_oracle') or float('nan')))
    for r in sorted(ref, key=lambda r: -r['fidelity'])[:6]:
        print('| best REFUSED | `%s` | %.3f | %.5f | **%.4f** | %.4f |' % (
            r['fixture'], r['z_um'], r['pixel_continuity'], r['fidelity'],
            r.get('power_over_oracle') or float('nan')))


def per_optic(rows, bar=1.06):
    fold = [r for r in rows if r.get('reason') == 'fold_ring'
            and not r.get('fell_back')]
    by = collections.defaultdict(list)
    for r in fold:
        by[r['fixture']].append(r)
    print('| optic | n | RETURNED C / fidelity | REFUSED C / fidelity |')
    print('|---|---|---|---|')
    for nm in sorted(by):
        rs = by[nm]
        a = [r for r in rs if r['pixel_continuity'] <= bar]
        b = [r for r in rs if r['pixel_continuity'] > bar]

        def f(p):
            if not p:
                return '--'
            return '%.4f..%.4f / %.4f..%.4f' % (
                min(r['pixel_continuity'] for r in p),
                max(r['pixel_continuity'] for r in p),
                min(r['fidelity'] for r in p), max(r['fidelity'] for r in p))
        print('| `%s` | %d | %s | %s |' % (nm, len(rs), f(a), f(b)))


def ladder_table(paths):
    print('| ladder | planes | fold-ring planes | fold-ring gap | '
          'fold margin above | fold margin below | all-planes gap | '
          'all margin above | all margin below |')
    print('|' + '---|' * 9)
    for p in paths:
        d = load(p)['summary']
        f = d['cost']['fold_ring']['1.06']
        a = d['cost']['all_planes']['1.06']
        print('| %s | %d | %d | %.5f | %.5f | %.5f | %.5f | %.5f | %.5f |' % (
            p.split('/')[-1].replace('joined_', '').replace('_win.json', ''),
            d['n_joined'], d['n_fold_ring'],
            f.get('gap', float('nan')), f.get('margin_above', float('nan')),
            f.get('margin_below', float('nan')),
            a.get('gap', float('nan')), a.get('margin_above', float('nan')),
            a.get('margin_below', float('nan'))))


def main():
    paths = sys.argv[1:]
    d = load(paths[0])
    S, rows = d['summary'], d['rows']
    print('## population\n')
    population_table(rows)
    print('\n## fold ring at the shipped bar\n')
    extremes(rows, 1.06, 'fold_ring')
    print('\n## per optic (fold ring)\n')
    per_optic(rows)
    print('\n## all planes at the shipped bar\n')
    extremes(rows, 1.06, 'all')
    print('\n## fallback at the shipped bar\n')
    extremes(rows, 1.06, 'fallback')
    print('\n## bar cost -- FOLD RING\n')
    cost_table(S, 'fold_ring')
    print('\n## bar cost -- ALL PLANES\n')
    cost_table(S, 'all_planes')
    print('\n## converged spread\n')
    print(json.dumps(S['converged_spread'], indent=1, default=float))
    print('\n## derived centre\n')
    print(json.dumps(S.get('derived_centre'), indent=1, default=float))
    if len(paths) > 1:
        print('\n## ladder depth\n')
        ladder_table(paths[::-1])


if __name__ == '__main__':
    main()
