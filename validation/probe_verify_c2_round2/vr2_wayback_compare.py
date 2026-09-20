"""Compare the four ``vr2_wayback.py`` arms.

Usage:  python vr2_wayback_compare.py <pre> <post_oldkw> <post_default>
                                      <post_none> <out.json>
"""
import json
import pathlib
import sys


def load(p):
    return json.loads(pathlib.Path(p).read_text(encoding='ascii'))


def compare(a, b):
    """(identical, moved, missing) over the union of case/array keys."""
    ident = moved = 0
    detail = {}
    cases = sorted(set(a['digests']) | set(b['digests']))
    for c in cases:
        da, db = a['digests'].get(c, {}), b['digests'].get(c, {})
        keys = sorted(set(da) | set(db))
        i = m = 0
        for k in keys:
            if k in da and k in db and da[k] == db[k]:
                i += 1
            else:
                m += 1
        ident += i
        moved += m
        detail[c] = {'identical': i, 'moved': m, 'total': len(keys)}
    return ident, moved, detail


def main():
    pre, oldkw, dflt, none_, out = sys.argv[1:6]
    A = load(pre)
    B = load(oldkw)
    C = load(dflt)
    D = load(none_)
    for name, arm in (('pre', A), ('post_oldkw', B), ('post_default', C),
                      ('post_none', D)):
        assert not arm['errors'], (name, sorted(arm['errors']))

    wb_i, wb_m, wb_d = compare(A, B)
    df_i, df_m, df_d = compare(A, C)
    nn_i, nn_m, nn_d = compare(C, D)

    ep_moved = {c: d['moved'] for c, d in df_d.items() if d['moved']}
    # collapse the per-fixture cases back to entry points
    def _ep(c):
        return c.split('[')[0]
    eps = sorted({_ep(c) for c in df_d})
    ep_summary = {}
    for e in eps:
        cs = [c for c in df_d if _ep(c) == e]
        ep_summary[e] = {
            'fixtures': sorted(cs),
            'wayback_identical': sum(wb_d[c]['identical'] for c in cs),
            'wayback_total': sum(wb_d[c]['total'] for c in cs),
            'arrays_moving_at_default': sum(df_d[c]['moved'] for c in cs),
            'none_identical': sum(nn_d[c]['identical'] for c in cs),
            'none_total': sum(nn_d[c]['total'] for c in cs),
        }

    res = {
        'n_cases': len(df_d),
        'n_entry_points': len(eps),
        'n_arrays': A['n_arrays'],
        'wayback_identical': wb_i, 'wayback_moved': wb_m,
        'default_identical': df_i, 'default_moved': df_m,
        'none_identical': nn_i, 'none_moved': nn_m,
        'entry_points_moving_at_default': sorted(
            {_ep(c) for c in ep_moved}),
        'entry_points_not_moving_at_default': sorted(
            e for e in eps if ep_summary[e]['arrays_moving_at_default'] == 0),
        'per_entry_point': ep_summary,
        'per_case_default_moved': {c: df_d[c]['moved'] for c in sorted(df_d)},
        'roots': {'pre': A['root'], 'post': B['root']},
        'python': A['python'], 'numpy': A['numpy'],
    }
    pathlib.Path(out).write_text(json.dumps(res, indent=1), encoding='ascii')
    print('arrays                        :', res['n_arrays'])
    print('way back  identical / moved   : %d / %d' % (wb_i, wb_m))
    print('default   identical / moved   : %d / %d' % (df_i, df_m))
    print('None==omitted ident / moved   : %d / %d' % (nn_i, nn_m))
    print('entry points moving at default: %d of %d'
          % (len(res['entry_points_moving_at_default']), len(eps)))
    for e in eps:
        s = ep_summary[e]
        print('   %-28s wayback %4d/%-4d  moved@default %4d  none %4d/%-4d'
              % (e, s['wayback_identical'], s['wayback_total'],
                 s['arrays_moving_at_default'], s['none_identical'],
                 s['none_total']))


if __name__ == '__main__':
    main()
