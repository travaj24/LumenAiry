"""WP-C4 -- a pytest plugin that censuses every shape the selection rule is
asked about during a test run.

    python -m pytest <files> -p c4_ratio_census_plugin ...
    (with this directory on PYTHONPATH; the JSON path comes from
     ``C4_RATIO_CENSUS_OUT``)

WHY.  The 55-file MFT-touching sweep moved no test.  "No test moved" has two
very different explanations -- the rule never fires at the shapes the suite
drives, or it fires and nothing notices -- and only a measurement separates
them.  This plugin wraps ``_auto_selects_direct`` and records, per test, every
``(Ny_in, Nx_in, N_out_y, N_out_x)`` it was asked about and what it answered.

It changes no answer: the wrapper calls the original and returns its result.
"""
from __future__ import annotations

import json
import os

_CALLS = []
_CURRENT = {'nodeid': None}


def pytest_configure(config):
    from lumenairy.propagators import _bluestein as B
    # THE SECOND ARM.  With ``C4_FORCE_NEVER=1`` the boundary constant is set
    # to ``_MFT_DIRECT_NEVER`` for the whole session, so the same files run on
    # the PRE-5.49.0 dispatch.  A file that is green under both settings has no
    # assertion coupled to which route ran -- which is what turns "nothing
    # failed" into a two-sided reading instead of a hope.
    if os.environ.get('C4_FORCE_NEVER') == '1':
        B._MFT_DIRECT_MAX_RATIO = B._MFT_DIRECT_NEVER
        print('[c4-ratio-census] _MFT_DIRECT_MAX_RATIO forced to '
              '_MFT_DIRECT_NEVER for this session')
    original = B._auto_selects_direct

    def recording(ny, nx, my, mx):
        ans = original(ny, nx, my, mx)
        _CALLS.append({'nodeid': _CURRENT['nodeid'],
                       'shape': [int(ny), int(nx), int(my), int(mx)],
                       'ratio': max(int(my) / int(ny), int(mx) / int(nx)),
                       'direct': bool(ans)})
        return ans

    recording.__wrapped__ = original
    B._auto_selects_direct = recording
    config._c4_original_rule = original


def pytest_unconfigure(config):
    from lumenairy.propagators import _bluestein as B
    if hasattr(config, '_c4_original_rule'):
        B._auto_selects_direct = config._c4_original_rule
    out = os.environ.get('C4_RATIO_CENSUS_OUT')
    if not out:
        return
    by_test = {}
    for c in _CALLS:
        e = by_test.setdefault(c['nodeid'] or '<collection>',
                               {'n': 0, 'n_direct': 0, 'min_ratio': None,
                                'direct_shapes': []})
        e['n'] += 1
        if c['direct']:
            e['n_direct'] += 1
            if c['shape'] not in e['direct_shapes']:
                e['direct_shapes'].append(c['shape'])
        r = c['ratio']
        e['min_ratio'] = r if e['min_ratio'] is None else min(e['min_ratio'], r)
    ratios = sorted({round(c['ratio'], 9) for c in _CALLS})
    summary = {
        'total_calls': len(_CALLS),
        'calls_answering_direct': sum(1 for c in _CALLS if c['direct']),
        'distinct_shapes': len({tuple(c['shape']) for c in _CALLS}),
        'distinct_ratios': len(ratios),
        'min_ratio_seen': ratios[0] if ratios else None,
        'ratios_below_1_over_16': [r for r in ratios if r <= 1 / 16.0],
        'tests_with_a_direct_call': sorted(
            k for k, v in by_test.items() if v['n_direct']),
    }
    with open(out, 'w', encoding='cp1252') as fh:
        json.dump({'summary': summary, 'by_test': by_test}, fh, indent=1,
                  sort_keys=True, default=str)
    print(f"\n[c4-ratio-census] {summary['total_calls']} calls, "
          f"{summary['calls_answering_direct']} answered 'direct', "
          f"{summary['distinct_shapes']} distinct shapes, "
          f"min ratio {summary['min_ratio_seen']}")
    print(f"[c4-ratio-census] -> {out}")


def pytest_runtest_protocol(item, nextitem):
    _CURRENT['nodeid'] = item.nodeid
    return None
