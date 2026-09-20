"""VERIFY-WP-C4 claim 9 -- MY OWN pytest plugin that counts where the rule
fires in the shipped suite.

Wraps :func:`lumenairy.propagators._bluestein._auto_selects_direct` for a whole
session and records, per test id, every shape it is asked about and what it
answered.  It CHANGES NO ANSWER: the wrapper calls the original and returns its
result unmodified.

Independent of ``probe_c4_mft_direct/c4_ratio_census_plugin.py``: this one also
records the CALLER of each rule call (the immediate frame under
``_bluestein.py``), so "which entry point reached the rule" is a measurement
rather than a grep, and it records the call's primitive.

    pytest <files> -p no:randomly --capture=sys \\
        -p validation.probe_verify_c4.v4_census_plugin \\
        --v4-census-out=<path.json>
"""
from __future__ import annotations

import json
import os
import sys
import traceback

_STATE = {'calls': [], 'current': None, 'out': None, 'orig': None}


def pytest_addoption(parser):
    parser.addoption('--v4-census-out', action='store', default=None,
                     help='where to write the rule census JSON')


def pytest_configure(config):
    _STATE['out'] = config.getoption('--v4-census-out')
    if not _STATE['out']:
        return
    from lumenairy.propagators import _bluestein as B
    orig = B._auto_selects_direct
    _STATE['orig'] = orig

    def spy(Ny_in, Nx_in, N_out_y, N_out_x):
        ans = orig(Ny_in, Nx_in, N_out_y, N_out_x)
        caller = ''
        try:
            st = traceback.extract_stack()
            # the first frame OUTSIDE _bluestein.py, walking outwards
            for fr in reversed(st[:-1]):
                fn = os.path.basename(fr.filename)
                if fn not in ('_bluestein.py', 'v4_census_plugin.py'):
                    caller = f"{fn}:{fr.lineno}:{fr.name}"
                    break
        except Exception:                                # noqa: BLE001
            caller = '<unavailable>'
        _STATE['calls'].append({
            'nodeid': _STATE['current'],
            'shape': [int(Ny_in), int(Nx_in), int(N_out_y), int(N_out_x)],
            'answer': bool(ans), 'caller': caller})
        return ans

    B._auto_selects_direct = spy


def pytest_runtest_protocol(item, nextitem):
    _STATE['current'] = item.nodeid
    return None


def pytest_sessionfinish(session, exitstatus):
    if not _STATE['out']:
        return
    from lumenairy.propagators import _bluestein as B
    if _STATE['orig'] is not None:
        B._auto_selects_direct = _STATE['orig']
    calls = _STATE['calls']
    direct = [c for c in calls if c['answer']]
    by_id = {}
    for c in direct:
        by_id.setdefault(c['nodeid'], []).append(c)
    files = {}
    for nid, cs in by_id.items():
        files.setdefault(str(nid).split('::')[0], []).append(nid)
    ratios = sorted({max(c['shape'][2] / c['shape'][0],
                         c['shape'][3] / c['shape'][1]) for c in calls})
    out = {
        'python': sys.version.split()[0],
        'platform': sys.platform,
        'exitstatus': int(exitstatus),
        'total_rule_calls': len(calls),
        'calls_answering_direct': len(direct),
        'distinct_shapes': len({tuple(c['shape']) for c in calls}),
        'distinct_ratios': len(ratios),
        'smallest_ratio': (min(ratios) if ratios else None),
        'ids_where_the_rule_fires': sorted(by_id),
        'n_ids_where_the_rule_fires': len(by_id),
        'files_where_the_rule_fires': {k: sorted(v)
                                       for k, v in sorted(files.items())},
        'n_files': len(files),
        'direct_shapes': sorted({tuple(c['shape']) for c in direct}),
        'direct_callers': sorted({c['caller'] for c in direct}),
        'all_calls': calls,
    }
    with open(_STATE['out'], 'w', encoding='cp1252', errors='replace') as fh:
        json.dump(out, fh, indent=1, sort_keys=True, default=str)
    print(f"\n[v4-census] {len(calls)} rule calls, {len(direct)} answering "
          f"'direct', in {len(by_id)} ids across {len(files)} files "
          f"-> {_STATE['out']}")
