"""VERIFY-WP-C4 ROUND 2, item 9 -- a pytest plugin that counts what the rule
is ASKED and what it ANSWERS over a whole session.

Written independently of ``validation/probe_verify_c4/v4_census_plugin.py``.
It wraps ``_auto_selects_direct`` at every module attribute that holds it (a
``from ... import`` binds the function OBJECT, so patching only the defining
module misses the primitives' own module-level name), forwards every call to
the original and changes no answer, and records the four grid sizes, the
answer, the immediate caller and the test id.

    pytest <files> -p no:randomly -q --capture=sys \\
        -p vc4b_census_plugin --vc4b-census-out=<path>

Author:  Andrew Traverso
"""
from __future__ import annotations

import json
import os
import sys


def pytest_addoption(parser):
    parser.addoption('--vc4b-census-out', action='store', default=None,
                     help='where to write the rule census JSON')


class _Census:
    def __init__(self, path):
        self.path = path
        self.calls = []
        self.current = None

    def record(self, ny, nx, my, mx, answer, caller):
        self.calls.append({'Ny_in': int(ny), 'Nx_in': int(nx),
                           'My': int(my), 'Mx': int(mx),
                           'answer': bool(answer), 'caller': caller,
                           'nodeid': self.current})


_STATE = {}


def pytest_configure(config):
    out = config.getoption('--vc4b-census-out')
    if not out:
        return
    import inspect

    from lumenairy.propagators import _bluestein as B
    original = B._auto_selects_direct
    census = _Census(out)
    _STATE['census'] = census
    _STATE['original'] = original
    _STATE['holders'] = []

    def wrapped(ny, nx, my, mx):
        ans = original(ny, nx, my, mx)
        frame = inspect.currentframe().f_back
        caller = "%s %s" % (os.path.basename(frame.f_code.co_filename),
                            frame.f_code.co_name)
        census.record(ny, nx, my, mx, ans, caller)
        return ans

    for mod in list(sys.modules.values()):
        if not getattr(mod, '__name__', '').startswith('lumenairy'):
            continue
        if not hasattr(mod, '__dict__'):
            continue
        for attr in list(vars(mod)):
            try:
                if getattr(mod, attr) is original:
                    setattr(mod, attr, wrapped)
                    _STATE['holders'].append("%s.%s" % (mod.__name__, attr))
            except Exception:                                # noqa: BLE001
                continue


def pytest_runtest_logstart(nodeid, location):
    c = _STATE.get('census')
    if c is not None:
        c.current = nodeid


def pytest_sessionfinish(session, exitstatus):
    c = _STATE.get('census')
    if c is None:
        return
    calls = c.calls
    direct = [x for x in calls if x['answer']]
    shapes = sorted({(x['Ny_in'], x['Nx_in'], x['My'], x['Mx'])
                     for x in calls})
    dshapes = sorted({(x['Ny_in'], x['Nx_in'], x['My'], x['Mx'])
                      for x in direct})
    ids = sorted({x['nodeid'] for x in direct if x['nodeid']})
    files = sorted({i.split('::')[0] for i in ids})
    callers = {}
    for x in direct:
        callers[x['caller']] = callers.get(x['caller'], 0) + 1
    per_file = {}
    for i in ids:
        f = os.path.basename(i.split('::')[0])
        per_file[f] = per_file.get(f, 0) + 1
    ratios = sorted({min(x['My'] / x['Ny_in'], x['Mx'] / x['Nx_in'])
                     for x in calls})
    out = {'patched_holders': _STATE['holders'],
           'exitstatus': int(exitstatus),
           'python': sys.version.split()[0], 'platform': sys.platform,
           'total_rule_calls': len(calls),
           'calls_answering_direct': len(direct),
           'n_ids_where_the_rule_fires': len(ids),
           'ids_where_the_rule_fires': ids,
           'n_files': len(files), 'files_where_the_rule_fires': files,
           'ids_per_file': per_file,
           'distinct_shapes_asked_about': len(shapes),
           'direct_shapes': ["%dx%d->%dx%d" % s for s in dshapes],
           'direct_callers': callers,
           'smallest_ratio_seen': ratios[0] if ratios else None}
    with open(c.path, 'w', encoding='cp1252', errors='replace') as fh:
        json.dump(out, fh, indent=1, sort_keys=True, default=str)
    print("\n[vc4b census] %d rule calls, %d direct, %d ids, %d files -> %s"
          % (out['total_rule_calls'], out['calls_answering_direct'],
             out['n_ids_where_the_rule_fires'], out['n_files'], c.path))
