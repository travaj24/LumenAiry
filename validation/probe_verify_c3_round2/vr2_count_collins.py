"""VERIFY-WP-C3 ROUND 2, item 3 -- a pytest plugin that COUNTS, per test id,
how many legs reach ``_collins_transport`` and how many times WP-C5's
accuracy-keyed rule actually drops ``gap_kernel='auto'`` to ``'fresnel'``.

It is the measurement behind "were the d3 bars re-derived or only re-recorded
on the merged tree": a bar whose fixtures never reach the Collins transport
cannot have been moved by either 5.49.0 default.

Usage:  pytest -p vr2_count_collins <ids>   (this directory on PYTHONPATH)
"""
import json
import os

import lumenairy.propagators.carrier as C

COUNTS = {}
_CUR = ['<setup>']
_real_ct = C._collins_transport
_real_dep = C._collins_exact_kernel_departure


def _ct(*a, **k):
    COUNTS.setdefault(_CUR[0], {'collins_legs': 0, 'tau_evaluated': 0,
                                'tau_fired': 0, 'departures': []})
    COUNTS[_CUR[0]]['collins_legs'] += 1
    return _real_ct(*a, **k)


def _dep(z_eff, th, wl):
    d = _real_dep(z_eff, th, wl)
    r = COUNTS.setdefault(_CUR[0], {'collins_legs': 0, 'tau_evaluated': 0,
                                    'tau_fired': 0, 'departures': []})
    r['tau_evaluated'] += 1
    r['departures'].append(float(d))
    tau = C._GAP_KERNEL_ACCURACY_TAU
    if tau is not None and d > float(tau):
        r['tau_fired'] += 1
    return d


C._collins_transport = _ct
C._collins_exact_kernel_departure = _dep


def pytest_runtest_setup(item):
    _CUR[0] = item.nodeid


def pytest_sessionfinish(session, exitstatus):
    for v in COUNTS.values():
        d = v.pop('departures')
        v['departure_min'] = min(d) if d else None
        v['departure_max'] = max(d) if d else None
    p = os.environ.get('VR2_COUNT_OUT', 'vr2_collins_counts.json')
    with open(p, 'w', encoding='utf-8') as fh:
        json.dump({'tau': C._GAP_KERNEL_ACCURACY_TAU, 'counts': COUNTS},
                  fh, indent=1)
    print('\n[vr2] tau =', C._GAP_KERNEL_ACCURACY_TAU, '-> wrote', p)
