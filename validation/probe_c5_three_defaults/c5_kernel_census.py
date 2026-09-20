"""A pytest plugin that censuses every leg the item-1 accuracy rule touches.

The blast radius of ``_GAP_KERNEL_ACCURACY_TAU`` is not "which files fail" but
"which legs resolved differently", so it is measured at the one function the
rule cannot take its decision without: ``_collins_exact_kernel_departure`` is
called once per leg that the ``k4`` representability gate has already resolved
to ``'exact'`` -- i.e. exactly the legs the rule is in a position to move.  The
decision itself is then read out of the CALLER's frame (``_kernel_asked``, the
resolved ``kernel``, ``z_eff``), so nothing has to be inferred from a call
sequence.

Only that one attribute is wrapped, and it is wrapped with ``functools.wraps``:
``test_wave5_h2_collins_jax.py`` inspects the SIGNATURE of the sibling
``_collins_exact_kernel_correction``, so wrapping that one turns a census into
a test failure.

    PYTHONPATH="<tree>;<this dir>" python -m pytest -p c5_kernel_census ...

writes ``C5_CENSUS_OUT`` (default ``c5_census.json``) at session end.
"""
import functools
import json
import os
import sys

_LEGS = []
_STATE = {'node': '<session>'}


def pytest_configure(config):
    from lumenairy.propagators import carrier as CA

    real = CA._collins_exact_kernel_departure

    @functools.wraps(real)
    def dep(z_eff, theta_env, wavelength, *a, **k):
        out = real(z_eff, theta_env, wavelength, *a, **k)
        try:
            caller = sys._getframe(1).f_locals
            asked = caller.get('_kernel_asked')
            resolved = caller.get('kernel')
        except Exception:                                  # noqa: BLE001
            asked = resolved = None
        tau = CA._GAP_KERNEL_ACCURACY_TAU
        fell = bool(tau is not None and out > float(tau)
                    and asked != 'exact')
        _LEGS.append({'node': _STATE['node'], 'z_eff': float(z_eff),
                      'theta_env': float(theta_env),
                      'wavelength': float(wavelength),
                      'departure': float(out),
                      'kernel_asked': asked, 'resolved_before_rule': resolved,
                      'tau': (None if tau is None else float(tau)),
                      'fell_back': fell})
        return out

    CA._collins_exact_kernel_departure = dep


def pytest_runtest_setup(item):
    _STATE['node'] = item.nodeid


def pytest_sessionfinish(session, exitstatus):
    fell = [g for g in _LEGS if g['fell_back']]
    by_node = {}
    for g in fell:
        by_node.setdefault(g['node'], []).append(
            {'z_eff': g['z_eff'], 'departure': g['departure'],
             'theta_env': g['theta_env'], 'kernel_asked': g['kernel_asked']})
    kept = {}
    for g in _LEGS:
        if not g['fell_back']:
            kept.setdefault(g['node'], 0)
            kept[g['node']] += 1
    out = {'n_exact_resolved_legs': len(_LEGS),
           'n_fell_back': len(fell),
           'n_nodes_touching_the_rule': len(set(g['node'] for g in _LEGS)),
           'nodes_with_fallback': by_node,
           'legs_kept_per_node': kept,
           'exitstatus': int(exitstatus)}
    path = os.environ.get('C5_CENSUS_OUT', 'c5_census.json')
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print('\n[c5-census] %d exact-resolved legs over %d ids, %d fell back '
          '-> %s' % (len(_LEGS), out['n_nodes_touching_the_rule'],
                     len(fell), path))
