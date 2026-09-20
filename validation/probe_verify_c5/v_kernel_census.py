"""VERIFY-WP-C5: a pytest plugin that censuses every leg the item-1 accuracy
rule is in a position to move -- optionally with ``transport='collins'``
FORCED, which is the WP-C3 interaction the C5 report flags but does not
quantify.

Why one function.  ``_collins_exact_kernel_departure`` is called exactly once
per leg the ``k4`` representability gate has already resolved to ``'exact'``,
which is exactly the set the rule can act on.  The decision is read out of the
CALLER's frame (``_kernel_asked``, the resolved ``kernel``, ``A``/``B``), so
nothing is inferred from a call sequence, and only that one attribute is
wrapped (``test_wave5_h2_collins_jax.py`` inspects the SIGNATURE of the sibling
``_collins_exact_kernel_correction``; wrapping that one turns a census into a
failure).  ``functools.wraps`` keeps the signature intact here too.

``C5V_FORCE_COLLINS=1`` rewrites the ``transport`` keyword default of
``propagate_traced_carrier_chain`` and
``propagate_traced_carrier_chain_multi`` (and of the ``lumenairy`` re-exports,
which are the same function objects) from ``'sziklas'`` to ``'collins'``
BEFORE collection, i.e. it simulates WP-C3's flip for every call site that does
not name the transport itself.  Tests that assert the sziklas route WILL fail
under it; the census is still valid, because a leg is recorded when it runs and
the failures are reported alongside.

    PYTHONPATH="<tree>;<this dir>" python -m pytest -p v_kernel_census ...

writes ``C5V_CENSUS_OUT`` (default ``v_census.json``) at session end.
"""
import functools
import json
import os
import sys

_LEGS = []
_STATE = {'node': '<session>'}
_FORCED = {}


def pytest_configure(config):
    from lumenairy.propagators import carrier as CA

    if os.environ.get('C5V_FORCE_COLLINS') == '1':
        for name in ('propagate_traced_carrier_chain',
                     'propagate_traced_carrier_chain_multi'):
            fn = getattr(CA, name)
            kd = dict(fn.__kwdefaults__ or {})
            _FORCED[name] = kd.get('transport')
            kd['transport'] = 'collins'
            fn.__kwdefaults__ = kd

    real = CA._collins_exact_kernel_departure

    @functools.wraps(real)
    def dep(z_eff, theta_env, wavelength, *a, **k):
        out = real(z_eff, theta_env, wavelength, *a, **k)
        try:
            caller = sys._getframe(1).f_locals
            asked = caller.get('_kernel_asked')
            resolved = caller.get('kernel')
            fnname = caller.get('fn')
        except Exception:                                  # noqa: BLE001
            asked = resolved = fnname = None
        tau = CA._GAP_KERNEL_ACCURACY_TAU
        fell = bool(tau is not None and out > float(tau)
                    and asked != 'exact')
        _LEGS.append({'node': _STATE['node'], 'z_eff': float(z_eff),
                      'theta_env': float(theta_env),
                      'wavelength': float(wavelength),
                      'departure': float(out),
                      'kernel_asked': asked, 'resolved_before_rule': resolved,
                      'fn': (str(fnname) if fnname is not None else None),
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
             'theta_env': g['theta_env'], 'kernel_asked': g['kernel_asked'],
             'fn': g['fn'], 'wavelength': g['wavelength']})
    kept = {}
    for g in _LEGS:
        kept.setdefault(g['node'], [0, 0])
        kept[g['node']][1 if g['fell_back'] else 0] += 1
    out = {'n_exact_resolved_legs': len(_LEGS),
           'n_fell_back': len(fell),
           'n_fell_back_excluding_direct_calls':
               len([g for g in fell if g['kernel_asked'] is not None]),
           'n_nodes_touching_the_rule': len(set(g['node'] for g in _LEGS)),
           'n_nodes_with_fallback': len(by_node),
           'forced_collins': os.environ.get('C5V_FORCE_COLLINS') == '1',
           'restored_transport_defaults': _FORCED,
           'nodes_with_fallback': by_node,
           'legs_kept_and_fallen_per_node': kept,
           'exitstatus': int(exitstatus)}
    path = os.environ.get('C5V_CENSUS_OUT', 'v_census.json')
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print('\n[c5v-census] %d exact-resolved legs over %d ids, %d fell back '
          '(forced_collins=%s) -> %s'
          % (len(_LEGS), out['n_nodes_touching_the_rule'], len(fell),
             out['forced_collins'], path))
