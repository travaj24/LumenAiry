"""FU2 -- the AST pin's blind spot, stated as a measurement (D7).

VERIFY-WP-B13 defect D7: the shipped
``test_only_the_bounded_helper_ever_joins_an_executor`` walked for
``ast.Call`` nodes whose ``func.attr == 'shutdown'``.  A joining teardown
written as ``with ProcessPoolExecutor(...) as ex:`` contains no such call --
``Executor.__exit__`` IS ``shutdown(wait=True)`` -- so the pin stayed green
on exactly the shape the sibling pool
``lumenairy.propagators.carrier._multi_parallel_results`` uses.

This probe runs BOTH detectors over BOTH modules and over a synthetic source
that carries every shape on purpose, and writes the readings.  It is the
fail-before for the extension: the shipped detector must report nothing in
``carrier.py``; the extended one must report the ``with`` site.

    PYTHONPATH=<tree> python validation/probe_wp_b13_followups/
    fu2_d7_pin_detectors.py --out fu2_win.json
"""
from __future__ import annotations

import argparse
import ast
import inspect
import json
import os
import sys

_SYNTHETIC = (
    'def teardown_with(n):\n'
    '    with ProcessPoolExecutor(max_workers=n) as ex:\n'
    '        ex.submit(abs, -1)\n'
    'def teardown_call(ex):\n'
    '    ex.shutdown(wait=True)\n'
    'def teardown_unbound(ex):\n'
    '    Executor.shutdown(ex, True)\n'
    'def retire(ex):\n'
    '    ex.shutdown(wait=False, cancel_futures=True)\n'
    'def retire_positional(ex):\n'
    '    ex.shutdown(False)\n'
    'def threads(n):\n'
    '    with ThreadPoolExecutor(max_workers=n) as tp:\n'
    '        tp.submit(abs, -1)\n')


def _owners(tree):
    owner, enclosing = {}, {}
    for parent in ast.walk(tree):
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(parent):
                if child is not parent:
                    owner[child] = parent.name
                    enclosing.setdefault(child, set()).add(parent.name)
    return owner, enclosing


def shipped_detector(src, exempt=('_shutdown_pool_bounded',)):
    """The pin exactly as WP-B13 shipped it: ``ast.Call`` / ``.shutdown``."""
    tree = ast.parse(src)
    owner, enclosing = _owners(tree)
    offenders = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'shutdown'):
            continue
        waits = [kw.value for kw in node.keywords if kw.arg == 'wait']
        joining = (not waits) or any(
            not (isinstance(v, ast.Constant) and v.value is False)
            for v in waits)
        if joining and not (set(exempt) & enclosing.get(node, set())):
            offenders.append(f'{owner.get(node)}:{node.lineno}')
    return sorted(offenders)


def _load_extended():
    """The extended detector, imported FROM THE TEST that owns it.

    Re-implementing it here would put the probe and the gate on two
    definitions of "unbounded join", and the first time they disagreed the
    probe would be measuring itself.
    """
    import importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, '..', '..'))
    path = os.path.join(root, 'tests', 'unit',
                        'test_fix_newton_pool_broken_fallback.py')
    spec = importlib.util.spec_from_file_location('_b13_pin_src', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._unbounded_executor_joins, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='fu2.json')
    args = ap.parse_args()

    import lumenairy as la
    from lumenairy.elements import _lens_traced as LT
    from lumenairy.propagators import carrier as CA
    print('lumenairy.__file__ =', la.__file__, flush=True)

    extended, pin_path = _load_extended()
    out = {'lumenairy': la.__file__, 'python': sys.version,
           'platform': sys.platform, 'pin_source': pin_path, 'cases': {}}
    for label, src, exempt in (
            ('synthetic', _SYNTHETIC, ()),
            ('lumenairy.elements._lens_traced', inspect.getsource(LT),
             ('_shutdown_pool_bounded',)),
            ('lumenairy.propagators.carrier', inspect.getsource(CA), ())):
        off, info = extended(src, exempt=exempt)
        out['cases'][label] = {
            'shipped_detector_offenders': shipped_detector(src, exempt),
            'extended_detector_offenders': off,
            'extended_detector_informational': info}
        print(label, json.dumps(out['cases'][label]), flush=True)

    car = out['cases']['lumenairy.propagators.carrier']
    out['verdict'] = {
        'shipped_pin_is_blind_to_the_sibling_pool':
            car['shipped_detector_offenders'] == [],
        'extended_pin_detects_the_sibling_pool':
            any(o.startswith('_multi_parallel_results:')
                for o in car['extended_detector_offenders']),
        'lens_traced_clean_under_extended_pin':
            out['cases']['lumenairy.elements._lens_traced'][
                'extended_detector_offenders'] == []}
    print('VERDICT', json.dumps(out['verdict']), flush=True)
    here = os.path.dirname(os.path.abspath(__file__))
    path = args.out if os.path.isabs(args.out) else os.path.join(here,
                                                                 args.out)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1)
    print('WROTE', path, flush=True)


if __name__ == '__main__':
    main()
