"""VERIFY WP-B13 follow-ups, D7 -- the join detector against a corpus I own.

Feeds >= 12 synthetic source shapes to BOTH detectors side by side:

  * SHIPPED  -- the pre-fix ``ast.Call``-only sweep, reconstructed here from
    ``git show b631ce79:tests/unit/test_fix_newton_pool_broken_fallback.py``
    so the comparison is against the code that actually shipped, not against
    a paraphrase of it;
  * EXTENDED -- ``_unbounded_executor_joins`` imported from the branch's test
    module, i.e. the thing under verification.

Every shape carries its OWN expected verdict (``want``), derived from the
semantics of the construct and not from either detector's output, so the
table below is a set of decisions and the two detectors are graded against
it: TP / FP / FN / TN per detector.

Also runs both detectors over the two real modules (``_lens_traced`` and
``propagators.carrier``) and reports what each sees.

Run:
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=<worktree> python validation/probe_verify_b13_followups/\
vf4_d7_detector_corpus.py --out <json>
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import inspect
import json
import os
import sys


# --------------------------------------------------------------------------
# The SHIPPED detector, transcribed from base b631ce79's
# test_only_the_bounded_helper_ever_joins_an_executor.  Returns the same
# 'owner:lineno' strings the shipped pin built.
# --------------------------------------------------------------------------
def shipped_detector(src, exempt=('_shutdown_pool_bounded',)):
    tree = ast.parse(src)
    owner, enclosing = {}, {}
    for parent in ast.walk(tree):
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(parent):
                if child is not parent:
                    owner[child] = parent.name
                    enclosing.setdefault(child, set()).add(parent.name)
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


# --------------------------------------------------------------------------
# The corpus.  (name, source, want) with want in {'offender', 'clean',
# 'informational'}.  Each function in a snippet is a separate shape; the
# snippet's name is the function's name.
# --------------------------------------------------------------------------
CORPUS = [
    # --- shapes that DO join a process pool unboundedly -------------------
    ('c01_with_plain',
     'def c01_with_plain(n):\n'
     '    with ProcessPoolExecutor(max_workers=n) as ex:\n'
     '        ex.submit(abs, -1)\n',
     'offender',
     'bare `with ProcessPoolExecutor(...)`: __exit__ IS shutdown(wait=True)'),
    ('c02_with_dotted',
     'import concurrent.futures\n'
     'def c02_with_dotted(n):\n'
     '    with concurrent.futures.ProcessPoolExecutor(n) as ex:\n'
     '        ex.submit(abs, -1)\n',
     'offender',
     'fully dotted concurrent.futures.ProcessPoolExecutor'),
    ('c03_with_aliased_module',
     'import concurrent.futures as cf\n'
     'def c03_with_aliased_module(n):\n'
     '    with cf.ProcessPoolExecutor(n) as ex:\n'
     '        ex.submit(abs, -1)\n',
     'offender',
     'module aliased, class name intact'),
    ('c04_with_aliased_class',
     'from concurrent.futures import ProcessPoolExecutor as PPE\n'
     'def c04_with_aliased_class(n):\n'
     '    with PPE(n) as ex:\n'
     '        ex.submit(abs, -1)\n',
     'offender',
     'CLASS aliased to a name that does not end in Executor'),
    ('c05_with_nested_in_function',
     'def c05_with_nested_in_function(n):\n'
     '    def inner():\n'
     '        with ProcessPoolExecutor(n) as ex:\n'
     '            ex.submit(abs, -1)\n'
     '    return inner\n',
     'offender',
     '`with` inside a nested def -- attribution must be the inner closure'),
    ('c06_shutdown_bare',
     'def c06_shutdown_bare(ex):\n'
     '    ex.shutdown()\n',
     'offender',
     'shutdown() with no wait argument defaults to wait=True'),
    ('c07_shutdown_wait_true_kw',
     'def c07_shutdown_wait_true_kw(ex):\n'
     '    ex.shutdown(wait=True)\n',
     'offender',
     'explicit wait=True keyword'),
    ('c08_shutdown_unbound_positional',
     'def c08_shutdown_unbound_positional(ex):\n'
     '    Executor.shutdown(ex, True)\n',
     'offender',
     'unbound call: first positional is self, wait is the SECOND'),
    ('c09_shutdown_wait_variable',
     'def c09_shutdown_wait_variable(ex, flag):\n'
     '    ex.shutdown(wait=flag)\n',
     'offender',
     'wait is not the constant False, so it may join'),
    ('c10_shutdown_cancel_only',
     'def c10_shutdown_cancel_only(ex):\n'
     '    ex.shutdown(cancel_futures=True)\n',
     'offender',
     'cancel_futures without wait=False still joins'),
    # --- shapes that do NOT join ------------------------------------------
    ('c11_shutdown_wait_false_kw',
     'def c11_shutdown_wait_false_kw(ex):\n'
     '    ex.shutdown(wait=False, cancel_futures=True)\n',
     'clean',
     'the non-joining retire used by _abandon_pool'),
    ('c12_shutdown_wait_false_positional',
     'def c12_shutdown_wait_false_positional(ex):\n'
     '    ex.shutdown(False)\n',
     'clean',
     'POSITIONAL wait=False -- the shipped detector false-positives here'),
    ('c13_shutdown_unbound_false',
     'def c13_shutdown_unbound_false(ex):\n'
     '    ProcessPoolExecutor.shutdown(ex, False)\n',
     'clean',
     'unbound, wait False in the second positional slot'),
    ('c14_not_an_executor',
     'def c14_not_an_executor(server):\n'
     '    server.shutdown(wait=True)\n',
     'offender',
     'any .shutdown(wait=True) is reported -- conservative BY DESIGN; '
     'graded as offender because that is the pin\'s stated contract'),
    # --- thread pools: informational --------------------------------------
    ('c15_with_threadpool',
     'def c15_with_threadpool(n):\n'
     '    with ThreadPoolExecutor(max_workers=n) as tp:\n'
     '        tp.submit(abs, -1)\n',
     'informational',
     'ThreadPoolExecutor has no _terminate_broken; out of scope but visible'),
    ('c16_with_threadpool_dotted',
     'import concurrent.futures as cf\n'
     'def c16_with_threadpool_dotted(n):\n'
     '    with cf.ThreadPoolExecutor(n) as tp:\n'
     '        tp.submit(abs, -1)\n',
     'informational',
     'dotted thread pool'),
    # --- other shapes a reader would expect to be seen ---------------------
    ('c17_with_prebuilt_name',
     'def c17_with_prebuilt_name(n):\n'
     '    ex = ProcessPoolExecutor(n)\n'
     '    with ex:\n'
     '        ex.submit(abs, -1)\n',
     'offender',
     '`with ex:` on a pre-built executor -- same __exit__, no Call node'),
    ('c18_with_two_items',
     'def c18_with_two_items(n, path):\n'
     '    with open(path) as fh, ProcessPoolExecutor(n) as ex:\n'
     '        ex.submit(abs, -1)\n',
     'offender',
     'multi-item with: the executor is the SECOND item'),
    ('c19_async_with',
     'async def c19_async_with(n):\n'
     '    async with ProcessPoolExecutor(n) as ex:\n'
     '        ex.submit(abs, -1)\n',
     'offender',
     'AsyncWith form'),
    ('c20_exempt_helper',
     'def _shutdown_pool_bounded(ex):\n'
     '    ex.shutdown(wait=True)\n',
     'clean',
     'the ONE place that is allowed to join, because it does so off-thread'),
]


def _import_branch_test_module(root):
    """Import the branch's test module without pytest, for its detector."""
    path = os.path.join(root, 'tests', 'unit',
                        'test_fix_newton_pool_broken_fallback.py')
    spec = importlib.util.spec_from_file_location('_b13_testmod', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['_b13_testmod'] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    import lumenairy
    print('lumenairy.__file__ =', lumenairy.__file__, flush=True)
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    tm = _import_branch_test_module(root)
    extended = tm._unbounded_executor_joins

    rows = []
    tally = {'shipped': {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0},
             'extended': {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}}
    for name, src, want, why in CORPUS:
        # Each snippet is graded on the function whose name matches `name`
        # (or, for c20, on the exempt helper).  `exempt` is left at the
        # library default so c20 exercises the exemption.
        sh = shipped_detector(src)
        off, info = extended(src)
        sh_hit = bool(sh)
        ex_hit = bool(off)
        ex_info = bool(info)
        for det, hit in (('shipped', sh_hit), ('extended', ex_hit)):
            if want == 'offender':
                tally[det]['TP' if hit else 'FN'] += 1
            elif want == 'clean':
                tally[det]['FP' if hit else 'TN'] += 1
            else:                                     # informational
                # A thread pool must NOT be an offender.  Counting it as a
                # clean case for the offender grade; the `informational`
                # channel is reported separately.
                tally[det]['FP' if hit else 'TN'] += 1
        rows.append({'shape': name, 'want': want, 'why': why,
                     'shipped_offenders': sh,
                     'extended_offenders': off,
                     'extended_informational': info,
                     'shipped_verdict': 'offender' if sh_hit else 'clean',
                     'extended_verdict': ('offender' if ex_hit else
                                          ('informational' if ex_info
                                           else 'clean')),
                     'shipped_correct': (sh_hit == (want == 'offender')),
                     'extended_correct': (
                         (ex_hit == (want == 'offender'))
                         and (want != 'informational' or ex_info))})

    # --- the two real modules --------------------------------------------
    from lumenairy.elements import _lens_traced as LT
    from lumenairy.propagators import carrier as CA
    real = {}
    for label, mod in (('lumenairy.elements._lens_traced', LT),
                       ('lumenairy.propagators.carrier', CA)):
        src = inspect.getsource(mod)
        sh = shipped_detector(src)
        off, info = extended(src)
        real[label] = {'shipped_offenders': sh,
                       'extended_offenders': off,
                       'extended_informational': info}

    out = {
        'python': sys.version,
        'lumenairy_file': lumenairy.__file__,
        'platform': sys.platform,
        'rows': rows,
        'tally': tally,
        'real_modules': real,
        'shipped_pin_is_blind_to_the_sibling_pool': (
            real['lumenairy.propagators.carrier']['shipped_offenders'] == []),
        'extended_pin_detects_the_sibling_pool': any(
            o.startswith('_multi_parallel_results:')
            for o in real['lumenairy.propagators.carrier'][
                'extended_offenders']),
        'lens_traced_clean_under_extended_pin': (
            real['lumenairy.elements._lens_traced'][
                'extended_offenders'] == []),
        'residual_blind_spots': [
            r['shape'] for r in rows
            if r['want'] == 'offender' and not r['extended_offenders']],
    }
    with open(args.out, 'w', encoding='cp1252', errors='replace') as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: out[k] for k in (
        'tally', 'shipped_pin_is_blind_to_the_sibling_pool',
        'extended_pin_detects_the_sibling_pool',
        'lens_traced_clean_under_extended_pin',
        'residual_blind_spots')}, indent=1))
    for r in rows:
        print(f"{r['shape']:32s} want={r['want']:14s} "
              f"shipped={r['shipped_verdict']:14s} "
              f"extended={r['extended_verdict']:14s} "
              f"ok(sh)={r['shipped_correct']!s:5s} "
              f"ok(ex)={r['extended_correct']}")
    for k, v in real.items():
        print(k, '->', v)


if __name__ == '__main__':
    main()
