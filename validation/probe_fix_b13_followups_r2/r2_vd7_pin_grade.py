"""FIX WP-B13 FOLLOW-UPS round 2, VD7/VD8 -- the D4 pin, re-graded.

VERIFY_WP-B13_FOLLOWUPS.md section 4 graded
``test_the_in_flight_counter_is_one_claim_per_dispatch_not_per_chunk``'s AST
check on five shapes and found it wrong in both directions:

  VD7  it MISSES the magnitude leaving the module as ``_note_pool_inflight``'s
       RETURN VALUE (``n = _note_pool_inflight(0); if n > 1:``), because no
       ``ast.Compare`` on the Name ``_POOL_INFLIGHT`` can see it;
  VD8  it FALSE-FAILS on ``0 < _POOL_INFLIGHT``, a legal spelling of the same
       zero-vs-non-zero decision, because the non-zero operands were read from
       ``node.comparators`` alone.

This probe runs the verifier's own grader (``vf6_d4_readers.branch_d4_check``,
imported from its file, never re-typed) and the RESTATED pin side by side over
the verifier's five shapes plus three more that isolate the return-value
route.  The restated check is transcribed from the shipped test and the
transcription is verified against the shipped file, so a copy is not the only
evidence: ``transcription_is_current`` is part of the verdict.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONPATH=<tree> python validation/probe_fix_b13_followups_r2/\
r2_vd7_pin_grade.py --out r2_vd7_win.json
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import inspect
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_VF6 = os.path.join(os.path.dirname(_HERE), 'probe_verify_b13_followups',
                    'vf6_d4_readers.py')
_TESTFILE = os.path.join(os.path.dirname(os.path.dirname(_HERE)),
                         'tests', 'unit',
                         'test_fix_newton_pool_broken_fallback.py')

IDENT = '_POOL_INFLIGHT'
CALL = '_note_pool_inflight'


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def restated_d4_check(src):
    """The RESTATED pin, transcribed from the shipped test.

    Two facts, both graded: every comparison involving the counter is against
    zero (read from BOTH operand slots), and every ``_note_pool_inflight``
    call site discards the return value.  The shipped test also asserts that
    each population is non-empty; those are premise gates on the real module,
    not shape grading, so they are reported separately by ``main``.
    """
    tree = ast.parse(src)
    against_non_zero = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        operands = [node.left, *node.comparators]
        if not any(isinstance(n, ast.Name) and n.id == IDENT
                   for n in operands):
            continue
        others = [x for x in operands
                  if not (isinstance(x, ast.Name) and x.id == IDENT)]
        if not all(isinstance(x, ast.Constant) and x.value == 0
                   for x in others):
            against_non_zero.append(node.lineno)
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == CALL]
    discarded = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                 and isinstance(n.value.func, ast.Name)
                 and n.value.func.id == CALL]
    ok = (against_non_zero == []) and (len(discarded) == len(calls))
    return 'PASS' if ok else 'FAIL'


def transcription_is_current(testsrc):
    """Is what this probe grades what the test actually ships?"""
    needed = [
        'operands = [node.left, *node.comparators]',
        "x.id == '_POOL_INFLIGHT')]",
        'discarded = [n for n in ast.walk(tree)',
        'assert len(discarded) == len(calls), (',
    ]
    return {k: (k in testsrc) for k in needed}


EXTRA_SHAPES = [
    ('claim as a statement',
     "_note_pool_inflight(1)\n", 'PASS',
     'the shipped shape: the return value is discarded'),
    ('return value bound to a name, never compared',
     "n = _note_pool_inflight(0)\n", 'FAIL',
     'the magnitude has left the module even without a Compare'),
    ('return value used inline',
     "if _note_pool_inflight(0) > 1:\n    pass\n", 'FAIL',
     'the VD7 route with no intermediate name at all'),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    import lumenairy
    print('lumenairy.__file__ =', lumenairy.__file__, flush=True)
    from lumenairy.elements import _lens_traced as LT

    VF6 = _load(_VF6, 'vf6_d4_readers')
    print('verifier probe =', VF6.__file__, flush=True)
    testsrc = open(_TESTFILE, encoding='cp1252', errors='replace').read()

    shapes = [(lbl, src, want, why)
              for lbl, src, want, why in VF6.D4_SHAPES] + EXTRA_SHAPES
    rows = []
    for label, src, want, why in shapes:
        before = VF6.branch_d4_check(src)
        after = restated_d4_check(src)
        rows.append({'shape': label, 'want': want,
                     'pin_before_vd7_vd8': before,
                     'pin_after': after,
                     'before_correct': before == want,
                     'after_correct': after == want,
                     'why': why})
        print(f'{label:50s} want={want:4s} before={before:4s} '
              f'after={after:4s}', flush=True)

    # The real module: the two premise gates the shipped test also makes.
    modsrc = inspect.getsource(LT)
    tree = ast.parse(modsrc)
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == CALL]
    discarded = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                 and isinstance(n.value.func, ast.Name)
                 and n.value.func.id == CALL]
    compares = [n.lineno for n in ast.walk(tree)
                if isinstance(n, ast.Compare)
                and any(isinstance(x, ast.Name) and x.id == IDENT
                        for x in [n.left, *n.comparators])]

    out = {'python': sys.version, 'platform': sys.platform,
           'lumenairy_file': lumenairy.__file__,
           'verifier_probe': VF6.__file__,
           'grades': rows,
           'real_module': {
               'note_pool_inflight_call_sites': [n.lineno for n in calls],
               'call_sites_that_discard_the_value': len(discarded),
               'compare_lines_on_the_counter': compares,
               'restated_pin_on_the_real_module': restated_d4_check(modsrc)},
           'transcription_is_current': transcription_is_current(testsrc)}
    out['verdict'] = {
        'misgrades_before': [r['shape'] for r in rows
                             if not r['before_correct']],
        'misgrades_after': [r['shape'] for r in rows
                            if not r['after_correct']],
        'restated_pin_green_on_the_real_module':
            out['real_module']['restated_pin_on_the_real_module'] == 'PASS',
        'transcription_is_current':
            all(out['transcription_is_current'].values()),
    }
    with open(args.out, 'w', encoding='cp1252', errors='replace') as fh:
        json.dump(out, fh, indent=1)
    print('VERDICT', json.dumps(out['verdict'], indent=1))
    print('wrote', args.out)


if __name__ == '__main__':
    main()
