"""VERIFY WP-B13 follow-ups, D4 -- is the ``_POOL_INFLIGHT`` reader list
complete, and is EVERY read really zero-vs-non-zero?

The follow-ups report's D4 decision ("fix the COMMENT, not the counter")
rests entirely on a claim about consumers: that nothing anywhere reads the
counter's MAGNITUDE.  The branch's pin checks one shape only -- every
``ast.Compare`` whose operands include the Name ``_POOL_INFLIGHT`` is
against the constant 0.  That misses any read that is not a comparison.

This probe enumerates the readers two independent ways over the WHOLE of
``lumenairy/``:

  grep   every textual occurrence of the identifier, file and line, with the
         line's source, so a reader outside `_lens_traced` cannot hide;
  AST    every ``ast.Name`` load of ``_POOL_INFLIGHT``, classified by the
         syntactic PARENT that consumes it -- Compare-against-zero,
         Compare-against-something-else, an f-string / format, a bare truth
         test, an assignment, an arithmetic operand, a call argument, a
         return.  Anything but the first two is a MAGNITUDE read and would
         refute the decision.

Also reports the module-level attribute reads a consumer outside the module
would use (``LT._POOL_INFLIGHT``), which the Name walk cannot see.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONPATH=<tree> python validation/probe_verify_b13_followups/\
vf6_d4_readers.py --out vf6_win.json
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys

IDENT = '_POOL_INFLIGHT'


def _classify(parent, node, field):
    """What does the syntactic parent DO with this read?"""
    if isinstance(parent, ast.Compare):
        others = [c for c in [parent.left, *parent.comparators] if c is not node]
        if all(isinstance(c, ast.Constant) and c.value == 0 for c in others):
            return 'compare_against_zero'
        return 'compare_against_non_zero'
    if isinstance(parent, ast.Assign) and field == 'value':
        return 'assigned_elsewhere'
    if isinstance(parent, (ast.BinOp, ast.AugAssign, ast.UnaryOp)):
        return 'arithmetic'
    if isinstance(parent, ast.FormattedValue):
        return 'formatted_into_a_string'
    if isinstance(parent, ast.Call):
        return 'call_argument'
    if isinstance(parent, ast.Return):
        return 'returned'
    if isinstance(parent, (ast.If, ast.While, ast.BoolOp, ast.IfExp)):
        return 'bare_truth_test'
    if isinstance(parent, ast.Subscript):
        return 'subscript'
    return f'other:{type(parent).__name__}'


MAGNITUDE_SHAPES = {'compare_against_non_zero', 'arithmetic',
                    'formatted_into_a_string', 'call_argument', 'returned',
                    'subscript'}


def scan_tree(root):
    grep_rows, ast_rows, attr_rows = [], [], []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in {'.git', '__pycache__', '.pytest_cache'}]
        for fn in filenames:
            if not fn.endswith('.py'):
                continue
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, os.path.dirname(root))
            try:
                src = open(path, encoding='utf-8', errors='replace').read()
            except OSError:
                continue
            if IDENT not in src:
                continue
            for i, line in enumerate(src.splitlines(), 1):
                if IDENT in line:
                    grep_rows.append({'file': rel.replace('\\', '/'),
                                      'line': i, 'source': line.strip()})
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            parents = {}
            for p in ast.walk(tree):
                for field, value in ast.iter_fields(p):
                    if isinstance(value, ast.AST):
                        parents[value] = (p, field)
                    elif isinstance(value, list):
                        for v in value:
                            if isinstance(v, ast.AST):
                                parents[v] = (p, field)
            for node in ast.walk(tree):
                if (isinstance(node, ast.Name) and node.id == IDENT
                        and isinstance(node.ctx, ast.Load)):
                    p, field = parents.get(node, (None, None))
                    ast_rows.append({
                        'file': rel.replace('\\', '/'),
                        'line': node.lineno,
                        'parent': type(p).__name__ if p else None,
                        'shape': _classify(p, node, field) if p else 'toplevel',
                    })
                if (isinstance(node, ast.Attribute) and node.attr == IDENT
                        and isinstance(node.ctx, ast.Load)):
                    p, field = parents.get(node, (None, None))
                    attr_rows.append({
                        'file': rel.replace('\\', '/'),
                        'line': node.lineno,
                        'shape': _classify(p, node, field) if p else 'toplevel',
                    })
    return grep_rows, ast_rows, attr_rows


def branch_d4_check(src):
    """The branch's D4 AST check, extracted VERBATIM from
    ``test_the_in_flight_counter_is_one_claim_per_dispatch_not_per_chunk``,
    so its reach can be graded against shapes it was never shown.
    """
    tree = ast.parse(src)
    comparisons = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        names = [n.id for n in [node.left, *node.comparators]
                 if isinstance(n, ast.Name)]
        if IDENT not in names:
            continue
        others = [c for c in node.comparators
                  if not (isinstance(c, ast.Constant) and c.value == 0)]
        comparisons.append((node.lineno, bool(others)))
    against_non_zero = [ln for ln, nz in comparisons if nz]
    return 'PASS' if against_non_zero == [] else 'FAIL'


def _src(*lines):
    """A snippet, assembled line by line so no escape can be mangled."""
    return chr(10).join(lines) + chr(10)


D4_SHAPES = [
    ('shipped guard `_POOL_INFLIGHT > 0`',
     _src('if _POOL_INFLIGHT > 0:', '    pass'), 'PASS',
     'zero-vs-non-zero, written the way the library writes it'),
    ('legal reversal `0 < _POOL_INFLIGHT`',
     _src('if 0 < _POOL_INFLIGHT:', '    pass'), 'PASS',
     'the SAME zero-vs-non-zero decision with the operands swapped'),
    ('legal `_POOL_INFLIGHT == 0`',
     _src('if _POOL_INFLIGHT == 0:', '    pass'), 'PASS',
     'zero-vs-non-zero'),
    ('magnitude `_POOL_INFLIGHT > 1`',
     _src('if _POOL_INFLIGHT > 1:', '    pass'), 'FAIL',
     'a real magnitude read -- the pin must catch it'),
    ('magnitude via the return value',
     _src('n = _note_pool_inflight(0)', 'if n > 1:', '    pass'), 'FAIL',
     'the counter leaves the module as a RETURN VALUE; no Compare on '
     '_POOL_INFLIGHT can see it'),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    import lumenairy
    print('lumenairy.__file__ =', lumenairy.__file__, flush=True)
    pkg = os.path.dirname(os.path.abspath(lumenairy.__file__))
    repo = os.path.dirname(pkg)

    lib_grep, lib_ast, lib_attr = scan_tree(pkg)
    tests_dir = os.path.join(repo, 'tests')
    val_dir = os.path.join(repo, 'validation')
    t_grep, t_ast, t_attr = scan_tree(tests_dir) if os.path.isdir(
        tests_dir) else ([], [], [])
    v_grep, v_ast, v_attr = scan_tree(val_dir) if os.path.isdir(
        val_dir) else ([], [], [])

    # the dispatcher's claim/release sites, counted
    import inspect

    from lumenairy.elements import _lens_traced as LT
    src_mod = inspect.getsource(LT)
    claims = src_mod.count('_note_pool_inflight(1)')
    releases = src_mod.count('_note_pool_inflight(-1)')

    lib_mag = [r for r in lib_ast + lib_attr
               if r['shape'] in MAGNITUDE_SHAPES]
    all_mag = [r for r in lib_ast + lib_attr + t_ast + t_attr + v_ast + v_attr
               if r['shape'] in MAGNITUDE_SHAPES]

    d4_rows = []
    for label, src, want, why in D4_SHAPES:
        got = branch_d4_check(src)
        d4_rows.append({'shape': label, 'want': want, 'branch_pin': got,
                        'correct': got == want, 'why': why})

    out = {
        'branch_d4_pin_reach': d4_rows,
        'branch_d4_pin_misgrades': [r['shape'] for r in d4_rows
                                    if not r['correct']],
        'python': sys.version, 'platform': sys.platform,
        'lumenairy_file': lumenairy.__file__,
        'library': {'grep_hits': lib_grep, 'name_loads': lib_ast,
                    'attribute_loads': lib_attr},
        'tests': {'grep_hits': len(t_grep), 'name_loads': t_ast,
                  'attribute_loads': t_attr},
        'validation': {'grep_hits': len(v_grep), 'name_loads': v_ast,
                       'attribute_loads': v_attr},
        'dispatcher': {'note_pool_inflight_plus_one': claims,
                       'note_pool_inflight_minus_one': releases},
        'verdict': {
            'library_readers': sorted({
                f"{r['file']}:{r['line']}:{r['shape']}"
                for r in lib_ast + lib_attr}),
            'library_magnitude_reads': lib_mag,
            'every_library_read_is_zero_vs_non_zero': lib_mag == [],
            'magnitude_reads_anywhere_in_the_repo': all_mag,
            'claim_is_once_per_dispatch': claims == 1,
        },
    }
    with open(args.out, 'w', encoding='cp1252', errors='replace') as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out['verdict'], indent=1))
    print('--- the branch D4 pin, graded ---')
    for r in d4_rows:
        print(f"  {r['shape']:42s} want={r['want']:4s} got={r['branch_pin']:4s}"
              f"  ok={r['correct']}")
    print('--- library grep hits ---')
    for r in lib_grep:
        print(f"  {r['file']}:{r['line']}  {r['source'][:100]}")
    print('--- test/validation loads ---')
    for r in t_ast + t_attr + v_ast + v_attr:
        print(f"  {r['file']}:{r['line']}  {r['shape']}")
    print('wrote', args.out)


if __name__ == '__main__':
    main()
