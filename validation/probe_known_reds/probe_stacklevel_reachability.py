"""Which modules does the b11 ratchet's RATIONALE cover?

The ratchet exists because a literal ``stacklevel`` encodes ONE call depth.  It
is therefore wrong wherever a warning site can be reached at MORE THAN ONE
depth.  Statically, that is the case when the function that emits (or forwards)
the warning is called from inside the package as well as from outside it: the
in-package call adds a library frame the literal did not count.

This probe computes, per module under ``lumenairy/propagators/``:

  * the ``warnings.warn`` sites with a LITERAL stacklevel,
  * for each, the enclosing top-level function,
  * whether that function (or any in-module function that reaches it through a
    same-module call chain) is ALSO called from elsewhere in the package.

A module with at least one such site is covered by the rationale.  This is a
conservative screen, not a proof of misattribution -- a site it flags is one to
MEASURE with ``probe_carrier_attribution.py``-style instrumentation.
"""
import ast
import json
import os
import pathlib
import sys
from collections import defaultdict

REPO = pathlib.Path(__file__).resolve().parents[2]
PKG = REPO / 'lumenairy'


def module_functions(tree):
    """{function name: (lineno, end_lineno)} for top-level and nested defs."""
    out = {}
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[n.name] = (n.lineno, n.end_lineno)
    return out


def enclosing(funcs, lineno):
    best, span = None, None
    for name, (a, b) in funcs.items():
        if a <= lineno <= b and (span is None or (b - a) < span):
            best, span = name, b - a
    return best


def literal_warn_sites(tree):
    out = []
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == 'warn'):
            continue
        for k in n.keywords:
            if (k.arg == 'stacklevel' and isinstance(k.value, ast.Constant)
                    and isinstance(k.value.value, int)):
                out.append((n.lineno, k.value.value))
    return out


def calls_in(tree):
    """{caller function name: {callee simple names}} for same-module edges."""
    edges = defaultdict(set)
    funcs = module_functions(tree)
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        name = getattr(n.func, 'id', getattr(n.func, 'attr', None))
        if name is None:
            continue
        host = enclosing(funcs, n.lineno)
        if host:
            edges[host].add(name)
    return edges


def main():
    package_call_names = defaultdict(set)      # name -> {files calling it}
    trees = {}
    for p in sorted(PKG.rglob('*.py')):
        try:
            t = ast.parse(p.read_text(encoding='utf-8'))
        except SyntaxError:
            continue
        trees[p] = t
        for n in ast.walk(t):
            if isinstance(n, ast.Call):
                nm = getattr(n.func, 'id', getattr(n.func, 'attr', None))
                if nm:
                    package_call_names[nm].add(p.relative_to(REPO).as_posix())

    rows = []
    for p, t in trees.items():
        rel = p.relative_to(REPO).as_posix()
        if '/propagators/' not in rel:
            continue
        sites = literal_warn_sites(t)
        if not sites:
            continue
        funcs = module_functions(t)
        edges = calls_in(t)
        # reverse reachability inside the module
        reach = defaultdict(set)
        for host, callees in edges.items():
            for c in callees:
                if c in funcs:
                    reach[c].add(host)
        flagged = []
        for lineno, lvl in sites:
            host = enclosing(funcs, lineno)
            chain, seen = {host}, [host]
            while seen:
                cur = seen.pop()
                for up in reach.get(cur, ()):  # callers inside this module
                    if up not in chain:
                        chain.add(up)
                        seen.append(up)
            callers = set()
            for nm in chain:
                if nm is None:
                    continue
                callers |= {f for f in package_call_names.get(nm, set())
                            if f != rel}
                if len(reach.get(nm, ())) > 0:
                    callers.add(rel + ' (same module)')
            flagged.append({'line': lineno, 'stacklevel': lvl,
                            'enclosing': host,
                            'reachable_from': sorted(callers)[:6],
                            'multi_depth': bool(callers)})
        rows.append({'file': rel, 'n_literal': len(sites),
                     'n_multi_depth': sum(1 for f in flagged
                                          if f['multi_depth']),
                     'sites': flagged})
    rows.sort(key=lambda r: -r['n_multi_depth'])
    print(f"{'lits':>5} {'multi':>6}  file")
    for r in rows:
        print(f"{r['n_literal']:5d} {r['n_multi_depth']:6d}  {r['file']}")
    out = {'rows': rows,
           'covered_by_rationale': [r['file'] for r in rows
                                    if r['n_multi_depth']]}
    print('covered by the rationale:', out['covered_by_rationale'])
    tag = os.environ.get('PROBE_TAG', 'default')
    dest = pathlib.Path(__file__).parent / f'stacklevel_reach_{tag}.json'
    dest.write_text(json.dumps(out, indent=1), encoding='utf-8')
    print('wrote', dest)
    return 0


if __name__ == '__main__':
    sys.exit(main())
