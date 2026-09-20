"""VERIFY-WP-C2 item 5 -- every PUBLIC entry point that traces internally,
enumerated from the PACKAGE (an AST walk), not from the report.

The campaign's rule is that every moved public entry point has a one-keyword
way back.  ``trace`` / ``trace_world`` moved their defaults, so anything that
calls them INTERNALLY moved too -- and can only be put back if it forwards
``sphere_normal`` / ``renormalize``.

This walks every module under ``lumenairy/``, finds every function whose body
calls a tracer, keeps the ones that are exported (or are public names on an
exported module), and records for each whether its own signature carries the
two keywords.  The result is the way-back census, per entry point.

Usage: ``LUMENAIRY_ROOT=<root> python vc2_entrypoints.py OUT.json``
"""
import ast
import importlib
import inspect
import json
import os
import pathlib
import sys

_ROOT = os.environ['LUMENAIRY_ROOT']
sys.path.insert(0, _ROOT)

import lumenairy as la                                        # noqa: E402

_want = os.path.realpath(os.path.join(_ROOT, 'lumenairy'))
assert os.path.realpath(os.path.dirname(la.__file__)) == _want, la.__file__

TRACERS = {'trace', 'trace_world', 'trace_prescription', 'raytrace_system',
           'trace_jax', 'trace_jax_world'}
KEYWORDS = ('sphere_normal', 'renormalize')

SUBPACKAGES = ('raytrace', 'analysis', 'io', 'optimize', 'elements',
               'propagators', 'pipeline')


class _Visitor(ast.NodeVisitor):
    """Record, per function, every callee NAME and every tracer keyword
    it forwards.  Callees are matched by LEAF NAME, which over-approximates
    (two same-named helpers merge) -- deliberately, because for this census
    a false positive is a name to check and a false negative is a missed
    entry point."""

    def __init__(self, hits):
        self.stack = []
        self.hits = hits

    def visit_FunctionDef(self, node):
        self.stack.append(node.name)
        qual = '.'.join(self.stack)
        callees = set()
        fwd = set()
        direct = set()
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and sub.id in TRACERS:
                # a bare reference counts: ``ray_fan_data`` PASSES ``trace``
                # to a helper rather than calling it, and a census that only
                # reads Call nodes misses exactly that shape.
                direct.add(sub.id)
            if not isinstance(sub, ast.Call):
                continue
            fn = sub.func
            nm = (fn.id if isinstance(fn, ast.Name)
                  else fn.attr if isinstance(fn, ast.Attribute) else None)
            if nm:
                callees.add(nm)
            if nm in TRACERS:
                direct.add(nm)
                for kw in sub.keywords:
                    if kw.arg in KEYWORDS:
                        fwd.add(kw.arg)
        self.hits[qual] = dict(callees=sorted(callees), forwards=sorted(fwd),
                               direct=sorted(direct), leaf=node.name)
        self.generic_visit(node)
        self.stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()


def _public_names():
    names = {}
    for n in dir(la):
        if not n.startswith('_'):
            names[n] = 'lumenairy'
    for sub in SUBPACKAGES:
        try:
            m = importlib.import_module('lumenairy.' + sub)
        except Exception:
            continue
        for n in dir(m):
            if not n.startswith('_'):
                names.setdefault(n, 'lumenairy.' + sub)
    return names


def main(out_path):
    root = pathlib.Path(_ROOT) / 'lumenairy'
    per_module = {}
    for f in sorted(root.rglob('*.py')):
        rel = f.relative_to(pathlib.Path(_ROOT)).as_posix()
        try:
            tree = ast.parse(f.read_text(encoding='utf-8', errors='replace'))
        except SyntaxError:
            continue
        hits = {}
        _Visitor(hits).visit(tree)
        if hits:
            per_module[rel] = hits

    # --- transitive closure by leaf name: a function that calls a
    # function that traces, traces.  Iterated to a fixpoint so an entry
    # point three calls above ``trace`` is still counted.
    # DIRECT: the body names a tracer.  This is the census the way-back
    # question is about, and the one comparable to the C2 report's list.
    traces = set(TRACERS)
    depth = {t: 0 for t in TRACERS}
    for _mod, hits in per_module.items():
        for _qual, info in hits.items():
            if info['direct']:
                traces.add(info['leaf'])
                depth[info['leaf']] = 1
    direct_names = set(traces)
    # TRANSITIVE, for the fan-out count only: a caller of a direct tracer.
    # Leaf-name matching over-approximates, so ubiquitous method names are
    # excluded -- otherwise ``evaluate`` / ``run`` / ``fn`` pull in the
    # whole package and the number means nothing.
    STOP = {'evaluate', 'run', 'fn', 'apply', 'f', '_f', 'wrapper', 'inner',
            'call', '__call__', 'forward', 'step', 'solve', 'compute',
            'aggregate', 'main', 'value', 'func', 'objective'}
    changed = True
    rounds = 0
    while changed and rounds < 4:
        changed = False
        rounds += 1
        for _mod, hits in per_module.items():
            for _qual, info in hits.items():
                leaf = info['leaf']
                if leaf in traces or leaf in STOP:
                    continue
                hit = [c for c in info['callees']
                       if c in traces and c not in STOP]
                if hit:
                    traces.add(leaf)
                    depth[leaf] = 1 + min(depth.get(c, 1) for c in hit)
                    changed = True

    public = _public_names()
    rows = []
    for mod, hits in per_module.items():
        for qual, info in hits.items():
            leaf = qual.split('.')[-1]
            if leaf.startswith('_') or leaf not in traces:
                continue
            is_direct = leaf in direct_names
            info = dict(info,
                        tracers=sorted(c for c in info['callees']
                                       if c in traces),
                        depth=depth.get(leaf, 0))
            where = public.get(leaf)
            sig_kw = []
            if where:
                try:
                    obj = getattr(importlib.import_module(where), leaf)
                    params = inspect.signature(obj).parameters
                    sig_kw = [k for k in KEYWORDS if k in params]
                except Exception:
                    sig_kw = ['<unreadable>']
            rows.append(dict(name=leaf, qualname=qual, module=mod,
                             tracers=info['tracers'], depth=info['depth'],
                             forwards_to_tracer=info['forwards'],
                             exported_from=where,
                             signature_keywords=sig_kw,
                             direct=is_direct,
                             has_way_back=bool(where) and len(sig_kw) == 2))
    rows.sort(key=lambda r: (r['exported_from'] is None, r['name']))
    exported = [r for r in rows if r['exported_from']]
    exported_direct = [r for r in exported if r['direct']]
    no_way_back = [r for r in exported_direct if not r['has_way_back']]
    no_way_back_trans = [r for r in exported
                         if not r['has_way_back'] and not r['direct']]
    res = dict(
        meta=dict(python=sys.version.split()[0], lumenairy=la.__version__,
                  file=la.__file__),
        n_functions_that_trace=len(rows),
        n_exported=len(exported),
        n_exported_direct=len(exported_direct),
        n_exported_direct_without_a_way_back=len(no_way_back),
        exported_direct_without_a_way_back=sorted(
            {r['name'] for r in no_way_back}),
        n_exported_transitive_without_a_way_back=len(
            {r['name'] for r in no_way_back_trans}),
        exported_transitive_without_a_way_back=sorted(
            {r['name'] for r in no_way_back_trans}),
        rows=rows)
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    print(f"{'name':32s} {'exported from':22s} {'sig kw':16s} tracers")
    for r in exported_direct:
        print(f"{r['name']:32s} {r['exported_from']:22s} "
              f"{','.join(r['signature_keywords']) or '-':16s} d"
              f"{r['depth']} {','.join(r['tracers'])[:60]}")
    print()
    print('non-exported public-named functions that trace:',
          sorted(r['name'] for r in rows if not r['exported_from']))
    print()
    print('EXPORTED, DIRECTLY TRACING, NO WAY BACK:',
          len({r['name'] for r in no_way_back}))
    for r in no_way_back:
        print('  ', r['name'], '<-', r['module'], r['qualname'])
    print()
    print('EXPORTED, TRANSITIVELY TRACING, NO WAY BACK:',
          len({r['name'] for r in no_way_back_trans}))
    print('  ', sorted({r['name'] for r in no_way_back_trans}))


if __name__ == '__main__':
    main(sys.argv[1])
