"""VERIFY-WP-C3 claim 4b -- AST call-graph census: which functions in the
PACKAGE call anything that reaches a flipped default, and do they forward a
``transport`` keyword?

Walks every ``*.py`` under one or more roots, records every ``Call`` node whose
callee NAME (``f(...)`` or ``mod.f(...)``) is in the target set, and prints the
enclosing function, its line, whether that enclosing function declares a
``transport`` parameter, and whether the call passes ``transport=``.

    python census_callers.py <root> [<root> ...]
"""
from __future__ import annotations

import ast
import json
import os
import sys

TARGETS = {
    'propagate_carrier_referenced',
    'propagate_traced_carrier_chain',
    'propagate_traced_carrier_chain_multi',
    'carrier_referenced_focus_readout',
    'carrier_referenced_exact_focus_readout',
    '_carrier_step_fast',
    '_collins_carrier_leg',
    '_collins_transport',
    '_collins_focus_readout',
    '_collins_readout_k1',
}


def _callee_name(node):
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


class V(ast.NodeVisitor):
    def __init__(self, path):
        self.path = path
        self.stack = []
        self.hits = []

    def _fn(self, node):
        params = [a.arg for a in
                  node.args.posonlyargs + node.args.args + node.args.kwonlyargs]
        self.stack.append((node.name, node.lineno, 'transport' in params,
                           _default_of(node, 'transport')))
        self.generic_visit(node)
        self.stack.pop()

    visit_FunctionDef = _fn
    visit_AsyncFunctionDef = _fn

    def visit_Call(self, node):
        name = _callee_name(node)
        if name in TARGETS:
            passes = [kw.arg for kw in node.keywords if kw.arg == 'transport']
            val = None
            for kw in node.keywords:
                if kw.arg == 'transport':
                    val = (ast.unparse(kw.value)
                           if hasattr(ast, 'unparse') else '?')
            encl = self.stack[-1] if self.stack else ('<module>', 0, False,
                                                      None)
            self.hits.append({
                'file': self.path, 'line': node.lineno, 'callee': name,
                'enclosing': encl[0], 'enclosing_line': encl[1],
                'enclosing_takes_transport': encl[2],
                'enclosing_transport_default': encl[3],
                'call_passes_transport': bool(passes),
                'call_transport_value': val,
            })
        self.generic_visit(node)


def _default_of(node, pname):
    args = node.args
    pos = args.posonlyargs + args.args
    ndef = len(args.defaults)
    for i, a in enumerate(pos):
        if a.arg == pname:
            j = i - (len(pos) - ndef)
            if 0 <= j < ndef:
                return (ast.unparse(args.defaults[j])
                        if hasattr(ast, 'unparse') else '?')
            return None
    for a, d in zip(args.kwonlyargs, args.kw_defaults):
        if a.arg == pname:
            return (ast.unparse(d) if d is not None
                    and hasattr(ast, 'unparse') else None)
    return None


def main():
    hits = []
    for root in sys.argv[1:]:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d != '__pycache__']
            for fn in filenames:
                if not fn.endswith('.py'):
                    continue
                p = os.path.join(dirpath, fn)
                try:
                    src = open(p, encoding='utf-8').read()
                    tree = ast.parse(src)
                except Exception as exc:                      # noqa: BLE001
                    hits.append({'file': p, 'parse_error': repr(exc)})
                    continue
                v = V(os.path.relpath(p, root).replace('\\', '/'))
                v.visit(tree)
                hits.extend(v.hits)
    print(json.dumps(hits, indent=1))


if __name__ == '__main__':
    main()
