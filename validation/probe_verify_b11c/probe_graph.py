"""VERIFY-B11c import-graph instruments, written independently of
``validation/probe_wp_b11c/import_graph.py``.

Two readings of the same graph:

``--mode ast``     the SOURCE reading.  Parses every ``lumenairy/**/*.py`` and
                   keeps only ``Import`` / ``ImportFrom`` nodes whose enclosing
                   scope is the MODULE BODY -- an ``ast.NodeVisitor`` that
                   refuses to descend into ``FunctionDef`` /
                   ``AsyncFunctionDef`` / ``ClassDef`` / ``Lambda``, so a
                   deferred import inside a function is not counted.  Relative
                   imports are resolved against the file's package.  ``if
                   TYPE_CHECKING:`` blocks ARE module-body statements and are
                   counted, which is deliberate: they are what a reader sees.

``--mode exec``    the EXECUTED reading.  Wraps ``builtins.__import__`` and
                   keeps a call only when the caller's frame is a module body
                   (``f_locals is f_globals``), which is true in no function,
                   comprehension or class body.  This is the only instrument
                   that sees the edge a cycle is MADE of: a module-level
                   ``from .lenses import x`` whose target is already in
                   ``sys.modules`` half-initialised runs no loader, so it
                   leaves no trace in ``-X importtime``.

``--first M``      (exec mode) import ``M`` BEFORE ``lumenairy``, so a module
                   that only works because something else initialised it first
                   is caught.

Both modes print a JSON document: the edge list restricted to a prefix, the
2-cycles, and (exec mode) the import order.

argv: --tree <root> --mode ast|exec [--first mod] [--prefix P] [--out F]
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys


def _mod_name(root: str, path: str) -> str:
    rel = os.path.relpath(path, root).replace("\\", "/")
    assert rel.endswith(".py")
    rel = rel[:-3]
    if rel.endswith("/__init__"):
        rel = rel[: -len("/__init__")]
    return rel.replace("/", ".")


class _TopLevelImports(ast.NodeVisitor):
    """Collects module-body imports only; never descends into a def/class."""

    def __init__(self):
        self.found = []

    def visit_FunctionDef(self, node):        # noqa: N802 -- ast protocol
        return

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_ClassDef = visit_FunctionDef
    visit_Lambda = visit_FunctionDef

    def visit_Import(self, node):             # noqa: N802
        for a in node.names:
            self.found.append((0, a.name))

    def visit_ImportFrom(self, node):         # noqa: N802
        self.found.append((node.level, node.module or ""))


def _resolve(level: int, target: str, owner: str, is_pkg: bool) -> str:
    """Resolve a relative import the way the interpreter does: one dot names
    the importer's PACKAGE, and each further dot climbs one level from it.  A
    package ``__init__`` IS its own package, so its ``owner`` needs no trim."""
    if level == 0:
        return target
    parts = owner.split(".")
    pkg = parts if is_pkg else parts[:-1]
    base = pkg[: len(pkg) - (level - 1)] if level > 1 else pkg
    head = ".".join(base)
    return f"{head}.{target}" if target else head


def _ast_graph(tree: str, prefix: str):
    pkg_root = os.path.join(tree, "lumenairy")
    edges = {}
    for dirpath, _dirs, files in os.walk(pkg_root):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            owner = _mod_name(tree, path)
            is_pkg = fn == "__init__.py"
            with open(path, encoding="utf-8") as fh:
                try:
                    node = ast.parse(fh.read(), filename=path)
                except SyntaxError as exc:
                    edges.setdefault(owner, set()).add(f"<syntaxerror:{exc}>")
                    continue
            v = _TopLevelImports()
            v.visit(node)
            out = edges.setdefault(owner, set())
            for level, target in v.found:
                full = _resolve(level, target, owner, is_pkg)
                if full.startswith(prefix):
                    out.add(full)
    return edges


def _exec_graph(tree: str, prefix: str, first: str | None):
    import builtins

    edges = {}
    order = []
    real = builtins.__import__

    def hook(name, globals=None, locals=None, fromlist=(), level=0):
        frame = sys._getframe(1)
        is_module_body = frame.f_locals is frame.f_globals
        owner = (globals or {}).get("__name__", "?")
        mod = real(name, globals, locals, fromlist, level)
        if is_module_body and owner.startswith("lumenairy"):
            if level:
                pkg = (globals or {}).get("__package__") or ""
                base = pkg.split(".")
                base = base[: len(base) - (level - 1)] if level > 1 else base
                head = ".".join(base)
                full = f"{head}.{name}" if name else head
            else:
                full = name
            if full.startswith(prefix):
                edges.setdefault(owner, set()).add(full)
            # a ``from X import a, b`` may pull SUBMODULES a, b of X
            for f in (fromlist or ()):
                cand = f"{full}.{f}"
                if cand.startswith(prefix) and cand in sys.modules:
                    edges.setdefault(owner, set()).add(cand)
        return mod

    builtins.__import__ = hook
    try:
        if first:
            __import__(first)
            order.append(first)
        __import__("lumenairy")
        order.append("lumenairy")
    finally:
        builtins.__import__ = real
    loaded = sorted(m for m in sys.modules if m.startswith(prefix))
    return edges, order, loaded


def _two_cycles(edges):
    out = set()
    for a, tos in edges.items():
        for b in tos:
            if b == a:
                continue
            if a in edges.get(b, ()):
                out.add(tuple(sorted((a, b))))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--mode", choices=("ast", "exec"), required=True)
    ap.add_argument("--first", default=None)
    ap.add_argument("--prefix", default="lumenairy")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    tree = os.path.abspath(a.tree)
    sys.path.insert(0, tree)

    if a.mode == "ast":
        edges = _ast_graph(tree, a.prefix)
        order, loaded = [], []
    else:
        import lumenairy  # noqa: F401 -- anchor check below
        got = os.path.realpath(sys.modules["lumenairy"].__file__)
        if os.path.commonpath([got, tree]) != tree:
            raise SystemExit(f"WRONG TREE: {got}")
        # reset and re-import inside the hook, in a SUBPROCESS-clean way:
        for m in [m for m in sys.modules if m.startswith("lumenairy")]:
            del sys.modules[m]
        edges, order, loaded = _exec_graph(tree, a.prefix, a.first)

    fam = "lumenairy.elements"
    doc = {
        "mode": a.mode,
        "tree": tree,
        "first": a.first,
        "n_modules": len(edges),
        "two_cycles_all": _two_cycles(edges),
        "two_cycles_lens_family": [
            c for c in _two_cycles(edges)
            if all(x.startswith(fam) and "rcwa" not in x and "pmm" not in x
                   for x in c)],
        "edges": {k: sorted(v) for k, v in sorted(edges.items())},
        "import_order": order,
        "loaded": loaded,
    }
    text = json.dumps(doc, indent=1)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"[graph] {a.mode}: {len(edges)} modules, "
              f"{len(doc['two_cycles_all'])} 2-cycles -> {a.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
