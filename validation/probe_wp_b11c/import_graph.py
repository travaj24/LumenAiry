"""Record the MODULE-LEVEL import graph that actually executes.

    python validation/probe_wp_b11c/import_graph.py <module-to-import> [...]

Prints one JSON object: ``{"edges": [[importer, imported, [fromlist]], ...]}``.

Why a hook and not an AST walk.  An AST walk reads what the source SAYS; this
reads what the interpreter DOES, including the case that matters for a cycle --
a module-level ``from .lenses import x`` whose target is ALREADY in
``sys.modules`` half-initialised, which executes no loader and so leaves no
trace in ``-X importtime`` or in ``sys.modules`` afterwards.  Both instruments
are used: the AST walk is the structural gate, this is the dynamic one.

Only MODULE-SCOPE imports are recorded.  The discriminator is exact rather than
heuristic: in a module body ``frame.f_locals`` IS ``frame.f_globals`` (the same
dict object), which is true in no function, method, comprehension or class
body.  In-function imports are deliberately not edges -- they are how the lens
family already breaks the cycles it cannot otherwise break.
"""
from __future__ import annotations

import builtins
import json
import sys

_edges = []
_real_import = builtins.__import__


def _resolve(name: str, package: str, level: int) -> str:
    """The absolute name a relative import resolves to, by PEP 328's rule."""
    if not level:
        return name
    parts = package.split(".") if package else []
    base = ".".join(parts[:len(parts) - level + 1])
    return f"{base}.{name}" if name else base


def _hook(name, globals=None, locals=None, fromlist=(), level=0):
    frame = sys._getframe(1)
    if globals is not None and frame.f_locals is frame.f_globals:
        src = globals.get("__name__")
        pkg = globals.get("__package__") or ""
        _edges.append([src, _resolve(name, pkg, level),
                       sorted(fromlist or ())])
    return _real_import(name, globals, locals, fromlist, level)


def main() -> int:
    builtins.__import__ = _hook
    try:
        for mod in sys.argv[1:]:
            _real_import(mod, {"__name__": "__main__", "__package__": ""},
                         None, ("*",), 0)
    finally:
        builtins.__import__ = _real_import
    seen = set()
    uniq = []
    for e in _edges:
        k = (e[0], e[1], tuple(e[2]))
        if k not in seen:
            seen.add(k)
            uniq.append(e)
    print(json.dumps({"edges": uniq}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
