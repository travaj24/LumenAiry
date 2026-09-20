"""V4/A1 -- the OPTION-A counterfactual, rebuilt from scratch on my OWN
synthetic modules, plus an independent AST walk of ``lenses.py``.

    python v4_opta.py <tree> <out.json>

Nothing in the synthetic half imports lumenairy: it is a measurement of
CPython's import / attribute machinery.  The anchor still runs first so the
JSON records which tree (and therefore which interpreter/build) produced it,
and the AST half reads that tree's ``lenses.py``.

Five rows, two columns:

  (i)  by-value re-export         -- ``from leaf import FLAG, helper``
  (ii) ``__getattr__``-only       -- module ``__class__`` swap, no dict entry

measured for: presence in ``from ... import *``; presence in ``dir()``;
presence in ``vars()``; a function defined IN the shell reading the bare name;
whether a write on the shell reaches the leaf.

Column (ii) is measured TWICE -- with and without the eight unioned into
``__dir__`` -- because "in dir()" is a property of the ``__dir__`` override and
not of ``__getattr__``, and the author's table reports only the unioned value.
"""
from __future__ import annotations

import ast
import json
import os
import sys
import tempfile
import textwrap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_TREE = os.path.abspath(sys.argv[1])
_OUT = os.path.abspath(sys.argv[2])
sys.path.insert(0, _TREE)

import vlib  # noqa: E402

vlib.anchor(_TREE)

LEAF_OWNED = ("CUPY_AVAILABLE", "_is_cupy_array", "_ensure_cupy_loaded",
              "_load_numba", "_get_aspheric_sag_accum_numba",
              "_ensure_numexpr_loaded", "_collect_semi_diameters",
              "_warn_if_aperture_exceeds_grid")

LEAF_SRC = textwrap.dedent('''
    """v4 synthetic LEAF -- stands in for _lens_kernels."""
    FLAG = True


    def helper():
        return "leaf-helper"
''')

# (i) by-value re-export: exactly what lenses.py does today.
SHELL_BYVALUE_SRC = textwrap.dedent('''
    """v4 synthetic SHELL, by-value re-export."""
    from v4leaf import FLAG as FLAG, helper as helper


    def uses_bare_name():
        return helper()


    def reads_bare_flag():
        return FLAG
''')

# (ii) __getattr__-only: option A.  ``UNION_DIR`` decides whether __dir__
# unions the served names in.
SHELL_GETATTR_SRC = textwrap.dedent('''
    """v4 synthetic SHELL, served only by __getattr__ (option A)."""
    import sys as _sys
    import types as _types

    import v4leaf as _LEAF

    _SERVED = frozenset({"FLAG", "helper"})
    UNION_DIR = __UNION__


    class _Facade(_types.ModuleType):
        def __getattr__(self, name):
            if name in _SERVED:
                return getattr(_LEAF, name)
            raise AttributeError(name)

        def __setattr__(self, name, value):
            if name in _SERVED:
                setattr(_LEAF, name, value)
                return
            super().__setattr__(name, value)

        def __delattr__(self, name):
            if name in _SERVED:
                delattr(_LEAF, name)
                return
            super().__delattr__(name)

        def __dir__(self):
            base = set(super().__dir__())
            if UNION_DIR:
                base |= _SERVED
            return sorted(base)


    def uses_bare_name():
        return helper()          # noqa: F821 -- THE POINT: LOAD_GLOBAL


    def reads_bare_flag():
        return FLAG              # noqa: F821 -- THE POINT: LOAD_GLOBAL


    _sys.modules[__name__].__class__ = _Facade
''')


def _star_names(modname, tmpdir):
    """``from <modname> import *`` in a FRESH child-of-this-process namespace.

    Executed with ``exec`` in an empty globals dict, which is exactly what the
    statement does at module scope, so the answer is the real one and not a
    reconstruction from ``__dict__``.
    """
    ns = {}
    exec(compile(f"from {modname} import *", "<v4-star>", "exec"), ns)
    return sorted(k for k in ns if k != "__builtins__")


def _attempt(fn, *a):
    try:
        return ("ok", repr(fn(*a))[:200])
    except BaseException as exc:                  # noqa: BLE001 -- recorded
        return ("raised", type(exc).__name__, str(exc)[:200])


def measure_column(kind, tmpdir, union_dir=True):
    """One column of the five-row table."""
    for m in list(sys.modules):
        if m.startswith("v4leaf") or m.startswith("v4shell"):
            del sys.modules[m]
    src = SHELL_BYVALUE_SRC if kind == "byvalue" else \
        SHELL_GETATTR_SRC.replace("__UNION__", str(bool(union_dir)))
    with open(os.path.join(tmpdir, "v4shell.py"), "w", encoding="utf-8") as fh:
        fh.write(src)
    import importlib
    leaf = importlib.import_module("v4leaf")
    shell = importlib.import_module("v4shell")

    star = _star_names("v4shell", tmpdir)
    row = {
        "in_import_star_FLAG": "FLAG" in star,
        "in_import_star_helper": "helper" in star,
        "import_star_names": star,
        "in_dir_FLAG": "FLAG" in dir(shell),
        "in_dir_helper": "helper" in dir(shell),
        "in_vars_FLAG": "FLAG" in vars(shell),
        "in_vars_helper": "helper" in vars(shell),
        "bare_name_call": _attempt(shell.uses_bare_name),
        "bare_name_read": _attempt(shell.reads_bare_flag),
        "getattr_FLAG": _attempt(getattr, shell, "FLAG"),
    }
    # write-reaches-leaf
    leaf.FLAG = True
    w = _attempt(setattr, shell, "FLAG", "SUBSTITUTE")
    row["write_attempt"] = w
    row["write_reaches_leaf"] = (leaf.FLAG == "SUBSTITUTE")
    row["shell_reads_after_write"] = _attempt(getattr, shell, "FLAG")
    leaf.FLAG = True
    try:
        vars(shell).pop("FLAG", None)
        vars(shell)["FLAG"] = True
    except Exception:
        pass
    return row


def ast_walk_lenses(tree):
    """Every BARE-NAME load of one of the eight anywhere in the tree's
    lenses.py, with the ``from ._lens_kernels import ...`` aliases excluded
    (those are ``alias`` nodes, not ``Name`` nodes, so they never appear).

    Also counts ``globals()[...]`` / ``vars()[...]`` subscripts with one of the
    eight as a literal key, which a plain Name walk would miss and which WOULD
    still work under option A only if the dict entry survived.
    """
    path = os.path.join(tree, "lumenairy", "elements", "lenses.py")
    src = open(path, encoding="utf-8").read()
    mod = ast.parse(src, filename=path)
    eight = set(LEAF_OWNED)
    name_loads, call_sites, store_sites, dyn = [], [], [], []
    for node in ast.walk(mod):
        if isinstance(node, ast.Name) and node.id in eight:
            entry = (node.id, node.lineno, type(node.ctx).__name__)
            if isinstance(node.ctx, ast.Load):
                name_loads.append(entry)
            else:
                store_sites.append(entry)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in eight:
            call_sites.append((node.func.id, node.lineno))
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Call) \
                and isinstance(node.value.func, ast.Name) \
                and node.value.func.id in ("globals", "vars"):
            key = getattr(node.slice, "value", None)
            dyn.append((node.value.func.id, key, node.lineno))
        # attribute access <anything>.<one of the eight>
    attr_hits = [(node.attr, node.lineno) for node in ast.walk(mod)
                 if isinstance(node, ast.Attribute) and node.attr in eight]
    # the same walk on the LEAF, as the near-miss control
    kpath = os.path.join(tree, "lumenairy", "elements", "_lens_kernels.py")
    ksrc = open(kpath, encoding="utf-8").read()
    kmod = ast.parse(ksrc, filename=kpath)
    kcalls = [(n.func.id, n.lineno) for n in ast.walk(kmod)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
              and n.func.id in eight]
    kloads = [(n.id, n.lineno) for n in ast.walk(kmod)
              if isinstance(n, ast.Name) and n.id in eight
              and isinstance(n.ctx, ast.Load)]
    return {
        "lenses_bare_name_loads": name_loads,
        "lenses_bare_name_loads_n": len(name_loads),
        "lenses_bare_calls": call_sites,
        "lenses_bare_calls_n": len(call_sites),
        "lenses_bare_stores": store_sites,
        "lenses_attribute_hits": attr_hits,
        "lenses_globals_subscripts": dyn,
        "leaf_bare_calls": kcalls,
        "leaf_bare_calls_n": len(kcalls),
        "leaf_bare_loads_n": len(kloads),
        "leaf_bare_loads": kloads,
        "lenses_has_dunder_all": bool(
            [n for n in ast.walk(mod) if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "__all__"
                     for t in n.targets)]),
    }


def main():
    tmpdir = tempfile.mkdtemp(prefix="v4_opta_")
    with open(os.path.join(tmpdir, "v4leaf.py"), "w", encoding="utf-8") as fh:
        fh.write(LEAF_SRC)
    sys.path.insert(0, tmpdir)
    out = {
        "build": vlib.build_tag(),
        "python": sys.version,
        "tree": _TREE,
        "col_i_byvalue": measure_column("byvalue", tmpdir),
        "col_ii_getattr_uniondir": measure_column("getattr", tmpdir, True),
        "col_ii_getattr_nouniondir": measure_column("getattr", tmpdir, False),
        "ast": ast_walk_lenses(_TREE),
    }
    vlib.write_json(out, _OUT)
    print(json.dumps({k: v for k, v in out["ast"].items()
                      if k.endswith("_n") or k == "lenses_has_dunder_all"},
                     indent=1))


if __name__ == "__main__":
    main()
