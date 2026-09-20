"""V4/A4 -- does ANY existing site legitimately WRITE one of the eight on the
lenses FACADE?

    python v4_writescan.py <tree> <out.json>

Four independent finders over every ``*.py`` in the tree:

1. ``Assign`` / ``AugAssign`` / ``AnnAssign`` / ``Delete`` whose target is
   ``Attribute(value=<lenses-ish>, attr=<one of the eight>)``.  "lenses-ish"
   is deliberately WIDE: any Name or dotted path whose last component contains
   ``lens`` (case-insensitive), so ``lenses``, ``LE``, ``le`` are caught by the
   alias table below and ``_lens_kernels`` is reported separately.
2. ``monkeypatch.setattr`` / ``delattr`` / ``setitem`` and
   ``mock.patch`` / ``patch.object`` / ``mock.patch.object``, in BOTH forms:
   an OBJECT first argument plus a string name, and a single dotted STRING.
3. ``setattr(...)`` / ``delattr(...)`` builtins with a literal name.
4. A raw regex sweep, as the net under the AST walk, for any line containing
   one of the eight AND an ``=`` that is not ``==`` / ``!=`` / ``>=`` / ``<=``.

Per hit: file, line, the target module path as written, the name, and whether
the branch's refusal would BREAK it (target resolves to the lenses facade) or
not (target is ``_lens_kernels`` / a different module).
"""
from __future__ import annotations

import ast
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_TREE = os.path.abspath(sys.argv[1])
_OUT = os.path.abspath(sys.argv[2])
sys.path.insert(0, _TREE)

import vlib  # noqa: E402

vlib.anchor(_TREE)

EIGHT = {"CUPY_AVAILABLE", "_is_cupy_array", "_ensure_cupy_loaded",
         "_load_numba", "_get_aspheric_sag_accum_numba",
         "_ensure_numexpr_loaded", "_collect_semi_diameters",
         "_warn_if_aperture_exceeds_grid"}

FACADE = "lumenairy.elements.lenses"
LEAF = "lumenairy.elements._lens_kernels"


def dotted(node):
    """``a.b.c`` for an Attribute/Name chain, else None."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def import_aliases(mod):
    """``{local name: full dotted module}`` for every import in the file."""
    out = {}
    for n in ast.walk(mod):
        if isinstance(n, ast.Import):
            for a in n.names:
                out[a.asname or a.name.split(".")[0]] = a.name
        elif isinstance(n, ast.ImportFrom):
            base = ("." * (n.level or 0)) + (n.module or "")
            for a in n.names:
                out[a.asname or a.name] = f"{base}.{a.name}"
    return out


def resolve(path, aliases, fname):
    """Best-effort: what module does this dotted expression name?"""
    if path is None:
        return None
    head = path.split(".")[0]
    full = aliases.get(head, head)
    rest = path.split(".")[1:]
    cand = ".".join([full] + rest)
    # relative import inside the package
    if cand.startswith("."):
        pkg = "lumenairy.elements"
        cand = pkg + cand
    return cand


def classify(target_mod, fname):
    t = (target_mod or "")
    if t.endswith("lenses") or t == FACADE or re.search(
            r"(^|\.)lenses$", t):
        return "FACADE"
    if "_lens_kernels" in t:
        return "LEAF"
    return "OTHER"


def scan_file(path):
    hits = []
    try:
        src = open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return hits
    if not any(n in src for n in EIGHT):
        return hits
    try:
        mod = ast.parse(src, filename=path)
    except SyntaxError as exc:
        return [{"file": path, "line": 0, "kind": "SYNTAX-ERROR",
                 "detail": str(exc)}]
    aliases = import_aliases(mod)

    def add(node, kind, name, tgt, detail=""):
        hits.append({
            "file": os.path.relpath(path, _TREE).replace("\\", "/"),
            "line": node.lineno, "kind": kind, "name": name,
            "target_as_written": tgt,
            "target_resolved": resolve(tgt, aliases, path),
            "class": classify(resolve(tgt, aliases, path), path),
            "detail": detail,
        })

    for node in ast.walk(mod):
        # (b) attribute assignment / deletion
        tgts = []
        if isinstance(node, ast.Assign):
            tgts = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            tgts = [node.target]
        elif isinstance(node, ast.Delete):
            tgts = node.targets
        for t in tgts:
            if isinstance(t, ast.Attribute) and t.attr in EIGHT:
                add(node, type(node).__name__, t.attr, dotted(t.value))
        # (c) setattr/delattr/monkeypatch/mock
        if isinstance(node, ast.Call):
            fn = dotted(node.func) or ""
            tail = fn.split(".")[-1]
            if tail in ("setattr", "delattr") and len(node.args) >= 2:
                a0, a1 = node.args[0], node.args[1]
                nm = a1.value if isinstance(a1, ast.Constant) else None
                if nm in EIGHT:
                    add(node, f"call:{fn}", nm, dotted(a0))
                # single dotted string form
            if tail in ("setattr", "delattr", "patch", "object",
                        "setitem", "delattr") and node.args:
                a0 = node.args[0]
                if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
                    s = a0.value
                    if s.rsplit(".", 1)[-1] in EIGHT and "." in s:
                        add(node, f"call-str:{fn}", s.rsplit(".", 1)[-1],
                            s.rsplit(".", 1)[0], detail=s)
                if len(node.args) >= 2 and isinstance(node.args[1],
                                                      ast.Constant) \
                        and node.args[1].value in EIGHT \
                        and tail in ("patch", "object"):
                    add(node, f"call:{fn}", node.args[1].value, dotted(a0))
    # (d) regex net
    for i, line in enumerate(src.splitlines(), 1):
        for n in EIGHT:
            if n in line and re.search(
                    rf"\.{re.escape(n)}\s*=(?!=)", line):
                if not any(h["line"] == i and h["name"] == n for h in hits):
                    hits.append({
                        "file": os.path.relpath(path, _TREE).replace("\\", "/"),
                        "line": i, "kind": "regex-only", "name": n,
                        "target_as_written": line.strip()[:160],
                        "target_resolved": None, "class": "UNRESOLVED",
                        "detail": line.strip()[:160]})
    return hits


def main():
    hits = []
    for root, dirs, files in os.walk(_TREE):
        dirs[:] = [d for d in dirs if d not in
                   (".git", "__pycache__", ".pytest_cache", "node_modules")]
        for f in files:
            if f.endswith(".py"):
                hits.extend(scan_file(os.path.join(root, f)))
    out = {"build": vlib.build_tag(), "tree": _TREE,
           "n_hits": len(hits),
           "n_facade": sum(1 for h in hits if h["class"] == "FACADE"),
           "n_leaf": sum(1 for h in hits if h["class"] == "LEAF"),
           "n_other": sum(1 for h in hits if h["class"] == "OTHER"),
           "n_unresolved": sum(1 for h in hits if h["class"] == "UNRESOLVED"),
           "hits": sorted(hits, key=lambda h: (h["file"], h["line"]))}
    vlib.write_json(out, _OUT)
    print(json.dumps({k: v for k, v in out.items() if k != "hits"}, indent=1))
    for h in out["hits"]:
        print(f"  [{h['class']}] {h['file']}:{h['line']} {h['kind']} "
              f"{h['target_as_written']} -> {h['name']}")


if __name__ == "__main__":
    main()
