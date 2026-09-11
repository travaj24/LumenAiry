"""TASK F -- enumerate EVERY numeric constant that appears inside an assert
statement in the files under audit, with its enclosing test and the source
line, so the origin census cannot miss one.
"""
from __future__ import annotations

import ast
import pathlib
import sys

FILES = [
    r"C:/tmp/lum_vbor/tests/unit/test_fix_bor_multilayer_guards.py",
    r"C:/tmp/lum_vbor/tests/unit/test_fix_eme_branch_cut.py",
    r"C:/tmp/lum_vbor/tests/unit/test_bor_solve.py",
]
ONLY = sys.argv[1] if len(sys.argv) > 1 else None


def walk(path):
    src = pathlib.Path(path).read_text(encoding="utf-8", errors="replace")
    lines = src.splitlines()
    tree = ast.parse(src)
    for fn in [n for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef,))]:
        if ONLY and ONLY not in fn.name:
            continue
        rows = []
        for node in ast.walk(fn):
            if not isinstance(node, ast.Assert):
                continue
            for sub in ast.walk(node.test):
                if (isinstance(sub, ast.Constant)
                        and isinstance(sub.value, (int, float))
                        and not isinstance(sub.value, bool)):
                    rows.append((sub.lineno, sub.value,
                                 lines[sub.lineno - 1].strip()))
        if rows:
            print("\n### %s:%d  %s" % (pathlib.Path(path).name,
                                       fn.lineno, fn.name))
            seen = set()
            for ln, val, text in rows:
                key = (ln, repr(val))
                if key in seen:
                    continue
                seen.add(key)
                print("    L%-5d %-22r %s" % (ln, val, text[:96]))


for f in FILES:
    print("=" * 78)
    print(f)
    walk(f)
