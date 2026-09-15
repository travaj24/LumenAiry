"""Per-block dependency analysis of lumenairy/elements/rcwa/_core.py.

For each banner-delimited section, report which module-level names it DEFINES
and which module-level names defined OUTSIDE it that it READS.  A section is a
candidate leaf iff the set of outside names it reads is empty (modulo imports).
"""
import ast
import sys
from pathlib import Path

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else
           'lumenairy/elements/rcwa/_core.py')
src = SRC.read_text(encoding='utf-8')
lines = src.splitlines()
tree = ast.parse(src)

# module-level definitions: name -> (lineno, end_lineno)
defs = {}
imported = set()
for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        defs[node.name] = (node.lineno, node.end_lineno)
    elif isinstance(node, ast.Assign):
        for t in node.targets:
            if isinstance(t, ast.Name):
                defs[t.id] = (node.lineno, node.end_lineno)
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        defs[node.target.id] = (node.lineno, node.end_lineno)
    elif isinstance(node, (ast.Import, ast.ImportFrom)):
        for a in node.names:
            imported.add(a.asname or a.name.split('.')[0])

# banner sections
banners = []
for i, ln in enumerate(lines, start=1):
    if ln.startswith('# ====================='):
        # title is the next line
        if i < len(lines) and lines[i].startswith('# '):
            banners.append((i, lines[i][2:].strip()))
sections = []
for k, (ln, title) in enumerate(banners):
    end = banners[k + 1][0] - 1 if k + 1 < len(banners) else len(lines)
    sections.append((ln, end, title))
if sections:
    sections.insert(0, (1, sections[0][0] - 1, 'HEAD (imports, BLAS controls, small utils)'))


def names_used(lo, hi):
    used = set()
    for node in tree.body:
        s = getattr(node, 'lineno', 0)
        if s < lo or s > hi:
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name):
                used.add(sub.id)
            elif isinstance(sub, ast.Attribute):
                pass
    return used


print(f"{SRC}: {len(lines)} lines, {len(defs)} module-level names, "
      f"{len(sections)} sections\n")
for lo, hi, title in sections:
    own = {n for n, (a, b) in defs.items() if lo <= a <= hi}
    used = names_used(lo, hi)
    outside = sorted(n for n in used
                     if n in defs and n not in own and n not in imported)
    inbound = []
    for n in sorted(own):
        for m, (a, b) in defs.items():
            if m in own:
                continue
            for node in tree.body:
                if getattr(node, 'lineno', 0) != a:
                    continue
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Name) and sub.id == n:
                        inbound.append((m, n))
                        break
                break
    print(f"--- [{lo:5d}-{hi:5d}] ({hi - lo + 1:4d} lines) {title}")
    print(f"    defines ({len(own)}): {', '.join(sorted(own)) or '-'}")
    print(f"    READS from elsewhere in _core ({len(outside)}): "
          f"{', '.join(outside) or 'NONE  <-- LEAF CANDIDATE'}")
    inb = sorted({f'{m}->{n}' for m, n in inbound})
    print(f"    read BY elsewhere ({len(inb)}): {', '.join(inb) or '-'}")
    print()
