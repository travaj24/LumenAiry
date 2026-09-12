"""Probe 9: structural census of carrier.py / carrier_field.py."""
import io, sys, tokenize, ast, collections
BASE = r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy/lumenairy/propagators/"
for fn in ("carrier.py", "carrier_field.py"):
    path = BASE + fn
    src = open(path, encoding='utf-8').read()
    lines = src.splitlines()
    n = len(lines)
    toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    comment_lines = set()
    for t in toks:
        if t.type == tokenize.COMMENT:
            comment_lines.add(t.start[0])
    tree = ast.parse(src)
    doc_lines = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            d = ast.get_docstring(node, clean=False)
            if d is not None:
                # find the docstring node
                first = node.body[0]
                for L in range(first.lineno, first.end_lineno+1):
                    doc_lines.add(L)
    blank = {i+1 for i, L in enumerate(lines) if not L.strip()}
    # code lines = not blank, not comment-only, not docstring
    codeish = 0
    for i, L in enumerate(lines, 1):
        if i in blank: continue
        if i in doc_lines: continue
        if i in comment_lines and not L.strip().startswith('#'):
            pass  # trailing comment on a code line -> counts as code
        if L.strip().startswith('#'):
            continue
        codeish += 1
    com_only = sum(1 for i, L in enumerate(lines,1) if L.strip().startswith('#'))
    print(f"{fn}: total {n}  blank {len(blank)}  comment-only {com_only}  docstring {len(doc_lines)}  code {codeish}")
    print(f"    -> code {100*codeish/n:.1f}%   doc {100*len(doc_lines)/n:.1f}%   comment {100*com_only/n:.1f}%   blank {100*len(blank)/n:.1f}%")
    # per-function sizes
    sizes = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body_doc = 0
            d = ast.get_docstring(node, clean=False)
            if d is not None:
                f0 = node.body[0]
                body_doc = f0.end_lineno - f0.lineno + 1
            tot = node.end_lineno - node.lineno + 1
            sizes.append((tot, tot-body_doc, node.name, node.lineno))
    sizes.sort(reverse=True)
    print("    top 12 functions by total lines (total, non-docstring, name, line):")
    for s in sizes[:12]:
        print(f"       {s[0]:5d} {s[1]:5d}  {s[2]}  @{s[3]}")
