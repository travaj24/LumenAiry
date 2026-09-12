"""Size / complexity / comment-ratio census for Lumenairy (read-only)."""
import ast, io, json, os, sys, tokenize, collections

ROOT = r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy"
OUT = r"docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/TESTS-ARCH"

SKIP_DIRS = {".git", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
             ".benchmarks", "lumenairy.egg-info", "node_modules", ".venv"}


def walk(base):
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if fn.endswith(".py"):
                yield os.path.join(dirpath, fn)


def rel(p):
    return os.path.relpath(p, ROOT).replace("\\", "/")


BRANCH_NODES = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler,
                ast.With, ast.AsyncWith, ast.IfExp, ast.Assert, ast.comprehension)


def analyze():
    files = {}          # rel -> dict
    funcs = []          # list of dicts
    for p in walk(os.path.join(ROOT, "lumenairy")) :
        try:
            src = open(p, "r", encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        r = rel(p)
        nlines = src.count("\n") + 1
        rec = {"lines": nlines, "bytes": len(src)}
        # tokenize for comment / docstring / code ratio
        comment_lines = set()
        string_lines = set()
        code_lines = set()
        blank = 0
        try:
            toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
        except Exception:
            toks = []
        prev_end_row = 0
        for tok in toks:
            t, s, start, end, line = tok
            if t == tokenize.COMMENT:
                for ln in range(start[0], end[0] + 1):
                    comment_lines.add(ln)
            elif t == tokenize.STRING:
                # treat as docstring only if statement-level; approximate below with AST
                for ln in range(start[0], end[0] + 1):
                    string_lines.add(ln)
            elif t in (tokenize.NEWLINE, tokenize.NL, tokenize.INDENT,
                       tokenize.DEDENT, tokenize.ENDMARKER, tokenize.ENCODING):
                pass
            else:
                for ln in range(start[0], end[0] + 1):
                    code_lines.add(ln)
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            rec["syntax_error"] = str(e)
            files[r] = rec
            continue
        # docstring line set from AST
        doc_lines = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                body = getattr(node, "body", None)
                if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                        and isinstance(body[0].value.value, str):
                    n = body[0].value
                    for ln in range(n.lineno, (n.end_lineno or n.lineno) + 1):
                        doc_lines.add(ln)
        # code lines that are not docstrings
        real_code = code_lines - doc_lines
        rec["comment_lines"] = len(comment_lines)
        rec["doc_lines"] = len(doc_lines)
        rec["code_lines"] = len(real_code)
        rec["nondoc_string_lines"] = len(string_lines - doc_lines)
        files[r] = rec

        # functions
        parents = {}
        for node in ast.walk(tree):
            for ch in ast.iter_child_nodes(node):
                parents[ch] = node
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                a = node.args
                nparams = (len(a.posonlyargs) + len(a.args) + len(a.kwonlyargs)
                           + (1 if a.vararg else 0) + (1 if a.kwarg else 0))
                nbranch = sum(1 for x in ast.walk(node) if isinstance(x, BRANCH_NODES))
                nbool = sum(len(x.values) - 1 for x in ast.walk(node) if isinstance(x, ast.BoolOp))
                span = (node.end_lineno or node.lineno) - node.lineno + 1
                # qualified name
                qn = node.name
                par = parents.get(node)
                chain = []
                while par is not None and not isinstance(par, ast.Module):
                    if isinstance(par, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                        chain.append(par.name)
                    par = parents.get(par)
                if chain:
                    qn = ".".join(reversed(chain)) + "." + qn
                funcs.append({"file": r, "line": node.lineno, "name": qn,
                              "span": span, "params": nparams,
                              "cc": nbranch + nbool + 1})
    return files, funcs


files, funcs = analyze()
json.dump({"files": files, "funcs": funcs}, open(os.path.join(OUT, "census.json"), "w"))

print("=== TOTALS (lumenairy/ package only) ===")
print("py files:", len(files))
print("total lines:", sum(f["lines"] for f in files.values()))
print("total code lines:", sum(f.get("code_lines", 0) for f in files.values()))
print("total comment lines:", sum(f.get("comment_lines", 0) for f in files.values()))
print("total docstring lines:", sum(f.get("doc_lines", 0) for f in files.values()))
print("functions:", len(funcs))

print("\n=== FILES > 3000 lines ===")
big = sorted(files.items(), key=lambda kv: -kv[1]["lines"])
for r, d in big:
    if d["lines"] > 3000:
        print(f"{d['lines']:7d}  {r}")
print("(count:", sum(1 for _, d in files.items() if d["lines"] > 3000), ")")

print("\n=== TOP 30 files by line count ===")
for r, d in big[:30]:
    print(f"{d['lines']:7d}  {r}")

print("\n=== TOP 30 functions by span ===")
for f in sorted(funcs, key=lambda x: -x["span"])[:30]:
    print(f"{f['span']:6d} lines  cc={f['cc']:4d}  params={f['params']:3d}  {f['file']}:{f['line']}  {f['name']}")

print("\n=== functions > 300 lines: count =", sum(1 for f in funcs if f["span"] > 300))
print("=== functions > 150 lines: count =", sum(1 for f in funcs if f["span"] > 150))

print("\n=== functions with > 25 params ===")
for f in sorted(funcs, key=lambda x: -x["params"])[:30]:
    if f["params"] > 20:
        print(f"{f['params']:4d} params  span={f['span']:5d}  {f['file']}:{f['line']}  {f['name']}")

print("\n=== TOP 25 by cyclomatic-ish complexity ===")
for f in sorted(funcs, key=lambda x: -x["cc"])[:25]:
    print(f"cc={f['cc']:5d}  span={f['span']:5d}  {f['file']}:{f['line']}  {f['name']}")

print("\n=== TOP 20 files where (comment+docstring) > code ===")
rows = []
for r, d in files.items():
    c = d.get("code_lines", 0)
    cm = d.get("comment_lines", 0) + d.get("doc_lines", 0)
    if c >= 50:
        rows.append((cm / max(c, 1), cm, c, d["lines"], r))
rows.sort(reverse=True)
for ratio, cm, c, ln, r in rows[:25]:
    print(f"ratio={ratio:5.2f}  comments+doc={cm:6d}  code={c:6d}  lines={ln:6d}  {r}")
