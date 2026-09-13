"""identity_check.py -- for every modified lumenairy/**/*.py, compare HEAD vs working tree on the
docstring-stripped AST (positions ignored).  Independent of the sweep's own applier/checker."""
import ast, subprocess, sys, pathlib
REPO = pathlib.Path("D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
def strip_docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    return tree
def fp(src):
    return ast.dump(strip_docstrings(ast.parse(src)))
mod = subprocess.run(["git", "diff", "--name-only", "--", "lumenairy"], cwd=REPO, capture_output=True, text=True).stdout.split()
bad = 0
for path in mod:
    head = subprocess.run(["git", "show", f"HEAD:{path}"], cwd=REPO, capture_output=True).stdout.decode("utf-8")
    work = (REPO / path).read_text(encoding="utf-8")
    same = fp(head) == fp(work)
    raw_same = ast.dump(ast.parse(head)) == ast.dump(ast.parse(work))
    print(f"{'OK ' if same else 'DIFF'}  {path:55s} docstring-free AST equal={same}  raw AST equal={raw_same}")
    bad += not same
print(f"{len(mod)} modules, {bad} with executable differences")
sys.exit(1 if bad else 0)
