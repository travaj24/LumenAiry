import ast, sys, io, tokenize
path = sys.argv[1]; lo = int(sys.argv[2]); hi = int(sys.argv[3])
src = open(path, encoding='utf-8').read()
lines = src.splitlines()
tree = ast.parse(src)
doc = set()
for node in ast.walk(tree):
    if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            f0 = node.body[0]
            for L in range(f0.lineno, f0.end_lineno+1):
                doc.add(L)
for i in range(lo, min(hi, len(lines))+1):
    L = lines[i-1]
    if i in doc: continue
    s = L.strip()
    if s.startswith('#'): continue
    if not s: continue
    print(f"{i}\t{L}")
