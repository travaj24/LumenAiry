"""Near-duplicate function-body detection across lumenairy via normalised-AST hashing."""
import ast, os, hashlib, collections

ROOT = r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy"


def pyfiles(base):
    for dp, dn, fn in os.walk(base):
        dn[:] = [d for d in dn if d != "__pycache__"]
        for f in fn:
            if f.endswith(".py"):
                yield os.path.join(dp, f)


def rel(p):
    return os.path.relpath(p, ROOT).replace("\\", "/")


class Norm(ast.NodeTransformer):
    """Erase identifiers & constants so structurally-identical bodies hash the same."""
    def visit_Name(self, n):
        return ast.copy_location(ast.Name(id="_", ctx=n.ctx), n)

    def visit_Attribute(self, n):
        self.generic_visit(n)
        return ast.copy_location(ast.Attribute(value=n.value, attr="_", ctx=n.ctx), n)

    def visit_arg(self, n):
        return ast.copy_location(ast.arg(arg="_", annotation=None), n)

    def visit_Constant(self, n):
        return ast.copy_location(ast.Constant(value=0), n)

    def visit_FunctionDef(self, n):
        self.generic_visit(n)
        n.name = "_"
        n.decorator_list = []
        n.returns = None
        return n


def strip_doc(node):
    b = list(node.body)
    if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) \
            and isinstance(b[0].value.value, str):
        b = b[1:]
    return b


exact = collections.defaultdict(list)   # exact-name+shape (same identifiers)
struct = collections.defaultdict(list)  # structure only
names = collections.defaultdict(list)

for p in pyfiles(os.path.join(ROOT, "lumenairy")):
    try:
        src = open(p, encoding="utf-8", errors="replace").read()
        tree = ast.parse(src)
    except Exception:
        continue
    r = rel(p)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = strip_doc(node)
            nstmt = sum(1 for _ in ast.walk(ast.Module(body=body, type_ignores=[])))
            span = (node.end_lineno or node.lineno) - node.lineno + 1
            if span < 8:
                continue
            mod = ast.Module(body=body, type_ignores=[])
            try:
                lit = ast.dump(mod)
            except Exception:
                continue
            exact[hashlib.md5(lit.encode()).hexdigest()].append(
                (r, node.lineno, node.name, span))
            m2 = Norm().visit(ast.parse(ast.unparse(mod)))
            struct[hashlib.md5(ast.dump(m2).encode()).hexdigest()].append(
                (r, node.lineno, node.name, span))
            names[node.name].append((r, node.lineno, span))

print("===== EXACT duplicate function bodies (identical AST incl. identifiers, >=8 lines) =====")
cl = sorted((v for v in exact.values() if len(v) > 1),
            key=lambda v: -(len(v) * v[0][3]))
tot_dup_lines = 0
for i, v in enumerate(cl[:22], 1):
    span = v[0][3]
    tot_dup_lines += span * (len(v) - 1)
    print(f"\n  [{i}] {len(v)} copies x ~{span} lines  (name(s): "
          f"{sorted({x[2] for x in v})})")
    for r, ln, nm, sp in sorted(v):
        print(f"        {r}:{ln}  {nm}  ({sp} lines)")
print(f"\n  duplicated lines removable from top-22 exact clusters: ~{tot_dup_lines}")
print(f"  total exact-dup clusters: {len(cl)}; "
      f"total redundant lines across ALL clusters: "
      f"~{sum(v[0][3]*(len(v)-1) for v in cl)}")

print("\n\n===== STRUCTURAL near-duplicates (identifiers/constants erased, >=8 lines) =====")
cs = sorted((v for v in struct.values() if len(v) > 2),
            key=lambda v: -(len(v) * v[0][3]))
for i, v in enumerate(cs[:14], 1):
    print(f"\n  [{i}] {len(v)} copies x ~{v[0][3]} lines  names={sorted({x[2] for x in v})[:6]}")
    for r, ln, nm, sp in sorted(v)[:8]:
        print(f"        {r}:{ln}  {nm}")
    if len(v) > 8:
        print(f"        ... +{len(v)-8} more")

print("\n\n===== same FUNCTION NAME defined in >=3 modules (>=8 lines) =====")
for nm, v in sorted(names.items(), key=lambda kv: -len(kv[1])):
    if len(v) < 3:
        continue
    mods = {x[0] for x in v}
    if len(mods) < 3:
        continue
    print(f"  {nm:42s} {len(v)} defs in {len(mods)} modules, "
          f"~{sum(x[2] for x in v)} total lines")
    for r, ln, sp in sorted(v)[:8]:
        print(f"        {r}:{ln} ({sp} lines)")
    if len(v) > 8:
        print(f"        ... +{len(v)-8}")
