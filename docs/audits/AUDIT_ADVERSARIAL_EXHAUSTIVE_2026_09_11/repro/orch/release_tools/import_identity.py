"""For every modified lumenairy/**/*.py: the multiset of import bindings (module, level, name, asname)
must be unchanged, and the AST with import statements removed must be identical to HEAD."""
import ast, subprocess, sys, pathlib
REPO = pathlib.Path("D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
class Strip(ast.NodeTransformer):
    def __init__(self): self.binds = []
    def visit_Import(self, node):
        for a in node.names: self.binds.append(("import", 0, a.name, a.asname))
        return None
    def visit_ImportFrom(self, node):
        for a in node.names: self.binds.append((node.module, node.level, a.name, a.asname))
        return None
def analyse(src):
    t = ast.parse(src); s = Strip(); t = s.visit(t); ast.fix_missing_locations(t)
    return sorted(map(str, s.binds)), ast.dump(t)
mod = subprocess.run(["git", "diff", "--name-only", "--", "lumenairy"], cwd=REPO, capture_output=True, text=True).stdout.split()
bad = 0
for path in mod:
    head = subprocess.run(["git", "show", f"HEAD:{path}"], cwd=REPO, capture_output=True).stdout.decode("utf-8")
    work = (REPO / path).read_text(encoding="utf-8")
    hb, ha = analyse(head); wb, wa = analyse(work)
    ok = hb == wb and ha == wa
    n_head = len(ast.parse(head).body); n_work = len(ast.parse(work).body)
    print(f"{'OK ' if ok else 'DIFF'} {path:55s} bindings equal={hb == wb} ({len(hb)}), non-import AST equal={ha == wa}")
    bad += not ok
print(f"{len(mod)} modules, {bad} differ beyond import statement merging")
sys.exit(bad)
