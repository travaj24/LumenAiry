"""Module-level import graph + cycle detection for lumenairy (read-only, AST based)."""
import ast, os, sys, json, collections

ROOT = r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy"
PKG = os.path.join(ROOT, "lumenairy")
OUT = r"docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/TESTS-ARCH"
SKIP = {"__pycache__"}


def modname(path):
    r = os.path.relpath(path, ROOT).replace("\\", "/")
    assert r.endswith(".py")
    r = r[:-3]
    if r.endswith("/__init__"):
        r = r[: -len("/__init__")]
    return r.replace("/", ".")


mods = {}
for dp, dn, fn in os.walk(PKG):
    dn[:] = [d for d in dn if d not in SKIP]
    for f in fn:
        if f.endswith(".py"):
            p = os.path.join(dp, f)
            mods[modname(p)] = p

known = set(mods)

# top-level: edges for module-level imports only (not inside functions)
top_edges = collections.defaultdict(set)   # module-level (import executes at import time)
lazy_edges = collections.defaultdict(set)  # function-level (deferred)
all_edges = collections.defaultdict(set)


def resolve(mod, node):
    """Return set of lumenairy module names this import targets."""
    out = set()
    if isinstance(node, ast.Import):
        for a in node.names:
            out.add(a.name)
    else:  # ImportFrom
        if node.level:
            parts = mod.split(".")
            if mods[mod].endswith("__init__.py"):
                # mod IS the package: level 1 -> itself, level 2 -> parent
                base = parts[: len(parts) - (node.level - 1)] if node.level > 1 else parts
            else:
                # module file: level 1 -> containing package
                base = parts[: len(parts) - node.level]
            pkgbase = ".".join(base)
            target = pkgbase + ("." + node.module if node.module else "")
            out.add(target)
            for a in node.names:
                out.add(target + "." + a.name)
        else:
            if node.module:
                out.add(node.module)
                for a in node.names:
                    out.add(node.module + "." + a.name)
    return {o for o in out if o.split(".")[0] == "lumenairy"}


for mod, path in sorted(mods.items()):
    try:
        tree = ast.parse(open(path, encoding="utf-8", errors="replace").read())
    except SyntaxError:
        continue
    # mark function/class bodies
    deferred = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for ch in ast.walk(node):
                deferred.add(id(ch))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            tgts = resolve(mod, node)
            for t in tgts:
                # map "lumenairy.a.b.symbol" down to the longest known module prefix
                cand = t
                while cand and cand not in known:
                    if "." not in cand:
                        cand = None
                        break
                    cand = cand.rsplit(".", 1)[0]
                if not cand or cand == mod:
                    continue
                all_edges[mod].add(cand)
                if id(node) in deferred:
                    lazy_edges[mod].add(cand)
                else:
                    top_edges[mod].add(cand)

# strip lazy-only edges from top
for m in list(top_edges):
    pass


def find_cycles(edges, limit=400):
    """Tarjan SCC + elementary cycles within each SCC (bounded)."""
    g = {m: set(edges.get(m, ())) for m in mods}
    index = {}
    low = {}
    onstack = {}
    stack = []
    sccs = []
    counter = [0]

    def strong(v):
        work = [(v, iter(g.get(v, ())))]
        index[v] = low[v] = counter[0]; counter[0] += 1
        stack.append(v); onstack[v] = True
        while work:
            node, it = work[-1]
            advanced = False
            for w in it:
                if w not in index:
                    index[w] = low[w] = counter[0]; counter[0] += 1
                    stack.append(w); onstack[w] = True
                    work.append((w, iter(g.get(w, ()))))
                    advanced = True
                    break
                elif onstack.get(w):
                    low[node] = min(low[node], index[w])
            if not advanced:
                work.pop()
                if work:
                    low[work[-1][0]] = min(low[work[-1][0]], low[node])
                if low[node] == index[node]:
                    comp = []
                    while True:
                        w = stack.pop(); onstack[w] = False
                        comp.append(w)
                        if w == node:
                            break
                    if len(comp) > 1:
                        sccs.append(comp)
    for v in list(g):
        if v not in index:
            strong(v)
    return sccs, g


def two_cycles(edges):
    out = []
    for a, tgts in edges.items():
        for b in tgts:
            if a < b and a in edges.get(b, ()):
                out.append((a, b))
    return sorted(out)


print("=== modules:", len(mods))
print("=== module-level import edges:", sum(len(v) for v in top_edges.values()))
print("=== function-level (lazy) import edges:", sum(len(v) for v in lazy_edges.values()))
print("=== total distinct edges:", sum(len(v) for v in all_edges.values()))

for label, E in (("MODULE-LEVEL (import-time) ", top_edges), ("ALL (incl. lazy/in-function)", all_edges)):
    sccs, g = find_cycles(E)
    print(f"\n===== {label}: strongly-connected components with >1 module =====")
    for comp in sorted(sccs, key=lambda c: -len(c)):
        print(f"  SCC size {len(comp)}:")
        for m in sorted(comp):
            inner = sorted(x for x in E.get(m, ()) if x in set(comp))
            print(f"    {m}  ->  {', '.join(inner)}")
    print(f"  ({len(sccs)} SCCs)")
    tc = two_cycles(E)
    print(f"  --- 2-cycles ({len(tc)}) ---")
    for a, b in tc:
        print(f"    {a}  <->  {b}")

# layering
LAYERS = {
    "lumenairy.ui": 0, "lumenairy.optimize": 1, "lumenairy.analysis": 2,
    "lumenairy.propagators": 3, "lumenairy.elements": 4, "lumenairy.raytrace": 5,
    "lumenairy.sources": 6, "lumenairy.algebra": 7, "lumenairy.io": 3.5,
}


def layer(m):
    for k, v in LAYERS.items():
        if m == k or m.startswith(k + "."):
            return k, v
    return None, None


print("\n===== LAYERING VIOLATIONS (lower-layer module importing a higher-level subsystem) =====")
viol = collections.defaultdict(list)
for a, tgts in sorted(all_edges.items()):
    ka, la = layer(a)
    if ka is None:
        continue
    for b in sorted(tgts):
        kb, lb = layer(b)
        if kb is None or ka == kb:
            continue
        if lb < la:   # a (deeper) imports b (shallower / more app-level)
            lazy = b in lazy_edges.get(a, ())
            viol[(ka, kb)].append((a, b, "lazy" if lazy else "MODULE-LEVEL"))
for (ka, kb), items in sorted(viol.items(), key=lambda kv: -len(kv[1])):
    ml = sum(1 for _, _, k in items if k == "MODULE-LEVEL")
    print(f"\n  {ka}  ->  {kb}   ({len(items)} edges, {ml} at module level)")
    for a, b, kind in items[:14]:
        print(f"     [{kind:12s}] {a} -> {b}")
    if len(items) > 14:
        print(f"     ... +{len(items)-14} more")

json.dump({"top": {k: sorted(v) for k, v in top_edges.items()},
           "lazy": {k: sorted(v) for k, v in lazy_edges.items()},
           "all": {k: sorted(v) for k, v in all_edges.items()}},
          open(os.path.join(OUT, "importgraph.json"), "w"))

# fan-in / fan-out
fo = sorted(((len(v), k) for k, v in all_edges.items()), reverse=True)[:15]
fi = collections.Counter()
for a, tg in all_edges.items():
    for b in tg:
        fi[b] += 1
print("\n===== TOP 15 fan-out =====")
for n, k in fo:
    print(f"  {n:4d}  {k}")
print("===== TOP 15 fan-in =====")
for k, n in fi.most_common(15):
    print(f"  {n:4d}  {k}")
