"""Targeted duplication probes + token-level near-duplicate clustering (difflib)."""
import ast, os, re, collections, difflib, itertools

ROOT = r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy"


def pyfiles(base):
    for dp, dn, fn in os.walk(base):
        dn[:] = [d for d in dn if d != "__pycache__"]
        for f in fn:
            if f.endswith(".py"):
                yield os.path.join(dp, f)


def rel(p):
    return os.path.relpath(p, ROOT).replace("\\", "/")


srcs = {rel(p): open(p, encoding="utf-8", errors="replace").read()
        for p in pyfiles(os.path.join(ROOT, "lumenairy"))}

print("===== targeted probes (definition sites) =====")
for name in ("_ensure_cupy_loaded", "_is_cupy_array", "_load_numba", "_have_numexpr",
             "_numexpr", "_copy_prescription", "_sag", "_surface_sag", "_eval_sag",
             "_exit_vertex", "_ensure_numba", "_HAS_CUPY", "_get_cupy", "_as_numpy",
             "_to_numpy", "_asnumpy", "_xp_for", "_array_namespace"):
    sites = []
    for k, s in srcs.items():
        for m in re.finditer(r"^(?:\s*)def\s+(" + re.escape(name) + r"\w*)\s*\(", s, re.M):
            ln = s[:m.start()].count("\n") + 1
            sites.append((k, ln, m.group(1)))
    if len(sites) >= 2:
        print(f"\n  {name}* : {len(sites)} definitions in {len({x[0] for x in sites})} modules")
        for k, ln, nm in sorted(sites)[:14]:
            print(f"      {k}:{ln}  {nm}")
        if len(sites) > 14:
            print(f"      ... +{len(sites)-14}")

print("\n\n===== 'exit vertex transfer'  t = -z / N   idiom =====")
for k, s in srcs.items():
    for m in re.finditer(r"^.*\bt\s*=\s*-\s*\w*z\w*\s*/\s*\w*[nN]\w*.*$", s, re.M):
        ln = s[:m.start()].count("\n") + 1
        print(f"   {k}:{ln}  {m.group(0).strip()[:120]}")

# ---------- difflib near-duplicate clustering over function bodies >= 12 lines ----------
bodies = []
for k, s in srcs.items():
    try:
        tree = ast.parse(s)
    except SyntaxError:
        continue
    lines = s.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            span = (node.end_lineno or node.lineno) - node.lineno + 1
            if span < 12 or span > 400:
                continue
            body = [re.sub(r"\s+", " ", l.strip())
                    for l in lines[node.lineno: (node.end_lineno or node.lineno)]
                    if l.strip() and not l.strip().startswith("#")]
            body = [l for l in body if not (l.startswith('"""') or l.startswith("'''"))]
            if len(body) < 10:
                continue
            bodies.append((k, node.lineno, node.name, span, body))
print(f"\n\n===== difflib near-duplicate clusters over {len(bodies)} functions "
      f"(>=12 lines, code-only) =====")
# bucket by length to keep it O(n * bucket)
buckets = collections.defaultdict(list)
for b in bodies:
    buckets[len(b[4]) // 4].append(b)
pairs = []
for key, grp in buckets.items():
    cand = grp + buckets.get(key + 1, [])
    for a, b in itertools.combinations(cand, 2):
        if a[0] == b[0] and a[1] == b[1]:
            continue
        r = difflib.SequenceMatcher(None, a[4], b[4]).quick_ratio()
        if r < 0.80:
            continue
        r = difflib.SequenceMatcher(None, a[4], b[4]).ratio()
        if r >= 0.80:
            pairs.append((r, a, b))
pairs.sort(key=lambda p: -(p[0] * min(len(p[1][4]), len(p[2][4]))))
seen = set()
n = 0
for r, a, b in pairs:
    key = (a[0], a[1], b[0], b[1])
    if key in seen:
        continue
    seen.add(key)
    n += 1
    if n > 22:
        break
    print(f"\n  [{n}] similarity {r:.2f}  (~{min(len(a[4]),len(b[4]))} code lines)")
    print(f"       {a[0]}:{a[1]}  {a[2]}  ({a[3]} lines)")
    print(f"       {b[0]}:{b[1]}  {b[2]}  ({b[3]} lines)")
print(f"\n  total pairs with similarity >= 0.80: {len(pairs)}")
crossmod = sum(1 for r, a, b in pairs if a[0] != b[0])
print(f"  of which CROSS-MODULE: {crossmod}")
print(f"  redundant code lines (sum over cross-module pairs, dedup by target): "
      f"~{sum(min(len(a[4]), len(b[4])) for r, a, b in pairs if a[0] != b[0])}")
