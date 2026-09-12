"""slow-marker census, validation/ + benchmarks/ inventory, global-config census, API surface."""
import ast, os, re, json, collections

ROOT = r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy"


def pyfiles(base):
    for dp, dn, fn in os.walk(base):
        dn[:] = [d for d in dn if d != "__pycache__"]
        for f in fn:
            if f.endswith(".py"):
                yield os.path.join(dp, f)


def rel(p):
    return os.path.relpath(p, ROOT).replace("\\", "/")


# ---------- 1. slow marker census ----------
tests = {rel(p): open(p, encoding="utf-8", errors="replace").read()
         for p in pyfiles(os.path.join(ROOT, "tests"))}
slow_files = [k for k, v in tests.items() if re.search(r"mark\.slow", v)]
nslow = sum(len(re.findall(r"mark\.slow", v)) for v in tests.values())
print(f"=== 'slow' marker: {nslow} applications in {len(slow_files)} of {len(tests)} files")
for k in slow_files:
    print("   ", k, "x", len(re.findall(r"mark\.slow", tests[k])))
d = json.load(open(os.path.join(ROOT, ".test_durations")))
slow_t = sum(v for k, v in d.items() if k.split("::")[0].replace("\\", "/") in slow_files)
print(f"    durations attributable to files containing mark.slow: {slow_t:.0f}s "
      f"({100*slow_t/sum(d.values()):.1f}% of {sum(d.values()):.0f}s)")
for m in ("integration", "regression", "bench", "unit"):
    n = sum(len(re.findall(r"mark\." + m + r"\b", v)) for v in tests.values())
    print(f"    mark.{m}: {n} applications")

# ---------- 2. validation / benchmarks inventory ----------
for sub in ("validation", "benchmarks", ".benchmarks", "scripts", "examples", "docs"):
    base = os.path.join(ROOT, sub)
    if not os.path.isdir(base):
        print(f"\n### {sub}/ MISSING")
        continue
    n_py = n_md = n_other = 0
    dirs = set()
    total = 0
    for dp, dn, fn in os.walk(base):
        dn[:] = [x for x in dn if x != "__pycache__"]
        for f in fn:
            total += 1
            if f.endswith(".py"):
                n_py += 1
            elif f.endswith(".md"):
                n_md += 1
            else:
                n_other += 1
        if dp != base:
            dirs.add(rel(dp))
    print(f"\n### {sub}/ : {total} files ({n_py} .py, {n_md} .md, {n_other} other), "
          f"{len(dirs)} subdirs")
    if sub in ("validation", "benchmarks", "scripts"):
        for dd in sorted(dirs)[:40]:
            print("     ", dd)
        print("     top-level files:", sorted(os.listdir(base))[:40])

# ---------- 3. global mutable config census ----------
print("\n\n===== GLOBAL MUTABLE CONFIG CENSUS =====")
setters = collections.defaultdict(list)
for p in pyfiles(os.path.join(ROOT, "lumenairy")):
    src = open(p, encoding="utf-8", errors="replace").read()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        continue
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            n = node.name
            if re.match(r"^(set_|get_|reset_|enable_|disable_|clear_|configure_)", n) or \
               n.endswith("_context") or "override" in n:
                # does it touch a module global?
                has_global = any(isinstance(x, ast.Global) for x in ast.walk(node))
                is_cm = any(
                    (isinstance(dec, ast.Attribute) and dec.attr == "contextmanager") or
                    (isinstance(dec, ast.Name) and dec.id == "contextmanager")
                    for dec in node.decorator_list)
                setters[rel(p)].append((n, node.lineno, has_global, is_cm))
rows = []
for f, items in setters.items():
    for n, ln, g, cm in items:
        rows.append((f, ln, n, g, cm))
mut = [r for r in rows if r[3]]     # writes a module global
print(f"functions named set_/get_/reset_/enable_/disable_/clear_/configure_*: {len(rows)}")
print(f"   of which actually WRITE a module-level global (`global` stmt): {len(mut)}")
print(f"   context-manager forms among all of them: {sum(1 for r in rows if r[4])}")
# group by stem
stems = collections.defaultdict(dict)
for f, ln, n, g, cm in rows:
    m = re.match(r"^(set|get|reset|enable|disable|clear|configure)_(.*)$", n)
    if m:
        stems[(f, m.group(2))][m.group(1)] = (ln, g, cm)
print(f"\n--- setter/getter families (module, knob) : which verbs exist ---")
nocm = []
noreset = []
for (f, k), verbs in sorted(stems.items()):
    if "set" in verbs:
        vs = ",".join(sorted(verbs))
        cm = any(v[2] for v in verbs.values())
        print(f"  {f}:{verbs['set'][0]:6d}  {k:44s} [{vs}]"
              f"{'  +ctxmgr' if cm else ''}")
        if not cm:
            nocm.append(f"{f}::{k}")
        if "reset" not in verbs:
            noreset.append(f"{f}::{k}")
print(f"\n  set_* knobs WITHOUT a context-manager form: {len(nocm)}")
print(f"  set_* knobs WITHOUT a reset_* form:          {len(noreset)}")

# environment variables read
envs = collections.Counter()
for p in pyfiles(os.path.join(ROOT, "lumenairy")):
    src = open(p, encoding="utf-8", errors="replace").read()
    for m in re.findall(r"environ(?:\.get)?\(?\[?[\"']([A-Z0-9_]+)[\"']", src):
        envs[m] += 1
print(f"\n--- environment variables read by the library: {len(envs)} ---")
for k, n in envs.most_common(40):
    print(f"   {n:3d}  {k}")

# ---------- 4. public API surface ----------
init = open(os.path.join(ROOT, "lumenairy", "__init__.py"), encoding="utf-8",
            errors="replace").read()
tree = ast.parse(init)
for node in tree.body:
    if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
        try:
            names = ast.literal_eval(node.value)
            print(f"\n===== lumenairy.__all__ : {len(names)} symbols =====")
        except Exception:
            print("\n__all__ not a literal")
print("   __all__ += occurrences:", init.count("__all__ +="),
      " __all__.extend:", init.count("__all__.extend"))
# tier comments
tiers = re.findall(r"^#\s*(TIER|Tier|tier)\s*([0-9A-Za-z ]*)", init, re.M)
print("   tier markers in __init__.py:", len(tiers))
