"""Test-suite census + adversarial classification of a random sample."""
import ast, os, re, json, random, collections

ROOT = r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy"
OUT = r"docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/TESTS-ARCH"


def pyfiles(base):
    for dp, dn, fn in os.walk(base):
        dn[:] = [d for d in dn if d != "__pycache__"]
        for f in fn:
            if f.endswith(".py"):
                yield os.path.join(dp, f)


def rel(p):
    return os.path.relpath(p, ROOT).replace("\\", "/")


# ---------------- 1. census ----------------
tests = {}
for p in pyfiles(os.path.join(ROOT, "tests")):
    src = open(p, encoding="utf-8", errors="replace").read()
    tests[rel(p)] = src

unit = {k: v for k, v in tests.items() if k.startswith("tests/unit/")}
integ = {k: v for k, v in tests.items() if k.startswith("tests/integration/")}
print(f"tests/ total .py files: {len(tests)}  unit={len(unit)} integration={len(integ)} "
      f"other={len(tests)-len(unit)-len(integ)}")
for k in tests:
    if not k.startswith(("tests/unit/", "tests/integration/")):
        print("   other:", k)

lines = {k: v.count("\n") + 1 for k, v in tests.items()}
tot = sum(lines.values())
print(f"total test LOC: {tot}")
sl = sorted(lines.items(), key=lambda kv: -kv[1])
print("largest 15 test files:")
for k, n in sl[:15]:
    print(f"   {n:6d}  {k}")
print("smallest 15 test files:")
for k, n in sl[-15:]:
    print(f"   {n:6d}  {k}")
import statistics
print(f"median test file LOC: {statistics.median(lines.values()):.0f}  mean: {tot/len(lines):.0f}")

# ---------------- 2. which lumenairy modules do tests import? ----------------
src_mods = set()
for p in pyfiles(os.path.join(ROOT, "lumenairy")):
    r = rel(p)[:-3]
    if r.endswith("/__init__"):
        r = r[:-9]
    src_mods.add(r.replace("/", "."))

imported = collections.Counter()
# textual: count mentions of the module basename
for k, src in tests.items():
    for m in re.findall(r"lumenairy(?:\.[A-Za-z_][A-Za-z0-9_]*)+", src):
        cand = m
        while cand and cand not in src_mods:
            if "." not in cand:
                cand = None
                break
            cand = cand.rsplit(".", 1)[0]
        if cand:
            imported[cand] += 1
    # relative "from lumenairy.x import y" already matched; also from-import forms
    for m in re.findall(r"from\s+(lumenairy(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\s+import", src):
        cand = m
        while cand and cand not in src_mods:
            if "." not in cand:
                cand = None
                break
            cand = cand.rsplit(".", 1)[0]
        if cand:
            imported[cand] += 1

never = sorted(m for m in src_mods if imported.get(m, 0) == 0)
print(f"\n=== source modules NEVER named in any test file: {len(never)} / {len(src_mods)} ===")
for m in never:
    print("   ", m)

# dedicated test file: a test file whose *name* maps to the module
basenames = {m.rsplit(".", 1)[-1]: m for m in src_mods}
have_dedicated = set()
for k in tests:
    stem = os.path.basename(k)[:-3]
    for b, m in basenames.items():
        if b and (stem == "test_" + b or stem.endswith("_" + b)):
            have_dedicated.add(m)
print(f"\n=== source modules with a name-matched dedicated test file: {len(have_dedicated)} ===")
print(f"=== modules with NO name-matched test file: {len(src_mods)-len(have_dedicated)} ===")

# ---------------- 3. adversarial classification ----------------
# collect test functions with bodies
funcs = []
for k, src in tests.items():
    try:
        tree = ast.parse(src)
    except SyntaxError:
        continue
    srclines = src.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            body = "\n".join(srclines[node.lineno - 1: (node.end_lineno or node.lineno)])
            funcs.append({"file": k, "line": node.lineno, "name": node.name,
                          "span": (node.end_lineno or node.lineno) - node.lineno + 1,
                          "body": body})
print(f"\n=== test functions found by AST: {len(funcs)} in {len(tests)} files ===")

random.seed(0)
sample = random.sample(funcs, 70)
json.dump([{k: v for k, v in f.items()} for f in sample],
          open(os.path.join(OUT, "sample70.json"), "w"))


def classify(b):
    tags = []
    low = b.lower()
    if re.search(r"monkeypatch|mock\.|MagicMock|unittest\.mock|patch\(", b):
        tags.append("MOCK")
    if re.search(r"assert\s+True\b|assert\s+1\b|^\s*pass\s*$", b, re.M):
        tags.append("TRIVIAL")
    if re.search(r"__doc__|docstring|CHANGELOG|inspect\.getsource|read_text\(\)", b):
        tags.append("DOC/SOURCE-TEXT")
    if re.search(r"inspect\.signature|hasattr\(|callable\(|__all__|isinstance\(", b):
        tags.append("CONTRACT/SIG")
    if re.search(r"pytest\.raises|pytest\.warns", b):
        tags.append("ERROR-PATH")
    if re.search(r"pytest\.skip|importorskip", b):
        tags.append("SKIP")
    if re.search(r"np\.array_equal|allclose\(.*,\s*E_\w+|bit-for-bit|bit_for_bit", b):
        tags.append("SELF-CONSISTENCY")
    if re.search(r"assert\s+not\s+np\.allclose|assert\s+.*!=", b):
        tags.append("NEGATIVE-ONLY")
    if re.search(r"np\.isfinite|np\.all\(np\.isfinite", b):
        tags.append("SMOKE/FINITE")
    if re.search(r"\d\.\d{3,}e[+-]\d|\d\.\d{4,}", b):
        tags.append("MAGIC-NUMBER")
    if re.search(r"analytic|closed[- ]form|exact|theory|textbook|airy|fresnel|gaussian|"
                 r"reference|oracle", low):
        tags.append("CLAIMS-ORACLE")
    if not re.search(r"\bassert\b", b):
        tags.append("NO-ASSERT")
    return tags


counts = collections.Counter()
for f in funcs:
    for t in classify(f["body"]):
        counts[t] += 1
print("\n=== whole-corpus heuristic tags (over all", len(funcs), "test functions) ===")
for t, n in counts.most_common():
    print(f"   {t:20s} {n:6d}  ({100*n/len(funcs):5.1f}%)")

print("\n=== SAMPLE OF 70 (random.seed(0)) -- tags ===")
scounts = collections.Counter()
for f in sample:
    tg = classify(f["body"])
    for t in tg:
        scounts[t] += 1
    print(f"{f['file']}:{f['line']}  {f['name'][:70]}  span={f['span']:3d}  [{','.join(tg)}]")
print("\n--- sample tag totals ---")
for t, n in scounts.most_common():
    print(f"   {t:20s} {n:4d}  ({100*n/70:5.1f}%)")
