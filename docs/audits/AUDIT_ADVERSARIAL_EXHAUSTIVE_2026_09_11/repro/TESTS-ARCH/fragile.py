"""Check the test corpus against TESTING_STANDARDS S1-S5 fragile shapes + find non-independent oracles."""
import ast, os, re, collections, json

ROOT = r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy"


def pyfiles(base):
    for dp, dn, fn in os.walk(base):
        dn[:] = [d for d in dn if d != "__pycache__"]
        for f in fn:
            if f.endswith(".py"):
                yield os.path.join(dp, f)


def rel(p):
    return os.path.relpath(p, ROOT).replace("\\", "/")


corpus = {}
for p in pyfiles(os.path.join(ROOT, "tests")):
    corpus[rel(p)] = open(p, encoding="utf-8", errors="replace").read().splitlines()


def scan(pat, label, limit=25, flags=0):
    rx = re.compile(pat, flags)
    hits = []
    for k, lines in corpus.items():
        for i, l in enumerate(lines, 1):
            if rx.search(l):
                hits.append((k, i, l.strip()))
    print(f"\n===== {label}: {len(hits)} hits in "
          f"{len({h[0] for h in hits})} files =====")
    for k, i, l in hits[:limit]:
        print(f"  {k}:{i}  {l[:150]}")
    if len(hits) > limit:
        print(f"  ... +{len(hits)-limit} more")
    return hits


# ---- S1: performance / timing assertions in asserts ----
scan(r"assert\b.*\b(perf_counter|elapsed|t_end|wall|_secs?\b|speedup|faster|slower)",
     "S1a: assertions on measured TIME / speed", 30)
scan(r"^\s*assert\b.*\b\d+(\.\d+)?\s*\*\s*(t|dt|t0|t1|base|ref)\b",
     "S1b: assertions on a measured-ratio bar", 20)
# ---- S3: resource-dependent skips ----
scan(r"pytest\.skip\(", "S3: pytest.skip() calls (resource / env preconditions)", 30)
scan(r"(psutil|os\.cpu_count|virtual_memory|free_gb|available|RAM|get_ram_budget)",
     "S3b: resource probes inside tests", 25)
# ---- S4: floor bars: bare tight literals ----
scan(r"assert\b.*<\s*[0-9]\.[0-9]{3,}e-\d+", "S4: assert < <many-digit> float literal", 30)
# ---- S5: exact counts of nondeterministic machinery ----
scan(r"assert\s+len\(.*\)\s*==\s*\d+", "S5: exact len() == N assertions", 25)
# ---- doc/changelog tests ----
scan(r"(CHANGELOG|README|\.md['\"]).*(read_text|open\(|in src|in doc)",
     "V: tests asserting on markdown / doc file CONTENT", 25)
scan(r"__doc__|inspect\.getsource", "V2: tests asserting on docstrings / source text", 30)
# ---- mock / monkeypatch of physics ----
scan(r"monkeypatch\.setattr\([^,]*(?:lens|propag|rcwa|pmm|carrier|trace|bor|solve)",
     "III: monkeypatching a physics function", 30, re.I)
print()

# ---------- tests that re-implement the formula they test ----------
# Look for a test file that imports a private helper AND contains a local
# re-implementation with the same math.
priv = collections.Counter()
for k, lines in corpus.items():
    src = "\n".join(lines)
    for m in re.findall(r"from\s+lumenairy[\w.]*\s+import\s+\(?([^)\n]*)", src):
        for name in re.split(r",\s*", m):
            name = name.strip().split(" as ")[0].strip()
            if name.startswith("_") and name:
                priv[(k, name)] += 1
print(f"===== tests importing PRIVATE (underscore) library symbols: "
      f"{len(priv)} distinct (file,symbol) pairs in "
      f"{len({k for k, _ in priv})} files =====")
by_file = collections.Counter(k for k, _ in priv)
for k, n in by_file.most_common(20):
    print(f"  {n:3d} private symbols  {k}")
