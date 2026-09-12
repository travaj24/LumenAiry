"""Do the `code refs` in README/docs still resolve? (random sample, seed 0)"""
import os, re, random, importlib

ROOT = r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy"
import sys
sys.path.insert(0, ROOT)
import lumenairy as lm

texts = {}
for f in ("README.md", "ROADMAP.md", "Migration-Guide.md", "CONVENTIONS.md"):
    p = os.path.join(ROOT, f)
    if os.path.exists(p):
        texts[f] = open(p, encoding="utf-8", errors="replace").read()
for dp, dn, fn in os.walk(os.path.join(ROOT, "docs")):
    for f in fn:
        if f.endswith(".md") and not dp.endswith("audits"):
            p = os.path.join(dp, f)
            texts[os.path.relpath(p, ROOT).replace("\\", "/")] = \
                open(p, encoding="utf-8", errors="replace").read()

# collect `identifier` refs that look like python names
cands = {}
for k, s in texts.items():
    for m in re.finditer(r"`([A-Za-z_][A-Za-z0-9_]{2,})`", s):
        n = m.group(1)
        if n.lower() in ("true", "false", "none", "int", "str", "float", "bool",
                         "dict", "list", "tuple", "numpy", "scipy", "python",
                         "pytest", "main", "and", "not", "the"):
            continue
        cands.setdefault(n, []).append((k, s[:m.start()].count("\n") + 1))
print(f"distinct backtick identifiers across README/ROADMAP/Migration/CONVENTIONS/docs "
      f"(excluding docs/audits): {len(cands)}")

random.seed(0)
sample = random.sample(sorted(cands), min(60, len(cands)))
ok = bad = 0
missing = []
for n in sample:
    if hasattr(lm, n):
        ok += 1
        continue
    # try as a submodule attribute
    found = False
    for mod in ("lumenairy.elements", "lumenairy.propagators", "lumenairy.analysis",
                "lumenairy.raytrace", "lumenairy.io", "lumenairy.optimize",
                "lumenairy.sources", "lumenairy.algebra", "lumenairy.backend"):
        try:
            m = importlib.import_module(mod)
        except Exception:
            continue
        if hasattr(m, n):
            found = True
            break
    if found:
        ok += 1
    else:
        bad += 1
        missing.append((n, cands[n][0]))
print(f"\nsample of {len(sample)} (random.seed(0)): RESOLVE={ok}  DO-NOT-RESOLVE={bad} "
      f"({100*bad/len(sample):.0f}%)")
print("\nnon-resolving refs (name -> first citation):")
for n, (f, ln) in missing:
    print(f"   {n:44s}  {f}:{ln}   (cited {len(cands[n])}x)")

# also: README-only, weighted by citation count -> the ones users will hit
print("\n\n=== top-cited identifiers in README.md that do NOT resolve ===")
rc = {}
s = texts.get("README.md", "")
for m in re.finditer(r"`([A-Za-z_][A-Za-z0-9_]{2,})`", s):
    rc[m.group(1)] = rc.get(m.group(1), 0) + 1
bad2 = []
for n, c in sorted(rc.items(), key=lambda kv: -kv[1])[:250]:
    if hasattr(lm, n):
        continue
    found = False
    for mod in ("lumenairy.elements", "lumenairy.propagators", "lumenairy.analysis",
                "lumenairy.raytrace", "lumenairy.io", "lumenairy.optimize",
                "lumenairy.sources", "lumenairy.algebra", "lumenairy.backend"):
        try:
            m2 = importlib.import_module(mod)
        except Exception:
            continue
        if hasattr(m2, n):
            found = True
            break
    if not found:
        bad2.append((c, n))
print(f"of the 250 most-cited README identifiers, {len(bad2)} do not resolve:")
for c, n in bad2[:35]:
    print(f"   {c:4d}x  {n}")
