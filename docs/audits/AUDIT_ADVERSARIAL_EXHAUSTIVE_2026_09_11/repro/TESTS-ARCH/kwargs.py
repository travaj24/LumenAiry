"""Extract apply_real_lens-family signatures and check which kwargs the test corpus exercises."""
import ast, os, re, json, collections

ROOT = r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy"
TESTS = os.path.join(ROOT, "tests")

TARGETS = {
    "apply_real_lens": "lumenairy/elements/_lens_real.py",
    "_apply_real_lens_impl": "lumenairy/elements/_lens_real.py",
    "apply_real_lens_traced": "lumenairy/elements/_lens_traced.py",
    "apply_real_lens_maslov": "lumenairy/elements/lenses_maslov.py",
    "apply_real_lens_gbd": "lumenairy/elements/lenses_gbd.py",
    "apply_real_lens_fga": "lumenairy/propagators/fga.py",
    "apply_real_lens_traced_multibranch": "lumenairy/elements/_lens_traced_multibranch.py",
}

sigs = {}
for fn, rel in TARGETS.items():
    p = os.path.join(ROOT, rel)
    tree = ast.parse(open(p, encoding="utf-8", errors="replace").read())
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == fn:
            a = node.args
            names = ([x.arg for x in a.posonlyargs] + [x.arg for x in a.args]
                     + ([a.vararg.arg] if a.vararg else [])
                     + [x.arg for x in a.kwonlyargs]
                     + ([a.kwarg.arg] if a.kwarg else []))
            defaults = {}
            pos = [x.arg for x in a.posonlyargs] + [x.arg for x in a.args]
            for i, d in enumerate(a.defaults):
                defaults[pos[len(pos) - len(a.defaults) + i]] = ast.unparse(d)
            for k, d in zip(a.kwonlyargs, a.kw_defaults):
                if d is not None:
                    defaults[k.arg] = ast.unparse(d)
            sigs[fn] = {"file": rel, "line": node.lineno, "params": names, "defaults": defaults,
                        "nkw": len(names)}
            break

# read the whole test corpus once
corpus = {}
for dp, dn, files in os.walk(TESTS):
    dn[:] = [d for d in dn if d != "__pycache__"]
    for f in files:
        if f.endswith(".py"):
            p = os.path.join(dp, f)
            corpus[os.path.relpath(p, ROOT).replace("\\", "/")] = \
                open(p, encoding="utf-8", errors="replace").read()
# also validation/, benchmarks/, examples/
extra = {}
for sub in ("validation", "benchmarks", "examples"):
    for dp, dn, files in os.walk(os.path.join(ROOT, sub)):
        dn[:] = [d for d in dn if d != "__pycache__"]
        for f in files:
            if f.endswith(".py"):
                p = os.path.join(dp, f)
                extra[os.path.relpath(p, ROOT).replace("\\", "/")] = \
                    open(p, encoding="utf-8", errors="replace").read()

print(f"test files: {len(corpus)}  |  validation+benchmarks+examples files: {len(extra)}")

for fn, s in sigs.items():
    print(f"\n########## {fn}  ({s['file']}:{s['line']})  {s['nkw']} params ##########")
    untested_tests = []
    untested_all = []
    weak = []
    for p in s["params"]:
        if p in ("E_in", "x", "y", "wavelength", "self"):
            continue
        pat = re.compile(r"\b" + re.escape(p) + r"\s*=")
        nt = sum(1 for src in corpus.values() if pat.search(src))
        ne = sum(1 for src in extra.values() if pat.search(src))
        if nt == 0 and ne == 0:
            untested_all.append(p)
        elif nt == 0:
            untested_tests.append((p, ne))
        elif nt <= 2:
            weak.append((p, nt))
    print(f"  params: {len(s['params'])}")
    print(f"  NEVER mentioned as kwarg anywhere (tests+validation+benchmarks+examples): "
          f"{len(untested_all)}")
    for p in untested_all:
        print(f"     ! {p}   (default {s['defaults'].get(p, '<required>')})")
    print(f"  not in tests/ but present elsewhere: {len(untested_tests)}")
    for p, n in untested_tests:
        print(f"     ~ {p}  ({n} non-test files)")
    print(f"  thin coverage (<=2 test files mention it): {len(weak)}")
    for p, n in sorted(weak, key=lambda x: x[1]):
        print(f"     . {p}  ({n} test files)   default={s['defaults'].get(p,'<req>')}")

json.dump(sigs, open(os.path.join(
    r"docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/TESTS-ARCH",
    "sigs.json"), "w"), indent=1)
