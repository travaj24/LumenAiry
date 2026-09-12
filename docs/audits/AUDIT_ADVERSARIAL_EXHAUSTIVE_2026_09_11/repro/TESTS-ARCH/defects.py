"""Why the five orchestrator-found defects slipped through: search the corpus for the
specific test shapes that would have caught them."""
import ast, os, re, collections

ROOT = r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy"
T = os.path.join(ROOT, "tests")

corpus = {}
for dp, dn, fn in os.walk(T):
    dn[:] = [d for d in dn if d != "__pycache__"]
    for f in fn:
        if f.endswith(".py"):
            p = os.path.join(dp, f)
            corpus[os.path.relpath(p, ROOT).replace("\\", "/")] = \
                open(p, encoding="utf-8", errors="replace").read()


def files_with(*pats, allof=True, flags=0):
    rxs = [re.compile(p, flags) for p in pats]
    out = []
    for k, s in corpus.items():
        hits = [bool(r.search(s)) for r in rxs]
        if (all(hits) if allof else any(hits)):
            out.append(k)
    return out


def funcs_with(*pats, flags=0):
    """test functions whose BODY contains all patterns"""
    rxs = [re.compile(p, flags) for p in pats]
    out = []
    for k, s in corpus.items():
        if not all(r.search(s) for r in rxs):
            continue
        try:
            tree = ast.parse(s)
        except SyntaxError:
            continue
        lines = s.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and \
                    node.name.startswith("test"):
                body = "\n".join(lines[node.lineno - 1:(node.end_lineno or node.lineno)])
                if all(r.search(body) for r in rxs):
                    out.append((k, node.lineno, node.name))
    return out


print("#" * 78)
print("# D1  seidel_correction=True on a prescription whose LAST surface is CURVED,")
print("#     asserting on FOCUS POSITION")
print("#" * 78)
f = files_with(r"seidel_correction\s*=\s*True")
print("files passing seidel_correction=True:", f)
print("test functions:", funcs_with(r"seidel_correction\s*=\s*True"))
print("...of those, functions ALSO mentioning focus position:",
      funcs_with(r"seidel_correction\s*=\s*True", r"focus|z_focus|bfd|back_focal|argmax"))
print("...of those, using a prescription with R2 finite (curved last surface):",
      funcs_with(r"seidel_correction\s*=\s*True", r"R2\s*=\s*(?!float\('inf'\)|np\.inf)"))

print("\n" + "#" * 78)
print("# D2  surface_frame=True with a TILTED surface, asserting BEAM DEVIATION")
print("#" * 78)
print("files:", files_with(r"surface_frame\s*=\s*True"))
print("test functions with surface_frame=True:", funcs_with(r"surface_frame\s*=\s*True"))
print("...also mentioning tilt:", funcs_with(r"surface_frame\s*=\s*True", r"tilt"))
print("...also asserting on a centroid / deviation / chief-ray angle:",
      funcs_with(r"surface_frame\s*=\s*True",
                 r"centroid|deviat|chief|angle|direction|shift"))

print("\n" + "#" * 78)
print("# D3  apply_real_lens_traced with a REAL-dtype E_in")
print("#" * 78)
# any test that builds a real array and feeds it to a traced call
cand = funcs_with(r"apply_real_lens_traced",
                  r"float64|float32|np\.ones\(\s*\([^)]*\)\s*\)|dtype\s*=\s*float|"
                  r"np\.real\(|\.real\b")
print(f"traced tests touching a real dtype anywhere: {len(cand)}")
for c in cand[:15]:
    print("   ", c)
# more precise: a real-dtype array passed as the first positional arg
strict = []
for k, s in corpus.items():
    for m in re.finditer(r"apply_real_lens_traced\s*\(\s*(\w+)", s):
        var = m.group(1)
        ln = s[:m.start()].count("\n") + 1
        # look back 40 lines for the assignment of var
        back = "\n".join(s.splitlines()[max(0, ln - 40):ln])
        am = re.findall(re.escape(var) + r"\s*=\s*([^\n]*)", back)
        if am and not re.search(r"complex|1j|astype\(c|\.astype\(np\.complex", am[-1]):
            strict.append((k, ln, var, am[-1].strip()[:80]))
print(f"\ntraced call sites whose E_in assignment shows NO complex marker: {len(strict)}")
for k, ln, v, a in strict[:20]:
    print(f"   {k}:{ln}  {v} = {a}")

print("\n" + "#" * 78)
print("# D4  caustic='multibranch' with output_plane_distance at the PARAXIAL FOCUS")
print("#" * 78)
mb = funcs_with(r"multibranch")
print(f"test functions mentioning multibranch: {len(mb)}")
both = funcs_with(r"multibranch", r"output_plane_distance")
print(f"...also passing output_plane_distance: {len(both)}")
for x in both:
    print("   ", x)
foc = funcs_with(r"multibranch", r"output_plane_distance", r"efl|focal|f_par|paraxial|focus")
print(f"...and evaluating at a focal/paraxial distance: {len(foc)}")
for x in foc:
    print("   ", x)

print("\n" + "#" * 78)
print("# D5  newton_fit='spline' with a VIGNETTED ray")
print("#" * 78)
sp = funcs_with(r"newton_fit\s*=\s*['\"]spline['\"]")
print(f"test functions passing newton_fit='spline': {len(sp)}")
for x in sp:
    print("   ", x)
vg = funcs_with(r"newton_fit\s*=\s*['\"]spline['\"]", r"vignett|clip|aperture|blocked|dead|nan")
print(f"...also mentioning vignetting/aperture/nan: {len(vg)}")
for x in vg:
    print("   ", x)
print("\nfiles mentioning 'vignett' at all:", len(files_with(r"vignett", flags=re.I)))
for x in files_with(r"vignett", flags=re.I)[:15]:
    print("   ", x)

print("\n" + "#" * 78)
print("# kwarg-COMBINATION coverage: how many distinct kwargs per call site?")
print("#" * 78)
combos = collections.Counter()
per_call = []
for k, s in corpus.items():
    for m in re.finditer(r"apply_real_lens(?:_traced|_maslov|_gbd|_fga)?\s*\(", s):
        # crude: take up to the matching close paren by counting
        i = m.end() - 1
        depth = 0
        j = i
        while j < len(s):
            if s[j] == "(":
                depth += 1
            elif s[j] == ")":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        call = s[m.end():j]
        kws = sorted(set(re.findall(r"(?:^|,|\s)(\w+)\s*=", call)))
        kws = [x for x in kws if x not in ("E_in", "E", "dx", "wavelength", "prescription")]
        combos[len(kws)] += 1
        per_call.append((k, s[:m.start()].count("\n") + 1, tuple(kws)))
print("distribution of #non-trivial kwargs per apply_real_lens* call in tests:")
for n in sorted(combos):
    print(f"   {n:3d} kwargs : {combos[n]:5d} call sites")
print(f"total call sites: {sum(combos.values())}")
uniq = collections.Counter(c[2] for c in per_call)
print(f"distinct kwarg COMBINATIONS exercised: {len(uniq)}")
print("top 12 combinations:")
for c, n in uniq.most_common(12):
    print(f"   {n:5d}  {c}")
