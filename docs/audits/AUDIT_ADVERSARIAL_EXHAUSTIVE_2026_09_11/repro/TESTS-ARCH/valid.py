"""validation/ oracle independence + conftest fixture audit + docs census."""
import ast, os, re, collections

ROOT = r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy"


def rel(p):
    return os.path.relpath(p, ROOT).replace("\\", "/")


# ---------- validation/oracles ----------
ordir = os.path.join(ROOT, "validation", "oracles")
print("===== validation/oracles/ =====")
if os.path.isdir(ordir):
    for f in sorted(os.listdir(ordir)):
        p = os.path.join(ordir, f)
        if not f.endswith(".py"):
            print(f"   [non-py] {f}")
            continue
        s = open(p, encoding="utf-8", errors="replace").read()
        imports = sorted(set(re.findall(r"^\s*(?:from|import)\s+([\w.]+)", s, re.M)))
        lum = [i for i in imports if i.startswith("lumenairy")]
        print(f"\n   {f}  ({s.count(chr(10))+1} lines)")
        print(f"      imports: {', '.join(imports)[:200]}")
        print(f"      -> IMPORTS LUMENAIRY: {lum if lum else 'NO (independent)'}")

# ---------- run_all / run_validation ----------
print("\n\n===== validation runners =====")
for cand in ("run_all.py", "run_validation.py", "_harness.py", "conftest.py", "README.md"):
    p = os.path.join(ROOT, "validation", cand)
    print(f"   {cand}: {'EXISTS' if os.path.exists(p) else 'MISSING'}"
          + (f"  ({os.path.getsize(p)} bytes)" if os.path.exists(p) else ""))

# ---------- validation/real_lens_opd + repro dirs ----------
for d in ("real_lens_opd", "repro_traced_carrier_121", "repro_traced_carrier_122"):
    p = os.path.join(ROOT, "validation", d)
    print(f"   validation/{d}: {'EXISTS' if os.path.isdir(p) else 'MISSING'}"
          + (f"  ({len(os.listdir(p))} entries)" if os.path.isdir(p) else ""))

# what are the 2225 "other" files in validation/?
ext = collections.Counter()
tot_bytes = collections.Counter()
for dp, dn, fn in os.walk(os.path.join(ROOT, "validation")):
    dn[:] = [x for x in dn if x != "__pycache__"]
    for f in fn:
        e = os.path.splitext(f)[1].lower() or "(none)"
        ext[e] += 1
        try:
            tot_bytes[e] += os.path.getsize(os.path.join(dp, f))
        except OSError:
            pass
print("\n   validation/ file extensions:")
for e, n in ext.most_common(20):
    print(f"      {e:10s} {n:5d}   {tot_bytes[e]/1e6:9.2f} MB")

# ---------- conftest fixture audit ----------
print("\n\n===== conftest.py files =====")
for dp, dn, fn in os.walk(ROOT):
    dn[:] = [d for d in dn if d not in ("__pycache__", ".git", ".mypy_cache",
                                        ".pytest_cache", ".ruff_cache")]
    for f in fn:
        if f == "conftest.py":
            p = os.path.join(dp, f)
            s = open(p, encoding="utf-8", errors="replace").read()
            fx = re.findall(r"@pytest\.fixture[^\n]*\ndef\s+(\w+)", s)
            auto = re.findall(r"autouse\s*=\s*True", s)
            print(f"   {rel(p)}  ({s.count(chr(10))+1} lines)  fixtures={fx}  "
                  f"autouse={len(auto)}")

# ---------- do tests restore global config? ----------
GLOBALS = ["set_lens_sag_dtype", "set_lens_parallel_amp", "set_default_wave_propagator",
           "set_default_real_dtype", "set_default_complex_dtype", "set_default_dy",
           "set_fft_threads", "set_fft_plan_cache_size", "set_asm_cache_size",
           "set_fft_auto_promote", "set_fft_double_buffer", "set_fft_fallback",
           "set_pyfftw_planner", "set_max_ram", "set_low_memory", "set_cache_budget",
           "set_storage_backend", "set_blas_threads",
           "set_pointwise_cos_grid_cache_budget", "set_fft_plan_max_bytes_per_buffer"]
tests = {}
for dp, dn, fn in os.walk(os.path.join(ROOT, "tests")):
    dn[:] = [d for d in dn if d != "__pycache__"]
    for f in fn:
        if f.endswith(".py"):
            p = os.path.join(dp, f)
            tests[rel(p)] = open(p, encoding="utf-8", errors="replace").read()

print("\n\n===== global-knob use in tests, and whether it is restored =====")
print(f"{'knob':40s} {'files':>6s} {'calls':>6s} {'try/finally':>12s} {'fixture':>8s}")
for g in GLOBALS:
    files = [k for k, v in tests.items() if g + "(" in v]
    calls = sum(v.count(g + "(") for v in tests.values())
    # a crude restore check: the file also contains 'finally' or a yield-fixture
    tf = sum(1 for k in files if "finally" in tests[k])
    fx = sum(1 for k in files if re.search(r"@pytest\.fixture", tests[k]))
    if calls:
        print(f"{g:40s} {len(files):6d} {calls:6d} {tf:12d} {fx:8d}")

# ---------- docs census ----------
print("\n\n===== docs/ census =====")
docs = []
for dp, dn, fn in os.walk(os.path.join(ROOT, "docs")):
    for f in fn:
        p = os.path.join(dp, f)
        docs.append((rel(p), os.path.getsize(p), os.path.getmtime(p)))
print(f"   total docs files: {len(docs)}, {sum(d[1] for d in docs)/1e6:.2f} MB")
aud = [d for d in docs if "AUDIT" in d[0].upper()]
print(f"   files with AUDIT in the name: {len(aud)}")
fix = [d for d in docs if re.search(r"/(FIX|VERIFY|GAP|PROBE)", d[0].upper())]
print(f"   FIX*/VERIFY*/GAP*/PROBE* named: {len(fix)}")
import datetime
by_year = collections.Counter(datetime.date.fromtimestamp(d[2]).strftime("%Y-%m")
                              for d in docs)
print("   docs by mtime month:")
for k, n in sorted(by_year.items()):
    print(f"      {k}  {n:4d}")
print("\n   20 largest docs:")
for r, b, m in sorted(docs, key=lambda d: -d[1])[:20]:
    print(f"      {b/1000:8.1f} KB  {r}")
subs = collections.Counter(os.path.dirname(d[0]) for d in docs)
print("\n   docs subdirectories:")
for k, n in subs.most_common():
    print(f"      {n:5d}  {k}/")
