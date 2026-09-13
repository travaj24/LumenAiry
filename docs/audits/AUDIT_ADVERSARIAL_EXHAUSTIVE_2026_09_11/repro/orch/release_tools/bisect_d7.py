"""Read-only bisect of test_niche_d7_decentred_fit's exit-slope measurement over the remediation
commits.  Each probe imports the CURRENT test module by path inside a child whose cwd and PYTHONPATH
are the archived library (no pytest, so no rootdir sys.path insertion can shadow the archive) and
prints (on_axis, before, after) in urad for frac = 0.5 and 1.0, plus PASS/FAIL of the test's bars."""
import subprocess, sys, pathlib, os

REPO = pathlib.Path("D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
S = pathlib.Path("C:/Users/AndrewT/AppData/Local/Temp/claude/D--Metacept-Neurophos-Python-Test-Scripts-Metasurface-QWP/78f7e8ef-9607-4e41-ba1d-ebd1c2e74c7e/scratchpad/bisect_trees")
S.mkdir(exist_ok=True)
TEST = REPO / "tests/unit/test_niche_d7_decentred_fit.py"
CHILD = r"""
import sys, importlib.util, warnings
warnings.simplefilter("ignore")
import lumenairy
assert sys.argv[2] in lumenairy.__file__, ("wrong library imported", lumenairy.__file__)
spec = importlib.util.spec_from_file_location("d7", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
f_b = float(m._bfd_by_inline_raytrace([0.3e-3]).mean())
on = m._exit_slope_rms(m._apply(0.0, rs=1), 0.0, f_b)
out = [f"on_axis={on*1e6:.3f}"]
ok = on < 1e-4
for frac in (0.5, 1.0):
    cx = frac * m._W
    before = m._exit_slope_rms(m._apply(cx, pre_d7=True, rs=1), cx, f_b)
    after = m._exit_slope_rms(m._apply(cx, rs=1), cx, f_b)
    ok = ok and before > 2 * on and after < 0.25 * on and after < 0.1 * before
    out.append(f"frac{frac}: before={before*1e6:.3f} after={after*1e6:.3f}")
print(("PASS " if ok else "FAIL ") + " | ".join(out))
"""
revs = subprocess.run(["git", "rev-list", "--reverse", "a1ff1e6e..HEAD", "--", "lumenairy"], cwd=REPO, capture_output=True, text=True).stdout.split()
revs = ["a1ff1e6e"] + revs
subj = {r: subprocess.run(["git", "log", "-1", "--format=%s", r], cwd=REPO, capture_output=True, text=True).stdout.strip()[:70] for r in revs}
cache = {}

def probe(rev):
    if rev in cache:
        return cache[rev]
    d = S / rev[:8]
    if not (d / "lumenairy").exists():
        d.mkdir(exist_ok=True)
        tar = subprocess.run(["git", "archive", rev, "lumenairy"], cwd=REPO, capture_output=True).stdout
        subprocess.run(["tar", "-x", "-C", str(d)], input=tar, check=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONPATH=str(d))
    r = subprocess.run([sys.executable, "-c", CHILD, str(TEST), str(d)], cwd=d, env=env, capture_output=True, text=True, timeout=1800)
    line = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "ERR " + (r.stderr.strip().splitlines()[-1][:150] if r.stderr.strip() else "?")
    ok = line.startswith("PASS")
    cache[rev] = ok
    print(f"{rev[:8]}  {line:120s}  {subj[rev]}", flush=True)
    return ok

print(f"{len(revs)} trees (base + {len(revs)-1} commits touching lumenairy/)", flush=True)
lo, hi = 0, len(revs) - 1
p_lo, p_hi = probe(revs[lo]), probe(revs[hi])
if p_lo == p_hi:
    print("no crossing")
else:
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if probe(revs[mid]) == p_lo:
            lo = mid
        else:
            hi = mid
    print(f"\ncrossing: {revs[lo][:8]} -> {revs[hi][:8]}: {subj[revs[hi]]}")
    for k in range(max(0, lo - 1), min(len(revs), hi + 2)):
        probe(revs[k])
