"""Read-only bisect of test_niche_p2_design_battery's doublet through-focus FWHM ratio over the
remediation commits: every commit touching lumenairy/ is extracted with `git archive` into the
scratchpad and the CURRENT test module's _through_focus(_d_doublet, 2e-3, 2.5) is run against it.
Prints the ratio per commit (binary search first, then the crossing's neighbours)."""
import subprocess, sys, pathlib, shutil, os

REPO = pathlib.Path("D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
S = pathlib.Path("C:/Users/AndrewT/AppData/Local/Temp/claude/D--Metacept-Neurophos-Python-Test-Scripts-Metasurface-QWP/78f7e8ef-9607-4e41-ba1d-ebd1c2e74c7e/scratchpad/bisect_p2")
S.mkdir(exist_ok=True)
CHILD = r"""
import sys, importlib.util, warnings
warnings.simplefilter("ignore")
spec = importlib.util.spec_from_file_location("bat", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
fwhm, fwhm_th, ee, dz = m._through_focus(m._d_doublet, 2.0e-3, 2.5)
print(f"{fwhm:.4e} {fwhm_th:.4e} {fwhm/fwhm_th:.4f} {ee[1]:.4f} {ee[2]:.4f} {dz:.3e}")
"""
revs = subprocess.run(["git", "rev-list", "--reverse", "a1ff1e6e..HEAD", "--", "lumenairy"], cwd=REPO,
                      capture_output=True, text=True).stdout.split()
subjects = {r: subprocess.run(["git", "log", "-1", "--format=%s", r], cwd=REPO, capture_output=True, text=True).stdout.strip()[:70] for r in revs}
test_file = REPO / "tests/unit/test_niche_p2_design_battery.py"
cache = {}

def measure(rev):
    if rev in cache:
        return cache[rev]
    d = S / rev[:8]
    if not (d / "lumenairy").exists():
        d.mkdir(exist_ok=True)
        tar = subprocess.run(["git", "archive", rev, "lumenairy"], cwd=REPO, capture_output=True).stdout
        subprocess.run(["tar", "-x", "-C", str(d)], input=tar, check=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONPATH=str(d))
    r = subprocess.run([sys.executable, "-c", CHILD, str(test_file)], cwd=d, env=env, capture_output=True, text=True, timeout=900)
    out = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "ERR " + r.stderr.strip().splitlines()[-1][:120]
    cache[rev] = out
    print(f"{rev[:8]}  {out:60s}  {subjects[rev]}", flush=True)
    return out

def ratio(rev):
    out = measure(rev)
    return None if out.startswith("ERR") else float(out.split()[2])

lo, hi = 0, len(revs) - 1
print(f"{len(revs)} commits touching lumenairy/; HEAD={revs[-1][:8]}", flush=True)
r_hi = ratio(revs[hi]); r_lo = ratio(revs[lo])
# binary search for the first commit whose ratio >= 1.10 (unscorable commits count as 'no change')
while hi - lo > 1:
    mid = (lo + hi) // 2
    rm = ratio(revs[mid])
    if rm is None or rm < 1.10:
        lo = mid
    else:
        hi = mid
print(f"\ncrossing between {revs[lo][:8]} ({ratio(revs[lo])}) and {revs[hi][:8]} ({ratio(revs[hi])})")
# neighbours, to see whether the move happened in more than one step
for k in range(max(0, lo - 2), min(len(revs), hi + 3)):
    measure(revs[k])
