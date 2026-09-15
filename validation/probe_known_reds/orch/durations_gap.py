"""Which collected test ids lack a .test_durations entry, split into tracked vs untracked test files."""
import json, subprocess, sys, collections, pathlib
REPO = pathlib.Path(r"C:/tmp/lum_reds")
out = subprocess.run([sys.executable, "-m", "pytest", "tests/unit", "--collect-only", "-q", "-p", "no:cacheprovider"],
                     cwd=REPO, capture_output=True, text=True)
ids = [l.strip() for l in out.stdout.splitlines() if "::" in l]
dur = json.loads((REPO / ".test_durations").read_text(encoding="utf-8"))
untracked = set(subprocess.run(["git", "ls-files", "--others", "--exclude-standard", "tests/unit"], cwd=REPO,
                               capture_output=True, text=True).stdout.split())
missing = [i for i in ids if i not in dur]
by_file = collections.Counter(i.split("::")[0] for i in missing)
tracked_missing = [i for i in missing if i.split("::")[0].replace("\\", "/") not in untracked]
print(f"collected {len(ids)}; durations {len(dur)}; missing {len(missing)} ({100*len(missing)/max(1,len(ids)):.2f} %)")
print(f"missing in TRACKED files: {len(tracked_missing)}  ({100*(len(ids)-len(missing))/max(1,len(ids)):.2f} % coverage overall)")
for f, n in by_file.most_common(12):
    tag = "untracked" if f.replace("\\", "/") in untracked else "TRACKED"
    print(f"  {n:5d}  {tag:9s}  {f}")
stale = [k for k in dur if k not in set(ids)]
print(f"stale durations entries (no longer collected): {len(stale)}")
for k in stale[:5]: print("   ", k[:140])
