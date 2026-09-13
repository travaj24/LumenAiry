"""refnow.py REF PATH N [N ...] -- content of line N at commit REF and where that content sits now."""
import subprocess, sys, pathlib
REPO = pathlib.Path("D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
ref, path = sys.argv[1], sys.argv[2]
old = subprocess.run(["git", "show", f"{ref}:{path}"], cwd=REPO, capture_output=True).stdout.decode("utf-8", "replace").splitlines()
cur = (REPO / path).read_text(encoding="utf-8", errors="replace").splitlines()
print(f"== {path}: ref {len(old)} lines, now {len(cur)}")
for n in map(int, sys.argv[3:]):
    s = old[n-1].strip() if n <= len(old) else "<out of range>"
    hits = [i+1 for i, l in enumerate(cur) if l.strip() == s and s]
    print(f"  {n:5d}: {s[:75]:75s} -> now {hits[:6]}")
