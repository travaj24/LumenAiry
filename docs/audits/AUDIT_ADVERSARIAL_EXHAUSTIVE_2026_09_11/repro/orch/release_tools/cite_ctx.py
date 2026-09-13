"""cite_ctx.py CHANGELOG_NAME TOKEN [RESOLVED_PATH] -- show reference content of a cited range and where it sits now."""
import subprocess, sys, pathlib, re
REPO = pathlib.Path("D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
FIX = "docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/"
def git(*a):
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True).stdout.decode("utf-8", "replace")
name, token = sys.argv[1], sys.argv[2]
path = sys.argv[3] if len(sys.argv) > 3 else token.split(":")[0]
m = re.match(r"(.*):(\d+)(?:-(\d+))?$", token)
a, b = int(m.group(2)), int(m.group(3)) if m.group(3) else None
ref = git("log", "-1", f"-S{token}", "--format=%H", "--", FIX + name).strip()
old = git("show", f"{ref}:{path}").splitlines()
cur = (REPO / path).read_text(encoding="utf-8", errors="replace").splitlines()
print(f"=== {name} {token} -> {path}  ref {ref[:8]}  (ref {len(old)} lines, now {len(cur)} lines)")
for n in ([a] + ([b] if b else [])):
    print(f"  ref line {n}: {old[n-1].strip()[:90] if n <= len(old) else '<out of range>'}")
    if n <= len(old):
        key = old[n-1].strip()
        hits = [i+1 for i, l in enumerate(cur) if l.strip() == key] if key else []
        print(f"    now at: {hits[:8]}")
        # context anchors: nearest preceding def/class in the ref file, and where it is now
        for j in range(n-1, -1, -1):
            s = old[j].strip()
            if s.startswith(("def ", "class ", "async def ")):
                dh = [i+1 for i, l in enumerate(cur) if l.strip() == s]
                print(f"    enclosing @ref {j+1}: {s[:70]}  -> now {dh[:4]}  (offset in ref {n-(j+1)})")
                break
