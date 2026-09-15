"""Re-anchor ``path.py:N`` citations in the given changelog files against an EXPLICIT reference
commit (the tree the citations were last known-correct for), instead of the changelog file's own
last commit as preflight_citations.py does -- needed when a changelog was committed together with
later edits to the files it cites, or is itself uncommitted after a fix.

usage: python reanchor_since.py <ref-commit> <changelog file> [...]  [--fix] [--only-path substr ...]
For each citation whose cited file matches one of --only-path substrings (all files if none):
read line N (and M for a range) at <ref-commit>; find that exact line in the current file;
OK if at the same number, DRIFT -> proposal (rewritten with --fix) if at exactly one other
number, MANUAL otherwise.
"""
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(r"C:/tmp/lum_reds")
FIX = REPO / "docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes"
CIT = re.compile(r"(?<![\w/.-])((?:[\w.-]+/)*[\w-]+\.py):(\d+)(?:-(\d+))?(?![\w.-]*/)")
ROOTS = ["lumenairy", "tests", "scripts", "validation"]

args = sys.argv[1:]
fix = "--fix" in args
args = [a for a in args if a != "--fix"]
only = []
if "--only-path" in args:
    i = args.index("--only-path"); only = args[i + 1:]; args = args[:i]
ref, files = args[0], args[1:]


def resolve(p):
    if "/" in p:
        return p if (REPO / p).exists() else None
    hits = [str(x.relative_to(REPO)).replace("\\", "/") for r in ROOTS for x in (REPO / r).rglob(p)]
    return hits[0] if len(hits) == 1 else None


def at_ref(path):
    r = subprocess.run(["git", "show", f"{ref}:{path}"], cwd=REPO, capture_output=True)
    return r.stdout.decode("utf-8", "replace").splitlines() if r.returncode == 0 else None


cache = {}
for name in files:
    p = pathlib.Path(name) if pathlib.Path(name).exists() else FIX / name
    t = p.read_text(encoding="utf-8")
    out = t
    for m in list(CIT.finditer(t)):
        raw, n1, n2 = m.group(1), int(m.group(2)), (int(m.group(3)) if m.group(3) else None)
        if only and not any(s in raw for s in only):
            continue
        path = resolve(raw)
        if path is None:
            print(f"AMBIGUOUS {name:32s} {m.group(0)}"); continue
        if path not in cache:
            cache[path] = (at_ref(path), (REPO / path).read_text(encoding="utf-8").splitlines())
        old, now = cache[path]
        if old is None:
            print(f"MANUAL    {name:32s} {m.group(0)}  (not at {ref})"); continue

        def locate(n):
            if n - 1 >= len(old):
                return None, "past EOF at ref"
            line = old[n - 1]
            if n - 1 < len(now) and now[n - 1] == line:
                return n, "OK"
            hits = [i + 1 for i, l in enumerate(now) if l == line]
            if len(hits) == 1:
                return hits[0], "DRIFT"
            return None, f"{len(hits)} matches"

        a, sa = locate(n1)
        b, sb = (locate(n2) if n2 else (None, "OK"))
        if sa == "OK" and sb == "OK":
            print(f"OK        {name:32s} {m.group(0)}")
        elif a is not None and (n2 is None or b is not None):
            new = f"{raw}:{a}" + (f"-{b}" if n2 else "")
            print(f"DRIFT     {name:32s} {m.group(0):55s} -> {new}")
            if fix:
                out = out.replace(m.group(0), new, 1)
        else:
            print(f"MANUAL    {name:32s} {m.group(0):55s} ({sa}; {sb})")
    if fix and out != t:
        p.write_text(out, encoding="utf-8", newline="")
        print("  rewrote", name)
