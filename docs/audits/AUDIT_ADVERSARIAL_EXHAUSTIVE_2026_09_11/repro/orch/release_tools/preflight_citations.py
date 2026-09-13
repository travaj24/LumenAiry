#!/usr/bin/env python3
"""preflight_citations.py -- source-line citation drift check for the WP changelog files.

For every ``path.py:N`` / ``path.py:N-M`` citation in fixes/WP-*_CHANGELOG.md and
fixes/VERIFY_WP-*_CHANGELOG.md, take the commit that LAST touched that changelog file as the
point where the citation was known-correct, read line N of the cited file at that commit, and
compare it with line N of the cited file now.

  OK      -- same line content at the same line number
  DRIFT   -- the old line content now sits at exactly one other line (proposal printed; --fix rewrites)
  MANUAL  -- old content not found / ambiguous / cited file missing at the reference commit
  NEW     -- the changelog file is not committed yet: only V18 triviality is checked

Bare basenames are resolved like V18 does (unique match under the repo, else AMBIGUOUS).
Usage: python preflight_citations.py [--fix] [--only WP-A6_CHANGELOG.md ...]
"""
import re, subprocess, sys, pathlib

REPO = pathlib.Path("D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
FIX = REPO / "docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes"
CIT = re.compile(r"(?<![\w/.-])((?:[\w.-]+/)*[\w-]+\.py):(\d+)(?:-(\d+))?(?![\w.-]*/)")
TRIVIAL = re.compile(r"^\s*(?:[\)\]\}]+,?|pass|continue|break|return|\.\.\.|)\s*$")
SEARCH_ROOTS = ["lumenairy", "tests", "scripts", "validation"]

def git(*args):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True).stdout.decode("utf-8", "replace")

def resolve(path):
    """Repo-relative path, else a unique suffix match (``propagators/rs.py``), else the candidate list."""
    if "/" in path and (REPO / path).exists():
        return path
    base = path.rsplit("/", 1)[-1]
    hits = []
    for root in SEARCH_ROOTS:
        for q in (REPO / root).rglob(base):
            rel = q.relative_to(REPO).as_posix()
            if rel == path or rel.endswith("/" + path):
                hits.append(rel)
    return hits[0] if len(hits) == 1 else (hits or None)

def lines_at(commit, path):
    out = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=REPO, capture_output=True)
    if out.returncode != 0:
        return None
    return out.stdout.decode("utf-8", "replace").splitlines()

def check_one(cur, old, n):
    """Return (status, proposal) for a single cited line number n (1-based)."""
    if old is None:
        return ("MANUAL", None)
    if n < 1 or n > len(old):
        return ("MANUAL", None)
    o = old[n - 1].strip()
    if n <= len(cur) and cur[n - 1].strip() == o:
        return ("OK", n)
    if not o or TRIVIAL.match(o):
        return ("MANUAL", None)
    cands = [i + 1 for i, l in enumerate(cur) if l.strip() == o]
    if len(cands) == 1:
        return ("DRIFT", cands[0])
    return ("MANUAL", None)

def main(argv):
    fix = "--fix" in argv
    only = [a for a in argv if a.endswith(".md")]
    files = sorted(FIX.glob("*_CHANGELOG.md"))
    if only:
        files = [f for f in files if f.name in only]
    counts = {"OK": 0, "DRIFT": 0, "MANUAL": 0, "NEW": 0, "AMBIGUOUS": 0}
    for f in files:
        raw = f.read_bytes()
        text = raw.decode("utf-8", "replace")
        commit = git("log", "-1", "--format=%H", "--", str(f.relative_to(REPO))).strip()
        repl = {}
        for m in CIT.finditer(text):
            path, a, b = m.group(1), int(m.group(2)), m.group(3)
            token = m.group(0)
            # Reference = the commit that last WROTE this token into this changelog file (so a
            # citation re-anchored later is compared against the tree it was re-anchored on).
            tok_commit = git("log", "-1", "-G" + re.escape(token) + "([^0-9-]|$)", "--format=%H", "--",
                             str(f.relative_to(REPO))).strip() or commit
            res = resolve(path)
            if res is None or isinstance(res, list):
                print(f"AMBIGUOUS {f.name:32s} {token:60s} -> {res}")
                for cand in (res or []):
                    old_c = lines_at(tok_commit, cand) if tok_commit else None
                    line = old_c[a - 1].strip()[:70] if old_c and a <= len(old_c) else "<n/a>"
                    cur_c = (REPO / cand).read_text(encoding="utf-8", errors="replace").splitlines()
                    st, prop = check_one(cur_c, old_c, a)
                    print(f"          candidate {cand}:{a} @ref({tok_commit[:8]}) = {line} | now {st} -> {prop}")
                counts["AMBIGUOUS"] += 1
                continue
            cur = (REPO / res).read_text(encoding="utf-8", errors="replace").splitlines()
            if not commit:
                for n in [a] + ([int(b)] if b else []):
                    triv = n > len(cur) or TRIVIAL.match(cur[n - 1])
                    print(f"NEW       {f.name:32s} {token:60s} {'TRIVIAL' if triv else 'nontrivial'}: {cur[n-1].strip()[:70] if n <= len(cur) else '<out of range>'}")
                counts["NEW"] += 1
                continue
            old = lines_at(tok_commit, res)
            sa, pa = check_one(cur, old, a)
            if b:
                sb, pb = check_one(cur, old, int(b))
            else:
                sb, pb = "OK", None
            status = "OK" if sa == sb == "OK" else ("DRIFT" if {sa, sb} <= {"OK", "DRIFT"} else "MANUAL")
            counts[status] += 1
            if status != "OK":
                oldtxt = old[a - 1].strip()[:60] if old and a <= len(old) else "<n/a>"
                newtok = f"{path}:{pa}" + (f"-{pb}" if b else "")
                print(f"{status:9s} {f.name:32s} {token:60s} -> {newtok if status == 'DRIFT' else '?':40s} | was: {oldtxt}")
                if status == "DRIFT":
                    repl[token] = newtok
        if fix and repl:
            for tok, new in repl.items():
                text = re.sub(r"(?<![\w/.-])" + re.escape(tok) + r"(?![\w.-]*/|\d)", new, text)
            f.write_bytes(text.encode("utf-8"))
            print(f"FIXED     {f.name}: {len(repl)} citation(s) re-anchored")
    print("summary:", counts)
    return 0 if counts["DRIFT"] == counts["MANUAL"] == counts["AMBIGUOUS"] == 0 else 1

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
