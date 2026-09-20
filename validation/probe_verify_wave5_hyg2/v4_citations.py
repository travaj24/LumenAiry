"""V4/B -- every ``<file>.py:N`` citation in the repo's prose that names one
of the FOUR library files this branch moved lines in, checked against BOTH
commits' blobs.

    python v4_citations.py <worktree> <out.json>

A citation is judged by CONTENT, not by arithmetic: the line the number named
at the BASE commit is looked up, then the same text is located in the HEAD
blob.  Three outcomes per citation:

  STILL-EXACT   the base line's text sits at the SAME number at HEAD
  RE-ANCHORED   the base text moved and the number was updated to follow it
  STALE         the base text moved and the number was NOT updated

Run from the WORKTREE (``git show`` needs the repository).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_WT = os.path.abspath(sys.argv[1])
_OUT = os.path.abspath(sys.argv[2])
sys.path.insert(0, _WT)

import vlib  # noqa: E402

vlib.anchor(_WT)

BASE = "f4f18851"
HEAD = "b86fe31f"

WATCHED = {
    "mft.py": "lumenairy/propagators/mft.py",
    "carrier.py": "lumenairy/propagators/carrier.py",
    "_bluestein.py": "lumenairy/propagators/_bluestein.py",
    "lenses.py": "lumenairy/elements/lenses.py",
}

CITE = re.compile(r"`?((?:[\w./_-]+/)?([\w_]+\.py)):(\d+)(?:-(\d+))?`?")


def blob(commit, path):
    out = subprocess.run(["git", "show", f"{commit}:{path}"],
                         cwd=_WT, capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
    if out.returncode:
        return None
    return out.stdout.split("\n")


def main():
    blobs = {}
    for short, path in WATCHED.items():
        blobs[short] = (blob(BASE, path), blob(HEAD, path))

    docs = []
    for root, dirs, files in os.walk(_WT):
        dirs[:] = [d for d in dirs if d not in
                   (".git", "__pycache__", ".pytest_cache")]
        for f in files:
            if f.endswith(".md"):
                docs.append(os.path.join(root, f))

    rows = []
    for doc in docs:
        rel = os.path.relpath(doc, _WT).replace("\\", "/")
        try:
            lines = open(doc, encoding="utf-8", errors="replace").read()\
                .split("\n")
        except OSError:
            continue
        for i, line in enumerate(lines, 1):
            for m in CITE.finditer(line):
                full, short, n1, n2 = m.group(1), m.group(2), \
                    int(m.group(3)), m.group(4)
                if short not in WATCHED:
                    continue
                b, h = blobs[short]
                if b is None or h is None:
                    continue
                if n1 < 1 or n1 > len(b):
                    rows.append({"doc": rel, "doc_line": i, "cite": m.group(0),
                                 "file": short, "n": n1,
                                 "status": "OUT-OF-RANGE-AT-BASE"})
                    continue
                base_text = b[n1 - 1]
                head_text = h[n1 - 1] if n1 <= len(h) else "<past-EOF>"
                # where did base_text go at HEAD?
                cand = [k + 1 for k, t in enumerate(h) if t == base_text]
                moved_to = (cand[0] if len(cand) == 1 else
                            (n1 if n1 in cand else (cand if cand else None)))
                trivial = not base_text.strip() or base_text.strip() in (
                    ")", "]", "}", '"""', "'''", "else:", "return", "pass")
                if base_text == head_text:
                    status = "STILL-EXACT"
                elif isinstance(moved_to, int) and moved_to != n1:
                    status = "STALE-BASE-TEXT-MOVED"
                else:
                    status = "CHANGED-OR-AMBIGUOUS"
                rows.append({
                    "doc": rel, "doc_line": i, "cite": m.group(0),
                    "file": short, "n": n1, "n2": n2,
                    "base_text": base_text.strip()[:110],
                    "head_text": head_text.strip()[:110],
                    "base_text_now_at": moved_to,
                    "trivial_line": trivial,
                    "status": status,
                })
    bad = [r for r in rows if r["status"] != "STILL-EXACT"]
    out = {"build": vlib.build_tag(), "base": BASE, "head": HEAD,
           "n_citations": len(rows),
           "n_still_exact": sum(1 for r in rows
                                if r["status"] == "STILL-EXACT"),
           "n_suspect": len(bad), "suspect": bad, "all": rows}
    vlib.write_json(out, _OUT)
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("suspect", "all")}, indent=1))
    for r in bad:
        print(f"  [{r['status']}] {r['doc']}:{r['doc_line']} {r['cite']}\n"
              f"      base line {r['n']} = {r.get('base_text')!r}\n"
              f"      head line {r['n']} = {r.get('head_text')!r}\n"
              f"      base text now at HEAD line {r.get('base_text_now_at')}")


if __name__ == "__main__":
    main()
