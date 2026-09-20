"""V4/B1-B2 -- the 5.47.0 block's citations into the FOUR library files this
branch moved lines in, paired BASE-citation to HEAD-citation and judged by
where the cited CONTENT actually went.

    python v4_citations2.py <worktree> <out.json>

Method (the repo's own, from validation/probe_wp_b11c/reanchor_citations.py's
docstring): take the line each number named at the BASE commit, locate that
exact text in the HEAD blob, and that is where the number should now point.
Citations are paired base-to-head by ORDER OF APPEARANCE inside the 5.47.0
block, which is safe because the block's prose is unchanged apart from the
numbers themselves (asserted).
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
CITE = re.compile(r"`((?:[\w./_-]+/)?([\w_]+\.py)):(\d+)(?:-(\d+))?`")


#: Optional argv[3]/argv[4]: the two commits ALREADY EXTRACTED as trees, for a
#: build (WSL) whose git cannot resolve a Windows worktree's gitdir pointer.
#: Reading the extraction is the same bytes as reading the blob; the probe
#: PRINTS which route it took.
_BASE_TREE = sys.argv[3] if len(sys.argv) > 3 else None
_HEAD_TREE = sys.argv[4] if len(sys.argv) > 4 else None


def blob(commit, path):
    tree = {BASE: _BASE_TREE, HEAD: _HEAD_TREE}.get(commit)
    if tree:
        p = os.path.join(tree, path)
        if not os.path.exists(p):
            return None
        return open(p, encoding="utf-8", errors="replace").read().split("\n")
    r = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=_WT,
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    return None if r.returncode else r.stdout.split("\n")


def block_5470(lines):
    a = b = None
    for i, ln in enumerate(lines, 1):
        if ln.startswith("## [5.47.0]"):
            a = i
        elif a and ln.startswith("## [") and i > a:
            b = i - 1
            break
    return a, b or len(lines)


def cites(lines, a, b):
    out = []
    for i in range(a, b + 1):
        for m in CITE.finditer(lines[i - 1]):
            if m.group(2) in WATCHED:
                out.append({"doc_line": i, "cite": m.group(0),
                            "file": m.group(2), "n": int(m.group(3)),
                            "n2": int(m.group(4)) if m.group(4) else None,
                            "prose": lines[i - 1].strip()[:120]})
    return out


def locate(text, blob_lines, hint):
    """Where is this exact text in ``blob_lines``?  Unique match, else the one
    nearest ``hint``, else None."""
    hits = [k + 1 for k, t in enumerate(blob_lines) if t == text]
    if not hits:
        return None, 0
    if len(hits) == 1:
        return hits[0], 1
    return min(hits, key=lambda h: abs(h - hint)), len(hits)


def main():
    cl_b = blob(BASE, "CHANGELOG.md")
    cl_h = blob(HEAD, "CHANGELOG.md")
    ab, bb = block_5470(cl_b)
    ah, bh = block_5470(cl_h)
    cb, ch = cites(cl_b, ab, bb), cites(cl_h, ah, bh)
    rows = []
    paired = len(cb) == len(ch)
    for i, (rb, rh) in enumerate(zip(cb, ch)):
        src_b = blob(BASE, WATCHED[rb["file"]])
        src_h = blob(HEAD, WATCHED[rh["file"]])
        base_text = src_b[rb["n"] - 1] if rb["n"] <= len(src_b) else None
        head_text = src_h[rh["n"] - 1] if rh["n"] <= len(src_h) else None
        want, n_hits = (locate(base_text, src_h, rb["n"])
                        if base_text is not None else (None, 0))
        trivial = (base_text is None or not base_text.strip())
        if trivial:
            status = "TRIVIAL-BASE-LINE(not judgeable by text)"
        elif want is None:
            status = "BASE-TEXT-GONE-AT-HEAD"
        elif rh["n"] == want:
            status = ("RE-ANCHORED-CORRECTLY" if rh["n"] != rb["n"]
                      else "UNMOVED-AND-CORRECT")
        else:
            status = ("STALE-NOT-RE-ANCHORED" if rh["n"] == rb["n"]
                      else "RE-ANCHORED-TO-WRONG-LINE")
        rows.append({
            "idx": i, "file": rb["file"],
            "base_doc_line": rb["doc_line"], "head_doc_line": rh["doc_line"],
            "cite_base": rb["cite"], "cite_head": rh["cite"],
            "n_base": rb["n"], "n_head": rh["n"], "n_should_be": want,
            "n_matches_in_head_blob": n_hits,
            "base_text": (base_text or "").strip()[:120],
            "head_text_at_cited_line": (head_text or "").strip()[:120],
            "prose": rh["prose"], "status": status,
        })
    out = {"build": vlib.build_tag(), "base": BASE, "head": HEAD,
           "block_base_lines": [ab, bb], "block_head_lines": [ah, bh],
           "n_base_cites": len(cb), "n_head_cites": len(ch),
           "paired_one_to_one": paired, "rows": rows}
    vlib.write_json(out, _OUT)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    for r in rows:
        print(f"[{r['status']}] {r['file']} {r['cite_base']} -> "
              f"{r['cite_head']}  (should be :{r['n_should_be']}, "
              f"{r['n_matches_in_head_blob']} text match(es))")
        print(f"    CHANGELOG.md:{r['head_doc_line']} {r['prose']!r}")
        print(f"    base text       : {r['base_text']!r}")
        print(f"    head@cited line : {r['head_text_at_cited_line']!r}")


if __name__ == "__main__":
    main()
