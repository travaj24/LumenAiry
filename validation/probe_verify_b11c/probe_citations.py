"""VERIFY-B11c: were the CHANGELOG's source-line citations re-anchored to the
SAME LINES they named before the move?

``scripts/check_source_line_citations.py`` (V18) answers a weaker question: it
refuses a citation that lands on a trivial line (blank, a lone brace, a
docstring delimiter).  A citation that shifted by one and landed on a different
NON-trivial line passes V18 and is wrong -- the "right conclusion, wrong
numbers" shape.  This reads, for every ``path:line`` citation in the CHANGELOG,

* the text of that line in the file as the BASE commit had it, using the base
  commit's CHANGELOG number, and
* the text of the line the CURRENT CHANGELOG names in the CURRENT file,

and reports every citation whose two texts differ.  Citations added or removed
between the two CHANGELOGs are listed separately rather than silently dropped.

argv: --repo <worktree> --base <sha>
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess

#: ``CHANGELOG.md`` cites a source line as `path/to/file.py:1234` in SINGLE
#: backticks (a range `...:12-34` cites its first line).
CITE = re.compile(r"`([\w./\\-]+\.py):(\d+)(?:-\d+)?`")


def _show(repo, ref, path):
    """Bytes, decoded as cp1252 with replacement: the repository's convention
    is cp1252 and a stray byte must not abort the walk."""
    out = subprocess.run(["git", "-C", repo, "show", f"{ref}:{path}"],
                         capture_output=True)
    if out.returncode != 0:
        return None
    return out.stdout.decode("utf-8", "replace").splitlines()


def _resolve(repo, ref, cited):
    """A citation may name a bare basename; find the file it means."""
    out = subprocess.run(["git", "-C", repo, "ls-tree", "-r", "--name-only",
                          ref], capture_output=True)
    files = out.stdout.decode("utf-8", "replace").splitlines()
    cited = cited.replace("\\", "/")
    if cited in files:
        return cited
    # A suffix match on the FULL cited path wins outright ("rcwa/_core.py"
    # names exactly one file even though "_core.py" alone names several); only
    # if that finds nothing does a bare-basename match get a vote, and then
    # only when it is unambiguous.
    suffix = [f for f in files
              if f.startswith("lumenairy/") and f.endswith("/" + cited)]
    if len(suffix) == 1:
        return suffix[0]
    tail = cited.rsplit("/", 1)[-1]
    hits = [f for f in files
            if f.startswith("lumenairy/") and f.rsplit("/", 1)[-1] == tail]
    return hits[0] if len(hits) == 1 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    base_cl = _show(a.repo, a.base, "CHANGELOG.md")
    head_cl = _show(a.repo, "HEAD", "CHANGELOG.md")

    # Matching by the CHANGELOG LINE NUMBER a citation sits on is fragile (a
    # block can move), so match by the sentence around it: key on the stripped
    # CHANGELOG line with the citations blanked out, which the re-anchor did
    # not rewrite.
    def _by_text(lines):
        out = {}
        for ln in lines:
            for m in CITE.finditer(ln):
                key = CITE.sub("`@@`", ln).strip()
                out.setdefault(key, []).append((m.group(1), int(m.group(2))))
        return out

    base_t, head_t = _by_text(base_cl), _by_text(head_cl)

    checked, moved_ok, moved_bad, unmatched = 0, [], [], []
    for key, head_items in head_t.items():
        base_items = base_t.get(key)
        if base_items is None or len(base_items) != len(head_items):
            continue
        for (bp, bl), (hp, hl) in zip(base_items, head_items):
            bfile = _resolve(a.repo, a.base, bp)
            hfile = _resolve(a.repo, "HEAD", hp)
            if not bfile or not hfile:
                unmatched.append((bp, bl, hp, hl))
                continue
            bsrc, hsrc = _show(a.repo, a.base, bfile), _show(a.repo, "HEAD",
                                                            hfile)
            if bsrc is None or hsrc is None or bl > len(bsrc) \
                    or hl > len(hsrc):
                unmatched.append((bp, bl, hp, hl))
                continue
            checked += 1
            btxt, htxt = bsrc[bl - 1].rstrip(), hsrc[hl - 1].rstrip()
            rec = {"base": f"{bfile}:{bl}", "head": f"{hfile}:{hl}",
                   "text_base": btxt[:110], "text_head": htxt[:110]}
            if (bp, bl) == (hp, hl):
                continue                      # unchanged citation
            (moved_ok if btxt == htxt else moved_bad).append(rec)

    doc = {"citations_checked": checked,
           "re_anchored_and_content_matches": len(moved_ok),
           "re_anchored_but_content_differs": moved_bad,
           "unresolvable": unmatched,
           "detail_ok": moved_ok}
    text = json.dumps(doc, indent=1)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(text)
    print(f"checked {checked}; re-anchored {len(moved_ok)} content-matching, "
          f"{len(moved_bad)} content-DIFFERING, {len(unmatched)} unresolvable")
    for r in moved_bad:
        print("  MISMATCH", r["base"], "->", r["head"])
        print("    was :", r["text_base"])
        print("    now :", r["text_head"])


if __name__ == "__main__":
    main()
