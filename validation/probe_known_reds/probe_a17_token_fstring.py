#!/usr/bin/env python3
"""Probe: why `token_fingerprint` disagrees between CPython 3.11 and 3.12+.

Item D / WP-A17.  CI run 34914295323 turned every py3.11 shard red on
`test_the_module_token_stream_is_unchanged_since_the_history_move[...]` while
py3.12, py3.13 and py3.14 were green on the same ids.

The hypothesis under test: PEP 701 (CPython 3.12) re-tokenised f-strings.  On
3.11 an f-string is ONE `STRING` token carrying its whole source text; on 3.12+
it is a run of `FSTRING_START` / `FSTRING_MIDDLE` / `FSTRING_END` tokens with
the replacement-field expressions tokenised as ordinary NAME/OP/NUMBER in
between.  The a17 digest feeds token *names* and *strings* into a SHA-256, so a
module holding one f-string digests differently on the two tokenizers.

What this probe measures, per registered history document:

* `v1`  -- the digest scheme as it stands (f-string trio fed in piecewise);
* `v2`  -- the candidate scheme: the whole `FSTRING_START..FSTRING_END` run is
           replaced by ONE synthetic `STRING` record whose payload is the exact
           source slice of the f-string.  That is byte-for-byte what a 3.11
           tokenizer emits, so `v2` is tokenizer-version-independent;
* whether the module contains an f-string at all (`ast.JoinedStr`);
* the digest CI actually computed ON 3.11 (scraped from the failure blocks of
  the five 3.11 shard logs, `ci311_live_token_digests.json`).

The decisive comparison is `v2` against that scraped 3.11 reading: if they are
equal, the collapse reproduces the real 3.11 token stream exactly, which both
proves the mechanism and proves the candidate fix agrees across the version
boundary -- without needing a 3.11 interpreter on this box.

Usage:  python validation/probe_known_reds/probe_a17_token_fstring.py [--out FILE]
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import pathlib
import platform
import re
import sys
import tokenize

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
HISTORY_DIR = REPO_ROOT / "docs" / "history"
_HEADER_RE = re.compile(r"<!--\s*lumenairy-history-doc\s*(?P<body>.*?)-->", re.S)

_MEANING_V1 = frozenset({
    "NAME", "OP", "NUMBER", "STRING",
    "FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END",
})
_MEANING_V2 = frozenset({"NAME", "OP", "NUMBER", "STRING"})
_FSTRING_OPEN = frozenset({"FSTRING_START", "TSTRING_START"})
_FSTRING_CLOSE = frozenset({"FSTRING_END", "TSTRING_END"})


def _string_statement_lines(tree):
    spans = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)):
            spans.append((node.lineno, node.end_lineno or node.lineno))
    return spans


def _digest(parts):
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()


def token_parts_v1(source):
    """The scheme as shipped in 5.47.0: every meaning token fed in as-is."""
    spans = _string_statement_lines(ast.parse(source))
    parts = []
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        name = tokenize.tok_name[tok.type]
        if name not in _MEANING_V1:
            continue
        if name.startswith(("STRING", "FSTRING")):
            if any(lo <= tok.start[0] <= hi for lo, hi in spans):
                continue
        parts.append(name + "\x1f" + tok.string)
    return parts


def _slice(lines, start, end):
    (r1, c1), (r2, c2) = start, end
    if r1 == r2:
        return lines[r1 - 1][c1:c2]
    out = [lines[r1 - 1][c1:]]
    out.extend(lines[r1:r2 - 1])
    out.append(lines[r2 - 1][:c2])
    return "".join(out)


def token_parts_v2(source):
    """The candidate: an f-string run collapses to one STRING record holding
    the f-string's exact source text -- the pre-3.12 spelling."""
    spans = _string_statement_lines(ast.parse(source))
    lines = source.splitlines(keepends=True)
    parts = []
    depth = 0
    opened = None
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        name = tokenize.tok_name[tok.type]
        if name in _FSTRING_OPEN:
            if depth == 0:
                opened = tok.start
            depth += 1
            continue
        if depth:
            if name in _FSTRING_CLOSE:
                depth -= 1
                if depth == 0:
                    if not any(lo <= opened[0] <= hi for lo, hi in spans):
                        parts.append(
                            "STRING\x1f" + _slice(lines, opened, tok.end))
                    opened = None
            continue
        if name not in _MEANING_V2:
            continue
        if name == "STRING" and any(lo <= tok.start[0] <= hi for lo, hi in spans):
            continue
        parts.append(name + "\x1f" + tok.string)
    return parts


def parse_header(md):
    match = _HEADER_RE.search(md.read_text(encoding="utf-8"))
    header = {}
    for line in match.group("body").splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        header[key.strip()] = value.strip()
    return header


def token_type_census(snippet):
    names = []
    for tok in tokenize.generate_tokens(io.StringIO(snippet).readline):
        name = tokenize.tok_name[tok.type]
        if name in ("ENCODING", "NEWLINE", "NL", "ENDMARKER", "INDENT", "DEDENT"):
            continue
        names.append(name + "=" + repr(tok.string))
    return names


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    ci_path = pathlib.Path(__file__).with_name("ci311_live_token_digests.json")
    ci311 = json.loads(ci_path.read_text()) if ci_path.is_file() else {}

    rows = []
    for md in sorted(HISTORY_DIR.glob("*.md")):
        if md.name.upper() == "README.MD":
            continue
        header = parse_header(md)
        src_path = REPO_ROOT / header["module"]
        source = src_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        n_joined = sum(1 for n in ast.walk(tree) if isinstance(n, ast.JoinedStr))
        v1 = _digest(token_parts_v1(source))
        v2 = _digest(token_parts_v2(source))
        rows.append({
            "doc": md.stem,
            "module": header["module"],
            "recorded_token": header.get("token_sha256", ""),
            "v1": v1,
            "v2": v2,
            "n_joinedstr": n_joined,
            "v1_matches_recorded": v1 == header.get("token_sha256"),
            "v2_matches_recorded": v2 == header.get("token_sha256"),
            "ci311_live": ci311.get(md.stem),
            "v2_matches_ci311": (ci311.get(md.stem) == v2
                                 if md.stem in ci311 else None),
        })

    n = len(rows)
    with_f = [r for r in rows if r["n_joinedstr"]]
    v1_drift = [r for r in rows if not r["v1_matches_recorded"]]
    v2_moves = [r for r in rows if r["v1"] != r["v2"]]
    checked = [r for r in rows if r["ci311_live"]]
    agree = [r for r in checked if r["v2_matches_ci311"]]

    summary = {
        "interpreter": sys.version,
        "implementation": platform.python_implementation(),
        "documents": n,
        "documents_with_fstring": len(with_f),
        "documents_without_fstring": n - len(with_f),
        "v1_drift_vs_recorded_on_this_interpreter": [r["doc"] for r in v1_drift],
        "documents_where_v1_differs_from_v2": len(v2_moves),
        "documents_where_v1_equals_v2": n - len(v2_moves),
        "fstring_but_v1_equals_v2": [
            r["doc"] for r in with_f if r["v1"] == r["v2"]],
        "no_fstring_but_v1_differs_from_v2": [
            r["doc"] for r in v2_moves if not r["n_joinedstr"]],
        "ci311_digests_available": len(checked),
        "ci311_reproduced_by_v2": len(agree),
        "ci311_not_reproduced_by_v2": [
            r["doc"] for r in checked if not r["v2_matches_ci311"]],
        "fstring_token_census": {
            "f-string": token_type_census('x = f"a{b}c"\n'),
            "plain string": token_type_census('x = "a"\n'),
        },
        "rows": rows,
    }

    text = json.dumps(summary, indent=1)
    if args.out:
        pathlib.Path(args.out).write_text(text, encoding="utf-8")
    for key, value in summary.items():
        if key == "rows":
            continue
        print(key + ": " + repr(value))
    return 0


if __name__ == "__main__":
    sys.exit(main())
