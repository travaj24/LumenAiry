"""WP-A17 -- a RATCHET on version-history narrative in the source.

Audit 2026-09-11, finding P2-4 (``TESTS-ARCH.md``) / consolidated report
sec. 14 V6 and sec. 15.7.  Four relocation sweeps moved the library's
"vX.Y (audit Z): pre-fix this did A, which was wrong because B, now it does C"
blocks into ``docs/history/``.  Nothing stops them coming back.

That is the gap this file closes.  The relocation checker
(``test_audit2609_a17_history_relocation.py``) proves each MOVE was
behaviour-free; it says nothing about the next comment somebody writes.  A
backlog that took four sweeps to clear will re-accumulate one release note at
a time unless a gate notices, and a gate that demanded ZERO would be wrong:
a handful of live migration statements, file-format contracts and English
words legitimately match the patterns (see FALSE POSITIVES below).

So this is a RATCHET, not a bar.  Per module:

* the count may SHRINK (that is the point) -- the test passes and prints how
  far ahead of its baseline the tree now is, plus the re-baseline command;
* the count may STAY;
* the count may not GROW.  A module over its baseline fails, and the failure
  lists the offending lines so the author can see what tripped it.

A module with no baseline entry -- a NEW module -- has a baseline of 0, so
narrative cannot enter the library through a new file either.

RE-BASELINING, which is a normal and expected operation::

    python tests/unit/test_audit2609_a17_history_lint.py --write
    LUMENAIRY_HISTORY_LINT_WRITE=1 python -m pytest \\
        tests/unit/test_audit2609_a17_history_lint.py

Both rewrite ``test_audit2609_a17_history_lint_baseline.json`` beside this
file from the live tree.  Do it when a count legitimately grows -- a new
``.. versionchanged::`` migration note, a new file-format compatibility
statement -- and say which line and why in the commit message.  Re-baselining
to silence a genuine narrative block is the failure mode; the diff is one line
per module, so a reviewer can see exactly what was conceded.

WHAT IS COUNTED.  Lines that are COMMENTS or STRING STATEMENTS (docstrings and
the "string as a comment" form) and match one of the finding's own shapes:

    \\bv\\d+\\.\\d+(\\.\\d+)? \\(    a release tag with an audit id
    \\bpre-?fix\\b                the "pre-fix this did X" framing
    \\bpre-v?\\d                   "pre-4.10" / "pre-v5.29.1"
    used to (say|claim|read)    a comment correcting an earlier comment
    formerly | was wrong | superseded | re-?scheduled | retract

Executable strings are NOT counted: a ``raise``/``warn`` message is behaviour,
and several of them legitimately carry "does NOT hold" style wording.  A file
that does not parse falls back to a whole-file count rather than being skipped,
because "unparseable" must not be a way to hide from the ratchet.

FALSE POSITIVES, named rather than pattern-tuned away.  ``\\bpre-?fix\\b``
matches the ENGLISH WORD "prefix", which this library uses constantly
(``CONVENTIONS.md`` Section 2 requires every error message to carry an
``fn_name:`` prefix).  Ten such lines survive in the WP-A17 SWEEP-4 partition
alone.  Narrowing the pattern would put this file's numbers out of step with
the finding's own classifier and with the four sweep reports; baselining them
keeps one definition of "history line" across all of it, and the ratchet does
not care that a baselined count is 3 rather than 0 -- only that it does not
climb.

TESTING_STANDARDS: no wall-clock, resource or per-build assertion; every bar
here is an integer comparison against a committed file.
``test_the_counter_actually_counts`` and
``test_the_ratchet_direction_is_enforced`` are the falsifiability checks (V1):
a counter that returned 0 for everything, or a comparison that never failed,
would pass this file forever while the backlog rebuilt.
"""
from __future__ import annotations

import ast
import io
import json
import os
import pathlib
import re
import sys
import tokenize

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PACKAGE = REPO_ROOT / "lumenairy"
BASELINE_PATH = pathlib.Path(__file__).with_name(
    "test_audit2609_a17_history_lint_baseline.json")

#: The finding's own shapes.  Kept verbatim -- the same set the WP-A17 sweep
#: reports count with, so the numbers in those reports and the numbers in the
#: baseline are the same numbers.
HISTORY_PATTERN = re.compile(
    r"\bv\d+\.\d+(\.\d+)? \(|"
    r"\bpre-?fix\b|"
    r"\bpre-v?\d|"
    r"used to (say|claim|read)|"
    r"formerly|"
    r"was wrong|"
    r"superseded|"
    r"re-?scheduled|"
    r"retract",
    re.I)

_WRITE_ENV = "LUMENAIRY_HISTORY_LINT_WRITE"


# ---------------------------------------------------------------------------
# the counter
# ---------------------------------------------------------------------------
def _prose_lines(text: str) -> set[int] | None:
    """1-based line numbers inside a comment or a string STATEMENT.

    ``None`` when the file cannot be tokenised or parsed -- the caller then
    counts every line instead, so a syntax error is not a hiding place.
    """
    try:
        rows = {
            tok.start[0]
            for tok in tokenize.generate_tokens(io.StringIO(text).readline)
            if tokenize.tok_name[tok.type] == "COMMENT"
        }
        tree = ast.parse(text)
    except (SyntaxError, tokenize.TokenError, IndentationError):
        return None
    for node in ast.walk(tree):
        if (isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)):
            rows.update(range(node.lineno,
                              (node.end_lineno or node.lineno) + 1))
    return rows


def history_lines(text: str) -> list[tuple[int, str]]:
    """``(lineno, line)`` for every counted history line in ``text``."""
    lines = text.split("\n")
    rows = _prose_lines(text)
    candidates = (sorted(rows) if rows is not None
                  else range(1, len(lines) + 1))
    return [(i, lines[i - 1]) for i in candidates
            if i <= len(lines) and HISTORY_PATTERN.search(lines[i - 1])]


def _read(path: pathlib.Path) -> str:
    """Decode as the tokenizer sees it -- universal newlines, utf-8.

    Read as BYTES and translated here rather than through
    ``Path.read_text()``: on a CRLF checkout the two disagree about what a
    line is, and the baseline has to mean the same thing on both.
    """
    return path.read_bytes().decode("utf-8", errors="replace").replace(
        "\r\n", "\n")


def _key(py: pathlib.Path, root: pathlib.Path) -> str:
    """The baseline key: the module's path relative to ``root``.

    ``root`` is a parameter so the falsifiability tests below can scan a tree
    they build in ``tmp_path`` with the same code the real scan uses.
    """
    try:
        return py.relative_to(root).as_posix()
    except ValueError:              # pragma: no cover - tmp trees only
        return py.name


def scan_tree(package: pathlib.Path = PACKAGE,
              root: pathlib.Path = REPO_ROOT) -> dict[str, int]:
    """``{module path relative to the repo root: history-line count}``."""
    out: dict[str, int] = {}
    for py in sorted(package.rglob("*.py")):
        if "__pycache__" in py.parts:
            continue
        out[_key(py, root)] = len(history_lines(_read(py)))
    return out


def scan_offenders(package: pathlib.Path = PACKAGE,
                   root: pathlib.Path = REPO_ROOT
                   ) -> dict[str, list[tuple[int, str]]]:
    out: dict[str, list[tuple[int, str]]] = {}
    for py in sorted(package.rglob("*.py")):
        if "__pycache__" in py.parts:
            continue
        hits = history_lines(_read(py))
        if hits:
            out[_key(py, root)] = hits
    return out


# ---------------------------------------------------------------------------
# the baseline
# ---------------------------------------------------------------------------
def load_baseline(path: pathlib.Path = BASELINE_PATH) -> dict[str, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: int(v) for k, v in data["counts"].items()}


def write_baseline(path: pathlib.Path = BASELINE_PATH,
                   package: pathlib.Path = PACKAGE,
                   root: pathlib.Path = REPO_ROOT) -> dict[str, int]:
    """Re-record the baseline from the live tree.  Only non-zero counts are
    stored, so the file stays readable and a module that drops to zero shows
    up as a deletion in the diff."""
    counts = {k: v for k, v in scan_tree(package, root).items() if v}
    payload = {
        "_comment": (
            "History-line counts per module -- the RATCHET baseline for "
            "tests/unit/test_audit2609_a17_history_lint.py.  A module may "
            "shrink or stay; it may not grow.  Modules with a count of 0 are "
            "omitted and are treated as 0.  Re-record with "
            "`python tests/unit/test_audit2609_a17_history_lint.py --write`."),
        "pattern": HISTORY_PATTERN.pattern,
        "counted": ("comment and string-statement lines only; whole-file when "
                    "the module does not parse"),
        "total": sum(counts.values()),
        "counts": dict(sorted(counts.items())),
    }
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    return counts


def compare(live: dict[str, int], baseline: dict[str, int]
            ) -> tuple[list[tuple[str, int, int]], list[tuple[str, int, int]]]:
    """``(grown, shrunk)`` as ``(module, baseline, live)`` triples."""
    grown, shrunk = [], []
    for rel, n in sorted(live.items()):
        was = baseline.get(rel, 0)
        if n > was:
            grown.append((rel, was, n))
        elif n < was:
            shrunk.append((rel, was, n))
    return grown, shrunk


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
def test_the_baseline_file_is_present_and_non_trivial():
    """A guard on the guard.  An empty or missing baseline would make every
    module's bar 0, which either fails everywhere (useless) or -- if the scan
    also came back empty -- passes while checking nothing."""
    assert BASELINE_PATH.is_file(), (
        f"{BASELINE_PATH.name} is missing; re-record it with\n"
        f"    python {pathlib.Path(__file__).name} --write")
    baseline = load_baseline()
    assert baseline, f"{BASELINE_PATH.name} holds no counts"
    assert sum(baseline.values()) > 0
    live = scan_tree()
    assert len(live) > 100, (
        f"the scan found only {len(live)} modules under {PACKAGE}; the tree "
        f"walk is broken, not the library")


def test_no_module_accumulates_more_version_history():
    """THE RATCHET.  A module may shrink or stay; it may not grow."""
    if os.environ.get(_WRITE_ENV):
        write_baseline()
        pytest.skip(f"{_WRITE_ENV} set: baseline re-recorded, not checked")

    live = scan_tree()
    baseline = load_baseline()
    grown, shrunk = compare(live, baseline)

    if grown:
        offenders = scan_offenders()
        detail = []
        for rel, was, now in grown:
            hits = offenders.get(rel, [])
            detail.append(f"\n  {rel}: {was} -> {now}")
            # The ratchet knows the COUNT moved, not WHICH line is new, so it
            # lists what it counted -- capped, because a module at 40 would
            # otherwise bury the three modules that also grew.
            for lineno, line in hits[:12]:
                detail.append(f"\n      {rel}:{lineno}: {line.strip()[:110]}")
            if len(hits) > 12:
                detail.append(f"\n      ... and {len(hits) - 12} more")
        raise AssertionError(
            "WP-A17 ratchet: version-history narrative GREW in "
            f"{len(grown)} module(s)." + "".join(detail) + "\n\n"
            "A comment says what the code does NOW and why; the "
            "\"vX.Y (audit Z): pre-fix this did A\" narrative belongs in "
            "docs/history/<dotted.module.path>.md (see docs/history/ for the "
            "format, and CONTRIBUTING.md for the rule).  If one of these is a "
            "LIVE migration statement or a file-format contract that has to "
            "name a release, say so in the commit message and re-baseline:\n"
            f"    python tests/unit/{pathlib.Path(__file__).name} --write")

    if shrunk:
        gained = sum(was - now for _, was, now in shrunk)
        print(f"\nWP-A17 ratchet: {gained} history line(s) removed from "
              f"{len(shrunk)} module(s) since the baseline -- re-baseline to "
              f"lock the gain in:\n"
              f"    python tests/unit/{pathlib.Path(__file__).name} --write")
        for rel, was, now in shrunk[:20]:
            print(f"    {rel}: {was} -> {now}")


def test_the_counter_actually_counts(tmp_path):
    """Falsifiability part 1: the counter discriminates.

    A counter that returned 0 for everything would make the ratchet
    unfailable.  Each arm below is a shape the finding names, or one it
    deliberately does not, on a module written here rather than found in the
    tree -- so the check does not depend on what the library happens to
    contain today.
    """
    cases = [
        # (source, expected count, what it proves)
        ("x = 1\n", 0, "clean code counts zero"),
        ("# v5.30 (audit E-L4): the scaffold was dead.\nx = 1\n", 1,
         "a release tag with an audit id is counted"),
        ("# Pre-4.10 this returned 0 sag.\nx = 1\n", 1,
         "the pre-4.10 spelling is counted"),
        ("# pre-fix the pool was silent.\nx = 1\n", 1,
         "the pre-fix framing is counted"),
        ('"""This docstring used to claim the opposite."""\nx = 1\n', 1,
         "a string STATEMENT is counted"),
        ("x = 'pre-fix the pool was silent'\n", 0,
         "an EXECUTABLE string is not counted -- it is behaviour"),
        ("# formerly written out three times\n# was wrong\n# superseded\n"
         "x = 1\n", 3, "each shape counts its own line"),
        ("def f(prefix):\n    return prefix\n", 0,
         "the English word 'prefix' in CODE is not counted"),
        ("# the fn_name: prefix\nx = 1\n", 1,
         "the English word 'prefix' in a COMMENT is a known false positive, "
         "counted on purpose so the pattern stays the finding's own"),
        ("def f(:\n", 0,
         "an unparseable file falls back to a whole-file count (0 here)"),
        ("# pre-fix this was wrong\ndef f(:\n", 1,
         "an unparseable file is still counted, not skipped"),
    ]
    for i, (src, want, why) in enumerate(cases):
        p = tmp_path / f"m{i}.py"
        p.write_text(src, encoding="utf-8")
        got = len(history_lines(_read(p)))
        assert got == want, f"{why}: expected {want}, got {got} for {src!r}"


def test_the_ratchet_direction_is_enforced():
    """Falsifiability part 2: the COMPARISON fails on growth and not on
    shrinkage.  Asserted on synthetic counts, so it holds whatever the tree
    looks like."""
    base = {"a.py": 5, "b.py": 2}

    grown, shrunk = compare({"a.py": 6, "b.py": 2}, base)
    assert grown == [("a.py", 5, 6)] and not shrunk, "growth must be reported"

    grown, shrunk = compare({"a.py": 4, "b.py": 2}, base)
    assert not grown and shrunk == [("a.py", 5, 4)], "shrinkage must pass"

    grown, shrunk = compare({"a.py": 5, "b.py": 2}, base)
    assert not grown and not shrunk, "an unchanged tree must be silent"

    grown, _ = compare({"a.py": 5, "b.py": 2, "new.py": 1}, base)
    assert grown == [("new.py", 0, 1)], (
        "a module with no baseline entry must be held to 0, or narrative "
        "walks in through a new file")


def test_the_write_path_round_trips(tmp_path):
    """Falsifiability part 3: the re-baseline path writes what the checker
    reads, on a tree built here.  A writer that emitted a shape
    :func:`load_baseline` could not parse would turn the sanctioned recovery
    into a broken gate."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "clean.py").write_text("x = 1\n", encoding="utf-8")
    (pkg / "dirty.py").write_text(
        "# v5.30 (audit X): pre-fix this was wrong.\nx = 1\n",
        encoding="utf-8")
    out = tmp_path / "baseline.json"

    written = write_baseline(out, pkg, tmp_path)
    # the clean module is OMITTED, not stored as 0
    assert written == {"pkg/dirty.py": 1}, written

    reloaded = load_baseline(out)
    assert reloaded == written, "the writer and the reader disagree"
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["total"] == 1
    assert payload["pattern"] == HISTORY_PATTERN.pattern


if __name__ == "__main__":
    if "--write" in sys.argv:
        counts = write_baseline()
        print(f"re-recorded {BASELINE_PATH} -- {len(counts)} module(s), "
              f"{sum(counts.values())} history line(s)")
    else:
        live, base = scan_tree(), load_baseline()
        grown, shrunk = compare(live, base)
        for rel, was, now in grown:
            print(f"GROWN  {rel}: {was} -> {now}")
        for rel, was, now in shrunk:
            print(f"shrunk {rel}: {was} -> {now}")
        print(f"total {sum(live.values())} (baseline {sum(base.values())})")
        sys.exit(1 if grown else 0)
